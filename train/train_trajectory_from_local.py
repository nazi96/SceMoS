"""Train the SceMoS Trajectory Refinement Module R (Section 3.4 of the paper).

R is a lightweight 1D-conv network that maps the local motion features
    x_local = [j_r, j_p, j_v, c_f]                    (D - 3 dims)
back to the root translation offset
    t_hat_delta = R(x_local)                          (3 dims)

so that the AR-generated trajectory can be replaced at inference, reducing foot
sliding artifacts. The architecture is already provided by `Global_Trajectory_Pred`
in `models.vqvae` (input_feats = D - 3, output_feats = 3, dilated 1-D conv).

Training data for R follows the SceMoS protocol: the local-motion input is
obtained from the *AR planner* + *VQ decoder* pipeline (default), so R sees the
same distribution of local features it will see at inference. The supervision
target is the GT root delta.

Loss (SceMoS eq. 7):
    L_traj = lambda_r * || t_delta      - t_hat_delta      ||_1
           + lambda_v * || Delta t_delta - Delta t_hat_delta ||_1
"""
import os
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import numpy as np
import pickle
import sys
import warnings
sys.path.append('.')
sys.path.append('..')
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from collections import OrderedDict, defaultdict
from smplx import SMPLX, SMPLXLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image

# Patch transformers to use Image.BICUBIC
import transformers
transformers.image_transforms.InterpolationMode = Image

from utils.transformations import *

from dataset_prep.trumans_loader import *
from dataset_prep.motion_process import *
from models.vqvae import *
from models.autoregressive_motion_generator import AutoregressiveMotionGenerator
from train.args_vqvae import *
from utils.utilities import *

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
is_train = False
test_split = 'test'   # the dataset split that you want to evaluate on.
load_exp = None
# Resume from a previously trained refinement network (uncomment to enable):
load_exp = os.path.join('checkpoints', 'trumans', 'exp_87_TrajectoryTrainer_128_80', 'latest', 'weights.p')

vq_pretrained_weight_path = os.path.join(
    'checkpoints', 'trumans',
    'exp_11_VQVAE_decoder_heightmap_contact_256_80', '11360', 'weights.p'
)
ar_pretrained_weight_path = os.path.join(
    'checkpoints', 'trumans',
    'exp_84_AutoregressiveMotionGenerator_32_80', 'latest', 'weights.p'
)

# Refinement-training settings (SceMoS Section 3.4):
#   INPUT_SOURCE: where the local-motion input x_local fed to R comes from
#     'ar'        : AR planner -> VQ decoder (matches test-time distribution, default)
#     'gt'        : ground-truth local features
#     'vq'        : VQ-direct (encode + quantize + decode of GT motion)
#     'mix_ar_gt' : 50/50 mix between AR and GT (regularizes against AR drift)
INPUT_SOURCE = 'ar'

# Loss mode:
#   True  -> paper's L1 on root delta + L1 on root acceleration (SceMoS eq. 7)
#   False -> legacy rich rec+vel losses on the full motion features
USE_PAPER_LOSS = True
LAMBDA_R = 1.0
LAMBDA_V = 1.0

# AR generation knobs (used when INPUT_SOURCE includes 'ar')
AR_TARGET_TOKEN_LEN = 20
AR_USE_GREEDY = True


def dataset_prepare(opt):
    train_dataset = TrumansDataset(phase='test', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                     downsample_rate=opt.downsample_rate, device=opt.device, load_dino_feats=True)
    val_dataset = TrumansDataset(phase='test', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                    downsample_rate=opt.downsample_rate, device=opt.device, load_dino_feats=True)
   
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                              drop_last=True, num_workers=0,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, 
                            drop_last=True, num_workers=0,
                            shuffle=False)
    return train_loader, val_loader

def def_value():
    return 0.0
def large_value():
    return 1e+5


def scene_vertices_numpy_for_visualization_pkl(scene_vertex_batch):
    """Match ``inference_scemos_pipeline._scene_vertex_numpy_for_pkl`` (1, N, 3) for aitviewer."""
    if isinstance(scene_vertex_batch, list):
        arr = scene_vertex_batch[0].detach().cpu().numpy()
    else:
        arr = scene_vertex_batch[0].detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32, order="C")
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return np.ascontiguousarray(arr.reshape(1, -1, 3))
    if arr.ndim == 3:
        return np.ascontiguousarray(arr)
    raise ValueError(f"scene_vertex: expected (N,3) or (B,N,3), got {arr.shape}")


class TrajectoryTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("Using ", self.opt.gpu_id, "GPUs! ")
        self.num_jts = BODY_JOINTS
       
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.scale = self.opt.data_scale
        self.batch_size = self.opt.batch_size
        print("batch size:{}".format(self.batch_size))
            # torch.autograd.set_detect_anomaly(True)
        self.dim_pose = MOTION_FEATS_BODY_ONLY
        self.vq_model, self.vq_opt = self.load_vq_model(vq_pretrained_weight_path)
        self.device= torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        # AR planner is needed whenever the refinement input is sourced from it.
        self.input_source = getattr(self.opt, 'input_source', INPUT_SOURCE)
        self.use_paper_loss = getattr(self.opt, 'use_paper_loss', USE_PAPER_LOSS)
        self.ar_model = None
        if self.input_source in ('ar', 'mix_ar_gt'):
            self.ar_model = self.load_ar_model(ar_pretrained_weight_path)
        
        if torch.cuda.is_available():
            print("Using GPU:", torch.cuda.get_device_name(self.opt.gpu_id[0]))
        else:
            print("Using CPU")
        
        
        net = Global_Trajectory_Pred( 
            input_feats=self.dim_pose - 3,    
            output_feats=3,    
            output_emb_width = self.opt.output_emb_width,
            down_t=self.opt.down_t,
            stride_t=self.opt.stride_t,
            width=self.opt.width,
            depth=self.opt.depth,
            dilation_growth_rate=self.opt.dilation_growth_rate,
            norm='LN',
            activation='relu',
            gpu_id=self.opt.gpu_id[0]
            )
        

        self.opt.device = self.device
        # net = nn.DataParallel(net, device_ids=self.opt.gpu_id)
        self.model = net.to(self.device)
        pc_tr = sum(param.numel() for param in net.parameters() if param.requires_grad)
        # print(net)
        print('Total trainable parameters of Trajectory Predictor: {:,}'.format(pc_tr))
        self.opt_model = optim.AdamW(self.model.parameters(), lr=self.opt.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.epoch = 0
        self.it = 0
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))


        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
            self.inv_z_normalization = self.train_loader.dataset.inv_z_normalization
            self.logger = SummaryWriter(self.opt.logs_dir)
            self.l1_smoothloss = torch.nn.SmoothL1Loss()
            if self.opt.recons_loss == 'l1_smooth':
                l1_criterion = torch.nn.SmoothL1Loss()
            elif self.opt.recons_loss == 'l1':
                l1_criterion = torch.nn.L1Loss()
            self.rec_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            self.rel_rec_loss_list = [l1_criterion, l1_criterion, l1_criterion]
            self.vel_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            
        else:
            self.load_dataset = TrumansDataset(phase=test_split, window_size=self.opt.window_size,
                                    predict_velocity=opt.predict_velocity, 
                                    downsample_rate=self.opt.downsample_rate, device=self.device, load_dino_feats=True, load_scene_vertex=True)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, drop_last=True, num_workers=0,
                            shuffle=False)
            self.inv_z_normalization = self.data_loader.dataset.inv_z_normalization

    def load_vq_model(self, vq_pretrained_weight_path):
        opt_path = os.path.join(os.path.dirname(os.path.dirname(vq_pretrained_weight_path)), 'opt.txt')
        vq_opt = get_opt(opt_path)
        vq_opt.gpu_id = self.opt.gpu_id
        # Fix for gpu_id
        if isinstance(vq_opt.gpu_id, int):
            gpu_id = vq_opt.gpu_id
        else:
            gpu_id = vq_opt.gpu_id[0]
        vq_model = eval(vq_opt.model_name)( 
            input_feats=self.dim_pose,
            output_feats=self.dim_pose,
            quantizer=vq_opt.quantizer,
            code_num=vq_opt.code_num,
            code_dim=vq_opt.code_dim,
            output_emb_width = vq_opt.output_emb_width,
            down_t=vq_opt.down_t,
            stride_t=vq_opt.stride_t,
            width=vq_opt.width,
            depth=vq_opt.depth,
            dilation_growth_rate=vq_opt.dilation_growth_rate,
            norm=vq_opt.vq_norm,
            activation=vq_opt.vq_act,
            gpu_id=gpu_id,
            scene_dim=vq_opt.scene_dim,
            condition_emb_dim=vq_opt.condition_emb_dim,
            n_heads=vq_opt.n_heads,
            num_quantizers=vq_opt.num_quantizers,
            shared_codebook=vq_opt.shared_codebook,
            quantize_dropout_prob=vq_opt.quantize_dropout_prob
            )
        vq_opt.device = torch.device('cuda:' + str(vq_opt.gpu_id[0]) if torch.cuda.is_available() and sum(vq_opt.gpu_id) > -1 else 'cpu')
        vq_model = vq_model.to(vq_opt.device)
        checkpoint = torch.load(vq_pretrained_weight_path, map_location=vq_opt.device, weights_only=True)
        vq_model.load_state_dict(checkpoint['vq_model'], strict=True)
        vq_model = freeze_model(vq_model)
        print(f'Loaded VQ Model {vq_opt.model_name}')
        return vq_model, vq_opt

    def load_ar_model(self, ar_weights_path):
        """Load the (frozen) autoregressive motion planner."""
        opt_path = os.path.join(os.path.dirname(os.path.dirname(ar_weights_path)), 'opt.txt')
        with open(opt_path) as f:
            ar_opt = json.load(f)
        model = AutoregressiveMotionGenerator(
            motion_vocab_size=ar_opt.get('motion_vocab_size', 1024),
            dino_feature_dim=ar_opt.get('dino_feature_dim', 768),
            text_feature_dim=ar_opt.get('text_feature_dim', 1024),
            hidden_dim=ar_opt.get('hidden_dim', 512),
            num_layers=ar_opt.get('num_layers', 8),
            num_heads=ar_opt.get('num_heads', 8),
            max_seq_len=ar_opt.get('max_seq_len', 64),
            dropout=0.0,
            t5_model_name=ar_opt.get('t5_model_name', 'google/flan-t5-large'),
            freeze_t5=True,
            ff_mult=ar_opt.get('ff_mult', 4),
        ).to(self.device).eval()
        ckpt = torch.load(ar_weights_path, map_location=self.device, weights_only=True)
        # strict=False so a minor head mismatch doesn't break loading
        model.load_state_dict(ckpt['model'], strict=False)
        for p in model.parameters():
            p.requires_grad = False
        print(f"Loaded AR planner from {ar_weights_path}")
        return model

    @torch.no_grad()
    def _ar_generate_delta(self, batch_data, heightmap_cond, contact_maps_cond):
        """Run AR planner -> VQ decoder to produce normalized motion features
        in *delta* space (same convention as `vq_model(...)`'s first return).

        Output: [B, T, D]  with [..., :3] = root delta velocity, [..., 3:] = local.
        """
        self.ar_model.eval()
        self.vq_model.eval()

        text_prompts = batch_data['texts']
        dino_feats = batch_data['dino_feats'].to(self.device).squeeze(1)

        memory, memory_kpm = self.ar_model.build_memory(dino_feats, text_prompts)
        if AR_USE_GREEDY:
            B = memory.shape[0]
            gen = torch.full((B, 1), self.ar_model.start_token_id,
                             dtype=torch.long, device=self.device)
            for _ in range(AR_TARGET_TOKEN_LEN):
                logits = self.ar_model._decode(gen, memory, memory_kpm)[:, -1, :]
                nxt = logits.argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
            motion_tokens = gen[:, 1:]
        else:
            motion_tokens = self.ar_model.generate_motion_tokens(
                memory, memory_kpm, target_length=AR_TARGET_TOKEN_LEN
            )

        pred_norm_delta = self.vq_model.motion_decode(
            motion_tokens, heightmap_cond, contact_maps_cond
        )
        return pred_norm_delta                                              # [B, T, D]

    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, dir_name, ep, total_it):
        state = {
            "model": self.model.state_dict(),
            "opt_model": self.opt_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train test loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
       

    def resume(self, model_dir, change_lr=False):
        checkpoint = torch.load(model_dir, map_location=self.device,weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        if change_lr:
            for param_group in self.opt_model.param_groups:
                param_group["lr"] = self.opt.lr
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def _get_input_local_motion(self, batch_data, heightmap_cond, contact_maps_cond,
                                input_motion):
        """Returns `input_delta_local_motion` (the local features fed to R) according
        to `self.input_source`. All branches return tensors that share semantics with
        `input_motion[..., 3:]` (z-normalized local motion features).
        """
        gt_local = input_motion[..., 3:]
        src = self.input_source
        if src == 'gt':
            return gt_local
        if src == 'vq':
            with torch.no_grad():
                pred_norm_delta, *_ = self.vq_model(
                    input_motion, heightmap_cond, contact_maps_cond
                )
            return pred_norm_delta[..., 3:]
        if src == 'ar':
            ar_delta = self._ar_generate_delta(batch_data, heightmap_cond, contact_maps_cond)
            return ar_delta[..., 3:]
        if src == 'mix_ar_gt':
            ar_delta = self._ar_generate_delta(batch_data, heightmap_cond, contact_maps_cond)
            ar_local = ar_delta[..., 3:]
            # Per-sample coin flip for slightly better regularization than per-batch.
            B = ar_local.shape[0]
            use_ar = torch.randint(0, 2, (B, 1, 1), device=ar_local.device).expand_as(ar_local)
            return torch.where(use_ar.bool(), ar_local, gt_local)
        raise ValueError(f"Unknown input_source: {src}")

    def forward(self, batch_data, iterations):
        heightmap = batch_data['heightmap'][:, ::4].to(self.device)
        heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
        contact_maps = batch_data['contact_map'][:, ::4].to(self.device)
        contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
        input_motion = batch_data['normalized_mot_feats'].to(self.device)

        # ---- pick the local-motion stream fed to R ----
        input_delta_local_motion = self._get_input_local_motion(
            batch_data, heightmap_cond, contact_maps_cond, input_motion
        )

        # ---- R predicts the root delta velocity ----
        pred_root_motion = self.model(input_delta_local_motion)             # [B, T, 3]
        output_root_motion = input_motion[..., :3]                          # GT root delta

        # ---- SceMoS eq. (7) ----
        if self.use_paper_loss:
            l1 = F.l1_loss
            loss_pos = l1(pred_root_motion, output_root_motion)
            gt_vel   = output_root_motion[:, 1:] - output_root_motion[:, :-1]
            pred_vel = pred_root_motion[:, 1:]   - pred_root_motion[:, :-1]
            loss_vel = l1(pred_vel, gt_vel)
            return LAMBDA_R * loss_pos + LAMBDA_V * loss_vel

        # ---- legacy rich loss (kept for backward compatibility) ----
        root_rec_loss = self.opt.weight_loss_rec[-1] * self.l1_smoothloss(
            pred_root_motion, output_root_motion
        )
        loss_model = root_rec_loss
        pred_motion_with_trajectory = torch.cat(
            (pred_root_motion[..., :3], input_delta_local_motion), dim=-1
        )
        pred_motion_normalized = delta_mot2mot_feats(
            batch_data['mot_init'].to(self.device),
            pred_motion_with_trajectory.to(self.device),
        )
        pred_motion = self.inv_z_normalization(pred_motion_normalized.to(self.device))
        gt_motion = batch_data['motion_features'].to(self.device)

        (gt_root_transl, gt_root_rot, gt_body_rot, gt_local_jointpos,
         gt_local_jointvel, gt_foot_contact) = reverse_process_mot(gt_motion.to(self.device))
        (pred_root_transl, pred_root_rot, pred_body_rot, pred_local_jointpos,
         pred_local_jointvel, pred_foot_contact) = reverse_process_mot(pred_motion.to(self.device))

        gt_mot_features   = [gt_root_transl, gt_root_rot, gt_body_rot,
                             gt_local_jointpos, gt_local_jointvel, gt_foot_contact]
        pred_mot_features = [pred_root_transl, pred_root_rot, pred_body_rot,
                             pred_local_jointpos, pred_local_jointvel, pred_foot_contact]
        num_features = len(gt_mot_features)
        loss_rec_ = [self.opt.weight_loss_rec[i] *
                     self.rec_loss_list[i](pred_mot_features[i], gt_mot_features[i])
                     for i in range(num_features)]
        loss_vel_ = [self.opt.weight_loss_vel[i] *
                     self.vel_loss_list[i](
                         pred_mot_features[i][:, 5:] - pred_mot_features[i][:, :-5],
                         gt_mot_features[i][:, 5:]   - gt_mot_features[i][:, :-5])
                     for i in range(num_features)]
        for i_loss in loss_rec_:
            loss_model += i_loss
        for i_loss in loss_vel_:
            loss_model += i_loss
        return loss_model

   
    def train_epoch(self, train_loader, epoch, it):
        self.model.train()
        self.vq_model.eval()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        # train_loader.sampler.set_epoch(epoch)
        train_tqdm = tqdm(train_loader, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(train_tqdm):
            self.opt_model.zero_grad()      
            
            # if it < self.opt.warm_up_iter:
            #     current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
            loss_model = self.forward(batch_data, it)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
           
            train_logs['loss'] += loss_model.item()
            train_logs['lr'] += self.opt_model.param_groups[0]['lr']
            
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt_model.step()
            it += 1
            
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it):
        self.model.eval()
        self.vq_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        # val_tqdm = tqdm(val_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_loader):
            loss_model = self.forward(batch_data, it)
            
            val_logs['loss'] += loss_model.item()
            val_logs['lr'] += self.opt_model.param_groups[0]['lr']

        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss


    def batch_visualize_predicted(self, batch_data, pred_motion):
        unnormalized_gt = batch_data['motion_features'].to(self.device)
        unnormalized_pred_motion = pred_motion
        
        gt_data, gt_vertices, gt_faces, _ = mot_to_smplx_verts(unnormalized_gt[0], betas=batch_data['betas'][0], 
                                                interpolate_by=2)
        pred_data, pred_vertices, pred_faces, _ = mot_to_smplx_verts(unnormalized_pred_motion[0], betas=batch_data['betas'][0],
                                                interpolate_by=2)
        output_dict = {
            'scene_name': batch_data['scene_name'][0],
            'gt_data': gt_data,
            'pred_data': pred_data,
            'gt_vertices': gt_vertices,
            'gt_faces': gt_faces,
            'pred_vertices': pred_vertices,
            'pred_faces': pred_faces,
            'scene_vertices': scene_vertices_numpy_for_visualization_pkl(batch_data['scene_vertex']),
            'local_vertices': batch_data['local_vertices'][0].detach().cpu().numpy(),
            'texts': batch_data['texts']
        }
        pkl_savepath =  makepath(os.path.join(os.path.dirname(load_exp), test_split + '', os.path.basename(batch_data['sample_name'][0])[:-4] + '_' +
                                                   str(batch_data['start_seq'][0].item()) + '_' + str(batch_data['end_seq'][0].item()) + '_.pkl'), isfile=True)
        with open(pkl_savepath, 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # scene_faces = None # check for scene faces
        # from visualize.visualize_data import visualize_sequence
        # visualize_sequence(scene_data=[scene_vertices, scene_faces], person1=[gt_vertices, gt_faces], 
                        #    person2=[pred_vertices, pred_faces])

    def train(self):
        self.model.to(self.device)
        total_iters = self.opt.num_epoch * len(self.train_loader)
        print('Total Epochs: {}, Total Iters: {}'.format(self.opt.num_epoch, total_iters))
        current_lr = self.opt.lr
        train_logs = defaultdict(def_value, OrderedDict())
        val_logs = defaultdict(def_value, OrderedDict())
        best_val_loss = 1e+5
        running_epoch = 0
        while self.epoch <= self.opt.num_epoch:
            start_time = time.time()
            train_logs, self.it = self.train_epoch(self.train_loader, self.epoch, self.it )
            print_current_loss('Train ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                               time.time() - start_time, self.it, train_logs['loss'], epoch=self.epoch,  lr=self.opt_model.param_groups[0]['lr'])
            # if running_epoch % self.opt.eval_every_e == 0:
            #     start_time_ = time.time()
            #     val_logs = self.val_epoch(self.val_loader, self.epoch, self.it)
            #     print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
            #                        time.time() - start_time_, self.it, val_logs['loss'], accuracy=val_logs['accuracy'],
            #                        epoch=self.epoch, lr=self.opt_model.param_groups[0]['lr'])
            #     if val_logs['loss'] < best_val_loss:
            #         self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, 'best_val_'), self.epoch, self.it)
            #         best_val_loss = val_logs['loss']
            if running_epoch % self.opt.log_every_e == 0:    
                for tag, value in train_logs.items():
                    self.logger.add_scalar('Train/%s'%tag, train_logs[tag], self.it)
                for tag, value in val_logs.items():
                    self.logger.add_scalar('Val/%s'%tag, val_logs[tag], self.it)
            # if self.it >= self.opt.warm_up_iter:
            if running_epoch % self.opt.save_every_e == 0:
                self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, str(self.epoch)), self.epoch, self.it)   
            elif running_epoch % 2 == 0:
                self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, 'latest'), self.epoch, self.it) 

            self.scheduler.step() 
            running_epoch += 1
            self.epoch += 1
            self.logger.flush()
        self.logger.close()  
        self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, str(self.epoch)), self.epoch, self.it)
        print('Training complete!')

 

    def test(self):
        """Inference pipeline (SceMoS Sec. 3.4):
            AR planner -> VQ decoder -> refinement R -> integrate -> visualize.

        If the AR model isn't loaded (e.g. INPUT_SOURCE == 'vq' or 'gt' for an
        ablation), we fall back to VQ-direct as the local-motion source.
        """
        self.model.eval()
        self.vq_model.eval()
        if self.ar_model is not None:
            self.ar_model.eval()

        val_tqdm = tqdm(self.data_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_tqdm):
            heightmap = batch_data['heightmap'][:, ::4].to(self.device)
            heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
            contact_maps = batch_data['contact_map'][:, ::4].to(self.device)
            contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
            input_motion = batch_data['normalized_mot_feats'].to(self.device)

            with torch.no_grad():
                if self.ar_model is not None:
                    pred_motion_normalized_delta = self._ar_generate_delta(
                        batch_data, heightmap_cond, contact_maps_cond
                    )
                else:
                    pred_motion_normalized_delta, *_ = self.vq_model(
                        input_motion, heightmap_cond, contact_maps_cond
                    )

            input_delta_local_motion = pred_motion_normalized_delta[..., 3:]
            pred_root_motion = self.model(input_delta_local_motion)
            pred_motion_with_trajectory = torch.cat(
                (pred_root_motion[..., :3], input_delta_local_motion), dim=-1
            )
            pred_motion_normalized = delta_mot2mot_feats(
                batch_data['mot_init'].to(self.device),
                pred_motion_with_trajectory.to(self.device),
            )
            pred_motion = self.inv_z_normalization(pred_motion_normalized.to(self.device))
            self.batch_visualize_predicted(batch_data, pred_motion)
    
    def evaluate(self): 
        self.model.eval()
        self.vq_model.eval()
        mpjgpe_sum = 0
        val_tqdm = tqdm(self.data_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_tqdm):
            heightmap = batch_data['heightmap'][:, ::4].to(self.device)
            heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
            
            # heightmap_cond = torch.zeros_like(heightmap_cond).to(self.device)
            
            contact_maps = batch_data['contact_map'][:, ::4].to(self.device)
            contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
            input_motion = batch_data['normalized_mot_feats'].to(self.device)
            pred_motion_normalized_delta, loss_commit, perplexity, code_idx = self.model(input_motion, heightmap_cond, contact_maps_cond)
            

            # if self.opt.predict_velocity:
            pred_motion_normalized = delta_mot2mot_feats(batch_data['mot_init'].to(self.device), pred_motion_normalized_delta.to(self.device))
            unnormalized_pred_motion = self.inv_z_normalization(pred_motion_normalized.to(self.device))
            unnormalized_gt = batch_data['motion_features'].to(self.device)
         
            _, _, _, gt_jt = mot_to_smplx_verts(unnormalized_gt[0], betas=batch_data['betas'][0], 
                                                interpolate_by=self.opt.downsample_rate)
            _, _, _, pred_jt = mot_to_smplx_verts(unnormalized_pred_motion[0], betas=batch_data['betas'][0],
                                                interpolate_by=self.opt.downsample_rate)
            gt_jt = gt_jt[:, :self.num_jts]
            pred_jt = pred_jt[:, :self.num_jts]

            mpjgpe_sum += np.linalg.norm(pred_jt - gt_jt)

        mpjpe = mpjgpe_sum / running_iter    # if running_iter > 0 and running_iter%50 == 0:
        print(mpjpe)
        eval_savepath = makepath(os.path.join(os.path.dirname(self.opt.load_exp), 'MPJPE.pkl'), isfile=True)
        with open(eval_savepath, 'wb') as f:
            pickle.dump(mpjpe, f)

            
   

if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=is_train, load_exp=load_exp)
    fixseed(opt.seed)
    trainer = TrajectoryTrainer(opt)
    if is_train:
        trainer.train()
    else:
        trainer.test()
        # trainer.evaluate()
        # trainer.encode_token()