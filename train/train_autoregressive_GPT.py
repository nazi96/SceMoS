"""Train or evaluate the AR token generator used after VQ-VAE pretraining.

The model predicts motion codebook tokens from scene (DINO) and text context.
"""

import os
import numpy as np
import pickle
import sys
import warnings

from sympy import true
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
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image


import transformers
transformers.image_transforms.InterpolationMode = Image

from utils.transformations import *

from dataset_prep.trumans_loader import *
from dataset_prep.motion_process import *
from models.autoregressive_motion_generator import *
from models.vqvae import *
from train.args_transformer import *
from utils.utilities import *

is_train=False
test_split = 'test'   # the dataset split that you want to evaluate on.
load_exp = None
load_exp = os.path.join('checkpoints', 'trumans', 'exp_84_AutoregressiveMotionGenerator_32_80', 'latest', 'weights.p')

vq_pretrained_weight_path = os.path.join('checkpoints', 'trumans', 'exp_11_VQVAE_decoder_heightmap_contact_256_80', '11360', 'weights.p')


def dataset_prepare(opt):
    """Build train/test dataloaders with DINO conditioning enabled."""
    train_dataset = TrumansDataset(phase='train', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                     downsample_rate=opt.downsample_rate, device=opt.device, load_dino_feats=True)
    test_dataset = TrumansDataset(phase='test', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                    downsample_rate=opt.downsample_rate, device=opt.device, load_dino_feats=True)

    
   
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                              drop_last=True, num_workers=opt.num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, 
                            drop_last=True, num_workers=opt.num_workers,
                            shuffle=False)
    return train_loader, test_loader

def def_value():
    return 0.0
def large_value():
    return 1e+5


class TransformerTrainer:
    """Wraps pretrained VQ tokenization and autoregressive transformer training."""
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
        
        if torch.cuda.is_available():
            print("Using GPU:", torch.cuda.get_device_name(self.opt.gpu_id[0]))
        else:
            print("Using CPU")
        
        
        # instantiate the model and move it to the right device
        model_kwargs = {
            'motion_vocab_size': self.opt.motion_vocab_size,
            'dino_feature_dim': self.opt.dino_feature_dim,
            'text_feature_dim': self.opt.text_feature_dim,
            'hidden_dim': self.opt.hidden_dim,
            'num_layers': self.opt.num_layers,
            'num_heads': self.opt.num_heads,
            'max_seq_len': self.opt.max_seq_len,
            # 'tokens_per_step': self.opt.tokens_per_step,
            'dropout': self.opt.dropout,
            't5_model_name': self.opt.t5_model_name,
            'freeze_t5': self.opt.freeze_t5,
            'disable_loss_masking': getattr(self.opt, 'disable_loss_masking', False),
            'ff_mult': getattr(self.opt, 'ff_mult', 4),
            'use_pre_decoder_layer_norm': getattr(self.opt, 'use_pre_decoder_layer_norm', False),
        }
        
        net = eval(self.opt.model_name)(**model_kwargs)
        

        self.opt.device = self.device
        # net = nn.DataParallel(net, device_ids=self.opt.gpu_id)
        self.model = net.to(self.device)
        pc_vq = sum(param.numel() for param in net.parameters() if param.requires_grad)
        # print(net)
        print('Total trainable parameters of Autoregressive Transformer: {:,}'.format(pc_vq))
        self.opt_model = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt_model, 
            lr_lambda=lambda epoch: max(self.opt.gamma ** (epoch // self.opt.step_size), 1e-6/self.opt.lr)
        )
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
            
        else:
            self.load_dataset = TrumansDataset(phase=test_split, window_size=self.opt.window_size,
                                    predict_velocity=opt.predict_velocity, 
                                    downsample_rate=self.opt.downsample_rate, device=self.device, load_dino_feats=True, load_scene_vertex=True)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, drop_last=True, num_workers=0,
                            shuffle=False)
            self.inv_z_normalization = self.data_loader.dataset.inv_z_normalization

    def load_vq_model(self, vq_pretrained_weight_path):
        """Load frozen VQ-VAE used to convert motions into token IDs."""
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

    def forward(self, batch_data, iterations):
        """Tokenize motions with VQ-VAE and train AR model with teacher forcing."""
        input_motion = batch_data['normalized_mot_feats'].to(self.device)
        text_prompts = batch_data['texts']
        action_labels = batch_data['action_label'].to(self.device)
        dino_features_patches = batch_data['dino_feats'].to(self.device).squeeze(1)
        # Motion tokens become the supervision target for autoregressive decoding.
        motion_tokens = self.vq_model.motion_encode(input_motion)
        batch_size, seq_len = motion_tokens.shape[:2]

        if getattr(self, '_debug_first_batch_logged', False) is False:
            try:
                tok_min = int(motion_tokens.min().item())
                tok_max = int(motion_tokens.max().item())
                print("[DEBUG/forward] motion_tokens shape:", tuple(motion_tokens.shape),
                      "min:", tok_min, "max:", tok_max,
                      "vocab:", self.opt.motion_vocab_size)
                print("[DEBUG/forward] dino:", tuple(dino_features_patches.shape),
                      "action_labels:", tuple(action_labels.shape),
                      "input_motion:", tuple(input_motion.shape))
                print("[DEBUG/forward] sample text[0]:", repr(text_prompts[0]) if len(text_prompts) > 0 else None)
                self._debug_first_batch_logged = True
            except Exception as _e:
                print("[DEBUG/forward] diagnostic print failed:", _e)
                self._debug_first_batch_logged = True

        # Curriculum learning: For the first 10000 iterations, use random 5 consecutive tokens
        # if iterations <= 2000:
            
        #     # Randomly select starting position for 10 consecutive tokens
        #     max_start = seq_len - 12
        #     if max_start > 0:
        #         start_pos = torch.randint(0, max_start, (1,), device=self.device).item()
        #         # Extract 5 consecutive tokens using the same start_pos for all samples
        #         motion_tokens_ = motion_tokens[:, start_pos:start_pos+12]
        #         action_labels_ = action_labels[:, 4*start_pos:4*start_pos+12]
        # elif iterations > 2000 and iterations <= 3500:
        #     # Randomly select starting position for 10 consecutive tokens
        #     max_start = seq_len - 16
        #     if max_start > 0:
        #         start_pos = torch.randint(0, max_start, (1,), device=self.device).item()
        #         # Extract 5 consecutive tokens using the same start_pos for all samples
        #         motion_tokens_ = motion_tokens[:, start_pos:start_pos+16]
        #         action_labels_ = action_labels[:, 4*start_pos:4*start_pos+16]

        # else:
        #     motion_tokens_ = motion_tokens
        #     action_labels_ = action_labels

        loss_patches, accuracy = self.model(dino_features_patches, text_prompts, action_labels, iterations, motion_tokens)

        return loss_patches, accuracy

   
    def train_epoch(self, train_loader, epoch, it):
        """Run one optimization epoch and report mean loss/accuracy."""
        self.model.train()
        self.vq_model.eval()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())

        if getattr(self.opt, 'overfit_single_batch', False):
            if not hasattr(self, '_cached_single_batch'):
                self._cached_single_batch = next(iter(train_loader))
                print("[DEBUG/single_batch] cached batch with batch_size=",
                      self._cached_single_batch['normalized_mot_feats'].shape[0],
                      "and reusing it every step.")
            steps_per_epoch = max(1, len(train_loader))
            iterator = ((i, self._cached_single_batch) for i in range(steps_per_epoch))
            train_tqdm = iterator
        else:
            train_tqdm = enumerate(tqdm(train_loader, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120))

        running_iter = -1
        for running_iter, batch_data in train_tqdm:
            self.opt_model.zero_grad()
            loss_model, acc = self.forward(batch_data, it)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()

            train_logs['loss'] += loss_model.item()
            train_logs['accuracy'] += acc
            train_logs['lr'] += self.opt_model.param_groups[0]['lr']

            loss_model.mean().backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.opt.grad_clip
            )
            train_logs['grad_norm'] += float(grad_norm)

            if self.opt.debug_print_every > 0 and (it % self.opt.debug_print_every == 0):
                print(f"[DEBUG/iter {it}] loss={loss_model.item():.4f} "
                      f"acc={float(acc):.4f} grad_norm={float(grad_norm):.4f} "
                      f"lr={self.opt_model.param_groups[0]['lr']:.6f}")

            self.opt_model.step()
            it += 1

        denom = max(1, running_iter + 1)
        for tag, value in train_logs.items():
            mean_loss[tag] = value / denom
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it):
        """Evaluate the AR model on held-out batches."""
        self.model.eval()
        self.vq_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        # val_tqdm = tqdm(val_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_loader):
            loss_model, acc = self.forward(batch_data, it)
            
            val_logs['loss'] += loss_model.item()
            val_logs['accuracy'] += acc
            val_logs['lr'] += self.opt_model.param_groups[0]['lr']

        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss


    def batch_visualize_predicted(self, batch_data, pred_motion):
        unnormalized_gt = batch_data['motion_features'].to(self.device)
        unnormalized_pred_motion = pred_motion
        # For root joint smoothing
        
        # unnormalized_pred_motion[..., :3] = unnormalized_gt[..., :3] 
        # unnormalized_pred_motion[..., knee_joint_indices] = unnormalized_gt[..., knee_joint_indices]
        # unnormalized_pred_motion[..., hip_joint_indices] = unnormalized_gt[..., hip_joint_indices]
        # predict_motion_smooth = smooth_root_joint_deltas(unnormalized_pred_motion)
        # unnormalized_pred_motion = predict_motion_smooth
        
        gt_data, gt_vertices, gt_faces, _ = mot_to_smplx_verts(unnormalized_gt[0], betas=batch_data['betas'][0], 
                                                interpolate_by=3)
        pred_data, pred_vertices, pred_faces, _ = mot_to_smplx_verts(unnormalized_pred_motion[0], betas=batch_data['betas'][0],
                                                interpolate_by=3)
        output_dict = {
            'scene_name': batch_data['scene_name'][0],
            'gt_data': gt_data,
            'pred_data': pred_data,
            'gt_vertices': gt_vertices,
            'gt_faces': gt_faces,
            'pred_vertices': pred_vertices,
            'pred_faces': pred_faces,
            'scene_vertices': batch_data['scene_vertex'].detach().cpu().numpy(),
            'local_vertices': batch_data['local_vertices'][0].detach().cpu().numpy(),
            'texts': batch_data['texts']
        }
        pkl_savepath =  makepath(os.path.join(os.path.dirname(load_exp), test_split + '_extra1', os.path.basename(batch_data['sample_name'][0])[:-4] + '_' +
                                                   str(batch_data['start_seq'][0].item()) + '_' + str(batch_data['end_seq'][0].item()) + '_.pkl'), isfile=True)
        with open(pkl_savepath, 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # scene_faces = None # check for scene faces
        # from visualize.visualize_data import visualize_sequence
        # visualize_sequence(scene_data=[scene_vertices, scene_faces], person1=[gt_vertices, gt_faces], 
                        #    person2=[pred_vertices, pred_faces])

    def train(self):
        """Main epoch loop: train, validate, log, and checkpoint."""
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
                               time.time() - start_time, self.it, train_logs['loss'], accuracy=train_logs['accuracy'],
                                epoch=self.epoch,  lr=self.opt_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], accuracy=val_logs['accuracy'],
                                   epoch=self.epoch, lr=self.opt_model.param_groups[0]['lr'])
                if val_logs['loss'] < best_val_loss:
                    self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, 'best_val_'), self.epoch, self.it)
                    best_val_loss = val_logs['loss']
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

   

if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=is_train, load_exp=load_exp)
    fixseed(opt.seed)
    trainer = TransformerTrainer(opt)
    if is_train:
        trainer.train()
    else:
        trainer.test()
        # trainer.evaluate()
        # trainer.encode_token()