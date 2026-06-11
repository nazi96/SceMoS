import os
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

from utils.transformations import *
from dataset_prep.trumans_loader import *
from dataset_prep.motion_process import *
from models.vqvae import *
from train.args_vqvae import *
from utils.utilities import *

is_train=False
test_split = 'test'   # the dataset split that you want to evaluate on.
load_exp = None
load_exp = os.path.join('checkpoints', 'trumans', 'exp_11_VQVAE_decoder_heightmap_contact_256_80', '11360', 'weights.p')


def dataset_prepare(opt):
    train_dataset = TrumansDataset(phase='train', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                     downsample_rate=opt.downsample_rate, device=opt.device)
    val_dataset = TrumansDataset(phase='test', window_size=opt.window_size, predict_velocity=opt.predict_velocity,
                                    downsample_rate=opt.downsample_rate, device=opt.device)
   
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


class VQTokenizerTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("Using ", self.opt.gpu_id, "GPUs! ")
        self.num_jts = BODY_JOINTS
       
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.scale = self.opt.data_scale
        self.batch_size = self.opt.batch_size
        self.m_codebook_size = self.opt.code_num
        print("batch size:{}".format(self.batch_size))
            # torch.autograd.set_detect_anomaly(True)
        self.dim_pose = MOTION_FEATS_BODY_ONLY
        # instantiate the model and move it to the right device
        net = eval(self.opt.model_name)( 
            input_feats=self.dim_pose,
            output_feats=self.dim_pose,
            quantizer=self.opt.quantizer,
            code_num=self.opt.code_num,
            code_dim=self.opt.code_dim,
            output_emb_width = self.opt.output_emb_width,
            down_t=self.opt.down_t,
            stride_t=self.opt.stride_t,
            width=self.opt.width,
            depth=self.opt.depth,
            dilation_growth_rate=self.opt.dilation_growth_rate,
            norm=self.opt.vq_norm,
            activation=self.opt.vq_act,
            gpu_id = self.opt.gpu_id[0],
            scene_dim=self.opt.scene_dim,
            condition_emb_dim=self.opt.condition_emb_dim,
            n_heads=self.opt.n_heads,
            num_quantizers=self.opt.num_quantizers,
            shared_codebook=self.opt.shared_codebook,
            quantize_dropout_prob=self.opt.quantize_dropout_prob
            )


        self.device= torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        self.opt.device = self.device
        # net = nn.DataParallel(net, device_ids=self.opt.gpu_id)
        self.vq_model = net.to(self.device)
        pc_vq = sum(param.numel() for param in net.parameters())
        # print(net)
        print('Total parameters of VQVAE: {:,}'.format(pc_vq))
        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_vq_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt_vq_model, 
            lr_lambda=lambda epoch: max(self.opt.gamma ** (epoch // self.opt.step_size), 1e-6/self.opt.lr)
        )
        self.epoch = 0
        self.it = 0
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        # self.male_smplxmodel = SMPLXLayer(model_path=SMPLX_FOLDER,
        #                         gender='male', 
        #                         # batch_size= data['global_orient'].shape[0] * data['global_orient'].shape[1],
        #                         num_betas=10,
        #                         use_pca=False,
        #                         use_face_contour=True, 
        #                         flat_hand_mean=True).to(self.device)

        
        # self.female_smplxmodel = SMPLXLayer(model_path=SMPLX_FOLDER,
        #                         gender='female', 
        #                         # batch_size= data['global_orient'].shape[0] * data['global_orient'].shape[1],
        #                         num_betas=10,
        #                         use_pca=False,
        #                         use_face_contour=True, 
        #                         flat_hand_mean=True).to(self.device)


        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
            self.inv_z_normalization = self.train_loader.dataset.inv_z_normalization
            self.logger = SummaryWriter(self.opt.logs_dir)
            l1_criterion = None
            if self.opt.recons_loss == 'l1_smooth':
                l1_criterion = torch.nn.SmoothL1Loss()
            elif self.opt.recons_loss == 'l1':
                l1_criterion = torch.nn.L1Loss()
            self.rec_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            self.rel_rec_loss_list = [l1_criterion, l1_criterion, l1_criterion]
            self.vel_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            self.fk_loss = torch.nn.MSELoss()
            self.bce_ = torch.nn.BCELoss()
            self.sigmoid_ = torch.nn.Sigmoid()
        else:
            self.load_dataset = TrumansDataset(phase=test_split, window_size=self.opt.window_size,
                                    predict_velocity=opt.predict_velocity, 
                                    downsample_rate=self.opt.downsample_rate, device=self.device, load_scene_vertex=True)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, drop_last=True, num_workers=0,
                            shuffle=False)
            self.inv_z_normalization = self.data_loader.dataset.inv_z_normalization
        # self.critic = CriticWrapper(self.opt.dataset_name, self.device)

    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, dir_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train validation loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
       

    def resume(self, model_dir, change_lr=False):
        checkpoint = torch.load(model_dir, map_location=self.device,weights_only=True)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        if change_lr:
            for param_group in self.opt_vq_model.param_groups:
                param_group["lr"] = self.opt.lr
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def forward(self, batch_data, iterations):
       
        heightmap = batch_data['heightmap'][:, ::4].to(self.device)
        heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
        contact_maps = batch_data['contact_map'][:, ::4].to(self.device)
        contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
        input_motion = batch_data['normalized_mot_feats'].to(self.device)
        pred_motion_normalized_delta, loss_commit, perplexity, code_idx = self.vq_model(input_motion, heightmap_cond, contact_maps_cond)
        

        # if self.opt.predict_velocity:
        pred_motion_normalized = delta_mot2mot_feats(batch_data['mot_init'].to(self.device), pred_motion_normalized_delta.to(self.device))
        pred_motion = self.inv_z_normalization(pred_motion_normalized.to(self.device))
        gt_motion = batch_data['motion_features'].to(self.device)

        loss_commit = torch.mean(loss_commit)
        perplexity = torch.mean(perplexity)
        
        B, T, dim = pred_motion.shape
        gt_root_transl, gt_root_rot, gt_body_rot, gt_local_jointpos, gt_local_jointvel, gt_foot_contact = reverse_process_mot(gt_motion.to(self.device))
        pred_root_transl, pred_root_rot, pred_body_rot, pred_local_jointpos, pred_local_jointvel, pred_foot_contact = reverse_process_mot(pred_motion.to(self.device))

        gt_mot_features = [gt_root_transl, gt_root_rot, gt_body_rot, gt_local_jointpos, gt_local_jointvel, gt_foot_contact]
        pred_mot_features = [pred_root_transl, pred_root_rot, pred_body_rot, pred_local_jointpos, pred_local_jointvel, pred_foot_contact]
        num_features = len(gt_mot_features)
        loss_rec_ = [opt.weight_loss_rec[i] * self.rec_loss_list[i](pred_mot_features[i], gt_mot_features[i])
                       for i in range(num_features)]
        loss_vel_ = [opt.weight_loss_vel[i] * self.vel_loss_list[i](pred_mot_features[i][:, 20:] - pred_mot_features[i][:, :-20], gt_mot_features[i][:, 20:] - gt_mot_features[i][:, :-20])
                       for i in range(num_features)]
        
        
        # loss_fk = torch.tensor(0.0).float().to(self.device)
        loss_model = 0.1 * self.rec_loss_list[-1](pred_motion_normalized_delta, input_motion)
        for i_loss in loss_rec_:
            loss_model += i_loss

        for i_loss in loss_vel_:
            loss_model += i_loss    

        loss_model += loss_model + self.opt.weight_loss_commit * loss_commit
        return pred_motion, loss_model, [loss_rec_, loss_vel_], loss_commit, perplexity

    def loss_logs_arrange(self, train_logs, loss_list):
        train_logs['loss_rec_root_transl'] += loss_list[0][0].item()
        train_logs['loss_rec_root_rot'] += loss_list[0][1].item()
        train_logs['loss_rec_body_rot'] += loss_list[0][2].item()
        train_logs['loss_rec_local_jointpos'] += loss_list[0][3].item()
        train_logs['loss_rec_local_jointvel'] += loss_list[0][4].item()
        train_logs['loss_rec_foot_contact'] += loss_list[0][5].item()
        train_logs['loss_vel_root_transl'] += loss_list[1][0].item()
        train_logs['loss_vel_root_rot'] += loss_list[1][1].item()
        train_logs['loss_vel_body_rot'] += loss_list[1][2].item()
        train_logs['loss_vel_local_jointpos'] += loss_list[1][3].item()
        train_logs['loss_vel_local_jointvel'] += loss_list[1][4].item()
        train_logs['loss_vel_foot_contact'] += loss_list[1][5].item()
        if len(loss_list[0]) > 6:
            train_logs['loss_rec_global_pos'] += loss_list[0][6].item()
            train_logs['loss_vel_global_pos'] += loss_list[1][6].item()

        return train_logs

    def train_epoch(self, train_loader, epoch, it):
        self.vq_model.train()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        # train_loader.sampler.set_epoch(epoch)
        train_tqdm = tqdm(train_loader, desc='train' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(train_tqdm):
            self.opt_vq_model.zero_grad()      
            
            # if it < self.opt.warm_up_iter:
            #     current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
            pred_motion, loss_model, loss_list, loss_commit, perplexity = self.forward(batch_data, it)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
           
            train_logs['loss'] += loss_model.item()
            train_logs = self.loss_logs_arrange(train_logs, loss_list)
            train_logs['loss_commit'] += loss_commit.item()
            train_logs['perplexity'] += perplexity.item()
            train_logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
            
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), max_norm=1.0)
            self.opt_vq_model.step()
            it += 1
            
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it):
        self.vq_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        # val_tqdm = tqdm(val_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_loader):
            pred_motion, loss_model, loss_list, loss_commit, perplexity = self.forward(batch_data, it)
            
            val_logs['loss'] += loss_model.item()
            val_logs = self.loss_logs_arrange(val_logs, loss_list)
            val_logs['loss_commit'] += loss_commit.item()
            val_logs['perplexity'] += perplexity.item()
            val_logs['epoch'] += self.opt_vq_model.param_groups[0]['lr']

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
            'scene_vertices': batch_data['scene_vertex'].detach().cpu().numpy(),
            'local_vertices': batch_data['grid_xyz'][0].detach().cpu().numpy(),
            'texts': batch_data['texts']
        }
        
        pkl_savepath =  makepath(os.path.join(os.path.dirname(load_exp), test_split, os.path.basename(batch_data['sample_name'][0])[:-4] + '_' +
                                                   str(batch_data['start_seq'][0].item()) + '_' + str(batch_data['end_seq'][0].item()) + '_.pkl'), isfile=True)
        with open(pkl_savepath, 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # scene_faces = None # check for scene faces
        # from visualize.visualize_data import visualize_sequence
        # visualize_sequence(scene_data=[scene_vertices, scene_faces], person1=[gt_vertices, gt_faces], 
                        #    person2=[pred_vertices, pred_faces])

    def train(self):
        self.vq_model.to(self.device)
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
                               time.time() - start_time, self.it, train_logs['loss'],
                                epoch=self.epoch,  lr=self.opt_vq_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], 
                                   epoch=self.epoch, lr=self.opt_vq_model.param_groups[0]['lr'])
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

    def motion_token_to_string(self, motion_token: List):
        motion_string =  (f'<m_{self.m_codebook_size}>' +
                 ''.join([f'<m_{int(i)}>' for i in motion_token]) +
                 f'<m_{self.m_codebook_size + 1}>')
        return motion_string
    
    def encode_token(self): 
        self.vq_model.eval()
        # val_tqdm = tqdm(val_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(self.data_loader):
            
            x_condition = [
            batch_data['local_height_map'][:, 0],
            batch_data['local_scene_data_semantic_label'][:, 0],
            # batch_data['action_class'],
            ]
            if self.opt.predict_velocity:
                input_delta_motion = batch_data['delta_mot_feats'].to(self.device)
                # pred_motion_delta, loss_commit, perplexity, code_idx = self.vq_model(input_delta_motion, x_condition)
                code_idx = self.vq_model.motion_encode(input_delta_motion, x_condition)
            motion_strings = self.motion_token_to_string(code_idx.squeeze(0).tolist())
            save_tokenpath = makepath(os.path.join(PREPROCESSED_DATA_FOLDER, batch_data['action'][0].replace(" ", "") + '_tokens',
                                                   batch_data['scene'][0][:-2] + batch_data['motion'][0],
                                                   str(batch_data['ncase'].tolist()[0]), 'motion_token.txt'), isfile=True)
            with open(save_tokenpath, "w") as text_file:
                text_file.write(motion_strings)
            tmp=1


    def test(self): 
        self.vq_model.eval()
        val_tqdm = tqdm(self.data_loader, desc='val' + ' {:.10f}'.format(0), leave=False, ncols=120)
        for running_iter, batch_data in enumerate(val_tqdm):
            heightmap = batch_data['heightmap'][:, ::4].to(self.device)
            heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
            
            # heightmap_cond = torch.zeros_like(heightmap_cond).to(self.device)
            
            contact_maps = batch_data['contact_map'][:, ::4].to(self.device)
            contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
            input_motion = batch_data['normalized_mot_feats'].to(self.device)
            pred_motion_normalized_delta, loss_commit, perplexity, code_idx = self.vq_model(input_motion, heightmap_cond, contact_maps_cond)
            # pred_motion_normalized_delta, loss_commit, perplexity, code_idx = self.vq_model(input_motion, 
            #                                                                              torch.zeros_like(heightmap_cond).to(self.device),
            #                                                                              torch.zeros_like(contact_maps_cond).to(self.device))
            

            # if self.opt.predict_velocity:
            pred_motion_normalized = delta_mot2mot_feats(batch_data['mot_init'].to(self.device), pred_motion_normalized_delta.to(self.device))
            pred_motion = self.inv_z_normalization(pred_motion_normalized.to(self.device))
            self.batch_visualize_predicted(batch_data, pred_motion)    
    
    def evaluate(self): 
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
            pred_motion_normalized_delta, loss_commit, perplexity, code_idx = self.vq_model(input_motion, heightmap_cond, contact_maps_cond)
            

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
    trainer = VQTokenizerTrainer(opt)
    if is_train:
        trainer.train()
    else:
        trainer.test()
        # trainer.evaluate()
        # trainer.encode_token()