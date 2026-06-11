import argparse
import json
import os
import torch
import time
from utils.utilities import *




## dataloader
dataset_name = 'trumans'
scene_dim = 32*32
bbox_scale = 0.6
window_size = 80
downsample_rate = 1
data_scale = 1.0
dataset_root = os.path.join('data', dataset_name)
checkpoints_dir = os.path.join('checkpoints', dataset_name)

predict_velocity = True
## VQVae architecture hyper parameters
npoints = 8192
model_name = 'TrajectoryTrainer'
code_dim = 512
code_num = 1024
output_emb_width = 512
condition_emb_dim = 512
num_heads = 8
down_t = 2
stride_t = 2
width = 512
depth = 3
dilation_growth_rate = 3
mu = 0.99
vq_norm = 'LN'
vq_act = 'silu'
quantizer = "ema_reset"
num_quantizers = 0  # make this greater than 0 to use residual VQVAE
quantize_dropout_prob = 0.2
## train & test
gpu_id = [0]
batch_size = 128
seed = 3407
learning_rate = 0.00007
change_lr=True
num_epoch = 10000
use_cuda = True
resume_model = ''
num_workers = 0
total_iter = 30000
milestones = [150000, 250000]
step_size = 100
gamma = 0.98
warm_up_iter = 2000
eval_every_e = 20
save_every_e = 100
log_every_e = 50

## training losses
weight_loss_rec = [1.0, 1.0, 2.0, 1.0, 10.0, 1.0, 1.0, 2.0, 2.0, 10.0]
weight_loss_vel = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
weight_loss_foot_contact = 1.0
weight_loss_commit = 0.1
weight_loss_rec_pose = 1.0
weight_loss_rec_vertex = 1.0
weight_loss_kl = 0.1
weight_loss_fk = 0.01
weight_loss_vposer = 1e-3
weight_loss_ground = 1.0

def arg_parse(is_train=False, load_exp=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='dataset directory')
    parser.add_argument('--dataset_root', type=str, default=dataset_root, help='dataset directory')
    parser.add_argument('--window_size', type=int, default=window_size, help='training motion length')
    parser.add_argument('--bbox_scale', type=float, default=bbox_scale, help='scale of the local bounding box with heightmap')
    parser.add_argument('--scene_dim', type=int, default=scene_dim, help='number of points in local height map')
    parser.add_argument('--downsample_rate', type=int, default=downsample_rate, help='downsample rate of each mini sequence')
    parser.add_argument('--data_scale', type=float, default=data_scale, help='scale down data before training')
    parser.add_argument("--predict_velocity", type=bool, default=predict_velocity, help='Predict root velocities instead of position?')
    parser.add_argument("--gpu_id", type=int, default=gpu_id, help='GPU id')
    parser.add_argument('--checkpoints_dir', type=str, default=checkpoints_dir, help='models are saved here')
    ## path setting
    parser.add_argument('--logs_dir', 
                        type=str, 
                        default='/logs',
                        help='dir for saving checkpoints and logs')
    parser.add_argument('--stamp', 
                        type=str, 
                        default=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                        help='timestamp')

    ## vqvae architecture
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of this model')
    parser.add_argument("--code_dim", type=int, default=code_dim, help="embedding dimension")
    parser.add_argument("--code_num", type=int, default=code_num, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=mu, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=down_t, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=stride_t, help="stride size")
    parser.add_argument("--width", type=int, default=width, help="width of the network")
    parser.add_argument('--n_heads', type=int, default=num_heads, help='Number of heads.')
    parser.add_argument("--depth", type=int, default=depth, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=dilation_growth_rate, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=output_emb_width, help="output embedding width")
    parser.add_argument("--condition_emb_dim", type=int, default=condition_emb_dim, help="condition embedding width")
    parser.add_argument('--vq_act', type=str, default=vq_act, choices=['relu', 'silu', 'gelu'],
                        help='vq activation function')
    parser.add_argument('--vq_norm', type=str, default=vq_norm, help='vq norm')

    parser.add_argument('--quantizer', type=str, default=quantizer, choices=['ema_reset', 'orig', 'ema', 'reset'],
                        help='quantizer for vqvae')
    parser.add_argument('--num_quantizers', type=int, default=num_quantizers, help='num_quantizers for residual vqvae')
    parser.add_argument('--shared_codebook', type=bool, default=True)
    parser.add_argument('--quantize_dropout_prob', type=float, default=quantize_dropout_prob, help='quantize_dropout_prob')

    
    ## training and optimization
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of this trial')
    parser.add_argument("--seed", default=seed, type=int)
    parser.add_argument('--total_iter', default=total_iter, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=warm_up_iter, type=int, help='number of total iterations for warmup')
    
    parser.add_argument('--milestones', default=milestones, nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--step_size', default=step_size, nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=gamma, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--change_lr', 
                        type=bool, 
                        default=change_lr,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--use_cuda', 
                        type=str, 
                        default=use_cuda,
                        help='set device for training')
    parser.add_argument('--resume_model',
                        type=str,
                        default=resume_model,
                        help='resume model path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=num_workers,
                        help='number of dataloader worker processer')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')

    parser.add_argument('--log_every_e', default=log_every_e, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=save_every_e, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=eval_every_e, type=int, help='save eval results every n epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')
    
    
    ## training loss
    parser.add_argument('--weight_loss_commit', type=float, default=weight_loss_commit, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--weight_loss_rec',
                        type=list,
                        default=weight_loss_rec,
                        help='loss weight of rec loss')
    parser.add_argument('--weight_loss_vel',
                        type=list,
                        default=weight_loss_vel,
                        help='loss weight of velocity loss')
    parser.add_argument('--weight_loss_foot_contact',
                        type=float,
                        default=weight_loss_foot_contact,
                        help='loss weight of binary cross entropy loss for foot contacts')
    parser.add_argument('--weight_loss_fk',
                        type=float,
                        default=weight_loss_fk,
                        help='loss weight of forward kinematics loss')
    parser.add_argument('--weight_loss_kl',
                        type=float,
                        default=weight_loss_kl,
                        help='loss weight of kl loss')
    parser.add_argument('--weight_loss_vposer',
                        type=float,
                        default=weight_loss_vposer,
                        help='loss weight of vposer loss')
    parser.add_argument('--weight_loss_ground',
                        type=float,
                        default=weight_loss_ground,
                        help='loss weight of ground loss')
    parser.add_argument('--all_body_vertices',
                        action="store_true",
                        help='use all body vertices to regress')
    
  
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(opt.gpu_id)
    
    if load_exp is None:
        opt = parser.parse_args()
        opt.is_train = is_train
        opt.load_exp = load_exp
        if is_train:
            # create new experiment
            opt.experiment = _update_exp()

            experiment_name = 'exp_' + str(opt.experiment) + '_' + opt.model_name + '_' + str(opt.batch_size) + '_' + str(opt.window_size) 
            
            opt.experiment_name = experiment_name
            expr_dir = os.path.join(opt.checkpoints_dir, experiment_name)
            opt.logs_dir = os.path.join(expr_dir, 'tenserboard_logs')
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            if not os.path.exists(opt.logs_dir):
                os.makedirs(opt.logs_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            args = vars(opt)
            json.dump(args, open(file_name,'w'))
            # with open(file_name, 'wt') as opt_file:
            #     for k, v in sorted(args.items()):
            #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        
    else:
        
        assert load_exp.endswith('.p')
        expr_dir = os.path.dirname(os.path.dirname(load_exp))
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name) as f: 
            file_lines = f.read()
        args = json.loads(file_lines)
        args['load_exp'] = load_exp
        args['is_train'] = is_train
        opt = argparse.Namespace(**args)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
    return opt

def _update_exp():
    exp_file = '.experiments'
    if not os.path.exists(exp_file):
        exp = -1
        with open(exp_file, 'w') as f:
            f.writelines([f'{exp}\n'])
    else:
        with open(exp_file, 'r') as f:
            lines = f.readlines()
        exp = int(lines[0].strip())
    exp += 1
    with open(exp_file, 'w') as f:
        f.writelines([f'{exp}\n'])
    print(f'Experiment Number: {exp}')
    return exp