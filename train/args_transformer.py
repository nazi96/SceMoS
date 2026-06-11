import argparse
import json
import os
import time


# Data and training defaults
DATASET_NAME = 'trumans'
SCENE_DIM = 32*32
BBOX_SCALE = 0.6
WINDOW_SIZE = 80
DOWNSAMPLE_RATE = 1
DATA_SCALE = 1.0
DATASET_ROOT = os.path.join('data', DATASET_NAME)
CHECKPOINTS_DIR = os.path.join('checkpoints', DATASET_NAME)
PREDICT_VELOCITY = True
BATCH_SIZE = 32
SEED = 3407
LEARNING_RATE = 0.001
CHANGE_LR = True
NUM_EPOCH = 3000
USE_CUDA = True
NUM_WORKERS = 0
STEP_SIZE = 1000
GAMMA = 1.0
WARM_UP_ITER = 2000
EVAL_EVERY_E = 20
SAVE_EVERY_E = 60
LOG_EVERY_E = 50

# AutoregressiveMotionGenerator architecture defaults
MODEL_NAME = 'AutoregressiveMotionGenerator'
# MODEL_NAME = 'MaskedTransformerForCustomInputs'
CODE_DIM = 512
CODE_NUM = 1024
LATENT_DIM = 512
FF_SIZE = 1024
NUM_LAYERS = 12
NUM_HEADS = 8
DROPOUT = 0.0
T5_DIM = 1024
FRAME_EMBEDDING_DIM = 128
COND_DROP_PROB = 0.0
USE_ROPE = True
GPU_ID = [0]
USE_RENDERED_IMAGES=True
use_local_grid=True

# AutoregressiveMotionGenerator specific defaults
MOTION_VOCAB_SIZE = 1024
DINO_FEATURE_DIM = 768
TEXT_FEATURE_DIM = 1024  # T5 dimension
HIDDEN_DIM = 512
MAX_SEQ_LEN = 50
TOKENS_PER_STEP = 4
T5_MODEL_NAME = 'google/flan-t5-large'
FREEZE_T5 = True
DINO_PATCH_SIZE = 16
DINO_NUM_PATCHES = 49
USE_DINO_PATCHES = True 
WEIGHT_DECAY = 0.0
GRAD_CLIP = 5.0
OVERFIT_NUM_SAMPLES = 400
OVERFIT_SINGLE_BATCH = False
DEBUG_PRINT_EVERY = 50
DISABLE_LOSS_MASKING = False
FF_MULT = 4
USE_PRE_DECODER_LAYER_NORM = False

## training losses
weight_loss_rec = [1.0, 1.0, 2.0, 1.0, 10.0, 1.0, 1.0, 2.0, 2.0, 10.0]
weight_loss_vel = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
weight_loss_foot_contact = 1.0
weight_loss_commit = 0.1
weight_loss_rec_pose = 1.0
weight_loss_rec_vertex = 1.0
weight_loss_kl = 0.1
weight_loss_fk = 0.1
weight_loss_vposer = 1e-3
weight_loss_ground = 1.0
weight_loss_consistency = 0.1  # Consistency between VQVAE and refined output
weight_loss_temporal = 0.01   # Temporal consistency loss
weight_loss_adversarial = 0.01   # Weight for the adversarial loss on the generator


def arg_parse(is_train=False, load_exp=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data loader
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, help='dataset directory')
    parser.add_argument('--dataset_root', type=str, default=DATASET_ROOT, help='dataset directory')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='training motion length')
    parser.add_argument('--bbox_scale', type=float, default=BBOX_SCALE, help='scale of the local bounding box with heightmap')
    parser.add_argument('--scene_dim', type=int, default=SCENE_DIM, help='number of points in local height map')
    parser.add_argument('--downsample_rate', type=int, default=DOWNSAMPLE_RATE, help='downsample rate of each mini sequence')
    parser.add_argument('--data_scale', type=float, default=DATA_SCALE, help='scale down data before training')
    parser.add_argument('--predict_velocity', type=bool, default=PREDICT_VELOCITY, help='Predict root velocities instead of position?')
    parser.add_argument("--gpu_id", type=int, default=GPU_ID, help='GPU id')
    parser.add_argument('--checkpoints_dir', type=str, default=CHECKPOINTS_DIR, help='models are saved here')
    parser.add_argument('--logs_dir', type=str, default='/logs', help='dir for saving checkpoints and logs')
    parser.add_argument('--stamp', type=str, default=time.strftime('%Y%m%d_%H%M%S', time.localtime()), help='timestamp')

    # Model architecture
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Name of this model')
    parser.add_argument('--code_dim', type=int, default=CODE_DIM, help='embedding dimension')
    parser.add_argument('--code_num', type=int, default=CODE_NUM, help='number of tokens in codebook')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='latent dimension for transformer')
    parser.add_argument('--ff_size', type=int, default=FF_SIZE, help='feedforward size in transformer')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='number of transformer heads')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='dropout rate')
    parser.add_argument('--t5_dim', type=int, default=T5_DIM, help='T5 embedding dimension')
    parser.add_argument('--cond_drop_prob', type=float, default=COND_DROP_PROB, help='Conditional dropout probability')
    parser.add_argument('--use_rope', type=bool, default=USE_ROPE, help='Use rotary positional encoding (RoPE)')
    parser.add_argument('--use_local_grid', type=bool, default=use_local_grid, help='Use global occupancy grid as well')
    parser.add_argument('--use_rendered_image', type=bool, default=USE_RENDERED_IMAGES, help='Use top-view rendered images of scene')
    parser.add_argument('--use_multiscale', action='store_true', default=False, help='Use multi-scale scene encoder')
    parser.add_argument('--frame_embedding_dim', type=int, default=FRAME_EMBEDDING_DIM, help='Frame embedding dimension')
    # AutoregressiveMotionGenerator specific arguments
    parser.add_argument('--motion_vocab_size', type=int, default=MOTION_VOCAB_SIZE, help='Motion vocabulary size')
    parser.add_argument('--dino_feature_dim', type=int, default=DINO_FEATURE_DIM, help='DINO feature dimension')
    parser.add_argument('--text_feature_dim', type=int, default=TEXT_FEATURE_DIM, help='Text feature dimension (T5)')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='Hidden dimension for transformer')
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help='Maximum sequence length for motion tokens')
    parser.add_argument('--tokens_per_step', type=int, default=TOKENS_PER_STEP, help='Number of tokens to generate per step')
    parser.add_argument('--t5_model_name', type=str, default=T5_MODEL_NAME, help='T5 model name for text encoding')
    parser.add_argument('--freeze_t5', type=bool, default=FREEZE_T5, help='Freeze T5 parameters during training')
    parser.add_argument('--dino_patch_size', type=int, default=DINO_PATCH_SIZE, help='DINO patch size')
    parser.add_argument('--dino_num_patches', type=int, default=DINO_NUM_PATCHES, help='Number of DINO patches')
    parser.add_argument('--use_dino_patches', type=bool, default=USE_DINO_PATCHES, help='Use patch-wise DINO features')

    # Training and optimization
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of this trial')
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--step_size', default=STEP_SIZE, type=int, help='learning rate schedule (epochs)')
    parser.add_argument('--gamma', default=GAMMA, type=float, help='learning rate decay')
    parser.add_argument('--warm_up_iter', default=WARM_UP_ITER, type=float, help='warm_up_iteration')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size to train')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=GRAD_CLIP, help='max grad norm for clipping')
    parser.add_argument('--overfit_num_samples', type=int, default=OVERFIT_NUM_SAMPLES, help='number of samples to keep for overfit debugging (<=0 uses all)')
    parser.add_argument('--overfit_single_batch', action='store_true', default=OVERFIT_SINGLE_BATCH, help='Repeat a single cached batch every step for overfit sanity check')
    parser.add_argument('--debug_print_every', type=int, default=DEBUG_PRINT_EVERY, help='Iteration interval for printing diagnostics (token ranges, grad norm)')
    parser.add_argument('--disable_loss_masking', action='store_true', default=DISABLE_LOSS_MASKING, help='Disable random loss masking schedule in AutoregressiveMotionGenerator')
    parser.add_argument('--ff_mult', type=int, default=FF_MULT, help='Multiplier for transformer decoder feedforward size')
    parser.add_argument('--use_pre_decoder_layer_norm', action='store_true', default=USE_PRE_DECODER_LAYER_NORM, help='Apply LayerNorm to embeddings before the decoder')
    parser.add_argument('--change_lr', type=bool, default=CHANGE_LR, help='change learning rate on resume')
    parser.add_argument('--num_epoch', type=int, default=NUM_EPOCH, help='#epochs to train')
    parser.add_argument('--use_cuda', type=bool, default=USE_CUDA, help='set device for training')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='number of dataloader worker processes')
    parser.add_argument('--is_continue', action='store_true', help='Continue training from checkpoint')
    parser.add_argument('--log_every_e', default=LOG_EVERY_E, type=int, help='epoch log frequency')
    parser.add_argument('--save_every_e', default=SAVE_EVERY_E, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=EVAL_EVERY_E, type=int, help='eval every n epoch')
    parser.add_argument('--which_epoch', type=str, default='all', help='Which epoch to evaluate')

    # Resume/Load
    parser.add_argument('--resume_model', type=str, default='', help='resume model path')
    parser.add_argument('--load_exp', type=str, default=load_exp, help='Path to load experiment checkpoint')

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
    parser.add_argument('--weight_loss_consistency',
                        type=float,
                        default=weight_loss_consistency,
                        help='loss weight of consistency between VQVAE and refined output')
    parser.add_argument('--weight_loss_temporal',
                        type=float,
                        default=weight_loss_temporal,
                        help='loss weight of temporal consistency loss')
    parser.add_argument('--all_body_vertices',
                        action="store_true",
                        help='use all body vertices to regress')
    parser.add_argument('--weight_loss_adversarial',
                    type=float,
                    default=weight_loss_adversarial,
                    help='loss weight of adversarial loss')
    

    # Custom MaskedTransformer options can be added here

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