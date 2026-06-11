# import cv2
import argparse
import io
import json
import math
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0) 
import os
import pickle
import random      
import time
import torch
torch.manual_seed(0)
import yaml
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, List
from dataset_prep.trumans_paths import SMPLX_BODY_MODELS_DIR, SMPLX_MODEL_DIR
# from scipy.ndimage import gaussian_filter


colors = {
    'pink': [1.00, 0.75, 0.80],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}
GPU_ID = 0  # -1 to use cpu
if torch.cuda.device_count() == 1: 
    GPU_ID = 0
to_cpu = lambda tensor: tensor.detach().cpu().numpy()
DEVICE = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() and GPU_ID >=0 else "cpu")
SMPLX_FOLDER = SMPLX_MODEL_DIR
BODY_MODEL_FOLDER = SMPLX_BODY_MODELS_DIR
# scan2cad_anno = '/home/wangzan/Data/scan2cad/scan2cad_download_link/full_annotations.json'
# scannet_folder = '/mnt/d/SRC/DATASETS/ScanNet/scans/'
# referit3d_sr3d = '/home/wangzan/Data/referit3d/sr3d.csv'

def convert_numpy_to_tensor(obj):
    """
    Recursively convert all numpy arrays in a structure to torch tensors,
    while leaving string arrays or objects untouched.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_tensor(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_numpy_to_tensor(i) for i in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_tensor(i) for i in obj)

    elif isinstance(obj, np.ndarray):
        # If it's a string or object dtype, leave it alone
        if obj.dtype.kind in {'U', 'S', 'O'}:
            return obj.tolist()  # or optionally convert to list with `obj.tolist()`
        else:
            return to_tensor(obj, device="cuda:0")

    else:
        return obj


def convert_tensor_to_numpy(obj):
    """
    Recursively convert all torch tensors in a structure to numpy arrays,
    while leaving string arrays or objects untouched.
    """
    if isinstance(obj, dict):
        return {k: convert_tensor_to_numpy(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_tensor_to_numpy(i) for i in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_tensor(i) for i in obj)

    elif isinstance(obj, torch.Tensor):
        # If it's a string or object dtype, leave it alone
        # if obj.dtype.kind in {'U', 'S', 'O'}:
        #     return obj.tolist()  # or optionally convert to list with `obj.tolist()`
        # else:
        return obj.detach().cpu().numpy()

    else:
        return obj

def to_tensor(array, dtype=torch.float32, device='cpu'):
    if not torch.is_tensor(array):
        array = torch.from_numpy(array)
    return array.to(dtype).to(device)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class _Printer():

    def __init__(self) -> None:
        self._printer = print
        self._debug = False

    def print(self, debug: bool, *args: List[str]) -> None:
        if debug and not self._debug:
            return
        
        args = list(map(str, args))
        self._printer(' '.join(args))

    def setPrinter(self, **kwargs) -> None:
        if 'printer' in kwargs:
            self._printer = kwargs['printer']
        if 'debug' in kwargs:
            self._debug = kwargs['debug']

class Console(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def setPrinter(**kwargs) -> None:
        """ Set printer for Console

        Args:
            printer: Callable object, core output function
            debug: bool type, whether output debug information 
        """
        p = _Printer()
        p.setPrinter(**kwargs)
    
    @staticmethod
    def log(*args: List[Any]) -> None:
        """ Output log information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(False, *args)
    
    @staticmethod
    def debug(*args: List[Any]) -> None:
        """ Output debug information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(True, *args)

@singleton
class _Writer():
    def __init__(self) -> None:
        self.writer = None

    def write(self, write_dict: dict) -> None:
        if self.writer is None:
            raise Exception('[ERR-CFG] Writer is None!')
        
        for key in write_dict.keys():
            if write_dict[key]['plot']:
                self.writer.add_scalar(key, write_dict[key]['value'], write_dict[key]['step'])

    def setWriter(self, writer: SummaryWriter) -> None:
        self.writer = writer

class Ploter():
    def __init__(self) -> None:
        pass

    @staticmethod
    def setWriter(writer: SummaryWriter) -> None:
        w = _Writer()
        w.setWriter(writer)
    
    @staticmethod
    def write(write_dict: dict) -> None:
        w = _Writer()
        w.write(write_dict)
        



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(split, print_filename, time_elapsed, niter_state, losses, accuracy=None, epoch=None, lr=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(start_time):
        now = time.time() - start_time
        seconds = int(now % 60)
        return seconds

    if epoch is not None:
        message = split + '- epoch: {:>4d}, niter: {:>6d}, secs/ep: {:.4f}, lr: {:.6f}'.format(epoch, niter_state, time_elapsed, lr)

    # for k, v in losses.items():
    #     message += ' %s: %.4f ' % (k, v)
    message += ' Mean Loss: {:.8f}'.format(losses)
    if accuracy is not None:
        message += ' Acc: {:.5f}'.format(accuracy)
    print(message)
    with open(print_filename, 'a') as f:
        print(message, file=f)


def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def plot_loss_curve(losses, save_path, intervals=500):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    for key in losses.keys():
        plt.plot(list_cut_average(losses[key], intervals), label=key)
    plt.xlabel("Iterations/" + str(intervals))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


# def motion_temporal_filter(motion, sigma=1):
#     motion = motion.reshape(motion.shape[0], -1)
#     # print(motion.shape)
#     for i in range(motion.shape[1]):
#         motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
#     return motion.reshape(motion.shape[0], -1, 3)

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def get_opt(opt_path):
    with open(opt_path) as f: 
        file_lines = f.read()
    args = json.loads(file_lines)
    opt = argparse.Namespace(**args)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    return opt

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
        else:
            return super().find_class(module, name)

def clean_suffix(s):
    for suffix in ['_augment1', '_augment2']:
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s