"""Compute dataset-wide normalization statistics for motion features."""

import glob
import numpy as np
import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["MUJOCO_GL"] = "osmesa"
import pickle
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import trimesh
from natsort import natsorted
from utils.transformations import *
from utils.utilities import *
from dataset_prep.motion_process import *
from dataset_prep.trumans_loader import *
from dataset_prep.preprocess_trumans import *

def mean_variance_trumans_dataset(all_seqs):
    data_list = []
    for a in tqdm(all_seqs):
        with open(a, 'rb') as fp:
            # data = pickle.load(fp)
            data = Unpickler(fp).load()
        
        with torch.no_grad():
            data_list.append(data['mot_feats'].numpy())
        #     motion_root_delta = np.zeros((len(data['mot_feats']), data['mot_feats'].shape[-1]))
        #     motion_root_delta[1:, :3] = data['mot_feats'][1:, :3] - data['mot_feats'][:-1, :3]
        #     motion_root_delta[:, 3:] = data['mot_feats'][:, 3:].clone()
        # data_list.append(motion_root_delta)

          
    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0) + 1e-8
    np.save(os.path.join(PREPROCESSED_DATA_FOLDER, 'Mean.npy'), Mean)
    np.save(os.path.join(PREPROCESSED_DATA_FOLDER, 'Std.npy'), Std)


if __name__ == "__main__":
    datas = glob.glob(PREPROCESSED_DATA_FOLDER + '/*.pkl')
    mean_variance_trumans_dataset(datas)
    # mean_variance_humanise_dataset(aligns)
