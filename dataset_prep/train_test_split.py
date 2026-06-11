import glob
import numpy as np
import os
import pickle
import sys
import random
sys.path.append('.')
sys.path.append('..')
import torch
import trimesh
from natsort import natsorted
from dataset_prep.trumans_loader import *

def split_data(datas, phase):
        # Load the appropriate scene name list based on phase
        if phase == 'train':
            split_file = TRAIN_SPLIT
        elif phase == 'test':
            split_file = TEST_SPLIT
        else:
            raise Exception('Unexpected phase.')
            
        # Load the scene list from the pickle file
        with open(split_file, 'rb') as f:
            scene_name_list = pickle.load(f)
            
        # Convert scene_name_list to a set for O(1) lookup
        scene_name_list = [scene_name_list[i].split('\\')[-1][:-4] for i in range(len(scene_name_list))]
        scene_name_set = set(scene_name_list)
        
        # Filter data files
        motion_data_list = []
        
        # Process each data file
        for data_path in tqdm(datas, desc=f"Filtering {phase} data"):
            # Load the data file to check the scene name
            with open(data_path, 'rb') as fp:
                data = Unpickler(fp).load()
                
            # Check if the scene name is in our target set
            if str(data['scene_name']) in scene_name_set:
                motion_data_list.append(data_path)
                
        np.save(os.path.join(PREPROCESSED_DATA_FOLDER, phase+'_samples.npy'), motion_data_list, allow_pickle=True)


def split_scenes(scene_datas):
        # Sort the scene data files
    scene_datas = natsorted(scene_datas)
    
    torch.manual_seed(0)
    
    # Calculate split indices for 7:3 ratio
    total_scenes = len(scene_datas)
    train_size = int(0.7 * total_scenes)
    
    # Split the data
    train_scenes = scene_datas[:train_size]
    test_scenes = scene_datas[train_size:]
    
    # Save the splits as lists
    with open(os.path.join(DATA_PATH,'train_scenes.pkl'), 'wb') as f:
        pickle.dump(train_scenes, f)
    
    with open(os.path.join(DATA_PATH,'test_scenes.pkl'), 'wb') as f:
        pickle.dump(test_scenes, f)
    
    print(f"Total scenes: {total_scenes}")
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Test scenes: {len(test_scenes)}")
    
    # Print some example scene names
    print("\nExample train scenes:")
    for i, scene in enumerate(train_scenes):
        print(f"  {i+1}. {os.path.basename(scene)}")
    
    print("\nExample test scenes:")
    for i, scene in enumerate(test_scenes):
        print(f"  {i+1}. {os.path.basename(scene)}")
        
if __name__ == "__main__":
    
    scene_datas = glob.glob(os.path.join(DATA_PATH, 'Scene') + '\\*.npy')
    with open(os.path.join(DATA_PATH,'all_scenes.pkl'), 'wb') as f:
        pickle.dump(scene_datas, f)
    datas = glob.glob(PREPROCESSED_DATA_FOLDER + '/*.pkl')
    split_scenes(scene_datas)
    split_data(datas, phase='train')
    split_data(datas, phase='test')
    

    