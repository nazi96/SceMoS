"""Dataset loader for TRUMANS motion, geometry, and conditioning features."""

import glob
import os
import math
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import torch
from scipy.spatial.transform import Rotation as R
from utils.smplx_util import SMPLX_Util
from torch.utils.data import Dataset, DataLoader, Subset
from utils.utilities import *
from utils.smplx_util import *
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_prep.trumans_paths import (
    preprocessed_trumans_dir,
    preprocessed_trumans_ply_dir,
    resolve_motion_data_path,
    resolve_trumans_data_root,
    test_samples_path,
    train_samples_path,
)

DATA_PATH = resolve_trumans_data_root()
PREPROCESSED_DATA_FOLDER = preprocessed_trumans_dir()
PREPROCESSED_PLY_DATA_FOLDER = preprocessed_trumans_ply_dir()
SCENE_OCC_PATH = os.path.join(DATA_PATH, "Scene")
SCENE_MESH_PATH = os.path.join(DATA_PATH, "Scene_mesh")
PREPROCESSED_SCENE_PATH = os.path.join(DATA_PATH, "Scene_data", "main")
PREPROCESSED_OBJ_ALL_PATH = os.path.join(DATA_PATH, "Object_all", "Obj_data", "main")
PREPROCESSED_OBJ_CHAIRS_PATH = os.path.join(DATA_PATH, "Object_chairs", "Obj_data", "main")
ANNOTATION_PATH = os.path.join(DATA_PATH, "Actions")
OBJECT_ALL_MESH_PATH = os.path.join(DATA_PATH, "Object_all", "Object_mesh")
OBJECT_CHAIRS_MESH_PATH = os.path.join(DATA_PATH, "Object_chairs", "Object_mesh")
SCENE_BEV_RECORDING_PATH = os.path.join(DATA_PATH, "Recordings", "BEV_1")
SCENE_DINO_FEATS_PATH = os.path.join(DATA_PATH, "Recordings", "dinov3", "specialized_features_output")

NB_VOXELS = 16  # same as TRUMANS
BBOX_SCALE = 0.6  # same as TRUMANS
NUM_MAX_SCENE_PTS = 250000
SEQ_LEN = 32 # same as Humanise
STEP = 3
TRAIN_SPLIT = os.path.join(DATA_PATH, 'train_scenes.pkl')
TRAIN_SAMPLES = train_samples_path()
TEST_SPLIT = os.path.join(DATA_PATH, 'test_scenes.pkl')
TEST_SAMPLES = test_samples_path()


def get_nearest_texts(action_dict, start, end):
    result = " ".join(
        action['text'] + "."
        for action in action_dict
        if action['start_frame']//STEP <= end and action['end_frame']//STEP >= start
    )
    return result
   

class TrumansDataset(Dataset):
    def __init__(self, phase: str, window_size: int = 21, predict_velocity=True, downsample_rate: int = 1,
                 device='cuda:0', load_scene_vertex=False, few=False, extra=False, load_dino_feats=False):
        self.phase = phase
        self.extra = extra
        self.load_dino_feats = load_dino_feats
        self.load_scene_vertex = load_scene_vertex
        self.predict_velocity = predict_velocity
        self.device = device
        self.window_size = window_size
        self.downsample_rate = downsample_rate
        self.target_seq_len = math.ceil(self.window_size / self.downsample_rate)
        
        # Load normalization parameters and move to GPU
        self.mean = to_tensor(np.load(os.path.join(PREPROCESSED_DATA_FOLDER, 'Mean.npy'))).to(self.device)
        self.std = to_tensor(np.load(os.path.join(PREPROCESSED_DATA_FOLDER, 'Std.npy'))).to(self.device)
        
        # Pre-allocate tensors for better memory management
        self.temp_tensors = {}
        
        # Load sample paths
        if phase == 'train':
            selected_samples = list(np.load(TRAIN_SAMPLES, allow_pickle=True))
        elif phase == 'test':
            selected_samples = list(np.load(TEST_SAMPLES, allow_pickle=True))
        elif phase == 'all':
            selected_samples = (
                list(np.load(TRAIN_SAMPLES, allow_pickle=True))
                + list(np.load(TEST_SAMPLES, allow_pickle=True))
            )
        else:
            raise ValueError(f"Unknown phase: {phase!r}")

        if few:
            self.motion_data_list = [
                resolve_motion_data_path(str(selected_samples[i])) for i in range(20)
            ]
        else:
            self.motion_data_list = [
                resolve_motion_data_path(str(selected_samples[i]))
                for i in range(len(selected_samples))
            ]
        
        # Pre-load all motion data during initialization
        print(f"Loading {len(self.motion_data_list)} motion data examples for {self.phase} set...")
        self.motion_data = []
        loaded_motion_data_list = []
        skipped_corrupt = 0
        skipped_missing = 0
        for i, data_path in enumerate(tqdm(self.motion_data_list)):
            if not os.path.exists(data_path):
                skipped_missing += 1
                continue
            try:
                with open(data_path, 'rb') as fp:
                    data = Unpickler(fp).load()
            except (pickle.UnpicklingError, EOFError, OSError, ValueError) as e:
                # Truncated / corrupt pickle - skip this file and keep loading the rest.
                skipped_corrupt += 1
                print(f"[trumans_loader] Skipping corrupt pickle '{data_path}': {type(e).__name__}: {e}")
                continue
            self.motion_data.append(data)
            loaded_motion_data_list.append(data_path)

        self.motion_data_list = loaded_motion_data_list
        if skipped_missing > 0 or skipped_corrupt > 0:
            print(f"[trumans_loader] Skipped {skipped_missing} missing and {skipped_corrupt} corrupt files.")
        Console.log('Total {} motion data examples loaded in {} set.'.format(len(self.motion_data_list), self.phase))
        self.end_index = -1
        self.index = 6
        self.count = 0
    
    def __len__(self):
        return len(self.motion_data_list)

    def _slice_with_pad(self, tensor_data, start, end):
        sliced = tensor_data[start:end:self.downsample_rate]
        current_len = sliced.shape[0]
        if current_len == self.target_seq_len:
            return sliced
        if current_len == 0:
            pad_shape = (self.target_seq_len,) + tuple(tensor_data.shape[1:])
            return torch.zeros(pad_shape, dtype=tensor_data.dtype, device=tensor_data.device)
        if current_len > self.target_seq_len:
            return sliced[:self.target_seq_len]

        pad_count = self.target_seq_len - current_len
        last_frame = sliced[-1:].expand(pad_count, *sliced.shape[1:])
        return torch.cat([sliced, last_frame], dim=0)
    
    def z_normalization(self, data):
        """
        Apply z-normalization to the data using pre-computed mean and std.
        
        Args:
            data: Tensor to normalize
            
        Returns:
            Normalized tensor
        """
        # Ensure data is on the correct device
        data = data.to(self.device)
        return (data - self.mean) / (self.std + 1e-8)
    
    def inv_z_normalization(self, data):
        """
        Apply inverse z-normalization to the data using pre-computed mean and std.
        
        Args:
            data: Tensor to denormalize
            
        Returns:
            Denormalized tensor
        """
        # Ensure data is on the correct device
        data = data.to(self.device)
        return data * self.std + self.mean

    def find_contact_vertices(self, local_grid_vertices, person_vertices, contact_threshold=0.05, max_vertices=2000):
        """
        Identifies vertices from local_grid_vertices that come in contact with the person's vertices.
        Returns a fixed number of vertices (max_vertices) by selecting the closest ones or padding with zeros.
        
        Args:
            local_grid_vertices: Tensor [N, 3] with 3D coordinates of scene vertices
            person_vertices: Tensor [M, 3] with 3D coordinates of person vertices
            contact_threshold: float, distance threshold to consider vertices in contact (default: 0.05)
            max_vertices: int, maximum number of vertices to return (default: 100)
            
        Returns:
            contact_vertices: Tensor [max_vertices, 3] with 3D coordinates of scene vertices in contact with the person
            contact_indices: Tensor [max_vertices] with indices of contact vertices in local_grid_vertices
        """
        # Ensure inputs are on the correct device
        local_grid_vertices = local_grid_vertices.to(self.device)
        person_vertices = person_vertices.to(self.device)
        
        # Filter out ground vertices (y=0)
        non_ground_mask = local_grid_vertices[:, 1] > 0.01  # [N]
        non_ground_vertices = local_grid_vertices[non_ground_mask]  # [K, 3]
        
        if non_ground_vertices.shape[0] == 0:
            # If no non-ground vertices, return zero tensors with the correct shape
            return torch.zeros((max_vertices, 3), device=self.device), torch.zeros(max_vertices, dtype=torch.long, device=self.device)
        
        # Compute distances between all scene vertices and all person vertices
        # Reshape for broadcasting: [K, 1, 3] - [1, M, 3] = [K, M, 3]
        distances = torch.cdist(non_ground_vertices, person_vertices)  # [K, M]
        
        # Find the minimum distance from each scene vertex to any person vertex
        min_distances = torch.min(distances, dim=1)[0]  # [K]
        
        # Create a mask for vertices within the contact threshold
        contact_mask = min_distances < contact_threshold  # [K]
        
        # Get the contact vertices
        contact_vertices = non_ground_vertices[contact_mask]  # [L, 3]
        
        # Get the original indices of the contact vertices in local_grid_vertices
        # We need to map back from the filtered non_ground_vertices to the original local_grid_vertices
        non_ground_indices = torch.where(non_ground_mask)[0]  # [K]
        contact_indices = non_ground_indices[contact_mask]  # [L]
        
        # If we have more contact vertices than max_vertices, select the closest ones
        if contact_vertices.shape[0] > max_vertices:
            # Get the distances of the contact vertices
            contact_distances = min_distances[contact_mask]  # [L]
            
            # Sort by distance and get the indices of the closest vertices
            _, closest_indices = torch.sort(contact_distances)
            closest_indices = closest_indices[:max_vertices]
            
            # Select the closest vertices
            contact_vertices = contact_vertices[closest_indices]  # [max_vertices, 3]
            contact_indices = contact_indices[closest_indices]  # [max_vertices]
        
        # If we have fewer contact vertices than max_vertices, pad with zeros
        elif contact_vertices.shape[0] < max_vertices:
            # Create zero tensors for padding
            padding_vertices = torch.zeros((max_vertices - contact_vertices.shape[0], 3), device=self.device)
            padding_indices = torch.zeros(max_vertices - contact_indices.shape[0], dtype=torch.long, device=self.device)
            
            # Concatenate the contact vertices with the padding
            contact_vertices = torch.cat([contact_vertices, padding_vertices], dim=0)  # [max_vertices, 3]
            contact_indices = torch.cat([contact_indices, padding_indices], dim=0)  # [max_vertices]
        
        return contact_vertices, contact_indices
    
    def create_scene_occupancy_grid(self, scene_vertices, grid_size=32):
        """
        Creates an occupancy grid from scene vertices where y position denotes height.
        If y is close to floor (y ≈ 0), occupancy is 0, else 1.
        
        Args:
            scene_vertices: Tensor [N, 3] with 3D coordinates of scene vertices
            grid_size: int, size of the output grid (default: 32)
            
        Returns:
            occupancy_grid: Tensor [grid_size, grid_size] with binary occupancy values
        """
        # Ensure vertices are on the correct device
        scene_vertices = scene_vertices.to(self.device)
        
        # Get min and max x,z coordinates to define grid boundaries
        min_xz = torch.min(scene_vertices[:, [0, 2]], dim=0)[0]
        max_xz = torch.max(scene_vertices[:, [0, 2]], dim=0)[0]
        
        # Create grid coordinates
        x = torch.linspace(min_xz[0], max_xz[0], grid_size, device=self.device)
        z = torch.linspace(min_xz[1], max_xz[1], grid_size, device=self.device)
        xx, zz = torch.meshgrid(x, z, indexing='ij')
        
        # Reshape grid coordinates for distance computation
        grid_points = torch.stack([xx.flatten(), zz.flatten()], dim=1)  # [grid_size*grid_size, 2]
        
        # Compute distances between grid points and scene vertices
        distances = torch.cdist(grid_points, scene_vertices[:, [0, 2]])  # [grid_size*grid_size, N]
        
        # Find nearest vertex for each grid point
        nearest_idx = torch.argmin(distances, dim=1)  # [grid_size*grid_size]
        
        # Get heights of nearest vertices
        nearest_heights = scene_vertices[nearest_idx, 1]  # [grid_size*grid_size]
        
        # Create occupancy grid: 0 if height is close to floor, 1 otherwise
        floor_threshold = 0.01  # Threshold to consider a vertex as being on the floor
        occupancy = (nearest_heights > floor_threshold).float()
        
        # Reshape to grid
        occupancy_grid = occupancy.reshape(grid_size, grid_size)
        
        return occupancy_grid

    def pointcloud_to_occupancy_grid(self, local_scene_vertices, grid_size=32):
        """
        Converts a point cloud [N, 3] to a 3D occupancy grid of shape [grid_size, grid_size, grid_size].
        Args:
            local_scene_vertices: torch.Tensor [N, 3]
            grid_size: int, number of voxels per dimension
        Returns:
            occupancy_grid: torch.Tensor [grid_size, grid_size, grid_size] (bool)
        """
        # Ensure tensor and device
        points = local_scene_vertices.to(self.device)
        N = points.shape[0]
        # Compute bounding box
        bbox_min = torch.min(points, dim=0)[0]
        bbox_max = torch.max(points, dim=0)[0]
        # Avoid degenerate box
        eps = 1e-6
        bbox_max = torch.where(bbox_max - bbox_min < eps, bbox_min + eps, bbox_max)
        # Compute voxel indices
        voxel_size = (bbox_max - bbox_min) / grid_size
        indices = ((points - bbox_min) / voxel_size).long()
        indices = torch.clamp(indices, 0, grid_size - 1)
        # Create occupancy grid
        occupancy = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=self.device)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return occupancy

    def pointcloud_to_height_occupancy_grid(self, local_scene_vertices, grid_size=32, y_threshold=0.05):
        """
        Converts a point cloud [N, 3] to a 3D occupancy grid of shape [grid_size, grid_size, grid_size],
        but only marks voxels as occupied if the y (height) value is above y_threshold (i.e., not ground).
        Args:
            local_scene_vertices: torch.Tensor [N, 3]
            grid_size: int, number of voxels per dimension
            y_threshold: float, minimum y value to be considered above ground
        Returns:
            occupancy_grid: torch.Tensor [grid_size, grid_size, grid_size] (bool)
        """
        points = local_scene_vertices.to(self.device)
        bbox_min = torch.min(points, dim=0)[0]
        bbox_max = torch.max(points, dim=0)[0]
        eps = 1e-6
        bbox_max = torch.where(bbox_max - bbox_min < eps, bbox_min + eps, bbox_max)
        voxel_size = (bbox_max - bbox_min) / grid_size
        indices = ((points - bbox_min) / voxel_size).long()
        indices = torch.clamp(indices, 0, grid_size - 1)
        occupancy = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=self.device)
        # Only use points above the threshold
        mask = points[:, 1] > y_threshold
        indices = indices[mask]
        if indices.shape[0] > 0:
            occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return occupancy

    def __getitem__(self, index):
        annot_data = {}  
        
        # Validation time index
        index = self.index



        index = index % len(self.motion_data)
        datas = self.motion_data[index]
        annot_data['scene_name'] = datas['scene_name']
        snippet_len = len(datas['mot_feats'])

        # validation-time sequential
        max_start = max(0, snippet_len - self.window_size)
        start = min(self.end_index + 1, max_start)
        end = start + self.window_size
        self.end_index = end
        self.count += 1
        if self.count >= 6:
            self.index += 1
            self.count = 0
            self.end_index = -1

        annot_data['sample_name'] = self.motion_data_list[index]
        annot_data['start_seq'] = start
        annot_data['end_seq'] = end
        annot_data['scene_occ'] = np.load(os.path.join(SCENE_OCC_PATH, str(datas['scene_name'] + '.npy')))
        
        # Move data to GPU in batches to reduce CPU-GPU transfer overhead
        annot_data['betas'] = datas['person_data']['betas'].to(self.device)
        if self.load_scene_vertex:
            annot_data['scene_vertex'] = datas['scene_vertex'].to(self.device)
        annot_data['local_vertices'] = self._slice_with_pad(datas['local_grid_vertices'].to(self.device), start, end)
        # Create scene occupancy grid
        # annot_data['scene_occupancy_grid'] = self.create_scene_occupancy_grid(annot_data['scene_vertex'])
        # Create local occupancy grid for each frame in window
        local_occ_list = []
        local_height_occ_list = []
        for frame_vertices in annot_data['local_vertices']:
            local_occ = self.pointcloud_to_occupancy_grid(frame_vertices, grid_size=32)
            local_height_occ = self.pointcloud_to_height_occupancy_grid(frame_vertices, grid_size=32)
            local_occ_list.append(local_occ)
            local_height_occ_list.append(local_height_occ)
        annot_data['local_occupancy_grid'] = torch.stack(local_occ_list)  # [window_size, 32, 32, 32]
        annot_data['local_height_occupancy_grid'] = torch.stack(local_height_occ_list)  # [window_size, 32, 32, 32]
        annot_data['texts'] = get_nearest_texts(datas['annotation_actions'], start, end)
        
        # Process motion features on GPU
        annot_data['motion_features'] = self._slice_with_pad(datas['mot_feats'].to(self.device), start, end)
        grid_xyz = self._slice_with_pad(to_tensor(datas['grid_xyz'], device=self.device), start, end)
        annot_data['action_label'] = self._slice_with_pad(to_tensor(datas['action_label'], device=self.device), start, end)
        annot_data['grid_xyz'] = grid_xyz.permute(0, 3, 1, 2)
        annot_data['heightmap'] = grid_xyz[..., 1]
        annot_data['body_joints'] = self._slice_with_pad(datas['body_joints'].to(self.device), start, end)
        annot_data['contact_vertices'] = self._slice_with_pad(datas['contact_vertices'].to(self.device), start, end)
        
        # # Compute contact map on GPU
        # annot_data['contact_map'] = torch.linalg.norm(annot_data['contact_vertices'] - annot_data['body_joints'][:, 0:1], dim=-1)
        
        # Mask: True where A has non-zero (x, y, z)
        non_zero_mask = (annot_data['contact_vertices'].to(self.device).abs().sum(dim=-1) != 0)  # [32, 2000]

        # Euclidean distance between each annot_data['contact_vertices'] and annot_data['body_joints'][:, 0:1]
        dist = torch.linalg.norm(annot_data['contact_vertices'].to(self.device) - annot_data['body_joints'][:, 0:1].to(self.device), dim=-1)  # [32, 2000]

        # Set distance to zero where A was all-zero
        annot_data['contact_map'] = torch.where(non_zero_mask, dist, torch.zeros_like(dist).to(self.device))  # [32, 2000]
        # Normalize motion features on GPU
        normalized_mot_feats = self.z_normalization(annot_data['motion_features'].to(self.device))
        
        if not self.predict_velocity:
            annot_data['normalized_mot_feats'] = normalized_mot_feats
        else:
            annot_data['mot_init'] = normalized_mot_feats[0:1]
            # Create motion_root_delta directly on GPU
            motion_root_delta = torch.zeros((len(normalized_mot_feats), normalized_mot_feats.shape[-1]), device=self.device)
            motion_root_delta[1:, :3] = normalized_mot_feats[1:, :3] - normalized_mot_feats[:-1, :3]
            motion_root_delta[:, 3:] = normalized_mot_feats[:, 3:].clone()
            annot_data['normalized_mot_feats'] = motion_root_delta

        if self.extra:
            # Process extra data on GPU
            annot_data['person_data'] = {
                'transl': self._slice_with_pad(datas['person_data']['transl'].to(self.device), start, end),
                'global_orient': self._slice_with_pad(datas['person_data']['global_orient'].to(self.device), start, end),
                'pose_hand': self._slice_with_pad(datas['person_data']['pose_hand'].to(self.device), start, end),
                'pose_body': self._slice_with_pad(datas['person_data']['pose_body'].to(self.device), start, end),
                'betas' : datas['person_data']['betas'].to(self.device),
                'meta': datas['person_data']['meta']
                }
        if self.load_dino_feats:
            # Extract motion name from sample path
            motion_name = os.path.splitext(os.path.basename(annot_data['sample_name']))[0].rsplit('_heightmap_contacts')[0]
            
            # Construct the base DINO features path: SCENE_DINO_FEATS_PATH/scene_name/motion_name
            dino_feats_base_path = os.path.join(SCENE_DINO_FEATS_PATH, annot_data['scene_name'], motion_name)
            
            # Calculate the feature index nearest to start frame
            # Features are named as 0001, 0011, 0021, etc. (at 10-frame intervals)
            nearest_feature_idx = (start // 10) * 10 + 1
            
            # Load only the feature at the nearest index to start
            feature_path = os.path.join(dino_feats_base_path, f"{nearest_feature_idx:04d}_unpooled.npy")
            if os.path.exists(feature_path):
                annot_data['dino_feats'] = to_tensor(np.load(feature_path), device=self.device)
            else:
                # print(f"Warning: DINO feature file not found: {feature_path}")
                annot_data['dino_feats'] = torch.zeros((1, 49, 768), device=self.device)  # Adjust dimensions as needed

        vis = False
        if vis:
            print(annot_data['texts'])
            from visualize.visualize_data import visualize_sequence
            person_vertices, person_joints = smplx_to_pos3d(annot_data['person_data'])
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            # Ensure numpy array is contiguous and float64 for Open3D Vector3dVector
            # Flatten to (N, 3) shape for point cloud
            points_np = np.ascontiguousarray(
                grid_xyz[0].reshape( -1, 3).detach().cpu().numpy(), 
                dtype=np.float64
            )
            pcd.points = o3d.utility.Vector3dVector(points_np)
            # Add red color to all points
            num_points = points_np.shape[0]
            red_colors = np.tile([1.0, 0.0, 0.0], (num_points, 1))  # RGB red color
            pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(red_colors, dtype=np.float64))
            heightmap_save_name = makepath(os.path.join('render', 'scenes_heightmaps', annot_data['scene_name'], 
            os.path.basename(annot_data['sample_name'])[:-4] + '_' + str(start) + '_' + str(end) + '.ply'), isfile=True)
            o3d.io.write_point_cloud(heightmap_save_name, pcd)
            
            # Save person vertices as point cloud
            pcd_person = o3d.geometry.PointCloud()
            # Ensure numpy array is contiguous and float64 for Open3D Vector3dVector
            person_vertices_np = np.ascontiguousarray(
                person_vertices[0].detach().cpu().numpy().reshape(-1, 3), 
                dtype=np.float64
            )
            pcd_person.points = o3d.utility.Vector3dVector(person_vertices_np)
            # Add red color to all person vertices
            num_person_points = person_vertices_np.shape[0]
            person_red_colors = np.tile([1.0, 0.0, 0.0], (num_person_points, 1))  # RGB red color
            pcd_person.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(person_red_colors, dtype=np.float64))
            person_vertices_save_name = makepath(os.path.join('render', 'scenes_heightmaps', annot_data['scene_name'], 
            os.path.basename(annot_data['sample_name'])[:-4] + '_' + str(start) + '_' + str(end) + '_person.ply'), isfile=True)
            o3d.io.write_point_cloud(person_vertices_save_name, pcd_person)
            
            visualize_sequence(scene_data=[datas['scene_vertex'][None].detach().cpu().numpy(), None], 
                            person1=[person_vertices.detach().cpu().numpy(), None],
                            item3=[annot_data['local_vertices'].detach().cpu().numpy(), None],
                            item4=[(annot_data['contact_vertices']).detach().cpu().numpy(), None],
                            item5=[(grid_xyz.reshape(self.window_size, -1, 3) ).detach().cpu().numpy(), None],
                            is_permute=False)
            # --- Occupancy grid visualization ---
            if 'local_occupancy_grid' in annot_data and 'local_vertices' in annot_data:
                occ = annot_data['local_occupancy_grid'][0].cpu().numpy()  # first frame
                xs, ys, zs = np.where(occ)
                local_vertices = annot_data['local_vertices'][0]
                bbox_min = local_vertices.min(dim=0)[0]
                bbox_max = local_vertices.max(dim=0)[0]
                grid_size = occ.shape[0]
                voxel_size = (bbox_max - bbox_min) / grid_size
                centers = (np.stack([xs, ys, zs], axis=1) + 0.5) * voxel_size.cpu().numpy() + bbox_min.cpu().numpy()
                fig = plt.figure(figsize=(18, 6))
                # --- Pure voxel grid ---
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(centers[:, 0], centers[:, 2], centers[:, 1], s=2, c='red', label='Occupied Voxels (Pure)')
                ax1.scatter(local_vertices[:, 0].cpu(), local_vertices[:, 2].cpu(), local_vertices[:, 1].cpu(), s=1, c='blue', alpha=0.1, label='Original Points')
                ax1.set_title('Pure Occupancy Grid (red) vs Local Vertices (blue)')
                ax1.set_xlabel('X'); ax1.set_ylabel('Z'); ax1.set_zlabel('Y (up)')
                ax1.legend()
                # --- Height-based occupancy grid ---
                occ_height = self.pointcloud_to_height_occupancy_grid(local_vertices, grid_size=grid_size, y_threshold=0.05).cpu().numpy()
                xs_h, ys_h, zs_h = np.where(occ_height)
                centers_h = (np.stack([xs_h, ys_h, zs_h], axis=1) + 0.5) * voxel_size.cpu().numpy() + bbox_min.cpu().numpy()
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(centers_h[:, 0], centers_h[:, 2], centers_h[:, 1], s=2, c='green', label='Occupied Voxels (Height)')
                ax2.scatter(local_vertices[:, 0].cpu(), local_vertices[:, 2].cpu(), local_vertices[:, 1].cpu(), s=1, c='blue', alpha=0.1, label='Original Points')
                ax2.set_title('Height-based Occupancy Grid (green)')
                ax2.set_xlabel('X'); ax2.set_ylabel('Z'); ax2.set_zlabel('Y (up)')
                ax2.legend()
                # --- Overlay both ---
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(centers[:, 0], centers[:, 2], centers[:, 1], s=2, c='red', label='Pure Voxel')
                ax3.scatter(centers_h[:, 0], centers_h[:, 2], centers_h[:, 1], s=2, c='green', label='Height-based Voxel')
                ax3.scatter(local_vertices[:, 0].cpu(), local_vertices[:, 2].cpu(), local_vertices[:, 1].cpu(), s=1, c='blue', alpha=0.1, label='Original Points')
                ax3.set_title('Overlay: Pure (red) + Height (green)')
                ax3.set_xlabel('X'); ax3.set_ylabel('Z'); ax3.set_zlabel('Y (up)')
                ax3.legend()
                plt.tight_layout()
                plt.show()
            
        return annot_data

    # def create_2d_grid(self, top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, grid_size=32):
    #     """
    #     Creates a 2D grid that coincides with the local_grid_vertices structure.
        
    #     Args:
    #         top_left_corner: Tensor [2], top-left corner of the grid in world coordinates (x, z).
    #         top_right_corner: Tensor [2], top-right corner of the grid in world coordinates (x, z).
    #         bottom_left_corner: Tensor [2], bottom-left corner of the grid in world coordinates (x, z).
    #         bottom_right_corner: Tensor [2], bottom-right corner of the grid in world coordinates (x, z).
    #         forward: Tensor [2], unit vector in forward direction (x, z).
    #         grid_size: int, size of the output grid (default: 32).
    #         cell_size: float, size of each grid cell in meters (default: 0.5m).
        
    #     Returns:
    #         grid_xz: [grid_size, grid_size, 2] tensor with world coordinates of each grid cell (x, z).
    #     """
       
    #     # Create left and right line segments
    #     left_ls = torch.stack([
    #         torch.linspace(bottom_left_corner[i], top_left_corner[i], grid_size) 
    #         for i in range(2)  # Only using x and z dimensions
    #     ], dim=1)
        
    #     right_ls = torch.stack([
    #         torch.linspace(bottom_right_corner[i], top_right_corner[i], grid_size) 
    #         for i in range(2)  # Only using x and z dimensions
    #     ], dim=1)
        
    #     # Create the grid by interpolating between left and right lines
    #     grid_xz = torch.empty((grid_size, grid_size, 2)).to(self.device)
    #     for idx in range(grid_size):
    #         grid_xz[idx] = torch.stack([
    #             torch.linspace(left_ls[idx, i], right_ls[idx, i], grid_size) 
    #             for i in range(2)  # Only using x and z dimensions
    #         ], dim=-1)
        
    #     return grid_xz

if __name__ == "__main__":
    
    dataset = TrumansDataset(phase='train', window_size=80, device=DEVICE, load_scene_vertex=False, load_dino_feats=True, extra=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=True)
    
    for i, data in enumerate(dataloader):           
        print(data['texts'])