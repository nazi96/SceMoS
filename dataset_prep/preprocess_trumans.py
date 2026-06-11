import argparse
import glob
import re
import numpy as np
import os
import sys
sys.path.append('.')
sys.path.append('..')
import pickle
import torch
torch.manual_seed(0)
# import trimesh
from collections import defaultdict
from  natsort import natsorted
from sklearn.neighbors import KDTree
from trimesh import transform_points
from pyquaternion import Quaternion as Q
from typing import List
from common.quaternion import *
from dataset_prep.trumans_loader import *
from dataset_prep.motion_process import *
from utils.smplx_util import *
from utils.transformations import *

DATA_PATH = os.path.join("..", "..", "DATASETS", "TRUMANS", "Data_release")
PREPROCESSED_DATA_FOLDER = os.path.join('data', 'trumans')
PREPROCESSED_PLY_DATA_FOLDER = os.path.join('data', 'trumans_scene_ply')
SCENE_OCC_PATH = os.path.join(DATA_PATH, "Scene")
SCENE_MESH_PATH = os.path.join(DATA_PATH, "Scene_mesh")
PREPROCESSED_SCENE_PATH = os.path.join(DATA_PATH, "Scene_data", "main")
PREPROCESSED_OBJ_ALL_PATH = os.path.join(DATA_PATH, "Object_all", "Obj_data", "main")
PREPROCESSED_OBJ_CHAIRS_PATH = os.path.join(DATA_PATH, "Object_chairs", "Obj_data", "main")
ANNOTATION_PATH = os.path.join(DATA_PATH, "Actions")
OBJECT_ALL_MESH_PATH = os.path.join(DATA_PATH, "Object_all", "Object_mesh")
OBJECT_CHAIRS_MESH_PATH = os.path.join(DATA_PATH, "Object_chairs", "Object_mesh")

MOTION_IND = np.load(os.path.join(DATA_PATH, 'idx_start.npy'))
GLOBAL_ORIENT = np.load(os.path.join(DATA_PATH, 'human_orient.npy'))
POSE = np.load(os.path.join(DATA_PATH, 'human_pose.npy'))
TRANSL = np.load(os.path.join(DATA_PATH, 'human_transl.npy'))
BETAS = np.load(os.path.join(DATA_PATH, 'betas.npy'))
JOINTS = np.load(os.path.join(DATA_PATH, 'human_joints.npy'))
LEFT_HAND = np.load(os.path.join(DATA_PATH, 'left_hand_pose.npy'))
RIGHT_HAND = np.load(os.path.join(DATA_PATH, 'right_hand_pose.npy'))

ACTION_LABEL = np.load(os.path.join(DATA_PATH, 'action_label.npy')).astype(np.float32)
SCENE_FLAG = np.load(os.path.join(DATA_PATH, 'scene_flag.npy'))
SCENE_LIST = np.load(os.path.join(DATA_PATH, 'scene_list.npy'))
OBJ_LIST = np.load(os.path.join(DATA_PATH, 'object_list.npy'))
SEG_NAME = np.load(os.path.join(DATA_PATH, 'seg_name.npy'))
FRAME_ID = np.load(os.path.join(DATA_PATH, 'frame_id.npy'))
OBJECT_FLAG = np.load(os.path.join(DATA_PATH, 'object_flag.npy'))
OBJECT_MAT = np.load(os.path.join(DATA_PATH, 'object_mat.npy'))
r_hip, l_hip, r_sdr, l_sdr = across_joint_indx
pelvis, _, _, head = up_joint_indx

def normalize(x, eps=1e-8):
    return x / (torch.linalg.norm(x) + eps)

def group_indices(lst):
    index_dict = defaultdict(list)
    
    for idx, value in enumerate(lst):
        index_dict[value].append(idx)
    
    return dict(index_dict)

grouped_indices = group_indices(list(SEG_NAME))
first_last_values = {key: (values[0], values[-1]) for key, values in grouped_indices.items()}
with open(os.path.join(DATA_PATH, 'trumans_sequence.pkl'), 'wb') as handle:
    pickle.dump(first_last_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def transform_smplx_from_origin_to_sampled_position(
        sampled_trans: np.ndarray,
        sampled_rotat: np.ndarray,
        origin_trans: np.ndarray,
        origin_orient: np.ndarray,
        origin_pelvis: np.ndarray,
        anchor_frame: int=0,
    ):
        """ Convert original smplx parameters to transformed smplx parameters

        Args:
            sampled_trans: sampled valid position
            sampled_rotat: sampled valid rotation
            origin_trans: original trans param array
            origin_orient: original orient param array
            origin_pelvis: original pelvis trajectory
            anchor_frame: the anchor frame index for transform motion, this value is very important!!!
        
        Return:
            Transformed trans, Transformed orient, Transformed pelvis
        """
        position = sampled_trans
        rotat = sampled_rotat

        T1 = np.eye(4, dtype=np.float32)
        T1[0:2, -1] = -origin_pelvis[anchor_frame, 0:2]
        T2 = Q(axis=[0, 0, 1], angle=rotat).transformation_matrix.astype(np.float32)
        T3 = np.eye(4, dtype=np.float32)
        T3[0:3, -1] = position
        T = T3 @ T2 @ T1

        trans_t = []
        orient_t = []
        for i in range(len(origin_trans)):
            t_, o_ = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(T, origin_trans[i], origin_orient[i], origin_pelvis[i])
            trans_t.append(t_)
            orient_t.append(o_)
        
        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = transform_points(origin_pelvis, T)
        return trans_t, orient_t, pelvis_t

def get_anchor_frame_index(action: str):
    action_anchor = {
        'sit': -1,
        'standup': 0,
        'walk': -1,
        'lie': -1,
    }
    return action_anchor[action]


def sampling_scene_old(points: np.ndarray, target_object_mask: np.ndarray,  SCENE_DIM: int=256) -> List:
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the height map 
    """
    ## compute floor height map
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()

    x = np.linspace(minx, maxx, SCENE_DIM)
    y = np.linspace(miny, maxy, SCENE_DIM)
    xx, yy = np.meshgrid(x, y)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    # floor_mask = points[:, -1] == 0
    # floor_xyz = points[floor_mask, 0:3]

    floor_kdtree = KDTree(points[:, 0:2], leaf_size=len(points))
    neigh_idx = floor_kdtree.query(pos2d.reshape((-1, 2)), k=1, return_distance=False, dualtree=True)
    neigh_idx = neigh_idx.reshape(-1)
    height_map = points[neigh_idx, 2]
    height_map = height_map.reshape(xx.shape)
    

    ## visualize
    # rescaled_scene = np.concatenate([xx[..., None], yy[..., None], height_map[..., None]], axis=-1).reshape(-1, 3)
    rescaled_scene = points[neigh_idx]
    rescaled_tgt_obj_mask = target_object_mask[neigh_idx]
    

    return rescaled_scene, rescaled_tgt_obj_mask, minx, maxx, miny, maxy

def sampling_scene(vertices: np.ndarray, root_pos: np.ndarray, 
                   target_object_mask: np.ndarray, scene_data_semantic_label: np.ndarray,
                   SCENE_DIM: int=128,  num_pts: int=1000) -> List:
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the height map 
    """
    # floor_mask = points[:, -1] == 0
    # floor_xyz = points[floor_mask, 0:3]
    if vertices.shape[0] > SCENE_DIM * SCENE_DIM:
        minx, miny = vertices[:, 0].min(), vertices[:, 1].min()
        maxx, maxy = vertices[:, 0].max(), vertices[:, 1].max()

        x = np.linspace(minx, maxx, SCENE_DIM)
        y = np.linspace(miny, maxy, SCENE_DIM)
        xx, yy = np.meshgrid(x, y)
        pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)
        floor_kdtree = KDTree(vertices[:, 0:2], leaf_size=len(vertices))
        neigh_idx = floor_kdtree.query(pos2d.reshape((-1, 2)), k=1, return_distance=False, dualtree=True)
        neigh_idx = neigh_idx.reshape(-1)
        sampled_verts = vertices[neigh_idx]
        sampled_tgt_obj_mask = target_object_mask[neigh_idx]
        
    else:
        sampled_verts = vertices
        sampled_tgt_obj_mask = target_object_mask
    local_kdtree = KDTree(sampled_verts[:, 0:2], leaf_size=len(sampled_verts))
    local_neigh_dist, local_neigh_idx= local_kdtree.query(root_pos[:, :2].reshape((-1, 2)), k=num_pts, return_distance=True, dualtree=True)
    local_scene_data_semantic_label = scene_data_semantic_label[local_neigh_idx]
    return sampled_verts, sampled_tgt_obj_mask, local_scene_data_semantic_label, local_neigh_dist, local_neigh_idx 


def sampling_scene_voxel_heightmap(vertices: np.ndarray, root_trans: np.ndarray, root_orient: np.ndarray,
                    betas: np.ndarray,
                    scene_data_semantic_label: np.ndarray,
                    SCENE_DIM: int=32, 
                    bbox_scale: float=0.6) -> List:
    
    body_vertices, body_faces, body_joints = SMPLX_Util.get_body_vertices_sequence(
                BODY_MODEL_FOLDER, 
                (root_trans, root_orient, betas, 
                    np.zeros((len(root_trans), 63), dtype="float32"),
                    np.zeros((len(root_trans), 90), dtype="float32")),
                num_betas=10
            )
    n_frames = len(body_joints)
    # Find upward and across vectors
    r_hip, l_hip, r_sdr, l_sdr = across_joint_indx
    pelvis, _, _, head = up_joint_indx
    cube_bbox_grid_vertices = np.zeros((n_frames, SCENE_DIM, SCENE_DIM, SCENE_DIM, 3))
    
    local_scene_data_semantic_label = np.zeros((n_frames, SCENE_DIM * SCENE_DIM))
    for n in range(n_frames):        
        across = body_joints[n, l_hip] - body_joints[n, r_hip]
        across = normalize(across)
        up = body_joints[n, head] - body_joints[n, pelvis]
        up = normalize(up)
        forward = normalize(np.cross(across, up))

        # forward_ls = np.linspace(forward[:2] + body_joints[n, 0, :2] , body_joints[n, 0, :2], 50)
        # forward_grid = np.concatenate([forward_ls, np.zeros((50, 1))], axis=-1).reshape(-1, 3)
        # across_ls = np.linspace(across[:2] + body_joints[n, 0, :2], body_joints[n, 0, :2], 50)
        # across_grid = np.concatenate([across_ls, np.zeros((50, 1))], axis=-1).reshape(-1, 3)
            
        top_left = bbox_scale * (forward[:2] + across[:2]) + body_joints[n, 0, :2]
        top_right = bbox_scale * (forward[:2] - across[:2]) + body_joints[n, 0, :2]
        bottom_left = bbox_scale * (- forward[:2] + across[:2]) + body_joints[n, 0, :2]
        bottom_right = bbox_scale * (- forward[:2] - across[:2]) + body_joints[n, 0, :2]
        # top_up_left = (2 * bbox_scale * up[:2]) + top_left
        
        left_ls = np.linspace(bottom_left, top_left, SCENE_DIM)
        right_ls = np.linspace(bottom_right, top_right, SCENE_DIM)
        # up_ls = np.linspace(top_left, top_up_left, SCENE_DIM)
        # xx, yy = np.meshgrid(x, y)
        bbox_grid = np.empty((SCENE_DIM, SCENE_DIM, 2))
        # bbox_grid_up = np.empty((SCENE_DIM, SCENE_DIM, 2))
        for bbox_idx in range(SCENE_DIM):
            bbox_grid[bbox_idx] = np.linspace(left_ls[bbox_idx], right_ls[bbox_idx], SCENE_DIM)
            # bbox_grid_up[bbox_idx] = np.linspace(left_ls[bbox_idx], up_ls[bbox_idx], SCENE_DIM)

        vertices_kdtree = KDTree(vertices[:, 0:2], leaf_size=len(vertices))
        neigh_idx = vertices_kdtree.query(bbox_grid.reshape((-1, 2)), k=1, return_distance=False, dualtree=True)
        neigh_idx = neigh_idx.reshape(-1)
        height_map = vertices[neigh_idx, 2]
        bbox_grid_vertices = np.concatenate([bbox_grid, height_map.reshape(SCENE_DIM, SCENE_DIM, 1)], axis=-1)
        cube_bbox_grid_vertices[n] = np.tile(bbox_grid_vertices[:, :, np.newaxis, :], (1, 1, SCENE_DIM, 1))
        # top_height = np.ones_like(height_map) * 2 * bbox_scale
        # bbox_topgrid_vertices = np.concatenate([bbox_grid, top_height.reshape(SCENE_DIM, SCENE_DIM, 1)], axis=-1)
        # voxel_grid = np.zeros((SCENE_DIM, SCENE_DIM, SCENE_DIM, 3))
        # for d in range(SCENE_DIM):
        #     alpha = d / (SCENE_DIM - 1)  # Normalized depth (0 at floor, 1 at ceiling)
        #     voxel_grid[:, :, d] = (1 - alpha) * bbox_grid_vertices + alpha * bbox_topgrid_vertices
        # cube_bbox_grid_vertices[n] = voxel_grid
        
        local_scene_data_semantic_label[n] = scene_data_semantic_label[neigh_idx]
    return cube_bbox_grid_vertices, local_scene_data_semantic_label

import numpy as np

def extract_scene_patch(human_points, scene_vertices, distance_threshold=0.6, patch_size=4000):
    """
    Extract a 32x32 dimension of scene point cloud that is within a given distance
    from any human vertices or body joints.
    
    Parameters:
    - human_points (torch.Tensor): Array of shape (N, 3) representing human point cloud.
    # - joint_positions (torch.Tensor): Array of shape (J, 3) representing 3D body joints.
    - scene_vertices (torch.Tensor): Array of shape (M, 3) representing scene point cloud.
    - distance_threshold (float): Distance threshold to consider points near human.
    - patch_size (int): Number of points to sample in the extracted scene patch.
    
    Returns:
    - (torch.Tensor): Extracted scene patch of shape (patch_size, 3).
    """
    # Combine human vertices and joints
    # human_points = np.vstack((human_vertices, joint_positions))
    
    # Compute distances from scene points to human points
    distances = torch.cdist(scene_vertices, human_points)
    
    # Find scene points within the distance threshold
    valid_indices = torch.any(distances <= distance_threshold, dim=1)
    filtered_scene = scene_vertices[valid_indices]
    
    # Sample a fixed number of points (patch_size)
    # if filtered_scene.shape[0] > patch_size:
    #     return filtered_scene[sampled_indices]
    if filtered_scene.shape[0] >= patch_size:
        sampled_indices = np.random.choice(filtered_scene.shape[0], patch_size, replace=False)
    else:
        sampled_indices = np.random.choice(filtered_scene.shape[0], patch_size, replace=True)
    # sampled_indices = torch.randperm(filtered_scene.shape[0])[:patch_size]
    return filtered_scene[sampled_indices]

def extract_and_sort_points(human_points, scene_points, forward_vector, distance_threshold=0.6, patch_size=4000):
    """
    Extracts points from the scene point cloud that are within a distance of `distance_threshold` from
    any point in the human point cloud and sorts them in a clockwise manner from the front of the human.
    
    Args:
    - human_points (torch.Tensor): A tensor of shape (N, 3) representing the human point cloud vertices.
    - scene_points (torch.Tensor): A tensor of shape (M, 3) representing the scene point cloud vertices.
    - forward_vector (torch.Tensor): A 3D tensor representing the forward direction of the human.
    - distance_threshold (float): The distance threshold to filter points from the scene (default: 0.6 meters).
    
    Returns:
    - sorted_points (torch.Tensor): The points from the scene that are within the distance threshold from
      the human, sorted in a clockwise manner starting from the front direction.
    """
    
    # # Step 1: Compute distances between human vertices and scene points
    # human_points_expanded = human_points.unsqueeze(1)  # Shape: (N, 1, 3)
    # scene_points_expanded = scene_points.unsqueeze(0)  # Shape: (1, M, 3)
    
    # # Compute the distance between each human point and each scene point
    # distances = torch.norm(human_points_expanded - scene_points_expanded, dim=-1)  # Shape: (N, M)
    distances = torch.cdist(scene_points, human_points)
    # Step 2: Extract points from the scene that are within the distance threshold from any human vertex
    # within_threshold = distances <= distance_threshold
    selected_scene_points = scene_points[torch.any(distances <= distance_threshold, dim=1)]  # Select points that are within threshold
    
    # Step 3: Sort the points based on the relative angle to the forward direction
    # Get the vector from the human's front (arbitrary origin) to each selected scene point
    human_center = human_points.mean(dim=0)  # Assuming human is centered
    front_to_points = selected_scene_points - human_center  # Shape: (K, 3), where K is the number of selected points
    
    # Normalize forward vector
    forward_vector_normalized = forward_vector / forward_vector.norm()
    
    # Compute right vector (perpendicular to forward vector) assuming Y-up axis
    right_vector = torch.linalg.cross(forward_vector_normalized, torch.tensor([0., 1., 0.], dtype=torch.float32).to(forward_vector.device))
    right_vector = right_vector / right_vector.norm()
    
    # Compute the angle between the forward direction and the point's vector (in the 2D plane)
    projection = torch.matmul(front_to_points, right_vector)  # X-axis projection (clockwise)
    angle = torch.atan2(projection, torch.matmul(front_to_points, forward_vector_normalized))  # Angle wrt the forward vector
    
    # Step 4: Sort the points in a clockwise manner starting from the front direction
    sorted_indices = torch.argsort(angle)
    sorted_points = selected_scene_points[sorted_indices]
    
    if sorted_points.shape[0] >= patch_size:
        sampled_indices = np.random.choice(sorted_points.shape[0], patch_size, replace=False)
    else:
        sampled_indices = np.random.choice(sorted_points.shape[0], patch_size, replace=True)
    
    return sorted_points[sampled_indices]

def sampling_scene_heightmap(scene_vertices: torch.Tensor, body_joints: torch.Tensor,
                    SCENE_DIM: int=32, 
                    bbox_scale: float=0.6) -> List:
    """ Load region meshes of a scenes local around the human

    Args:
        scene_vertices: scene point cloud
        body_joints: body_joints position of person
        
    
    Return:
        Return the local grid around person
    """

    # Find upward and across vectors
    r_hip, l_hip, r_sdr, l_sdr = across_joint_indx
    pelvis, _, _, head = up_joint_indx
    bbox_grid_vertices = torch.zeros((len(body_joints), SCENE_DIM, SCENE_DIM, 3)).to(DEVICE)
    local_scene_data_semantic_label = torch.zeros((len(body_joints), SCENE_DIM, SCENE_DIM)).to(DEVICE)
    for n in range(len(body_joints)):        
        across = body_joints[n, l_hip] - body_joints[n, r_hip]
        across = normalize(across)
        up = body_joints[n, head] - body_joints[n, pelvis]
        up = normalize(up)
        forward = normalize(torch.linalg.cross(across, up))

        # forward_ls = np.linspace(forward[:2] + body_joints[n, 0, :2] , body_joints[n, 0, :2], 50)
        # forward_grid = np.concatenate([forward_ls, np.zeros((50, 1))], axis=-1).reshape(-1, 3)
        # across_ls = np.linspace(across[:2] + body_joints[n, 0, :2], body_joints[n, 0, :2], 50)
        # across_grid = np.concatenate([across_ls, np.zeros((50, 1))], axis=-1).reshape(-1, 3)
            
        top_left = bbox_scale * (forward[:2] + across[:2]) + body_joints[n, 0, :2]
        top_right = bbox_scale * (forward[:2] - across[:2]) + body_joints[n, 0, :2]
        bottom_left = bbox_scale * (- forward[:2] + across[:2]) + body_joints[n, 0, :2]
        bottom_right = bbox_scale * (- forward[:2] - across[:2]) + body_joints[n, 0, :2]
        
        # left_ls = torch.linspace(bottom_left, top_left, SCENE_DIM)
        # right_ls = torch.linspace(bottom_right, top_right, SCENE_DIM)
        left_ls = torch.stack([torch.linspace(bottom_left[i], top_left[i], SCENE_DIM) for i in range(2)], dim=1)
        right_ls = torch.stack([torch.linspace(bottom_right[i], top_right[i], SCENE_DIM) for i in range(2)], dim=1)

        bbox_grid = torch.empty((SCENE_DIM, SCENE_DIM, 2)).to(body_joints.device)
        for bbox_idx in range(SCENE_DIM):
            bbox_grid[bbox_idx] = torch.stack([torch.linspace(left_ls[bbox_idx, i], right_ls[bbox_idx, i], SCENE_DIM) for i in range(2)], dim=-1)

        # vertices_kdtree = KDTree(scene_vertices[:, 0:2], leaf_size=len(scene_vertices))
        # neigh_idx = vertices_kdtree.query(bbox_grid.reshape((-1, 2)), k=1, return_distance=False, dualtree=True)
        # neigh_idx = neigh_idx.reshape(-1)
        vertices_xy = scene_vertices[:, :2]  # Extract xy coordinates
        bbox_grid_flat = bbox_grid.reshape(-1, 2)
        distances = torch.cdist(bbox_grid_flat, vertices_xy)  # Compute pairwise distances
        neigh_idx = torch.argmin(distances, dim=1)
        height_map = scene_vertices[neigh_idx, 2]
        bbox_grid_vertices[n] = torch.cat([bbox_grid, height_map.reshape(SCENE_DIM, SCENE_DIM, 1)], -1)
        # local_scene_data_semantic_label[n] = scene_data_semantic_label[neigh_idx].reshape(SCENE_DIM, SCENE_DIM)
    return bbox_grid_vertices, local_scene_data_semantic_label

def create_meshgrid_from_corners(top_left, top_right, bottom_left, bottom_right, grid_size=32):
        """
        Creates a meshgrid using four corner points.
        
        Args:
            top_left: Tensor [2], top-left corner coordinates (x, z)
            top_right: Tensor [2], top-right corner coordinates (x, z)
            bottom_left: Tensor [2], bottom-left corner coordinates (x, z)
            bottom_right: Tensor [2], bottom-right corner coordinates (x, z)
            grid_size: int, size of the grid (default: 32)
            
        Returns:
            grid_xz: [grid_size, grid_size, 2] tensor with world coordinates of each grid cell (x, z)
        """
        # Create interpolation parameters for both dimensions
        u = torch.linspace(0, 1, grid_size, device=DEVICE)
        v = torch.linspace(0, 1, grid_size, device=DEVICE)
        
        # Create meshgrid of interpolation parameters
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        # Reshape for broadcasting
        U = U.unsqueeze(-1)  # [grid_size, grid_size, 1]
        V = V.unsqueeze(-1)  # [grid_size, grid_size, 1]
        
        # Interpolate along the top and bottom edges
        top_edge = top_left.unsqueeze(0).unsqueeze(0) * (1 - U) + top_right.unsqueeze(0).unsqueeze(0) * U
        bottom_edge = bottom_left.unsqueeze(0).unsqueeze(0) * (1 - U) + bottom_right.unsqueeze(0).unsqueeze(0) * U
        
        # Interpolate between top and bottom edges
        grid_xz = top_edge * (1 - V) + bottom_edge * V
        
        return grid_xz

def create_heightmap_from_vertices(grid_xz, local_grid_vertices, grid_size=32, search_radius=0.1):
        """
        Creates a heightmap by finding the maximum y-coordinate from local_grid_vertices
        that corresponds to each point in grid_xz.
        
        Args:
            grid_xz: Tensor [grid_size, grid_size, 2] with x and z coordinates of each grid point
            local_grid_vertices: Tensor [N, 3] with 3D coordinates of scene vertices
            grid_size: int, size of the grid (default: 32)
            search_radius: float, radius to search for nearby vertices (default: 0.1)
            
        Returns:
            grid_y: Tensor [grid_size, grid_size] with y-coordinates (heights) for each grid point
        """
        # Reshape grid_xz to [grid_size*grid_size, 2] for easier processing
        grid_points = grid_xz.reshape(-1, 2)  # [grid_size*grid_size, 2]
        
        # Extract xz coordinates and y coordinates from local_grid_vertices
        vertices_xz = local_grid_vertices[:, [0, 2]]  # [N, 2]
        vertices_y = local_grid_vertices[:, 1]  # [N]
        
        # Compute pairwise distances between all grid points and all vertices
        # This creates a matrix of shape [grid_size*grid_size, N]
        # Each element (i,j) is the distance between grid point i and vertex j
        distances = torch.cdist(grid_points, vertices_xz)  # [grid_size*grid_size, N]
        
        # Create a mask for vertices within the search radius
        # This creates a boolean matrix of shape [grid_size*grid_size, N]
        # True values indicate vertices within the search radius
        mask = distances < search_radius  # [grid_size*grid_size, N]
        
        # Initialize the heightmap with zeros
        grid_y = torch.zeros(grid_size * grid_size, device=DEVICE)
        
        # For each grid point, find the maximum y-coordinate of nearby vertices
        for i in range(grid_size * grid_size):
            # Get the mask for the current grid point
            point_mask = mask[i]  # [N]
            
            if point_mask.any():
                # Get the y-coordinates of nearby vertices
                nearby_y = vertices_y[point_mask]  # [M]
                
                # Find the maximum y-coordinate
                grid_y[i] = torch.max(nearby_y)
        
        # Reshape the heightmap to [grid_size, grid_size]
        grid_y = grid_y.reshape(grid_size, grid_size)
        
        return grid_y
 
def create_contact_region(output_file):
    with open(output_file, 'rb') as fp:
        annot_data = Unpickler(fp).load()
    window_size = len(annot_data['body_joints'])
    contact_vertices_list = []
    contact_indices_list = []
    
    person_vertices, person_joints = smplx_to_pos3d(annot_data['person_data'])
    for n in range(window_size):           
        # Find contact vertices
        contact_vertices, contact_indices = find_contact_vertices(
            to_tensor(annot_data['local_grid_vertices'][n], device=DEVICE),
            to_tensor(person_vertices[n], device=DEVICE)
        )
        
        contact_vertices_list.append(contact_vertices)
        contact_indices_list.append(contact_indices)
    
    annot_data['contact_vertices'] = torch.stack(contact_vertices_list)  
    with open(output_file, 'wb') as handle:
        pickle.dump(annot_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print(f'Saved pre-processed data with contact vertices in {output_file}')  

def create_heightmap(output_file):
    with open(output_file, 'rb') as fp:
        annot_data = Unpickler(fp).load()
    window_size = len(annot_data['body_joints'])
    across_ = to_tensor(annot_data['across_vector_seq']).to(DEVICE)
    forward_ = torch.linalg.cross(across_, 
                                    to_tensor(np.repeat(np.array([[0, 1, 0]]), window_size, axis=0), device=DEVICE),
                                    dim=-1)   
    forward_[..., 1] = 0 
    across_[..., 1] = 0 
    root_jt =  to_tensor(annot_data['body_joints'][:, 0]).to(DEVICE) 
    root_jt[..., 1] = 0
    top_left_corner =  (forward_ + across_) + root_jt
    top_right_corner = (forward_ - across_) + root_jt
    top_left_corner[..., 1] = 0 
    top_right_corner[..., 1] = 0 
    bottom_left_corner =   - (forward_ - across_) + root_jt
    bottom_right_corner = - (forward_ + across_) + root_jt
    bottom_right_corner[..., 1] = 0 
    bottom_left_corner[..., 1] = 0 
        
    # choices = np.random.choice(datas['merged_scene_vertex'].shape[1], NUM_MAX_SCENE_PTS)
    # annot_data['merged_scene_vertex'] = datas['merged_scene_vertex'][start: end: self.downsample_rate, choices].to(DEVICE)
    heightmap = torch.zeros((window_size, NB_VOXELS, NB_VOXELS)).to(DEVICE)
    grid_xz = torch.zeros((window_size, NB_VOXELS, NB_VOXELS, 2)).to(DEVICE)
    grid_xyz = torch.zeros((window_size, NB_VOXELS, NB_VOXELS, 3)).to(DEVICE)
    for n in range(window_size):
        # Create meshgrid using the four corner points
        grid_xz[n] = create_meshgrid_from_corners(
            top_left_corner[n, [0,2]],
            top_right_corner[n, [0,2]],
            bottom_left_corner[n, [0,2]],
            bottom_right_corner[n, [0,2]],
            grid_size=NB_VOXELS
        )
        
        # Create heightmap from local_grid_vertices
        heightmap[n] = create_heightmap_from_vertices(
            grid_xz[n],
            to_tensor(annot_data['local_grid_vertices'][n]).to(DEVICE),
            grid_size=NB_VOXELS
        )
        
        # Assign x, y, and z coordinates to grid_xyz
        grid_xyz[n, ..., 0] = grid_xz[n, ..., 0]  # x coordinate
        grid_xyz[n, ..., 1] = heightmap[n]  # y coordinate (height)
        grid_xyz[n, ..., 2] = grid_xz[n, ..., 1] 
    annot_data['grid_xyz'] = to_cpu(grid_xyz)
    annot_data['forward_vector_seq'] = to_cpu(forward_)
    annot_data['across_vector_seq'] = to_cpu(across_)
    annot_data['top_left_corner'] = to_cpu(top_left_corner)
    annot_data['top_right_corner'] = to_cpu(top_right_corner)
    annot_data['bottom_left_corner'] = to_cpu(bottom_left_corner)
    annot_data['bottom_right_corner'] = to_cpu(bottom_right_corner)
    

    with open(output_file, 'wb') as handle:
        pickle.dump(annot_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print(f'Saved pre-processed data with heightmap in {output_file}')



def find_contact_vertices(local_grid_vertices, person_vertices, contact_threshold=0.05, max_vertices=2000):
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
    # Filter out vertices on the ground plane (y ≈ 0)
    # We use a small threshold to account for floating point precision
    ground_threshold = 0.01
    non_ground_mask = torch.abs(local_grid_vertices[:, 1]) > ground_threshold
    non_ground_vertices = local_grid_vertices[non_ground_mask]
    
    if non_ground_vertices.shape[0] == 0:
        # If no non-ground vertices, return zero tensors with the correct shape
        return torch.zeros((max_vertices, 3), device=DEVICE), torch.zeros(max_vertices, dtype=torch.long, device=DEVICE)
    
    # Compute pairwise distances between all non-ground scene vertices and all person vertices
    # This creates a matrix of shape [K, M] where K is the number of non-ground vertices
    # Each element (i,j) is the distance between scene vertex i and person vertex j
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
        padding_vertices = torch.zeros((max_vertices - contact_vertices.shape[0], 3), device=DEVICE)
        padding_indices = torch.zeros(max_vertices - contact_indices.shape[0], dtype=torch.long, device=DEVICE)
        
        # Concatenate the contact vertices with the padding
        contact_vertices = torch.cat([contact_vertices, padding_vertices], dim=0)  # [max_vertices, 3]
        contact_indices = torch.cat([contact_indices, padding_indices], dim=0)  # [max_vertices]
    
    return contact_vertices, contact_indices

def scene_data_preprocess(start_id, end, skip, vis=False, out_pkl=None):
    annot_data = {}
    annot_data['start_id'] = start_id
    annot_data['end_id'] = end
    annot_data['scene_name'] = SCENE_LIST[SCENE_FLAG[start_id]]
    obj_flag = OBJECT_FLAG[start_id]
    annot_data['obj_name'] = OBJ_LIST[obj_flag]
    obj_mat_seq = OBJECT_MAT[start_id: end: skip]
    annot_filename = os.path.join(ANNOTATION_PATH, clean_suffix(SEG_NAME[start_id]) + '.txt')
    if out_pkl is None:
        save_folder = makepath(os.path.join(PREPROCESSED_DATA_FOLDER, str(start_id) + '_' + str(end) + '_motion.pkl'), isfile=True)
    else:
        save_folder = makepath(out_pkl, isfile=True)
    f_open = open(annot_filename, "r")
    content = f_open.read()
    f_open.close()
    entries = content.strip().split("\n")
    annotation_actions = []
    for entry in entries:
        parts = entry.split("\t")  # Split by tab
        annotation_actions.append({
            "start_frame": int(parts[0]),
            "end_frame": int(parts[1]),
            "text": parts[2]
        })
    annot_data['annotation_actions'] = annotation_actions
    pose_body = to_tensor(POSE[start_id: end: skip], device=DEVICE)
    global_orient = to_tensor(GLOBAL_ORIENT[start_id: end: skip], device=DEVICE)
    transl = to_tensor(TRANSL[start_id: end: skip], device=DEVICE)
    betas = to_tensor(BETAS[start_id:start_id+1], device=DEVICE)
    joints = to_tensor(JOINTS[start_id: end: skip], device=DEVICE)
    left_hand_pose = to_tensor(LEFT_HAND[start_id: end: skip], device=DEVICE)
    right_hand_pose = to_tensor(RIGHT_HAND[start_id: end: skip], device=DEVICE)
    pose_hand = torch.cat((left_hand_pose, right_hand_pose), -1)
        
    annot_data['action_label'] = ACTION_LABEL[start_id: end: skip]
    annot_data['person_data'] = {
                'transl':transl,
                'global_orient': global_orient,
                'pose_body': pose_body,
                'pose_hand': pose_hand,
                'betas': betas,
                'meta': {'gender': 'male'}
            }
        
    person_vertices, person_joints = smplx_to_pos3d(annot_data['person_data'])
    annot_data['body_joints'] = person_joints[:, :BODY_JOINTS]
    annot_data['mot_feats'] = process_human_motion(annot_data['person_data'], feet_thre=0.002)
    # annot_data['scene_occupancy_grid'] = np.load(os.path.join(SCENE_OCC_PATH, annot_data['scene_name'] + '.npy'))
    scene_vertex = np.load(os.path.join(PREPROCESSED_SCENE_PATH, annot_data['scene_name'] + '_vertices.npy'))
    scene_vertex_idxes = torch.sort(torch.randperm(scene_vertex.shape[0])[:NUM_MAX_SCENE_PTS])[1]
    annot_data['scene_vertex'] = to_tensor(scene_vertex[scene_vertex_idxes], device=DEVICE)
    # annot_data['scene_faces'] = np.load(os.path.join(PREPROCESSED_SCENE_PATH, annot_data['scene_name'] + '_faces.npy'))
    # for obj_items in annot_data['obj_name']:
    merged_scene_vertex = annot_data['scene_vertex'].repeat(len(person_joints), 1, 1)
    obj_points_all = []
    for obj_id, has in enumerate(obj_flag):
        if has >= 0:
            obj_items = annot_data['obj_name'][obj_id]
            obj_vertex = np.load(os.path.join(PREPROCESSED_OBJ_CHAIRS_PATH, obj_items + '_vertices.npy'))
            obj_points = []
            for l in range(len(person_joints)):
                obj_mat = obj_mat_seq[l][has]
                swapped_obj_mat = obj_mat[:3, :3].copy()
                # swapped_obj_mat[[1,2], :] = swapped_obj_mat[[2,1], :]
                # swapped_obj_mat[:, [1,2]] = swapped_obj_mat[:, [2,1]]
                swapped_obj_mat[:, 2] *= -1
                obj_points.append((obj_vertex @ swapped_obj_mat + obj_mat[:3, 3])[None])
            obj_points = np.concatenate(obj_points, axis=0)
            obj_points_all.append(obj_points)
    if obj_points_all:
        obj_points_all = to_tensor(np.concatenate(obj_points_all, axis=1), device= merged_scene_vertex.device)
        merged_scene_vertex = torch.cat([merged_scene_vertex, obj_points_all], dim=1)
    
    # annot_data['merged_scene_vertex'] = merged_scene_vertex

    # annot_data['merged_scene_vertex'] = merged_scene_vertex[choices]
    
        # if obj_items != 'static_chair_03(1.25,1.22,1.25)':
        #     obj_vertex = np.load(os.path.join(PREPROCESSED_OBJ_CHAIRS_PATH, obj_items + '_vertices.npy'))
        #     obj_vertex = to_tensor(obj_vertex, device=DEVICE)
    # faces_object_updated = annot_data['object_faces'] + annot_data['scene_vertex'].shape[0]  # Shift indices of object faces
    # annot_data['merged_scene_face'] = np.concatenate([annot_data['scene_faces'], faces_object_updated], axis=0)
    # merged_scene_vertex_idxes = torch.sort(torch.randperm(merged_scene_vertex.shape[0])[:NUM_MAX_SCENE_PTS])[1]
    # local_grid_vertices, local_scene_data_semantic_label = sampling_scene_voxel_heightmap(p['scene_verts'],
    # local_grid_vertices, local_scene_data_semantic_label = sampling_scene_heightmap(annot_data['scene_vertex'],
    #                                                                                 annot_data['body_joints'],
    #                                                                                 SCENE_DIM=NB_VOXELS,
    #                                                                                 bbox_scale=BBOX_SCALE)
    
    filtered_scene = []
    forward_vector_seq = []
    across_vector_seq = []
    up_vector_seq = []
    print('building local neighborhood')
    for l in range(len(person_joints)):
        across = person_joints[l, l_hip] - person_joints[l, r_hip]
        across = normalize(across)
        across_vector_seq.append(across)
        up = person_joints[l, head] - person_joints[l, pelvis]
        up = normalize(up)
        up_vector_seq.append(up)
        forward_vector = normalize(torch.linalg.cross(across, up))
        forward_vector_seq.append(forward_vector)
        # filtered_scene.append(extract_and_sort_points(person_vertices[l], annot_data['merged_scene_vertex'][l], forward_vector, distance_threshold=BBOX_SCALE))
        filtered_scene.append(extract_scene_patch(person_vertices[l], merged_scene_vertex[l], distance_threshold=BBOX_SCALE))
    filtered_scene = torch.stack(filtered_scene)
    forward_vector_seq = torch.stack(forward_vector_seq)
    across_vector_seq = torch.stack(across_vector_seq)
    up_vector_seq = torch.stack(up_vector_seq)
    #0 is x axis, 1 is z axis, 2 is y axis (upwards)
    # p['local_height_map'] = local_grid_vertices[..., 2]
    annot_data['local_grid_vertices'] = filtered_scene
    annot_data['forward_vector_seq'] = forward_vector_seq
    annot_data['across_vector_seq'] = across_vector_seq
    annot_data['up_vector_seq'] = up_vector_seq
    

        
    if vis:   
        from visualize.visualize_data import visualize_sequence
        prompt = " ".join(
        action['text'] + "."
        for action in annotation_actions)
        print(prompt)
        # scene_path = os.path.join(SCANNET_FOLDER, str(scene_id), str(scene_id) + '_vh_clean_2.ply')
        # static_scene = trimesh.load(scene_path, process=False)
        # static_scene.apply_translation(scene_trans)
        # body_vertices, body_faces, joints = SMPLX_Util.get_body_vertices_sequence(
        #         BODY_MODEL_FOLDER, 
        #         (p['person_data']['transl'], p['person_data']['global_orient'], 
        #         p['person_data']['betas'], p['person_data']['pose_body'],
        #         p['person_data']['pose_hand']),
        #         num_betas=10
        #     )
        
        
        # visualize_sequence(scene_data=static_scene, person1=p['person_data'], item3=[local_grid_vertices.reshape(seq_len, -1, 3), None])
        visualize_sequence(scene_data=[merged_scene_vertex.detach().cpu().numpy(), None], 
                            person1=[person_vertices.detach().cpu().numpy(), None],
                            # item3=[filtered_scene.detach().cpu().numpy(), None],
                            is_permute=False)
            
           
    with open(save_folder, 'wb') as handle:
        pickle.dump(annot_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print(f'Saved pre-processed data in {save_folder}')


def clean_suffix(s):
    for suffix in ['_augment1', '_augment2']:
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s
    
if __name__ == '__main__':
    
    with open(os.path.join(DATA_PATH, 'trumans_sequence.pkl'), 'rb') as fp:
        read_sequence_ = pickle.load(fp)
    skip = STEP
    for idx, (key, values) in enumerate(read_sequence_.items()):
        # if idx in [55, 56, 57, 66, 67, 558, 573, 578, 584, 586, 589, 599, 607, 608, 623, 625, 631, 634, 635, 651, 672, 685, 687, 693, 713, 714, 716, 736, 739, 742, 745, 748, 756, 783, 788, 804, 829, 832, 834, 849, 850, 852, 866, 870, 875, 887, 888, 892, 902, 911, 918, 919, 920, 927, 938, 949, 950, 951, 956, 962, 963, 972, 977, 980, 995]:
        if idx >= 0:
            print('idx ', idx)
            start_id = values[0]
            end_id = values[-1]
            
            annotation_filename = os.path.join(ANNOTATION_PATH, clean_suffix(key)+'.txt')
            output_file = os.path.join(PREPROCESSED_DATA_FOLDER, str(idx) + '_' + str(start_id) + '_' + str(end_id) + '_motion_heightmap.pkl')
            newname = os.path.join(PREPROCESSED_DATA_FOLDER, str(idx) + '_' + str(start_id) + '_' + str(end_id) + '_motion_heightmap_contacts.pkl')
            if os.path.isfile(newname):
                continue
            if not os.path.isfile(annotation_filename):
                print('%s.txt doesnot exist' % key)
                continue
            if not os.path.isfile(output_file):
                scene_data_preprocess(start_id, end_id, skip, vis=False, out_pkl=output_file)
            if not os.path.isfile(output_file):
                print('skip idx %s: no base pickle at %s' % (idx, output_file))
                continue
            create_contact_region(output_file)
            create_heightmap(output_file)
            os.rename(output_file, newname)

