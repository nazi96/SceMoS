
# import ffmpeg
import glob
import pickle
import numpy as np
import os
import sys
sys.path.append('.')
sys.path.append('..')
import skvideo
# skvideo.setFFmpegPath('C:\\Users\\angh01\\AppData\\Local\\anaconda3\\envs\\scene_int\\Scripts')
# MESA_GL_VERSION_OVERRIDE=4.0
import skvideo.io
import torch
import trimesh
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence, SMPLLayer
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from copy import deepcopy

from smplx import SMPLX
from dataset_prep.motion_process import save_smplx_as_npz, smplx_to_pos3d, interpolate_person_data

from utils.transformations import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# aitviewer PointClouds expects ``points`` with rank 3, typically (B, N, 3).
def _numpy_points_for_pointclouds(x):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 2 and a.shape[-1] == 3:
        a = a.reshape(1, -1, 3)
    return a


# '''Copy this function into the SMPLSequence class of aitviewer'''
@classmethod
def from_custom_npy(cls, data: dict, smpl_layer=None, **kwargs):
    """Creates a SMPL sequence from a dictionary of smpl parameters"""
    if smpl_layer is None:
        smpl_layer = SMPLLayer(model_type="smplx", gender=data['meta']['gender'])
    out = cls(
        smpl_layer=smpl_layer,
        poses_root=data["poses"][:, :3],
        poses_body=data["poses"][:, 3:66],
        poses_left_hand=data["poses"][:, 75:120],
        poses_right_hand=data["poses"][:, 120:],
        betas=data["betas"],
        trans=data["transl"],
        **kwargs,
    )
    return out

def visualize_sequence(scene_data=None, person1=None, person2=None, item3=None, item4=None, item5=None, item6=None, is_permute=True):
    v = Viewer()
    if scene_data is not None:
        if isinstance(scene_data, list):
            scene_vertex = scene_data[0]
            scene_faces = scene_data[1]
            if is_permute:
                scene_vertex = scene_vertex[..., [0, 2, 1]]
            scene_vertex = _numpy_points_for_pointclouds(scene_vertex)
            mesh0 = PointClouds(points=scene_vertex, color=(0.0, 0.75, 1.0, 0.5))
        else:
            uvs = None
            vertex_colors = None
            face_colors = None
            scene_vertices = scene_data.vertices
            if is_permute:
                scene_vertices = scene_vertices[..., [0, 2, 1]]
            if isinstance(scene_data.visual, trimesh.visual.ColorVisuals):
                if scene_data.visual.kind == "vertex_colors":
                    vertex_colors = scene_data.visual.vertex_colors
                elif scene_data.visual.kind == "face_colors":
                    face_colors = scene_data.visual.vertex_colors
            elif isinstance(scene_data.visual, trimesh.visual.TextureVisuals):
                uvs = scene_data.visual.uv
            mesh0 = Meshes(scene_vertices, scene_data.faces, 
                        vertex_normals=scene_data.vertex_normals, flat_shading=True)
        v.scene.add(mesh0)
    if person1 is not None:
        if isinstance(person1, list):
            data1_vertex = person1[0]
            data1_face = person1[1]
            if is_permute:
                data1_vertex = data1_vertex[..., [0, 2, 1]]
            data1_vertex = _numpy_points_for_pointclouds(data1_vertex)
            mesh1 = PointClouds(points=data1_vertex, color=(0.6, 0.5, 0.5, 1.0))
        else:
            if is_permute:
                person1["transl"] = person1["transl"][..., [0, 2, 1]]
                person1["global_orient"] = person1["global_orient"][..., [0, 2, 1]]
                person1["pose_body"] = (person1["pose_body"].reshape(len(person1["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person1["poses"]), -1)
                person1["pose_hand"] = (person1["pose_hand"].reshape(len(person1["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person1["poses"]), -1)
                person1["poses"] = (person1["poses"].reshape(len(person1["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person1["poses"]), -1)
            mesh1 = SMPLSequence.from_custom_npy(data=person1, z_up=is_permute, color=(0.6, 0.5, 0.5, 1.0))
        v.scene.add(mesh1)
    if person2 is not None:
        if isinstance(person2, list):
            data2_vertex = person2[0]
            data2_face = person2[1]
            if is_permute:
                data2_vertex = data2_vertex[..., [0, 2, 1]]
            data2_vertex = _numpy_points_for_pointclouds(data2_vertex)
            mesh2 = PointClouds(points=data2_vertex, color=(0.6, 0.4, 0.4, 1.0))
        else:
            if is_permute:
                person2["transl"] = person2["transl"][..., [0, 2, 1]]
                person2["global_orient"] = person2["global_orient"][..., [0, 2, 1]]
                person2["pose_body"] = (person2["pose_body"].reshape(len(person1["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person1["poses"]), -1)
                person2["pose_hand"] = (person2["pose_hand"].reshape(len(person1["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person1["poses"]), -1)
                person2["poses"] = (person2["poses"].reshape(len(person2["poses"]), -1, 3)[..., [0, 2, 1]]).reshape(len(person2["poses"]), -1)
            mesh2 = SMPLSequence.from_custom_npy(data=person2, z_up=is_permute, color=(0.6, 0.4, 0.4, 1.0))
        v.scene.add(mesh2)
    if item3 is not None:
        data3_vertex = item3[0]
        data3_face = item3[1]
        if is_permute:
            data3_vertex = data3_vertex[..., [0, 2, 1]]
        data3_vertex = _numpy_points_for_pointclouds(data3_vertex)
        verts3 = PointClouds(points=data3_vertex, color=(0.0, 0.75, 1.0, 0.5))
        v.scene.add(verts3)
    if item4 is not None:
        data4_vertex = item4[0]
        data4_face = item4[1]
        if is_permute:
            data4_vertex = data4_vertex[..., [0, 2, 1]]
        data4_vertex = _numpy_points_for_pointclouds(data4_vertex)
        verts4 = PointClouds(points=data4_vertex, color=(1.0, 1.0, 0.0, 1.0))
        v.scene.add(verts4)
    if item5 is not None:
        data5_vertex = item5[0]
        data5_face = item5[1]
        if is_permute:
            data5_vertex = data5_vertex[..., [0, 2, 1]]
        data5_vertex = _numpy_points_for_pointclouds(data5_vertex)
        verts5 = PointClouds(points=data5_vertex, color=(1.0, 1.0, 0.4, 1.0))
        v.scene.add(verts5)
    if item6 is not None:
        data6_vertex = item6[0]
        data6_face = item6[1]
        if is_permute:
            data6_vertex = data6_vertex[..., [0, 2, 1]]
        data6_vertex = _numpy_points_for_pointclouds(data6_vertex)
        verts6 = PointClouds(points=data6_vertex, color=(1.0, 1.0, 0.8, 1.0))
        v.scene.add(verts6)
    v.run()


def save_sequence(v, data0, save_path, data1=None,):
    v.reset()
    smpl0 = SMPLSequence.from_custom_npy(data=data0, z_up=True, color=(0.6, 0.5, 0.5, 1.0))
    v.scene.add(smpl0)
    if data1 is not None:
        smpl1 = SMPLSequence.from_custom_npy(data=data1, z_up=True, color=(0.5, 0.5, 0.6, 1.0))
        v.scene.add(smpl1)
    # v.scene.camera
    v.lock_to_node(smpl0, (4, 4, 4), smooth_sigma=5.0)
    v.save_video(video_dir=save_path, output_fps=30, rotate_camera=False, rotation_degrees=360.0)
    print("video saved at ", save_path)


def resample_motion_data(data, target_fps, original_fps=30.0):
    """
    Resample motion data to a different framerate.
    
    Args:
        data: Motion data dictionary with 'poses', 'transl', etc.
        target_fps: Target framerate
        original_fps: Original framerate (default: 30.0)
    
    Returns:
        Resampled motion data dictionary
    """
    if target_fps == original_fps:
        return data
    
    from scipy import interpolate
    import numpy as np
    
    # Calculate time points
    original_frames = data['poses'].shape[0]
    original_time = np.linspace(0, original_frames / original_fps, original_frames)
    target_frames = int(original_frames * target_fps / original_fps)
    target_time = np.linspace(0, original_frames / original_fps, target_frames)
    
    # Resampled data
    resampled_data = {}
    
    # Resample poses
    poses_reshaped = data['poses'].reshape(-1, 3)
    f_poses = interpolate.interp1d(original_time, poses_reshaped, axis=0, kind='linear')
    resampled_poses = f_poses(target_time)
    resampled_data['poses'] = resampled_poses.reshape(target_frames, -1)
    
    # Resample translation
    f_transl = interpolate.interp1d(original_time, data['transl'], axis=0, kind='linear')
    resampled_data['transl'] = f_transl(target_time)
    
    # Resample global orientation
    f_global_orient = interpolate.interp1d(original_time, data['global_orient'], axis=0, kind='linear')
    resampled_data['global_orient'] = f_global_orient(target_time)
    
    # Copy other data
    resampled_data['betas'] = data['betas']
    resampled_data['meta'] = data['meta']
    
    return resampled_data

def save_sequence_usd(v, data0, save_path, data1=None, export_as_directory=False, ascii=False, verbose=False, scene_fps=30.0, playback_fps=30.0, motion_fps=None):
    """
    Save sequence as USD file.
    
    Args:
        v: Viewer instance
        data0: Primary SMPL data dictionary
        save_path: Path to save the USD file (without extension)
        data1: Optional secondary SMPL data dictionary
        export_as_directory: If True, exports as directory with textures
        ascii: If True, exports as ASCII USD (.usda) instead of binary (.usd)
        verbose: If True, prints detailed export information
        scene_fps: Base framerate of the scene (default: 30.0)
        playback_fps: Playback framerate for animation (default: 30.0)
        motion_fps: Target framerate for motion data resampling (default: None, uses original)
    """
    v.reset()
    
    # Set scene and playback framerate
    v.scene.fps = scene_fps
    v.playback_fps = playback_fps
    
    # Resample motion data if requested
    if motion_fps is not None:
        data0 = resample_motion_data(data0, motion_fps)
        if data1 is not None:
            data1 = resample_motion_data(data1, motion_fps)
    
    # Create SMPL sequences
    smpl0 = SMPLSequence.from_custom_npy(
        data=data0, 
        z_up=False, 
        color=(0.05, 0.05, 1.0, 1.0)
    )
    v.scene.add(smpl0)
    
    if data1 is not None:
        smpl1 = SMPLSequence.from_custom_npy(
            data=data1, 
            z_up=False, 
            color=(0.05, 0.05, 1.0, 1.0)
        )
        v.scene.add(smpl1)
    
    # v.scene.camera
    v.lock_to_node(smpl0, (4, 4, 4), smooth_sigma=5.0)
    
    # Disable export for all scene elements except SMPL sequences
    for node in v.scene.nodes:
        if not isinstance(node, SMPLSequence):
            node.export_usd_enabled = False
        else:
            node.export_usd_enabled = True
    
    # Export to USD
    v.export_usd(save_path, export_as_directory=export_as_directory, verbose=verbose, ascii=ascii)
    print(f"USD file saved at {save_path} (Scene FPS: {scene_fps}, Playback FPS: {playback_fps})")


if __name__ == "__main__":
    '''load_pkl_path is the path to the pkl files you want to visualize. '''
    load_pkl_path = os.path.join('outputs', 'scemos_infer', 'test')
    
    

    all_seqs = sorted(glob.glob(load_pkl_path + '/*.pkl')   ) 
    print('total number of sequence: ', len(all_seqs))
    for seq_idx in range(0, len(all_seqs)):
        # motion_name = os.path.basename(all_seqs[seq_idx]).split('.')[0].rsplit('_heightmap_contacts')[0]
        filename = os.path.basename(all_seqs[seq_idx]).split('.')[0] 
        foldername = os.path.dirname(all_seqs[seq_idx])
        with open(all_seqs[seq_idx], 'rb') as _pf:
            pkl_data = pickle.load(_pf)
        print(pkl_data['texts'])

        sv = pkl_data.get('scene_vertices')
        scene_arg = [sv, None] if sv is not None else None
        visualize_sequence(
            scene_data=scene_arg,
            person1=[pkl_data['gt_vertices'], pkl_data['gt_faces']],
            person2=[pkl_data['pred_vertices'], pkl_data['pred_faces']],
            # item3=[pkl_data['local_vertices'].reshape(pkl_data['local_vertices'].shape[0], -1, 3), None],
            is_permute=False,
        )
    