from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from scipy import interpolate
from smplx import SMPLX
from utils.smplx_util import *
from utils.transformations import *
from utils.utilities import *

import torch
from tqdm import tqdm

BODYHAND_JOINTS = 55
BODY_JOINTS = 22
HAND_JOINTS = 15
ROOTTRANS_END_INDEX = 3
ROOTROT_END_INDEX = ROOTTRANS_END_INDEX + 6
BODYROT_END_INDEX = ROOTROT_END_INDEX + (BODY_JOINTS - 1) * 6
BODYJOINT_END_INDEX = BODYROT_END_INDEX + (BODY_JOINTS - 1) * 3
BODYJOINTVEL_END_INDEX = BODYJOINT_END_INDEX + (BODY_JOINTS - 1) * 3
BODYHANDROT_END_INDEX = ROOTROT_END_INDEX + (BODYHAND_JOINTS - 1) * 6
BODYHANDJOINT_END_INDEX = BODYHANDROT_END_INDEX + (BODYHAND_JOINTS - 1) * 3
BODYHANDJOINTVEL_END_INDEX = BODYHANDJOINT_END_INDEX + (BODYHAND_JOINTS -1) * 3
FOOT_CONTACT_DIM = 4
MOTION_FEATS_BODY_ONLY = 265
MOTION_FEATS_BODYHAND = 664
face_joint_indx = [2, 1, 17, 16]
trans_matrix_xyz_to_xzy = to_tensor(torch.Tensor([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, -1.0, 0.0]]), device=DEVICE)

trans_matrix_xzy_to_xyz = to_tensor(torch.Tensor([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [0.0, 1.0, 0.0]]), device=DEVICE)

def batch_smplx_to_pos3d(person_data):
    device = person_data['transl'].device 
    B = person_data['transl'].shape[0]
    T = person_data['transl'].shape[1]
    person_data = smplx_batchify(person_data)
    person_vertices, person_joints = smplx_to_pos3d(person_data)
    person_vertices = person_vertices.reshape(B, T, -1, 3).to(device)
    person_joints = person_joints.reshape(B, T, -1, 3).to(device)
    return person_vertices, person_joints

def smplx_to_pos3d(data):
    device = data['pose_body'].device
    smplx = SMPLX(model_path=SMPLX_FOLDER, betas=data['betas'][0:1, :10], gender=data['meta']['gender'], \
        batch_size=len(data['pose_body']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True).to(device)

    bodymodel = smplx.forward(
        global_orient=to_tensor(data['global_orient'], device=device),
        body_pose=to_tensor(data['pose_body'], device=device),
        jaw_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        leye_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        reye_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        left_hand_pose=torch.zeros((len(data['pose_body']), 45)).to(device=device),
        right_hand_pose=torch.zeros((len(data['pose_body']), 45)).to(device=device),
        transl=to_tensor(data['transl'], device=device), 
        betas=to_tensor(data['betas'][0:1, :10], device=device), 
        )
    vertices3d = bodymodel.vertices
    keypoints3d = bodymodel.joints
    nframes = keypoints3d.shape[0]
    return vertices3d, keypoints3d

def root_invariant_smplx_to_pos3d(data):
    device = data['pose_body'].device
    smplx = SMPLX(model_path=SMPLX_FOLDER, betas=data['betas'][:, :10], gender=data['meta']['gender'], \
        batch_size=len(data['pose_body']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True).to(device)

    bodymodel = smplx.forward(
        body_pose=to_tensor(data['pose_body'], device=device),
        jaw_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        leye_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        reye_pose=torch.zeros((len(data['pose_body']), 3)).to(device=device),
        left_hand_pose=to_tensor(data['pose_hand'][:, 0:45], device=device),
        right_hand_pose=to_tensor(data['pose_hand'][:, 45:], device=device),
        betas=to_tensor(data['betas'][0:1, :10], device=device), 
        )
    keypoints3d = bodymodel.joints
    nframes = keypoints3d.shape[0]
    return keypoints3d


def delta_mot2mot_feats(mot_init, delta_mots):
    # Create a copy to avoid in-place operations
    delta_mots_copy = delta_mots.clone()
    delta_mots_copy[:, 0:1, :3] = delta_mots[:, 0:1, :3] + mot_init[:, :, :3] 
    # motions = torch.cat((mot_init, delta_mots), dim=1)
    return torch.cat((torch.cumsum(delta_mots_copy[..., :3], dim=1), delta_mots_copy[..., 3:]), dim=-1)

def _mot_to_smplx(motion, betas, gender, with_fingers=False, root_offset=None, interpolate_by=None):
    assert motion.ndim == 2
    T = len(motion)
    if with_fingers:
        motion = motion[..., :BODYHANDROT_END_INDEX]
    else:
        motion = motion[..., :BODYROT_END_INDEX]
    if interpolate_by:
        T1 = T * interpolate_by
        x = np.linspace(0, T-1 ,T)
        x_new = np.linspace(0, T-1 ,T1)
        motion_int = np.zeros((T1, motion.shape[-1]))
        for v1 in range(0, motion.shape[-1]):
            p_ = to_np(motion[:, v1])
            f_p = interpolate.interp1d(x, p_, kind = 'linear')
            motion_int[:, v1] = f_p(x_new)
        motion = to_tensor(motion_int, device=motion.device)
        T = T1
        betas = betas[0:1].repeat(T1, 1)

    root_transl = motion[:, :ROOTTRANS_END_INDEX]
    if root_offset is not None:
        root_transl = root_transl - root_offset
    root_orient = d62aa(motion[:, ROOTTRANS_END_INDEX:ROOTROT_END_INDEX])
    poses = to_tensor(torch.zeros((T, BODYHAND_JOINTS*3)), device=motion.device)
    poses[:, :3] = root_orient
    if with_fingers:         # only body
        poses[:, 1*3:BODYHAND_JOINTS*3] = d62aa(motion[:, ROOTROT_END_INDEX:BODYHANDROT_END_INDEX].reshape(T, (BODYHAND_JOINTS-1), 6) ).reshape(T, (BODYHAND_JOINTS-1)*3)
    else:    
        poses[:, 1*3:BODY_JOINTS*3] = d62aa(
            motion[:, ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(T, (BODY_JOINTS-1), 6) ).reshape(T, (BODY_JOINTS-1)*3)
    data = {
        'transl': to_np(root_transl),
        'global_orient': to_np(root_orient),
        'poses': to_np(poses),
        'betas': to_np(betas),
        'meta': {'gender': gender}
    }
    return data


def mot_to_smplx_verts(motion, betas, interpolate_by=None):
    assert motion.ndim == 2
    T = len(motion)
    if interpolate_by:
        T1 = T * interpolate_by
        x = np.linspace(0, T-1 ,T)
        x_new = np.linspace(0, T-1 ,T1)
        motion_int = np.zeros((T1, motion.shape[-1]))
        for v1 in range(0, motion.shape[-1]):
            p_ = to_np(motion[:, v1])
            f_p = interpolate.interp1d(x, p_, kind = 'cubic')
            # f_p = interpolate.interp1d(x, p_, kind = 'linear')
            motion_int[:, v1] = f_p(x_new)
        motion = to_tensor(motion_int, device=motion.device)
        T = T1
    

    root_transl = motion[:, :ROOTTRANS_END_INDEX]
    T = len(root_transl)
    root_orient = d62aa(motion[:, ROOTTRANS_END_INDEX:ROOTROT_END_INDEX], device=motion.device)
    poses = to_tensor(torch.zeros((T, BODY_JOINTS*3)), device=motion.device)
    poses[:, :3] = root_orient
    
    poses[:, 1*3:BODY_JOINTS*3] = d62aa(
            motion[:, ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(T, (BODY_JOINTS-1), 6) , device=motion.device).reshape(T, (BODY_JOINTS-1)*3)
    pose_hand = torch.zeros((T, 90)).float().to(motion.device)
    body_vertices, body_faces, joints = SMPLX_Util.get_body_vertices_sequence(
                BODY_MODEL_FOLDER, 
                (to_np(root_transl), 
                 to_np(root_orient), 
                 to_np(betas),
                 to_np(poses[:, 3:].reshape(-1, 63)),
                 to_np(pose_hand)),
                num_betas=10
            )
    data = {
        'transl': to_np(root_transl),
        'global_orient': to_np(root_orient),
        'poses': to_np(poses),
        'betas': to_np(betas),
        'meta': {'gender': 'male'}
    }
    return data, body_vertices, body_faces, joints

def smplx_batchify(data):
    B = data['transl'].shape[0]
    T = data['transl'].shape[1]
    
    data['betas'] = data['betas'].reshape(B*T, 10)
    data['transl'] = data['transl'].reshape(B*T, 3)
    data['global_orient'] = data['global_orient'].reshape(B*T, 3)
    data['pose_body'] = data['pose_body'].reshape(B*T, BODY_JOINTS - 1, 3)
    data['body_pose'] = data['pose_body'].reshape(B*T, BODY_JOINTS - 1, 3)

    return data

def batch_mot_to_smplx(motion, betas, with_fingers=False):
    B, T, dim = motion.shape
    root_transl = to_tensor(motion[..., :ROOTTRANS_END_INDEX], device=motion.device)
    root_orient = to_tensor(d62aa(motion[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX], device=motion.device).reshape(B, T, 3), device=motion.device)
    poses = to_tensor(torch.zeros((B, T, BODY_JOINTS, 3)), device=motion.device)
    poses[:, :, 0] = root_orient
    
    poses[:, :, 1:BODY_JOINTS] = d62aa(motion[..., ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(B, T, BODY_JOINTS - 1, 6), device= motion.device).reshape(B, T, BODY_JOINTS - 1, 3)
    betas = betas.repeat(1, T, 1)
    data = {
        'betas': betas,
        'transl': root_transl.reshape(B, T, 3),
        'global_orient': root_orient.reshape(B, T, 3),
        'pose_body': poses[:, :, 1:BODY_JOINTS].reshape(B, T, -1, 3),
        'meta': {'gender': 'male'}
    }
    return data

def mot_to_smplx(motion, betas, with_fingers=False):
    T, dim = motion.shape
    root_transl = to_tensor(motion[..., :ROOTTRANS_END_INDEX], device=motion.device)
    root_orient = to_tensor(d62aa(motion[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX], device=motion.device).reshape(T, 3), device=motion.device)
    poses = to_tensor(torch.zeros((T, BODYHAND_JOINTS, 3)), device=motion.device)
    poses[:, 0] = root_orient
    if with_fingers:         # only body
        poses[ :, 1:BODYHAND_JOINTS] = d62aa(motion[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX].reshape(T, BODYHAND_JOINTS - 1, 6), device= motion.device).reshape(T, BODYHAND_JOINTS - 1, 3)
    else:
        poses[ :, 1:BODY_JOINTS] = d62aa(motion[..., ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(T, BODY_JOINTS - 1, 6), device= motion.device).reshape(T, BODY_JOINTS - 1, 3)
    # betas = betas.repeat(1, T, 1)
    data = {
        'betas': betas,
        'transl': root_transl.reshape(T, 3),
        'global_orient': root_orient.reshape(T, 3),
        'pose_body': poses[:, 1:BODY_JOINTS].reshape(T, -1),
        'pose_hand' : poses[:, 25:BODYHAND_JOINTS].reshape(T, -1),
        'meta': {'gender': 'male'}
    }
    return data

def mot_to_smplx_np(motion, betas, with_fingers=False):
    T, dim = motion.shape
    root_transl = to_tensor(motion[..., :ROOTTRANS_END_INDEX], device=motion.device)
    root_orient = to_tensor(d62aa(motion[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX], device=motion.device).reshape(T, 3), device=motion.device)
    poses = to_tensor(torch.zeros((T, BODYHAND_JOINTS, 3)), device=motion.device)
    poses[:, 0] = root_orient
    if with_fingers:         # only body
        poses[ :, 1:BODYHAND_JOINTS] = d62aa(motion[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX].reshape(T, BODYHAND_JOINTS - 1, 6), device= motion.device).reshape(T, BODYHAND_JOINTS - 1, 3)
    else:
        poses[ :, 1:BODY_JOINTS] = d62aa(motion[..., ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(T, BODY_JOINTS - 1, 6), device= motion.device).reshape(T, BODY_JOINTS - 1, 3)
    # betas = betas.repeat(1, T, 1)
    data = {
        'betas': betas,
        'transl': root_transl.reshape(T, 3).detach().cpu().numpy(),
        'global_orient': root_orient.reshape(T, 3).detach().cpu().numpy(),
        'pose_body': poses[:, 1:BODY_JOINTS].reshape(T, -1).detach().cpu().numpy(),
        'pose_hand' : poses[:, 25:BODYHAND_JOINTS].reshape(T, -1).detach().cpu().numpy(),
        'meta': {'gender': 'male'}
    }
    return data

def reverse_process_mot(data, with_fingers=False):
    root_transl = data[..., :ROOTTRANS_END_INDEX]
    root_rot = data[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]
    foot_contact = data[..., -FOOT_CONTACT_DIM:]
    if with_fingers:
        body_rot = data[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX]
        local_jointpos = data[..., BODYHANDROT_END_INDEX:BODYHANDJOINT_END_INDEX]
        local_jointvel = data[..., BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL_END_INDEX]
    else:
        body_rot = data[..., ROOTROT_END_INDEX:BODYROT_END_INDEX]
        local_jointpos = data[..., BODYROT_END_INDEX:BODYJOINT_END_INDEX]
        local_jointvel = data[..., BODYJOINT_END_INDEX:BODYJOINTVEL_END_INDEX]
    return root_transl, root_rot, body_rot, local_jointpos, local_jointvel, foot_contact


def process_human_motion(person_data, feet_thre=0.002, with_fingers=False):
    device = person_data['transl'].device
    if with_fingers:
        num_jts = BODYHAND_JOINTS
        aa_rotation = torch.cat((person_data['global_orient'], person_data['pose_body'], person_data['pose_hand']), -1)
    else:
        num_jts = BODY_JOINTS
        aa_rotation = torch.cat((person_data['global_orient'], person_data['pose_body']), -1)
    body_joint_positions = root_invariant_smplx_to_pos3d(person_data)[:, :num_jts]
    """ Get Foot Contacts """
    def foot_detect(positions, thres):
        velfactor, heightfactor = to_tensor(np.array([thres, thres]), device=device), to_tensor(np.array([0.12, 0.05]), device=device)
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l, 2]
        feet_l = to_tensor(torch.zeros((len(positions), 2)), device=device)
        feet_r = to_tensor(torch.zeros((len(positions), 2)), device=device)
        feet_l[1:] = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor))
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r, 2]
        feet_r[1:] = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor))
        return feet_l, feet_r
    feet_l, feet_r = foot_detect(body_joint_positions, feet_thre)

    '''Get root invariant local joint velocities'''
    joint_vels = to_tensor(torch.zeros_like(body_joint_positions), device=device)
    joint_vels[1:] = body_joint_positions[1:] - body_joint_positions[:-1]
    '''Get 6D rotation for all joints'''
    rot6d = aa2d6(aa_rotation.reshape(-1, num_jts , 3), device=device)
    processed_data = person_data['transl']                          # shape=(:, 3)
    processed_data = torch.cat([processed_data, rot6d.reshape(len(rot6d), -1)], dim=-1)             # shape=(:, 132)
    processed_data = torch.cat([processed_data, 
                                body_joint_positions[:, 1:num_jts].reshape(len(body_joint_positions), -1)], dim=-1)     # shape=(:, 63)
    processed_data = torch.cat([processed_data, joint_vels[:, 1:num_jts].reshape(len(joint_vels), -1)], dim=-1)         # shape=(:, 63)
    processed_data = torch.cat([processed_data, feet_l, feet_r], dim=-1)    # shape=(:, 4)
    return processed_data


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_rot(data):
    # dataset [bs, seqlen, 263/251] HumanML/KIT
    joints_num = BODY_JOINTS if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions

def new_process_mot(annot_dict, feet_thre=0.002):
    seq_len = len(annot_dict['trans'])
    _, _, joint_pos_ = SMPLX_Util.get_body_vertices_sequence(
            BODY_MODEL_FOLDER, 
            (np.zeros((seq_len, 3), dtype="float32"), 
            np.zeros((seq_len, 3), dtype="float32"),
            annot_dict['betas'],
            d62aa(torch.tensor(annot_dict['pose_body_d6 '])).reshape(-1, 63).cpu().numpy(),
                annot_dict['pose_hand']),
            num_betas=10
        )
    joint_pos_ = to_tensor(joint_pos_[:, :BODY_JOINTS])
    '''Get Binary Foot Contacts ''' 
    def foot_detect(positions, thres):
        velfactor, heightfactor = to_tensor(torch.asarray([thres, thres]), device=positions.device), to_tensor(torch.asarray([3.0, 2.0]), device=positions.device)
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l, 2]
        feet_l = to_tensor(torch.zeros((len(positions), 2)), device=positions.device)
        feet_r = to_tensor(torch.zeros((len(positions), 2)), device=positions.device)
        feet_l[1:] = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor))
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r, 2]
        feet_r[1:] = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor))
        return feet_l, feet_r
    feet_l, feet_r = foot_detect(joint_pos_, feet_thre)

    '''Get root invariant local joint velocities'''
    joint_vels = to_tensor(torch.zeros_like(joint_pos_), device=joint_pos_.device)
    joint_vels[1:] = joint_pos_[1:] - joint_pos_[:-1]
    '''Get 6D rotation for all joints'''
    orient_6d = aa2d6(to_tensor(annot_dict['orient'])).reshape(seq_len, 1, 6)
    rot6d = torch.cat((orient_6d, to_tensor(annot_dict['pose_body_d6 '])), dim=1)

    processed_data = annot_dict['trans']                                                               # Global Root translation, shape=(:, 3)
    processed_data = torch.cat([processed_data, rot6d.reshape(seq_len, BODY_JOINTS*6)], dim=-1)                       # 6D rotations of body joints, shape=(:, 132)
    processed_data = torch.cat([processed_data, joint_pos_.reshape(seq_len, BODY_JOINTS*3)], dim=-1)  # Local joint positions wrt root, shape =(:, 63)
    processed_data = torch.cat([processed_data, joint_vels[:, 1:BODY_JOINTS].reshape(seq_len, (BODY_JOINTS-1)*3)], dim=-1)             # Global joint velocities, shape=(:, 66)
    
    processed_data = torch.cat([processed_data, feet_l, feet_r], dim=-1)                           # Binary foot contact, shape=(:, 4)
    return processed_data

def normalize_vector(x, eps=1e-8):
    return x / (torch.linalg.norm(x) + eps)

def interpolate_person_data(person_data, num_interpolate):
    """
    Interpolate all numpy arrays in person_data dictionary by a specified factor.
    
    Args:
        person_data (dict): Dictionary containing SMPL-X parameters with numpy arrays
        num_interpolate (int): Number of interpolation points between each original frame
        
    Returns:
        dict: Interpolated person_data with same structure but longer sequences
    """
    interpolated_data = {}
    
    for key, value in person_data.items():
        if key == 'meta':
            # Copy metadata as-is
            interpolated_data[key] = value.copy()
        elif isinstance(value, np.ndarray):
            # Interpolate numpy arrays
            original_shape = value.shape
            seq_len = original_shape[0]
            
            if seq_len == 1:
                # For single-frame data like betas, repeat the frame
                new_seq_len = (seq_len - 1) * num_interpolate + 1
                interpolated_data[key] = np.repeat(value, new_seq_len, axis=0)
            else:
                # For multi-frame data, interpolate between frames
                new_seq_len = (seq_len - 1) * num_interpolate + 1
                interpolated_shape = (new_seq_len,) + original_shape[1:]
                interpolated_array = np.zeros(interpolated_shape, dtype=value.dtype)
                
                for i in range(seq_len - 1):
                    start_idx = i * num_interpolate
                    end_idx = (i + 1) * num_interpolate
                    
                    # Linear interpolation between consecutive frames
                    for j in range(num_interpolate):
                        alpha = j / num_interpolate
                        interpolated_array[start_idx + j] = (
                            (1 - alpha) * value[i] + alpha * value[i + 1]
                        )
                
                # Set the last frame
                interpolated_array[-1] = value[-1]
                interpolated_data[key] = interpolated_array
        else:
            # Copy non-array data as-is
            interpolated_data[key] = value
    
    return interpolated_data

def save_smplx_as_npz(data_dict_, filename):
    
    data = {
            'gender': data_dict_['meta']['gender'],
            'surface_model_type': 'smplx',
            'mocap_frame_rate': 30,
            # 'mocap_time_length': 20,
            'trans':np.array(data_dict_['transl'], dtype=np.float32),
            'poses':np.array(data_dict_['poses'], dtype=np.float32),
            'betas':np.array(data_dict_['betas'], dtype=np.float32),
            'root_orient':np.array(data_dict_['global_orient'], dtype=np.float32),
            'pose_body':np.array(data_dict_['pose_body'], dtype=np.float32),
        }
    np.savez(filename, **data)


def smooth_root_joint(sequence, root_joint_idx=0, window_size=5):
    """
    Apply a moving average filter to the root joint in a human motion sequence.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window (default is 5).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]
    
    # Create a kernel for each of the three dimensions
    kernel = torch.ones(3, 1, window_size) / window_size  # Shape: [3, 1, window_size]
    kernel = kernel.to(sequence.device)
    # Apply 1D convolution along the time dimension for each batch and each root joint coordinate
    smoothed_positions = F.conv1d(
        root_joint_positions.permute(0, 2, 1),  # Reshape to [B, 3, T]
        kernel, padding=window_size // 2, groups=3
    ).permute(0, 2, 1)  # Reshape back to [B, T, 3]

    # Replace the root joint positions in the original sequence with smoothed values
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence


import torch

def batch_smooth_root_joint_constrained(sequence, root_joint_idx=0, window_size=5, anchor_interval=10):
    """
    Apply a constrained moving average filter to the root joint in a human motion sequence.
    Anchors the root joint at specified intervals to prevent drift.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window (default is 5).
        anchor_interval (int): Interval for anchor points (default is 10).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]
    
    # Initialize smoothed positions with the original
    smoothed_positions = root_joint_positions.clone()
    
    # Iterate over frames, applying a moving average while anchoring
    for start in range(0, root_joint_positions.size(1) - window_size, anchor_interval):
        end = min(start + window_size, root_joint_positions.size(1))
        
        # Apply smoothing only between anchor points
        segment = root_joint_positions[:, start:end]
        smoothed_segment = torch.stack([
            segment[:, :, i].unfold(1, window_size, 1).mean(dim=2)
            for i in range(3)
        ], dim=2)  # Shape: [B, end-start, 3]
        
        # Update the smoothed segment in the output
        smoothed_positions[:, start:end] = smoothed_segment
    
    # Blend original positions at anchor points for constraint
    for start in range(0, root_joint_positions.size(1), anchor_interval):
        smoothed_positions[:, start] = root_joint_positions[:, start]
    
    # Update the original sequence with the smoothed root joint positions
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence

def smooth_root_joint_constrained(root_joint_positions, window_size=5, anchor_interval=60):
    """
    Apply a constrained moving average filter to the root joint in a human motion sequence.
    Anchors the root joint at specified intervals to prevent drift.

    Args:
        root_joint_positions (torch.Tensor): The motion sequence of shape [B, T, 3].
        
        window_size (int): Size of the moving average window (default is 5).
        anchor_interval (int): Interval for anchor points (default is 10).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    
    # Initialize smoothed positions with the original
    smoothed_positions = root_joint_positions.clone()
    
    # Iterate over frames, applying a moving average while anchoring
    for start in range(0, root_joint_positions.size(1) - window_size, anchor_interval):
        end = min(start + window_size, root_joint_positions.size(1))
        
        # Apply smoothing only between anchor points
        segment = root_joint_positions[:, start:end]
        smoothed_segment = torch.stack([
            segment[:, :, i].unfold(1, window_size, 1).mean(dim=2)
            for i in range(3)
        ], dim=2)  # Shape: [B, end-start, 3]
        
        # Update the smoothed segment in the output
        smoothed_positions[:, start:end] = smoothed_segment
    
    # Blend original positions at anchor points for constraint
    for start in range(0, root_joint_positions.size(1), anchor_interval):
        smoothed_positions[:, start] = root_joint_positions[:, start]

    return smoothed_positions


def smooth_root_joint_deltas(sequence, root_joint_idx=0, alpha_xz=0.5, alpha_y=0.1):
    """
    Apply a smoothing filter to the root joint's X, Y, and Z deltas, preserving the initial position.
    This reduces jumps by smoothing the changes in position instead of the absolute positions.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        alpha_xz (float): Smoothing factor for X and Z deltas (higher = more smoothing).
        alpha_y (float): Smoothing factor for Y deltas (higher = more smoothing).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]

    # Calculate deltas (differences) between consecutive frames
    deltas = root_joint_positions[:, 1:] - root_joint_positions[:, :-1]  # Shape: [B, T-1, 3]

    # Separate XZ and Y deltas
    xz_deltas = deltas[:, :, [0, 2]]
    y_deltas = deltas[:, :, 1:2]

    # Initialize smoothed deltas with the original deltas
    smoothed_xz_deltas = xz_deltas.clone()
    smoothed_y_deltas = y_deltas.clone()

    # Apply exponential smoothing on the deltas
    for t in range(1, deltas.size(1)):
        smoothed_xz_deltas[:, t] = alpha_xz * xz_deltas[:, t] + (1 - alpha_xz) * smoothed_xz_deltas[:, t - 1]
        smoothed_y_deltas[:, t] = alpha_y * y_deltas[:, t] + (1 - alpha_y) * smoothed_y_deltas[:, t - 1]

    # Reconstruct positions by cumulatively summing the smoothed deltas, starting from the initial position
    smoothed_positions = torch.cat([root_joint_positions[:, :1],  # Initial position
                                    root_joint_positions[:, :1] + torch.cumsum(
                                        torch.cat([smoothed_y_deltas, smoothed_xz_deltas], dim=2), dim=1)], dim=1)

    # Insert the smoothed root joint positions back into the original sequence
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence


def smooth_root_joint_XZ(sequence, root_joint_idx=0, window_size=5, y_window_size=3):
    """
    Apply a moving average filter to the root joint's X and Z directions in a batched human motion sequence,
    with minimal smoothing on Y to avoid jumps, ensuring the initial frames stay consistent.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window for X and Z directions.
        y_window_size (int): Size of the moving average window for Y direction (default is 3).

    Returns:
        torch.Tensor: Smoothed motion sequence with frame 0 unchanged for the root joint.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]
    
    # Keep the original root position at frame 0
    initial_position = root_joint_positions[:, :1, :]  # Shape: [B, 1, 3]
    
    # Create kernels for the X and Z directions
    kernel_xz = torch.ones(2, 1, window_size) / window_size  # Shape: [2, 1, window_size]
    kernel_xz = kernel_xz.to(sequence.device)

    # Create a kernel for the Y direction with minimal smoothing
    kernel_y = torch.ones(1, 1, y_window_size) / y_window_size  # Shape: [1, 1, y_window_size]
    kernel_y = kernel_y.to(sequence.device)
    
    # Extract X, Y, and Z positions separately, excluding frames 0 and 1
    xz_positions = root_joint_positions[:, 2:, [0, 2]]  # Shape: [B, T-2, 2]
    y_position = root_joint_positions[:, 2:, 1:2]       # Shape: [B, T-2, 1]

    # Apply 1D convolution to smooth the X and Z components from frame 2 onward
    padding_xz = (window_size - 1) // 2
    smoothed_xz = F.conv1d(
        xz_positions.permute(0, 2, 1),  # Reshape to [B, 2, T-2]
        kernel_xz, padding=padding_xz, groups=2
    ).permute(0, 2, 1)  # Reshape back to [B, T-2, 2]

    # Apply minimal smoothing on the Y component
    padding_y = (y_window_size - 1) // 2
    smoothed_y = F.conv1d(
        y_position.permute(0, 2, 1),  # Reshape to [B, 1, T-2]
        kernel_y, padding=padding_y
    ).permute(0, 2, 1)  # Reshape back to [B, T-2, 1]

    # Concatenate the initial position at frame 0 and keep frame 1 the same, then smooth from frame 2 onward
    smoothed_positions = root_joint_positions.clone()
    smoothed_positions[:, 2:, [0, 2]] = smoothed_xz
    smoothed_positions[:, 2:, 1:2] = smoothed_y
    smoothed_positions[:, 1:2, 1:2] = initial_position[:, :, 1:2]  # Keep frame 1's Y the same as frame 0's Y

    # Replace the root joint positions in the original sequence with smoothed values
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence

def prevent_foot_sliding_root_positions(motion_sequence, foot_indices, threshold=0.1):
    """
    Adjusts the root positions in a human motion sequence to prevent foot sliding based on vertical movement.

    Args:
        motion_sequence (torch.Tensor): A tensor of shape [T, J, 3] representing the motion sequence.
        foot_indices (list): Indices of the joints representing the feet (e.g., [1, 2] for left and right feet).
        threshold (float): Maximum allowable movement of the root joint when feet are stationary in the vertical direction.

    Returns:
        torch.Tensor: Adjusted root positions of shape [T, 3].
    """
    # Extract the number of frames
    T, J, _ = motion_sequence.shape
    
    # Extract root positions
    root_positions = motion_sequence[:, 0, :].clone()
    
    # Loop through the frames and adjust the root position if needed
    for t in range(1, T):
        # Calculate the vertical movement of the foot joints
        foot_positions_current = motion_sequence[t, foot_indices, 1]  # Y-coordinates of current frame
        foot_positions_previous = motion_sequence[t - 1, foot_indices, 1]  # Y-coordinates of previous frame
        foot_movement = foot_positions_current - foot_positions_previous
        
        # Check if vertical foot movement is below the threshold
        if torch.all(torch.abs(foot_movement) < threshold):
            root_positions[t] = root_positions[t - 1]  # Lock root position

    return root_positions