"""Online heightmap + contact geometry for VQ-VAE decoding (matches preprocess_trumans)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from dataset_prep.trumans_loader import BBOX_SCALE

# Preprocessed pickles and VQ-VAE (scene_dim=1024) use 32x32 heightmaps, not NB_VOXELS=16.
HEIGHTMAP_GRID_SIZE = 32
from utils.smplx_util import across_joint_indx, up_joint_indx

r_hip, l_hip, r_sdr, l_sdr = across_joint_indx
pelvis, _, _, head = up_joint_indx

CONTACT_MAX_VERTICES = 2000
SEARCH_RADIUS = 0.1
PATCH_SIZE = 4000


def _to_tensor3d(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x) + eps)


def causal_geom_index(token_idx: int) -> int:
    """Subsampled heightmap index that conditions VQ token ``token_idx`` (training convention)."""
    return 0 if token_idx < 2 else token_idx - 1


def build_causal_conditioning(
    heightmaps: torch.Tensor,
    contact_maps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same shift as training: cond[t] uses geometry from t-1 (duplicate frame 0)."""
    heightmap_cond = torch.cat((heightmaps[:, 0:1], heightmaps[:, :-1]), dim=1)
    contact_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
    return heightmap_cond, contact_cond


def create_meshgrid_from_corners(
    top_left: torch.Tensor,
    top_right: torch.Tensor,
    bottom_left: torch.Tensor,
    bottom_right: torch.Tensor,
    grid_size: int = HEIGHTMAP_GRID_SIZE,
) -> torch.Tensor:
    """Return [grid_size, grid_size, 2] xz world coordinates."""
    device = top_left.device
    u = torch.linspace(0, 1, grid_size, device=device)
    v = torch.linspace(0, 1, grid_size, device=device)
    U, V = torch.meshgrid(u, v, indexing="ij")
    U = U.unsqueeze(-1)
    V = V.unsqueeze(-1)
    top_edge = top_left.unsqueeze(0).unsqueeze(0) * (1 - U) + top_right.unsqueeze(0).unsqueeze(0) * U
    bottom_edge = bottom_left.unsqueeze(0).unsqueeze(0) * (1 - U) + bottom_right.unsqueeze(0).unsqueeze(0) * U
    return top_edge * (1 - V) + bottom_edge * V


def create_heightmap_from_vertices(
    grid_xz: torch.Tensor,
    local_grid_vertices: torch.Tensor,
    grid_size: int = HEIGHTMAP_GRID_SIZE,
    search_radius: float = SEARCH_RADIUS,
) -> torch.Tensor:
    """Max-y heightmap [grid_size, grid_size] from local scene points."""
    grid_points = grid_xz.reshape(-1, 2)
    vertices_xz = local_grid_vertices[:, [0, 2]]
    vertices_y = local_grid_vertices[:, 1]
    distances = torch.cdist(grid_points, vertices_xz)
    mask = distances < search_radius
    grid_y = torch.zeros(grid_size * grid_size, device=grid_xz.device)
    for i in range(grid_size * grid_size):
        point_mask = mask[i]
        if point_mask.any():
            grid_y[i] = torch.max(vertices_y[point_mask])
    return grid_y.reshape(grid_size, grid_size)


def extract_scene_patch(
    human_points: torch.Tensor,
    scene_vertices: torch.Tensor,
    distance_threshold: float = BBOX_SCALE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    device = scene_vertices.device if isinstance(scene_vertices, torch.Tensor) else human_points.device
    scene_vertices = _to_tensor3d(scene_vertices, device)
    human_points = _to_tensor3d(human_points, device)
    distances = torch.cdist(scene_vertices, human_points)
    valid_indices = torch.any(distances <= distance_threshold, dim=1)
    filtered_scene = scene_vertices[valid_indices]
    if filtered_scene.shape[0] == 0:
        idx = torch.randint(0, scene_vertices.shape[0], (patch_size,), device=device)
        return scene_vertices[idx]
    replace = filtered_scene.shape[0] < patch_size
    if replace:
        idx = torch.randint(0, filtered_scene.shape[0], (patch_size,), device=device)
        return filtered_scene[idx]
    idx = torch.randperm(filtered_scene.shape[0], device=device)[:patch_size]
    return filtered_scene[idx]


def across_forward_from_joints(body_joints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single frame joints [J, 3] -> across (xz), forward (xz) unit vectors."""
    across = body_joints[l_hip] - body_joints[r_hip]
    across = normalize(across)
    up = body_joints[head] - body_joints[pelvis]
    up = normalize(up)
    forward = normalize(torch.linalg.cross(across, up))
    forward = forward.clone()
    across = across.clone()
    forward[..., 1] = 0.0
    across[..., 1] = 0.0
    return across, forward


def grid_corners_from_joints(
    body_joints: torch.Tensor,
    across: Optional[torch.Tensor] = None,
    forward: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pelvis-centred 2D box corners in xz (same layout as preprocess ``create_heightmap``)."""
    if across is None or forward is None:
        across, forward = across_forward_from_joints(body_joints)
    root = body_joints[pelvis].clone()
    root[..., 1] = 0.0
    top_left = (forward + across) + root
    top_right = (forward - across) + root
    bottom_left = -(forward - across) + root
    bottom_right = -(forward + across) + root
    for c in (top_left, top_right, bottom_left, bottom_right):
        c[..., 1] = 0.0
    return top_left, top_right, bottom_left, bottom_right


def find_contact_vertices(
    local_grid_vertices: torch.Tensor,
    person_vertices: torch.Tensor,
    contact_threshold: float = 0.05,
    max_vertices: int = CONTACT_MAX_VERTICES,
) -> torch.Tensor:
    device = local_grid_vertices.device
    non_ground_mask = torch.abs(local_grid_vertices[:, 1]) > 0.01
    non_ground_vertices = local_grid_vertices[non_ground_mask]
    if non_ground_vertices.shape[0] == 0:
        return torch.zeros((max_vertices, 3), device=device)

    distances = torch.cdist(non_ground_vertices, person_vertices)
    min_distances = torch.min(distances, dim=1)[0]
    contact_mask = min_distances < contact_threshold
    contact_vertices = non_ground_vertices[contact_mask]

    if contact_vertices.shape[0] > max_vertices:
        contact_distances = min_distances[contact_mask]
        _, closest_indices = torch.sort(contact_distances)
        contact_vertices = contact_vertices[closest_indices[:max_vertices]]
    elif contact_vertices.shape[0] < max_vertices:
        padding = torch.zeros(
            (max_vertices - contact_vertices.shape[0], 3), device=device
        )
        contact_vertices = torch.cat([contact_vertices, padding], dim=0)
    return contact_vertices


def contact_map_from_vertices(
    contact_vertices: torch.Tensor,
    body_joints: torch.Tensor,
) -> torch.Tensor:
    """Same rule as ``TrumansDataset``: L2 distance to pelvis, zero if padded contact slot."""
    pelvis_pos = body_joints[pelvis]
    non_zero_mask = contact_vertices.abs().sum(dim=-1) != 0
    dist = torch.linalg.norm(contact_vertices - pelvis_pos.unsqueeze(0), dim=-1)
    return torch.where(non_zero_mask, dist, torch.zeros_like(dist))


def compute_frame_geometry(
    body_joints: torch.Tensor,
    person_vertices: torch.Tensor,
    scene_vertices: torch.Tensor,
    *,
    grid_size: int = HEIGHTMAP_GRID_SIZE,
    patch_size: int = PATCH_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        body_joints: [J, 3]
        person_vertices: [V, 3]
        scene_vertices: [N, 3] static scene cloud
    Returns:
        heightmap [grid_size, grid_size], contact_map [CONTACT_MAX_VERTICES]
    """
    device = scene_vertices.device if isinstance(scene_vertices, torch.Tensor) else torch.device("cpu")
    body_joints = _to_tensor3d(body_joints, device)
    person_vertices = _to_tensor3d(person_vertices, device)
    scene_vertices = _to_tensor3d(scene_vertices, device)
    local_pts = extract_scene_patch(
        person_vertices, scene_vertices, patch_size=patch_size
    )
    tl, tr, bl, br = grid_corners_from_joints(body_joints)
    grid_xz = create_meshgrid_from_corners(
        tl[[0, 2]], tr[[0, 2]], bl[[0, 2]], br[[0, 2]], grid_size=grid_size
    )
    heightmap = create_heightmap_from_vertices(grid_xz, local_pts, grid_size=grid_size)
    contact_verts = find_contact_vertices(local_pts, person_vertices)
    contact_map = contact_map_from_vertices(contact_verts, body_joints)
    return heightmap, contact_map


def compute_geometry_from_motion(
    motion_unnorm: torch.Tensor,
    betas: torch.Tensor,
    scene_vertices: torch.Tensor,
    frame_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SMPL-X geometry at one subsampled motion frame.

    Args:
        motion_unnorm: [T, D] un-normalized motion features
        betas: [1, D_b] or [D_b]
        scene_vertices: [N, 3]
        frame_idx: index into ``motion_unnorm`` (clamped)
    """
    from dataset_prep.motion_process import mot_to_smplx_verts

    frame_idx = int(max(0, min(frame_idx, motion_unnorm.shape[0] - 1)))
    motion_slice = motion_unnorm[: frame_idx + 1]
    _, verts, _, joints = mot_to_smplx_verts(motion_slice, betas)
    device = scene_vertices.device if isinstance(scene_vertices, torch.Tensor) else torch.device("cpu")
    body_joints = _to_tensor3d(joints[-1, :22], device)
    person_vertices = _to_tensor3d(verts[-1], device)
    scene_vertices = _to_tensor3d(scene_vertices, device)
    return compute_frame_geometry(body_joints, person_vertices, scene_vertices)


def compute_geometry_sequence_from_motion(
    motion_unnorm: torch.Tensor,
    betas: torch.Tensor,
    scene_vertices: torch.Tensor,
    num_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build subsampled heightmap/contact stacks [num_frames, ...]."""
    heightmaps: List[torch.Tensor] = []
    contacts: List[torch.Tensor] = []
    for t in range(num_frames):
        hm, cm = compute_geometry_from_motion(motion_unnorm, betas, scene_vertices, t)
        heightmaps.append(hm)
        contacts.append(cm)
    return torch.stack(heightmaps, dim=0), torch.stack(contacts, dim=0)


def get_scene_vertices_from_batch(batch: dict, sample_idx: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Resolve static scene cloud from dataloader batch."""
    sv = batch.get("scene_vertex")
    if sv is None:
        raise ValueError(
            "online geometry requires scene point cloud: pass load_scene_vertex=True "
            "to TrumansDataset or --load_scene_vertex to inference."
        )
    if isinstance(sv, list):
        out = sv[sample_idx]
    else:
        out = sv[sample_idx]
    target_device = device
    if target_device is None and isinstance(out, torch.Tensor):
        target_device = out.device
    if target_device is None:
        target_device = torch.device("cpu")
    return _to_tensor3d(out, target_device)
