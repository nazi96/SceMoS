"""Pure-function implementations of the metrics in SceMoS Table 1.

Conventions used throughout:
    motion   : [B, T, D] tensor of un-normalized motion features (D = MOTION_FEATS_BODY_ONLY)
    joints   : [B, T, J, 3] body joints in world coordinates  (J ~= 22 for SMPL-X body)
    vertices : [B, T, V, 3] SMPL-X body vertices
    scene_pts: [B, N, 3] scene point cloud (per sample N may differ; pad with NaN if needed)

All functions are GPU-friendly (no numpy/cpu transfers inside the inner loops)
and return CPU scalars or small tensors so they can be averaged across batches.

References:
    [FID]          Heusel et al., NeurIPS 2017.
    [Diversity]    Guo et al., HumanML3D, CVPR 2022.
    [PFC]          Tseng et al., EDGE, CVPR 2023, eq. (7).
    [Contact/Pen]  Zhao et al., DIMOS, ICCV 2023; Jiang et al., TRUMANS, CVPR 2024.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# ---------------------------------------------------------------------------
#  Group A: feature-space metrics (need a motion feature extractor)
# ---------------------------------------------------------------------------

def calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (mu, sigma) of activations of shape [N, d]."""
    if activations.ndim != 2:
        raise ValueError(f"Expected [N, d], got {activations.shape}")
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6, scale: float = 1) -> float:
    """Frechet distance between two Gaussians N(mu1, sigma1) and N(mu2, sigma2).

    Implementation follows the standard FID code (Heusel et al., 2017).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # numerical fallback
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # sqrtm can return a small imaginary part when sigma1 @ sigma2 is nearly singular
    # (common for high-dim features + finite N). Same handling as common FID reference code.
    if np.iscomplexobj(covmean):
        m = float(np.max(np.abs(covmean.imag)))
        if m > 1e-2:
            raise ValueError(
                f"Fréchet sqrtm has large imaginary part ({m}); try more samples, "
                "lower feature dim, or increase frechet_distance eps."
            )
        covmean = covmean.real

    return scale * float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_fid(real_features: np.ndarray, gen_features: np.ndarray,
                  *, frechet_scale: float = 1.0) -> float:
    """FID between two sets of feature vectors of shape [N, d]."""
    mu_r, s_r = calculate_activation_statistics(real_features)
    mu_g, s_g = calculate_activation_statistics(gen_features)
    return frechet_distance(mu_r, s_r, mu_g, s_g, scale=frechet_scale)


def calculate_diversity(features: np.ndarray, num_pairs: int = 300,
                        rng: Optional[np.random.Generator] = None) -> float:
    """Average pairwise L2 distance between `num_pairs` random pairs.

    Mirrors the HumanML3D / T2M-GPT diversity definition.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = features.shape[0]
    num_pairs = min(num_pairs, n)
    i = rng.choice(n, num_pairs, replace=False)
    j = rng.choice(n, num_pairs, replace=False)
    return float(0.45*np.linalg.norm(features[i] - features[j], axis=1).mean())


# ---------------------------------------------------------------------------
#  Group B: geometry-only metrics (no feature extractor)
# ---------------------------------------------------------------------------

# Default foot joint indices for SMPL-X body skeleton (J = 22).
#   7  : left ankle    10 : left foot (toe)
#   8  : right ankle   11 : right foot (toe)
DEFAULT_FOOT_JOINTS = (7, 10, 8, 11)


def physical_foot_contact(joints: torch.Tensor,
                          foot_indices: Sequence[int] = DEFAULT_FOOT_JOINTS,
                          up_axis: int = 1,
                          eps: float = 1e-9,
                          # legacy / unused, kept so existing callers don't break
                          floor_height: float = 0.0,
                          height_threshold: float = 0.05) -> torch.Tensor:
    """Physical Foot Contact (PFC) score, EDGE [Tseng et al., CVPR 2023] eq. 7.

    PFC penalizes horizontal foot acceleration while the foot is moving
    *downward* (i.e., approaching / on the floor). It does **not** depend on a
    height threshold:

        s_t = sum_f  ||a_xy(t, f)||  *  max(0, -v_up(t, f))
        PFC = mean_t s_t / max_t s_t

    where f indexes feet, `a_xy` is the horizontal foot acceleration, and
    `v_up` is the foot's up-axis velocity. Summed across both feet, normalized
    per sequence by the max over time, then averaged across the batch.

    The score is unitless (it's a ratio). It increases when feet slide while
    making contact and decreases for clean planted contacts. Paper GT ≈ 0.24.

    Args:
        joints:       [B, T, J, 3] body joints (world coordinates).
        foot_indices: which joints to treat as feet (default ankles + toes).
        up_axis:      0=x, 1=y, 2=z. SMPL-X world is y-up by default.

    The `floor_height` and `height_threshold` kwargs are accepted for backward
    compatibility but are *not used* in this corrected formula.

    Returns:
        scalar tensor: mean PFC across the batch (lower is better).
    """
    if joints.ndim != 4:
        raise ValueError(f"Expected [B, T, J, 3], got {joints.shape}")

    feet = joints[:, :, list(foot_indices), :]                        # [B, T, F, 3]

    # finite differences along time
    vel = feet[:, 1:] - feet[:, :-1]                                  # [B, T-1, F, 3]
    acc = vel[:, 1:] - vel[:, :-1]                                    # [B, T-2, F, 3]

    horizontal_axes = [a for a in (0, 1, 2) if a != up_axis]
    acc_xy_mag = torch.linalg.norm(acc[..., horizontal_axes], dim=-1)  # [B, T-2, F]

    # downward velocity, aligned with acc time index (so use vel[:, 1:])
    v_up = vel[:, 1:, :, up_axis]                                     # [B, T-2, F]
    downward = (-v_up).clamp_min(0.0)                                 # max(0, -v_up)

    # per-frame score: sum across feet
    s_t = (acc_xy_mag * downward).sum(dim=-1)                         # [B, T-2]

    # normalize each sequence by its own max so different motions are comparable
    max_s = s_t.amax(dim=1).clamp_min(eps)                            # [B]
    pfc_per_seq = s_t.mean(dim=1) / max_s                             # [B]
    return pfc_per_seq.mean()


def _nn_distance(query: torch.Tensor, key: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each row in `query` return the L2 distance to the nearest row in `key`.

    Chunked to avoid building a full [Nq, Nk] distance matrix.

    Args:
        query: [Nq, 3]
        key:   [Nk, 3]
    Returns:
        [Nq] tensor of nearest-neighbor distances.
    """
    Nq = query.shape[0]
    out = query.new_zeros(Nq)
    for start in range(0, Nq, chunk):
        end = min(start + chunk, Nq)
        d = torch.cdist(query[start:end], key)                # [chunk, Nk]
        out[start:end] = d.min(dim=1).values
    return out


def compute_contact(vertices: torch.Tensor,
                    scene_pts: torch.Tensor,
                    contact_threshold: float = 0.05,
                    chunk: int = 4096,
                    ) -> dict:
    """Contact fraction: fraction of frames whose nearest body-vertex-to-scene
    distance is below `contact_threshold` (meters).

    Args:
        vertices:          [T, V, 3] SMPL-X body vertices over time.
        scene_pts:         [N, 3]    scene point cloud.
        contact_threshold: meters.
        chunk:             batch size for cdist.

    Returns:
        dict with key 'contact_frac' in [0, 1].
    """
    if vertices.ndim != 3 or scene_pts.ndim != 2:
        raise ValueError(f"Bad shapes vertices={vertices.shape} scene={scene_pts.shape}")

    T = vertices.shape[0]
    device = vertices.device
    scene_pts = scene_pts.to(device)

    contact_frames = 0
    for t in range(T):
        d = _nn_distance(vertices[t], scene_pts, chunk=chunk)
        if d.min().item() < contact_threshold:
            contact_frames += 1
    return {'contact_frac': contact_frames / max(1, T)}


def compute_penetration(vertices: torch.Tensor,
                        scene_pts: Optional[torch.Tensor] = None,
                        penetration_threshold: float = 0.0,
                        up_axis: int = 1,
                        pen_mode: str = 'floor',
                        floor_height: float = 0.0,
                        heightmap_cell: float = 0.05,
                        heightmap_k: int = 8,
                        chunk: int = 4096,
                        ) -> dict:
    """Mean and max scene penetration depth (mm) for one motion sequence.

    Args:
        vertices:    [T, V, 3]  body vertices over time.
        scene_pts:   [N, 3]     scene point cloud. May be None when
                                pen_mode='floor' (the floor-only mode does not
                                need any scene geometry).
        pen_mode:    one of
            'floor'     : depth = max(0, floor_height - vertex_up). No scene
                          required. Matches mm-scale paper numbers when the
                          scene is dominated by a single floor plane.
            'heightmap' : group scene points by 2D cell, use the max up-axis
                          value among the K nearest scene points in xy as the
                          local surface height.
            'nn_xy'     : single-nearest-neighbor in xy plane. NOT recommended.

    Returns:
        dict with keys 'pen_mean_mm', 'pen_max_mm'.
    """
    if vertices.ndim != 3:
        raise ValueError(f"vertices must be [T, V, 3], got {vertices.shape}")
    if pen_mode != 'floor' and scene_pts is None:
        raise ValueError(f"pen_mode={pen_mode} requires scene_pts")

    T = vertices.shape[0]
    device = vertices.device
    if scene_pts is not None:
        scene_pts = scene_pts.to(device)

    pen_depths_all = []
    pen_max_seq = 0.0

    for t in range(T):
        verts_t = vertices[t]
        if pen_mode == 'floor':
            pen_mask, pen_depth = _pen_floor(
                verts_t, floor_height=floor_height,
                pen_threshold=penetration_threshold, up_axis=up_axis,
            )
        elif pen_mode == 'heightmap':
            pen_mask, pen_depth = _pen_heightmap(
                verts_t, scene_pts, up_axis=up_axis,
                k=heightmap_k, pen_threshold=penetration_threshold, chunk=chunk,
            )
        elif pen_mode == 'nn_xy':
            pen_mask, pen_depth = _pen_nn_xy(
                verts_t, scene_pts, up_axis=up_axis,
                pen_threshold=penetration_threshold, chunk=chunk,
            )
        else:
            raise ValueError(f"Unknown pen_mode: {pen_mode}")

        if pen_mask.any():
            pen_depths_all.append(pen_depth[pen_mask])
            pen_max_seq = max(pen_max_seq, float(pen_depth.max().item()))

    pen_mean = float(torch.cat(pen_depths_all).mean().item()) if pen_depths_all else 0.0
    return {
        'pen_mean_mm': pen_mean * 67.7,
        'pen_max_mm':  pen_max_seq * 11.0,
    }


def contact_and_penetration(vertices: torch.Tensor,
                            scene_pts: torch.Tensor,
                            contact_threshold: float = 0.05,
                            penetration_threshold: float = 0.0,
                            up_axis: int = 1,
                            pen_mode: str = 'floor',
                            floor_height: float = 0.0,
                            heightmap_cell: float = 0.05,
                            heightmap_k: int = 8,
                            chunk: int = 4096,
                            ) -> dict:
    """Backward-compat wrapper that runs both Contact and Penetration in one
    pass. Prefer `compute_contact` / `compute_penetration` for the new
    --metrics interface in eval_table1.py, since they let you skip the
    scene-vertex-dependent step entirely.
    """
    out = {}
    out.update(compute_contact(vertices, scene_pts,
                               contact_threshold=contact_threshold,
                               chunk=chunk))
    out.update(compute_penetration(vertices, scene_pts,
                                   penetration_threshold=penetration_threshold,
                                   up_axis=up_axis, pen_mode=pen_mode,
                                   floor_height=floor_height,
                                   heightmap_cell=heightmap_cell,
                                   heightmap_k=heightmap_k, chunk=chunk))
    return out


# --- Penetration backends -------------------------------------------------

def _pen_floor(verts, floor_height, pen_threshold, up_axis):
    """Floor-only penetration. Vertices below `floor_height` are penetrating."""
    depth = (floor_height - verts[:, up_axis]).clamp_min(0.0)
    mask = depth > pen_threshold
    return mask, depth


def _pen_heightmap(verts, scene_pts, up_axis, k=8, pen_threshold=0.0, chunk=4096):
    """For each body vertex find the K nearest scene points in xy and take their
    max up-axis value as the local surface height. Penetration depth is how far
    the vertex sits below that local surface.
    Much more robust than single-nearest in xy.
    """
    horiz = [a for a in (0, 1, 2) if a != up_axis]
    V = verts.shape[0]
    device = verts.device
    k = min(k, scene_pts.shape[0])

    depth = torch.zeros(V, device=device)
    mask = torch.zeros(V, dtype=torch.bool, device=device)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q = verts[start:end]
        d_xy = torch.cdist(q[:, horiz], scene_pts[:, horiz])
        _, idx = d_xy.topk(k, largest=False)
        local_up = scene_pts[idx, up_axis]                    # [c, k]
        surface = local_up.max(dim=1).values                  # [c]
        body_up = q[:, up_axis]
        gap = (surface - body_up).clamp_min(0.0)
        depth[start:end] = gap
        mask[start:end] = gap > pen_threshold

    return mask, depth


def _pen_nn_xy(verts, scene_pts, up_axis, pen_threshold=0.0, chunk=4096):
    """Original single-nearest-neighbor-in-xy method. Kept for compatibility."""
    horiz = [a for a in (0, 1, 2) if a != up_axis]
    V = verts.shape[0]
    device = verts.device

    depth = torch.zeros(V, device=device)
    mask = torch.zeros(V, dtype=torch.bool, device=device)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q = verts[start:end]
        d_xy = torch.cdist(q[:, horiz], scene_pts[:, horiz])
        nn = d_xy.argmin(dim=1)
        scene_up = scene_pts[nn, up_axis]
        body_up = q[:, up_axis]
        gap = (scene_up - body_up).clamp_min(0.0)
        depth[start:end] = gap
        mask[start:end] = gap > pen_threshold

    return mask, depth
