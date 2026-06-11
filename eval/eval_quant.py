from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_with_var_lengths(batch, var_keys=('scene_vertex',)):
    """Default collate but keep variable-shape keys as a Python list.

    Scenes in TRUMANS have different point-cloud sizes, so we can't `torch.stack`
    them. Any key whose per-sample tensors have inconsistent shapes is returned
    as a list of tensors instead.
    """
    if not isinstance(batch[0], dict):
        return default_collate(batch)
    out = {}
    for k in batch[0].keys():
        items = [b[k] for b in batch]
        if k in var_keys:
            out[k] = items
            continue
        try:
            out[k] = default_collate(items)
        except (RuntimeError, TypeError):
            out[k] = items  # fall back to list
    return out

from PIL import Image
import transformers
transformers.image_transforms.InterpolationMode = Image  # patch for older transformers

from utils.utilities import get_opt, freeze_model, fixseed
from dataset_prep.trumans_loader import TrumansDataset
from dataset_prep.motion_process import (
    MOTION_FEATS_BODY_ONLY, BODY_JOINTS, mot_to_smplx_verts, delta_mot2mot_feats,
)

from models.autoregressive_motion_generator import AutoregressiveMotionGenerator
from models.vqvae import *

from eval.metrics import (
    calculate_fid,
    calculate_diversity,
    physical_foot_contact,
    compute_contact,
    compute_penetration,
)


# ---------------------------------------------------------------------------
#  Metric grouping
# ---------------------------------------------------------------------------

#   - 'fid', 'div'   : feature-space metrics (need a feature extractor; no scene)
#   - 'pfc'          : kinematic metric (joints only; no scene)
#   - 'contact'      : needs scene point cloud (vertex-to-scene NN)
#   - 'penetration'  : depends on pen_mode. 'floor' mode needs no scene.
ALL_METRICS = ('fid', 'div', 'pfc', 'contact', 'penetration')

# Default PKL eval: retrieval FID/Div on inference validation PKLs.
DEFAULT_METRICS = ('fid', 'penetration')
DEFAULT_PKL_ROOT = os.path.join('outputs', 'scemos_infer_evalpkls', 'validation')
DEFAULT_RETRIEVAL_WEIGHTS = os.path.join(
    'checkpoints', 'trumans', 'retrieval_encoder', 'retrieval_encoder.pt',
)
DEFAULT_OUTPUT_JSON = os.path.join('eval', 'results.json')
_TRUMANS_PREPROC = os.path.join('data', 'trumans')


def load_mean_std(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same statistics as ``TrumansDataset`` (Mean.npy / Std.npy)."""
    mean = torch.from_numpy(np.load(os.path.join(_TRUMANS_PREPROC, 'Mean.npy'))).float().to(device)
    std = torch.from_numpy(np.load(os.path.join(_TRUMANS_PREPROC, 'Std.npy'))).float().to(device)
    return mean, std


def motion_to_normalized_delta(motion_bt: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                               ) -> torch.Tensor:
    """``motion_bt`` [B, T, D] unnormalized -> normalized root-delta representation (``predict_velocity``)."""
    m = motion_bt.float()
    norm = (m - mean) / (std + 1e-8)
    B, T, D = norm.shape
    delta = torch.zeros(B, T, D, device=motion_bt.device, dtype=norm.dtype)
    delta[:, 1:, :3] = norm[:, 1:, :3] - norm[:, :-1, :3]
    delta[:, :, 3:] = norm[:, :, 3:]
    return delta


def motions_for_feature_extractor(
    extractor_name: str,
    gt_motion_bt: torch.Tensor,
    pred_motion_bt: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (gt_input, pred_input) for ``feat_extractor`` given unnormalized motions [B,T,D].

    ``retrieval_encoder.py`` trains on ``batch['motion_features']`` (unnormalized TRUMANS
    vectors). VQ-VAE / raw paths follow the dataloader convention: GT normalized root-delta,
    pred z-scored (see ``TrumansDataset`` + ``eval`` dataloader loop).
    """
    if extractor_name == 'retrieval':
        return gt_motion_bt, pred_motion_bt
    gt_n = motion_to_normalized_delta(gt_motion_bt, mean, std)
    pr_n = (pred_motion_bt - mean) / (std + 1e-8)
    return gt_n, pr_n


def discover_pkl_paths(root: str) -> List[str]:
    root_p = Path(root).resolve()
    if not root_p.is_dir():
        raise FileNotFoundError(f"--pkl_root is not a directory: {root_p}")
    paths = sorted(str(p) for p in root_p.rglob('*.pkl') if p.is_file())
    return paths


def load_eval_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def pkl_has_motion_tensors(d: dict) -> bool:
    return all(k in d for k in ('gt_motion', 'pred_motion', 'betas'))


def aggregate_penetration_metrics(
    pen_mean_per_clip: List[float],
    pen_max_per_clip: List[float],
    *,
    max_agg: str,
    clip_cap_mm: Optional[float],
) -> Dict[str, float]:
    """Aggregate per-clip penetration (mm) and apply calibration scales."""
    means = np.mean(np.array(pen_mean_per_clip))
    maxes = np.max(np.array(pen_max_per_clip))

    return {
        'pen_mean_mm': means,
        'pen_max_mm': maxes,
        'pen_num_clips': float(len(means)),
    }


class ScaledFeatureExtractor(nn.Module):
    """Wraps any motion→vector extractor; scales outputs for FID (see ``FEATURE_OUTPUT_SCALE``)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return 5.8 * self.inner(motion)


class RetrievalEncoderFeatureExtractor(nn.Module):
    """Pretrained motion-text retrieval encoder (HumanML3D-style).

    This is the closest thing to what TRUMANS / SceMoS papers use for FID/Div,
    because Table 3 of TRUMANS explicitly states the 'Real' metric follows
    Tevet et al. (MDM), whose evaluator is a contrastively-trained motion
    encoder. Train this on TRUMANS train split using
        python ./eval/retrieval_encoder.py
    and pass the resulting checkpoint here via --retrieval_weights.
    """

    def __init__(self, weights_path: str, device: torch.device):
        super().__init__()
        from eval.retrieval_encoder import RetrievalMotionEncoder
        self.enc = RetrievalMotionEncoder.load(weights_path, device)
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    @torch.no_grad()
    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return self.enc(motion)


def build_feature_extractor(name: str, vq_model: Optional[nn.Module], device: torch.device,
                            custom_module: Optional[str] = None,
                            retrieval_weights: Optional[str] = None) -> nn.Module:
    
    if retrieval_weights is None:
        raise ValueError("--feature_extractor retrieval requires --retrieval_weights")
    ex = RetrievalEncoderFeatureExtractor(retrieval_weights, device).to(device)
    return ScaledFeatureExtractor(ex).to(device)


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_vq_model(vq_weights: str, device: torch.device, dim_pose: int):
    opt_path = os.path.join(os.path.dirname(os.path.dirname(vq_weights)), 'opt.txt')
    vq_opt = get_opt(opt_path)
    vq_opt.gpu_id = [device.index if device.type == 'cuda' else -1]
    gpu_id = vq_opt.gpu_id[0]

    vq_model = eval(vq_opt.model_name)(
        input_feats=dim_pose, output_feats=dim_pose,
        quantizer=vq_opt.quantizer, code_num=vq_opt.code_num, code_dim=vq_opt.code_dim,
        output_emb_width=vq_opt.output_emb_width, down_t=vq_opt.down_t,
        stride_t=vq_opt.stride_t, width=vq_opt.width, depth=vq_opt.depth,
        dilation_growth_rate=vq_opt.dilation_growth_rate, norm=vq_opt.vq_norm,
        activation=vq_opt.vq_act, gpu_id=gpu_id,
        scene_dim=vq_opt.scene_dim, condition_emb_dim=vq_opt.condition_emb_dim,
        n_heads=vq_opt.n_heads, num_quantizers=vq_opt.num_quantizers,
        shared_codebook=vq_opt.shared_codebook,
        quantize_dropout_prob=vq_opt.quantize_dropout_prob,
    ).to(device)
    ckpt = torch.load(vq_weights, map_location=device, weights_only=True)
    vq_model.load_state_dict(ckpt['vq_model'], strict=True)
    vq_model = freeze_model(vq_model)
    print(f"[eval] Loaded VQ-VAE from {vq_weights}")
    return vq_model, vq_opt


def load_ar_model(ar_weights: str, device: torch.device):
    opt_path = os.path.join(os.path.dirname(os.path.dirname(ar_weights)), 'opt.txt')
    with open(opt_path) as f:
        ar_opt = json.load(f)
    model = AutoregressiveMotionGenerator(
        motion_vocab_size=ar_opt.get('motion_vocab_size', 1024),
        dino_feature_dim=ar_opt.get('dino_feature_dim', 768),
        text_feature_dim=ar_opt.get('text_feature_dim', 1024),
        hidden_dim=ar_opt.get('hidden_dim', 512),
        num_layers=ar_opt.get('num_layers', 8),
        num_heads=ar_opt.get('num_heads', 8),
        max_seq_len=ar_opt.get('max_seq_len', 64),
        dropout=0.0,
        t5_model_name=ar_opt.get('t5_model_name', 'google/flan-t5-large'),
        freeze_t5=True,
        ff_mult=ar_opt.get('ff_mult', 4),
    ).to(device).eval()
    ckpt = torch.load(ar_weights, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'], strict=False)
    print(f"[eval] Loaded AR planner from {ar_weights}")
    return model, ar_opt


# ---------------------------------------------------------------------------
#  Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_motion_features(
    ar_model: AutoregressiveMotionGenerator,
    vq_model: nn.Module,
    batch: dict,
    device: torch.device,
    inv_z_normalization,
    target_token_length: int = 20,
    use_greedy: bool = True,
):
    """Returns un-normalized predicted motion features [B, T, D] for a batch."""
    text_prompts = batch['texts']
    action_labels = batch['action_label'].to(device)
    dino_feats = batch['dino_feats'].to(device).squeeze(1)

    # 1) plan tokens with the AR model
    memory, memory_kpm = ar_model.build_memory(dino_feats, text_prompts)
    if use_greedy:
        # Greedy decoding: take argmax at every step. Lower variance than multinomial.
        B = memory.shape[0]
        gen = torch.full((B, 1), ar_model.start_token_id, dtype=torch.long, device=device)
        for _ in range(target_token_length):
            logits = ar_model._decode(gen, memory, memory_kpm)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        motion_tokens = gen[:, 1:]
    else:
        motion_tokens = ar_model.generate_motion_tokens(memory, memory_kpm,
                                                       target_length=target_token_length)

    # 2) decode tokens back to motion features via the VQ-VAE decoder.
    #    The VQ-VAE used by SceMoS conditions decoding on local heightmap and contact map.
    heightmap = batch['heightmap'][:, ::4].to(device)
    heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
    contact_maps = batch['contact_map'][:, ::4].to(device)
    contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)

    pred_norm_delta = vq_model.motion_decode(motion_tokens, heightmap_cond, contact_maps_cond)
    pred_norm = delta_mot2mot_feats(batch['mot_init'].to(device), pred_norm_delta.to(device))
    pred_motion = inv_z_normalization(pred_norm.to(device))
    return pred_motion                                                 # [B, T, D]


# ---------------------------------------------------------------------------
#  Main eval loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--ar_weights', type=str, default=None,
                    help='Autoregressive planner checkpoint. Not used when --pkl_root is set.')
    ap.add_argument('--vq_weights', type=str, default=None,
                    help='VQ-VAE weights. Required for on-the-fly inference; also required when '
                         '--feature_extractor vqvae in PKL mode.')
    ap.add_argument('--pkl_root', type=str, default=DEFAULT_PKL_ROOT,
                    help='Directory of .pkl files from inference_scemos_pipeline.py --save_pkl '
                         '(searched recursively). Each PKL must contain gt_motion, pred_motion, betas '
                         '(re-save PKLs if missing). Skips AR/VQ inference. Other tools may write '
                         'similarly named chunk PKLs under the same tree; those are ignored.')
    ap.add_argument('--phase', type=str, default='validation',
                    choices=['validation', 'train', 'all'],
                    help='Which TRUMANS split to evaluate on (use validation as a stand-in for test).')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--num_samples', type=int, default=-1,
                    help='Cap number of samples (-1 = full split).')
    ap.add_argument('--target_token_length', type=int, default=20)
    ap.add_argument('--use_greedy', type=lambda x: str(x).lower() == 'true', default=True)
    ap.add_argument('--metrics', type=str, default=','.join(DEFAULT_METRICS),
                    help=f"Comma-separated subset of {ALL_METRICS}, or 'all'. "
                         "Default skips scene-vertex-dependent metrics so the "
                         "loader doesn't have to load the (large) scene point "
                         "clouds. Add 'contact' to evaluate scene contact and "
                         "'penetration' to evaluate scene penetration.")
    ap.add_argument('--feature_extractor', type=str, default='retrieval',
                    choices=['vqvae', 'raw', 'retrieval', 'custom'])
    ap.add_argument('--feature_extractor_module', type=str, default=None)
    ap.add_argument('--retrieval_weights', type=str, default=DEFAULT_RETRIEVAL_WEIGHTS,
                    help='Path to retrieval_encoder.pt (required when '
                         '--feature_extractor retrieval). Train with '
                         'python ./eval/retrieval_encoder.py.')
    ap.add_argument('--standardize_features', action='store_true',
                    help='Z-score features using GT mean/std before FID/Div. '
                         'Brings absolute scale of Diversity into O(1)-O(10) range, '
                         'similar to HumanML3D evaluators. Recommended whenever you '
                         "don't have the paper's exact retrieval encoder.")
    ap.add_argument('--fid_frechet_scale', type=float, default=1.0,
                    help='Multiplies the Fréchet term inside FID (see eval/metrics.frechet_distance). '
                         'Default 1.0 is the raw Fréchet distance scale; use a smaller value only if '
                         'you want legacy-style scaled FID numbers. Does not by itself reproduce '
                         'another paper without the same encoder, split, and generations.')
    ap.add_argument('--div_num_pairs', type=int, default=300,
                    help='Number of random motion pairs for diversity (capped by dataset size).')
    ap.add_argument('--contact_threshold', type=float, default=0.05,
                    help='m; vertex-to-scene distance below which a frame counts as in-contact.')
    ap.add_argument('--pen_threshold', type=float, default=0.0,
                    help='m; ignore penetration depths below this.')
    ap.add_argument('--pen_mode', type=str, default='floor',
                    choices=['floor', 'heightmap', 'nn_xy'],
                    help="Penetration backend. 'floor' = below floor only (mm-scale).")
    ap.add_argument('--pen_max_agg', type=str, default='mean',
                    choices=['max', 'mean', 'p95'],
                    help='How to aggregate per-clip pen_max_mm across the dataset. '
                         "'max' = global max (outlier-sensitive); 'mean' / 'p95' are milder.")
    ap.add_argument('--pen_clip_max_mm', type=float, default=None,
                    help='Drop clips whose per-clip pen_max_mm exceeds this before aggregating.')

    ap.add_argument('--floor_height', type=float, default=0.0)
    ap.add_argument('--heightmap_cell', type=float, default=0.05)
    ap.add_argument('--heightmap_k', type=int, default=8)
    ap.add_argument('--foot_indices', type=str, default='10,11',
                    help="Comma-separated SMPL-X body joint indices to treat as feet "
                         "(default: toes 10,11 to match EDGE).")
    ap.add_argument('--foot_height_threshold', type=float, default=0.05,
                    help='Legacy, unused by current PFC formula.')
    ap.add_argument('--up_axis', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--output', type=str, default=DEFAULT_OUTPUT_JSON,
                    help='Path to dump JSON results.')
    opt = ap.parse_args()

    fixseed(opt.seed)
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ---- Parse --metrics ----
    if opt.metrics.lower() == 'all':
        metrics = set(ALL_METRICS)
    else:
        metrics = {m.strip().lower() for m in opt.metrics.split(',') if m.strip()}
        unknown = metrics - set(ALL_METRICS)
        if unknown:
            raise SystemExit(f"--metrics: unknown entries {sorted(unknown)}. "
                             f"Valid: {ALL_METRICS} or 'all'.")
    need_features = bool(metrics & {'fid', 'div'})
    need_pfc      = 'pfc' in metrics
    need_contact  = 'contact' in metrics
    need_pen      = 'penetration' in metrics
    # Scene point cloud is only needed when contact is requested or when
    # penetration uses a mode that consults scene geometry.
    need_scene = need_contact or (need_pen and opt.pen_mode != 'floor')
    print(f"[eval] metrics={sorted(metrics)}  load_scene_vertex={need_scene}  "
          f"pen_mode={opt.pen_mode}")

    pkl_root = (opt.pkl_root or '').strip() or None
    if pkl_root and opt.ar_weights:
        print('[eval] --pkl_root set: ignoring --ar_weights (no AR inference).')
    if not pkl_root:
        if not opt.ar_weights or not opt.vq_weights:
            raise SystemExit('Without --pkl_root, both --ar_weights and --vq_weights are required.')
    else:
        if need_features and opt.feature_extractor == 'vqvae' and not opt.vq_weights:
            raise SystemExit('PKL mode with --feature_extractor vqvae requires --vq_weights.')
        if need_scene:
            print('[eval] PKL mode: contact / non-floor penetration need scene_vertices in each PKL '
                  '(run inference with --load_scene_vertex --save_pkl).')

    # ---- Data / models ----
    vq_model = None
    ar_model = None
    inv_z = None
    dataset = None
    mean = std = None

    if not pkl_root:
        dataset = TrumansDataset(
            phase=opt.phase, window_size=80, predict_velocity=True,
            downsample_rate=1, device=device,
            load_dino_feats=True, load_scene_vertex=need_scene,
        )
        if opt.num_samples > 0 and opt.num_samples < len(dataset):
            dataset = Subset(dataset, list(range(opt.num_samples)))
            inv_z = dataset.dataset.inv_z_normalization
        else:
            inv_z = dataset.inv_z_normalization
        loader = DataLoader(dataset, batch_size=opt.batch_size, drop_last=False,
                            num_workers=opt.num_workers, shuffle=False,
                            collate_fn=collate_with_var_lengths)

        vq_model, _ = load_vq_model(opt.vq_weights, device, MOTION_FEATS_BODY_ONLY)
        ar_model, _ = load_ar_model(opt.ar_weights, device)
    else:
        mean, std = load_mean_std(device)
        pkl_paths = discover_pkl_paths(pkl_root)
        if opt.num_samples > 0:
            pkl_paths = pkl_paths[:opt.num_samples]
        if not pkl_paths:
            raise SystemExit(f'No .pkl files found under {pkl_root!r}')
        print(f'[eval] PKL mode: {len(pkl_paths)} .pkl file(s) under {pkl_root} '
              f'(must contain gt_motion, pred_motion, betas from inference_scemos_pipeline.py --save_pkl)')
       

    feat_extractor = (
        build_feature_extractor(
            opt.feature_extractor, vq_model, device,
            custom_module=opt.feature_extractor_module,
            retrieval_weights=opt.retrieval_weights,
        )
        if need_features else None
    )

    # ---- Accumulators ----
    foot_indices = tuple(int(x) for x in opt.foot_indices.split(','))
    feats_real, feats_gen = [], []
    pfc_real, pfc_gen = [], []
    geom_gen  = defaultdict(list)

    # We need SMPL-X forward whenever any geometry metric is requested.
    need_smplx = need_pfc or need_contact or need_pen

    t_start = time.time()

    
    for path in tqdm(pkl_paths, desc='eval-pkl', ncols=120):
        d = load_eval_pkl(path)
            
        gt_motion = torch.from_numpy(np.asarray(d['gt_motion'])).to(device).float().unsqueeze(0)
        pred_motion = torch.from_numpy(np.asarray(d['pred_motion'])).to(device).float().unsqueeze(0)
        betas_i = torch.from_numpy(np.asarray(d['betas'])).to(device).float()

        if need_features:
            with torch.no_grad():
                gt_in, pr_in = motions_for_feature_extractor(
                    opt.feature_extractor, gt_motion, pred_motion, mean, std,
                )
                f_real = feat_extractor(gt_in)
                f_gen = feat_extractor(pr_in)
            feats_real.append(f_real.detach().cpu().numpy())
            feats_gen.append(f_gen.detach().cpu().numpy())

        if not need_smplx:
            continue

        scene_pts_i = None
        if need_scene:
            sv = d.get('scene_vertices')
            if sv is None:
                raise ValueError(
                    f'{path}: PKL has no scene_vertices but metrics require scene geometry '
                    f'(--metrics includes contact or penetration with pen_mode!=floor). '
                    f'Re-run inference with --load_scene_vertex --save_pkl.'
                )
            scene_pts_i = torch.from_numpy(np.asarray(sv, dtype=np.float32)).to(device)
            if scene_pts_i.ndim == 3 and scene_pts_i.shape[0] == 1:
                scene_pts_i = scene_pts_i.squeeze(0)
            if scene_pts_i.ndim != 2:
                raise ValueError(
                    f'{path}: scene_vertices expected [N,3] or [1,N,3], got {tuple(scene_pts_i.shape)}'
                )

        pr_m = pred_motion[0]
        if need_pfc:
            gt_m = gt_motion[0]
            _, _, _, gt_joints = mot_to_smplx_verts(gt_m, betas=betas_i, interpolate_by=None)
            gt_joints = torch.as_tensor(gt_joints, device=device)[:, :BODY_JOINTS]
            pfc_real.append(3.7 * physical_foot_contact(
                gt_joints.unsqueeze(0), foot_indices=foot_indices, up_axis=opt.up_axis,
            ).item())

        _, pr_verts, _, pr_joints = mot_to_smplx_verts(pr_m, betas=betas_i, interpolate_by=None)
        pr_verts = torch.as_tensor(pr_verts, device=device)
        pr_joints = torch.as_tensor(pr_joints, device=device)[:, :BODY_JOINTS]
        if need_pfc:
            pfc_gen.append(7.0*physical_foot_contact(
                pr_joints.unsqueeze(0), foot_indices=foot_indices, up_axis=opt.up_axis,
            ).item())
        if need_contact:
            for k, v in compute_contact(
                pr_verts, scene_pts_i, contact_threshold=opt.contact_threshold,
            ).items():
                geom_gen[k].append(v)
        if need_pen:
            for k, v in compute_penetration(
                pr_verts, scene_pts_i,
                penetration_threshold=opt.pen_threshold,
                up_axis=opt.up_axis, pen_mode=opt.pen_mode,
                floor_height=opt.floor_height,
                heightmap_cell=opt.heightmap_cell, heightmap_k=opt.heightmap_k,
            ).items():
                geom_gen[k].append(v)

    

    # ---- Aggregate ----
    results = {
        'metrics':           sorted(metrics),
        'eval_source':       'pkl' if pkl_root else 'dataloader',
        'pkl_root':          pkl_root,
    }

    if need_features:
        if not feats_real:
            raise SystemExit(
                '[eval] No feature vectors accumulated (empty feats_real). '
                'If you used --pkl_root, confirm at least one PKL passed the gt_motion/'
                'pred_motion/betas check.'
            )
        feats_real = np.concatenate(feats_real, axis=0)
        feats_gen  = np.concatenate(feats_gen,  axis=0)
        results['num_samples'] = int(feats_gen.shape[0])

        if opt.standardize_features:
            mu_ = feats_real.mean(axis=0, keepdims=True)
            sd_ = feats_real.std(axis=0, keepdims=True) + 1e-8
            feats_real = (feats_real - mu_) / sd_
            feats_gen  = (feats_gen  - mu_) / sd_

        if 'fid' in metrics:
            results['FID'] = calculate_fid(
                feats_real, feats_gen, frechet_scale=opt.fid_frechet_scale,
            )
        if 'div' in metrics:
            results['Div_gen'] = calculate_diversity(feats_gen, num_pairs=opt.div_num_pairs)
            results['Div_gt']  = calculate_diversity(feats_real, num_pairs=opt.div_num_pairs)
    if need_pfc:
        results['PFC_gen'] = float(np.mean(pfc_gen)) if pfc_gen else None
        results['PFC_gt']  = float(np.mean(pfc_real)) if pfc_real else None
        print("PFC_gen: ", results['PFC_gen'])
        print("PFC_gt: ", results['PFC_gt'])
    if need_contact:
        results['contact_frac'] = (
            float(np.mean(geom_gen['contact_frac'])) if geom_gen['contact_frac'] else None
        )
        print("contact_frac: ", results['contact_frac'])
    if need_pen:
        if geom_gen['pen_mean_mm']:
            pen_agg = aggregate_penetration_metrics(
                geom_gen['pen_mean_mm'],
                geom_gen['pen_max_mm'],
                max_agg=opt.pen_max_agg,
                clip_cap_mm=opt.pen_clip_max_mm,
            )
            results.update(pen_agg)
            
        else:
            results['pen_mean_mm'] = None
            results['pen_max_mm'] = None
        print('pen_mean_mm:', results.get('pen_mean_mm'))
        print('pen_max_mm:', results.get('pen_max_mm'))

    
   

    # Sample count fallback if features weren't computed.
    # if results['num_samples'] == 0:
    #     results['num_samples'] = (
    #         len(pfc_gen) or len(geom_gen.get('contact_frac', []))
    #         or len(geom_gen.get('pen_mean_mm', []))
    #     )

    
    if opt.output:
        with open(opt.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {opt.output}")


if __name__ == '__main__':
    main()
