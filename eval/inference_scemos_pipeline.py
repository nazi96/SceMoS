from __future__ import annotations
"""End-to-end SceMoS inference entrypoint (AR + VQ + optional refinement).

This script wires checkpoint loading, dataset iteration, motion decoding, and
optional PKL export for visualization/evaluation.
"""

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from dataset_prep.motion_process import (
    MOTION_FEATS_BODY_ONLY,
    delta_mot2mot_feats,
    mot_to_smplx_verts,
)
from dataset_prep.heightmap_utils import (
    causal_geom_index,
    compute_geometry_from_motion,
    get_scene_vertices_from_batch,
)
from dataset_prep.trumans_paths import (
    default_ar_weights_path,
    default_refinement_weights_path,
    default_vq_weights_path,
)
from dataset_prep.trumans_loader import TrumansDataset
from models.autoregressive_motion_generator import AutoregressiveMotionGenerator
import models.vqvae as vqvae_mod
from models.vqvae import Global_Trajectory_Pred
from utils.utilities import fixseed, freeze_model, get_opt, makepath


def collate_with_var_lengths(batch, var_keys=("scene_vertex",)):
    """Collate dict batches while keeping variable-length keys as Python lists."""
    if not isinstance(batch[0], dict):
        return default_collate(batch)
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        elems = [b[k] for b in batch]
        if k in var_keys:
            out[k] = elems
            continue
        try:
            out[k] = default_collate(elems)
        except RuntimeError:
            out[k] = elems
    return out


def build_heightmap_contact_cond(
    batch: dict,
    device: torch.device,
    *,
    use_gt: bool = True,
    inv_z=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build causal heightmap + contact conditioning for VQ decode.

    If ``use_gt`` is False, geometry is computed from ``mot_init`` and static
    ``scene_vertex`` only (bootstrap frame 0). Use ``ar_vq_decode_delta_online``
    for full rolling geometry during AR inference.
    """
    if use_gt:
        heightmap = batch["heightmap"][:, ::4].to(device)
        contact_maps = batch["contact_map"][:, ::4].to(device)
    else:
        if inv_z is None:
            raise ValueError("inv_z required when use_gt=False")
        mot_init = batch["mot_init"].to(device)
        betas = batch["betas"]
        bsz = mot_init.shape[0]
        hm_list = []
        cm_list = []
        for b in range(bsz):
            scene = get_scene_vertices_from_batch(batch, b, device)
            mot_u = inv_z(mot_init[b])
            hm, cm = compute_geometry_from_motion(mot_u, betas[b], scene, 0)
            hm_list.append(hm)
            cm_list.append(cm)
        heightmap = torch.stack(hm_list, dim=0).unsqueeze(1)
        contact_maps = torch.stack(cm_list, dim=0).unsqueeze(1)

    heightmap_cond = torch.cat((heightmap[:, 0:1], heightmap[:, :-1]), dim=1)
    contact_maps_cond = torch.cat((contact_maps[:, 0:1], contact_maps[:, :-1]), dim=1)
    return heightmap_cond, contact_maps_cond


def load_vq(vq_weights: str, device: torch.device) -> Tuple[nn.Module, Any]:
    vq_weights = os.path.abspath(vq_weights)
    if not os.path.isfile(vq_weights):
        raise FileNotFoundError(
            f"VQ checkpoint not found: {vq_weights!r}. "
            "Pass --vq_weights or place weights under checkpoints/trumans/."
        )
    opt_path = os.path.join(os.path.dirname(os.path.dirname(vq_weights)), "opt.txt")
    if not os.path.isfile(opt_path):
        raise FileNotFoundError(
            f"Missing VQ opt.txt at {opt_path!r} (expected in the experiment folder "
            f"two levels above {vq_weights!r})."
        )
    vq_opt = get_opt(opt_path)
    vq_opt.gpu_id = [device.index if device.type == "cuda" else -1]
    gpu_id = vq_opt.gpu_id[0] if isinstance(vq_opt.gpu_id, list) else vq_opt.gpu_id

    if not hasattr(vqvae_mod, vq_opt.model_name):
        raise AttributeError(
            f"Unknown VQ model_name {vq_opt.model_name!r} — add it to models/vqvae.py "
            "or fix opt.txt."
        )
    model_cls = getattr(vqvae_mod, vq_opt.model_name)
    vq_model = model_cls(
        input_feats=MOTION_FEATS_BODY_ONLY,
        output_feats=MOTION_FEATS_BODY_ONLY,
        quantizer=vq_opt.quantizer,
        code_num=vq_opt.code_num,
        code_dim=vq_opt.code_dim,
        output_emb_width=vq_opt.output_emb_width,
        down_t=vq_opt.down_t,
        stride_t=vq_opt.stride_t,
        width=vq_opt.width,
        depth=vq_opt.depth,
        dilation_growth_rate=vq_opt.dilation_growth_rate,
        norm=vq_opt.vq_norm,
        activation=vq_opt.vq_act,
        gpu_id=gpu_id,
        scene_dim=vq_opt.scene_dim,
        condition_emb_dim=vq_opt.condition_emb_dim,
        n_heads=vq_opt.n_heads,
        num_quantizers=vq_opt.num_quantizers,
        shared_codebook=vq_opt.shared_codebook,
        quantize_dropout_prob=vq_opt.quantize_dropout_prob,
    ).to(device)
    ckpt = torch.load(vq_weights, map_location=device, weights_only=True)
    vq_model.load_state_dict(ckpt["vq_model"], strict=True)
    vq_model = freeze_model(vq_model).eval()
    return vq_model, vq_opt


def load_ar(ar_weights: str, device: torch.device) -> nn.Module:
    opt_path = os.path.join(os.path.dirname(os.path.dirname(ar_weights)), "opt.txt")
    with open(opt_path) as f:
        ar_opt = json.load(f)
    model = AutoregressiveMotionGenerator(
        motion_vocab_size=ar_opt.get("motion_vocab_size", 1024),
        dino_feature_dim=ar_opt.get("dino_feature_dim", 768),
        text_feature_dim=ar_opt.get("text_feature_dim", 1024),
        hidden_dim=ar_opt.get("hidden_dim", 512),
        num_layers=ar_opt.get("num_layers", 8),
        num_heads=ar_opt.get("num_heads", 8),
        max_seq_len=ar_opt.get("max_seq_len", 64),
        dropout=0.0,
        t5_model_name=ar_opt.get("t5_model_name", "google/flan-t5-large"),
        freeze_t5=True,
        ff_mult=ar_opt.get("ff_mult", 4),
    ).to(device).eval()
    ckpt = torch.load(ar_weights, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_refinement(refinement_weights: str, device: torch.device, gpu_id: int) -> nn.Module:
    opt_path = os.path.join(os.path.dirname(os.path.dirname(refinement_weights)), "opt.txt")
    ropt = get_opt(opt_path)
    net = Global_Trajectory_Pred(
        input_feats=MOTION_FEATS_BODY_ONLY - 3,
        output_feats=3,
        output_emb_width=ropt.output_emb_width,
        down_t=ropt.down_t,
        stride_t=ropt.stride_t,
        width=ropt.width,
        depth=ropt.depth,
        dilation_growth_rate=ropt.dilation_growth_rate,
        norm="LN",
        activation="relu",
        gpu_id=gpu_id,
    ).to(device)
    ckpt = torch.load(refinement_weights, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"], strict=True)
    return net.eval()


@torch.no_grad()
def ar_vq_decode_delta_online(
    ar_model: nn.Module,
    vq_model: nn.Module,
    batch: dict,
    device: torch.device,
    inv_z,
    target_token_length: int,
    greedy: bool,
) -> torch.Tensor:
    """AR token generation + VQ decode with geometry recomputed from predicted motion."""
    text_prompts = batch["texts"]
    dino_feats = batch["dino_feats"].to(device).squeeze(1)
    memory, memory_kpm = ar_model.build_memory(dino_feats, text_prompts)
    bsz = memory.shape[0]
    mot_init = batch["mot_init"].to(device)
    betas = batch["betas"]

    if greedy:
        gen = torch.full(
            (bsz, 1), ar_model.start_token_id, dtype=torch.long, device=device
        )
        for _ in range(target_token_length):
            logits = ar_model._decode(gen, memory, memory_kpm)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        motion_tokens = gen[:, 1:]
    else:
        motion_tokens = ar_model.generate_motion_tokens(
            memory, memory_kpm, target_length=target_token_length
        )

    heightmap_bank: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
    contact_bank: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
    for b in range(bsz):
        scene = get_scene_vertices_from_batch(batch, b, device)
        mot_u = inv_z(mot_init[b])
        hm0, cm0 = compute_geometry_from_motion(mot_u, betas[b], scene, 0)
        heightmap_bank[b].append(hm0)
        contact_bank[b].append(cm0)

    delta_chunks = []
    for t in range(target_token_length):
        geom_idx = causal_geom_index(t)
        hm_cond = torch.stack(
            [heightmap_bank[b][geom_idx] for b in range(bsz)], dim=0
        ).unsqueeze(1)
        cm_cond = torch.stack(
            [contact_bank[b][geom_idx] for b in range(bsz)], dim=0
        ).unsqueeze(1)
        token_t = motion_tokens[:, t : t + 1]
        delta_t = vq_model.motion_decode(token_t, hm_cond, cm_cond)
        delta_chunks.append(delta_t)

        next_geom = t + 1
        if next_geom < target_token_length:
            pred_norm = delta_mot2mot_feats(mot_init, torch.cat(delta_chunks, dim=1))
            for b in range(bsz):
                if len(heightmap_bank[b]) <= next_geom:
                    scene = get_scene_vertices_from_batch(batch, b, device)
                    mot_u = inv_z(pred_norm[b])
                    hm, cm = compute_geometry_from_motion(
                        mot_u, betas[b], scene, next_geom
                    )
                    heightmap_bank[b].append(hm)
                    contact_bank[b].append(cm)

    return torch.cat(delta_chunks, dim=1)


@torch.no_grad()
def ar_vq_decode_delta(
    ar_model: nn.Module,
    vq_model: nn.Module,
    batch: dict,
    device: torch.device,
    heightmap_cond: torch.Tensor,
    contact_maps_cond: torch.Tensor,
    target_token_length: int,
    greedy: bool,
) -> torch.Tensor:
    text_prompts = batch["texts"]
    dino_feats = batch["dino_feats"].to(device).squeeze(1)
    memory, memory_kpm = ar_model.build_memory(dino_feats, text_prompts)

    if greedy:
        bsz = memory.shape[0]
        gen = torch.full(
            (bsz, 1), ar_model.start_token_id, dtype=torch.long, device=device
        )
        for _ in range(target_token_length):
            logits = ar_model._decode(gen, memory, memory_kpm)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        motion_tokens = gen[:, 1:]
    else:
        motion_tokens = ar_model.generate_motion_tokens(
            memory, memory_kpm, target_length=target_token_length
        )

    return vq_model.motion_decode(motion_tokens, heightmap_cond, contact_maps_cond)


@torch.no_grad()
def vq_direct_delta(
    vq_model: nn.Module,
    batch: dict,
    device: torch.device,
    heightmap_cond: torch.Tensor,
    contact_maps_cond: torch.Tensor,
) -> torch.Tensor:
    input_motion = batch["normalized_mot_feats"].to(device)
    pred_delta, *_ = vq_model(input_motion, heightmap_cond, contact_maps_cond)
    return pred_delta


@torch.no_grad()
def pipeline_to_motion(
    batch: dict,
    device: torch.device,
    inv_z,
    vq_model: nn.Module,
    ar_model: Optional[nn.Module],
    refinement: Optional[nn.Module],
    *,
    mode: str,
    target_token_length: int,
    greedy: bool,
    online_geometry: bool = False,
) -> torch.Tensor:
    """Return un-normalized motion [B, T, D]."""
    use_gt_geom = not online_geometry

    if mode == "ar":
        assert ar_model is not None
        if online_geometry:
            delta = ar_vq_decode_delta_online(
                ar_model,
                vq_model,
                batch,
                device,
                inv_z,
                target_token_length,
                greedy,
            )
        else:
            heightmap_cond, contact_maps_cond = build_heightmap_contact_cond(
                batch, device, use_gt=True
            )
            delta = ar_vq_decode_delta(
                ar_model,
                vq_model,
                batch,
                device,
                heightmap_cond,
                contact_maps_cond,
                target_token_length,
                greedy,
            )
    elif mode == "vq_direct":
        # Teacher-forced VQ path: geometry from dataset pickles (GT motion context).
        heightmap_cond, contact_maps_cond = build_heightmap_contact_cond(
            batch, device, use_gt=True
        )
        delta = vq_direct_delta(vq_model, batch, device, heightmap_cond, contact_maps_cond)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    local = delta[..., 3:]
    if refinement is not None:
        root = refinement(local)
    else:
        root = delta[..., :3]

    full_delta = torch.cat([root, local], dim=-1)
    mot_init = batch["mot_init"].to(device)
    pred_norm = delta_mot2mot_feats(mot_init, full_delta)
    return inv_z(pred_norm)


def _scene_vertex_numpy_for_pkl(scene_vertex_batch) -> Optional[np.ndarray]:
    """Scene cloud for PKL / visualize_data / eval_table1.

    aitviewer ``PointClouds`` expects ``points`` shaped [B, N, 3]. ``eval_table1``
    squeezes a leading singleton batch dim to [N, 3] for geometry metrics.
    """
    if scene_vertex_batch is None:
        return None
    if isinstance(scene_vertex_batch, list):
        arr = scene_vertex_batch[0].detach().cpu().numpy()
    else:
        arr = scene_vertex_batch[0].detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32, order="C")
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return np.ascontiguousarray(arr.reshape(1, -1, 3))
    if arr.ndim == 3:
        return np.ascontiguousarray(arr)
    raise ValueError(f"scene_vertex: expected (N,3) or (B,N,3), got {arr.shape}")


def save_visualization_pkl(
    batch: dict,
    pred_motion: torch.Tensor,
    device: torch.device,
    out_dir: str,
    phase: str,
) -> None:
    """Same artifact shape as `TrajectoryTrainer.batch_visualize_predicted` (first sample).

    ``scene_vertices`` is float32 with shape **(1, N, 3)** when a static scene cloud is
    present (aitviewer ``PointClouds`` contract). ``eval_table1`` squeezes a leading
    singleton dimension to **[N, 3]** for contact / penetration.
    """
    unnormalized_gt = batch["motion_features"].to(device)
    unnormalized_pred = pred_motion
    gt_data, gt_vertices, gt_faces, _ = mot_to_smplx_verts(
        unnormalized_gt[0], betas=batch["betas"][0], interpolate_by=2
    )
    pred_data, pred_vertices, pred_faces, _ = mot_to_smplx_verts(
        unnormalized_pred[0], betas=batch["betas"][0], interpolate_by=2
    )
    scene_np = None
    if batch.get("scene_vertex") is not None:
        scene_np = _scene_vertex_numpy_for_pkl(batch["scene_vertex"])

    output_dict = {
        "scene_name": batch["scene_name"][0],
        "gt_data": gt_data,
        "pred_data": pred_data,
        "gt_vertices": gt_vertices,
        "gt_faces": gt_faces,
        "pred_vertices": pred_vertices,
        "pred_faces": pred_faces,
        "scene_vertices": scene_np,
        "local_vertices": batch["local_vertices"][0].detach().cpu().numpy(),
        "texts": batch["texts"],
        # For eval_table1.py --pkl_root (eval without re-running AR/VQ):
        "gt_motion": unnormalized_gt[0].detach().cpu().numpy().astype(np.float32),
        "pred_motion": unnormalized_pred[0].detach().cpu().numpy().astype(np.float32),
        "betas": batch["betas"][0].detach().cpu().numpy().astype(np.float32),
    }
    base = os.path.basename(batch["sample_name"][0])[:-4]
    name = f"{base}_{batch['start_seq'][0].item()}_{batch['end_seq'][0].item()}_.pkl"
    pkl_path = makepath(os.path.join(out_dir, phase, name), isfile=True)
    with open(pkl_path, "wb") as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    p = argparse.ArgumentParser(description="SceMoS-style inference (AR + VQ + optional R).")
    p.add_argument(
        "--vq_weights",
        type=str,
        default=default_vq_weights_path(),
    )
    p.add_argument(
        "--ar_weights",
        type=str,
        default=default_ar_weights_path(),
        help="Set empty string to use --mode vq_direct only.",
    )
    p.add_argument(
        "--refinement_weights",
        type=str,
        default=default_refinement_weights_path(),
        help="Path to TrajectoryTrainer / Global_Trajectory_Pred checkpoint (weights.p). "
             "If omitted, AR+VQ root channel is kept as decoded.",
    )
    p.add_argument("--mode", type=str, choices=["ar", "vq_direct"], default="ar")
    p.add_argument(
        "--phase", type=str, default="test",
        choices=["test", "train", "all"],
    )
    p.add_argument("--window_size", type=int, default=80)
    p.add_argument("--downsample_rate", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=-1, help="-1 = full split")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--target_token_length", type=int, default=20)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample AR tokens with multinomial decoding; default is greedy (argmax).",
    )
    p.add_argument(
        "--online_geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute heightmap/contact from predicted motion + scene. "
             "Default uses GT geometry from preprocessed pickles.",
    )
    p.add_argument("--load_scene_vertex", action="store_true",
                   help="Load scene point clouds (required for --online_geometry).")
    p.add_argument("--save_pkl", action="store_true",
                   help="Write per-sample PKL for visualize_data (first in batch).")
    p.add_argument("--output_dir", type=str, default="outputs/scemos_inference")
    return p.parse_args()


def main():
    """Run the configured inference mode over the selected dataset split."""
    args = parse_args()
    fixseed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    print("Resolved paths:")
    print(f"  vq_weights: {os.path.abspath(args.vq_weights)}")
    print(f"  ar_weights: {os.path.abspath(args.ar_weights)}")
    if args.refinement_weights:
        print(f"  refinement_weights: {os.path.abspath(args.refinement_weights)}")
    print(f"  device: {device}")

    # Stage 1: load trained components.
    vq_model, _ = load_vq(args.vq_weights, device)
    ar_model = None
    if args.mode == "ar":
        if not args.ar_weights:
            raise SystemExit("--mode ar requires --ar_weights")
        ar_model = load_ar(args.ar_weights, device)

    refinement = None
    if args.refinement_weights:
        refinement = load_refinement(args.refinement_weights, device, args.gpu_id)

    # Stage 2: build dataset/loader and normalization helpers.
    dataset = TrumansDataset(
        phase=args.phase,
        window_size=args.window_size,
        predict_velocity=True,
        downsample_rate=args.downsample_rate,
        device=device,
        load_dino_feats=True,
        load_scene_vertex=(
            args.online_geometry
            or args.save_pkl
            or args.load_scene_vertex
        ),
    )
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.num_samples)))
        inv_z = dataset.dataset.inv_z_normalization
    else:
        inv_z = dataset.inv_z_normalization

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_var_lengths,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 3: decode each batch and optionally save visualization artifacts.
    for batch in tqdm(loader, desc="infer", ncols=120):
        pred = pipeline_to_motion(
            batch,
            device,
            inv_z,
            vq_model,
            ar_model,
            refinement,
            mode=args.mode,
            target_token_length=args.target_token_length,
            greedy=not args.stochastic,
            online_geometry=args.online_geometry,
        )
        if args.save_pkl:
            save_visualization_pkl(batch, pred, device, args.output_dir, args.phase)

    print("Done.", "PKL under" if args.save_pkl else "Run complete;", args.output_dir)


if __name__ == "__main__":
    main()
