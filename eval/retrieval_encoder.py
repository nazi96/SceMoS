"""Motion / text retrieval encoder for TRUMANS — HumanML3D-style.

This is the same family of evaluator network used by MDM (Tevet et al.,
ICLR'23) and IMoS (Ghosh et al., CGF'23). The TRUMANS paper Table 3 says
"The definition of 'Real' follows the one defined in Tevet et al.", which
means the FID / Diversity numbers reported in TRUMANS and SceMoS are
computed in the 512-dim feature space of this encoder.

Architecture (compact, fast to train):
    motion : Linear -> BiGRU -> mean-pool -> Linear -> 512-d feature
    text   : T5 (frozen) -> mean-pool -> Linear -> 512-d feature

Trained with symmetric InfoNCE so motion features and matching text features
end up in the same metric space. FID / Diversity are computed on the motion
features only.

Usage at eval time:
    from eval.retrieval_encoder import RetrievalMotionEncoder
    enc = RetrievalMotionEncoder.load(path, device)
    feats = enc(motion[B, T, D])           # -> [B, 512]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    """BiGRU-based motion encoder. Maps [B, T, D] -> [B, out_dim] feature."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout,
            bidirectional=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: [B, T, D]
        x = self.in_proj(motion)
        y, _ = self.gru(x)               # [B, T, 2H]
        y = y.mean(dim=1)                # mean-pool over time
        return self.out_proj(y)          # [B, out_dim]


class TextEncoder(nn.Module):
    """Wraps a frozen T5 encoder + a small projection head."""

    def __init__(self, t5_model_name: str = 'google/flan-t5-base', out_dim: int = 512):
        super().__init__()
        from transformers import T5Tokenizer, T5EncoderModel
        self.tok = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5 = T5EncoderModel.from_pretrained(t5_model_name)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.t5.eval()
        hidden = self.t5.config.d_model
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.device = next(self.t5.parameters()).device

    @torch.no_grad()
    def _encode_t5(self, texts: List[str]) -> torch.Tensor:
        device = next(self.proj.parameters()).device
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=64, return_tensors='pt').to(device)
        out = self.t5(**enc).last_hidden_state            # [B, L, H]
        mask = enc.attention_mask.unsqueeze(-1).float()   # [B, L, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled                                     # [B, H]

    def forward(self, texts: List[str]) -> torch.Tensor:
        pooled = self._encode_t5(texts)
        return self.proj(pooled)


class RetrievalMotionEncoder(nn.Module):
    """Convenience wrapper exposing only motion -> feature for eval time."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.motion = MotionEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                    out_dim=out_dim, num_layers=num_layers,
                                    dropout=dropout)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return self.motion(motion)

    @classmethod
    def load(cls, weights_path: str, device: torch.device) -> 'RetrievalMotionEncoder':
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        cfg = ckpt['config']
        model = cls(**{k: cfg[k] for k in ('input_dim', 'hidden_dim', 'out_dim',
                                            'num_layers', 'dropout')}).to(device)
        model.motion.load_state_dict(ckpt['motion'])
        model.eval()
        return model


# ---------------------------------------------------------------------------
#  Loss
# ---------------------------------------------------------------------------

def info_nce(motion_feats: torch.Tensor, text_feats: torch.Tensor,
             temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE. Inputs are [B, d]."""
    m = F.normalize(motion_feats, dim=-1)
    t = F.normalize(text_feats, dim=-1)
    logits = (m @ t.t()) / temperature                    # [B, B]
    labels = torch.arange(m.shape[0], device=m.device)
    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_m2t + loss_t2m)


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

def train(opt):
    from torch.utils.data import DataLoader
    from utils.utilities import fixseed
    from dataset_prep.trumans_loader import TrumansDataset
    from dataset_prep.motion_process import MOTION_FEATS_BODY_ONLY
    from eval.eval_quant import collate_with_var_lengths

    fixseed(opt.seed)
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ---- Data ----
    train_ds = TrumansDataset(phase='all', window_size=opt.window_size,
                              predict_velocity=True, downsample_rate=1, device=device,
                              load_dino_feats=False, load_scene_vertex=False)
    val_ds   = TrumansDataset(phase='test', window_size=opt.window_size,
                              predict_velocity=True, downsample_rate=1, device=device,
                              load_dino_feats=False, load_scene_vertex=False)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              drop_last=True, num_workers=opt.num_workers,
                              collate_fn=collate_with_var_lengths)
    val_loader   = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                              drop_last=True, num_workers=opt.num_workers,
                              collate_fn=collate_with_var_lengths)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Models ----
    input_dim = MOTION_FEATS_BODY_ONLY
    motion_enc = MotionEncoder(input_dim=input_dim, hidden_dim=opt.hidden_dim,
                               out_dim=opt.out_dim, num_layers=opt.num_layers,
                               dropout=opt.dropout).to(device)
    text_enc = TextEncoder(t5_model_name=opt.t5_model_name, out_dim=opt.out_dim).to(device)
    params = list(motion_enc.parameters()) + [p for p in text_enc.proj.parameters()]
    print(f"Trainable params: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: min(1.0, (it + 1) / max(1, opt.warmup_iters))
    )

    os.makedirs(opt.output_dir, exist_ok=True)
    weights_path = os.path.join(opt.output_dir, 'retrieval_encoder.pt')
    log_path = os.path.join(opt.output_dir, 'train_log.txt')

    def _save():
        torch.save({
            'motion': motion_enc.state_dict(),
            'text_proj': text_enc.proj.state_dict(),
            'config': {
                'input_dim': input_dim,
                'hidden_dim': opt.hidden_dim,
                'out_dim': opt.out_dim,
                'num_layers': opt.num_layers,
                'dropout': opt.dropout,
                't5_model_name': opt.t5_model_name,
                'window_size': opt.window_size,
            },
        }, weights_path)

    @torch.no_grad()
    def _val():
        motion_enc.eval(); text_enc.eval()
        total, correct1, correct5 = 0, 0, 0
        for batch in val_loader:
            motion = batch['motion_features'].to(device)
            texts = list(batch['texts'])
            mf = F.normalize(motion_enc(motion), dim=-1)
            tf = F.normalize(text_enc(texts),    dim=-1)
            sim = mf @ tf.t()
            labels = torch.arange(mf.shape[0], device=device)
            top5 = sim.topk(min(5, sim.shape[1]), dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += mf.shape[0]
        motion_enc.train()
        return correct1 / max(1, total), correct5 / max(1, total)

    it = 0
    best_r5 = -1.0
    for epoch in range(opt.num_epoch):
        motion_enc.train(); text_enc.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            motion = batch['motion_features'].to(device)
            texts = list(batch['texts'])
            mf = motion_enc(motion)
            tf = text_enc(texts)
            loss = info_nce(mf, tf, temperature=opt.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            it += 1
            if it % opt.log_every == 0:
                print(f"  it {it:>6d}  loss {loss.item():.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")

        r1, r5 = _val()
        dt = time.time() - t0
        msg = (f"epoch {epoch:>3d}  train_loss {sum(losses)/max(1,len(losses)):.4f}  "
               f"R@1 {r1:.3f}  R@5 {r5:.3f}  time {dt:.1f}s")
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

        if r5 > best_r5:
            best_r5 = r5
            _save()
            print(f"  saved best (R@5={best_r5:.3f}) -> {weights_path}")

    print("Done.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--output_dir', type=str, default='checkpoints/trumans/retrieval_encoder')
    ap.add_argument('--window_size', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--num_epoch', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_iters', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--out_dim', type=int, default=512)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--t5_model_name', type=str, default='google/flan-t5-base')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--gpu_id', type=int, default=0)
    opt = ap.parse_args()
    train(opt)


if __name__ == '__main__':
    main()
