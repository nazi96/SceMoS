
# TRUMANS BEV feature extraction (DINOv2 / DINOv3)

**`specialized_feature_extractor.py`** extracts unpooled patch features from TRUMANS bird’s-eye-view (BEV) recordings. You choose **DINOv2** or **DINOv3** at runtime (CLI flag or interactive prompt). Both write the same artifact layout for `trumans_loader`.

```text
<Data_release>/Recordings/dinov3/specialized_features_output/<scene_uuid>/<motion_name>/<frame>_unpooled.npy
```

Each file is a NumPy array **`(1, N_patches, D)`** — patch count and dimension depend on the backbone (see below).

## Encoder choice

| | **DINOv3** (published SceMoS run) | **DINOv2** |
|---|-----------------------------------|------------|
| Default model | `dinov3-convnext-tiny` | `dinov2_vitb14` |
| Weights | **Required** local `.pth` ([DINOv3 repo](https://github.com/facebookresearch/dinov3)) | Optional — downloads via `torch.hub` if `--weights` omitted |
| Typical shape | `(1, 49, 768)` | `(1, 256, 768)` for ViT-B/14 @ 224×224 |
| Code path | Vendored `dinov3/` package | `torch.hub.load('facebookresearch/dinov2', ...)` |

**768-dim** features come from **ViT-B** and **ConvNeXt-Tiny/Small** variants only. ViT-S/L and larger ConvNeXt models use other dimensions — retrain the AR model if you change `D`.

**Patch count** differs between ConvNeXt-Tiny (49) and ViT backbones (~196–256). SceMoS checkpoints trained on ConvNeXt-Tiny expect `(1, 49, 768)`; switching encoder requires re-extracting **all** BEV frames and retraining.

## Upstream repositories

| Encoder | Repository |
|--------|------------|
| DINOv3 | [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |
| DINOv2 | [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |

## Inputs and outputs

**Input**

```text
Data_release/Recordings/BEV_1/<scene_uuid>/<motion_name>/0551.jpg
```

**Output**

```text
Data_release/Recordings/dinov3/specialized_features_output/<scene_uuid>/<motion_name>/0551_unpooled.npy
```

## How to run

From `Scemos/dino_feats_extraction`:

```bash
pip install torch torchvision numpy pillow tqdm
```

**DINOv3** also needs the vendored package locally: clone or copy [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) into `dino_feats_extraction/dinov3/` (not tracked in this repo).

### Interactive (prompts for DINOv2 vs DINOv3)

```bash
python specialized_feature_extractor.py \
  --input-dir  /path/to/Data_release/Recordings/BEV_1 \
  --output-dir /path/to/Data_release/Recordings/dinov3/specialized_features_output \
  --weights    /path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
```

If `--encoder` is omitted, the script asks:

```text
Select DINO encoder for BEV feature extraction:
  [1] DINOv3  (default — ConvNeXt-Tiny)
  [2] DINOv2  (ViT-B/14 hub, 768-dim patch tokens)
Enter 1 or 2 [1]:
```

### DINOv3 (non-interactive)

```bash
python specialized_feature_extractor.py \
  --encoder dinov3 \
  --input-dir  /path/to/Data_release/Recordings/BEV_1 \
  --output-dir /path/to/Data_release/Recordings/dinov3/specialized_features_output \
  --weights    /path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  --model-name dinov3-convnext-tiny \
  --resize-strategy stretch
```

### DINOv2 (non-interactive, hub weights)

```bash
python specialized_feature_extractor.py \
  --encoder dinov2 \
  --input-dir  /path/to/Data_release/Recordings/BEV_1 \
  --output-dir /path/to/Data_release/Recordings/dinov3/specialized_features_output \
  --model-name dinov2_vitb14 \
  --resize-strategy stretch
```

Optional local DINOv2 checkpoint:

```bash
python specialized_feature_extractor.py \
  --encoder dinov2 \
  --weights /path/to/dinov2_vitb14.pth \
  ...
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--encoder` | `dinov2` or `dinov3` (prompt if omitted) |
| `--model-name` | See `DINOV2_MODELS` / `DINOV3_MODELS` in the script |
| `--weights` | Required for DINOv3; optional for DINOv2 |
| `--resize-strategy` | `stretch` (default), `smart`, `pad`, `crop` |
| `--device` | `auto`, `cuda`, or `cpu` |

**DINOv3 models:** `dinov3-convnext-tiny`, `dinov3-convnext-small`, `dinov3-convnext-base`, `dinov3-convnext-large`, `dinov3-vits16`, `dinov3-vitb16`, `dinov3-vitl16`, `dinov3-vit7b16`

**DINOv2 models:** `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`, `dinov2_vitb14_reg`

## How training uses these features

In `TrumansDataset` (`load_dino_feats=True`), for window start frame `start`:

- Feature index: `(start // 10) * 10 + 1`
- File: `{index:04d}_unpooled.npy` under `specialized_features_output/<scene_name>/<motion_stem>/`

## Project layout

| Path | Role |
|------|------|
| `specialized_feature_extractor.py` | Batch extractor (DINOv2 + DINOv3) |
| `dinov3/dinov3/` | Local vendored [Meta DINOv3](https://github.com/facebookresearch/dinov3) code (install separately) |

## License

The `dinov3/dinov3/` subpackage follows Meta’s DINOv3 license. DINOv2 weights and code are subject to the [DINOv2 repository](https://github.com/facebookresearch/dinov2) license.
