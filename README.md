# SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens

✨ Accepted at CVPR 2026. ✨

[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Ghosh_SceMoS_Scene-Aware_3D_Human_Motion_Synthesis_by_Planning_with_Geometry-Grounded_CVPR_2026_paper.pdf) | [Project Page](https://anindita127.github.io/SceMoS/)

---

## Pre-requisites

We have tested the pipeline on:

* Ubuntu 20.04 LTS / Windows 10–11
* Python >= 3.8
* PyTorch >= 1.11 (CUDA recommended)
* conda >= 4.9 (optional but recommended)

You also need:

* **SMPL-X** body models (register at [SMPL-X](https://smpl-x.is.tue.mpg.de/)); paths are set in `dataset_prep/trumans_paths.py`
* **TRUMANS** [Code / Dataset](https://github.com/jnnan/trumans_utils)
* **[BEV images](https://edmond.mpg.de/file.xhtml?fileId=345209&version=2.0) of TRUMANS scenes**
* **DINOv2 or DINOv3** for BEV scene features — see [`dino_feats_extraction/README.md`](dino_feats_extraction/README.md)

---

## Getting started

Create a conda environment and install dependencies:

```bash
conda create -n scemos python=3.10
conda activate scemos
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy scipy tqdm trimesh pyquaternion natsort scikit-learn smplx transformers einops tensorboard aitviewer scikit-video pillow
```

Clone this repository and run all commands from the **repository root** (`Scemos/`).

---

## Dataset download and layout

1. Request and download the TRUMANS dataset from the official [Google Drive form](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) (see [trumans_utils](https://github.com/jnnan/trumans_utils)).
2. Unzip the release so that the `Data_release` folder contains motion arrays (`.npy`), `Actions/`, `Scene/`, `Scene_data/`, `Object_chairs/`, etc.
3. We save the vertices of the scenes and the objects using a standard mesh to vertex generation function. We save these as `Scene_data/main/*_vertices.npy` and `Object_chairs/Obj_data/main/*_vertices.npy`.
4. Download the BEV images from [here](https://edmond.mpg.de/file.xhtml?fileId=345209&version=2.0) and put them under `Recordings/BEV_1/`. 

### Set paths

Edit **`dataset_prep/trumans_paths.py`** for your machine:

* Raw TRUMANS release (`TRUMANS_DATA_RELEASE`)
* Preprocessed pickles and sample lists (`PREPROCESSED_TRUMANS_DIR`)
* Checkpoints (`CHECKPOINTS_TRUMANS_DIR`) and default VQ / AR / R weights
* SMPL-X models (`SMPLX_MODEL_DIR`)

Alternatively, override only the raw release with an environment variable (used by `trumans_loader` and inference):

```bash
# Linux / macOS
export TRUMANS_DATA_ROOT=/path/to/TRUMANS/Data_release

# Windows (PowerShell)
$env:TRUMANS_DATA_ROOT = "C:\path\to\TRUMANS\Data_release"
```

Preprocessing scripts (`preprocess_trumans.py`, `train_test_split.py`) write to local `data/trumans/` by default; point `trumans_paths.py` at that folder (or your preprocessed release) before training or inference.

If `test_samples.npy` is missing, the loader falls back to `validation_samples.npy` (same test split, legacy filename).

Expected raw layout:

```text
DATASETS/TRUMANS/Data_release/
├── human_pose.npy
├── Actions/
├── Scene/
├── Scene_data/main/
├── Object_chairs/Obj_data/main/
└── Recordings/BEV_1/          # BEV frames for DINO extraction
```

---

## Pipeline

Run the steps below **from the repository root**. Step 1 is slow; use a small subset first when debugging.

| Step | Script | Output |
|------|--------|--------|
| 1 | `dataset_prep/preprocess_trumans.py` | `data/trumans/*_motion_heightmap_contacts.pkl` |
| 2a | `dataset_prep/train_test_split.py` | `train_samples.npy`, `test_samples.npy`, scene splits |
| 2b | `dino_feats_extraction/specialized_feature_extractor.py` | `Recordings/dinov3/specialized_features_output/` |
| 3 | `dataset_prep/calculate_mean_variance.py` | `Mean.npy`, `Std.npy` |
| 4 | `train/train_motionVQVAE.py` | VQ-VAE checkpoints |
| 5 | `train/train_autoregressive_GPT.py` | AR GPT checkpoints |
| 5b | `train/train_trajectory_from_local.py` | Root trajectory refinement **R** |
| 6  | `eval/inference_scemos_pipeline.py` | Predicted motion / PKLs |
| 7  | `eval/eval_quant.py` | Quantitative metrics |
| — | `visualize/visualize_data.py` | View PKLs from step 6 |

### Step 1. Preprocess TRUMANS motions

Build per-sequence pickles with motion features, local scene patches, heightmaps, and contact regions:

```bash
python dataset_prep/preprocess_trumans.py
```

### Step 2a. Train / test split and sample lists

```bash
python dataset_prep/train_test_split.py
```

Writes `Data_release/train_scenes.pkl`, `Data_release/test_scenes.pkl`, and `data/trumans/{train,test}_samples.npy`.

### Step 2b. Extract BEV scene features (DINO)

Required when training with `load_dino_feats=True`. See [`dino_feats_extraction/README.md`](dino_feats_extraction/README.md).

```bash
cd dino_feats_extraction
python specialized_feature_extractor.py \
  --encoder dinov3 \
  --input-dir  /path/to/Data_release/Recordings/BEV_1 \
  --output-dir /path/to/Data_release/Recordings/dinov3/specialized_features_output \
  --weights    /path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
```

Default features: **DINOv3 ConvNeXt-Tiny** → `(1, 49, 768)` per frame.

### Step 3. Motion mean and standard deviation

```bash
python dataset_prep/calculate_mean_variance.py
```

Creates `data/trumans/Mean.npy` and `data/trumans/Std.npy`.

### Step 4. Train VQ-VAE

1. Open `train/train_motionVQVAE.py` and set `is_train = True`.
2. Run:

```bash
python train/train_motionVQVAE.py \
  --model_name VQVAE_decoder_heightmap_contact \
  --batch_size 128 \
  --window_size 80
```

Checkpoints: `checkpoints/trumans/exp_*_VQVAE_decoder_heightmap_contact_*/`.

### Step 5. Train autoregressive GPT

1. Open `train/train_autoregressive_GPT.py` and set `is_train = True`.
2. Set `vq_pretrained_weight_path` to your Step 4 checkpoint.
3. Run:

```bash
python train/train_autoregressive_GPT.py \
  --model_name AutoregressiveMotionGenerator \
  --batch_size 32 \
  --window_size 80
```

### Step 5b (optional). Train trajectory refinement R

Refines the global root trajectory from local body features (used at inference via `--refinement_weights`):

```bash
python train/train_trajectory_from_local.py
```

Configure `vq_pretrained_weight_path` and training flags inside the script before running.

### Step 6. Inference

Runs AR + VQ (+ optional trajectory refinement **R**). Default weights come from `trumans_paths.py`. Geometry (heightmap / contact) is read from preprocessed pickles unless you pass `--online_geometry`.

```bash
python eval/inference_scemos_pipeline.py \
  --phase test \
  --save_pkl \
  --mode vq_direct \
  --output_dir outputs/scemos_infer
```

PKLs are written under `outputs/scemos_infer/test/`. Scene point clouds are loaded automatically when `--save_pkl` is set (for visualization and scene-contact metrics).

Optional flags: `--refinement_weights`, `--stochastic`, `--online_geometry` (requires `--load_scene_vertex`).

### Step 7. Quantitative evaluation

Evaluates PKLs from Step 6 (default metrics: FID + penetration). Results go to `eval/results.json`.

```bash
python eval/eval_quant.py --pkl_root outputs/scemos_infer/test
```

For contact or non-floor penetration metrics, re-run inference with `--load_scene_vertex --save_pkl`. See `python eval/eval_quant.py --help` for `--metrics`, retrieval weights, and on-the-fly modes.

### Visualize PKLs

Edit `load_pkl_path` at the bottom of `visualize/visualize_data.py`, then:

```bash
python visualize/visualize_data.py
```

---

## Repository layout (code)

```text
Scemos/
├── dataset_prep/          # preprocess_trumans, train_test_split, loader, trumans_paths, heightmap_utils
├── dino_feats_extraction/ # README + specialized_feature_extractor.py only (in git)
├── models/                # vqvae, autoregressive_motion_generator, encdec, resnet, quantize_cnn
├── train/                 # train_motionVQVAE, train_autoregressive_GPT, train_trajectory_from_local
├── eval/                  # inference_scemos_pipeline, eval_quant, metrics, retrieval_encoder
├── visualize/             # visualize_data.py (PKL viewer)
├── utils/                 # SMPL-X, rotations, training helpers
├── common/                # quaternion, skeleton
├── data/                  # gitignored — preprocessed TRUMANS (local pipeline output)
└── checkpoints/         # gitignored — trained weights
```

---
## Pretrained Checkpoints
Download all pretrained checkpoints from [here](https://edmond.mpg.de/file.xhtml?fileId=345210&version=2.0)

---

## Citation

```bibtex
@InProceedings{Ghosh_2026_CVPR,
    title={SceMoS: Scene-Aware 3D Human Motion Synthesis by Planning with Geometry-Grounded Tokens},
    author={Ghosh, Anindita and Golyanik, Vladislav and Komura, Taku and Slusallek, Philipp and Theobalt, Christian and Dabral, Rishabh},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2026},
    pages={16443-16453}
}
```

If you use the TRUMANS dataset, please cite:

```bibtex
@inproceedings{jiang2024scaling,
  title={Scaling Up Dynamic Human-Scene Interaction Modeling},
  author={Jiang, Nan and Zhang, Zhiyuan and Li, Hongjie and Ma, Xiaoxuan and Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  booktitle={CVPR},
  year={2024}
}
```

---

## License

Copyright (c) 2026, Max Planck Institute for Informatics. All rights reserved.

TRUMANS data and SMPL-X models are subject to their respective terms from the [TRUMANS](https://github.com/jnnan/trumans_utils) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) projects.
