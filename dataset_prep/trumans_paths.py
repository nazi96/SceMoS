"""TRUMANS dataset and checkpoint paths for this workspace."""
from __future__ import annotations

import os
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Hardcoded paths (edit here if your layout changes) ---
TRUMANS_DATA_RELEASE = r"C:\Users\anind\SRC\DATASETS\TRUMANS\Data_release"
PREPROCESSED_TRUMANS_DIR = r"C:\Users\anind\SRC\MYPROJECTS\Character-Scene-Interaction\data\trumans"
PREPROCESSED_TRUMANS_PLY_DIR = r"C:\Users\anind\SRC\MYPROJECTS\Character-Scene-Interaction\data\trumans_scene_ply"
CHECKPOINTS_TRUMANS_DIR = r"C:\Users\anind\SRC\MYPROJECTS\Character-Scene-Interaction\checkpoints\trumans"
SMPLX_MODEL_DIR = r"C:\Users\anind\SRC\MYPROJECTS\Character-Scene-Interaction\models\smplx"
# smplx.create() expects the parent folder; SMPLX() expects the smplx/ folder directly.
SMPLX_BODY_MODELS_DIR = os.path.dirname(SMPLX_MODEL_DIR)

VQ_WEIGHTS_PATH = os.path.join(
    CHECKPOINTS_TRUMANS_DIR,
    "exp_11_VQVAE_decoder_heightmap_contact_256_80",
    "11360",
    "weights.p",
)
AR_WEIGHTS_PATH = os.path.join(
    CHECKPOINTS_TRUMANS_DIR,
    "exp_84_AutoregressiveMotionGenerator_32_80",
    "latest",
    "weights.p",
)
REFINEMENT_WEIGHTS_PATH = os.path.join(
    CHECKPOINTS_TRUMANS_DIR,
    "exp_87_TrajectoryTrainer_128_80",
    "latest",
    "weights.p",
)

TEST_SAMPLES_PATH = os.path.join(PREPROCESSED_TRUMANS_DIR, "validation_samples.npy")
TRAIN_SAMPLES_PATH = os.path.join(PREPROCESSED_TRUMANS_DIR, "train_samples.npy")
CSI_ROOT = os.path.dirname(os.path.dirname(PREPROCESSED_TRUMANS_DIR))


def repo_root() -> str:
    return _REPO_ROOT


def character_scene_interaction_root() -> str:
    return CSI_ROOT


def resolve_trumans_data_root(data_root: Optional[str] = None) -> str:
    if data_root:
        return os.path.abspath(data_root)
    env_root = os.environ.get("TRUMANS_DATA_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    return TRUMANS_DATA_RELEASE


def preprocessed_trumans_dir() -> str:
    return PREPROCESSED_TRUMANS_DIR


def preprocessed_trumans_ply_dir() -> str:
    return PREPROCESSED_TRUMANS_PLY_DIR


def checkpoints_trumans_dir() -> str:
    return CHECKPOINTS_TRUMANS_DIR


def train_samples_path() -> str:
    return TRAIN_SAMPLES_PATH


def test_samples_path() -> str:
    test_path = os.path.join(PREPROCESSED_TRUMANS_DIR, "test_samples.npy")
    if os.path.isfile(test_path):
        return test_path
    return TEST_SAMPLES_PATH


def resolve_motion_data_path(path: str) -> str:
    """Resolve motion pickle paths stored relative to Character-Scene-Interaction."""
    path = str(path)
    if os.path.isabs(path) and os.path.exists(path):
        return os.path.abspath(path)

    candidates = [
        os.path.join(PREPROCESSED_TRUMANS_DIR, os.path.basename(path)),
        os.path.join(CSI_ROOT, path.replace("/", os.sep)),
        os.path.join(_REPO_ROOT, path.replace("/", os.sep)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def default_vq_weights_path() -> str:
    return VQ_WEIGHTS_PATH


def default_ar_weights_path() -> str:
    return AR_WEIGHTS_PATH


def default_refinement_weights_path() -> Optional[str]:
    return REFINEMENT_WEIGHTS_PATH
