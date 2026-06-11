#!/usr/bin/env python3
"""
Extract unpooled DINO patch features from TRUMANS BEV recordings.

Supports Meta DINOv3 (vendored under ./dinov3/) and DINOv2 (via torch.hub).
Outputs ``{frame}_unpooled.npy`` arrays shaped ``(1, N_patches, D)`` for each image.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Vendored DINOv3 package lives in ./dinov3/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

DINOV3_MODELS = (
    "dinov3-convnext-tiny",
    "dinov3-convnext-small",
    "dinov3-convnext-base",
    "dinov3-convnext-large",
    "dinov3-vits16",
    "dinov3-vitb16",
    "dinov3-vitl16",
    "dinov3-vit7b16",
)

DINOV2_MODELS = (
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vitb14_reg",
)


@dataclass
class SpecializedFeatureConfig:
    """Configuration for DINO BEV feature extraction."""

    input_dir: str = ""
    output_dir: str = ""
    model_weights_path: str = ""
    encoder: str = "dinov3"  # "dinov3" | "dinov2"
    model_name: str = ""
    feature_types: List[str] = field(default_factory=lambda: ["unpooled"])
    resize_strategy: str = "stretch"
    device: str = "auto"

    def __post_init__(self):
        self.encoder = self.encoder.lower()
        if self.encoder not in {"dinov2", "dinov3"}:
            raise ValueError(f"encoder must be 'dinov2' or 'dinov3', got {self.encoder!r}")
        if not self.model_name:
            self.model_name = default_model_name(self.encoder)
        allowed = DINOV2_MODELS if self.encoder == "dinov2" else DINOV3_MODELS
        if self.model_name not in allowed:
            raise ValueError(
                f"model_name {self.model_name!r} is not valid for {self.encoder}. "
                f"Choose one of: {', '.join(allowed)}"
            )


def default_model_name(encoder: str) -> str:
    return "dinov3-convnext-tiny" if encoder == "dinov3" else "dinov2_vitb14"


def prompt_encoder_choice() -> str:
    """Interactive choice when --encoder is omitted."""
    print("\nSelect DINO encoder for BEV feature extraction:")
    print("  [1] DINOv3  (default — ConvNeXt-Tiny)")
    print("  [2] DINOv2  (ViT-B/14 hub, 768-dim patch tokens)")
    while True:
        choice = input("Enter 1 or 2 [1]: ").strip().lower() or "1"
        if choice in {"1", "dinov3", "v3", "dino3"}:
            return "dinov3"
        if choice in {"2", "dinov2", "v2", "dino2"}:
            return "dinov2"
        print("Invalid choice — please enter 1 or 2.")


def make_imagenet_eval_transform(crop_size: int = 224, resize_size: int = 256):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SpecializedDINOExtractor:
    """DINOv2 / DINOv3 patch feature extractor for TRUMANS BEV frames."""

    def __init__(self, config: SpecializedFeatureConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.transform = None
        self.last_original_size = None
        self.last_resize_method = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup device for feature extraction"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self):
        if self.config.encoder == "dinov3":
            self._load_dinov3_model()
        else:
            self._load_dinov2_model()

    def _load_dinov3_model(self):
        from dinov3.data.transforms import make_eval_transform

        try:
            self.logger.info(f"Loading DINOv3 model: {self.config.model_name}")

            if "convnext" in self.config.model_name:
                from dinov3.hub.backbones import (
                    dinov3_convnext_base,
                    dinov3_convnext_large,
                    dinov3_convnext_small,
                    dinov3_convnext_tiny,
                )

                convnext_models = {
                    "dinov3-convnext-tiny": dinov3_convnext_tiny,
                    "dinov3-convnext-small": dinov3_convnext_small,
                    "dinov3-convnext-base": dinov3_convnext_base,
                    "dinov3-convnext-large": dinov3_convnext_large,
                }
                self.model = convnext_models[self.config.model_name](pretrained=False)
            else:
                from dinov3.hub.backbones import (
                    dinov3_vit7b16,
                    dinov3_vitb16,
                    dinov3_vitl16,
                    dinov3_vits16,
                )

                vit_models = {
                    "dinov3-vits16": dinov3_vits16,
                    "dinov3-vitb16": dinov3_vitb16,
                    "dinov3-vitl16": dinov3_vitl16,
                    "dinov3-vit7b16": dinov3_vit7b16,
                }
                self.model = vit_models[self.config.model_name](pretrained=False)

            if not self.config.model_weights_path:
                raise ValueError(
                    "DINOv3 requires --weights pointing to a local .pth checkpoint."
                )
            if not os.path.exists(self.config.model_weights_path):
                raise FileNotFoundError(
                    f"DINOv3 weights not found: {self.config.model_weights_path}"
                )

            checkpoint = torch.load(
                self.config.model_weights_path, map_location="cpu", weights_only=False
            )
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state, strict=False)
            self.logger.info("Loaded DINOv3 weights from %s", self.config.model_weights_path)

            self.transform = make_eval_transform(
                resize_size=256,
                crop_size=224,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.model = self.model.to(self.device).eval()
            self.logger.info("DINOv3 model ready on %s", self.device)
        except Exception as e:
            self.logger.error("Failed to load DINOv3 model: %s", e)
            raise

    def _load_dinov2_model(self):
        try:
            self.logger.info(f"Loading DINOv2 model: {self.config.model_name}")
            hub_name = self.config.model_name
            use_local = bool(self.config.model_weights_path) and os.path.exists(
                self.config.model_weights_path
            )

            if use_local:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", hub_name, pretrained=False
                )
                checkpoint = torch.load(
                    self.config.model_weights_path, map_location="cpu", weights_only=False
                )
                state = (
                    checkpoint["model"]
                    if isinstance(checkpoint, dict) and "model" in checkpoint
                    else checkpoint
                )
                self.model.load_state_dict(state, strict=False)
                self.logger.info(
                    "Loaded DINOv2 weights from %s", self.config.model_weights_path
                )
            else:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", hub_name, pretrained=True
                )
                self.logger.info(
                    "Loaded DINOv2 %s from torch.hub (pretrained=True)", hub_name
                )

            self.transform = make_imagenet_eval_transform()
            self.model = self.model.to(self.device).eval()
            self.logger.info("DINOv2 model ready on %s", self.device)
        except Exception as e:
            self.logger.error("Failed to load DINOv2 model: %s", e)
            raise

    @staticmethod
    def _patch_tokens_from_forward(features) -> torch.Tensor:
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"]
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            first = next(iter(features.values()))
            if isinstance(first, torch.Tensor):
                return first
            raise ValueError(f"Unexpected feature dict keys: {list(features.keys())}")
        if isinstance(features, torch.Tensor):
            return features
        raise ValueError(f"Unexpected feature type: {type(features)}")
    
    def _smart_resize_image(self, image: Image.Image, target_size: int) -> Image.Image:
        """Smart image resizing that preserves important content"""
        original_size = image.size
        self.last_original_size = original_size
        
        if self.config.resize_strategy == "smart":
            # Smart resizing: preserve aspect ratio but ensure minimum size
            ratio = max(target_size / original_size[0], target_size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            
            # Resize image
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create square canvas with padding
            final_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            
            # Center the resized image
            paste_x = (target_size - new_size[0]) // 2
            paste_y = (target_size - new_size[1]) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            self.last_resize_method = "smart_padding"
            return final_image
            
        elif self.config.resize_strategy == "pad":
            # Simple padding approach
            ratio = min(target_size / original_size[0], target_size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            final_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            
            paste_x = (target_size - new_size[0]) // 2
            paste_y = (target_size - new_size[1]) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            self.last_resize_method = "padding"
            return final_image
            
        elif self.config.resize_strategy == "crop":
            # Center crop approach
            ratio = max(target_size / original_size[0], target_size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Center crop to target size
            left = (new_size[0] - target_size) // 2
            top = (new_size[1] - target_size) // 2
            right = left + target_size
            bottom = top + target_size
            
            final_image = resized_image.crop((left, top, right, bottom))
            self.last_resize_method = "center_crop"
            return final_image
            
        else:  # stretch
            # Simple stretch (original behavior)
            final_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            self.last_resize_method = "stretch"
            return final_image
    
    def extract_specialized_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """Extract multiple types of features optimized for scene analysis"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Get model input size (default to 224 if not available)
            model_img_size = getattr(self.model, 'img_size', 224)
            if isinstance(model_img_size, tuple):
                model_img_size = model_img_size[0]
            
            # Smart resize
            processed_image = self._smart_resize_image(image, model_img_size)
            
            # Apply transforms
            input_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(input_tensor)
                else:
                    features = self.model(input_tensor)

                extracted_features = {}
                if "unpooled" in self.config.feature_types:
                    patch_tokens = self._patch_tokens_from_forward(features)
                    extracted_features["unpooled"] = patch_tokens.cpu().numpy()

                return extracted_features
                
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def process_single_image(self, image_path: str, rel_path: str) -> Dict:
        """Process a single image and extract specialized features"""
        try:
            # Extract features
            features = self.extract_specialized_features(image_path)
            
            if features is None:
                return None
            
            # Prepare metadata
            metadata = {
                'image_path': rel_path,
                'original_size': self.last_original_size,
                'model_input_size': getattr(self.model, 'img_size', 224),
                'patch_size': getattr(self.model, 'patch_size', 16),
                'feature_dimensions': {k: v.shape for k, v in features.items() if k != 'feature_dim'},
                'resizing_info': {
                    'was_resized': self.last_original_size != (getattr(self.model, 'img_size', 224), getattr(self.model, 'img_size', 224)),
                    'aspect_ratio_changed': self.last_original_size[0] != self.last_original_size[1],
                    'resize_method': self.last_resize_method,
                    'original_aspect_ratio': self.last_original_size[0] / self.last_original_size[1],
                    'final_aspect_ratio': 1.0
                },
                'extraction_timestamp': str(torch.cuda.Event() if torch.cuda.is_available() else None)
            }
            
            return {
                'features': features,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed for {image_path}: {e}")
            return None
    
    def save_features(self, features_data: Dict, base_path: str):
        """Save only unpooled features as npy files"""
        try:
            features = features_data['features']
            # metadata = features_data['metadata']  # COMMENTED OUT
            
            # Save only unpooled features
            if 'unpooled' in features:
                feature_path = f"{base_path}_unpooled.npy"
                np.save(feature_path, features['unpooled'])
                self.logger.debug(f"Saved unpooled features to {feature_path}")
            
            # COMMENTED OUT: Save each feature type
            # for feature_type, feature_array in features.items():
            #     if feature_type == 'feature_dim':
            #         continue
            #     
            #     # Save feature array
            #     feature_path = f"{base_path}_{feature_type}.npy"
            #     np.save(feature_path, feature_array)
            #     
            #     # Save metadata
            #     json_path = f"{base_path}.json"
            #     with open(json_path, 'w') as f:
            #         json.dump(metadata, f, indent=2)
            #     
            #     self.logger.debug(f"Saved {feature_type} features to {feature_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save features: {e}")
    
    def process_directory(self):
        """Process all images in the input directory"""
        self.logger.info(f"🔍 Processing directory: {self.config.input_dir}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for root, dirs, files in os.walk(self.config.input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.config.input_dir)
                    image_files.append((full_path, rel_path))
        
        self.logger.info(f"📸 Found {len(image_files)} images to process")
        
        # Process images
        successful = 0
        failed = 0
        
        for image_path, rel_path in tqdm(image_files, desc="Processing images"):
            try:
                # Process image
                result = self.process_single_image(image_path, rel_path)
                
                if result is not None:
                    # Create output path
                    output_name = Path(rel_path).stem
                    output_dir = Path(self.config.output_dir) / Path(rel_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    base_path = output_dir / output_name
                    
                    # Save features
                    self.save_features(result, str(base_path))
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                failed += 1
        
        self.logger.info(f"✅ Processing complete!")
        self.logger.info(f"   Successful: {successful}")
        self.logger.info(f"   Failed: {failed}")
        self.logger.info(f"   Output directory: {self.config.output_dir}")

def parse_args() -> SpecializedFeatureConfig:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 or DINOv3 patch features from TRUMANS BEV recordings."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root of BEV frames, e.g. Data_release/Recordings/BEV_1",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root, e.g. Data_release/Recordings/dinov3/specialized_features_output",
    )
    parser.add_argument(
        "--encoder",
        choices=["dinov2", "dinov3"],
        default=None,
        help="DINO version. If omitted, the script prompts interactively.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Checkpoint .pth path. Required for DINOv3; optional for DINOv2 (uses torch.hub if omitted).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Backbone name. Defaults: dinov3-convnext-tiny (v3) or dinov2_vitb14 (v2).",
    )
    parser.add_argument(
        "--resize-strategy",
        default="stretch",
        choices=["smart", "pad", "crop", "stretch"],
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    encoder = args.encoder or prompt_encoder_choice()
    model_name = args.model_name or default_model_name(encoder)

    if encoder == "dinov3" and not args.weights:
        parser.error("DINOv3 extraction requires --weights /path/to/checkpoint.pth")

    return SpecializedFeatureConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_weights_path=args.weights,
        encoder=encoder,
        model_name=model_name,
        resize_strategy=args.resize_strategy,
        feature_types=["unpooled"],
        device=args.device,
    )


def main():
    config = parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Encoder: %s | Model: %s", config.encoder, config.model_name)
    extractor = SpecializedDINOExtractor(config)
    extractor.process_directory()


# Backward-compatible alias
SpecializedDINOv3Extractor = SpecializedDINOExtractor


if __name__ == "__main__":
    main()
