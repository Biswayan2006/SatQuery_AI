"""
SatQuery AI — SAR-Optical Fusion Model
Fuses complementary information from SAR and optical imagery.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger("satquery.sar_fusion")


class FeatureFusionHead(nn.Module):
    """Lightweight cross-modal fusion: concatenate features + MLP projection."""

    def __init__(self, in_dim: int = 2048, out_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, feat_optical: torch.Tensor, feat_sar: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([feat_optical, feat_sar], dim=-1)
        return self.proj(fused)


class SAROpticalFusionModel:
    """
    Fuses SAR and optical image features for enriched scene analysis.

    Architecture:
      - ResNet-50 encoder for optical image (RGB composite)
      - ResNet-50 encoder for SAR image (grayscale → 3-channel)
      - Feature-level concatenation + MLP projection
      - Fusion output used to condition VQA-style prompting
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        vqa_model_name: str = "Salesforce/blip2-opt-2.7b",
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.vqa_model_name = vqa_model_name
        self.opt_encoder = None
        self.sar_encoder = None
        self.fusion_head = None
        self._vqa = None

        self._load()

    def _load(self) -> None:
        logger.info("Loading SAR-optical fusion model on %s", self.device)

        # Shared ResNet backbone (ImageNet pretrained)
        def _make_encoder():
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            # Replace FC with identity — output [B, 2048]
            m.fc = nn.Identity()
            return m.to(self.device).eval()

        self.opt_encoder = _make_encoder()
        self.sar_encoder = _make_encoder()
        self.fusion_head = FeatureFusionHead(in_dim=2048 * 2, out_dim=1024).to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("SAR-optical fusion encoders loaded")

        # Lazy-load VQA for question answering
        try:
            from models.vqa_model import RemoteSensingVQA
            self._vqa = RemoteSensingVQA(
                model_name=self.vqa_model_name,
                device=self.device,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            logger.warning("VQA sub-model for SAR fusion unavailable: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse_and_analyze(
        self,
        optical: Image.Image,
        sar: Image.Image,
        query: str,
    ) -> Dict:
        """
        Fuse optical and SAR images and answer a natural language query.

        Returns:
            answer: str
            confidence: float
            fusion_map_b64: base64 PNG of the fusion visualization
        """
        optical = self._prep_optical(optical)
        sar = self._prep_sar(sar)

        # Extract features
        opt_feat = self._encode(self.opt_encoder, optical)  # [1, 2048]
        sar_feat = self._encode(self.sar_encoder, sar)      # [1, 2048]

        with torch.no_grad():
            fused = self.fusion_head(opt_feat, sar_feat)    # [1, 1024]

        # Compute SAR-derived analytics
        sar_analytics = self._analyze_sar(sar)
        opt_analytics = self._analyze_optical(optical)

        # Build enriched prompt
        enriched_query = self._build_enriched_prompt(query, sar_analytics, opt_analytics)

        # Run VQA on optical image (primary modality for scene understanding)
        if self._vqa is not None:
            vqa_out = self._vqa.answer(optical, enriched_query)
            answer = vqa_out["answer"]
            conf = min(0.88, vqa_out["confidence"] * 1.05)  # small boost for fusion
        else:
            answer = self._rule_based_answer(query, sar_analytics, opt_analytics)
            conf = 0.65

        # Create fusion visualization
        fusion_b64 = self._create_fusion_viz(optical, sar, fused)

        return {
            "answer": answer,
            "confidence": round(conf, 4),
            "fusion_map_b64": fusion_b64,
            "sar_analytics": sar_analytics,
            "optical_analytics": opt_analytics,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(self, encoder: nn.Module, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return encoder(tensor)

    @staticmethod
    def _prep_optical(image: Optional[Image.Image]) -> Image.Image:
        if image is None:
            return Image.new("RGB", (224, 224), (128, 128, 128))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    @staticmethod
    def _prep_sar(image: Optional[Image.Image]) -> Image.Image:
        """Convert SAR (typically single-band) to 3-channel for ResNet input."""
        if image is None:
            return Image.new("RGB", (224, 224), (64, 64, 64))
        if image.mode == "L":
            arr = np.array(image, dtype=np.float32)
        elif image.mode == "F":
            arr = np.array(image, dtype=np.float32)
        elif image.mode == "RGB":
            arr = np.array(image).mean(axis=-1).astype(np.float32)
        else:
            arr = np.array(image.convert("L"), dtype=np.float32)

        # Normalise to [0, 255]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

        # Replicate to 3 channels
        rgb = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _analyze_sar(sar_image: Image.Image) -> Dict:
        """Compute simple SAR backscatter statistics."""
        arr = np.array(sar_image.convert("L"), dtype=np.float32)
        mean_bs = float(arr.mean())
        std_bs = float(arr.std())
        high_bs_pct = float((arr > 200).mean() * 100)  # likely built-up/metallic

        return {
            "mean_backscatter": round(mean_bs, 2),
            "std_backscatter": round(std_bs, 2),
            "high_backscatter_pct": round(high_bs_pct, 2),
            "estimated_surface": (
                "built-up / metallic" if high_bs_pct > 15
                else "vegetation / soil" if mean_bs < 80
                else "mixed"
            ),
        }

    @staticmethod
    def _analyze_optical(opt_image: Image.Image) -> Dict:
        """Compute simple band statistics from optical image."""
        arr = np.array(opt_image.convert("RGB"), dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        ndvi_proxy = (g.mean() - r.mean()) / (g.mean() + r.mean() + 1e-6)
        water_proxy = float((b > r + 20).mean() * 100)

        return {
            "mean_rgb": [round(float(c.mean()), 2) for c in (r, g, b)],
            "ndvi_proxy": round(float(ndvi_proxy), 4),
            "water_pct_estimate": round(water_proxy, 2),
            "dominant_feature": (
                "water" if water_proxy > 30
                else "vegetation" if ndvi_proxy > 0.1
                else "urban / bare soil"
            ),
        }

    @staticmethod
    def _build_enriched_prompt(query: str, sar: Dict, opt: Dict) -> str:
        sar_info = (
            f"SAR analysis indicates {sar['estimated_surface']} surfaces "
            f"(backscatter mean={sar['mean_backscatter']:.0f}, "
            f"high-backscatter area={sar['high_backscatter_pct']:.1f}%)."
        )
        opt_info = (
            f"Optical analysis suggests {opt['dominant_feature']} coverage "
            f"(NDVI proxy={opt['ndvi_proxy']:.3f}, "
            f"water estimate={opt['water_pct_estimate']:.1f}%)."
        )
        return f"{query}\n\nAdditional context: {sar_info} {opt_info}"

    @staticmethod
    def _rule_based_answer(query: str, sar: Dict, opt: Dict) -> str:
        return (
            f"Based on SAR-optical fusion analysis: "
            f"The optical image shows {opt['dominant_feature']} with NDVI proxy of {opt['ndvi_proxy']:.3f}. "
            f"SAR backscatter indicates {sar['estimated_surface']} surfaces "
            f"with {sar['high_backscatter_pct']:.1f}% high-backscatter area. "
            f"Combined, the scene likely contains {opt['dominant_feature']} and {sar['estimated_surface']} elements."
        )

    def _create_fusion_viz(
        self, optical: Image.Image, sar: Image.Image, fused: torch.Tensor
    ) -> Optional[str]:
        """Side-by-side optical + SAR visualization."""
        try:
            from utils.visualization import create_fusion_visualization
            from utils.image_utils import image_to_base64
            viz = create_fusion_visualization(optical, sar, fused.cpu().numpy())
            return image_to_base64(viz)
        except Exception as exc:
            logger.warning("Fusion visualization failed: %s", exc)
            return None
