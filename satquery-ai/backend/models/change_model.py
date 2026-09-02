"""
SatQuery AI — Change Detection Model
Siamese deep feature comparison for bi-temporal satellite imagery.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger("satquery.change")


class ChangeDetectionModel:
    """
    Bi-temporal change detection using a Siamese ResNet backbone.

    Approach:
      1. Extract deep features from both images using a pretrained ResNet
      2. Compute per-pixel cosine distance in feature space
      3. Threshold to produce a binary change mask
      4. Optionally answer questions about the detected changes via VQA
    """

    RESIZE = (256, 256)
    FEATURE_LAYER = "layer3"  # ResNet mid-level features
    CHANGE_THRESHOLD = 0.35  # cosine distance threshold

    def __init__(
        self,
        backbone_name: str = "microsoft/resnet-50",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        threshold: float = 0.35,
    ):
        self.backbone_name = backbone_name
        self.device = device
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.backbone = None
        self.feature_extractor = None

        self._load()

    def _load(self) -> None:
        import torchvision.models as tvm

        logger.info("Loading change detection backbone: resnet50 on %s", self.device)

        # Use torchvision ResNet50 (faster than HF for feature extraction)
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Remove classification head — keep up to layer3
        self.feature_extractor = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,  # Output: [B, 1024, H/16, W/16]
        )
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.transform = T.Compose([
            T.Resize(self.RESIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("Change detection backbone loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_changes(self, img1: Image.Image, img2: Image.Image) -> Dict:
        """
        Detect changes between two co-registered images.

        Returns:
            change_map: np.ndarray [H, W] float32 in [0, 1]
            changed_regions: list of rough bounding boxes
            change_percentage: float (0–100)
        """
        feat1 = self._extract_features(img1)  # [1, C, h, w]
        feat2 = self._extract_features(img2)

        # Cosine distance per pixel
        dist_map = self._cosine_distance(feat1, feat2)  # [h, w]

        # Upsample to original input resolution
        h, w = self.RESIZE
        dist_resized = F.interpolate(
            dist_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Binary mask
        binary_mask = (dist_resized > self.threshold).astype(np.float32)
        change_pct = float(binary_mask.mean() * 100)

        changed_regions = self._find_regions(binary_mask)

        return {
            "change_map": dist_resized,
            "binary_mask": binary_mask,
            "changed_regions": changed_regions,
            "change_percentage": round(change_pct, 2),
        }

    def answer_change_question(
        self, img1: Image.Image, img2: Image.Image, question: str
    ) -> Dict:
        """
        Answer a natural language question about change between two images.
        """
        change_out = self.detect_changes(img1, img2)
        pct = change_out["change_percentage"]
        binary = change_out["binary_mask"]

        # Rule-based QA on change metrics
        answer = self._change_qa(question, pct, binary, change_out["changed_regions"])

        # Generate change map image
        from utils.visualization import create_change_map
        from utils.image_utils import image_to_base64
        change_pil = create_change_map(change_out["change_map"])
        change_b64 = image_to_base64(change_pil)

        # Confidence: higher when change is clear (far from threshold)
        confidence = min(0.92, 0.5 + abs(pct / 100 - self.threshold) * 1.5)

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "change_map_b64": change_b64,
            "change_percentage": pct,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract feature map from image. Returns [1, C, H, W]."""
        if image is None:
            return torch.zeros(1, 1024, 16, 16, device=self.device)
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(tensor)
        return features

    @staticmethod
    def _cosine_distance(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Per-pixel cosine distance. Input: [1, C, H, W]. Output: [H, W]."""
        # Normalise along channel dimension
        f1_n = F.normalize(f1, dim=1)
        f2_n = F.normalize(f2, dim=1)
        cos_sim = (f1_n * f2_n).sum(dim=1).squeeze(0)  # [H, W]
        return 1.0 - cos_sim  # distance in [0, 2]; typical range [0, 1]

    @staticmethod
    def _find_regions(binary_mask: np.ndarray, min_area_pct: float = 0.5) -> List[Dict]:
        """
        Find connected changed regions using simple scanline grouping.
        Returns bounding boxes as dicts with x1, y1, x2, y2, area_pct.
        """
        try:
            import cv2

            mask_uint8 = (binary_mask * 255).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_uint8, connectivity=8
            )
            h, w = binary_mask.shape
            regions = []
            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                area_pct = area / (h * w) * 100
                if area_pct < min_area_pct:
                    continue
                x = stats[i, cv2.CC_STAT_LEFT] / w
                y = stats[i, cv2.CC_STAT_TOP] / h
                rw = stats[i, cv2.CC_STAT_WIDTH] / w
                rh = stats[i, cv2.CC_STAT_HEIGHT] / h
                regions.append({
                    "x1": round(x, 3), "y1": round(y, 3),
                    "x2": round(x + rw, 3), "y2": round(y + rh, 3),
                    "area_pct": round(area_pct, 2),
                })
            return sorted(regions, key=lambda r: -r["area_pct"])[:10]
        except Exception:
            return []

    @staticmethod
    def _change_qa(question: str, pct: float, binary: np.ndarray, regions: List) -> str:
        """Simple rule-based change QA."""
        q = question.lower()

        if any(kw in q for kw in ["how much", "percentage", "percent", "area"]):
            return f"Approximately {pct:.1f}% of the scene has changed between the two images."

        if any(kw in q for kw in ["where", "location", "which part"]):
            if not regions:
                return "No significant change regions were detected."
            desc = []
            for r in regions[:3]:
                cx = (r["x1"] + r["x2"]) / 2
                cy = (r["y1"] + r["y2"]) / 2
                h_pos = "left" if cx < 0.33 else "center" if cx < 0.66 else "right"
                v_pos = "upper" if cy < 0.33 else "middle" if cy < 0.66 else "lower"
                desc.append(f"{v_pos}-{h_pos} ({r['area_pct']:.1f}% area)")
            return "Changed regions detected in: " + "; ".join(desc) + "."

        if any(kw in q for kw in ["how many", "count", "number"]):
            return f"{len(regions)} distinct changed region(s) detected."

        # Default: descriptive answer
        level = (
            "no significant" if pct < 2 else
            "minor" if pct < 10 else
            "moderate" if pct < 30 else
            "substantial" if pct < 60 else
            "extensive"
        )
        region_str = f" with {len(regions)} distinct region(s)" if regions else ""
        return (
            f"The analysis detected {level} change ({pct:.1f}%) between "
            f"the two images{region_str}."
        )
