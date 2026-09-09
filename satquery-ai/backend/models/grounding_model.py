"""
SatQuery AI — Text-Guided Region Grounding
Uses OWL-ViT for open-vocabulary object detection in satellite imagery.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("satquery.grounding")

# Colours for bounding box visualisation
BOX_COLORS = [
    (255, 80, 80), (80, 200, 80), (80, 80, 255),
    (255, 200, 0), (200, 0, 255), (0, 220, 220),
    (255, 128, 0), (0, 180, 128),
]

# ── Query-to-label mapping for remote sensing ────────────────────────────────
# Maps natural-language queries to OWL-ViT-compatible object labels.
_RS_LABEL_MAP: Dict[str, List[str]] = {
    # Infrastructure
    "road":       ["road", "street", "highway", "path"],
    "building":   ["building", "house", "structure"],
    "bridge":     ["bridge"],
    "railway":    ["railway", "train track", "railroad"],
    "airport":    ["airport", "runway"],
    "parking":    ["parking lot", "parking"],
    # Water
    "water":      ["water", "river", "lake", "pond", "stream"],
    "river":      ["river", "stream", "waterway"],
    "lake":       ["lake", "pond", "water body"],
    "coast":      ["coast", "shoreline", "beach"],
    # Vegetation
    "tree":       ["tree", "forest", "vegetation"],
    "forest":     ["forest", "trees", "woodland"],
    "field":      ["field", "farmland", "agriculture", "crop"],
    "crop":       ["crop", "farmland", "agriculture"],
    "grass":      ["grass", "grassland", "lawn"],
    # Urban
    "urban":      ["building", "structure", "house"],
    "residential":["house", "building", "residential area"],
    "industrial": ["factory", "warehouse", "industrial building"],
    "commercial": ["commercial building", "shop", "store"],
    # Land cover
    "bare":       ["bare ground", "dirt", "soil", "sand"],
    "sand":       ["sand", "dirt", "bare ground"],
    "rock":       ["rock", "boulder", "cliff"],
    "snow":       ["snow", "ice"],
    "cloud":      ["cloud"],
    # SAR-specific
    "ship":       ["ship", "boat", "vessel"],
    "vessel":     ["ship", "boat", "vessel"],
    "car":        ["car", "vehicle", "truck"],
    "vehicle":    ["car", "truck", "vehicle"],
    # Flood
    "flood":      ["water", "flood", "flooding"],
    "flooding":   ["water", "flood"],
    "damage":     ["damage", "debris", "ruins"],
}


def _extract_labels(query: str) -> List[str]:
    """
    Extract OWL-ViT-compatible object labels from a natural-language query.

    Tries keyword matching first, falls back to extracting noun phrases.
    Returns 1-4 labels for parallel detection.
    """
    q = query.lower().strip()

    # 1. Direct keyword match
    for key, labels in _RS_LABEL_MAP.items():
        if key in q:
            return labels[:3]

    # 2. Extract simple noun phrases (2-3 word chunks)
    # Remove common verbs / filler
    cleaned = re.sub(
        r"\b(locate|detect|find|show|identify|where|are|the|in|this|image|satellite|map|please|can|you|see|what|is)\b",
        "", q, flags=re.IGNORECASE
    )
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
    if words:
        # Take first 2-4 meaningful words as label candidates
        candidates = [" ".join(words[:min(4, len(words))])]
        # Also try individual words if they are meaningful nouns
        for w in words[:4]:
            if w not in candidates:
                candidates.append(w)
        return candidates[:4]

    # 3. Fallback — pass the raw query
    return [query]


class RemoteSensingGrounding:
    """
    Open-vocabulary grounding using OWL-ViT.

    Returns bounding boxes in normalised [x1, y1, x2, y2] format (0-1).
    Also produces an annotated image with drawn boxes.
    """

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        score_threshold: float = 0.1,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.score_threshold = score_threshold
        self.model = None
        self.processor = None

        self._load()

    def _load(self) -> None:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        logger.info("Loading grounding model: %s on %s", self.model_name, self.device)
        self.processor = OwlViTProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Grounding model loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def ground(self, image: Image.Image, text_query: str) -> Dict:
        """
        Locate all instances matching text_query in the image.

        Args:
            image: PIL RGB image
            text_query: Natural-language query, e.g. "Locate the roads in this image"

        Returns:
            dict with keys: boxes, labels, scores, annotated_image (base64)
        """
        if image is None:
            return {"boxes": [], "labels": [], "scores": [], "annotated_image": None}

        image = self._preprocess(image)
        w, h = image.size

        # Extract OWL-ViT-compatible labels from the query
        labels_to_detect = _extract_labels(text_query)
        logger.info(
            "Grounding query=%r -> labels=%s (threshold=%.2f)",
            text_query, labels_to_detect, self.score_threshold,
        )

        # OWL-ViT expects a list of texts per image
        texts = [labels_to_detect]

        try:
            inputs = self.processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process — target_sizes needed for absolute coords
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.score_threshold,
                target_sizes=target_sizes,
            )[0]

            boxes_abs = results["boxes"].cpu().numpy()  # [x1, y1, x2, y2] absolute
            scores = results["scores"].cpu().numpy()
            labels_idx = results["labels"].cpu().numpy()

            # Map label indices back to our text labels
            label_names = [labels_to_detect[i] for i in labels_idx]

            # Normalise to [0, 1]
            boxes_norm = boxes_abs.copy()
            boxes_norm[:, [0, 2]] /= w
            boxes_norm[:, [1, 3]] /= h
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

            # Sort by score descending
            order = np.argsort(scores)[::-1]
            boxes_norm = boxes_norm[order].tolist()
            scores = scores[order].tolist()
            label_names = [label_names[i] for i in order]

            annotated_b64 = self._draw_boxes(image, boxes_abs[order], label_names, scores)

            return {
                "boxes": boxes_norm,
                "labels": label_names,
                "scores": [round(float(s), 4) for s in scores],
                "annotated_image": annotated_b64,
            }

        except Exception as exc:
            logger.error("Grounding inference failed: %s", exc)
            return {"boxes": [], "labels": [], "scores": [], "annotated_image": None}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        max_side = 768
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        return image

    @staticmethod
    def _draw_boxes(
        image: Image.Image,
        boxes_abs: np.ndarray,
        labels: List[str],
        scores: List[float],
    ) -> Optional[str]:
        """Draw bounding boxes and return base64 PNG."""
        try:
            draw_img = image.copy().convert("RGB")
            draw = ImageDraw.Draw(draw_img, "RGBA")

            for i, (box, label, score) in enumerate(zip(boxes_abs, labels, scores)):
                color = BOX_COLORS[i % len(BOX_COLORS)]
                x1, y1, x2, y2 = box.tolist()

                # Semi-transparent fill
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.rectangle(
                    [x1, y1 - 18, x1 + len(label) * 7 + 50, y1],
                    fill=(*color, 200),
                )

                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                except Exception:
                    font = ImageFont.load_default()

                draw.text((x1 + 3, y1 - 16), f"{label} {score:.2f}", fill="white", font=font)

            buf = io.BytesIO()
            draw_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("Box drawing failed: %s", exc)
            return None
