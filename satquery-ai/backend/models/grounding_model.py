"""
SatQuery AI — Text-Guided Region Grounding
Uses OWL-ViT for open-vocabulary object detection in satellite imagery.
"""
from __future__ import annotations

import base64
import io
import logging
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


class RemoteSensingGrounding:
    """
    Open-vocabulary grounding using OWL-ViT.

    Returns bounding boxes in normalised [x1, y1, x2, y2] format (0–1).
    Also produces an annotated image with drawn boxes.
    """

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        score_threshold: float = 0.2,
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
            text_query: Object description, e.g. "water body", "building", "road"

        Returns:
            dict with keys: boxes, labels, scores, annotated_image (base64)
        """
        if image is None:
            return {"boxes": [], "labels": [], "scores": [], "annotated_image": None}

        image = self._preprocess(image)
        w, h = image.size

        # OWL-ViT expects a list of texts per image
        texts = [[text_query]]

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

            # Normalise to [0, 1]
            boxes_norm = boxes_abs.copy()
            boxes_norm[:, [0, 2]] /= w
            boxes_norm[:, [1, 3]] /= h
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

            labels = [text_query] * len(scores)

            # Sort by score descending
            order = np.argsort(scores)[::-1]
            boxes_norm = boxes_norm[order].tolist()
            scores = scores[order].tolist()
            labels = [labels[i] for i in order]

            annotated_b64 = self._draw_boxes(image, boxes_abs[order], labels, scores)

            return {
                "boxes": boxes_norm,
                "labels": labels,
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
