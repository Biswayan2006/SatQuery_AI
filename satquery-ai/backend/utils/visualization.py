"""
SatQuery AI — Visualization Utilities
Creates change maps, overlays, and fusion visualisations.
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("satquery.visualization")


# ── Change map ────────────────────────────────────────────────────────────────

def create_change_map(change_array: np.ndarray, threshold: Optional[float] = None) -> Image.Image:
    """
    Create a colour-coded change map.
      - Red  = high change
      - Yellow = moderate change
      - Green = low/no change

    change_array: [H, W] float32 in range [0, 1] (distance map)
    threshold: if given, values above = red, below = green (binary mode)
    """
    arr = np.array(change_array, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.squeeze()

    # Normalise to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr_norm = (arr - lo) / (hi - lo)
    else:
        arr_norm = np.zeros_like(arr)

    h, w = arr_norm.shape

    if threshold is not None:
        # Binary: changed (red) vs unchanged (green)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        changed = arr_norm >= threshold
        rgb[changed] = [220, 50, 50]    # red
        rgb[~changed] = [50, 180, 50]   # green
    else:
        # Gradient: green → yellow → red
        r = np.clip(arr_norm * 2, 0, 1) * 220
        g = np.clip((1 - arr_norm) * 2, 0, 1) * 180
        b = np.zeros_like(arr_norm)
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

    # Add a simple colourbar legend
    img = Image.fromarray(rgb, mode="RGB")
    img = _add_colorbar(img, min_val=float(lo), max_val=float(hi))
    return img


def _add_colorbar(image: Image.Image, min_val: float, max_val: float) -> Image.Image:
    """Append a horizontal colourbar strip below the image."""
    try:
        w, h = image.size
        bar_h = max(20, h // 20)
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

        for x in range(w):
            t = x / w
            r = int(min(t * 2, 1.0) * 220)
            g = int(min((1 - t) * 2, 1.0) * 180)
            bar[:, x] = [r, g, 0]

        bar_img = Image.fromarray(bar, mode="RGB")
        new_img = Image.new("RGB", (w, h + bar_h + 20), (20, 20, 20))
        new_img.paste(image, (0, 0))
        new_img.paste(bar_img, (0, h + 10))

        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, h + 1), f"{min_val:.2f}", fill="white", font=font)
        draw.text((w - 35, h + 1), f"{max_val:.2f}", fill="white", font=font)
        draw.text((w // 2 - 20, h + 1), "Change", fill="white", font=font)
        return new_img
    except Exception:
        return image


# ── Bounding box overlay ──────────────────────────────────────────────────────

def overlay_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    normalised: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    boxes: list of [x1, y1, x2, y2], normalised to [0,1] if normalised=True
    labels: string label per box
    scores: confidence score per box
    """
    COLORS = [
        (255, 80, 80), (80, 200, 80), (80, 80, 255),
        (255, 200, 0), (200, 0, 255), (0, 220, 220),
    ]

    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    w, h = result.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        color = COLORS[i % len(COLORS)]
        if normalised:
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
        else:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"
        text_w = len(text) * 7
        draw.rectangle([x1, y1 - 18, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), text, fill="white", font=font)

    return result


# ── SAR-Optical fusion visualisation ─────────────────────────────────────────

def create_fusion_visualization(
    optical: Image.Image,
    sar: Image.Image,
    fusion_result: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Create a side-by-side visualisation: [Optical | SAR | Fusion feature heatmap].
    """
    target_h = 256
    target_w = 256

    def _resize(img: Image.Image) -> Image.Image:
        return img.resize((target_w, target_h), Image.BICUBIC).convert("RGB")

    opt_resized = _resize(optical)
    sar_resized = _resize(sar) if sar is not None else Image.new("RGB", (target_w, target_h), (50, 50, 50))

    panels = [opt_resized, sar_resized]
    labels_text = ["Optical", "SAR"]

    # Fusion heatmap from feature vector
    if fusion_result is not None:
        feat = fusion_result.flatten()
        # Reshape to square-ish map
        side = int(np.sqrt(len(feat)))
        feat_map = feat[:side * side].reshape(side, side)
        feat_norm = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-6)
        # Colorise
        heatmap = np.zeros((side, side, 3), dtype=np.uint8)
        heatmap[:, :, 0] = (feat_norm * 220).astype(np.uint8)
        heatmap[:, :, 2] = ((1 - feat_norm) * 200).astype(np.uint8)
        heat_pil = Image.fromarray(heatmap, mode="RGB").resize((target_w, target_h), Image.NEAREST)
        panels.append(heat_pil)
        labels_text.append("Fusion")

    # Composite
    gap = 4
    total_w = target_w * len(panels) + gap * (len(panels) - 1)
    label_h = 24
    composite = Image.new("RGB", (total_w, target_h + label_h), (15, 15, 15))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(composite)
    for i, (panel, lbl) in enumerate(zip(panels, labels_text)):
        x_off = i * (target_w + gap)
        composite.paste(panel, (x_off, label_h))
        text_w = len(lbl) * 7
        draw.text((x_off + (target_w - text_w) // 2, 4), lbl, fill="white", font=font)

    return composite


# ── Confidence bar ────────────────────────────────────────────────────────────

def add_confidence_bar(image: Image.Image, confidence: float) -> Image.Image:
    """
    Append a horizontal confidence bar below the image.
    confidence: float in [0, 1]
    """
    try:
        w, h = image.size
        bar_h = 24
        new_img = Image.new("RGB", (w, h + bar_h), (20, 20, 20))
        new_img.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_img)
        bar_w = int(w * confidence)
        # Background
        draw.rectangle([0, h + 4, w, h + bar_h - 4], fill=(50, 50, 50))
        # Fill
        color = (
            (80, 200, 80) if confidence > 0.7
            else (230, 180, 0) if confidence > 0.4
            else (220, 60, 60)
        )
        if bar_w > 0:
            draw.rectangle([0, h + 4, bar_w, h + bar_h - 4], fill=color)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        label = f"Confidence: {confidence * 100:.1f}%"
        draw.text((4, h + 6), label, fill="white", font=font)

        return new_img
    except Exception:
        return image
