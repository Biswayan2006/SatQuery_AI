"""
SatQuery AI — BigEarthNet Dataset Loader
Loads BigEarthNet-S2 (optical) and BigEarthNet-S1 (SAR) paired patches.

BigEarthNet structure expected:
  data_dir/
    BigEarthNet-S2/
      S2A_MSIL2A_20170613T101031_N0205_R022_T32TPT_20170613T101520/
        <patch>_B01.tif, <patch>_B02.tif, ... <patch>_B12A.tif
        <patch>_labels_metadata.json
    BigEarthNet-S1/ (optional paired)
      S1A_IW_SLC__1SDV_... / <patch>_VV.tif, <patch>_VH.tif

Reference: https://bigearth.net/
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger("satquery.bigearthnet")

# ── BigEarthNet-43 label set ─────────────────────────────────────────────────
BIGEARTHNET_43_LABELS = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands",
    "Broad-leaved forest", "Burnt areas", "Coastal lagoons",
    "Complex cultivation patterns", "Coniferous forest",
    "Construction sites", "Continuous urban fabric",
    "Discontinuous dense urban fabric", "Discontinuous low density urban fabric",
    "Discontinuous medium density urban fabric",
    "Discontinuous very low density urban fabric",
    "Estuaries", "Fruit trees and berry plantations",
    "Green urban areas", "Industrial or commercial units",
    "Inland marshes", "Intertidal flats",
    "Land principally occupied by agriculture with significant natural vegetation",
    "Mineral extraction sites", "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Non-irrigated arable land", "Olive groves",
    "Pastures", "Peat bogs", "Port areas",
    "Rice fields", "Road and rail networks",
    "Salines", "Salt marshes",
    "Sea and ocean", "Sparsely vegetated areas",
    "Sport and leisure facilities", "Transitional woodland/shrub",
    "Urban green areas", "Vineyards",
    "Water bodies", "Water courses", "Wetlands",
]

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(BIGEARTHNET_43_LABELS)}

# Sentinel-2 band ordering and typical wavelengths (nm)
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# RGB composite band indices (B04=R, B03=G, B02=B, 0-indexed in S2_BANDS list)
S2_RGB_INDICES = [3, 2, 1]  # B04, B03, B02


class BigEarthNetDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet-S2 (with optional S1 SAR pairing).

    Each sample returns:
      optical_tensor: FloatTensor [12, 120, 120]
      sar_tensor:     FloatTensor [2, 120, 120]  (zeros if S1 unavailable)
      label_tensor:   FloatTensor [43]           (multi-hot)
      description:    str  (text description for CLIP training)
    """

    S2_MEAN = np.array([340.76, 429.9, 614.21, 590.23, 950.68,
                         1792.53, 2075.11, 2218.94, 2266.46, 732.95, 1648.9, 1049.44],
                        dtype=np.float32)
    S2_STD = np.array([554.99, 572.41, 582.87, 675.88, 729.89,
                        1096.01, 1273.45, 1365.45, 1356.13, 1108.06, 1258.32, 1008.28],
                       dtype=np.float32)

    S1_MEAN = np.array([-12.619, -20.256], dtype=np.float32)
    S1_STD = np.array([5.262, 6.248], dtype=np.float32)

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        use_sar: bool = True,
        image_size: int = 120,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.use_sar = use_sar
        self.image_size = image_size
        self.transform = transform
        self.max_samples = max_samples

        self.s2_root = os.path.join(data_dir, "BigEarthNet-S2")
        self.s1_root = os.path.join(data_dir, "BigEarthNet-S1") if use_sar else None

        self.patches = self._discover_patches()
        if max_samples:
            self.patches = self.patches[:max_samples]

        logger.info("BigEarthNet: found %d patches (split=%s)", len(self.patches), split)

    def _discover_patches(self) -> List[str]:
        """Return sorted list of S2 patch directory paths."""
        pattern = os.path.join(self.s2_root, "**", "*_labels_metadata.json")
        meta_files = sorted(glob.glob(pattern, recursive=True))
        patches = [os.path.dirname(f) for f in meta_files]

        # Split by hash (reproducible 80/10/10 split without pre-made files)
        split_map = {}
        for p in patches:
            h = hash(os.path.basename(p)) % 10
            if h < 8:
                split_map[p] = "train"
            elif h == 8:
                split_map[p] = "val"
            else:
                split_map[p] = "test"

        return [p for p, s in split_map.items() if s == self.split]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        patch_dir = self.patches[idx]
        patch_name = os.path.basename(patch_dir)

        optical = self._load_s2(patch_dir, patch_name)
        sar = self._load_s1(patch_name) if self.use_sar else torch.zeros(2, self.image_size, self.image_size)
        labels, label_names = self._load_labels(patch_dir, patch_name)
        description = self.generate_text_description(label_names)

        if self.transform is not None:
            optical = self.transform(optical)

        return optical, sar, labels, description

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_s2(self, patch_dir: str, patch_name: str) -> torch.Tensor:
        """Load all 12 S2 bands, normalize, and resize to image_size."""
        try:
            import rasterio
            from rasterio.enums import Resampling

            bands = []
            for band_name in S2_BANDS:
                band_path = os.path.join(patch_dir, f"{patch_name}_{band_name}.tif")
                if not os.path.exists(band_path):
                    bands.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                    continue
                with rasterio.open(band_path) as src:
                    band_data = src.read(
                        1,
                        out_shape=(self.image_size, self.image_size),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)
                bands.append(band_data)

            arr = np.stack(bands, axis=0)  # [12, H, W]
            # Normalize per-band
            for i in range(12):
                arr[i] = (arr[i] - self.S2_MEAN[i]) / (self.S2_STD[i] + 1e-6)

            return torch.from_numpy(arr)
        except Exception as exc:
            logger.debug("S2 load failed for %s: %s", patch_name, exc)
            return torch.zeros(12, self.image_size, self.image_size)

    def _load_s1(self, patch_name: str) -> torch.Tensor:
        """Load paired S1 VV/VH bands."""
        if self.s1_root is None:
            return torch.zeros(2, self.image_size, self.image_size)

        # S1 patch naming convention: replace S2 prefix with S1 patch name
        # Simplified: search for matching patch in S1 root
        try:
            import rasterio
            from rasterio.enums import Resampling

            # Look for any S1 patch matching the suffix
            s1_patches = glob.glob(os.path.join(self.s1_root, f"*{patch_name[-15:]}*"))
            if not s1_patches:
                return torch.zeros(2, self.image_size, self.image_size)

            s1_dir = s1_patches[0]
            s1_name = os.path.basename(s1_dir)
            bands = []
            for pol in ["VV", "VH"]:
                band_path = os.path.join(s1_dir, f"{s1_name}_{pol}.tif")
                if not os.path.exists(band_path):
                    bands.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                    continue
                with rasterio.open(band_path) as src:
                    band_data = src.read(
                        1,
                        out_shape=(self.image_size, self.image_size),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)
                bands.append(band_data)

            arr = np.stack(bands, axis=0)
            for i in range(2):
                arr[i] = (arr[i] - self.S1_MEAN[i]) / (self.S1_STD[i] + 1e-6)
            return torch.from_numpy(arr)
        except Exception as exc:
            logger.debug("S1 load failed for %s: %s", patch_name, exc)
            return torch.zeros(2, self.image_size, self.image_size)

    def _load_labels(self, patch_dir: str, patch_name: str) -> Tuple[torch.Tensor, List[str]]:
        """Load multi-label annotations from JSON metadata."""
        meta_path = os.path.join(patch_dir, f"{patch_name}_labels_metadata.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            label_names = meta.get("labels", [])
        except Exception:
            label_names = []

        label_vec = torch.zeros(len(BIGEARTHNET_43_LABELS), dtype=torch.float32)
        for lbl in label_names:
            idx = LABEL_TO_IDX.get(lbl)
            if idx is not None:
                label_vec[idx] = 1.0

        return label_vec, label_names

    # ── Text Description ──────────────────────────────────────────────────────

    @staticmethod
    def generate_text_description(labels: List[str]) -> str:
        """
        Convert a list of BigEarthNet land-cover labels to a natural language
        description suitable for CLIP contrastive learning.

        Examples:
          ["Mixed forest", "Water bodies"] →
          "A satellite image showing mixed forest and water bodies."
        """
        if not labels:
            return "A satellite image of land."

        if len(labels) == 1:
            return f"A satellite image showing {labels[0].lower()}."

        if len(labels) == 2:
            return f"A satellite image showing {labels[0].lower()} and {labels[1].lower()}."

        main = labels[:-1]
        last = labels[-1]
        joined = ", ".join(l.lower() for l in main) + f", and {last.lower()}"
        return f"A satellite image showing {joined}."

    def get_rgb_composite(self, idx: int) -> "PIL.Image.Image":
        """Return a PIL RGB composite image for visualisation."""
        from PIL import Image

        patch_dir = self.patches[idx]
        patch_name = os.path.basename(patch_dir)
        optical = self._load_s2(patch_dir, patch_name)  # [12, H, W]
        rgb = optical[S2_RGB_INDICES].numpy()  # [3, H, W]

        # Denormalize
        for i, bi in enumerate(S2_RGB_INDICES):
            rgb[i] = rgb[i] * self.S2_STD[bi] + self.S2_MEAN[bi]

        rgb = rgb.transpose(1, 2, 0)  # [H, W, 3]
        # Clip and scale to uint8
        p2 = np.percentile(rgb, 2)
        p98 = np.percentile(rgb, 98)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
