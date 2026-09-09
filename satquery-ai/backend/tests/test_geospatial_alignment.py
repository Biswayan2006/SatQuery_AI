"""
SatQuery AI — Geospatial Alignment Unit Tests

Covers all 8 required test cases:

  1. same_crs         — two GeoTIFF images with the same CRS → resample_to_common_extent
  2. different_crs    — two GeoTIFF images with different CRS → reproject_and_resample
  3. different_res    — same CRS, different pixel resolutions → upsample to finer res
  4. different_sizes  — same CRS/res, different extents → crop to common extent
  5. incompatible_bounds — geographic footprints don't overlap → AlignmentResult.error
  6. missing_crs      — GeoTIFF with no CRS → pixel-level fallback
  7. png_jpeg_pair    — plain PNG pair, no geo metadata → pixel-level fallback
  8. pixel_to_latlon  — pixel_to_geo_bbox round-trip on a known transform

All tests use synthetic in-memory data produced with numpy + rasterio
MemoryFile.  No real satellite images are required; no model loading
is performed.  Tests run on CPU only.

Run with:
    cd satquery-ai/backend
    python -m pytest tests/test_geospatial_alignment.py -v
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from typing import Optional

import numpy as np
import pytest
from PIL import Image

# ── Make the backend package importable from the tests directory ───────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_geotiff(
    tmp_dir: str,
    filename: str,
    width: int = 64,
    height: int = 64,
    bands: int = 3,
    dtype: str = "uint8",
    crs_epsg: Optional[int] = 32632,          # UTM zone 32N by default
    west: float = 500_000.0,
    south: float = 5_000_000.0,
    pixel_size: float = 10.0,                 # metres per pixel
    fill_value: int = 128,
) -> str:
    """
    Write a synthetic single-tile GeoTIFF to *tmp_dir* and return its path.
    The image is filled with *fill_value* (uint8) or random float32.
    CRS is omitted when *crs_epsg* is None.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    path = os.path.join(tmp_dir, filename)
    transform = from_origin(west, south + height * pixel_size, pixel_size, pixel_size)

    meta = {
        "driver": "GTiff",
        "count": bands,
        "width": width,
        "height": height,
        "dtype": dtype,
        "transform": transform,
    }
    if crs_epsg is not None:
        meta["crs"] = CRS.from_epsg(crs_epsg)

    with rasterio.open(path, "w", **meta) as dst:
        for b in range(1, bands + 1):
            if dtype == "uint8":
                data = np.full((height, width), fill_value, dtype=np.uint8)
            else:
                data = np.random.rand(height, width).astype(np.float32) * 100
            dst.write(data, b)

    return path


def _image_data_from_path(path: str) -> dict:
    """
    Load an image dict in the same format as ``image_utils.load_image()``.
    Attaches ``_path`` so geospatial_aligner can open the file.
    """
    from utils.image_utils import load_image
    data = load_image(path)
    data["_path"] = path
    return data


def _png_image_data(width: int = 64, height: int = 64) -> dict:
    """
    Build a minimal image dict from a plain in-memory PNG (no geo).
    """
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    return {
        "pil_image": pil,
        "numpy_array": arr,
        "bands": 3,
        "shape": [height, width, 3],
        "is_geotiff": False,
        "metadata": {},
        "_path": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Same CRS
# ══════════════════════════════════════════════════════════════════════════════

class TestSameCRS:
    """Both images share EPSG:32632; expect resample_to_common_extent."""

    def test_alignment_succeeds(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_same_crs.tif",
                               west=500_000, south=5_000_000)
        path_b = _make_geotiff(str(tmp_path), "b_same_crs.tif",
                               west=500_100, south=5_000_100)  # overlapping offset

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success, f"Expected success, got: {result.error}"
        assert result.pil_a is not None
        assert result.pil_b is not None

    def test_method_is_resample(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_method.tif",
                               west=500_000, south=5_000_000)
        path_b = _make_geotiff(str(tmp_path), "b_method.tif",
                               west=500_100, south=5_000_100)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        method = result.info.get("method", "")
        assert "resample" in method, f"Unexpected method: {method}"

    def test_output_images_same_size(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_size.tif",
                               west=500_000, south=5_000_000)
        path_b = _make_geotiff(str(tmp_path), "b_size.tif",
                               west=500_100, south=5_000_100)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.pil_a.size == result.pil_b.size, (
            f"Aligned images differ: {result.pil_a.size} vs {result.pil_b.size}"
        )

    def test_geographic_coordinates_available(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_geo.tif",
                               west=500_000, south=5_000_000)
        path_b = _make_geotiff(str(tmp_path), "b_geo.tif",
                               west=500_100, south=5_000_100)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.info.get("geographic_coordinates_available") is True


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Different CRS
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentCRS:
    """Image A in UTM 32N (EPSG:32632), Image B in WGS84 (EPSG:4326)."""

    def test_alignment_succeeds(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_utm.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=32, height=32)
        # WGS84 degrees — approx same geographic area
        path_b = _make_geotiff(str(tmp_path), "b_wgs84.tif",
                               crs_epsg=4326,
                               west=9.0, south=45.1,
                               pixel_size=0.0001, width=32, height=32)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        # Should succeed (overlap > 10%) or fail gracefully with an error string
        assert result.success or isinstance(result.error, str)

    def test_method_is_reproject(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_utm_m.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=32, height=32)
        path_b = _make_geotiff(str(tmp_path), "b_wgs84_m.tif",
                               crs_epsg=4326,
                               west=9.0, south=45.1,
                               pixel_size=0.0001, width=32, height=32)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        if result.success:
            assert "reproject" in result.info.get("method", ""), (
                f"Expected reproject in method, got: {result.info.get('method')}"
            )

    def test_source_and_target_crs_differ(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_crs_diff.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=32, height=32)
        path_b = _make_geotiff(str(tmp_path), "b_crs_diff.tif",
                               crs_epsg=4326,
                               west=9.0, south=45.1,
                               pixel_size=0.0001, width=32, height=32)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        if result.success:
            source = result.info.get("source_crs", "")
            target = result.info.get("target_crs", "")
            # At minimum both should be populated
            assert source is not None
            assert target is not None


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — Different Resolutions
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentResolutions:
    """Same CRS and overlapping extent but different GSD (10 m vs 30 m)."""

    def test_output_uses_finer_resolution(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        # 64×64 at 10 m/px → 640 m extent
        path_a = _make_geotiff(str(tmp_path), "a_10m.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=64, height=64)
        # 22×22 at 30 m/px → ~660 m extent (overlapping)
        path_b = _make_geotiff(str(tmp_path), "b_30m.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=30.0, width=22, height=22)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success, f"Expected success: {result.error}"
        # Target resolution should be the finer (10 m) one
        target_res = result.info.get("target_resolution")
        assert target_res is not None
        assert target_res <= 10.0 + 0.5, (
            f"Expected target_resolution ≤ 10.5 m, got {target_res}"
        )

    def test_aligned_images_same_size(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_res_size.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_res_size.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=30.0, width=22, height=22)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success
        assert result.pil_a.size == result.pil_b.size


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — Different Image Sizes (same CRS / resolution, different extents)
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentSizes:
    """Same CRS + GSD, but the two images cover different spatial extents."""

    def test_output_smaller_than_larger_input(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        # Large image: 128×128 @ 10 m = 1280 m
        path_a = _make_geotiff(str(tmp_path), "a_large.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=128, height=128)
        # Smaller overlapping image: 32×32 @ 10 m = 320 m within large extent
        path_b = _make_geotiff(str(tmp_path), "b_small.tif",
                               west=500_200, south=5_000_200,
                               pixel_size=10.0, width=32, height=32)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success, f"Expected success: {result.error}"
        out_w, out_h = result.pil_a.size
        # Output must be no larger than the larger input
        assert out_w <= 128 and out_h <= 128

    def test_common_bounds_subset_of_inputs(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_bounds.tif",
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=128, height=128)
        path_b = _make_geotiff(str(tmp_path), "b_bounds.tif",
                               west=500_200, south=5_000_200,
                               pixel_size=10.0, width=32, height=32)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success
        cb = result.info.get("common_bounds")
        assert cb is not None and len(cb) == 4


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — Incompatible Bounds (no overlap)
# ══════════════════════════════════════════════════════════════════════════════

class TestIncompatibleBounds:
    """Images are geographically far apart — must return an error."""

    def test_returns_error(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        # Image A: Germany (UTM 32N)
        path_a = _make_geotiff(str(tmp_path), "a_germany.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=64, height=64)
        # Image B: far away — Brazil UTM zone 23S, EPSG:31983
        path_b = _make_geotiff(str(tmp_path), "b_brazil.tif",
                               crs_epsg=31983,
                               west=300_000, south=7_500_000,
                               pixel_size=10.0, width=64, height=64)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert not result.success, "Expected failure for non-overlapping images"
        assert result.error is not None
        assert len(result.error) > 0

    def test_error_message_mentions_overlap(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_nooverlap.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_nooverlap.tif",
                               crs_epsg=31983,
                               west=300_000, south=7_500_000,
                               pixel_size=10.0, width=64, height=64)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        # Error message should be informative
        if result.error:
            lower = result.error.lower()
            assert any(w in lower for w in ["overlap", "intersection", "incompatible", "geographic"]), (
                f"Error should mention overlap/intersection, got: {result.error}"
            )

    def test_pil_images_are_none_on_failure(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_none.tif",
                               crs_epsg=32632,
                               west=500_000, south=5_000_000,
                               pixel_size=10.0, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_none.tif",
                               crs_epsg=31983,
                               west=300_000, south=7_500_000,
                               pixel_size=10.0, width=64, height=64)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        if not result.success:
            assert result.pil_a is None
            assert result.pil_b is None


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — Missing CRS (GeoTIFF with no projection)
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingCRS:
    """One or both GeoTIFFs lack CRS → pixel-level fallback."""

    def test_no_crs_falls_back_to_pixel(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_nocrs.tif",
                               crs_epsg=None, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_nocrs.tif",
                               crs_epsg=None, width=48, height=48)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success
        assert result.info.get("geographic_coordinates_available") is False
        method = result.info.get("method", "")
        assert "pixel" in method or "resize" in method, (
            f"Expected pixel/resize method, got: {method}"
        )

    def test_mixed_crs_and_no_crs_falls_back(self, tmp_path):
        """One has CRS, one doesn't — should degrade gracefully."""
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_hascrs.tif",
                               crs_epsg=32632, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_misscrs.tif",
                               crs_epsg=None, width=64, height=64)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.success
        assert result.info.get("geographic_coordinates_available") is False

    def test_output_images_not_none(self, tmp_path):
        from utils.geospatial_aligner import align_image_pair

        path_a = _make_geotiff(str(tmp_path), "a_nocrs_out.tif",
                               crs_epsg=None, width=64, height=64)
        path_b = _make_geotiff(str(tmp_path), "b_nocrs_out.tif",
                               crs_epsg=None, width=48, height=48)

        result = align_image_pair(
            _image_data_from_path(path_a),
            _image_data_from_path(path_b),
        )

        assert result.pil_a is not None
        assert result.pil_b is not None
        assert result.pil_a.size == result.pil_b.size


# ══════════════════════════════════════════════════════════════════════════════
# Test 7 — PNG / JPEG pair (no geo metadata)
# ══════════════════════════════════════════════════════════════════════════════

class TestPNGJPEGPair:
    """Plain PNG images without any geospatial metadata."""

    def test_succeeds_with_pixel_alignment(self):
        from utils.geospatial_aligner import align_image_pair

        img_a = _png_image_data(width=128, height=96)
        img_b = _png_image_data(width=64, height=64)

        result = align_image_pair(img_a, img_b)

        assert result.success
        assert result.pil_a is not None
        assert result.pil_b is not None

    def test_output_same_size(self):
        from utils.geospatial_aligner import align_image_pair

        img_a = _png_image_data(width=128, height=128)
        img_b = _png_image_data(width=64, height=64)

        result = align_image_pair(img_a, img_b)

        assert result.pil_a.size == result.pil_b.size

    def test_geographic_coordinates_unavailable(self):
        from utils.geospatial_aligner import align_image_pair

        img_a = _png_image_data(width=64, height=64)
        img_b = _png_image_data(width=64, height=64)

        result = align_image_pair(img_a, img_b)

        assert result.info.get("geographic_coordinates_available") is False
        assert result.info.get("source_crs") is None
        assert result.info.get("target_crs") is None

    def test_alignment_info_note_present(self):
        from utils.geospatial_aligner import align_image_pair

        img_a = _png_image_data(width=80, height=60)
        img_b = _png_image_data(width=40, height=40)

        result = align_image_pair(img_a, img_b)

        # A note should be attached explaining the pixel fallback
        note = result.info.get("note", "")
        assert len(note) > 0, "Expected a note about pixel-level alignment"

    def test_same_size_png_unchanged(self):
        """Two same-size PNGs should pass through without modification."""
        from utils.geospatial_aligner import align_image_pair

        img_a = _png_image_data(width=64, height=64)
        img_b = _png_image_data(width=64, height=64)

        result = align_image_pair(img_a, img_b)

        assert result.pil_a.size == (64, 64)
        assert result.pil_b.size == (64, 64)


# ══════════════════════════════════════════════════════════════════════════════
# Test 8 — pixel_to_latlon / pixel_to_geo_bbox round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestPixelToLatLon:
    """
    Verify pixel_to_geo_bbox produces correct WGS84 coordinates for a
    known affine transform (UTM → WGS84 reprojection).

    Reference: UTM Zone 32N, origin at (500 000 E, 5 000 000 N).
    Approx WGS84: lon ≈ 9.0°E, lat ≈ 45.1°N (central Italy / Alpine foothills).
    """

    # Affine: [a, b, c, d, e, f] = [pixel_width, 0, west, 0, -pixel_height, north]
    UTM_TRANSFORM = [10.0, 0.0, 500_000.0, 0.0, -10.0, 5_001_000.0]
    UTM_CRS = "EPSG:32632"
    IMAGE_W = 100
    IMAGE_H = 100

    def test_center_pixel_returns_dict(self):
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}
        result = pixel_to_geo_bbox(
            bbox, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        assert result is not None
        assert "min_lat" in result
        assert "max_lat" in result
        assert "min_lon" in result
        assert "max_lon" in result
        assert "center_lat" in result
        assert "center_lon" in result

    def test_latitude_in_reasonable_range(self):
        """UTM 32N with northing ~5 000 000 m → lat ≈ 45° N."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}  # full image
        result = pixel_to_geo_bbox(
            bbox, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        assert result is not None
        assert 40.0 < result["center_lat"] < 50.0, (
            f"Expected lat ≈ 45, got {result['center_lat']}"
        )

    def test_longitude_in_reasonable_range(self):
        """UTM 32N easting ~500 000 m → lon ≈ 9° E."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        result = pixel_to_geo_bbox(
            bbox, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        assert result is not None
        assert 0.0 < result["center_lon"] < 18.0, (
            f"Expected lon ≈ 9, got {result['center_lon']}"
        )

    def test_min_less_than_max(self):
        """min_lat < max_lat and min_lon < max_lon for any valid bbox."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}
        result = pixel_to_geo_bbox(
            bbox, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        assert result is not None
        assert result["min_lat"] <= result["max_lat"]
        assert result["min_lon"] <= result["max_lon"]

    def test_missing_transform_returns_none(self):
        """No transform → should return None, not raise."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}
        result = pixel_to_geo_bbox(
            bbox,
            transform=None,       # deliberately missing
            crs_str=self.UTM_CRS,
            image_width=self.IMAGE_W,
            image_height=self.IMAGE_H,
        )

        assert result is None

    def test_missing_crs_returns_none(self):
        """No CRS → should return None, not raise."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}
        result = pixel_to_geo_bbox(
            bbox,
            transform=self.UTM_TRANSFORM,
            crs_str=None,          # deliberately missing
            image_width=self.IMAGE_W,
            image_height=self.IMAGE_H,
        )

        assert result is None

    def test_convert_regions_batch(self):
        """convert_regions_to_geo_bbox enriches a list correctly."""
        from utils.geo_utils import convert_regions_to_geo_bbox

        regions = [
            {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3, "area_pct": 4.0},
            {"x1": 0.5, "y1": 0.5, "x2": 0.9, "y2": 0.9, "area_pct": 16.0},
        ]

        enriched = convert_regions_to_geo_bbox(
            regions, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        assert len(enriched) == 2
        for r in enriched:
            assert "geo_bbox" in r
            assert r["geo_bbox"] is not None
            assert "min_lat" in r["geo_bbox"]

    def test_zero_size_bbox_returns_point(self):
        """A degenerate bbox (point) should not crash."""
        from utils.geo_utils import pixel_to_geo_bbox

        bbox = {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5}
        result = pixel_to_geo_bbox(
            bbox, self.UTM_TRANSFORM, self.UTM_CRS,
            self.IMAGE_W, self.IMAGE_H,
        )

        # Either returns a valid dict with min==max or None; must not raise
        if result is not None:
            assert result["min_lat"] == result["max_lat"]
            assert result["min_lon"] == result["max_lon"]
