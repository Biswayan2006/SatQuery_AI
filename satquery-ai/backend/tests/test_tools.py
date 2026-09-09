"""
Synthetic-raster tests for the deterministic tool layer.

Every tool is exercised against hand-constructed numpy rasters with KNOWN
outputs — we never rely on AI model outputs for these calculations.  Covered:

  * NDVI / NDWI / NDBI: correct value on known bands; unsupported when a band
    is missing; unsupported when no valid pixels; sensor-specific band mapping.
  * spectral_statistics: per-band stats on named + positional bands.
  * SAR: intensity stats + VV/VH ratio; calibrated=False by default;
    calibrated=True only with a metadata marker.
  * change_area: pixel count, percentage, area with/without resolution.
  * connected_region_statistics: region count, per-region area/bbox.
  * pixel_to_latlon / image_bounds / ground_resolution: geo math with a known
    affine transform; unsupported without CRS+transform.
  * registry: unknown tool → error; tools never raise into the request path.
"""
import os
import sys

import numpy as np
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from tools import RasterInput, ToolStatus, get_default_registry  # noqa: E402
from tools.change import ChangeAreaTool, ConnectedRegionStatisticsTool  # noqa: E402
from tools.geolocation import (  # noqa: E402
    GroundResolutionTool,
    ImageBoundsTool,
    PixelToLatLonTool,
)
from tools.sar import SARBackscatterStatisticsTool  # noqa: E402
from tools.spectral import NDBITool, NDVITool, NDWITool, SpectralStatisticsTool  # noqa: E402


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def make_rgbn(red, green, blue, nir, h=8, w=8, **kw):
    arr = np.zeros((h, w, 4), dtype=np.float64)
    arr[..., 0] = red
    arr[..., 1] = green
    arr[..., 2] = blue
    arr[..., 3] = nir
    return RasterInput(array=arr, sensor="rgbn", modality="optical", **kw)


# North-up affine: 10 m pixels, origin at a projected easting/northing.
# EPSG:32633 (UTM 33N) is a projected metric CRS.
UTM_TRANSFORM = [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0]
UTM_CRS = "EPSG:32633"


# ── Spectral indices ─────────────────────────────────────────────────────────

def test_ndvi_known_value():
    # nir=0.8 red=0.2 -> (0.8-0.2)/(0.8+0.2)=0.6
    r = make_rgbn(red=0.2, green=0.3, blue=0.1, nir=0.8)
    res = NDVITool()(r)
    assert res.status == ToolStatus.SUCCESS
    d = res.to_dict()
    assert d["mean"] == pytest.approx(0.6, abs=1e-6)
    assert d["min"] == pytest.approx(0.6, abs=1e-6)
    assert d["max"] == pytest.approx(0.6, abs=1e-6)
    assert d["valid_pixel_percentage"] == 100.0
    assert d["count"] == 64


def test_ndwi_known_value():
    # green=0.3 nir=0.8 -> (0.3-0.8)/(0.3+0.8) = -0.5/1.1 = -0.4545...
    r = make_rgbn(red=0.2, green=0.3, blue=0.1, nir=0.8)
    res = NDWITool()(r)
    assert res.ok
    assert res.to_dict()["mean"] == pytest.approx(-0.5 / 1.1, abs=1e-4)


def test_ndbi_supported_on_sentinel2_bandmap():
    # Build a 13-band stack; sentinel-2 preset resolves swir1=11, nir=7.
    arr = np.zeros((4, 4, 13), dtype=np.float64)
    arr[..., 7] = 0.4   # nir
    arr[..., 11] = 0.6  # swir1
    r = RasterInput(array=arr, sensor="sentinel-2", modality="multispectral")
    res = NDBITool()(r)
    assert res.ok
    # (0.6-0.4)/(0.6+0.4)=0.2
    assert res.to_dict()["mean"] == pytest.approx(0.2, abs=1e-6)


def test_ndvi_unsupported_without_nir():
    # Plain 3-band RGB → no NIR band.
    arr = np.zeros((4, 4, 3), dtype=np.float64)
    r = RasterInput(array=arr, sensor="rgb", modality="optical")
    res = NDVITool()(r)
    assert res.status == ToolStatus.UNSUPPORTED
    d = res.to_dict()
    assert "nir" in d["missing_bands"]
    assert d["required_bands"] == ["nir", "red"]


def test_ndbi_unsupported_without_swir():
    r = make_rgbn(red=0.2, green=0.3, blue=0.1, nir=0.8)
    res = NDBITool()(r)
    assert res.status == ToolStatus.UNSUPPORTED
    assert "swir1" in res.to_dict()["missing_bands"]


def test_ndvi_unsupported_when_no_valid_pixels():
    # All-nodata raster → no finite pixels after band read.
    arr = np.full((4, 4, 4), -9999.0, dtype=np.float64)
    r = RasterInput(array=arr, sensor="rgbn", modality="optical", nodata=-9999.0)
    res = NDVITool()(r)
    assert res.status == ToolStatus.UNSUPPORTED
    assert res.to_dict()["valid_pixel_percentage"] == 0.0


def test_ndvi_explicit_band_map_overrides_sensor():
    # Put NIR at index 0, RED at index 1 via explicit band_map.
    arr = np.zeros((4, 4, 4), dtype=np.float64)
    arr[..., 0] = 0.9  # nir
    arr[..., 1] = 0.1  # red
    r = RasterInput(array=arr, band_map={"nir": 0, "red": 1}, modality="optical")
    res = NDVITool()(r)
    assert res.ok
    assert res.to_dict()["mean"] == pytest.approx(0.8, abs=1e-6)  # (.9-.1)/(.9+.1)


def test_ndvi_nodata_reduces_valid_percentage():
    arr = np.zeros((10, 10, 4), dtype=np.float64)
    arr[..., 0] = 0.2  # red
    arr[..., 3] = 0.8  # nir
    # Mark half the RED pixels as nodata.
    arr[:5, :, 0] = -1.0
    r = RasterInput(array=arr, sensor="rgbn", modality="optical", nodata=-1.0)
    res = NDVITool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["valid_pixel_percentage"] == pytest.approx(50.0, abs=0.1)
    assert d["count"] == 50


# ── Spectral statistics ────────────────────────────────────────────────────────

def test_spectral_statistics_named_bands():
    r = make_rgbn(red=0.2, green=0.4, blue=0.1, nir=0.8)
    res = SpectralStatisticsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["n_bands"] == 4
    assert d["bands"]["red"]["mean"] == pytest.approx(0.2, abs=1e-6)
    assert d["bands"]["nir"]["mean"] == pytest.approx(0.8, abs=1e-6)


def test_spectral_statistics_positional_when_unresolved():
    # 5-band stack, no sensor → no logical names resolve → positional reporting.
    arr = np.zeros((4, 4, 5), dtype=np.float64)
    for i in range(5):
        arr[..., i] = float(i)
    r = RasterInput(array=arr, modality="multispectral")
    res = SpectralStatisticsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert "band_0" in d["bands"]
    assert d["bands"]["band_3"]["mean"] == pytest.approx(3.0, abs=1e-6)


# ── SAR ─────────────────────────────────────────────────────────────────────────

def test_sar_uncalibrated_by_default():
    arr = np.zeros((6, 6, 2), dtype=np.float64)
    arr[..., 0] = 100.0  # vv
    arr[..., 1] = 25.0   # vh
    r = RasterInput(array=arr, sensor="sentinel-1", modality="sar")
    res = SARBackscatterStatisticsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["calibrated"] is False
    assert d["unit"] == "intensity_dn"
    # VV/VH ratio in linear space = 100/25 = 4.0
    assert d["vv_vh_ratio"] == pytest.approx(4.0, abs=1e-6)
    assert d["per_polarization"]["vv"]["mean"] == pytest.approx(100.0, abs=1e-6)


def test_sar_calibrated_only_with_metadata_marker():
    arr = np.zeros((4, 4, 2), dtype=np.float64)
    arr[..., 0] = 10.0
    arr[..., 1] = 5.0
    r = RasterInput(
        array=arr, sensor="sentinel-1", modality="sar",
        metadata={"calibrated": True, "unit": "sigma0"},
    )
    res = SARBackscatterStatisticsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["calibrated"] is True
    assert d["unit"] == "sigma0"


def test_sar_unnamed_channels_reported_positionally():
    # Single-channel SAR with no polarization label.
    arr = np.full((4, 4, 1), 50.0, dtype=np.float64)
    r = RasterInput(array=arr, modality="sar")
    res = SARBackscatterStatisticsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["polarizations"] is None
    assert d["per_channel"]["channel_0"]["mean"] == pytest.approx(50.0, abs=1e-6)
    assert d["calibrated"] is False


# ── Change area ──────────────────────────────────────────────────────────────

def test_change_area_counts_and_percentage():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :] = True  # 20 of 100 pixels changed
    res = ChangeAreaTool()(mask)
    assert res.ok
    d = res.to_dict()
    assert d["changed_pixels"] == 20
    assert d["total_pixels"] == 100
    assert d["changed_percentage"] == pytest.approx(20.0, abs=1e-6)
    assert d["area_m2"] is None  # no resolution supplied


def test_change_area_with_resolution():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :] = True  # 20 pixels
    res = ChangeAreaTool()(mask, pixel_resolution_m=10.0)
    assert res.ok
    d = res.to_dict()
    # 20 pixels * (10 m)^2 = 2000 m^2
    assert d["area_m2"] == pytest.approx(2000.0, abs=1e-6)
    assert d["area_km2"] == pytest.approx(0.002, abs=1e-9)
    assert d["area_is_approximate"] is True


def test_change_area_with_valid_mask_denominator():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :] = True  # 20 changed
    valid = np.zeros((10, 10), dtype=bool)
    valid[:5, :] = True  # only 50 valid pixels
    res = ChangeAreaTool()(mask, valid_mask=valid)
    assert res.ok
    d = res.to_dict()
    assert d["valid_pixels"] == 50
    assert d["changed_pixels"] == 20  # all changed pixels are within valid
    assert d["changed_percentage"] == pytest.approx(40.0, abs=1e-6)


def test_change_area_valid_mask_shape_mismatch():
    mask = np.zeros((10, 10), dtype=bool)
    valid = np.zeros((5, 5), dtype=bool)
    res = ChangeAreaTool()(mask, valid_mask=valid)
    assert res.status == ToolStatus.UNSUPPORTED


# ── Connected regions ─────────────────────────────────────────────────────────

def test_connected_regions_two_blobs():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1     # 3x3 = 9 px
    mask[10:14, 10:16] = 1  # 4x6 = 24 px
    res = ConnectedRegionStatisticsTool()(mask)
    assert res.ok
    d = res.to_dict()
    assert d["region_count"] == 2
    assert d["largest_region_pixels"] == 24
    # largest first
    assert d["regions"][0]["area_pixels"] == 24
    assert d["regions"][1]["area_pixels"] == 9


def test_connected_regions_area_and_bbox():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 2:5] = 1  # 3x3 block at (y=1..3, x=2..4)
    res = ConnectedRegionStatisticsTool()(mask, pixel_resolution_m=5.0)
    assert res.ok
    region = res.to_dict()["regions"][0]
    assert region["area_pixels"] == 9
    assert region["area_m2"] == pytest.approx(9 * 25.0, abs=1e-6)
    bbox = region["bbox_pixels"]
    assert bbox["x1"] == 2 and bbox["y1"] == 1
    assert bbox["x2"] == 5 and bbox["y2"] == 4


def test_connected_regions_min_area_filter():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1          # 1 px (filtered)
    mask[5:8, 5:8] = 1      # 9 px (kept)
    res = ConnectedRegionStatisticsTool()(mask, min_area_pixels=5)
    assert res.ok
    assert res.to_dict()["region_count"] == 1


# ── Geolocation ────────────────────────────────────────────────────────────────

def test_pixel_to_latlon_requires_geo():
    r = make_rgbn(0.2, 0.3, 0.1, 0.8)  # no CRS/transform
    res = PixelToLatLonTool()(r, pixel_x=1, pixel_y=1)
    assert res.status == ToolStatus.UNSUPPORTED


def test_pixel_to_latlon_projected_crs():
    arr = np.zeros((100, 100, 4), dtype=np.float64)
    r = RasterInput(array=arr, sensor="rgbn", modality="optical",
                    crs=UTM_CRS, transform=UTM_TRANSFORM)
    res = PixelToLatLonTool()(r, pixel_x=0, pixel_y=0)
    assert res.ok
    geo = res.to_dict()["geographic"]
    # UTM 33N easting 500000 is on the central meridian (15°E); northing 4,000,000
    # is ~36.1°N.  Pixel-centre offset is tiny.
    assert geo["lon"] == pytest.approx(15.0, abs=0.01)
    assert 35.0 < geo["lat"] < 37.0


def test_image_bounds_projected_crs():
    arr = np.zeros((50, 50, 4), dtype=np.float64)
    r = RasterInput(array=arr, sensor="rgbn", modality="optical",
                    crs=UTM_CRS, transform=UTM_TRANSFORM)
    res = ImageBoundsTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["west"] < d["east"]
    assert d["south"] < d["north"]
    assert d["width_px"] == 50 and d["height_px"] == 50


def test_ground_resolution_projected_is_pixel_size():
    arr = np.zeros((10, 10, 4), dtype=np.float64)
    r = RasterInput(array=arr, sensor="rgbn", modality="optical",
                    crs=UTM_CRS, transform=UTM_TRANSFORM)
    res = GroundResolutionTool()(r)
    assert res.ok
    d = res.to_dict()
    assert d["resolution_x_m"] == pytest.approx(10.0, abs=1e-6)
    assert d["resolution_y_m"] == pytest.approx(10.0, abs=1e-6)


def test_ground_resolution_requires_geo():
    r = make_rgbn(0.2, 0.3, 0.1, 0.8)
    res = GroundResolutionTool()(r)
    assert res.status == ToolStatus.UNSUPPORTED


# ── Registry ────────────────────────────────────────────────────────────────

def test_registry_has_all_ten_tools():
    reg = get_default_registry()
    expected = {
        "NDVI", "NDWI", "NDBI", "spectral_statistics",
        "sar_backscatter_statistics", "change_area", "pixel_to_latlon",
        "image_bounds", "ground_resolution", "connected_region_statistics",
    }
    assert expected.issubset(set(reg.names()))
    assert len(reg.names()) >= 10


def test_registry_unknown_tool_errors():
    reg = get_default_registry()
    res = reg.run("does_not_exist", None)
    assert res.status == ToolStatus.ERROR


def test_tool_never_raises_into_request_path():
    # Passing garbage into NDVI must yield an error result, not an exception.
    reg = get_default_registry()
    res = reg.run("NDVI", "not a raster")
    assert res.status == ToolStatus.ERROR


def test_registry_run_ndvi_end_to_end():
    reg = get_default_registry()
    r = make_rgbn(red=0.2, green=0.3, blue=0.1, nir=0.8)
    d = reg.run("NDVI", r).to_dict()
    assert d["tool"] == "NDVI"
    assert d["status"] == "success"
    assert d["mean"] == pytest.approx(0.6, abs=1e-6)


# ── Tool planner: the two spec example plans ───────────────────────────────────

def _image_data_rgbn(nir, red, h=16, w=16):
    arr = np.zeros((h, w, 4), dtype=np.float64)
    arr[..., 0] = red
    arr[..., 1] = 0.3
    arr[..., 2] = 0.1
    arr[..., 3] = nir
    return {"numpy_array": arr, "modality": "optical", "metadata": {"sensor": "rgbn"}}


def _image_data_s2(nir, swir1, h=16, w=16):
    arr = np.zeros((h, w, 13), dtype=np.float64)
    arr[..., 7] = nir
    arr[..., 11] = swir1
    return {
        "numpy_array": arr,
        "modality": "multispectral",
        "metadata": {
            "sensor": "sentinel-2",
            "crs": UTM_CRS,
            "transform": UTM_TRANSFORM,
        },
    }


def test_plan_vegetation_decrease():
    """'Has vegetation decreased?' → NDVI before/after + change-area."""
    from agent.task_classifier import TaskType
    from agent.tool_planner import ToolPlanner

    tp = ToolPlanner()
    before = _image_data_rgbn(nir=0.8, red=0.2)  # NDVI 0.6
    after = _image_data_rgbn(nir=0.4, red=0.3)   # NDVI ~0.14
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:8, :] = 1
    ev = tp.run_for_task(
        TaskType.CHANGE_VQA, "Has vegetation decreased?", [before, after],
        raw_extras={"change_mask": mask, "pixel_resolution_m": 10.0},
    )
    tools = [e["tool"] for e in ev]
    assert tools.count("NDVI") == 2
    assert "change_area" in tools
    ndvis = [e for e in ev if e["tool"] == "NDVI"]
    assert ndvis[0]["mean"] == pytest.approx(0.6, abs=1e-3)
    assert ndvis[1]["mean"] < ndvis[0]["mean"]  # decreased
    ca = next(e for e in ev if e["tool"] == "change_area")
    assert ca["changed_pixels"] == 128
    assert ca["area_m2"] == pytest.approx(12800.0, abs=1e-3)


def test_plan_construction_expansion_grounds_location():
    """'Where has construction expanded?' → NDBI + regions + geolocated centroids."""
    from agent.task_classifier import TaskType
    from agent.tool_planner import ToolPlanner

    tp = ToolPlanner()
    before = _image_data_s2(nir=0.4, swir1=0.3)
    after = _image_data_s2(nir=0.3, swir1=0.6)
    cmask = np.zeros((16, 16), dtype=np.uint8)
    cmask[2:6, 2:6] = 1
    ev = tp.run_for_task(
        TaskType.CHANGE_VQA, "Where has construction expanded?", [before, after],
        raw_extras={"change_mask": cmask},
    )
    tools = [e["tool"] for e in ev]
    assert tools.count("NDBI") == 2
    assert "connected_region_statistics" in tools
    regions = next(e for e in ev if e["tool"] == "connected_region_statistics")
    assert regions["region_count"] == 1
    # Centroid was geolocated because the raster carries CRS + transform.
    geo = regions["regions"][0].get("geographic")
    assert geo is not None
    assert 35.0 < geo["lat"] < 37.0
    assert geo["lon"] == pytest.approx(15.0, abs=0.05)


def test_plan_single_vqa_vegetation_query_runs_ndvi():
    from agent.task_classifier import TaskType
    from agent.tool_planner import ToolPlanner

    tp = ToolPlanner()
    img = _image_data_rgbn(nir=0.8, red=0.2)
    ev = tp.run_for_task(TaskType.SINGLE_VQA, "How much vegetation is there?", [img])
    assert [e["tool"] for e in ev] == ["NDVI"]
    assert ev[0]["mean"] == pytest.approx(0.6, abs=1e-3)


def test_plan_skips_index_when_bands_absent():
    """Vegetation query on a plain RGB image adds no NDVI (no NIR band)."""
    from agent.task_classifier import TaskType
    from agent.tool_planner import ToolPlanner

    tp = ToolPlanner()
    arr = np.zeros((8, 8, 3), dtype=np.float64)
    img = {"numpy_array": arr, "modality": "optical", "metadata": {"sensor": "rgb"}}
    ev = tp.run_for_task(TaskType.SINGLE_VQA, "Is there vegetation?", [img])
    assert all(e["tool"] != "NDVI" for e in ev)
