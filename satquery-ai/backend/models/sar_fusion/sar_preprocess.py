"""
SatQuery AI — Physically-meaningful SAR preprocessing
=====================================================
SAR (e.g. Sentinel-1) imagery is fundamentally different from optical imagery:
values are backscatter intensities, not reflectances, and the useful
representation is usually the **log domain (dB)**.  This module turns a raw SAR
array into a normalised, channel-consistent tensor for the fusion encoder
*without pretending the data is something it is not*.

Design constraints (honesty first)
----------------------------------
  * **Do not assume fixed channels.**  SAR sources vary: dual-pol VV+VH,
    single-pol VV, or a single unnamed band.  We detect what is present from
    the array shape + metadata rather than hardcoding a band count.
  * **VV / VH / VV/VH ratio only where available.**  The cross-pol ratio is
    a genuine, widely-used discriminator (built-up vs vegetation vs water), but
    it can only be formed when *both* co-pol and cross-pol bands exist.  When
    they do not, we say so in the returned metadata instead of fabricating it.
  * **Log-domain where appropriate.**  If the values look like linear intensity
    (all positive, wide dynamic range) we convert with ``10*log10``.  If they
    already look like dB (negative values present) we leave them.  Metadata can
    force either decision (``units: "linear" | "db"``).
  * **Do not claim calibrated backscatter** unless the metadata supports it
    (``calibrated: True``).  The returned analytics flag this explicitly.

Public API
----------
  * :func:`preprocess_sar` — array + metadata → (tensor[C,H,W], result-meta)
  * :func:`sar_backscatter_stats` — per-polarisation dB statistics (evidence)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("satquery.sar_preprocess")

# BigEarthNet-S1 dB normalisation statistics (VV, VH).  Reused from the
# training adapter so preprocessing matches the training distribution.
S1_DB_MEAN = np.array([-12.619, -20.256], dtype=np.float32)
S1_DB_STD = np.array([5.262, 6.248], dtype=np.float32)

# Floor for the log conversion so log10(0) never produces -inf.
_LINEAR_FLOOR = 1e-4


@dataclass
class SARPreprocessResult:
    """Structured, user-safe description of what preprocessing actually did."""

    tensor: np.ndarray                       # [C, H, W] float32, normalised
    polarizations: List[str]                 # e.g. ["VV", "VH"] or ["VV"] or ["band0"]
    channels: List[str]                      # actual output channels (may add "VV/VH")
    ratio_available: bool = False            # cross-pol ratio formed?
    log_applied: bool = False                # was 10*log10 applied?
    calibrated: bool = False                 # calibrated backscatter per metadata?
    note: Optional[str] = None

    def to_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "polarizations": self.polarizations,
            "channels": self.channels,
            "ratio_available": self.ratio_available,
            "log_applied": self.log_applied,
            "calibrated": self.calibrated,
        }
        if self.note:
            out["note"] = self.note
        return out


# ── Channel/polarisation detection ─────────────────────────────────────────────

def _to_chw(array: np.ndarray) -> np.ndarray:
    """
    Normalise input array to [C, H, W] float32 WITHOUT assuming a fixed C.

    Accepts [H, W] (single band), [H, W, C] (channel-last, common for our
    image_data dicts), or [C, H, W] (channel-first).  Disambiguation: when a
    3-D array's last axis is small (<= 4) and the first axis is large, treat it
    as channel-last; when the first axis is small, treat it as channel-first.
    """
    a = np.asarray(array, dtype=np.float32)
    if a.ndim == 2:
        return a[None, ...]                          # [1, H, W]
    if a.ndim != 3:
        raise ValueError(f"SAR array must be 2-D or 3-D, got shape {a.shape}")

    c0, c1, c2 = a.shape
    # Heuristic: SAR has few bands (1-4). Pick the small axis as channels.
    if c2 <= 4 and c2 <= c0:
        return np.transpose(a, (2, 0, 1))            # channel-last → [C, H, W]
    if c0 <= 4 and c0 <= c2:
        return a                                     # already channel-first
    # Ambiguous (both large) — assume channel-last, the app's dict convention.
    return np.transpose(a, (2, 0, 1))


def _detect_polarizations(n_channels: int, metadata: Dict[str, Any]) -> List[str]:
    """
    Determine polarisation labels from metadata, else fall back to conventions.

    We NEVER invent VV/VH names when the source is a single unnamed band —
    that would imply information we do not have.
    """
    meta = metadata or {}
    # Explicit metadata wins.
    for key in ("polarizations", "polarisations", "bands", "pols"):
        val = meta.get(key)
        if isinstance(val, (list, tuple)) and len(val) == n_channels:
            return [str(v).upper() for v in val]

    # Sentinel-1 dual-pol convention is the common paired case.
    if n_channels == 2:
        return ["VV", "VH"]
    if n_channels == 1:
        # A single band: honestly label it generically unless metadata says VV.
        single = meta.get("polarization") or meta.get("polarisation")
        return [str(single).upper()] if single else ["band0"]
    return [f"band{i}" for i in range(n_channels)]


def _should_log(chw: np.ndarray, metadata: Dict[str, Any]) -> bool:
    """
    Decide whether to apply 10*log10 (linear → dB).

    Metadata ``units`` forces the decision.  Otherwise: if the data already has
    negative values it is almost certainly already dB; if it is all-positive
    with a wide dynamic range it is linear intensity → convert.
    """
    units = str((metadata or {}).get("units", "")).lower()
    if units in ("db", "decibel", "decibels"):
        return False
    if units in ("linear", "intensity", "amplitude", "power"):
        return True
    finite = chw[np.isfinite(chw)]
    if finite.size == 0:
        return False
    # Already dB if meaningfully negative values are present.
    if float(finite.min()) < -1.0:
        return False
    # All-positive → treat as linear intensity and move to dB.
    return True


# ── Public: preprocessing ───────────────────────────────────────────────────────

def preprocess_sar(
    array: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    out_channels: int = 3,
) -> SARPreprocessResult:
    """
    Turn a raw SAR array into a normalised ``[out_channels, H, W]`` tensor.

    Parameters
    ----------
    array : np.ndarray
        SAR data as [H, W], [H, W, C] or [C, H, W].  Channel count is NOT
        assumed — it is detected.
    metadata : dict, optional
        Sensor metadata.  Recognised keys (all optional):
          ``polarizations`` (list, e.g. ["VV", "VH"]), ``units``
          ("linear" | "db"), ``calibrated`` (bool).
    out_channels : int
        Number of channels to emit (default 3 to feed a 3-channel CNN stem).
        Built as [VV, VH, VV/VH] when dual-pol; otherwise the available
        channels are tiled honestly to fill the width.

    Returns
    -------
    SARPreprocessResult
        ``.tensor`` is the normalised array; ``.to_meta()`` documents exactly
        what was done (so nothing is silently fabricated).
    """
    metadata = metadata or {}
    chw = _to_chw(array)
    n = chw.shape[0]
    pols = _detect_polarizations(n, metadata)

    # ── 1. Log domain (dB) where appropriate ──────────────────────────────────
    log_applied = _should_log(chw, metadata)
    if log_applied:
        chw = 10.0 * np.log10(np.clip(chw, _LINEAR_FLOOR, None))

    # ── 2. Build physically-meaningful channels ───────────────────────────────
    pol_index = {p: i for i, p in enumerate(pols)}
    have_vv = "VV" in pol_index
    have_vh = "VH" in pol_index
    ratio_available = have_vv and have_vh

    channels: List[np.ndarray] = []
    channel_names: List[str] = []

    if ratio_available:
        vv = chw[pol_index["VV"]]
        vh = chw[pol_index["VH"]]
        # In dB the cross-pol ratio is a difference (log(VV/VH) = VV_dB - VH_dB).
        ratio = (vv - vh) if log_applied else (vv / (vh + 1e-6))
        # Normalise VV/VH with the BigEarthNet dB stats; standardise the ratio.
        vv_n = (vv - S1_DB_MEAN[0]) / (S1_DB_STD[0] + 1e-6)
        vh_n = (vh - S1_DB_MEAN[1]) / (S1_DB_STD[1] + 1e-6)
        ratio_n = _standardize(ratio)
        channels = [vv_n, vh_n, ratio_n]
        channel_names = ["VV", "VH", "VV/VH"]
        note = "dual-pol VV/VH with cross-pol ratio"
    else:
        # Single-pol or unnamed: standardise each available band; NO fake ratio.
        for i in range(n):
            channels.append(_standardize(chw[i]))
            channel_names.append(pols[i] if i < len(pols) else f"band{i}")
        note = (
            "single-band SAR — cross-pol ratio unavailable"
            if n == 1 else
            f"{n}-band SAR — cross-pol ratio unavailable (need both VV and VH)"
        )

    stacked = np.stack(channels, axis=0).astype(np.float32)

    # ── 3. Fit to the requested output channel count (honest tiling) ───────────
    stacked = _fit_channels(stacked, out_channels)

    calibrated = bool(metadata.get("calibrated", False))
    return SARPreprocessResult(
        tensor=stacked,
        polarizations=pols,
        channels=channel_names,
        ratio_available=ratio_available,
        log_applied=log_applied,
        calibrated=calibrated,
        note=note,
    )


def _standardize(band: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std standardisation over finite pixels."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    mu = float(finite.mean())
    sd = float(finite.std())
    out = (band - mu) / (sd + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _fit_channels(stacked: np.ndarray, out_channels: int) -> np.ndarray:
    """Tile / truncate channel axis to exactly out_channels (no fabrication)."""
    c = stacked.shape[0]
    if c == out_channels:
        return stacked
    if c > out_channels:
        return stacked[:out_channels]
    # Repeat existing channels to fill — a documented replication, not new data.
    reps = [stacked[i % c] for i in range(out_channels)]
    return np.stack(reps, axis=0).astype(np.float32)


# ── Public: backscatter statistics (deterministic evidence) ─────────────────────

def sar_backscatter_stats(
    array: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute per-polarisation SAR backscatter statistics as structured evidence.

    Returns dB-domain mean/std per polarisation, the VV/VH ratio (dB
    difference) when both are present, and honest flags describing what the
    numbers represent.  Never claims calibration the metadata does not support.
    """
    metadata = metadata or {}
    chw = _to_chw(array)
    n = chw.shape[0]
    pols = _detect_polarizations(n, metadata)
    log_applied = _should_log(chw, metadata)
    db = 10.0 * np.log10(np.clip(chw, _LINEAR_FLOOR, None)) if log_applied else chw

    per_pol: Dict[str, Dict[str, float]] = {}
    for i, pol in enumerate(pols):
        band = db[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            per_pol[pol] = {"mean_db": 0.0, "std_db": 0.0}
        else:
            per_pol[pol] = {
                "mean_db": round(float(finite.mean()), 3),
                "std_db": round(float(finite.std()), 3),
            }

    stats: Dict[str, Any] = {
        "polarizations": pols,
        "per_polarization": per_pol,
        "units": "dB",
        "log_applied": log_applied,
        "calibrated": bool(metadata.get("calibrated", False)),
    }

    if "VV" in per_pol and "VH" in per_pol:
        vv_mean = per_pol["VV"]["mean_db"]
        vh_mean = per_pol["VH"]["mean_db"]
        # In dB the ratio is the difference of means.
        stats["vv_vh_ratio_db"] = round(vv_mean - vh_mean, 3)
        stats["ratio_available"] = True
        # High cross-pol return (small VV-VH gap) tends to indicate volume
        # scattering (vegetation); a large gap tends toward surface/urban.
        stats["dominant_scattering"] = (
            "surface / built-up (high VV-VH contrast)"
            if (vv_mean - vh_mean) > 6.0
            else "volume / vegetation (low VV-VH contrast)"
        )
    else:
        stats["ratio_available"] = False
        stats["note"] = "cross-pol ratio unavailable — single/unknown polarisation"

    return stats
