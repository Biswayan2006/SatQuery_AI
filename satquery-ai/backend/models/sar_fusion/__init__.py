"""
SatQuery AI — SAR-Optical fusion support package.

Holds the physically-meaningful SAR preprocessing used by the fusion model.
The fusion model itself lives in ``models/sar_fusion_model.py`` (kept there for
backward-compatible imports).
"""
from models.sar_fusion.sar_preprocess import (  # noqa: F401
    SARPreprocessResult,
    preprocess_sar,
    sar_backscatter_stats,
)
