"""
SatQuery AI — SAR-Optical Fusion Evaluation Runner
==================================================
    python -m evaluation.fusion.run --config evaluation/configs/evaluation.yaml

Land-cover multi-label classification on the **fused** SAR+optical tokens over
the BigEarthNet-S1/S2 *test* split (deterministic MD5 80/10/10 — test never
leaks into training).  Compares the **untrained** fusion adapter (baseline)
against a **trained** fusion checkpoint (adapted, ``SAR_FUSION_CKPT``).

Faithful to training
--------------------
The trained adapter is fed SAR exactly the way ``training/train_fusion.py`` fed
it: the dataset's already-dB-normalised 2-channel VV/VH tensor is turned into the
3-channel ``[VV, VH, VV/VH]`` stack by the SHARED ``_sar_to_fusion_channels``
helper and passed straight to ``adapter.encode_sar`` — NOT through the model's
raw-array ``encode_sar``/``preprocess_sar`` inference path, which would
double-normalise VV/VH and mismatch the checkpoint.  Optical tokens come from
the frozen BLIP vision tower via ``encode_optical`` (same processor/tower as
``OpticalTokenizer`` in training).

Honesty
-------
  * No ``BIGEARTHNET_DIR`` / no ``BigEarthNet-S2`` → explicit ``no_data``.
  * No paired ``BigEarthNet-S1`` → classification skipped (SAR would be zeros);
    only the wiring ``diagnose()`` is reported.
  * No BLIP injection pathway (non-BLIP backend / offline) → wiring only.
  * The **untrained** baseline is a randomly-initialised adapter: its numbers are
    real measurements but flagged ``requires_verification`` and never presented
    as trustworthy.
  * No ``SAR_FUSION_CKPT`` → adapted column is ``n/a`` (never fabricated).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json
from evaluation.common.variants import resolve_variants

logger = logging.getLogger("satquery.eval.fusion")

DOMAIN = "fusion"
TITLE = "SAR-optical fusion (land-cover on fused tokens)"
_REPORTED = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "mean_ap"]


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None (valid, honest JSON)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _no_data(output_dir: str, notes: List[str], prov=None, extra: Optional[Dict] = None) -> Dict[str, Any]:
    report = {"domain": DOMAIN, "status": "no_data", "dataset": "bigearthnet",
              "comparison": [], "notes": notes, "provenance": provenance(prov)}
    if extra:
        report.update(extra)
    write_json(output_dir, DOMAIN, _sanitize(report))
    append_markdown(output_dir, DOMAIN, TITLE, [], notes)
    logger.warning("Fusion: %s", notes[-1] if notes else "no data")
    return report


def _classify(model, ds, device: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run fused-token land-cover classification over the dataset.

    Returns (multilabel_metrics|None, n_samples).  Mirrors the training data
    path exactly (shared ``_sar_to_fusion_channels``); predictions are sigmoid
    probabilities so ``multilabel_metrics`` thresholding + mAP are meaningful.
    """
    import numpy as np
    import torch
    from PIL import Image

    from training.train_fusion import _sar_to_fusion_channels
    from training.evaluate import multilabel_metrics

    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    for i in range(len(ds)):
        item = ds[i]
        sar = item["metadata"].get("sar_tensor")
        if sar is None:
            continue
        img = item["image"]  # [3,H,W] float in [0,1]
        arr = img.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        opt_pil = Image.fromarray(arr, mode="RGB")
        try:
            with torch.no_grad():
                opt_tokens = model.encode_optical(opt_pil)                 # [1, N_o, H]
                sar_in = _sar_to_fusion_channels(sar.unsqueeze(0)).to(device)  # [1,3,H,W]
                sar_tokens = model.adapter.encode_sar(sar_in)              # [1, N_s, H]
                fused, _ = model.fuse_modalities(opt_tokens, sar_tokens)   # [1, N_o, H]
                logits = model.adapter.classify(fused)                     # [1, 43]
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        except Exception as exc:
            logger.debug("fusion classify failed on sample %d: %s", i, exc)
            continue
        y_true.append(item["metadata"]["label_tensor"].detach().cpu().numpy())
        y_prob.append(probs)

    if not y_true:
        return None, 0
    metrics = multilabel_metrics(np.stack(y_true), np.stack(y_prob))
    return _sanitize(metrics), len(y_true)


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    fcfg = cfg.get("fusion", {})
    ds_cfg = (fcfg.get("datasets", {}) or {}).get("bigearthnet", {})
    data_dir = ds_cfg.get("data_dir")
    model_cfg = fcfg.get("model", {})
    notes: List[str] = []

    if not is_configured(data_dir):
        notes.append("skipped: fusion dataset not configured (set BIGEARTHNET_DIR).")
        return _no_data(output_dir, notes)
    if not os.path.isdir(os.path.join(data_dir, "BigEarthNet-S2")):
        notes.append(f"skipped: no BigEarthNet-S2/ under {data_dir!r}.")
        return _no_data(output_dir, notes, prov=[{"name": "bigearthnet", "data_dir": data_dir}])

    limit = max_samples if max_samples is not None else ds_cfg.get("max_samples")
    s1_present = os.path.isdir(os.path.join(data_dir, "BigEarthNet-S1"))
    variants = resolve_variants(
        cfg, DOMAIN,
        baseline_label="Untrained fusion adapter", adapted_label="Trained fusion (SatQuery)",
        adapted_needs_checkpoint=True,
    )
    adapted_v = next((v for v in variants if v.kind == "adapted"), None)
    adapted_ckpt = adapted_v.checkpoint if (adapted_v and adapted_v.available) else None
    if adapted_v is not None and not adapted_v.available:
        notes.append(f"adapted: {adapted_v.reason}")

    # Build the fusion model ONCE (single BLIP load); the untrained adapter is
    # the baseline, then we swap in the trained checkpoint in place for adapted.
    from models.sar_fusion_model import SAROpticalFusionModel
    try:
        model = SAROpticalFusionModel(
            device=device,
            cache_dir=model_cfg.get("cache_dir"),
            vqa_model_name=model_cfg.get("vqa_model_name", "Salesforce/blip-vqa-base"),
            fusion_checkpoint=None,
        )
    except Exception as exc:
        notes.append(f"skipped: could not construct fusion model ({type(exc).__name__}: {exc}).")
        return _no_data(output_dir, notes, prov=[{"name": "bigearthnet", "data_dir": data_dir}])

    baseline_diag = model.diagnose()
    can_classify = bool(model._injection_supported and s1_present)
    if not model._injection_supported:
        notes.append("classification skipped: BLIP visual-token injection pathway unavailable "
                     "(non-BLIP backend or model offline) — reporting pathway wiring only.")
    elif not s1_present:
        notes.append(f"classification skipped: no BigEarthNet-S1/ under {data_dir!r} "
                     "(paired SAR required; adapter SAR would be all-zeros).")

    from training.datasets.bigearthnet import BigEarthNetAdapter

    base_metrics = adpt_metrics = None
    adapted_diag = None
    n_used = 0
    if can_classify:
        ds = BigEarthNetAdapter(data_dir=data_dir, split="test", use_sar=True,
                                image_size=int(ds_cfg.get("image_size", 120)), max_samples=limit)
        if len(ds) == 0:
            notes.append("classification skipped: BigEarthNet test split empty "
                         "(no *_labels_metadata.json patches).")
        else:
            base_metrics, n_used = _classify(model, ds, device)   # untrained baseline
            if adapted_ckpt:
                if model.load_fusion_checkpoint(adapted_ckpt):
                    adapted_diag = model.diagnose()
                    adpt_metrics, n_used = _classify(model, ds, device)
                else:
                    notes.append(f"adapted: checkpoint {adapted_ckpt} failed to load — staying baseline-only.")

    rows: List[Dict[str, Any]] = []
    if base_metrics is not None:
        for m in _REPORTED:
            rows.append(comparison_row("SAR-optical fusion", "bigearthnet", m,
                                       base_metrics.get(m), (adpt_metrics or {}).get(m)))
        notes.append(f"Land-cover multi-label metrics on FUSED tokens over {n_used} BigEarthNet "
                     "test patches (predictions = sigmoid probabilities).")
        if adpt_metrics is None:
            notes.append("Baseline = UNTRAINED fusion adapter (random init): numbers are real but "
                         "NOT trustworthy — set SAR_FUSION_CKPT for a trained comparison. "
                         "adapted = n/a.")
        else:
            notes.append("Baseline = untrained adapter; adapted = trained fusion checkpoint.")

    # Model constructed (else we'd have returned no_data) → wiring is always
    # reported; classification metrics attach when data + pathway allow.
    status = "ok"
    report = {
        "domain": DOMAIN,
        "status": status,
        "dataset": "bigearthnet",
        "s1_present": s1_present,
        "diagnose": {"baseline": baseline_diag, "adapted": adapted_diag},
        "fusion_trained": model.fusion_trained,
        "requires_verification": bool(base_metrics is not None and not model.fusion_trained
                                      and adpt_metrics is None),
        "variants": {
            "baseline": {"label": "Untrained fusion adapter", "metrics": base_metrics,
                         "requires_verification": True},
            "adapted": ({"label": "Trained fusion (SatQuery)", "metrics": adpt_metrics,
                         "checkpoint": adapted_ckpt} if adapted_ckpt else None),
        },
        "n_samples": n_used,
        "comparison": rows,
        "notes": notes,
        "provenance": provenance([{"name": "bigearthnet", "data_dir": data_dir, "n_samples": n_used}]),
    }
    write_json(output_dir, DOMAIN, _sanitize(report))
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    if base_metrics is not None:
        logger.info("Fusion macro_f1 baseline=%s adapted=%s (n=%d, trained=%s)",
                    base_metrics.get("macro_f1"),
                    (adpt_metrics or {}).get("macro_f1", "n/a"), n_used, model.fusion_trained)
    else:
        logger.info("Fusion: wiring reported (injection_supported=%s, s1_present=%s)",
                    model._injection_supported, s1_present)
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
