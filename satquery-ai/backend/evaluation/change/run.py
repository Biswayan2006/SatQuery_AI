"""
SatQuery AI — Change-Detection Evaluation Runner
================================================
    python -m evaluation.change.run --config evaluation/configs/evaluation.yaml

Two tracks, each honestly skipped when its dataset is absent:

  * **mask**       — pixel change-mask metrics (precision / recall / F1 / IoU /
                     overall accuracy) from ``ChangeDetectionModel.detect_changes``
                     vs a LEVIR-CD-style ``A/ B/ label/`` triplet dataset.
  * **change-VQA** — exact-match / token-F1 from ``answer_change_question`` over
                     the bi-temporal CDVQA adapter.

Baseline vs adapted (mask track)
--------------------------------
Same siamese model, two *decision thresholds*.  Baseline = the shipped
``change.model.threshold`` (0.35).  Adapted = ``CHANGE_ADAPTED_THRESHOLD`` — a
threshold **supplied via config, assumed tuned on a validation split**.  The
runner never sweeps the threshold against these test labels (that would be
test-set fitting); if the env var is unset the adapted column is ``n/a``.

Change-VQA answers are rule-based (deterministic) with no trained CDVQA
checkpoint, so that track's adapted column is always ``n/a``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json

logger = logging.getLogger("satquery.eval.change")

DOMAIN = "change"
TITLE = "Change detection (mask + change-VQA)"

_RESIZE = (256, 256)  # matches ChangeDetectionModel.RESIZE
_AB_LABEL = (("A", "B", "label"), ("A", "B", "labels"),
             ("time1", "time2", "label"), ("im1", "im2", "label"),
             ("t1", "t2", "label"))
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_triplet_dirs(root: str, split: str) -> Optional[Tuple[str, str, str]]:
    """Locate A/ B/ label/ subfolders under ``root`` or ``root/<split>``."""
    for base in (os.path.join(root, split), root):
        if not os.path.isdir(base):
            continue
        for a, b, l in _AB_LABEL:
            da, db, dl = os.path.join(base, a), os.path.join(base, b), os.path.join(base, l)
            if os.path.isdir(da) and os.path.isdir(db) and os.path.isdir(dl):
                return da, db, dl
    return None


def _list_triplets(da: str, db: str, dl: str, max_samples: Optional[int]) -> List[Tuple[str, str, str]]:
    names = sorted(set(os.listdir(da)) & set(os.listdir(db)) & set(os.listdir(dl)))
    names = [n for n in names if n.lower().endswith(_IMG_EXTS)]
    if max_samples:
        names = names[:max_samples]
    return [(os.path.join(da, n), os.path.join(db, n), os.path.join(dl, n)) for n in names]


def _mask_track(ccfg: Dict[str, Any], ds_cfg: Dict[str, Any], device: str,
                baseline_thr: float, adapted_thr: Optional[float],
                max_samples: Optional[int], notes: List[str]) -> Optional[Dict[str, Any]]:
    data_dir = ds_cfg.get("data_dir")
    if not is_configured(data_dir):
        notes.append("mask: skipped — change-mask dataset not configured (set LEVIR_CD_DIR).")
        return None
    triplet = _find_triplet_dirs(data_dir, "test")
    if triplet is None:
        notes.append(f"mask: skipped — no A/ B/ label/ subfolders under {data_dir!r}.")
        return None
    pairs = _list_triplets(*triplet, max_samples)
    if not pairs:
        notes.append("mask: skipped — no matching A/B/label image triplets found.")
        return None

    import numpy as np
    from PIL import Image
    from models.change_model import ChangeDetectionModel
    from evaluation.common.metrics import accumulate_confusion, metrics_from_confusion

    model_cfg = ccfg.get("model", {})
    try:
        model = ChangeDetectionModel(
            backbone_name=model_cfg.get("name", "microsoft/resnet-50"),
            device=device,
            cache_dir=model_cfg.get("cache_dir"),
            threshold=baseline_thr,
        )
    except Exception as exc:
        notes.append(f"mask: skipped — could not load change model ({type(exc).__name__}: {exc}).")
        return None

    base_tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    adpt_tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0} if adapted_thr is not None else None
    w, h = _RESIZE
    n = 0
    for pa, pb, pl in pairs:
        try:
            ia = Image.open(pa).convert("RGB")
            ib = Image.open(pb).convert("RGB")
            gt = np.array(Image.open(pl).convert("L").resize((w, h), Image.NEAREST)) > 127
        except Exception as exc:
            logger.debug("skip change triplet %s: %s", pl, exc)
            continue
        cm = model.detect_changes(ia, ib)["change_map"]  # [256,256] float
        cb = accumulate_confusion(cm > baseline_thr, gt)
        if cb["valid"]:
            for k in base_tot:
                base_tot[k] += cb[k]
        if adpt_tot is not None:
            ca = accumulate_confusion(cm > adapted_thr, gt)
            if ca["valid"]:
                for k in adpt_tot:
                    adpt_tot[k] += ca[k]
        n += 1

    if n == 0:
        notes.append("mask: skipped — no change triplets could be opened.")
        return None

    return {
        "n_samples": n,
        "baseline_threshold": baseline_thr,
        "adapted_threshold": adapted_thr,
        "baseline": metrics_from_confusion(**base_tot),
        "adapted": metrics_from_confusion(**adpt_tot) if adpt_tot is not None else None,
        "data_dir": data_dir,
    }


def _vqa_track(ccfg: Dict[str, Any], ds_cfg: Dict[str, Any], device: str,
               max_samples: Optional[int], notes: List[str]) -> Optional[Dict[str, Any]]:
    data_dir = ds_cfg.get("data_dir")
    if not is_configured(data_dir):
        notes.append("change-vqa: skipped — CDVQA dataset not configured (set CDVQA_DIR).")
        return None

    from evaluation.datasets.cdvqa import CDVQADataset
    ds = CDVQADataset(data_dir=data_dir, split="test", max_samples=max_samples)
    if len(ds) == 0:
        notes.append(f"change-vqa: skipped — no CDVQA samples found under {data_dir!r}.")
        return None

    from models.change_model import ChangeDetectionModel
    from evaluation.common.metrics import vqa_metrics_module

    model_cfg = ccfg.get("model", {})
    try:
        model = ChangeDetectionModel(
            backbone_name=model_cfg.get("name", "microsoft/resnet-50"),
            device=device,
            cache_dir=model_cfg.get("cache_dir"),
            threshold=float(model_cfg.get("threshold", 0.35)),
        )
    except Exception as exc:
        notes.append(f"change-vqa: skipped — could not load change model ({type(exc).__name__}: {exc}).")
        return None

    preds: List[str] = []
    golds: List[str] = []
    n_skipped = 0
    for i in range(len(ds)):
        # One unreadable/truncated image or a failed answer must not abort the
        # whole change domain (which would also discard the mask-track results
        # computed before this) — skip the sample and degrade honestly, exactly
        # as the mask track does above.
        try:
            item = ds[i]
            out = model.answer_change_question(item["image_a"], item["image_b"], item["question"])
        except Exception as exc:
            logger.debug("skip CDVQA sample %d: %s", i, exc)
            n_skipped += 1
            continue
        preds.append(out.get("answer", "") or "")
        golds.append(str(item.get("answer", "")))

    if n_skipped:
        notes.append(f"change-vqa: skipped {n_skipped} unreadable/failed CDVQA sample(s).")
    if not preds:
        notes.append("change-vqa: skipped — no CDVQA samples could be opened/answered.")
        return None

    metrics = vqa_metrics_module()["aggregate_metrics"](preds, golds)
    return {"n_samples": len(preds), "metrics": metrics, "data_dir": data_dir}


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    ccfg = cfg.get("change", {})
    datasets = ccfg.get("datasets", {}) or {}
    levir_cfg = datasets.get("levir_cd", {}) or {}
    cdvqa_cfg = datasets.get("cdvqa", {}) or {}

    baseline_thr = float(ccfg.get("model", {}).get("threshold", 0.35))
    adapted_thr_raw = (ccfg.get("variants", {}) or {}).get("adapted_threshold")
    adapted_thr = None
    if is_configured(adapted_thr_raw):
        try:
            adapted_thr = float(adapted_thr_raw)
        except (TypeError, ValueError):
            adapted_thr = None

    notes: List[str] = []
    mask_limit = max_samples if max_samples is not None else levir_cfg.get("max_samples")
    vqa_limit = max_samples if max_samples is not None else cdvqa_cfg.get("max_samples")

    mask = _mask_track(ccfg, levir_cfg, device, baseline_thr, adapted_thr, mask_limit, notes) \
        if levir_cfg.get("enabled", False) else None
    change_vqa = _vqa_track(ccfg, cdvqa_cfg, device, vqa_limit, notes) \
        if cdvqa_cfg.get("enabled", False) else None

    rows: List[Dict[str, Any]] = []
    prov: List[Dict[str, Any]] = []

    if mask:
        for m in ("precision", "recall", "f1", "iou", "overall_accuracy"):
            a = mask["adapted"].get(m) if mask["adapted"] else None
            rows.append(comparison_row("Change (ResNet-50 siamese)", "levir_cd", m,
                                       mask["baseline"].get(m), a))
        prov.append({"name": "levir_cd", "data_dir": mask["data_dir"], "n_samples": mask["n_samples"]})
        if adapted_thr is not None:
            notes.append(f"mask adapted = decision threshold {adapted_thr} (CHANGE_ADAPTED_THRESHOLD, "
                         f"assumed validation-tuned — NOT fit on test labels); baseline threshold {baseline_thr}.")
        else:
            notes.append(f"mask adapted = n/a (set CHANGE_ADAPTED_THRESHOLD to a validation-tuned "
                         f"threshold; baseline {baseline_thr}). Threshold is never swept on test labels.")

    if change_vqa:
        for m in ("exact_match", "token_f1"):
            rows.append(comparison_row("Change-VQA (rule-based)", "cdvqa", m,
                                       change_vqa["metrics"].get(m), None))
        prov.append({"name": "cdvqa", "data_dir": change_vqa["data_dir"], "n_samples": change_vqa["n_samples"]})
        notes.append("change-vqa adapted = n/a: rule-based change QA (deterministic); no trained CDVQA checkpoint.")

    status = "ok" if (mask or change_vqa) else "no_data"
    if status == "no_data":
        notes.append("skipped: neither change-mask (LEVIR-CD) nor CDVQA configured. "
                     "Set LEVIR_CD_DIR and/or CDVQA_DIR.")

    report = {
        "domain": DOMAIN,
        "status": status,
        "mask": mask,
        "change_vqa": change_vqa,
        "comparison": rows,
        "notes": notes,
        "provenance": provenance(prov or None),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    if mask:
        logger.info("Change mask: F1=%.4f IoU=%.4f (baseline thr=%.2f, n=%d)",
                    mask["baseline"].get("f1", 0.0), mask["baseline"].get("iou", 0.0),
                    baseline_thr, mask["n_samples"])
    if change_vqa:
        logger.info("Change-VQA: exact_match=%.4f (n=%d)",
                    change_vqa["metrics"].get("exact_match", 0.0), change_vqa["n_samples"])
    if status == "no_data":
        logger.warning("Change: %s", notes[-1])
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
