"""
SatQuery AI — Grounding Evaluation Runner
=========================================
    python -m evaluation.grounding.run --config evaluation/configs/evaluation.yaml

Scores ``RemoteSensingGrounding`` (OWL-ViT open-vocabulary detection) on a
referring-expression set with mIoU and Acc@0.5 / Acc@0.75 IoU (built on
``confidence.metrics.box_iou`` via ``evaluation.common.metrics.detection``).

Data honesty
------------
VRSBench's captioning / VQA splits carry **no bounding boxes**, so grounding
needs a referring-expression manifest — a JSON list of
``{image, query, box:[x1,y1,x2,y2]}`` (boxes normalised to [0,1], or absolute
with ``width``/``height`` alongside).  When none is found under the configured
data dir, the runner reports an explicit ``no_data`` result rather than
inventing samples.

Model honesty
-------------
The system ships no remote-sensing grounding *checkpoint* and
``RemoteSensingGrounding`` exposes no fine-tune load path, so the adapted
(SatQuery) column is always ``n/a`` — never fabricated.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json

logger = logging.getLogger("satquery.eval.grounding")

DOMAIN = "grounding"
TITLE = "Grounding (OWL-ViT open-vocabulary; RS-adapted n/a)"

_MANIFEST_NAMES = (
    "{split}_grounding.json", "grounding_{split}.json", "grounding.json",
    "referring_expressions.json", "referring.json", "{split}_referring.json",
)
_IMG_SUBDIRS = ("", "images", "Images", "img", "{split}", "{split}/images")
_Q_KEYS = ("query", "expression", "text", "phrase", "sentence", "referring_expression")
_BOX_KEYS = ("box", "bbox", "boxes", "target_box", "gt_box", "region")
_IMG_KEYS = ("image", "image_file", "file_name", "filename", "img", "image_path", "image_id")
_W_KEYS = ("width", "image_width", "w", "img_width")
_H_KEYS = ("height", "image_height", "h", "img_height")
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _first(d: Dict[str, Any], keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _resolve_image(data_dir: str, split: str, name: str) -> Optional[str]:
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    for sub in _IMG_SUBDIRS:
        sub = sub.format(split=split)
        cand = os.path.join(data_dir, sub, name) if sub else os.path.join(data_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def _normalize_box(box, width, height) -> Optional[List[float]]:
    """Coerce a [x1,y1,x2,y2] box to normalised [0,1]; None if impossible."""
    try:
        b = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if len(b) != 4:
        return None
    if max(b) > 1.5:  # looks like absolute pixel coords
        try:
            w, h = float(width), float(height)
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        b = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
    return [min(max(v, 0.0), 1.0) for v in b]


def _load_manifest(data_dir: str, split: str, max_samples: Optional[int]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Find + parse a referring-expression manifest. Returns (samples, manifest_path)."""
    path = None
    for d in (data_dir, os.path.join(data_dir, split)):
        if not os.path.isdir(d):
            continue
        for nm in _MANIFEST_NAMES:
            cand = os.path.join(d, nm.format(split=split))
            if os.path.isfile(cand):
                path = cand
                break
        if path:
            break
    if path is None:
        return [], None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Could not parse grounding manifest %s: %s", path, exc)
        return [], path

    records = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        return [], path

    samples: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        q, box, img = _first(r, _Q_KEYS), _first(r, _BOX_KEYS), _first(r, _IMG_KEYS)
        if q is None or box is None or img is None:
            continue
        # A list-of-boxes manifest → take the first box as the single target.
        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
            box = box[0]
        gold = _normalize_box(box, _first(r, _W_KEYS), _first(r, _H_KEYS))
        if gold is None:
            continue
        img_path = _resolve_image(data_dir, split, str(img))
        if img_path is None:
            continue
        samples.append({"image_file": img_path, "query": str(q), "box": gold})
        if max_samples and len(samples) >= max_samples:
            break
    return samples, path


def _no_data(output_dir: str, notes: List[str], prov=None) -> Dict[str, Any]:
    report = {"domain": DOMAIN, "status": "no_data", "dataset": "grounding",
              "metrics": {}, "comparison": [], "notes": notes, "provenance": provenance(prov)}
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, [], notes)
    logger.warning("Grounding: %s", notes[-1] if notes else "no data")
    return report


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    gcfg = cfg.get("grounding", {})
    ds_cfg = (gcfg.get("datasets", {}) or {}).get("vrsbench", {})
    data_dir = ds_cfg.get("data_dir")
    thresholds = tuple(gcfg.get("iou_thresholds", [0.5, 0.75]))
    notes: List[str] = []

    if not is_configured(data_dir):
        notes.append("skipped: grounding dataset not configured (set VRSBENCH_DIR or a "
                     "grounding manifest directory).")
        return _no_data(output_dir, notes)

    limit = max_samples if max_samples is not None else ds_cfg.get("max_samples")
    samples, manifest_path = _load_manifest(data_dir, "test", limit)
    if not samples:
        notes.append(
            f"skipped: no referring-expression manifest with boxes found under {data_dir!r}. "
            "VRSBench captioning/VQA splits carry no boxes; provide a grounding manifest "
            "([{image, query, box:[x1,y1,x2,y2]}])."
        )
        return _no_data(output_dir, notes, prov=[{"name": "grounding", "data_dir": data_dir, "n_samples": 0}])

    from PIL import Image
    from models.grounding_model import RemoteSensingGrounding
    from evaluation.common.metrics import aggregate_detection_metrics

    model_cfg = gcfg.get("model", {})
    try:
        model = RemoteSensingGrounding(
            model_name=model_cfg.get("name", "google/owlvit-base-patch32"),
            device=device,
            cache_dir=model_cfg.get("cache_dir"),
        )
    except Exception as exc:
        notes.append(f"skipped: could not load grounding model ({type(exc).__name__}: {exc}).")
        return _no_data(output_dir, notes, prov=[{"name": "grounding", "data_dir": data_dir, "n_samples": len(samples)}])

    predictions: List[List[List[float]]] = []
    references: List[List[List[float]]] = []
    scores: List[List[float]] = []
    for s in samples:
        try:
            image = Image.open(s["image_file"]).convert("RGB")
        except Exception as exc:
            logger.debug("skip grounding image %s: %s", s["image_file"], exc)
            continue
        out = model.ground(image, s["query"])
        boxes = out.get("boxes", []) or []          # normalised, sorted by score desc
        sc = out.get("scores", []) or []
        predictions.append([boxes[0]] if boxes else [])   # top-1 for referring-expression
        references.append([s["box"]])
        scores.append([sc[0]] if sc else [])

    if not predictions:
        notes.append("skipped: no grounding images could be opened.")
        return _no_data(output_dir, notes, prov=[{"name": "grounding", "data_dir": data_dir, "n_samples": 0}])

    metrics = aggregate_detection_metrics(predictions, references, thresholds=thresholds, scores=scores)

    model_label = "OWL-ViT grounding"
    rows = [comparison_row(model_label, "grounding", "mIoU", metrics.get("m_iou"), None)]
    for t in thresholds:
        rows.append(comparison_row(model_label, "grounding", f"Acc@{t}", metrics.get(f"acc@{t}"), None))

    notes.append("Adapted = n/a: no remote-sensing grounding checkpoint exists and "
                 "RemoteSensingGrounding exposes no fine-tune load path.")
    notes.append(f"Top-1 predicted box vs gold over {metrics.get('n_samples', 0)} referring "
                 f"expressions from {os.path.basename(manifest_path or 'manifest')}.")

    report = {
        "domain": DOMAIN,
        "status": "ok",
        "dataset": "grounding",
        "metrics": metrics,
        "comparison": rows,
        "notes": notes,
        "provenance": provenance([{"name": "grounding", "data_dir": data_dir,
                                   "n_samples": metrics.get("n_samples", 0)}]),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    logger.info("Grounding: mIoU=%.4f Acc@0.5=%.4f (n=%d)",
                metrics.get("m_iou", 0.0), metrics.get("acc@0.5", 0.0), metrics.get("n_samples", 0))
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
