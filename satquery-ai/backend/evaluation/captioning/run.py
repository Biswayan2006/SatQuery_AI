"""
SatQuery AI — Captioning Evaluation Runner
==========================================
    python -m evaluation.captioning.run --config evaluation/configs/evaluation.yaml

Scores ``RemoteSensingCaptioning`` (BLIP image-captioning) on the VRSBench test
split with BLEU-1..4, ROUGE-L, METEOR, and CIDEr (pure-Python metrics; caveats
in EVALUATION.md).

Honesty: the system ships no remote-sensing captioning *checkpoint* and the
captioning model exposes no fine-tuning load path, so the adapted (SatQuery)
column is always ``n/a`` — never fabricated.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json

logger = logging.getLogger("satquery.eval.captioning")

DOMAIN = "captioning"
TITLE = "Captioning (BLIP; RS-adapted n/a)"
_REPORTED = ["bleu_1", "bleu_4", "rouge_l", "meteor", "cider"]


def _no_data(output_dir: str, notes: List[str], prov=None) -> Dict[str, Any]:
    report = {"domain": DOMAIN, "status": "no_data", "dataset": "vrsbench",
              "metrics": {}, "comparison": [], "notes": notes, "provenance": provenance(prov)}
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, [], notes)
    logger.warning("Captioning: %s", notes[-1] if notes else "no data")
    return report


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    capcfg = cfg.get("captioning", {})
    ds_cfg = (capcfg.get("datasets", {}) or {}).get("vrsbench", {})
    data_dir = ds_cfg.get("data_dir")
    notes: List[str] = []

    if not is_configured(data_dir):
        notes.append("skipped: captioning dataset not configured (set VRSBENCH_DIR).")
        return _no_data(output_dir, notes)

    limit = max_samples if max_samples is not None else ds_cfg.get("max_samples")

    from training.datasets.vrsbench import VRSBenchAdapter
    adapter = VRSBenchAdapter(data_dir=data_dir, split="test", max_samples=limit)
    if len(adapter) == 0:
        notes.append("skipped: VRSBench test split empty (no annotations.json / test_captions.json found).")
        return _no_data(output_dir, notes, prov=[{"name": "vrsbench", "data_dir": data_dir, "n_samples": 0}])

    from PIL import Image
    from models.captioning_model import RemoteSensingCaptioning
    from evaluation.common.metrics import aggregate_caption_metrics

    model_cfg = capcfg.get("model", {})
    try:
        model = RemoteSensingCaptioning(
            model_name=model_cfg.get("name", "Salesforce/blip-image-captioning-base"),
            device=device,
            cache_dir=model_cfg.get("cache_dir"),
        )
    except Exception as exc:
        notes.append(f"skipped: could not load captioning model ({type(exc).__name__}: {exc}).")
        return _no_data(output_dir, notes, prov=[{"name": "vrsbench", "data_dir": data_dir, "n_samples": len(adapter)}])

    hypotheses: List[str] = []
    references: List[List[str]] = []
    for sample in adapter.samples:  # iterate raw records to open PIL directly
        try:
            image = Image.open(sample["image_file"]).convert("RGB")
        except Exception as exc:
            logger.debug("skip image %s: %s", sample.get("image_file"), exc)
            continue
        out = model.generate_caption(image)
        hypotheses.append(out.get("caption", "") or "")
        references.append(list(sample.get("captions", [])))

    if not hypotheses:
        notes.append("skipped: no VRSBench images could be opened for captioning.")
        return _no_data(output_dir, notes, prov=[{"name": "vrsbench", "data_dir": data_dir, "n_samples": 0}])

    metrics = aggregate_caption_metrics(hypotheses, references)

    rows = [comparison_row("BLIP captioning", "vrsbench", m, metrics.get(m), None) for m in _REPORTED]
    notes.append("Adapted = n/a: no remote-sensing captioning checkpoint exists in the system "
                 "(set captioning.variants.adapted_checkpoint + a load path to enable).")
    notes.append(f"Scored {metrics.get('n_samples', 0)} generated captions against VRSBench references.")
    notes.append("METEOR is WordNet-free and CIDEr uses tf-idf n-gram consensus — see EVALUATION.md.")

    report = {
        "domain": DOMAIN,
        "status": "ok",
        "dataset": "vrsbench",
        "metrics": metrics,
        "comparison": rows,
        "notes": notes,
        "provenance": provenance([{"name": "vrsbench", "data_dir": data_dir,
                                   "n_samples": metrics.get("n_samples", 0)}]),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    logger.info("Captioning: BLEU-4=%.4f ROUGE-L=%.4f METEOR=%.4f CIDEr=%.4f (n=%d)",
                metrics.get("bleu_4", 0.0), metrics.get("rouge_l", 0.0),
                metrics.get("meteor", 0.0), metrics.get("cider", 0.0), metrics.get("n_samples", 0))
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
