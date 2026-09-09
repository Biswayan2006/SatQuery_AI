"""
SatQuery AI — Confidence Calibration Runner
===========================================
    python -m evaluation.confidence.run --config evaluation/configs/evaluation.yaml

Measures how well the system's *emitted* confidence is calibrated against
observed correctness, using the ``(confidence, correct)`` pairs the VQA runner
writes to ``<output_dir>/vqa_results.json``.

Metrics (from ``confidence.metrics``): ECE, MCE, Brier score, and a per-bin
reliability curve.

Honesty:
  * No pairs available (VQA not run / no data) → explicit ``no_data`` report.
  * The emitted confidence is the *uncalibrated* proxy exp(mean token log-prob).
    No fitted calibrator has been produced, so the "adapted" column is ``n/a``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from evaluation.common.cli import build_parser, init_run
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json

logger = logging.getLogger("satquery.eval.confidence")

DOMAIN = "confidence"
TITLE = "Confidence calibration (VQA emitted confidence)"


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: int | None = None) -> Dict[str, Any]:
    ccfg = cfg.get("confidence", {})
    n_bins = int(ccfg.get("n_bins", 10))
    source = str(ccfg.get("source", "vqa"))
    notes: List[str] = []

    pairs: List[Dict[str, Any]] = []
    src_path = os.path.join(output_dir, f"{source}_results.json")
    if os.path.isfile(src_path):
        try:
            with open(src_path, encoding="utf-8") as f:
                pairs = json.load(f).get("confidence_pairs", []) or []
        except Exception as exc:
            notes.append(f"could not read {src_path}: {exc}")
    else:
        notes.append(f"no source report at {src_path}.")

    if max_samples is not None:
        pairs = pairs[:max_samples]

    if not pairs:
        notes.append(
            "skipped: no (confidence, correct) pairs available. Run "
            "`python -m evaluation.vqa.run` with a dataset configured first — "
            "it emits confidence_pairs consumed here."
        )
        report = {"domain": DOMAIN, "status": "no_data", "source": source,
                  "notes": notes, "provenance": provenance()}
        write_json(output_dir, DOMAIN, report)
        append_markdown(output_dir, DOMAIN, TITLE, [], notes)
        logger.warning("Confidence: %s", notes[-1])
        return report

    confidences = [float(p["confidence"]) for p in pairs]
    correct = [int(p["correct"]) for p in pairs]

    from evaluation.common.metrics import reliability_stats
    stats = reliability_stats(confidences, correct, n_bins)

    # ECE / Brier are "lower is better"; there is no calibrated variant yet.
    rows = [
        comparison_row("VQA confidence", f"{source} predictions", "ECE",
                       stats["ece"], None, higher_is_better=False),
        comparison_row("VQA confidence", f"{source} predictions", "Brier",
                       stats["brier"], None, higher_is_better=False),
    ]
    notes.append(
        f"Measured on {stats['n_samples']} predictions' emitted (uncalibrated) "
        "confidence = exp(mean token log-prob)."
    )
    notes.append(
        "Adapted = n/a: no fitted temperature calibrator has been produced "
        "(the default CalibrationStore is identity)."
    )

    report = {
        "domain": DOMAIN,
        "status": "ok",
        "source": source,
        "n_bins": n_bins,
        "stats": stats,
        "comparison": rows,
        "notes": notes,
        "provenance": provenance(),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    logger.info("Confidence: ECE=%.4f Brier=%.4f over %d predictions",
                stats["ece"], stats["brier"], stats["n_samples"])
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
