"""
SatQuery AI — Full Benchmark Orchestrator
=========================================
    python -m evaluation.run_all --config evaluation/configs/evaluation.yaml

Runs every domain evaluation in a SINGLE process so the shared
``benchmark_report.md`` accumulates one section per domain.  Running each domain
as a separate ``python -m evaluation.<domain>.run`` truncates the report at the
start of every process (the ``_REPORT_INITIALISED`` guard in
``evaluation/common/reporting.py`` is per-process), leaving only the last
domain's section — this orchestrator is the way to get the consolidated
``Model | Dataset | Metric | Baseline | SatQuery | Improvement`` report the spec
asks for.  Each domain still writes its own ``<domain>_results.json`` too.

Order matters
-------------
VQA runs *before* confidence: the confidence runner consumes the
``(confidence, correct)`` pairs the VQA runner emits into ``vqa_results.json``.

Isolation & honesty
-------------------
Each domain runs inside its own try/except — if one raises (e.g. a model can't
be downloaded offline) it is logged and the benchmark continues with the rest;
one domain never aborts the whole report.  Domains self-report ``no_data`` when
their dataset / checkpoint env vars are unset (every runner checks
``is_configured`` before touching a model), so an offline run is fast and
fabricates nothing — it reports exactly what it could and could not measure.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from evaluation.common.cli import build_parser, init_run

logger = logging.getLogger("satquery.eval.run_all")

DOMAIN = "run_all"

# VQA before confidence (confidence consumes VQA's emitted confidence_pairs);
# routing + confidence last (cheap / dependent on earlier output).
_DOMAINS: List[str] = [
    "vqa", "retrieval", "captioning", "grounding",
    "change", "fusion", "routing", "confidence",
]


def _load_run(domain: str) -> Callable:
    """Import ``evaluation.<domain>.run`` and return its ``run`` callable."""
    import importlib
    mod = importlib.import_module(f"evaluation.{domain}.run")
    return mod.run


def run_all(cfg: Dict[str, Any], output_dir: str, device: str,
            max_samples: int | None = None) -> Dict[str, Dict[str, Any]]:
    """Run every domain in sequence; return ``{domain: report}``."""
    results: Dict[str, Dict[str, Any]] = {}
    for domain in _DOMAINS:
        try:
            run_fn = _load_run(domain)
        except Exception as exc:  # import-time failure (should not happen)
            logger.error("Could not import evaluation.%s.run: %s", domain, exc)
            results[domain] = {"domain": domain, "status": "error",
                               "error": f"{type(exc).__name__}: {exc}"}
            continue
        logger.info("──────── evaluating: %s ────────", domain)
        try:
            results[domain] = run_fn(cfg, output_dir, device, max_samples=max_samples)
        except Exception as exc:  # runtime failure in one domain must not abort the rest
            logger.exception("Domain '%s' failed — continuing with the remaining domains", domain)
            results[domain] = {"domain": domain, "status": "error",
                               "error": f"{type(exc).__name__}: {exc}"}

    ok = [d for d, r in results.items() if r.get("status") == "ok"]
    nodata = [d for d, r in results.items() if r.get("status") == "no_data"]
    errored = [d for d, r in results.items() if r.get("status") == "error"]
    logger.info("Benchmark complete → %s", output_dir)
    logger.info("  ok:      %s", ", ".join(ok) or "(none)")
    logger.info("  no_data: %s", ", ".join(nodata) or "(none)")
    if errored:
        logger.warning("  errored: %s", ", ".join(errored))
    return results


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run_all(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
