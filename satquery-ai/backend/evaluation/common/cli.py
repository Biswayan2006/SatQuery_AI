"""
SatQuery AI — Shared runner CLI scaffolding
===========================================
Every domain runner (``python -m evaluation.<domain>.run``) exposes the same
four flags and the same setup path:

    --config       path to the evaluation YAML (required)
    --max-samples  cap samples per dataset (smoke tests / quick runs)
    --output       override output_dir from the config
    --device       override device ("auto" | "cpu" | "cuda")

``init_run`` loads + env-resolves the config, configures logging, resolves the
output directory and device, and seeds the RNGs — returning
``(cfg, output_dir, device)``.  No local paths are hardcoded here: everything
comes from the config (which itself resolves ``${ENV}`` placeholders).
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Tuple

from evaluation.common.config import get_output_dir, load_config, resolve_device

logger = logging.getLogger("satquery.eval.cli")


def build_parser(domain: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by every domain runner."""
    parser = argparse.ArgumentParser(
        prog=f"python -m evaluation.{domain}.run",
        description=f"SatQuery AI — {domain} evaluation (baseline vs adapted).",
    )
    parser.add_argument("--config", required=True, help="Path to the evaluation YAML config.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap samples per dataset (smoke tests / quick runs).",
    )
    parser.add_argument("--output", default=None, help="Override output_dir from the config.")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda"],
        help="Override the device from the config.",
    )
    return parser


def init_run(args: argparse.Namespace, domain: str) -> Tuple[Dict[str, Any], str, str]:
    """
    Load config, configure logging, resolve output dir + device, seed RNGs.

    Returns ``(cfg, output_dir, device)``.
    """
    cfg = load_config(args.config)

    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = get_output_dir(cfg, default="./evaluation/results")

    device = resolve_device(args.device or cfg.get("device", "auto"))
    _maybe_set_seed(int(cfg.get("seed", 42)))

    logger.info("[%s] output_dir=%s | device=%s", domain, output_dir, device)
    return cfg, output_dir, device


def apply_max_samples(dataset_cfg: Dict[str, Any], max_samples: int | None) -> None:
    """Override each dataset block's ``max_samples`` in place (CLI wins)."""
    if max_samples is None:
        return
    for block in dataset_cfg.values():
        if isinstance(block, dict):
            block["max_samples"] = max_samples


def _maybe_set_seed(seed: int) -> None:
    """Seed RNGs; fall back to stdlib ``random`` if torch/numpy are absent."""
    try:
        from training.vqa_utils import set_seed
        set_seed(seed)
    except Exception:  # pragma: no cover - defensive
        import random
        random.seed(seed)
