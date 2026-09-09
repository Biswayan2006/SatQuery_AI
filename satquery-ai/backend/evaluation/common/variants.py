"""
SatQuery AI — Baseline / Adapted Variant Resolution
===================================================
Central, honest logic for deciding whether a domain has a real *adapted*
(remote-sensing fine-tuned) variant to compare against its *baseline* (generic
pretrained) variant.

The spec asks for BASELINE vs ADAPTED comparisons:

  * VQA          — baseline = pretrained BLIP; adapted = fine-tuned checkpoint
  * Retrieval    — baseline = OpenAI CLIP;    adapted = RS-CLIP checkpoint
  * Fusion       — baseline = untrained adapter; adapted = fusion checkpoint

For captioning, grounding, and change detection there is currently NO adapted
checkpoint in the system.  This module returns ``adapted=None`` for those so the
report shows an honest ``n/a`` rather than a fabricated number.

A ``Variant`` simply names a run and carries the checkpoint path (or ``None``);
each domain's ``run.py`` interprets it.  This module does not load models — it
only resolves *what should be compared*, from config, without hardcoding paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.common.config import is_configured, path_exists


@dataclass
class Variant:
    """One evaluation variant (baseline or adapted)."""
    name: str                       # human label, e.g. "BLIP pretrained"
    kind: str                       # "baseline" | "adapted"
    checkpoint: Optional[str] = None
    available: bool = True          # False → skip + report n/a
    reason: Optional[str] = None    # why unavailable (for notes)
    extra: Dict[str, Any] = field(default_factory=dict)


def _variants_cfg(cfg: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Fetch the ``variants`` block for a domain, falling back to top-level."""
    dom = (cfg.get(domain) or {})
    return dom.get("variants") or cfg.get("variants") or {}


def resolve_variants(
    cfg: Dict[str, Any],
    domain: str,
    *,
    baseline_label: str,
    adapted_label: str,
    adapted_needs_checkpoint: bool = True,
) -> List[Variant]:
    """
    Resolve the baseline + adapted variants for a domain.

    ``adapted_needs_checkpoint`` — when True (VQA, retrieval, fusion) the adapted
    variant is only "available" if its checkpoint is configured *and* the path
    exists on disk; otherwise it is marked unavailable with a clear reason.

    When False (a domain whose adapted variant is a config flag rather than a
    file) the adapted variant is available whenever configured.
    """
    vcfg = _variants_cfg(cfg, domain)
    baseline_ckpt = vcfg.get("baseline_checkpoint") or None
    adapted_ckpt = vcfg.get("adapted_checkpoint") or None

    baseline = Variant(name=baseline_label, kind="baseline", checkpoint=baseline_ckpt or None)

    if adapted_needs_checkpoint:
        if not is_configured(adapted_ckpt):
            adapted = Variant(
                name=adapted_label, kind="adapted", available=False,
                reason="no adapted checkpoint configured "
                       f"(set variants.adapted_checkpoint for '{domain}')",
            )
        elif not path_exists(adapted_ckpt):
            adapted = Variant(
                name=adapted_label, kind="adapted", checkpoint=adapted_ckpt, available=False,
                reason=f"configured adapted checkpoint not found on disk: {adapted_ckpt}",
            )
        else:
            adapted = Variant(name=adapted_label, kind="adapted", checkpoint=adapted_ckpt)
    else:
        adapted = Variant(name=adapted_label, kind="adapted", checkpoint=adapted_ckpt or None)

    return [baseline, adapted]


def unavailable_adapted(label: str, reason: str) -> Variant:
    """Convenience: an adapted variant that does not exist in this system yet."""
    return Variant(name=label, kind="adapted", available=False, reason=reason)
