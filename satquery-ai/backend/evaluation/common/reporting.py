"""
SatQuery AI — Evaluation Reporting
==================================
Writes JSON and Markdown reports and builds the baseline-vs-adapted comparison
table required by the spec::

    | Model | Dataset | Metric | Baseline | SatQuery | Improvement |

Honesty rules baked in:

  * A metric whose adapted variant does not exist (no adapted checkpoint) is
    rendered with ``SatQuery = "n/a"`` and ``Improvement = "n/a"`` — never a
    fabricated number.
  * A domain that produced no data at all (missing dataset) writes a report
    that says so explicitly, rather than emitting empty/zero metrics that look
    like real results.

Provenance (git commit + dataset fingerprint) is attached to every JSON report
by reusing ``training.vqa_utils.git_commit_hash`` / ``dataset_version``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("satquery.eval.reporting")


# ── Provenance ──────────────────────────────────────────────────────────────────

def provenance(datasets: Optional[Sequence[Dict]] = None) -> Dict[str, Any]:
    """Build a provenance block reusing the training-side helpers."""
    try:
        from training.vqa_utils import dataset_version, git_commit_hash
        return {
            "git_commit": git_commit_hash(),
            "dataset_version": dataset_version(list(datasets)) if datasets else None,
            "datasets": list(datasets) if datasets else [],
        }
    except Exception as exc:  # pragma: no cover - provenance is best-effort
        logger.debug("Provenance helpers unavailable: %s", exc)
        return {"git_commit": None, "dataset_version": None, "datasets": list(datasets or [])}


# ── Comparison rows ─────────────────────────────────────────────────────────────

_NA = "n/a"


def _fmt(value: Any) -> str:
    """Format a metric value for the Markdown table."""
    if value is None:
        return _NA
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def comparison_row(
    model: str,
    dataset: str,
    metric: str,
    baseline: Optional[float],
    adapted: Optional[float],
    *,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Build one comparison row. ``Improvement`` is adapted − baseline when both
    are real numbers, else ``"n/a"``.  The sign is oriented so a positive
    improvement always means "adapted is better".
    """
    improvement: Any = _NA
    if isinstance(baseline, (int, float)) and isinstance(adapted, (int, float)):
        delta = float(adapted) - float(baseline)
        improvement = delta if higher_is_better else -delta
    return {
        "model": model,
        "dataset": dataset,
        "metric": metric,
        "baseline": baseline,
        "satquery": adapted,
        "improvement": improvement,
        "higher_is_better": higher_is_better,
    }


def build_comparison_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render comparison rows as a Markdown table."""
    header = (
        "| Model | Dataset | Metric | Baseline | SatQuery | Improvement |\n"
        "|-------|---------|--------|----------|----------|-------------|\n"
    )
    if not rows:
        return header + "| _(no comparable results — see notes)_ |||||| \n"

    lines = []
    for r in rows:
        imp = r.get("improvement")
        if isinstance(imp, (int, float)):
            imp_str = f"{imp:+.4f}"
        else:
            imp_str = _NA
        lines.append(
            f"| {r.get('model', _NA)} | {r.get('dataset', _NA)} | {r.get('metric', _NA)} "
            f"| {_fmt(r.get('baseline'))} | {_fmt(r.get('satquery'))} | {imp_str} |"
        )
    return header + "\n".join(lines) + "\n"


# ── Writers ─────────────────────────────────────────────────────────────────────

def write_json(output_dir: str, domain: str, report: Dict[str, Any]) -> str:
    """Write ``<output_dir>/<domain>_results.json`` and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{domain}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote JSON report: %s", path)
    return path


def append_markdown(
    output_dir: str,
    domain: str,
    title: str,
    rows: Sequence[Dict[str, Any]],
    notes: Optional[Sequence[str]] = None,
    report_name: str = "benchmark_report.md",
) -> str:
    """
    Append this domain's section to the shared ``benchmark_report.md``.

    The first write in a process run truncates the file (fresh report); later
    domains append.  A per-process guard lives in ``_REPORT_INITIALISED``.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, report_name)

    # First write of this process truncates and writes the header; later
    # domains in the same run append their own sections.
    first = path not in _REPORT_INITIALISED
    with open(path, "w" if first else "a", encoding="utf-8") as f:
        if first:
            f.write("# SatQuery AI — Benchmark Report\n\n")
            f.write(
                "_Baseline vs SatQuery-adapted comparison. `n/a` marks a variant "
                "with no produced checkpoint (never fabricated)._\n\n"
            )
            _REPORT_INITIALISED.add(path)
        f.write(f"## {title}\n\n")
        f.write(build_comparison_table(rows))
        if notes:
            f.write("\n**Notes:**\n\n")
            for n in notes:
                f.write(f"- {n}\n")
        f.write("\n")
    logger.info("Appended Markdown section '%s' to %s", title, path)
    return path


# Report files initialised (truncated + header written) this process, so a
# single `run` truncates once then appends subsequent sections.
_REPORT_INITIALISED: set = set()
