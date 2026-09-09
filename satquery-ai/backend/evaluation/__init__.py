"""
SatQuery AI — Reproducible Evaluation Framework
================================================
Quantitative, baseline-vs-adapted evaluation of the system's remote-sensing
adaptations across seven domains: VQA, captioning, grounding, retrieval, change
detection, SAR-optical fusion, routing, and confidence calibration.

Each domain is runnable as a module::

    python -m evaluation.vqa.run     --config evaluation/configs/evaluation.yaml
    python -m evaluation.routing.run --config evaluation/configs/evaluation.yaml
    ...

Principles (enforced throughout):

  * NEVER use benchmark test labels during inference or training.
  * NEVER hardcode local dataset / checkpoint paths — config + env only.
  * NEVER fabricate an "adapted" number where no adapted checkpoint exists;
    such cells are reported honestly as "n/a".

The framework only *reads* the model classes and reuses existing metric /
dataset helpers under ``training/`` and ``confidence/`` — it does not modify
``models/`` or ``api/``.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
