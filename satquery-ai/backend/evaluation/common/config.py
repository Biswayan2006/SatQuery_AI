"""
SatQuery AI — Evaluation Config Loading
=======================================
Loads the evaluation YAML and resolves ``${ENV_VAR}`` / ``${ENV_VAR:-default}``
placeholders from the environment.  This is the ONLY place dataset and
checkpoint locations enter the framework — nothing is hardcoded in code.

A path left blank (empty string / null / an unresolved ``${VAR}`` with no
environment value and no default) is treated as "not configured": the
corresponding dataset or variant is skipped and reported honestly as absent,
never fabricated.

Placeholder syntax
------------------
    ${VRSBENCH_DIR}            -> os.environ["VRSBENCH_DIR"]      ("" if unset)
    ${VRSBENCH_DIR:-/data/x}   -> os.environ.get("VRSBENCH_DIR", "/data/x")

Resolution is applied recursively to every string value in the config.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("satquery.eval.config")

# ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _resolve_env_in_str(value: str) -> str:
    """Replace every ``${VAR}`` / ``${VAR:-default}`` occurrence in a string."""

    def _sub(match: "re.Match[str]") -> str:
        var, default = match.group(1), match.group(2)
        env_val = os.environ.get(var)
        if env_val is not None and env_val != "":
            return env_val
        return default if default is not None else ""

    return _ENV_PATTERN.sub(_sub, value)


def resolve_env(obj: Any) -> Any:
    """Recursively resolve env placeholders in all string values of a structure."""
    if isinstance(obj, str):
        return _resolve_env_in_str(obj)
    if isinstance(obj, dict):
        return {k: resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env(v) for v in obj]
    return obj


def load_config(path: str) -> Dict[str, Any]:
    """
    Load the evaluation YAML and resolve env placeholders.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    RuntimeError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Evaluation config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    resolved = resolve_env(raw)
    logger.debug("Loaded evaluation config from %s", path)
    return resolved


def is_configured(value: Optional[str]) -> bool:
    """
    True when a path/string value is actually set (non-empty after env
    resolution and not a leftover placeholder).
    """
    if not value or not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    # An unresolved ${...} means the env var was unset with no default.
    if _ENV_PATTERN.search(v):
        return False
    return True


def path_exists(value: Optional[str]) -> bool:
    """True when ``value`` is configured AND points at an existing path."""
    return is_configured(value) and os.path.exists(value.strip())


def get_output_dir(cfg: Dict[str, Any], default: str = "./evaluation/results") -> str:
    """Resolve the results output directory (config ``output_dir`` or default)."""
    out = cfg.get("output_dir") or default
    if not is_configured(out):
        out = default
    os.makedirs(out, exist_ok=True)
    return out


def resolve_device(device: str = "auto") -> str:
    """Resolve an "auto" device string to "cuda"/"cpu"."""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
