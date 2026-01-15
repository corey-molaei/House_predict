from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config from the given path."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to repo root."""
    root = Path(cfg.get("paths", {}).get("root", "."))
    return (root / cfg["paths"][key]).resolve()


def env_or_config(cfg: Dict[str, Any], env_key: str, cfg_key: str) -> str | None:
    """Get value from env first, then config, else None."""
    val = os.environ.get(env_key)
    if val:
        return val
    return cfg.get("ingest", {}).get(cfg_key)
