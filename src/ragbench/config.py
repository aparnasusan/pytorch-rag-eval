"""Load pipeline settings from config.yaml (path overridable via RAGBENCH_CONFIG)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a YAML config file. Also loads API keys from .env into the environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path or os.environ.get("RAGBENCH_CONFIG") or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping")
    return data


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Cached default config for the current process."""
    return load_config()
