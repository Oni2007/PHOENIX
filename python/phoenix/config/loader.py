"""YAML -> validated model -> normalised JSON (ADR-008)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from phoenix.config.models import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return ExperimentConfig.model_validate(raw)


def normalise(cfg: ExperimentConfig) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    This is what gets content-hashed, so it must be byte-stable.
    """
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
