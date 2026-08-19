from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phoenix.config.loader import load_config, normalise
from phoenix.config.models import ExperimentConfig, expand_sweep

CONFIG = Path("experiments/EXP-001_cpu_reference_gemm/config.yaml")


def test_exp001_config_validates() -> None:
    cfg = load_config(CONFIG)
    assert cfg.experiment_id == "EXP-001"
    assert cfg.hypothesis.strip()


def test_sweep_expansion_is_the_full_cartesian_product() -> None:
    cfg = load_config(CONFIG)
    points = expand_sweep(cfg)
    expected = (
        len(cfg.execution.backends)
        * len(cfg.execution.implementations)
        * len(cfg.workload.sweep.dimensions)
        * len(cfg.workload.sweep.dtype)
        * cfg.measurement.repetitions
    )
    assert len(points) == expected


def test_sweep_expansion_is_deterministic() -> None:
    cfg = load_config(CONFIG)
    assert [p.model_dump() for p in expand_sweep(cfg)] == [
        p.model_dump() for p in expand_sweep(cfg)
    ]


def test_normalisation_is_byte_stable() -> None:
    cfg = load_config(CONFIG)
    assert normalise(cfg) == normalise(cfg)


def test_unknown_field_is_rejected_not_ignored() -> None:
    cfg = load_config(CONFIG).model_dump(mode="json")
    cfg["turbo_mode"] = True
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_experiment_id_must_match_the_house_pattern() -> None:
    cfg = load_config(CONFIG).model_dump(mode="json")
    cfg["experiment_id"] = "my-experiment"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_cannot_require_power_that_does_not_exist() -> None:
    cfg = load_config(CONFIG).model_dump(mode="json")
    cfg["power"] = {"method": "none", "required": True}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_outlier_action_cannot_be_delete() -> None:
    cfg = load_config(CONFIG).model_dump(mode="json")
    cfg["measurement"]["outlier_policy"]["action"] = "delete"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_discovery_survives_absent_tools() -> None:
    """An absent tool is recorded as null, never a crash and never a guess.

    Regression: `"".splitlines()[0]` raised IndexError whenever a tool was missing
    from PATH, which is the exact case discovery exists to handle.
    """
    from phoenix.discovery.environment import _first_line, collect_environment

    assert _first_line(None) is None
    assert _first_line("") is None
    assert _first_line("one\ntwo") == "one"

    env = collect_environment(None)
    assert "gcc" in env["toolchain"]
    assert "cmake" in env["toolchain"]
    assert env["accelerators"]["nvidia_gpu_present"] is False
