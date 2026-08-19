"""Experiment configuration schema (TDD 11).

YAML is the authoring format because a config must be able to record *why* a
parameter was chosen; it normalises to JSON immediately after validation, and the
normalised JSON is what gets hashed and stored.
"""

from __future__ import annotations

import itertools
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DType(StrEnum):
    FP64 = "fp64"
    FP32 = "fp32"


class Dimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    M: int = Field(gt=0)
    N: int = Field(gt=0)
    K: int = Field(gt=0)


class DTypeCombo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: DType = DType.FP64
    b: DType = DType.FP64
    c: DType = DType.FP64
    compute: DType = DType.FP64


class Sweep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimensions: list[Dimensions] = Field(min_length=1)
    dtype: list[DTypeCombo] = Field(min_length=1)


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["GEMM"] = "GEMM"
    sweep: Sweep
    alpha: float = 1.0
    beta: float = 0.0
    seed: int = 0


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backends: list[str] = Field(min_length=1)
    implementations: list[str] = Field(min_length=1)


class OutlierPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: Literal["mad", "tukey", "none"] = "mad"
    threshold: float = 3.0
    # NEVER delete. Outliers are flagged; deletion destroys the evidence.
    action: Literal["flag", "exclude"] = "flag"


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warmup_iterations: int = Field(default=20, ge=0)
    measure_iterations: int = Field(default=100, gt=0)
    repetitions: int = Field(default=5, gt=0)
    timing_method: Literal["host_wall"] = "host_wall"
    inter_repetition_cooldown_s: float = Field(default=0.0, ge=0)
    outlier_policy: OutlierPolicy = OutlierPolicy()


class Correctness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    reference: Literal["cpu_reference"] = "cpu_reference"
    tolerance_c: float = Field(default=8.0, gt=0)


class Validity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_valid_samples: int = Field(default=1, ge=0)
    max_cv: float = Field(default=0.0, ge=0)


class Power(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # There is no characterised power source on this platform. `none` is the only
    # honest setting; energy stays NOT YET MEASURED (ADR-024).
    method: Literal["none"] = "none"
    required: bool = False

    @model_validator(mode="after")
    def _cannot_require_absent_telemetry(self) -> Power:
        if self.required and self.method == "none":
            raise ValueError("power.required=true is impossible with power.method=none")
        return self


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_dir: str = "results/raw"
    retain_warmup_samples: bool = True
    emit_reproducibility_package: bool = True


class ProvenanceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    author: str
    created: str
    supersedes: str | None = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    experiment_id: str = Field(pattern=r"^EXP-\d{3}$")
    experiment_version: str
    title: str
    hypothesis: str = Field(min_length=1)
    workload: Workload
    execution: Execution
    measurement: Measurement = Measurement()
    correctness: Correctness = Correctness()
    validity: Validity = Validity()
    power: Power = Power()
    output: Output = Output()
    provenance: ProvenanceBlock


class RunPoint(BaseModel):
    """One fully-specified point of the sweep - a single benchmark run."""

    model_config = ConfigDict(extra="forbid")
    backend: str
    implementation: str
    dimensions: Dimensions
    dtype: DTypeCombo
    repetition_index: int


def expand_sweep(cfg: ExperimentConfig) -> list[RunPoint]:
    """Cartesian expansion of the sweep, in a deterministic order.

    Order matters for reproducibility: the same config must produce the same run
    sequence on every machine.
    """
    points: list[RunPoint] = []
    for backend, impl, dims, dt, rep in itertools.product(
        cfg.execution.backends,
        cfg.execution.implementations,
        cfg.workload.sweep.dimensions,
        cfg.workload.sweep.dtype,
        range(cfg.measurement.repetitions),
    ):
        points.append(
            RunPoint(
                backend=backend,
                implementation=impl,
                dimensions=dims,
                dtype=dt,
                repetition_index=rep,
            )
        )
    return points
