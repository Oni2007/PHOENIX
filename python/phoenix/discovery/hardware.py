"""Hardware database loader (TDD 9).

Every numeric field is a ProvenanceValue. A record with an unsourced number does
not validate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phoenix.provenance import Confidence, ProvenanceValue


class ComputeUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    count: ProvenanceValue


class MemoryLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    technology: str | None = None
    capacity: ProvenanceValue | None = None
    bandwidth: ProvenanceValue | None = None
    bandwidth_attained: ProvenanceValue | None = None


class Identity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor: str
    model: str
    family: str | None = None
    architecture: str | None = None
    microarchitecture: str | None = None
    form_factor: str | None = None


class ProvenanceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    authored_by: str
    reviewed_by: str | None = None
    last_verified: str
    overall_confidence: Confidence


class DeviceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    id: str
    record_revision: int = 1
    device_class: str
    identity: Identity
    compute: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    power: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    provenance_summary: ProvenanceSummary


def load_device_record(path: Path) -> DeviceRecord:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DeviceRecord.model_validate(raw)


def load_all(hardware_dir: Path) -> list[DeviceRecord]:
    return [load_device_record(p) for p in sorted(hardware_dir.rglob("*.yaml"))]
