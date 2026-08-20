"""Hardware database loader (TDD §9).

Every numeric field is a ProvenanceValue, and `compute`/`memory`/`interconnect`
carry real typed structure rather than a loose passthrough dict — a record with
an unsourced number, or a peak-compute figure with no stated conditions, does
not validate. TDD §9.1: "the schema rejects a peak-compute entry lacking
conditions" is enforced by `PeakComputeValue`, not by review discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phoenix.provenance import Confidence, PeakComputeValue, ProvenanceValue

DeviceClass = Literal[
    "CPU", "GPU", "TPU", "NPU", "PHOTONIC", "NEUROMORPHIC", "OTHER_ACCELERATOR"
]


class Identity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor: str
    model: str
    family: str | None = None
    architecture: str | None = None
    microarchitecture: str | None = None
    product_codes: list[str] = Field(default_factory=list)
    form_factor: str | None = None
    release_year: ProvenanceValue | None = None


class ComputeUnit(BaseModel):
    """Deliberately generic: an SM, a CU, an MXU, and an MZI mesh have nothing
    structurally in common, so this is a list of typed records, not fixed keys
    (TDD §9.3) — the same shape describes every future backend, not just GPUs."""

    model_config = ConfigDict(extra="forbid")
    kind: str
    count: ProvenanceValue
    generation: str | None = None


class Clocks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_mhz: ProvenanceValue | None = None
    boost_mhz: ProvenanceValue | None = None


class Precision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    supported: bool
    peak: PeakComputeValue
    native_or_emulated: Literal["NATIVE", "EMULATED"] | None = None


class Compute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: list[ComputeUnit] = Field(default_factory=list)
    clocks: Clocks = Clocks()
    precisions: list[Precision] = Field(default_factory=list)


class MemoryLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str
    technology: str | None = None
    capacity: ProvenanceValue | None = None
    # Theoretical peak bandwidth: TDD §9.2's own example carries
    # `conditions: "theoretical peak"` on this field, so it is held to the same
    # standard as a compute peak, not left as a bare ProvenanceValue.
    bandwidth: PeakComputeValue | None = None
    bandwidth_attained: ProvenanceValue | None = None
    configurable: bool | None = None


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    levels: list[MemoryLevel] = Field(default_factory=list)


class InterconnectLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    generation: str | None = None
    lanes: int | None = None
    bandwidth: PeakComputeValue | None = None
    topology_notes: str | None = None


class Interconnect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: InterconnectLink | None = None
    device_to_device: list[InterconnectLink] = Field(default_factory=list)


class Power(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tdp: ProvenanceValue | None = None
    idle_typical: ProvenanceValue | None = None
    telemetry_available: bool | None = None
    telemetry_mechanism: str | None = None
    telemetry_semantics: str | None = None


class Software(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: str | None = None
    isa: str | None = None
    compute_capability: str | None = None
    compilers: list[str] = Field(default_factory=list)
    vendor_math_libraries: list[str] = Field(default_factory=list)


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
    device_class: DeviceClass
    identity: Identity
    compute: Compute = Compute()
    memory: Memory = Memory()
    interconnect: Interconnect = Interconnect()
    power: Power = Power()
    software: Software = Software()
    # ADR-032: whether THIS specific record's subject, as measured on this dev
    # host, qualifies as a measurement host. Not part of the TDD §9.2 example
    # (which predates ADR-032) but a legitimate per-record extension of it.
    measurement_host_qualified: ProvenanceValue | None = None
    provenance_summary: ProvenanceSummary


def load_device_record(path: Path) -> DeviceRecord:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DeviceRecord.model_validate(raw)


def load_all(hardware_dir: Path) -> list[DeviceRecord]:
    return [load_device_record(p) for p in sorted(hardware_dir.rglob("*.yaml"))]
