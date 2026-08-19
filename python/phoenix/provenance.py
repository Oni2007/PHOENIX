"""Provenance as a first-class type (TDD P3 / ADR-004).

It is impossible to store a number in PHOENIX without declaring where it came from.
This is enforced by schema, not by convention.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provenance(StrEnum):
    MEASURED = "MEASURED"
    VENDOR_REPORTED = "VENDOR_REPORTED"
    SIMULATED = "SIMULATED"
    ANALYTICAL = "ANALYTICAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    DERIVED = "DERIVED"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str | None = None
    url: str | None = None
    accessed: str | None = None
    page: str | None = None
    run_id: str | None = None
    experiment_id: str | None = None


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ProvenanceValue(BaseModel):
    """A number that cannot exist without a provenance class.

    `value=None` means UNKNOWN / NOT YET MEASURED, and it is a legal, first-class
    state precisely so nobody is pressured into inventing a number to satisfy a
    required field.
    """

    model_config = ConfigDict(extra="forbid")

    value: float | int | str | None = None
    unit: str | None = None
    provenance: Provenance
    source: Source | None = None
    conditions: str | None = None
    confidence: Confidence | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _numeric_values_need_units(self) -> ProvenanceValue:
        if isinstance(self.value, (int, float)) and not self.unit:
            raise ValueError("a numeric ProvenanceValue must declare a unit")
        return self

    @property
    def is_known(self) -> bool:
        return self.value is not None


class DerivedMetric(BaseModel):
    """Tier 3. A derived value is never MEASURED (TDD P5).

    The formula, its inputs and its assumptions are stored so the value can be
    recomputed if the formula changes. `assumptions` is mandatory and non-empty:
    there is no such thing as a FLOP/s figure without a FLOP-counting convention,
    and unstated conventions are how incomparable numbers get compared.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float | None
    unit: str
    provenance: Provenance = Provenance.DERIVED
    formula: str
    formula_version: str = "1.0.0"
    inputs: dict[str, Any]
    assumptions: list[str] = Field(min_length=1)
    valid: bool = True

    @model_validator(mode="after")
    def _must_be_derived(self) -> DerivedMetric:
        if self.provenance is not Provenance.DERIVED:
            raise ValueError("a DerivedMetric must carry provenance DERIVED, never MEASURED")
        return self


def combine(a: Provenance, b: Provenance) -> Provenance:
    """Combining values of different classes yields DERIVED, never the stronger class.

    Combining MEASURED with VENDOR_REPORTED must not produce MEASURED - that is
    exactly how a vendor figure ends up presented as a measurement.
    """
    return a if a is b else Provenance.DERIVED
