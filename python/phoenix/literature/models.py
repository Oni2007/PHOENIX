"""Research-paper database schema (TDD §24).

Not a bibliography — an evidence database. The load-bearing fields are
`claims[].scope` and `claims[].excludes`: the dominant failure mode in
photonic-computing literature comparison is an optical-core energy figure
compared against a full-system GPU figure. Recording what a number includes and
excludes is what lets PHOENIX later assemble an honest cross-paper comparison
instead of an apples-to-oranges table.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from phoenix.provenance import Confidence


class VenueType(StrEnum):
    CONFERENCE = "CONFERENCE"
    JOURNAL = "JOURNAL"
    WORKSHOP = "WORKSHOP"
    PREPRINT = "PREPRINT"
    WHITEPAPER = "WHITEPAPER"
    OTHER = "OTHER"


class Venue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: VenueType
    tier_notes: str | None = None


class Identifiers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doi: str | None = None
    arxiv: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _at_least_one_identifier(self) -> Identifiers:
        if not any([self.doi, self.arxiv, self.url]):
            raise ValueError("a paper record needs at least one of doi/arxiv/url")
        return self


class ReproducibleStatus(StrEnum):
    YES = "YES"
    PARTIAL = "PARTIAL"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str | None = None
    data: str | None = None
    reproducible: ReproducibleStatus = ReproducibleStatus.UNKNOWN


class HardwareStudied(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["CPU", "GPU", "TPU", "NPU", "PHOTONIC", "NEUROMORPHIC", "OTHER_ACCELERATOR"]
    description: str
    fabrication: str | None = None
    scale: str | None = None


class MethodologyType(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    SIMULATION = "SIMULATION"
    ANALYTICAL = "ANALYTICAL"
    SURVEY = "SURVEY"


class Methodology(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: MethodologyType
    measured_on_physical_hardware: bool
    simulator: str | None = None
    workloads: list[str] = Field(default_factory=list)
    precision: str | None = None
    baseline_compared_against: str | None = None
    baseline_fairness_assessment: str | None = None


class EvidenceClass(StrEnum):
    """A paper's own evidence class — deliberately not PHOENIX's Provenance enum.

    A claim extracted from a paper is evidence *about* something PHOENIX did not
    itself measure. PHOENIX's own Provenance enum (MEASURED/VENDOR_REPORTED/...)
    describes PHOENIX's data; this describes the paper's. PROJECTED is meaningful
    here (a paper's own extrapolation) and has no PHOENIX-side equivalent.
    """

    MEASURED = "MEASURED"
    SIMULATED = "SIMULATED"
    ANALYTICAL = "ANALYTICAL"
    PROJECTED = "PROJECTED"


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def _numeric_value_needs_unit(self) -> Quantity:
        if self.value is not None and not self.unit:
            raise ValueError("a numeric claim quantity must declare a unit")
        return self


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    statement: str
    quantity: Quantity | None = None
    evidence_class: EvidenceClass
    scope: str = Field(min_length=1, description="what this number DOES include")
    excludes: list[str] = Field(
        default_factory=list, description="what this number does NOT include"
    )
    uncertainty_reported: bool = False
    confidence_in_claim: Confidence


class Limitations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stated_by_authors: list[str] = Field(default_factory=list)
    identified_by_phoenix: list[str] = Field(default_factory=list)


class Relevance(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PhoenixRelevance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevance: Relevance
    informs_hypotheses: list[str] = Field(default_factory=list)
    reproducible_by_phoenix: ReproducibleStatus = ReproducibleStatus.UNKNOWN
    follow_up_experiments: list[str] = Field(default_factory=list)


class ReadStatus(StrEnum):
    FULL_READ = "FULL_READ"
    SKIMMED = "SKIMMED"
    ABSTRACT_ONLY = "ABSTRACT_ONLY"


class PaperProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    added_by: str
    added_on: str
    read_status: ReadStatus
    verified_by: str | None = None


class Paper(BaseModel):
    """One paper. One YAML file. Schema-validated, not free text (TDD §24)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    paper_id: str = Field(pattern=r"^PAP-\d{4}$")
    title: str
    authors: list[str] = Field(min_length=1)
    year: int | None = None
    venue: Venue
    identifiers: Identifiers
    artifacts: Artifacts = Artifacts()

    categories: list[str] = Field(default_factory=list)
    hardware_studied: list[HardwareStudied] = Field(default_factory=list)
    methodology: Methodology

    claims: list[Claim] = Field(default_factory=list)
    limitations: Limitations = Limitations()
    phoenix_relevance: PhoenixRelevance
    provenance: PaperProvenance

    @model_validator(mode="after")
    def _abstract_only_cannot_carry_measured_claims(self) -> Paper:
        # TDD §24: "A paper may never be cited in PHOENIX output at ABSTRACT_ONLY
        # status." Enforced here at the point a claim is attached, not by
        # convention: an abstract does not contain the evidence needed to
        # responsibly extract a MEASURED/SIMULATED numeric claim.
        if self.provenance.read_status == ReadStatus.ABSTRACT_ONLY and self.claims:
            raise ValueError(
                f"{self.paper_id}: read_status is ABSTRACT_ONLY but claims[] is "
                "non-empty — a paper may never be cited at ABSTRACT_ONLY status "
                "(TDD §24). Extract claims only after SKIMMED or FULL_READ."
            )
        return self

    @model_validator(mode="after")
    def _claim_ids_are_unique_and_namespaced(self) -> Paper:
        seen: set[str] = set()
        for c in self.claims:
            if not c.claim_id.startswith(f"{self.paper_id}-"):
                raise ValueError(f"claim {c.claim_id!r} must be namespaced under {self.paper_id}-")
            if c.claim_id in seen:
                raise ValueError(f"duplicate claim_id {c.claim_id!r} within {self.paper_id}")
            seen.add(c.claim_id)
        return self


__all__ = [
    "Artifacts",
    "Claim",
    "EvidenceClass",
    "HardwareStudied",
    "Identifiers",
    "Limitations",
    "Methodology",
    "MethodologyType",
    "Paper",
    "PaperProvenance",
    "PhoenixRelevance",
    "Quantity",
    "ReadStatus",
    "Relevance",
    "ReproducibleStatus",
    "Venue",
    "VenueType",
]
