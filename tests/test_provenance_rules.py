"""Invariants that must hold by test, not by discipline (TDD 20)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from phoenix.provenance import DerivedMetric, Provenance, ProvenanceValue, combine


def test_derived_metric_can_never_be_measured() -> None:
    with pytest.raises(ValidationError):
        DerivedMetric(
            metric="achieved_flops",
            value=1.0,
            unit="FLOP/s",
            provenance=Provenance.MEASURED,
            formula="x",
            inputs={},
            assumptions=["a"],
        )


def test_derived_metric_requires_at_least_one_assumption() -> None:
    with pytest.raises(ValidationError):
        DerivedMetric(
            metric="achieved_flops",
            value=1.0,
            unit="FLOP/s",
            formula="(2*M*N*K)/t",
            inputs={},
            assumptions=[],
        )


def test_numeric_value_without_unit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProvenanceValue(value=42.0, provenance=Provenance.VENDOR_REPORTED)


def test_null_value_is_legal_and_keeps_its_class() -> None:
    v = ProvenanceValue(value=None, provenance=Provenance.VENDOR_REPORTED, notes="UNKNOWN")
    assert not v.is_known
    assert v.provenance is Provenance.VENDOR_REPORTED


@given(st.sampled_from(list(Provenance)), st.sampled_from(list(Provenance)))
def test_combining_classes_never_strengthens_provenance(a: Provenance, b: Provenance) -> None:
    result = combine(a, b)
    if a is b:
        assert result is a
    else:
        # Mixing MEASURED with VENDOR_REPORTED must never yield MEASURED.
        assert result is Provenance.DERIVED


def test_provenance_value_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ProvenanceValue(value=None, provenance=Provenance.MEASURED, definitely_measured=True)
