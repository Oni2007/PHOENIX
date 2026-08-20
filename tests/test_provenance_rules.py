"""Invariants that must hold by test, not by discipline (TDD 20)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from phoenix.provenance import DerivedMetric, PeakComputeValue, Provenance, ProvenanceValue, combine


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


def test_boolean_value_does_not_require_a_unit() -> None:
    """Regression: bool is a subclass of int in Python, so isinstance(False, int)
    is True. A naive numeric check wrongly demanded a unit on a boolean-valued
    field (e.g. measurement_host_qualified) — found while validating the CPU
    hardware record against the tightened DeviceRecord schema."""
    v = ProvenanceValue(value=False, provenance=Provenance.MEASURED, notes="host disqualified")
    assert v.value is False
    v2 = ProvenanceValue(value=True, provenance=Provenance.MEASURED)
    assert v2.value is True


def test_peak_compute_value_requires_conditions_when_value_is_known() -> None:
    """TDD §9.1: 'the schema rejects a peak-compute entry lacking conditions.'"""
    with pytest.raises(ValidationError, match="conditions"):
        PeakComputeValue(value=989.0, unit="TFLOP/s", provenance=Provenance.VENDOR_REPORTED)


def test_peak_compute_value_allows_null_without_conditions() -> None:
    # An UNKNOWN peak-compute value has nothing to state conditions about yet.
    v = PeakComputeValue(value=None, unit="TFLOP/s", provenance=Provenance.VENDOR_REPORTED)
    assert not v.is_known


def test_peak_compute_value_accepts_a_known_value_with_conditions() -> None:
    v = PeakComputeValue(
        value=989.0,
        unit="TFLOP/s",
        provenance=Provenance.VENDOR_REPORTED,
        conditions="dense, no sparsity, SXM5 boost clock",
    )
    assert v.value == 989.0
