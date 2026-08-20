"""Roofline math (TDD §33) and the H100 hardware record — ANALYTICAL only.

No test here executes anything or claims a measurement. That is the entire
point of this module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phoenix.analysis.roofline import (
    RooflineInputError,
    arithmetic_intensity,
    find_device_memory,
    find_precision,
    gemm_bytes_moved,
    gemm_flops,
    gemm_roofline_point,
    roofline_crossover_ai,
)
from phoenix.discovery.hardware import load_device_record
from phoenix.provenance import Provenance

H100_RECORD = Path("hardware/gpu/nvidia-h100-sxm5-80gb.yaml")


def test_h100_record_loads_and_validates() -> None:
    r = load_device_record(H100_RECORD)
    assert r.id == "nvidia-h100-sxm5-80gb"
    assert r.device_class == "GPU"


def test_every_precision_peak_carries_conditions() -> None:
    """TDD §9.1: enforced by PeakComputeValue's own validator, re-checked here
    against the real record as a belt-and-braces integration test."""
    r = load_device_record(H100_RECORD)
    for p in r.compute.precisions:
        assert p.peak.conditions, f"{p.name} has a known peak but no conditions"


def test_every_numeric_field_is_vendor_reported_or_explicitly_measured_null() -> None:
    """No number in this record may be MEASURED with a non-null value — this
    device has never been run on, so nothing here can honestly be a
    measurement."""
    r = load_device_record(H100_RECORD)
    for p in r.compute.precisions:
        assert p.peak.provenance == Provenance.VENDOR_REPORTED
    mem = find_device_memory(r)
    assert mem.bandwidth is not None
    assert mem.bandwidth.provenance == Provenance.VENDOR_REPORTED
    assert mem.bandwidth_attained is not None
    assert mem.bandwidth_attained.provenance == Provenance.MEASURED
    assert mem.bandwidth_attained.value is None, (
        "bandwidth_attained must be null: PHOENIX has never measured this device"
    )


def test_find_precision_lists_available_names_on_miss() -> None:
    r = load_device_record(H100_RECORD)
    with pytest.raises(RooflineInputError, match="NOT_A_REAL_PRECISION"):
        find_precision(r, "NOT_A_REAL_PRECISION")


def test_gemm_flops_convention() -> None:
    assert gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_gemm_bytes_moved_no_reuse_model() -> None:
    # M=N=K=2, fp32 (4 bytes): (4 + 4 + 4) elements * 4 bytes = 48 bytes
    assert gemm_bytes_moved(2, 2, 2, dtype_bytes=4.0) == 48.0


def test_gemm_bytes_moved_rejects_unknown_reuse_model() -> None:
    with pytest.raises(RooflineInputError):
        gemm_bytes_moved(2, 2, 2, dtype_bytes=4.0, reuse_model="perfect_cache")  # type: ignore[arg-type]


def test_arithmetic_intensity_basic() -> None:
    assert arithmetic_intensity(flops=100.0, bytes_moved=10.0) == 10.0


def test_arithmetic_intensity_rejects_nonpositive_bytes() -> None:
    with pytest.raises(RooflineInputError):
        arithmetic_intensity(flops=1.0, bytes_moved=0.0)


def test_roofline_crossover_is_the_point_where_both_roofs_give_the_same_answer() -> None:
    peak_flops = 1000.0
    peak_bw = 100.0
    crossover = roofline_crossover_ai(peak_flops, peak_bw)
    # Just below crossover: bandwidth-limited value is lower than peak compute.
    assert crossover * peak_bw < peak_flops * 1.0001
    assert crossover == pytest.approx(peak_flops / peak_bw)


def test_gemm_roofline_point_is_never_measured() -> None:
    """A DerivedMetric literally cannot be MEASURED (enforced by its own
    validator) — this asserts the roofline path actually produces one."""
    r = load_device_record(H100_RECORD)
    metric = gemm_roofline_point(r, "FP16_TENSOR_CORE_DENSE", 4096, 4096, 4096, dtype_bytes=2.0)
    assert metric.provenance == Provenance.DERIVED
    assert metric.value is not None
    assert metric.value > 0
    assert any("PURELY THEORETICAL" in a for a in metric.assumptions)


def test_gemm_roofline_point_regime_classification_is_consistent_with_ai() -> None:
    r = load_device_record(H100_RECORD)
    # A tiny GEMM has low arithmetic intensity -> memory-bound.
    small = gemm_roofline_point(r, "FP16_TENSOR_CORE_DENSE", 8, 8, 8, dtype_bytes=2.0)
    assert small.inputs["regime"] == "memory-bound"
    # A huge GEMM has high arithmetic intensity -> compute-bound.
    large = gemm_roofline_point(r, "FP16_TENSOR_CORE_DENSE", 8192, 8192, 8192, dtype_bytes=2.0)
    assert large.inputs["regime"] == "compute-bound"


def test_larger_gemm_has_higher_arithmetic_intensity() -> None:
    """GEMM AI grows with N under the no-reuse model (compute is O(N^3),
    traffic is O(N^2)) — the whole reason larger matrices trend compute-bound."""
    r = load_device_record(H100_RECORD)
    small = gemm_roofline_point(r, "FP32", 64, 64, 64, dtype_bytes=4.0)
    large = gemm_roofline_point(r, "FP32", 2048, 2048, 2048, dtype_bytes=4.0)
    assert (
        large.inputs["arithmetic_intensity_flop_per_byte"]
        > (small.inputs["arithmetic_intensity_flop_per_byte"])
    )


def test_sparse_precision_has_a_higher_compute_roof_than_dense() -> None:
    r = load_device_record(H100_RECORD)
    dense = gemm_roofline_point(r, "FP16_TENSOR_CORE_DENSE", 8192, 8192, 8192, dtype_bytes=2.0)
    sparse = gemm_roofline_point(r, "FP16_TENSOR_CORE_SPARSE", 8192, 8192, 8192, dtype_bytes=2.0)
    assert sparse.value is not None and dense.value is not None
    assert sparse.value >= dense.value


def test_roofline_point_raises_on_a_precision_with_no_known_peak() -> None:
    """Every precision in the real H100 record has a known peak, so this
    negative case is built against a synthetic record with a deliberately
    UNKNOWN (null) peak — an honest, unresolved figure rather than a guess."""
    from phoenix.discovery.hardware import (
        Compute,
        DeviceRecord,
        Identity,
        Precision,
        ProvenanceSummary,
    )
    from phoenix.provenance import PeakComputeValue

    record = DeviceRecord(
        schema_version="1.0.0",
        id="synthetic-test-device",
        device_class="GPU",
        identity=Identity(vendor="Test", model="Synthetic"),
        compute=Compute(
            precisions=[
                Precision(
                    name="FP32",
                    supported=True,
                    peak=PeakComputeValue(value=None, unit="TFLOP/s", provenance="VENDOR_REPORTED"),
                )
            ]
        ),
        provenance_summary=ProvenanceSummary(
            authored_by="test", last_verified="2026-08-20", overall_confidence="LOW"
        ),
    )
    with pytest.raises(RooflineInputError, match="no known peak"):
        gemm_roofline_point(record, "FP32", 128, 128, 128, dtype_bytes=4.0)
