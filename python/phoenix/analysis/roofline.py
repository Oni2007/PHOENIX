"""Roofline analysis (TDD §33). Every value here is ANALYTICAL — never MEASURED.

TDD §33 draws a hard line between four elements, visually distinguished by
provenance class:

    theoretical compute roof     VENDOR_REPORTED  (hardware database)
    empirical compute ceiling    MEASURED         (best attained, real runs)
    theoretical bandwidth roof   VENDOR_REPORTED  (hardware database)
    empirical bandwidth ceiling  MEASURED         (a real bandwidth microbenchmark)

This module computes only the first and third — and the workload's own
position against them. It has no execution path: it cannot produce a MEASURED
value, because computing a theoretical bound requires no execution and this
module never executes anything. On a device this project has never run on
(e.g. any GPU), this is the ONLY kind of number that can honestly exist.
"""

from __future__ import annotations

from typing import Literal

from phoenix.discovery.hardware import DeviceRecord, MemoryLevel, Precision
from phoenix.provenance import DerivedMetric


class RooflineInputError(Exception):
    """Raised when the hardware record does not carry what a roofline needs."""


def find_precision(record: DeviceRecord, name: str) -> Precision:
    for p in record.compute.precisions:
        if p.name == name:
            return p
    available = [p.name for p in record.compute.precisions]
    raise RooflineInputError(f"{record.id}: no precision named {name!r}. Available: {available}")


def find_device_memory(record: DeviceRecord) -> MemoryLevel:
    for level in record.memory.levels:
        if level.level == "Device memory":
            return level
    raise RooflineInputError(f"{record.id}: no 'Device memory' level in the hardware record")


def gemm_flops(dim_m: int, dim_n: int, dim_k: int) -> int:
    """Same convention as phoenix.analysis.statistics.gemm_flops: a fused
    multiply-add counts as 2 FLOPs. Duplicated rather than imported to keep
    this module's assumptions self-contained and independently auditable."""
    return 2 * dim_m * dim_n * dim_k


ReuseModel = Literal["no_reuse"]


def gemm_bytes_moved(
    dim_m: int, dim_n: int, dim_k: int, dtype_bytes: float, reuse_model: ReuseModel = "no_reuse"
) -> float:
    """Bytes moved to/from device memory for one GEMM, under a STATED model.

    `no_reuse`: every element of A, B, and C is read/written from device memory
    exactly once — the theoretical minimum-information-required traffic, with
    zero cache reuse credited. This is a LOWER BOUND on real traffic (a real
    kernel that misses cache moves strictly more), so the arithmetic intensity
    computed from it is an UPPER BOUND on the true AI. That direction of error
    is stated explicitly here because getting it backwards would make a
    workload look more compute-bound than it really is.
    """
    if reuse_model != "no_reuse":
        raise RooflineInputError(f"unknown reuse_model: {reuse_model!r}")
    elements_moved = (dim_m * dim_k) + (dim_k * dim_n) + (dim_m * dim_n)
    return elements_moved * dtype_bytes


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise RooflineInputError("bytes_moved must be positive")
    return flops / bytes_moved


def roofline_crossover_ai(peak_flops_per_s: float, peak_bandwidth_bytes_per_s: float) -> float:
    """The arithmetic intensity (FLOP/byte) at which the compute roof and the
    bandwidth roof meet. Below it, a workload is memory-bound; above it,
    compute-bound — the entire point of the roofline model (Williams, Waterman
    & Patterson 2009; PAP-0005 in this project's literature database)."""
    if peak_bandwidth_bytes_per_s <= 0:
        raise RooflineInputError("peak_bandwidth_bytes_per_s must be positive")
    return peak_flops_per_s / peak_bandwidth_bytes_per_s


def attainable_flops_per_s(
    ai: float, peak_flops_per_s: float, peak_bandwidth_bytes_per_s: float
) -> float:
    """min(compute roof, AI x bandwidth roof) — the roofline itself."""
    return min(peak_flops_per_s, ai * peak_bandwidth_bytes_per_s)


def gemm_roofline_point(
    record: DeviceRecord,
    precision_name: str,
    dim_m: int,
    dim_n: int,
    dim_k: int,
    dtype_bytes: float,
) -> DerivedMetric:
    """One GEMM shape's position against one precision's theoretical roofline.

    Returns a DerivedMetric — never MEASURED, by construction (the type itself
    forbids it; see phoenix.provenance.DerivedMetric). The formula and every
    assumption are recorded so this is recomputable and auditable, exactly as
    TDD Tier 3 requires for any derived quantity.
    """
    precision = find_precision(record, precision_name)
    if precision.peak.value is None:
        raise RooflineInputError(
            f"{record.id}: precision {precision_name!r} has no known peak value "
            "(value is null/UNKNOWN) — cannot compute a roofline against it"
        )
    mem = find_device_memory(record)
    if mem.bandwidth is None or mem.bandwidth.value is None:
        raise RooflineInputError(f"{record.id}: no known device-memory bandwidth")

    peak_flops_per_s = float(precision.peak.value) * 1e12  # TFLOP/s -> FLOP/s
    # TB/s in this hardware record means decimal terabytes/s (bytes, not bits).
    peak_bw_bytes_per_s = float(mem.bandwidth.value) * 1e12

    flops = gemm_flops(dim_m, dim_n, dim_k)
    bytes_moved = gemm_bytes_moved(dim_m, dim_n, dim_k, dtype_bytes)
    ai = arithmetic_intensity(flops, bytes_moved)
    crossover = roofline_crossover_ai(peak_flops_per_s, peak_bw_bytes_per_s)
    attainable = attainable_flops_per_s(ai, peak_flops_per_s, peak_bw_bytes_per_s)

    return DerivedMetric(
        metric="roofline_attainable_flops_per_s",
        value=attainable,
        unit="FLOP/s",
        formula="min(peak_flops_per_s, arithmetic_intensity * peak_bandwidth_bytes_per_s)",
        inputs={
            "device_id": record.id,
            "precision": precision_name,
            "M": dim_m,
            "N": dim_n,
            "K": dim_k,
            "dtype_bytes": dtype_bytes,
            "peak_flops_per_s": {
                "source": "hardware_database",
                "device_id": record.id,
                "precision": precision_name,
                "provenance": "VENDOR_REPORTED",
            },
            "peak_bandwidth_bytes_per_s": {
                "source": "hardware_database",
                "device_id": record.id,
                "level": "Device memory",
                "provenance": "VENDOR_REPORTED",
            },
            "arithmetic_intensity_flop_per_byte": ai,
            "roofline_crossover_ai_flop_per_byte": crossover,
            "regime": "compute-bound" if ai >= crossover else "memory-bound",
        },
        assumptions=[
            "PURELY THEORETICAL: this device has never been measured by PHOENIX; "
            "both the compute peak and the memory bandwidth are VENDOR_REPORTED "
            "figures from the hardware database, not measurements.",
            "counts a fused multiply-add as 2 FLOPs",
            "memory traffic uses the no_reuse model: every element of A, B, and "
            "C is charged exactly once, crediting zero cache reuse — a LOWER "
            "bound on real traffic and therefore an UPPER bound on true "
            "arithmetic intensity and attainable performance",
            "assumes boost-clock, dense-unless-named-*_SPARSE peak figures as "
            "recorded in the hardware database, with each figure's own stated "
            "conditions",
            "ignores kernel launch overhead, host-device transfer, and any "
            "non-GEMM portion of a real workload",
        ],
    )
