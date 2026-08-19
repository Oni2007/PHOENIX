"""Tier 1: raw per-iteration samples (TDD 10.1).

Nothing here is a derived quantity. There is no FLOP/s column - that is Tier 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

PHASE_NAMES = {0: "WARMUP", 1: "MEASURE"}

SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("sample_index", pa.int32(), nullable=False),
        pa.field("repetition_index", pa.int32(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("host_wall_ns", pa.int64(), nullable=False),
        pa.field("device_time_ns", pa.int64(), nullable=True),
        pa.field("api_overhead_ns", pa.int64(), nullable=True),
        pa.field("correctness_checked", pa.bool_(), nullable=False),
        pa.field("max_abs_err", pa.float64(), nullable=True),
        pa.field("max_rel_err", pa.float64(), nullable=True),
        pa.field("power_w", pa.float64(), nullable=True),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
    ]
)


def _nullable_i64(values: np.ndarray) -> pa.Array:
    """-1 is the boundary's null encoding; convert to a real Arrow null."""
    arr = np.asarray(values, dtype=np.int64)
    return pa.array(arr, mask=(arr == -1), type=pa.int64())


def _nullable_f64(values: np.ndarray) -> pa.Array:
    arr = np.asarray(values, dtype=np.float64)
    return pa.array(arr, mask=np.isnan(arr), type=pa.float64())


def to_table(run_id: str, out: Any) -> pa.Table:
    n = len(out)
    phases = np.asarray(out.phase())
    return pa.Table.from_arrays(
        [
            pa.array([run_id] * n, type=pa.string()),
            pa.array(np.asarray(out.sample_index()), type=pa.int32()),
            pa.array(np.asarray(out.repetition_index()), type=pa.int32()),
            pa.array([PHASE_NAMES[int(p)] for p in phases], type=pa.string()),
            pa.array(np.asarray(out.host_wall_ns()), type=pa.int64()),
            _nullable_i64(out.device_time_ns()),
            _nullable_i64(out.api_overhead_ns()),
            pa.array(np.asarray(out.correctness_checked()).astype(bool), type=pa.bool_()),
            _nullable_f64(out.max_abs_err()),
            _nullable_f64(out.max_rel_err()),
            # No characterised power source on this platform: NOT YET MEASURED,
            # recorded as null rather than substituted with an estimate.
            pa.nulls(n, type=pa.float64()),
            pa.array(np.asarray(out.timestamp_ns()), type=pa.int64()),
            pa.array(np.asarray(out.valid()).astype(bool), type=pa.bool_()),
        ],
        schema=SCHEMA,
    )
