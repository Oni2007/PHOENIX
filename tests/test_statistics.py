from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from phoenix.analysis.statistics import achieved_flops, aggregate, flag_outliers, gemm_flops
from phoenix.provenance import Provenance


def test_aggregate_reports_dispersion_alongside_the_median() -> None:
    agg = aggregate(np.array([10.0, 11.0, 12.0, 13.0, 100.0]))
    for key in ("median", "mad", "iqr", "cv", "ci_low", "ci_high", "stddev"):
        assert key in agg, f"a median without {key} is not an acceptable report"


def test_median_is_robust_where_the_mean_is_not() -> None:
    clean = np.array([10.0] * 20)
    with_spike = np.append(clean, 10_000.0)
    assert aggregate(with_spike)["median"] == aggregate(clean)["median"]
    assert aggregate(with_spike)["mean"] > aggregate(clean)["mean"]


def test_bootstrap_ci_brackets_the_median() -> None:
    rng = np.random.default_rng(0)
    agg = aggregate(rng.normal(100.0, 5.0, 500))
    assert agg["ci_low"] <= agg["median"] <= agg["ci_high"]


def test_bootstrap_ci_is_reproducible_from_a_fixed_seed() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert aggregate(x)["ci_low"] == aggregate(x)["ci_low"]


def test_outliers_are_flagged_not_removed() -> None:
    x = np.array([10.0] * 20 + [10_000.0])
    flags = flag_outliers(x)
    assert flags.sum() == 1
    assert flags.shape == x.shape, "flagging must not change the sample count"


def test_empty_sample_set_raises_rather_than_inventing_a_statistic() -> None:
    with pytest.raises(ValueError):
        aggregate(np.array([]))


def test_gemm_flop_convention() -> None:
    assert gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_achieved_flops_is_derived_and_carries_its_formula() -> None:
    m = achieved_flops(M=64, N=64, K=64, seconds=0.001, run_id="r")
    assert m.provenance is Provenance.DERIVED
    assert m.formula == "(2*M*N*K) / t_seconds"
    assert m.assumptions
    assert m.value == pytest.approx(2 * 64**3 / 0.001)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(min_value=1.0, max_value=1e9, allow_nan=False), min_size=1, max_size=200))
def test_aggregate_invariants_hold_for_any_sample_set(values: list[float]) -> None:
    agg = aggregate(np.array(values), resamples=200)
    assert agg["min"] <= agg["median"] <= agg["max"]
    assert agg["p50"] <= agg["p90"] <= agg["p95"] <= agg["p99"]
    assert agg["n"] == len(values)
