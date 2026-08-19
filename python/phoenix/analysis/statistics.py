"""Tier 2 aggregates and Tier 3 derived metrics (TDD 10.2, 10.3, 16)."""

from __future__ import annotations

from typing import Any

import numpy as np

from phoenix.provenance import DerivedMetric


def aggregate(
    samples_ns: np.ndarray, ci_level: float = 0.95, resamples: int = 10_000, seed: int = 20260819
) -> dict[str, Any]:
    """Descriptive statistics plus a bootstrap CI on the median.

    Median is the primary statistic: benchmark timing distributions are bounded
    below by a physical minimum and unbounded above, because interference has no
    ceiling. The mean is dragged by the tail; the minimum pretends interference
    does not exist. Reporting a median without its dispersion is forbidden, so the
    dispersion fields are not optional here.
    """
    x = np.asarray(samples_ns, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot aggregate an empty sample set")

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0

    ci_low, ci_high = _bootstrap_median_ci(x, ci_level, resamples, seed)

    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": mean,
        "median": median,
        "stddev": sd,
        "p50": median,
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "mad": mad,
        "cv": (sd / mean) if mean > 0 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": f"percentile bootstrap, {resamples} resamples, seed={seed}",
        "ci_level": ci_level,
        "provenance": "MEASURED",
    }


def _bootstrap_median_ci(
    x: np.ndarray, level: float, resamples: int, seed: int
) -> tuple[float, float]:
    """Bootstrap rather than a parametric interval.

    The distributions are non-normal and right-skewed; assuming normality produces
    intervals that are wrong in the direction that flatters the result.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(resamples, x.size))
    medians = np.median(x[idx], axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def flag_outliers(samples_ns: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """MAD-based outlier FLAGS. Nothing is ever deleted.

    MAD is zero whenever more than half the samples are identical - which happens
    with a coarse clock, where many iterations land on the same tick. A naive MAD
    rule reports "no outliers" for [10, 10, ..., 10, 10000], hiding the one sample
    that matters. So a zero MAD falls back to the mean absolute deviation, and only
    a genuinely zero-dispersion sample set yields no flags.
    """
    x = np.asarray(samples_ns, dtype=np.float64)
    median = np.median(x)
    deviations = np.abs(x - median)

    # 0.6745 scales MAD to a normal-consistent standard deviation estimate.
    scale = np.median(deviations) / 0.6745
    if scale == 0:
        scale = float(np.mean(deviations))
    if scale == 0:
        # Every sample is identical: there is no dispersion, so nothing is an outlier.
        return np.zeros(x.shape, dtype=bool)
    flags: np.ndarray = (deviations / scale) > threshold
    return flags


def gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def achieved_flops(
    M: int, N: int, K: int, seconds: float, run_id: str, statistic: str = "median"
) -> DerivedMetric:
    """Tier 3. Never MEASURED - the timing is measured, this is computed from it."""
    flops = gemm_flops(M, N, K)
    return DerivedMetric(
        metric="achieved_flops",
        value=(flops / seconds) if seconds > 0 else None,
        unit="FLOP/s",
        formula="(2*M*N*K) / t_seconds",
        inputs={
            "M": M,
            "N": N,
            "K": K,
            "t_seconds": {
                "source": "aggregate",
                "run_id": run_id,
                "statistic": statistic,
                "metric": "host_wall_ns",
            },
        },
        assumptions=[
            "counts a fused multiply-add as 2 FLOPs",
            "FLOP count excludes the alpha/beta epilogue",
            "t_seconds is host wall time, which on this backend includes harness overhead",
        ],
    )
