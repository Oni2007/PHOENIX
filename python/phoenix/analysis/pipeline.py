"""Tier 1 -> Tier 2 -> Tier 3 -> Tier 4, computed from stored raw data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from phoenix.analysis.statistics import achieved_flops, aggregate, flag_outliers


def analyse_run(run_dir: Path) -> dict[str, Any]:
    """Everything below is recomputed from samples.parquet on every call.

    Tiers 2-4 are never cached authoritatively: if they are lost, nothing is lost.
    """
    record = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    table = pq.read_table(run_dir / "samples.parquet")

    phase = np.asarray(table.column("phase").to_pylist())
    host_ns = np.asarray(table.column("host_wall_ns").to_pylist(), dtype=np.float64)

    measure = host_ns[phase == "MEASURE"]
    warmup = host_ns[phase == "WARMUP"]
    if measure.size == 0:
        return {"run_id": record["run_id"], "error": "no MEASURE samples"}

    agg = aggregate(measure)
    outliers = flag_outliers(measure)
    agg_excl = aggregate(measure[~outliers]) if (~outliers).any() and outliers.any() else None

    w = record["workload"]
    derived = achieved_flops(
        M=w["M"],
        N=w["N"],
        K=w["K"],
        seconds=agg["median"] / 1e9,
        run_id=record["run_id"],
    )

    # Tier 4: what a figure would show, and WHY that statistic.
    reported = {
        "value": agg["median"],
        "unit": "ns",
        "statistic": "median",
        "selection_rationale": (
            "median, because benchmark timing distributions are bounded below by a "
            "physical minimum and unbounded above; the mean is dragged by the tail"
        ),
        "dispersion": {"ci_low": agg["ci_low"], "ci_high": agg["ci_high"], "mad": agg["mad"]},
        "provenance": "MEASURED",
        "publishable": record["publishable"],
        "comparability_note": (
            "PROVISIONAL - " + ", ".join(record["provisional_reasons"])
            if record["provisional"]
            else "environment fingerprint " + record["environment_fingerprint"]
        ),
    }

    return {
        "run_id": record["run_id"],
        "experiment_id": record["experiment_id"],
        "backend": record["execution"]["backend"],
        "implementation": record["execution"]["implementation"],
        "workload": w,
        "status": record["status"],
        "invalidation_reasons": record["invalidation_reasons"],
        "tier2_aggregate_measure": agg,
        "tier2_aggregate_measure_outliers_excluded": agg_excl,
        "tier2_aggregate_warmup": aggregate(warmup) if warmup.size else None,
        "tier2_outliers_flagged": int(outliers.sum()),
        "tier3_derived": [derived.model_dump(mode="json")],
        "tier4_reported": reported,
    }


def analyse_experiment(results_root: Path, experiment_id: str) -> list[dict[str, Any]]:
    analyses = []
    for results_json in sorted(results_root.rglob("results.json")):
        record = json.loads(results_json.read_text(encoding="utf-8"))
        if record["experiment_id"] == experiment_id:
            analyses.append(analyse_run(results_json.parent))
    return analyses
