"""T5: the complete path - config -> run -> store -> analysis -> figure.

Runs the real C++ backend on a tiny workload, so it is fast enough for every PR
and still exercises every boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("phoenix_core")

from phoenix.analysis.figures import scaling_figure
from phoenix.analysis.pipeline import analyse_experiment, analyse_run
from phoenix.config.loader import load_config
from phoenix.runner.orchestrator import Orchestrator
from phoenix.store.package import package_complete, verify

TINY = """
schema_version: "1.0.0"
experiment_id: "EXP-999"
experiment_version: "0.0.1"
title: "integration smoke"
hypothesis: |
  The pipeline runs end to end. This is a HYPOTHESIS about the harness,
  not a result about any CPU.
workload:
  kind: GEMM
  sweep:
    dimensions:
      - {M: 32, N: 32, K: 32}
      - {M: 48, N: 48, K: 48}
    dtype:
      - {a: fp64, b: fp64, c: fp64, compute: fp64}
  alpha: 1.0
  beta: 0.0
  seed: 7
execution:
  backends: ["cpu"]
  implementations: ["gemm_blocked"]
measurement:
  warmup_iterations: 2
  measure_iterations: 5
  repetitions: 1
correctness:
  enabled: true
validity:
  min_valid_samples: 1
output:
  raw_dir: "RAW_DIR"
provenance:
  author: "integration test"
  created: "2026-08-19"
"""


@pytest.fixture
def experiment(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(TINY.replace("RAW_DIR", str(raw)))
    return cfg_path, raw


def test_full_path_config_to_figure(experiment: tuple[Path, Path], tmp_path: Path) -> None:
    cfg_path, raw = experiment
    cfg = load_config(cfg_path)
    orch = Orchestrator(tmp_path, cfg_path, cfg)
    outcomes = orch.run_all(cooldown_s=0.0)

    assert len(outcomes) == 2
    for o in outcomes:
        complete, missing = package_complete(o.run_dir)
        assert complete, f"incomplete package: {missing}"
        ok, problems = verify(o.run_dir)
        assert ok, problems

    analyses = analyse_experiment(raw, "EXP-999")
    assert len(analyses) == 2
    for a in analyses:
        assert a["tier2_aggregate_measure"]["n"] == 5
        assert a["tier3_derived"][0]["provenance"] == "DERIVED"
        # This host cannot produce a publishable measurement (ADR-032).
        assert a["tier4_reported"]["publishable"] is False

    fig = scaling_figure(analyses, tmp_path / "fig.png", "EXP-999")
    assert fig.exists() and fig.stat().st_size > 0


def test_figure_changes_when_a_raw_sample_changes(
    experiment: tuple[Path, Path], tmp_path: Path
) -> None:
    """The DoD requires proving figures come from raw data, not from a cache."""
    import pyarrow.parquet as pq

    cfg_path, _raw = experiment
    cfg = load_config(cfg_path)
    orch = Orchestrator(tmp_path, cfg_path, cfg)
    outcomes = orch.run_all(cooldown_s=0.0)

    before = analyse_run(outcomes[0].run_dir)["tier2_aggregate_measure"]["median"]

    samples = outcomes[0].run_dir / "samples.parquet"
    samples.chmod(0o644)
    table = pq.read_table(samples)
    col = [v * 3 for v in table.column("host_wall_ns").to_pylist()]
    import pyarrow as pa

    table = table.set_column(
        table.schema.get_field_index("host_wall_ns"),
        "host_wall_ns",
        pa.array(col, type=pa.int64()),
    )
    pq.write_table(table, samples)

    after = analyse_run(outcomes[0].run_dir)["tier2_aggregate_measure"]["median"]
    assert after == pytest.approx(before * 3, rel=0.01)

    # ...and the tamper is caught by the manifest.
    ok, problems = verify(outcomes[0].run_dir)
    assert not ok
    assert any("CHECKSUM_MISMATCH" in p for p in problems)


def test_invalid_run_is_still_written(tmp_path: Path) -> None:
    """An invalid run is data about the measurement environment, never discarded."""
    raw = tmp_path / "raw"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        TINY.replace("RAW_DIR", str(raw)).replace(
            "  min_valid_samples: 1", "  min_valid_samples: 9999"
        )
    )
    cfg = load_config(cfg_path)
    outcomes = Orchestrator(tmp_path, cfg_path, cfg).run_all(cooldown_s=0.0)
    for o in outcomes:
        assert o.status == "INVALID"
        assert "INSUFFICIENT_SAMPLES" in o.invalidation_reasons
        record = json.loads((o.run_dir / "results.json").read_text())
        assert record["status"] == "INVALID"
        assert (o.run_dir / "samples.parquet").exists(), "invalid runs keep their data"
