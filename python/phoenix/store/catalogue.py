"""DuckDB catalogue over the Parquet corpus (ADR-007).

The catalogue is a QUERY LAYER, never a copy. The Parquet files under results/raw/
are the system of record; this file is disposable and fully rebuildable, so a
corrupted database costs nothing and cannot drift from the raw data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


def rebuild(results_root: Path, catalogue_path: Path) -> dict[str, Any]:
    catalogue_path.parent.mkdir(parents=True, exist_ok=True)
    if catalogue_path.exists():
        catalogue_path.unlink()

    runs: list[dict[str, Any]] = []
    for results_json in sorted(results_root.rglob("results.json")):
        runs.append(json.loads(results_json.read_text(encoding="utf-8")))

    con = duckdb.connect(str(catalogue_path))
    try:
        pattern = str(results_root / "*" / "samples.parquet")
        if any(results_root.rglob("samples.parquet")):
            # DuckDB does not accept a bound parameter inside CREATE VIEW, so the
            # path is escaped and inlined rather than passed as a parameter.
            escaped = pattern.replace("'", "''")
            con.execute(
                "CREATE VIEW raw_sample AS "
                f"SELECT * FROM read_parquet('{escaped}', union_by_name=true)"
            )
        con.execute(
            """
            CREATE TABLE benchmark_run (
                run_id VARCHAR, experiment_id VARCHAR, status VARCHAR,
                backend VARCHAR, implementation VARCHAR,
                M BIGINT, N BIGINT, K BIGINT, dtype_compute VARCHAR,
                repetition_index INTEGER, environment_fingerprint VARCHAR,
                provisional BOOLEAN, publishable BOOLEAN,
                correctness_passed BOOLEAN
            )
            """
        )
        for r in runs:
            con.execute(
                "INSERT INTO benchmark_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    r["run_id"],
                    r["experiment_id"],
                    r["status"],
                    r["execution"]["backend"],
                    r["execution"]["implementation"],
                    r["workload"]["M"],
                    r["workload"]["N"],
                    r["workload"]["K"],
                    r["workload"]["dtype_compute"],
                    r["execution"]["repetition_index"],
                    r["environment_fingerprint"],
                    r["provisional"],
                    r["publishable"],
                    r["correctness"]["passed"],
                ],
            )
        n_samples = 0
        if any(results_root.rglob("samples.parquet")):
            row = con.execute("SELECT count(*) FROM raw_sample").fetchone()
            n_samples = int(row[0]) if row is not None else 0
    finally:
        con.close()
    return {"runs": len(runs), "samples": n_samples, "catalogue": str(catalogue_path)}
