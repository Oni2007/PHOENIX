"""Run lifecycle: config -> request -> C++ execution -> sealed package."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import phoenix_core

from phoenix.config.loader import normalise
from phoenix.config.models import ExperimentConfig, RunPoint, expand_sweep
from phoenix.discovery.environment import (
    collect_environment,
    collect_git,
    environment_fingerprint,
)
from phoenix.store import samples as samples_mod
from phoenix.store.package import PackageWriter

REQUEST_SCHEMA_VERSION = "1.0.0"


def build_request(cfg: ExperimentConfig, point: RunPoint) -> dict[str, Any]:
    """The exact object that crosses the boundary. Stored verbatim in the package."""
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "experiment_id": cfg.experiment_id,
        "backend": point.backend,
        "implementation": point.implementation,
        "workload": {
            "kind": "GEMM",
            "M": point.dimensions.M,
            "N": point.dimensions.N,
            "K": point.dimensions.K,
            "dtype_a": point.dtype.a.value,
            "dtype_b": point.dtype.b.value,
            "dtype_c": point.dtype.c.value,
            "dtype_compute": point.dtype.compute.value,
            "alpha": cfg.workload.alpha,
            "beta": cfg.workload.beta,
            "seed": cfg.workload.seed,
        },
        "measurement": {
            "warmup_iterations": cfg.measurement.warmup_iterations,
            "measure_iterations": cfg.measurement.measure_iterations,
            "repetition_index": point.repetition_index,
            "retain_warmup_samples": cfg.output.retain_warmup_samples,
            "min_valid_samples": cfg.validity.min_valid_samples,
            "max_cv": cfg.validity.max_cv,
        },
        "correctness": {
            "enabled": cfg.correctness.enabled,
            "reference": cfg.correctness.reference,
            "tolerance_c": cfg.correctness.tolerance_c,
        },
    }


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    status: str
    invalidation_reasons: list[str]
    provisional: bool


class Orchestrator:
    def __init__(self, repo_root: Path, config_path: Path, cfg: ExperimentConfig) -> None:
        self.repo_root = repo_root
        self.config_path = config_path
        self.cfg = cfg
        self.build_info = json.loads(phoenix_core.build_info())
        self.environment = collect_environment(self.build_info)
        self.fingerprint = environment_fingerprint(self.environment)
        self.git = collect_git(repo_root)

    def points(self) -> list[RunPoint]:
        return expand_sweep(self.cfg)

    def run_point(self, point: RunPoint) -> RunOutcome:
        run_id = str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())
        run_dir = self.repo_root / self.cfg.output.raw_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        writer = PackageWriter(run_dir)

        request = build_request(self.cfg, point)
        request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))

        started = datetime.now(UTC).isoformat()
        out = phoenix_core.execute(request_json)
        finished = datetime.now(UTC).isoformat()

        result = json.loads(out.result_json)
        calibration = json.loads(out.calibration_json)

        table = samples_mod.to_table(run_id, out)
        writer.write_samples(table)

        writer.write_text("config.yaml", self.config_path.read_text(encoding="utf-8"))
        writer.write_text("config.normalized.json", normalise(self.cfg))
        writer.write_text("request.json", request_json)
        writer.write_json("environment.json", self.environment)
        writer.write_json("git.json", self.git)
        writer.write_json("calibration.json", calibration)

        provisional = (
            bool(self.git["provisional"]) or not self.environment["measurement_host_qualified"]
        )
        record = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": self.cfg.experiment_id,
            "experiment_version": self.cfg.experiment_version,
            "experiment_definition_commit": self.git["commit"],
            "status": result["status"],
            "invalidation_reasons": result["invalidation_reasons"],
            "started_at": started,
            "finished_at": finished,
            "environment_fingerprint": self.fingerprint,
            "workload": request["workload"],
            "execution": {
                "backend": result["backend"],
                "implementation": result["implementation"],
                "warmup_iters": self.cfg.measurement.warmup_iterations,
                "measure_iters": self.cfg.measurement.measure_iterations,
                "repetition_index": point.repetition_index,
                "timing_method": self.cfg.measurement.timing_method,
                "prepare_ns": result["prepare_ns"],
                "first_execution_ns": result["first_execution_ns"],
            },
            "correctness": result["correctness"],
            "error": result["error"],
            "timing_valid_for_reporting": result["timing_valid_for_reporting"],
            # ADR-032: a run from an unqualified host is never publishable, however
            # clean its statistics look.
            "provisional": provisional,
            "provisional_reasons": (
                (["DIRTY_TREE"] if self.git["provisional"] else [])
                + list(self.environment["measurement_host_disqualifiers"])
            ),
            "publishable": not provisional,
        }
        writer.write_json("results.json", record)
        writer.seal()

        return RunOutcome(
            run_id=run_id,
            run_dir=run_dir,
            status=result["status"],
            invalidation_reasons=result["invalidation_reasons"],
            provisional=provisional,
        )

    def run_all(self, cooldown_s: float | None = None) -> list[RunOutcome]:
        cd = self.cfg.measurement.inter_repetition_cooldown_s if cooldown_s is None else cooldown_s
        outcomes: list[RunOutcome] = []
        pts = self.points()
        for i, p in enumerate(pts):
            outcomes.append(self.run_point(p))
            if cd > 0 and i < len(pts) - 1:
                time.sleep(cd)
        return outcomes
