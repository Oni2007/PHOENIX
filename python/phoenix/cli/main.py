"""phoenix - the researcher-facing CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from phoenix.analysis.figures import scaling_figure
from phoenix.analysis.pipeline import analyse_experiment, analyse_run
from phoenix.config.loader import load_config
from phoenix.discovery.environment import (
    collect_environment,
    environment_fingerprint,
)
from phoenix.runner.orchestrator import Orchestrator
from phoenix.store import catalogue as catalogue_mod
from phoenix.store.package import package_complete, verify

app = typer.Typer(add_completion=False, help="PHOENIX benchmark harness")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@app.command()
def discover() -> None:
    """Report exactly what machine we are on. Unknowns are printed as unknown."""
    import phoenix_core

    env = collect_environment(json.loads(phoenix_core.build_info()))
    env["environment_fingerprint"] = environment_fingerprint(env)
    typer.echo(json.dumps(env, indent=2, sort_keys=True))


@app.command()
def validate(config: Path) -> None:
    """Validate an experiment config and its expanded sweep."""
    from phoenix.config.models import expand_sweep

    cfg = load_config(config)
    points = expand_sweep(cfg)
    typer.echo(f"OK  {cfg.experiment_id} v{cfg.experiment_version}: {len(points)} runs")


@app.command()
def run(config: Path, cooldown: float | None = None) -> None:
    """Execute every point of an experiment's sweep."""
    cfg = load_config(config)
    orch = Orchestrator(_repo_root(), config, cfg)
    typer.echo(f"fingerprint: {orch.fingerprint}")
    if orch.environment["measurement_host_disqualifiers"]:
        typer.secho(
            "PROVISIONAL host: " + ", ".join(orch.environment["measurement_host_disqualifiers"]),
            fg=typer.colors.YELLOW,
        )
    outcomes = orch.run_all(cooldown)
    for o in outcomes:
        flags = ",".join(o.invalidation_reasons) or "-"
        typer.echo(f"{o.run_id}  {o.status:9s}  flags={flags}")
    typer.echo(f"{len(outcomes)} runs written under {cfg.output.raw_dir}")


@app.command("verify-package")
def verify_package(run_dir: Path) -> None:
    """Re-hash a sealed package against its manifest."""
    complete, missing = package_complete(run_dir)
    ok, problems = verify(run_dir)
    if not complete:
        typer.secho(f"INCOMPLETE: missing {missing}", fg=typer.colors.RED)
    typer.secho(
        "checksums OK" if ok else f"TAMPERED: {problems}",
        fg=typer.colors.GREEN if ok else typer.colors.RED,
    )
    raise typer.Exit(0 if (ok and complete) else 1)


@app.command()
def analyze(experiment_id: str, results: Path = Path("results/raw")) -> None:
    """Recompute aggregates, derived metrics and reported values from raw data."""
    root = _repo_root() / results if not results.is_absolute() else results
    analyses = analyse_experiment(root, experiment_id)
    typer.echo(json.dumps(analyses, indent=2, sort_keys=True, default=str))


@app.command()
def catalogue(
    results: Path = Path("results/raw"), out: Path = Path("results/catalogue/phoenix.duckdb")
) -> None:
    """Rebuild the DuckDB catalogue from Parquet. Disposable by design."""
    root = _repo_root()
    info = catalogue_mod.rebuild(root / results, root / out)
    typer.echo(json.dumps(info, indent=2))


@app.command()
def report(
    experiment_id: str, results: Path = Path("results/raw"), figures: Path = Path("results/figures")
) -> None:
    """Generate the experiment figure and summary from stored raw data only."""
    root = _repo_root()
    analyses = analyse_experiment(root / results, experiment_id)
    if not analyses:
        typer.secho(f"no runs found for {experiment_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    path = scaling_figure(analyses, root / figures / f"{experiment_id}_scaling.png", experiment_id)
    typer.echo(f"figure: {path}")


@app.command("run-info")
def run_info(run_dir: Path) -> None:
    """Analyse a single run directory."""
    typer.echo(json.dumps(analyse_run(run_dir), indent=2, sort_keys=True, default=str))


@app.command("papers")
def papers(
    papers_dir: Path = Path("research/papers"),
    csv_out: Path = Path("research/papers.csv"),
) -> None:
    """Validate the literature corpus and regenerate research/papers.csv.

    The YAML files are the source of truth; the CSV is a generated export.
    Editing the CSV directly is a no-op that gets overwritten.
    """
    from phoenix.literature.store import corpus_summary, export_csv, load_all

    root = _repo_root()
    corpus = load_all(root / papers_dir)
    export_csv(corpus, root / csv_out)
    summary = corpus_summary(corpus)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if summary["abstract_only"] > 0:
        typer.secho(
            f"{summary['abstract_only']} paper(s) at ABSTRACT_ONLY — never cite these in "
            "PHOENIX output (TDD §24)",
            fg=typer.colors.YELLOW,
        )


@app.command("roofline")
def roofline(
    hardware_record: Path,
    precision: str,
    sizes: str = "128,256,512,1024,2048,4096,8192",
    dtype_bytes: float = 2.0,
) -> None:
    """ANALYTICAL roofline for a GEMM sweep against a hardware record's peak specs.

    PURELY THEORETICAL. This command executes nothing and measures nothing —
    it computes what a GEMM's arithmetic intensity and attainable throughput
    WOULD be against a hardware record's own VENDOR_REPORTED peak figures. If
    the record describes a device this project has never run on, that fact is
    printed as a standing warning below, not left implicit.
    """
    from phoenix.analysis.roofline import gemm_roofline_point
    from phoenix.discovery.hardware import load_device_record

    record = load_device_record(hardware_record)
    dims = [int(s.strip()) for s in sizes.split(",")]

    typer.secho(
        "═" * 78 + "\nPURELY THEORETICAL — no execution, no measurement.\n"
        f"'{record.id}' has never been run on by PHOENIX (see the hardware record's own "
        "header). Every number below is derived from VENDOR_REPORTED peak specs.\n" + "═" * 78,
        fg=typer.colors.YELLOW,
        bold=True,
    )

    rows = []
    for n in dims:
        metric = gemm_roofline_point(record, precision, n, n, n, dtype_bytes)
        rows.append(metric.model_dump(mode="json"))
        regime = metric.inputs["regime"]
        ai = metric.inputs["arithmetic_intensity_flop_per_byte"]
        gflops = metric.value / 1e9 if metric.value is not None else float("nan")
        typer.echo(
            f"M=N=K={n:<6d} AI={ai:8.2f} FLOP/B  {regime:13s}  attainable={gflops:10.1f} GFLOP/s"
        )

    if rows:
        crossover = rows[0]["inputs"]["roofline_crossover_ai_flop_per_byte"]
        typer.echo(
            f"\nroofline crossover for {precision}: {crossover:.2f} FLOP/byte "
            "(below this AI, memory-bound; above it, compute-bound)"
        )
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
