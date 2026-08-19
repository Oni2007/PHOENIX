"""Figures are generated from raw data only. Never hand-edited, never typed in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scaling_figure(analyses: list[dict[str, Any]], out_path: Path, experiment_id: str) -> Path:
    """Median host wall time vs matrix size, with bootstrap CIs.

    A bare median with no dispersion is not an acceptable PHOENIX figure, so the
    CI is drawn, not optional.
    """
    by_impl: dict[str, list[tuple[int, float, float, float]]] = {}
    provisional = False
    for a in analyses:
        if "tier2_aggregate_measure" not in a:
            continue
        key = f"{a['backend']}/{a['implementation']}/{a['workload']['dtype_compute']}"
        agg = a["tier2_aggregate_measure"]
        by_impl.setdefault(key, []).append(
            (a["workload"]["M"], agg["median"], agg["ci_low"], agg["ci_high"])
        )
        if not a["tier4_reported"]["publishable"]:
            provisional = True

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, rows in sorted(by_impl.items()):
        rows.sort()
        xs = [r[0] for r in rows]
        med = [r[1] / 1e6 for r in rows]
        lo = [(r[1] - r[2]) / 1e6 for r in rows]
        hi = [(r[3] - r[1]) / 1e6 for r in rows]
        ax.errorbar(xs, med, yerr=[lo, hi], marker="o", capsize=3, label=key)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size M (= N = K)")
    ax.set_ylabel("host wall time per iteration (ms), median")
    title = f"{experiment_id}: GEMM scaling (MEASURED, median + 95% bootstrap CI)"
    if provisional:
        title += "\nPROVISIONAL - unqualified measurement host, not publishable"
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    run_ids = sorted({a["run_id"][:8] for a in analyses if "run_id" in a})
    fig.text(
        0.01,
        0.01,
        f"{experiment_id} | runs: {len(run_ids)} | generated from samples.parquet",
        fontsize=6,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
