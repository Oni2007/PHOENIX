"""Generates the H100 analytical roofline report as a script, not a hand-typed
document — the same 'never hand-edit a data file, generate figures from raw
data' discipline as everything under results/, applied to a theoretical report
instead of a measured one. Run via:

    python -m phoenix.analysis.roofline_report > docs/analysis/nvidia-h100-sxm5-roofline.md
"""

from __future__ import annotations

from pathlib import Path

from phoenix.analysis.roofline import gemm_roofline_point, roofline_crossover_ai
from phoenix.discovery.hardware import load_device_record

SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

PRECISION_SWEEP = [
    ("FP64", 8.0),
    ("FP64_TENSOR_CORE", 8.0),
    ("FP32", 4.0),
    ("TF32_TENSOR_CORE_DENSE", 4.0),
    ("TF32_TENSOR_CORE_SPARSE", 4.0),
    ("BF16_TENSOR_CORE_DENSE", 2.0),
    ("BF16_TENSOR_CORE_SPARSE", 2.0),
    ("FP16_TENSOR_CORE_DENSE", 2.0),
    ("FP16_TENSOR_CORE_SPARSE", 2.0),
    ("FP8_TENSOR_CORE_DENSE", 1.0),
    ("FP8_TENSOR_CORE_SPARSE", 1.0),
]


def render_report(hardware_path: Path) -> str:
    record = load_device_record(hardware_path)
    lines: list[str] = []
    lines.append(f"# {record.identity.model} — Analytical Roofline")
    lines.append("")
    lines.append(
        "> **PURELY THEORETICAL. This device has never been executed on by PHOENIX.**\n"
        ">\n"
        "> No NVIDIA GPU exists in this project's environment (verified: no "
        "`nvidia-smi`, no `nvcc`, no discrete GPU — see ADR-032 and TDD §37 item "
        "1). Every figure in this report is `VENDOR_REPORTED` (from NVIDIA's "
        "own published specifications, cross-checked across independent "
        "secondary sources — see the hardware record's own header for exactly "
        "which ones and their confidence) or `DERIVED`/`ANALYTICAL` from those "
        "specifications by the roofline model (TDD §33). **Nothing in this "
        "document is a measurement, a benchmark result, or evidence that "
        "PHOENIX runs on this or any GPU.**"
    )
    lines.append("")
    lines.append(
        f"Generated from `{hardware_path}` by `phoenix.analysis.roofline_report` "
        "— every number below is computed here, not hand-typed."
    )
    lines.append("")
    lines.append("## Peak compute roofs (VENDOR_REPORTED)")
    lines.append("")
    lines.append("| Precision | Peak | Conditions | Confidence |")
    lines.append("|---|---:|---|---|")
    for p in record.compute.precisions:
        conf = p.peak.confidence.value if p.peak.confidence else "—"
        lines.append(
            f"| {p.name} | {p.peak.value:g} {p.peak.unit} | {p.peak.conditions} | {conf} |"
        )
    lines.append("")

    mem = next(m for m in record.memory.levels if m.level == "Device memory")
    assert mem.bandwidth is not None and mem.bandwidth.value is not None
    lines.append(
        f"**Device memory bandwidth (theoretical peak, VENDOR_REPORTED):** "
        f"{mem.bandwidth.value:g} {mem.bandwidth.unit} — {mem.bandwidth.conditions}"
    )
    lines.append("")
    lines.append(
        "**Empirical bandwidth ceiling (MEASURED):** `NOT YET MEASURED` — "
        "no hardware, no microbenchmark, and none is claimed."
    )
    lines.append("")

    for precision_name, dtype_bytes in PRECISION_SWEEP:
        precisions_by_name = {p.name: p for p in record.compute.precisions}
        if precision_name not in precisions_by_name:
            continue
        peak_val = precisions_by_name[precision_name].peak.value
        if peak_val is None:
            continue

        crossover = roofline_crossover_ai(float(peak_val) * 1e12, float(mem.bandwidth.value) * 1e12)
        lines.append(f"## {precision_name}")
        lines.append("")
        lines.append(
            f"Roofline crossover: **{crossover:.2f} FLOP/byte** — GEMMs below this "
            "arithmetic intensity are memory-bound under the no-reuse traffic "
            "model; above it, compute-bound."
        )
        lines.append("")
        lines.append("| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |")
        lines.append("|---:|---:|---|---:|")
        for n in SIZES:
            metric = gemm_roofline_point(record, precision_name, n, n, n, dtype_bytes)
            ai = metric.inputs["arithmetic_intensity_flop_per_byte"]
            regime = metric.inputs["regime"]
            gflops = (metric.value or 0.0) / 1e9
            lines.append(f"| {n} | {ai:.2f} | {regime} | {gflops:,.1f} |")
        lines.append("")

    lines.append("## Assumptions (apply to every table above)")
    lines.append("")
    lines.append(
        "- Memory traffic uses the **no-reuse model**: every element of A, B, "
        "and C is charged exactly once against device memory, crediting zero "
        "cache reuse. This is a lower bound on real traffic and therefore an "
        "*upper* bound on true arithmetic intensity — a real kernel's AI, if it "
        "ever runs, will be lower than shown here for small sizes where cache "
        "matters, converging toward these figures only at sizes that exceed "
        "cache regardless."
    )
    lines.append("- A fused multiply-add counts as 2 FLOPs.")
    lines.append(
        "- Peak figures assume boost clock and the sparsity condition named in "
        "each precision's own row above; see the hardware record for full "
        "source citations and confidence levels."
    )
    lines.append(
        "- Ignores kernel launch overhead, host-device transfer, and anything "
        "outside the GEMM itself."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_report(Path("hardware/gpu/nvidia-h100-sxm5-80gb.yaml")))
