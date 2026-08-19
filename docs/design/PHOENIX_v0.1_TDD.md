# PHOENIX v0.1 — Technical Design Document

**Status:** Draft for technical design review
**Scope:** Architecture and research design only. No implementation code is authorised by this document.
**Document version:** 0.1.0
**Last updated:** 2026-08-19

> **Data policy for this document.** No benchmark number, throughput figure, latency figure, or hardware specification appears anywhere below. Where a value would normally be quoted, this document writes `UNKNOWN` or `NOT YET MEASURED`. Every table cell that would require vendor data is marked as requiring verification against a primary source at implementation time. This is deliberate and non-negotiable: the credibility of every later PHOENIX claim rests on never having fabricated an early one.

---

## 1. Executive Summary

PHOENIX v0.1 is **not** an accelerator, a compiler, or a photonic simulator. It is the measurement and provenance infrastructure that every future PHOENIX claim will be built on top of.

The central insight driving this design: the hardest part of comparing a GPU against a photonic accelerator is not simulating the photonics. It is producing an electronic baseline that a skeptical reviewer cannot dismantle. A photonic energy advantage claimed against a badly-measured GPU baseline is worth nothing. So v0.1 builds the baseline, and builds it to a standard where the measurement methodology itself is the first research asset.

**What v0.1 delivers:**

1. A vendor-neutral hardware description database with per-field provenance.
2. An environment discovery system that captures the full software/hardware manifest of any run.
3. A C++ benchmark runtime with a backend abstraction (CPU, CUDA, HIP, TPU-adapter), of which CPU and CUDA are implemented in v0.1.
4. A GEMM implementation ladder from CPU reference through Tensor Core and cuBLASLt, each stage validated against a trusted reference.
5. An immutable measurement store (Parquet + DuckDB) separating raw samples from aggregates from derived metrics.
6. A profiling subsystem wrapping Nsight Systems and Nsight Compute, with profiled timings structurally prevented from contaminating benchmark timings.
7. A reproducibility package emitted per run, sufficient to reconstruct execution conditions.
8. A structured research-literature database that records claims, evidence, and evidence *class* — not URLs.
9. Roofline analysis distinguishing theoretical roofs from empirically-attained ceilings.

**What v0.1 explicitly does not deliver:** photonic simulation, neuromorphic simulation, a compiler, a scheduler, multi-GPU, distributed execution, or any cross-vendor performance verdict.

**The first research question v0.1 is designed to answer:**

> For a given GPU, how does attained GEMM performance relate to vendor-reported theoretical peak as a function of matrix size, precision, and memory access pattern — and what, measured rather than assumed, accounts for the gap?

**Critical environmental finding.** *(Revised 2026-08-19 against the actual development host. The original text described the authoring container, not this project's machine.)*

The verified development host is:

| Property | Verified value | How verified |
|---|---|---|
| Host OS | Microsoft Windows 11 Home 10.0.26200 | `Get-CimInstance Win32_OperatingSystem` |
| Development environment | WSL2, Ubuntu 26.04 LTS, kernel 6.6.87.2-microsoft-standard-WSL2 | `wsl --list --verbose`, `uname -r` |
| CPU | Intel Core i7-1355U — 10 physical / 12 logical, **hybrid P-core + E-core, mobile/laptop part** | `Get-CimInstance Win32_Processor` |
| RAM | 15.7 GiB | `Get-CimInstance Win32_OperatingSystem` |
| GPU | Intel Iris Xe Graphics (integrated). **No NVIDIA GPU. No discrete GPU of any kind.** | `Get-CimInstance Win32_VideoController` |
| CUDA | `nvidia-smi`, `nvcc` both absent | `command -v` |
| ROCm | absent | `command -v` |
| Toolchain present (WSL2) | gcc 15.2.0, g++, make, git, python3 3.14.4 | `--version` |
| Toolchain absent (WSL2) | **cmake, ninja** — must be installed in Week 1 | `command -v` |
| Toolchain (Windows side) | no C++ compiler at all (`cl`, `gcc` both absent); Python 3.13.5; Docker Desktop; uv | `command -v` |

**Three consequences, in order of importance.**

**1. This host is a development host, not a measurement host.** (ADR-032.) Three independent reasons, any one of which would be sufficient:

- The i7-1355U is a **hybrid architecture**: performance and efficiency cores with different microarchitectures, clock ranges, and cache. A thread migrated between core types mid-run produces a bimodal timing distribution that is an artefact of the scheduler, not of the code under test.
- It is a **15 W-class mobile part in a laptop chassis**. Sustained GEMM sweeps will thermally throttle. §32 would correctly mark such runs `THERMAL_THROTTLE` → INVALID.
- **WSL2 is a virtual machine.** Its clock source, scheduling, and memory behaviour interpose a layer between PHOENIX's timers and the hardware. The magnitude of that layer's effect on `host_wall_ns` is `UNKNOWN` and would itself require characterisation (§37 item 11).

Therefore: EXP-001 may be **executed** here to validate the pipeline end-to-end — that is Week 4's actual purpose — but its output is **development-grade data, marked `PROVISIONAL`, and is not publishable**. Any published CPU baseline requires a controlled measurement host. This does not delay the schedule; it changes what Week 4's acceptance criterion means, and §30 has been amended accordingly.

**2. There is no CUDA path on this machine whatsoever.** Weeks 5–9 require a separate physical NVIDIA host that does not currently exist (§37 items 1 and 4). Weeks 1–4 are GPU-free by design and are unblocked today.

**3. The CI architecture (§21) is confirmed, not merely assumed.** The CPU/GPU split, compile-only CUDA CI, and self-hosted GPU nightly tier are the correct response to this environment rather than a hypothetical one.

**Version deviations to reconcile in Week 1:** the toolchain present in WSL2 (GCC 15.2, Python 3.14.4) is newer than §5 assumes (Python 3.11 floor, tested to 3.12). Resolution: `uv` pins the project interpreter to 3.11 or 3.12 independently of the system Python, and GCC 15 is used as-is with its version recorded in every environment manifest. Neither is a blocker; both are recorded rather than glossed.

---

## 2. Project Scope

### 2.1 In scope for v0.1

| Area | Included |
|---|---|
| Workload | GEMM only (`C = alpha·A·B + beta·C`) |
| Backends implemented | CPU reference, CPU-optimised (BLAS), CUDA |
| Backends stubbed with a stable interface | HIP, TPU adapter |
| Precisions | FP64, FP32, TF32, FP16, BF16 (FP8 gated on hardware availability) |
| Devices | Single device, single process |
| Data | Immutable raw samples, derived metrics, provenance |
| Analysis | Descriptive statistics, roofline, scaling curves |
| Outputs | Reproducibility package, plots, experiment report |

### 2.2 Deliberately deferred

| Area | Deferred to | Reason |
|---|---|---|
| Photonic simulation | v0.6 | Needs a defensible electronic baseline first |
| Neuromorphic | v0.8 | Same |
| Compiler / graph IR | v0.5 | Premature before a cost model exists, and a cost model is meaningless before measurement |
| Heterogeneous scheduler | v0.3–v0.5 | Requires per-backend measured cost data |
| CNN / transformer / LLM workloads | v0.2+ | GEMM first; a framework that cannot measure GEMM rigorously cannot measure a transformer rigorously |
| Multi-GPU, NVLink, collectives | v0.2+ | Single-device methodology must be settled first |
| AMD execution (as opposed to AMD *interface*) | v0.2 | No ROCm hardware access confirmed |
| TPU execution | v0.2 | No TPU access confirmed |

---

## 3. Explicit Non-Goals

These are stated so that scope creep can be identified and rejected by reference to this section.

1. **PHOENIX v0.1 does not try to beat cuBLAS.** The PHOENIX kernel ladder exists to *understand* the performance gap, not close it. A kernel that reaches a documented fraction of cuBLAS with a fully explained bottleneck analysis is a success. A faster kernel with no explanation is a failure.
2. **PHOENIX v0.1 makes no cross-vendor performance claim.** Not NVIDIA vs AMD, not GPU vs TPU. The v0.1 backend interface exists to make such comparisons *possible later, under controlled conditions*, not to enable them now.
3. **PHOENIX v0.1 asserts nothing about photonics.** Not "photonics will win", not "photonics will lose".
4. **PHOENIX v0.1 is not a general tensor library.** It is a benchmark harness. No user-facing tensor API, no autograd, no operator coverage goals.
5. **PHOENIX v0.1 does not do performance regression gating in CI on shared runners.** Benchmark numbers from shared/virtualised CI hardware are not admissible as measurements (§32).
6. **PHOENIX v0.1 does not model energy analytically where it can be measured.** Where energy cannot be measured, v0.1 records `NOT YET MEASURED` rather than substituting an estimate.
7. **PHOENIX v0.1 does not publish a leaderboard.**

---

## 4. Architectural Principles

Ordered. When two principles conflict, the higher-numbered one yields.

**P1 — Scientific correctness.** A result that is wrong is worse than no result, because it propagates.

**P2 — Reproducibility.** A measurement that cannot be reconstructed is an anecdote. Every run emits a reproducibility package or the run is invalid.

**P3 — Provenance is a first-class data type, not documentation.** Every numeric field in the system carries a machine-readable class: `MEASURED`, `VENDOR_REPORTED`, `SIMULATED`, `ANALYTICAL`, `HYPOTHETICAL`, `DERIVED`. This is enforced by schema, not by convention. It is impossible to store a number in PHOENIX without declaring where it came from. This single decision is the highest-leverage one in the document — it is what makes a photonic-vs-GPU comparison publishable in five years.

**P4 — Raw data is append-only.** Raw sample files are written once, checksummed, never mutated. Corrections are new records that supersede, referencing what they supersede. History is never rewritten.

**P5 — Derived values are recomputable and never authoritative.** TFLOP/s is not stored as a measurement; it is stored as a derived value with a recorded formula and inputs. If the formula changes, every derived value is recomputable from the retained raw data.

**P6 — Measurement beats assumption.** If a value can be measured, it is not estimated. If it can only be estimated, it is labelled `ANALYTICAL` and never plotted on the same axis as a `MEASURED` value without a visual distinction.

**P7 — Maintainability over cleverness.** The codebase will be read by researchers, not only by its authors.

**P8 — Portability over performance; performance over convenience.** The backend interface will cost some performance at the dispatch boundary. That cost is measured once, documented, and accepted — dispatch overhead is amortised over kernels that run for microseconds to milliseconds.

**P9 — The smallest architecture that is strong enough.** No microservices. No cloud. No message queues. A local-first research tool. Complexity must be justified by an experiment it enables.

**P10 — Every abstraction must survive contact with a genuinely different accelerator.** The backend interface is designed against the TPU case (compile-then-execute, no explicit memory copies, no kernel launches) rather than against the CUDA case, because an abstraction designed around CUDA will break on TPU, whereas the reverse holds.

---

## 5. Concrete Technology Stack

Every choice below is a decision, not a menu. Version pins marked ⚠ must be re-verified against the vendor's current release at implementation time.

### 5.1 Languages

| Layer | Choice | Version |
|---|---|---|
| Compute runtime | C++ | **C++20** |
| GPU kernels (NVIDIA) | CUDA C++ | targets `-std=c++20` where nvcc supports it, else C++17 for device TUs ⚠ |
| GPU kernels (AMD) | HIP C++ | v0.2 |
| Control plane | Python | **3.11** (floor 3.11, tested to 3.12) |
| Build | CMake | **≥ 3.24** (`CMAKE_CUDA_ARCHITECTURES` native support) |

**Decision — C++20, not C++17 or C++23.**
*Why:* `std::span` (non-owning buffer views across the binding boundary), designated initialisers (readable config structs), concepts (backend interface constraints checked at compile time rather than via SFINAE), `<chrono>` improvements, `std::source_location` for error context. All are directly useful.
*Alternatives:* C++17 — maximum toolchain compatibility, but loses `std::span` and concepts, both of which are load-bearing in the backend interface. C++23 — `std::expected` would be genuinely valuable for error propagation, but compiler support across the GCC/Clang/nvcc/hipcc matrix is uneven, and nvcc host-compiler pass-through lags.
*Disadvantage:* nvcc's C++20 support in device code is more limited than the host compiler's. Mitigation: device translation units stay conservative; the C++20 features live in host code.
*Future scalability:* Concepts on the backend interface mean a photonic backend added in v0.6 fails to compile if it does not satisfy the contract, rather than failing at runtime in an experiment.

**Decision — Python 3.11 floor.**
*Why:* `tomllib` in the standard library (config parsing with zero dependencies), exception groups, materially faster interpreter, and `typing` features used by Pydantic v2. 3.11 is present in the verified dev environment.
*Alternatives:* 3.10 — loses `tomllib`. 3.12/3.13 — some scientific-stack wheels lag; adopting as a floor risks blocking on wheel availability for a marginal gain.

### 5.2 Libraries and tools

| Function | Choice | Rationale (short) |
|---|---|---|
| Python bindings | **nanobind** | Smaller binaries, faster compile, lower call overhead than pybind11; modern C++17/20-native. Risk: smaller ecosystem than pybind11 — accepted, the binding surface is deliberately narrow (§7). |
| C++ testing | **GoogleTest** + **GoogleBenchmark** | Standard, mature, CMake-native. GoogleBenchmark is used for *microbenchmarks of PHOENIX's own machinery* (timer overhead, dispatch cost) — **never** for the scientific benchmarks, which use PHOENIX's own measurement path. |
| Python testing | **pytest** + **hypothesis** | Property-based testing for schema round-trips and statistical functions. |
| Config validation | **Pydantic v2** | Runtime validation with generated JSON Schema; the same schema validates C++-side parsing. |
| Config format | **YAML 1.2** (human-authored) → **JSON** (normalised, machine-consumed) | §11, §13. |
| C++ YAML | **yaml-cpp** | Adequate; used only to parse, then normalise immediately to JSON. |
| C++ JSON | **nlohmann/json** | Ubiquitous, header-only, good error messages. |
| Raw sample storage | **Apache Parquet** (via Arrow C++ / pyarrow) | Columnar, compressed, typed, schema-evolvable, checksummed. |
| Analytical store | **DuckDB** | In-process, zero-server, queries Parquet directly, SQL over the full result corpus. |
| Dataframes | **Polars** (primary), pandas (interop only) | Lazy evaluation, strict typing — pandas' silent dtype coercion is a correctness hazard in a provenance-sensitive system. |
| Plotting | **Matplotlib** | Deterministic, scriptable, publication-grade, no browser. Plotly rejected: interactive HTML is not a figure that goes into a paper. |
| Statistics | **NumPy** + **SciPy** | Bootstrap CIs, distribution tests. |
| CLI | **Typer** | Type-hint-driven, minimal boilerplate. |
| Logging (Python) | **structlog** → JSONL | Machine-parseable logs are part of the reproducibility package. |
| Logging (C++) | **spdlog**, JSON sink | Same log schema as Python side. |
| C++ format/lint | **clang-format**, **clang-tidy** | Config committed to repo. |
| Python format/lint | **ruff** (both), **mypy --strict** on `phoenix/` | |
| Docs | **MkDocs Material** + **mkdocstrings** | Markdown-native, matches the repo's existing documentation style. |
| Containers | **Docker**, images for CPU / CUDA / ROCm | §22. |
| Dependency management | **uv** (Python), **CMake FetchContent** pinned by commit SHA (C++) | No floating dependency ever enters a measurement environment. |
| Env manifest hashing | **BLAKE3** | Fast, collision-resistant, used for environment fingerprints. |

### 5.3 CUDA stack decisions

The master prompt correctly demands these be separated. They are separated because they version independently and each can independently invalidate a comparison.

| Component | v0.1 policy |
|---|---|
| **Host NVIDIA driver** | Not controlled by PHOENIX. **Recorded** in every environment manifest. A driver change invalidates cross-run comparability and must be flagged by the analysis layer. |
| **CUDA Toolkit** | Pin one line — **CUDA 12.x, minimum 12.4** ⚠ (verify current stable at implementation). Toolkit version is recorded per run and is part of the environment fingerprint. |
| **CUDA Runtime API** | Runtime API (`cudaMalloc`, `cudaEventRecord`) — not the Driver API. Rationale: the Driver API's extra control is not needed in v0.1, and the Runtime API keeps the CUDA backend readable. Revisit if v0.2 needs explicit context or module management. |
| **nvcc** | Sole device compiler for v0.1. Clang-CUDA rejected for v0.1: divergent codegen would become an uncontrolled variable. |
| **Target architectures** | `CMAKE_CUDA_ARCHITECTURES` set explicitly per build, **recorded in the manifest**. Compiling for `native` is forbidden in released builds — it makes the binary non-reproducible. |
| **cuBLAS** | Vendor baseline for the classic API path. |
| **cuBLASLt** | Vendor baseline for Tensor Core / mixed-precision paths, where layout and epilogue control matter. Treated as a *separate* baseline from cuBLAS — they are not interchangeable. |
| **CUTLASS** | **Not a v0.1 dependency.** Considered as a v0.2 reference point. Reason for exclusion: introducing CUTLASS early tempts the project into copying a fast kernel instead of understanding a slow one. |
| **Nsight Systems / Nsight Compute** | Profiling only, never the timing source (§17, §19). Versions recorded. |
| **PTX / JIT** | JIT compilation on first launch is a confound. Policy: compile SASS for the exact target architecture; record whether the binary contained a matching cubin or fell back to PTX JIT; a PTX-JIT fallback marks the run's first-execution samples as excluded. |

**Cross-CUDA-version comparability rule.** A benchmark run's `environment_fingerprint` includes driver version, toolkit version, target architecture, and library versions. The analysis layer **refuses by default** to aggregate runs across differing fingerprints; doing so requires an explicit `--allow-heterogeneous-environments` flag which stamps the resulting artefact with a comparability warning that propagates into any generated figure.

---

## 6. Architecture Diagram

```text
                       ┌──────────────────────────────────────┐
                       │        Researcher / CLI / CI         │
                       └──────────────────┬───────────────────┘
                                          │  phoenix run EXP-003
                                          v
╔══════════════════════════ PYTHON CONTROL PLANE ══════════════════════════╗
║                                                                          ║
║  ┌───────────────┐   ┌────────────────┐   ┌───────────────────────────┐  ║
║  │ Config loader │──>│ Pydantic       │──>│ Experiment orchestrator   │  ║
║  │ YAML→JSON     │   │ validation     │   │ sweep expansion, ordering │  ║
║  └───────────────┘   └────────────────┘   └─────────────┬─────────────┘  ║
║                                                         │                ║
║  ┌───────────────────────────┐                          │                ║
║  │ Environment discovery     │──────────────────────────┤                ║
║  │ host, GPU, driver, libs   │   environment_manifest    │                ║
║  └───────────────────────────┘                          │                ║
║                                                         │                ║
║                        ┌────────────────────────────────┘                ║
║                        │  RunRequest (JSON, schema-versioned)            ║
╚════════════════════════╪═════════════════════════════════════════════════╝
                         │  nanobind boundary  (§7)
╔════════════════════════╪═════════════ C++ COMPUTE PLANE ═════════════════╗
║                        v                                                 ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │                     BenchmarkExecutor                              │  ║
║  │   warmup → correctness gate → measurement loop → sample capture    │  ║
║  └───────────┬───────────────────────────────┬────────────────────────┘  ║
║              │                               │                           ║
║   ┌──────────v──────────┐        ┌───────────v────────────┐              ║
║   │   Timing subsystem  │        │  Telemetry subsystem   │              ║
║   │ host clock / events │        │  NVML power, clocks,   │              ║
║   │ device events       │        │  temperature, throttle │              ║
║   └─────────────────────┘        └────────────────────────┘              ║
║              │                                                           ║
║  ┌───────────v────────────────────────────────────────────────────────┐  ║
║  │                  IBackend  (C++20 concept + vtable)                │  ║
║  └──┬──────────────┬─────────────────┬─────────────────┬──────────────┘  ║
║     │              │                 │                 │                 ║
║  ┌──v────┐   ┌─────v──────┐   ┌──────v─────┐   ┌───────v────────┐        ║
║  │ Cpu   │   │ Cuda       │   │ Hip        │   │ Tpu adapter    │        ║
║  │Backend│   │ Backend    │   │ Backend    │   │ (out-of-proc)  │        ║
║  │       │   │            │   │  v0.2      │   │      v0.2      │        ║
║  │naive  │   │GEMM-1..5   │   │  stub      │   │      stub      │        ║
║  │blocked│   │cuBLAS      │   │            │   │                │        ║
║  │OpenMP │   │cuBLASLt    │   │            │   │                │        ║
║  │BLAS   │   │            │   │            │   │                │        ║
║  └───────┘   └────────────┘   └────────────┘   └────────────────┘        ║
╚══════════════════════════════════════════════════════════════════════════╝
                         │  RunResult + raw samples
                         v
╔═══════════════════════ DATA & PROVENANCE LAYER ═══════════════════════════╗
║  results/raw/<run_id>/            (immutable, checksummed)                ║
║    ├── config.yaml         ├── environment.json    ├── hardware.json      ║
║    ├── git.json            ├── results.json        ├── samples.parquet    ║
║    ├── telemetry.parquet   ├── logs/               └── profiling/         ║
║                                    │                                      ║
║                                    v                                      ║
║             DuckDB catalogue (views over Parquet, never a copy)           ║
╚═══════════════════════════════════╪═══════════════════════════════════════╝
                                    v
╔═════════════════════════ ANALYSIS & REPORTING ════════════════════════════╗
║  aggregation → derived metrics → roofline → figures → experiment report   ║
║  every figure stamped with: experiment_id, run_ids, provenance classes    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Note on the profiling path.** Profiled executions are launched through a *separate* entry point that writes into `profiling/` and stamps `timing_valid_for_reporting = false` on any timing captured during profiling. This is structural: it is not possible to accidentally aggregate a Nsight-perturbed timing into a performance figure.

---

## 7. Python/C++ Boundary

### 7.1 Where the line sits

| Responsibility | Side | Reason |
|---|---|---|
| Config parsing, validation, sweep expansion | Python | Iteration speed, rich validation, no rebuild to change an experiment |
| Environment/hardware discovery | Python (shells out to `nvidia-smi`, `lscpu`, NVML via `pynvml`) | Not performance-sensitive; easier to extend per platform |
| Experiment ordering, retries, run-level policy | Python | Orchestration logic changes often |
| Timing of the measured region | **C++** | Python's interpreter jitter is on the order of the quantities being measured for small GEMMs. Non-negotiable. |
| Kernel launch, memory management, synchronisation | C++ | Correctness and overhead |
| Correctness comparison against reference | C++ | Must happen on the buffers as they exist, before any copy or conversion |
| Per-sample capture | C++ (in-memory ring, flushed after the loop) | No I/O inside the measured loop |
| Sample persistence, statistics, plots, reports | Python | Right tool |

**Rule:** the Python interpreter is never on the critical path of a measured region. Python hands a fully-specified `RunRequest` to C++ and receives a `RunResult` plus a sample buffer. One crossing per run, not per iteration.

### 7.2 API surface

Deliberately tiny — six entry points:

```
phoenix_core.enumerate_backends()            -> list[BackendInfo]
phoenix_core.probe_device(backend, index)    -> DeviceInfo
phoenix_core.validate_request(request_json)  -> ValidationReport
phoenix_core.execute(request_json)           -> RunResult      # blocking
phoenix_core.abi_version()                   -> str
phoenix_core.build_info()                    -> BuildInfo      # compiler, flags, lib versions
```

`request_json` and the returned result are **JSON strings validated against a versioned schema on both sides**. Structured objects are deliberately not passed across the boundary.

*Why JSON-over-the-boundary rather than rich bound types:* it makes the boundary auditable and serialisable. Every crossing can be logged verbatim into the reproducibility package, and the exact request that produced a result is recoverable byte-for-byte. It also decouples the binding from the C++ type layout, so an ABI change does not silently corrupt a field. The serialisation cost is once per run — irrelevant.
*Disadvantage:* two schema definitions (Pydantic and C++). Mitigation: the C++ side is generated from the Pydantic-emitted JSON Schema by a build step, and a CI test asserts round-trip equivalence in both directions.

### 7.3 Memory ownership

- **Benchmark buffers:** allocated, owned, and freed entirely inside C++. Python never sees a device pointer.
- **Sample arrays:** allocated in C++, exposed to Python as a zero-copy read-only `std::span` view via nanobind's ndarray protocol, with lifetime tied to the `RunResult` object. Python copies into Arrow before the result is released.
- **No Python object is ever referenced from C++ after `execute()` returns.**

### 7.4 Error propagation

Three-tier, no exceptions across the boundary:

| Tier | Meaning | Behaviour |
|---|---|---|
| `ConfigurationError` | Request invalid before execution | Raised in `validate_request`, run never starts |
| `ExecutionError` | Backend/API/allocation failure during run | `RunResult.status = FAILED`, error captured with `std::source_location`, partial samples retained and marked incomplete |
| `ValidityViolation` | Ran, but result is not scientifically usable (correctness failure, throttling, insufficient samples) | `RunResult.status = INVALID` + machine-readable `invalidation_reasons[]` (§32). **Data is still written.** |

The third tier is the important one. An invalid run is data about the measurement environment and is never discarded.

### 7.5 ABI and versioning

- `phoenix_core` exposes `abi_version()` as `MAJOR.MINOR`. Python asserts a compatible major at import; mismatch is a hard failure with a rebuild instruction.
- Result schema version is independent of ABI version and is stamped into every `results.json`.
- **Threading model:** the C++ core is single-threaded at the API level and not re-entrant. `execute()` holds a process-wide lock. Internal parallelism (OpenMP in the CPU backend, CUDA streams) is a backend implementation detail. Rationale: concurrent benchmark execution destroys measurement validity; the constraint is enforced rather than documented.
- The GIL is released for the duration of `execute()`.

---

## 8. Repository Structure

```text
PHOENIX/
├── README.md
├── LICENSE                              # Apache-2.0 (already present)
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── CMakeLists.txt
├── pyproject.toml
├── Makefile                             # thin convenience wrapper over cmake/uv
├── .clang-format
├── .clang-tidy
├── .gitattributes                       # Parquet/PNG as binary; LF normalisation
│
├── cmake/                               # toolchain files, Find*.cmake, arch detection
│
├── cpp/                                 # ── C++ COMPUTE PLANE ──
│   ├── include/phoenix/
│   │   ├── core/                        # Result, Status, Provenance, Error types
│   │   ├── backend/                     # IBackend concept, BackendRegistry, DeviceInfo
│   │   ├── bench/                       # BenchmarkExecutor, WarmupPolicy, SampleBuffer
│   │   ├── timing/                      # HostTimer, DeviceTimer, TimerCalibration
│   │   ├── telemetry/                   # PowerSampler, ClockSampler, ThrottleDetector
│   │   ├── correctness/                 # Comparator, tolerance policies
│   │   └── schema/                      # generated request/result structs
│   ├── src/
│   │   ├── core/  bench/  timing/  telemetry/  correctness/  schema/
│   ├── backends/
│   │   ├── cpu/                         # naive, blocked, OpenMP, BLAS-backed
│   │   ├── cuda/
│   │   │   ├── kernels/                 # gemm_naive.cu … gemm_tensorcore.cu
│   │   │   ├── vendor/                  # cuBLAS, cuBLASLt wrappers
│   │   │   └── runtime/                 # device mgmt, streams, events, NVML
│   │   ├── hip/                         # v0.2 — interface-complete stub in v0.1
│   │   └── tpu/                         # v0.2 — out-of-process adapter stub
│   └── tests/                           # GoogleTest; unit/ and correctness/
│
├── bindings/                            # nanobind module `phoenix_core`
│
├── python/phoenix/                      # ── PYTHON CONTROL PLANE ──
│   ├── cli/                             # Typer entry points
│   ├── config/                          # Pydantic models, loader, sweep expansion
│   ├── discovery/                       # host/GPU/software manifest collection
│   ├── runner/                          # orchestration, retry, run lifecycle
│   ├── store/                           # Parquet writer, DuckDB catalogue, integrity
│   ├── analysis/                        # statistics, derived metrics, roofline
│   ├── profiling/                       # Nsight wrappers, artifact indexing
│   ├── viz/                             # figure builders (each stamps provenance)
│   ├── report/                          # experiment report generation
│   └── literature/                      # research-paper database access layer
│
├── schemas/                             # ── CONTRACTS (versioned, the real API) ──
│   ├── hardware/hardware-v1.schema.json
│   ├── experiment/experiment-v1.schema.json
│   ├── result/result-v1.schema.json
│   ├── environment/environment-v1.schema.json
│   ├── literature/paper-v1.schema.json
│   └── provenance/provenance-v1.schema.json
│
├── hardware/                            # ── HARDWARE DATABASE (YAML, per-field provenance) ──
│   ├── nvidia/  amd/  google/ intel/  cpu/
│   └── _templates/
│
├── experiments/                         # ── EXPERIMENT DEFINITIONS (version-controlled) ──
│   ├── EXP-001_cpu_reference_gemm/      # config.yaml + hypothesis.md + README.md
│   ├── EXP-002_naive_cuda_gemm/
│   └── … EXP-010
│
├── results/                             # ── DATA (git-ignored except manifests) ──
│   ├── raw/<run_id>/                    # immutable reproducibility packages
│   ├── catalogue/                       # DuckDB file + Parquet views
│   ├── processed/                       # aggregates, derived metrics
│   └── figures/                         # generated only; never hand-edited
│
├── research/                            # ── LITERATURE ──
│   ├── papers/<paper_id>.yaml
│   ├── claims/                          # extracted claims w/ evidence class
│   ├── notes/                           # hypothesis records (§31 format)
│   └── papers.csv                       # generated export, not source of truth
│
├── containers/
│   ├── cpu.Dockerfile  cuda.Dockerfile  rocm.Dockerfile
│
├── docs/
│   ├── design/                          # this document
│   ├── architecture/  methodology/  tutorials/  api/  adr/
│   └── mkdocs.yml
│
├── scripts/                             # operational scripts, not library code
└── .github/workflows/                   # CI (§21)
```

### 8.1 Why each top-level directory exists

- **`cpp/` and `python/` are separate trees, not interleaved.** They have different build systems, test runners, review standards, and change frequencies. Interleaving them is the single most common structural mistake in research codebases and it makes the compute plane impossible to reason about in isolation.
- **`backends/` sits under `cpp/`, not at top level.** Backends are implementations of a C++ interface. A top-level `cuda/` directory (as in the original README sketch) implies CUDA is a peer of the whole system rather than one backend among several — which is exactly the assumption that breaks when photonics arrives.
- **`schemas/` is top-level and language-neutral.** These are the project's real API. Both language planes and all external tooling consume them. Burying them inside `python/` would make the C++ side a second-class consumer.
- **`hardware/` is separate from `results/`.** Hardware descriptions are curated, reviewed, human-authored knowledge with citations. Results are machine-generated. Mixing them lets a `VENDOR_REPORTED` figure leak into a `MEASURED` context — the precise failure mode §16 of the research context warns against.
- **`experiments/` holds definitions; `results/` holds outcomes.** Definitions are version-controlled and reviewed; outcomes are immutable data. A run links to an experiment definition by ID *and commit SHA*.
- **`research/` uses one YAML file per paper**, not a single CSV. Per-file structure supports rich nested claim records, gives clean diffs, and avoids merge conflicts. `papers.csv` is a generated export for humans who want a spreadsheet.
- **`results/` is git-ignored** except for run manifests. Benchmark data does not belong in git history — it belongs in an archival store with checksums.

---

## 9. Hardware Schema

`schemas/hardware/hardware-v1.schema.json`. Human-authored YAML in `hardware/`, normalised to JSON on load.

### 9.1 The provenance-wrapped value

This is the type that makes the whole system work. **Every** numeric field in the hardware database uses it:

```yaml
# The universal value wrapper
<field_name>:
  value: <number|string|null>       # null == UNKNOWN, and null is legal
  unit: "TFLOP/s"                   # required whenever value is numeric
  provenance: VENDOR_REPORTED       # enum, required
  source:
    citation: "NVIDIA H100 Tensor Core GPU Architecture Whitepaper"
    url: "…"
    accessed: "2026-08-19"
    page: "…"                       # optional
  conditions: "with sparsity; boost clock"   # free text, REQUIRED for peak-compute fields
  confidence: HIGH                  # HIGH | MEDIUM | LOW
  notes: "…"
```

**`conditions` is mandatory on all peak-compute fields.** A peak FLOP/s figure without stated conditions (sparsity on/off, boost vs base clock, which precision mode) is the single most common source of dishonest accelerator comparisons. The schema rejects a peak-compute entry lacking `conditions`.

### 9.2 Device record structure

```yaml
schema_version: "1.0.0"
id: "nvidia-h100-sxm5-80gb"           # stable slug, immutable once published
device_class: GPU                      # CPU|GPU|TPU|NPU|PHOTONIC|NEUROMORPHIC|OTHER_ACCELERATOR

identity:
  vendor: NVIDIA
  model: "…"
  family: "…"
  architecture: "…"
  microarchitecture: "…"
  product_codes: ["…"]                 # SKU / PCI device IDs where known
  form_factor: "…"                     # SXM | PCIe | OAM | socket | …
  release_year: {value: null, provenance: VENDOR_REPORTED, source: {...}}

compute:
  units:                               # deliberately generic — SMs, CUs, MXUs, cores
    - kind: "SM"
      count: {value: null, provenance: VENDOR_REPORTED, source: {...}}
    - kind: "Tensor Core"
      count: {value: null, provenance: VENDOR_REPORTED, source: {...}}
      generation: "…"
  clocks:
    base_mhz:  {value: null, ...}
    boost_mhz: {value: null, ...}
  precisions:                          # list, not a fixed set of keys
    - name: FP64
      supported: true
      peak: {value: null, unit: "TFLOP/s", provenance: VENDOR_REPORTED,
             conditions: "REQUIRED", source: {...}}
      native_or_emulated: NATIVE
    - name: TF32
      ...
    - name: FP8_E4M3
      ...

memory:
  levels:                              # ordered, nearest-to-compute first
    - level: "Register file"
      capacity: {value: null, unit: "KiB/SM", ...}
    - level: "L1/Shared"
      capacity: {value: null, unit: "KiB/SM", ...}
      configurable: true
    - level: "L2"
      capacity: {value: null, unit: "MiB", ...}
    - level: "Device memory"
      technology: "HBM3"
      capacity:  {value: null, unit: "GiB", ...}
      bandwidth: {value: null, unit: "TB/s", provenance: VENDOR_REPORTED,
                  conditions: "theoretical peak", ...}
      bandwidth_attained: {value: null, unit: "TB/s", provenance: MEASURED,
                  source: {run_id: "…", experiment_id: "…"}}   # ← links to a PHOENIX run

interconnect:
  host: {kind: "PCIe", generation: null, lanes: null, bandwidth: {...}}
  device_to_device:
    - kind: "NVLink"
      generation: null
      bandwidth: {...}
      topology_notes: "…"

power:
  tdp:               {value: null, unit: "W", provenance: VENDOR_REPORTED, ...}
  idle_typical:      {value: null, unit: "W", provenance: MEASURED, ...}
  telemetry_available: true
  telemetry_mechanism: "NVML"
  telemetry_semantics: "board power, sampling interval UNKNOWN — must be characterised"

software:
  runtime: "CUDA"
  isa: "…"
  compute_capability: "…"
  compilers: ["nvcc"]
  vendor_math_libraries: ["cuBLAS", "cuBLASLt"]

provenance_summary:
  authored_by: "…"
  reviewed_by: "…"
  last_verified: "2026-08-19"
  overall_confidence: MEDIUM
```

### 9.3 Design decisions

**Decision — `compute.units` and `memory.levels` are lists of typed records, not fixed keys.**
*Why:* an SM, a CU, an MXU, an MZI mesh, and a neuromorphic core have nothing structurally in common. A schema with a `sm_count` field is a schema that cannot describe a photonic device. The list-of-records form already accommodates every v0.1 device and extends to v0.6 photonics without a schema break.
*Disadvantage:* queries are more awkward than fixed keys. Mitigated by DuckDB views that project the common cases.

**Decision — `null` is a legal, first-class value.**
*Why:* Rule 1. `UNKNOWN` must be representable, and the schema must never pressure an author into inventing a number to satisfy a required field. `value: null` with a `provenance` and a note is a complete, valid, honest record.

**Decision — measured fields link back to run IDs.**
`bandwidth_attained` carries `source: {run_id, experiment_id}`, closing the loop between the hardware database and the measurement store. Vendor-reported and PHOENIX-measured values for the same physical quantity coexist as separate fields and are never merged.

**Decision — hardware records are immutable once published.** Corrections create a new version (`id` stays, `schema_version`/record revision increments) with a supersession pointer. Old results keep referencing the record revision they were measured against.

---

## 10. Benchmark Data Schema

Four separated tiers. Conflating them is the failure mode this design exists to prevent.

```text
Tier 1  RAW SAMPLES        per-iteration, immutable, Parquet
Tier 2  AGGREGATES         statistics over Tier 1, recomputable
Tier 3  DERIVED METRICS    formula + inputs recorded, recomputable
Tier 4  REPORTED VALUES    what a figure/paper shows, with explicit selection rationale
```

Tiers 2–4 are **always recomputable from Tier 1**. If they are ever lost, nothing is lost. If Tier 1 is lost, the run is gone.

### 10.1 Tier 1 — raw samples (`samples.parquet`)

One row per measured iteration:

| Column | Type | Notes |
|---|---|---|
| `run_id` | string (UUIDv7) | time-ordered |
| `sample_index` | int32 | 0-based, monotonic |
| `repetition_index` | int32 | which repetition block (§16) |
| `phase` | enum | `WARMUP` \| `MEASURE` — warmup samples are **kept**, flagged, and excluded from aggregation |
| `host_wall_ns` | int64 | host-side steady_clock delta |
| `device_time_ns` | int64 \| null | CUDA event delta; null for CPU backend |
| `api_overhead_ns` | int64 \| null | host wall minus device time, where both exist |
| `correctness_checked` | bool | |
| `max_abs_err`, `max_rel_err` | float64 \| null | populated when checked |
| `power_w` | float64 \| null | telemetry-sampled, time-aligned; null if unavailable |
| `sm_clock_mhz`, `mem_clock_mhz` | int32 \| null | |
| `temperature_c` | int32 \| null | |
| `throttle_flags` | int64 \| null | vendor bitmask, recorded raw |
| `timestamp_ns` | int64 | monotonic since run start |
| `valid` | bool | per-sample validity |
| `invalidation_reasons` | list<string> | empty when valid |

**Nothing here is a derived quantity.** No FLOP/s column. That belongs in Tier 3.

### 10.2 Tier 2 — aggregates

Per `(run_id, phase, metric)`: `n`, `min`, `max`, `mean`, `median`, `stddev`, `p50`, `p90`, `p95`, `p99`, `iqr`, `mad`, `cv`, `ci_low`, `ci_high`, `ci_method`, `ci_level`, `n_excluded`, `exclusion_rule`.

Every aggregate records **which raw samples it consumed** (index ranges) so it can be audited.

### 10.3 Tier 3 — derived metrics

```json
{
  "metric": "achieved_flops",
  "value": null,
  "unit": "FLOP/s",
  "provenance": "DERIVED",
  "formula": "(2*M*N*K) / t_seconds",
  "formula_version": "1.0.0",
  "inputs": {
    "M": 4096, "N": 4096, "K": 4096,
    "t_seconds": {"source": "aggregate", "run_id": "…", "statistic": "median",
                  "metric": "device_time_ns"}
  },
  "assumptions": ["FLOP count excludes alpha/beta epilogue",
                  "counts a fused multiply-add as 2 FLOPs"],
  "valid": true
}
```

`assumptions` is mandatory and non-empty. A derived metric with no recorded assumptions is rejected — because there is no such thing as a FLOP/s figure without a FLOP-counting convention, and unstated conventions are how incomparable numbers get compared.

### 10.4 Tier 4 — reported values

What goes in a figure. Records: which aggregate/derived value was selected, **why that statistic** (e.g. "median, because the distribution is right-skewed by OS scheduling — see distribution plot"), which runs were included/excluded and on what rule, and the comparability status of the run set.

### 10.5 Run record (`results.json`)

```json
{
  "schema_version": "1.0.0",
  "run_id": "…", "experiment_id": "EXP-003",
  "experiment_definition_commit": "…",
  "status": "COMPLETED | FAILED | INVALID",
  "invalidation_reasons": [],
  "started_at": "…", "finished_at": "…",
  "hardware_ref": {"device_record_id": "…", "record_revision": 3,
                   "physical_device_fingerprint": "…"},
  "environment_fingerprint": "blake3:…",
  "software": {"os":"…","kernel":"…","driver":"…","cuda_toolkit":"…",
               "cublas":"…","compiler":"…","compiler_flags":"…",
               "cuda_architectures":"…","python":"…","phoenix_version":"…"},
  "git": {"commit":"…","dirty": false, "branch":"…"},
  "container": {"image":"…","digest":"…"},
  "workload": {"kind":"GEMM","M":…,"N":…,"K":…,"dtype_a":"…","dtype_b":"…",
               "dtype_c":"…","dtype_compute":"…","layout_a":"…","layout_b":"…",
               "alpha":…,"beta":…},
  "execution": {"backend":"cuda","implementation":"gemm_shared_tiled_v1",
                "warmup_iters":…,"measure_iters":…,"repetitions":…,
                "timing_method":"cuda_events","sync_policy":"…",
                "stream":"default"},
  "correctness": {"policy":"…","reference":"cublas","passed": true,
                  "tolerance_rule":"…","max_abs_err":null,"max_rel_err":null},
  "artifacts": {"samples":"samples.parquet","samples_blake3":"…",
                "telemetry":"telemetry.parquet","logs":"logs/",
                "profiling": null},
  "timing_valid_for_reporting": true
}
```

Note `dtype_a/b/c/compute` are four separate fields. Mixed-precision GEMM has an input type, an accumulate type, and an output type, and collapsing them into one "precision" field makes FP16-in/FP32-accumulate indistinguishable from FP16-throughout. That distinction is the entire subject of EXP-006.

### 10.6 Immutability enforcement

On write: files closed, BLAKE3 checksummed, directory made read-only, `MANIFEST.json` written containing every file's checksum. On ingest: checksums verified; a mismatch marks the run `TAMPERED` and excludes it from all analysis. Corrections are new runs with `supersedes: <run_id>`.

---

## 11. Experiment Configuration Schema

**Decision — YAML 1.2 for human-authored configs.**
*Why:* comments (essential for recording *why* a parameter was chosen), readable nesting, multi-line strings for hypotheses. TOML rejected: poor at deep nesting, which sweep specifications need. JSON rejected as an authoring format: no comments. Both are used downstream — configs normalise to JSON immediately after validation, and the normalised JSON is what is hashed and stored.

```yaml
schema_version: "1.0.0"
experiment_id: "EXP-003"
experiment_version: "1.2.0"
title: "Shared-memory tiled CUDA GEMM"

hypothesis: |
  Shared-memory tiling will increase attained FP32 GEMM throughput relative to
  EXP-002 (naive CUDA) at M=N=K >= 1024, because tiling raises data reuse and
  reduces global-memory traffic per FLOP.
  This is a HYPOTHESIS. It is not a result. See research/notes/HYP-004.md.

workload:
  kind: GEMM
  sweep:
    dimensions:
      - {M: 128,  N: 128,  K: 128}
      - {M: 1024, N: 1024, K: 1024}
      - {M: 4096, N: 4096, K: 4096}
      - {M: 1024, N: 4096, K: 256}      # rectangular, deliberately
      - {M: 1023, N: 1025, K: 767}      # non-power-of-2, deliberately
    dtype:
      - {a: fp32, b: fp32, c: fp32, compute: fp32}
  layout: {a: row_major, b: row_major, c: row_major}
  alpha: 1.0
  beta: 0.0

execution:
  backends: ["cuda"]
  implementations: ["gemm_shared_tiled_v1"]
  device_selection: {index: 0, require_exclusive: true}

measurement:
  warmup_iterations: 20
  measure_iterations: 100
  repetitions: 5                   # separate process invocations
  timing_method: cuda_events       # cuda_events | host_wall | both
  synchronization: per_iteration   # per_iteration | per_batch
  inter_repetition_cooldown_s: 30
  outlier_policy:
    rule: mad                      # mad | none | tukey
    threshold: 3.0
    action: flag                   # flag | exclude ; NEVER delete

power:
  method: nvml_sampling            # nvml_sampling | none
  sample_interval_ms: 10
  required: false                  # if true, absence of power data invalidates run

profiling:
  mode: none                       # none | nsys | ncu ; profiled runs are separate runs

correctness:
  policy: every_repetition_first_iteration
  reference: cublas                # cpu_reference | cublas | both
  tolerance: dtype_default         # resolved per §14

validity:
  min_valid_samples: 80
  max_cv: 0.05                     # coefficient of variation gate
  throttle_policy: invalidate      # invalidate | flag
  require_clean_environment: true

output:
  raw_dir: "results/raw"
  retain_warmup_samples: true
  emit_reproducibility_package: true

provenance:
  author: "…"
  created: "2026-08-19"
  supersedes: null
```

**Reproducibility contract:** `config.yaml` (content-hashed) + `environment.json` + source commit SHA + container digest ⇒ the run is reconstructible. All four are stored in every reproducibility package. Any one missing marks the package `INCOMPLETE`.

---

## 12. CUDA Architecture

### 12.1 Backend interface (the contract every backend satisfies)

```cpp
namespace phoenix {

template <typename T>
concept Backend = requires(T b, const RunRequest& r) {
    { T::identity() }        -> std::same_as<BackendIdentity>;
    { b.probe() }            -> std::same_as<std::vector<DeviceInfo>>;
    { b.prepare(r) }         -> std::same_as<PreparedWorkload>;  // alloc + compile
    { b.warmup() }           -> std::same_as<void>;
    { b.execute_once() }     -> std::same_as<ExecutionRecord>;   // ONE measured iteration
    { b.synchronize() }      -> std::same_as<void>;
    { b.read_output() }      -> std::same_as<HostBufferView>;
    { b.teardown() }         -> std::same_as<void>;
    { b.capabilities() }     -> std::same_as<BackendCapabilities>;
};

}
```

**Design note (P10):** `prepare()` is separated from `execute_once()` specifically for the TPU case, where compilation is a large, one-time, separately-reportable cost. On CUDA, `prepare()` does allocation and H2D transfer. On TPU it does XLA compilation. On a future photonic backend it would do phase-matrix configuration. Had this interface been designed around CUDA alone, compile latency would have been invisible — and TPU compile latency is one of the most important things to report about a TPU.

`BackendCapabilities` declares supported dtypes, whether device-side timing exists, whether power telemetry exists, and alignment requirements. The executor queries capabilities and **refuses** a request the backend cannot honour, rather than silently substituting behaviour (e.g. silently promoting FP16 to FP32).

### 12.2 CUDA-specific structure

- **Device management:** one device per run, explicitly selected, no implicit device 0 fallback. Exclusive-process compute mode requested where the config asks for it; if unavailable, recorded, and the run flagged as sharing-possible.
- **Memory:** buffers allocated once in `prepare()`, reused across all iterations. No allocation inside the measured loop. Deliberate cache-state policy: buffers are large enough that L2 does not hold the whole working set for large sizes, and the small-size cases are explicitly labelled as cache-resident in the analysis rather than pretended otherwise.
- **Streams:** default stream in v0.1. Multi-stream concurrency is a v0.2 experiment, not a v0.1 confound.
- **Timing:** CUDA events bracketing the kernel, plus host wall clock around the whole iteration including sync. Both recorded per sample. The difference is `api_overhead_ns` and is itself a research quantity.
- **Synchronisation:** `cudaEventSynchronize` on the stop event. `cudaDeviceSynchronize` is used only between repetitions.
- **Error checking:** every CUDA call goes through a checked wrapper capturing `std::source_location`. No unchecked calls, including in the measured loop — the check is a branch on a returned enum and its cost is measured once during timer calibration and documented.
- **Telemetry:** NVML sampled on a **separate host thread**, timestamped on the same monotonic clock as the samples, aligned in post-processing. Sampling never happens inline. NVML's actual sampling interval and averaging window are themselves `UNKNOWN` and must be characterised (§45) before any energy figure is published.

### 12.3 Timer calibration

Before the first measurement of any session, PHOENIX measures its own overhead: empty-kernel launch latency, CUDA event record/query overhead, host clock resolution, and the checked-wrapper cost. These are stored as a `TimerCalibration` record in the run's environment manifest. Any measured duration within 10× of the calibrated overhead is flagged `MEASUREMENT_FLOOR_RISK`. This is what prevents the classic error of reporting a 128×128 GEMM "time" that is mostly launch overhead.

---

## 13. CUDA GEMM Development Roadmap

Each stage is a separate, permanently-retained implementation. **Earlier implementations are never deleted** — they are the baselines that make later claims meaningful (Rule 5). Every stage below states a *hypothesis*, never a result.

| ID | Implementation | Purpose | Primary hypothesised bottleneck | Success criterion |
|---|---|---|---|---|
| GEMM-0 | CPU reference, triple loop | Ground truth for correctness | Serial compute | Bit-reproducible run-to-run; validated against an independent BLAS |
| GEMM-1 | Naive CUDA, one thread per output element | Establish the floor; make global-memory behaviour visible | Global memory bandwidth; no reuse | Correct vs GEMM-0; profiler confirms memory-bound |
| GEMM-2 | Shared-memory tiled | Introduce reuse | Shared-memory bandwidth / bank conflicts / occupancy | Correct; measured change vs GEMM-1 explained by profiler counters, not asserted |
| GEMM-3 | Register tiling (thread-level 2D micro-tile) | Raise arithmetic intensity per thread | Register pressure vs occupancy trade-off | Correct; occupancy and register count recorded; the trade-off documented with data |
| GEMM-4 | Vectorised access (`float4`), padded shared memory | Improve transaction efficiency, remove bank conflicts | Alignment, address divergence | Correct incl. non-multiple-of-4 dimensions; transaction efficiency counters recorded |
| GEMM-5 | Tensor Core path (WMMA/MMA, FP16/BF16 in, FP32 accumulate) | Access the matrix engines | Fragment layout, data movement, accumulate precision | Correct **under FP16-appropriate tolerance**; Tensor Core utilisation counter non-zero |
| GEMM-6 | cuBLAS + cuBLASLt baselines | Vendor reference | n/a — this *is* the reference | Recorded as two distinct baselines with their own heuristic/algo selection captured |
| GEMM-7 | PHOENIX optimised (double-buffered, async copy where supported) | Synthesis of what stages 1–6 taught | To be determined by profiling, not guessed | The bottleneck analysis is the deliverable; the speed is secondary |

**For every stage the following are mandatory before the stage is considered done:**

1. Correctness validation against GEMM-0 and against cuBLAS, at every tested shape and dtype.
2. An Nsight Compute characterisation with the roofline position recorded.
3. A written bottleneck analysis stating: what was hypothesised, what was measured, what explains the difference, and what was *ruled out*.
4. Retention of the previous stage as a comparison baseline.

**Explicit rule:** no stage may be optimised before the previous stage has been profiled. The sequence is `measure → profile → hypothesise → modify → measure → validate → document`. A commit that changes a kernel without a linked profiling artefact is rejected in review.

**No performance numbers appear in this roadmap and none will be added to it.** Results live in `results/`, linked by experiment ID.

---

## 14. Correctness Strategy

### 14.1 Reference hierarchy

```text
GEMM-0 CPU reference (FP64 accumulate)   ← ground truth for numerics
        ↓ validates
cuBLAS / cuBLASLt                        ← trusted vendor implementation
        ↓ cross-checks
PHOENIX kernels GEMM-1..7
```

The CPU reference computes in **FP64 regardless of the tested dtype**, then compares. This gives a dtype-independent notion of "the right answer" and separates *implementation error* from *precision-induced error* — which are constantly conflated in accelerator work.

### 14.2 Tolerance policy

Exact floating-point equality is used **only** for integer dtypes. Otherwise, tolerance is derived from the dtype's machine epsilon and the reduction length K, not hand-picked:

```text
tolerance_rel(dtype, K) = C · eps(dtype) · sqrt(K)
```

with `C` a small, per-dtype, **committed and reviewed** constant, and the `sqrt(K)` term reflecting random-walk error accumulation over the reduction. Both absolute and relative error are checked; the absolute floor prevents relative-error blowup near zero.

**AMENDED 2026-08-19 (ADR-034), after the first execution of EXP-001.** The
tolerance above did not state what the relative error is measured *relative to*.
The correct denominator is the sum of absolute products `(|A|·|B|)_ij`, not
`|C_ij|`:

```text
|Ĉ - C|_ij  ≤  C · eps(dtype) · sqrt(K) · (|A|·|B|)_ij
```

Dividing by `|C_ij|` measures catastrophic cancellation, which is a property of the
input data rather than of the implementation, and it caused all 36 fp32 runs of
EXP-001 to fail at 54×–2176× tolerance while every fp64 run passed bit-exactly.
`C` was **not** raised. Cancellation is still recorded, as `max_cancellation`, but
it is not the pass/fail criterion. See `docs/adr/ADR-034-gemm-error-scale.md` and
`research/notes/FIND-001-fp32-cancellation.md`.

*Why `sqrt(K)` rather than `K`:* worst-case linear accumulation is achievable but pathological for the random inputs used here; `sqrt(K)` matches expected behaviour for randomly-signed rounding errors. **This choice is an assumption and is recorded as one.** If a kernel fails only via this term, the failure is investigated, not accommodated by loosening `C`.

**Hard rule:** a tolerance constant is never raised to make a failing kernel pass. Raising `C` requires a written numerical justification in review. This rule exists because loosening a tolerance is the easiest way to turn a broken kernel into a fast one.

### 14.3 Test matrix

| Axis | Cases |
|---|---|
| Shape | square; tall-skinny; short-wide; K≫M,N; M=N=K=1 |
| Dimension parity | powers of two; odd; prime; one-below and one-above a tile boundary; not-a-multiple-of-vector-width |
| Size | tiny (fits in cache); medium; large (exceeds L2); largest that fits device memory |
| dtype | FP64, FP32, TF32, FP16(+FP32 acc), BF16(+FP32 acc) |
| Layout | row/col major combinations |
| Values | uniform random; normal; mixed magnitude (tests catastrophic cancellation); values containing exact zeros; near-denormal (dtype permitting) |
| Adversarial | inputs constructed so a boundary-condition bug produces a *wrong but plausible* result — a kernel that only reads within tile bounds must fail these |

The last row matters most: random-uniform inputs hide boundary bugs, because an off-by-one that reads a neighbouring element of a uniform random matrix produces a plausible number. Structured adversarial inputs (e.g. matrices where element value encodes its index) make index errors immediately visible.

### 14.4 Determinism

Bitwise reproducibility is **not** required of GPU kernels (parallel reduction order varies legitimately). It **is** required of the CPU reference. Run-to-run numerical variation on GPU is measured and recorded as a property of the implementation, not treated as a failure.

---

## 15. Measurement Methodology

### 15.1 The timing taxonomy

Every one of these is a distinct quantity. PHOENIX names them separately and never silently substitutes one for another:

| Quantity | Definition | Captured by |
|---|---|---|
| `device_kernel_time` | Kernel start→end on device | CUDA events around the kernel |
| `api_call_time` | Host time inside the launch call | Host clock around launch |
| `host_iteration_time` | Full host-side iteration incl. sync | Host clock around the iteration |
| `h2d_transfer_time` / `d2h_transfer_time` | Explicit copies | Separate events; **excluded** from GEMM compute time and reported separately |
| `prepare_time` | Allocation + compilation | Around `prepare()` |
| `first_execution_time` | Iteration 0 | Always recorded, always excluded from steady-state statistics |
| `steady_state_time` | Post-warmup measurement phase | The measurement loop |
| `end_to_end_time` | Process start → result written | Wall clock |

**The default reported GEMM latency is `device_kernel_time`, steady-state.** Every figure states this. A figure comparing a `device_kernel_time` against another system's `end_to_end_time` is a category error, and the schema makes such a comparison detectable because both quantities are named in the data.

### 15.2 Protocol

```text
per repetition (fresh process):
  1. record environment + GPU state (clocks, temp, power cap, throttle status)
  2. calibrate timers
  3. prepare()             → prepare_time
  4. warmup iterations     → recorded, phase=WARMUP, excluded from aggregation
  5. correctness check     → outside the timed loop
  6. steady-state check    → require clock/temperature stability before measuring
  7. measurement loop      → phase=MEASURE, one sample per iteration
  8. re-record GPU state   → compare against step 1; drift beyond threshold flags the run
  9. flush samples, write reproducibility package
 10. cooldown before next repetition
```

**Repetitions are separate process invocations**, not inner loops. This captures process-level variance (allocator state, page placement, JIT/compilation, driver state) which an inner loop cannot see and which is a real source of irreproducibility.

### 15.3 Warmup policy

Warmup counts are per-backend defaults, overridable per experiment, and **always recorded**. Warmup samples are retained (not discarded) so that the convergence to steady state is itself analysable — the shape of the warmup curve tells you whether you warmed up enough, and discarding it destroys the evidence for that judgement.

**Adequacy check:** the last 25% of warmup samples must have a coefficient of variation within the configured gate, and their median must be within a threshold of the first 25% of measurement samples. Failure flags `INSUFFICIENT_WARMUP`.

### 15.4 Outliers

Outliers are **flagged, never deleted** (P4). Default rule: median absolute deviation, threshold 3.0, action `flag`. Aggregates are computed both with and without flagged samples and both are stored. If those two aggregates differ materially, the analysis layer raises a warning rather than silently choosing the more favourable one.

### 15.5 Sample-count sufficiency

`measure_iterations` is not a magic constant. The runner performs a sequential adequacy check: measure a block, compute the CI half-width of the median by bootstrap, and continue until the half-width falls below a configured fraction of the median or a hard iteration cap is hit. If the cap is hit first, the run is flagged `PRECISION_TARGET_NOT_MET` — recorded, not hidden.

---

## 16. Statistical Methodology

**Raw samples are always retained.** Everything else is recomputable.

| Statistic | Purpose |
|---|---|
| `min` | Best-case; the closest observable to "the hardware unobstructed" — reported, but never as *the* result |
| `median` (p50) | **Primary reported statistic** — robust to the right-skew that OS scheduling, interrupts, and DVFS produce |
| `mean` + `stddev` | Reported for distribution shape; not the headline |
| `p90`, `p95`, `p99` | Tail behaviour — the relevant statistic for latency-constrained applications |
| `max` | Worst case; often diagnostic of an environmental problem |
| `IQR`, `MAD`, `CV` | Dispersion; CV gates run validity |

**Decision — median as the primary statistic, with the distribution always published alongside.**
*Why:* benchmark timing distributions are bounded below (there is a physical minimum) and unbounded above (interference has no ceiling). The mean is dragged by the tail; the minimum ignores the reality that real workloads experience interference. The median is the honest central estimate.
*What is explicitly forbidden:* reporting a median without its dispersion. Every figure showing a central value shows uncertainty (error bars, or the distribution itself). A bar chart of bare medians is not an acceptable PHOENIX figure.

**Uncertainty:** bootstrap confidence intervals (BCa, 10,000 resamples) on the median. Bootstrap rather than a parametric CI because the distributions are non-normal and skewed, and assuming normality here would produce intervals that are wrong in a direction that flatters the result.

**Comparisons:** a claim that implementation A is faster than B requires (a) same environment fingerprint, (b) non-overlapping bootstrap CIs *or* a stated effect size with its own CI, (c) an effect that exceeds the measured run-to-run variation of a single implementation against itself. That last control — measuring A against A — is mandatory before any A-vs-B claim, and it is the control most often skipped in accelerator papers.

**Forbidden practices, enforced in review:** reporting more significant figures than the CI supports; comparing across environment fingerprints without the comparability stamp; using `min` for one implementation and `median` for another; aggregating across repetitions without reporting between-repetition variance separately from within-repetition variance.

---

## 17. NVIDIA Profiling Methodology

### 17.1 Layered strategy

| Tool | Question it answers | When |
|---|---|---|
| **Nsight Systems** | Where does wall time go? Launch overhead, sync stalls, gaps, transfer/compute overlap, CPU-side bottlenecks | First, on any new implementation; whenever host time and device time diverge |
| **Nsight Compute** | Why is *this kernel* slow? Occupancy, registers, shared memory, memory throughput, transaction efficiency, warp stalls, Tensor Core utilisation, roofline position | After Nsight Systems shows the kernel dominates; on every GEMM stage |
| **CUPTI** | Programmatic counter access for automated regression tracking | Deferred to v0.2 — not needed while ad-hoc profiling suffices |

### 17.2 The contamination rule

**Profiled runs are separate runs with `timing_valid_for_reporting = false`.** Nsight Compute serialises kernels and replays them to collect counters, which changes execution time by an amount that is workload-dependent and not correctable. A profiled timing is therefore not a performance measurement and is structurally prevented from entering the performance dataset.

What a profiled run *does* contribute: counters, occupancy, roofline coordinates, stall reasons. These are stored in the profiling tier and joined to the corresponding unprofiled run by experiment ID and identical workload configuration.

### 17.3 Artefacts

Naming: `{experiment_id}_{run_id}_{tool}_{timestamp}.{nsys-rep|ncu-rep}`, stored under the run's `profiling/`, checksummed, indexed in DuckDB with the counter set that was collected. Counters are extracted to Parquet at ingest so analysis does not require the GUI tools.

Every profiled run records the exact profiler invocation (tool version, metric set, replay mode), because a counter set collected with different options is not comparable to another.

---

## 18. AMD/ROCm Strategy

**Status in v0.1: interface only, no execution.** No ROCm hardware access is confirmed (verified absent in the dev environment). Shipping an untested AMD backend and implying AMD support would be a false claim.

**Decision — HIP as the portability layer, with a native escape hatch.**
The `HipBackend` satisfies the same `Backend` concept. Kernels are ported via HIP, and `rocBLAS`/`hipBLASLt` provide the vendor baseline. But the interface deliberately permits a backend to supply an entirely native implementation rather than a source-translated one — because forcing source uniformity across architectures with different wavefront sizes, LDS capacities, and matrix-core semantics produces code that is portable and simultaneously unrepresentative of both platforms.

**The equivalence PHOENIX targets is *semantic*: the same mathematical workload, the same correctness standard, the same measurement methodology — not the same source code.**

**Comparability policy (hard rule).** A PHOENIX figure may **never** place NVIDIA and AMD results on the same axis without an explicit comparability declaration stating: which implementations were compared (naive-vs-naive is not a meaningful hardware comparison; vendor-library-vs-vendor-library is a different and more defensible one), what tuning effort was applied to each (unequal tuning effort is the dominant confound in every published cross-vendor GPU comparison), the precision semantics on each side, and the power-measurement semantics on each side (NVML board power and AMD SMI power are not the same measurement).

v0.1 deliverable: interface, build path, container, and documented methodology. v0.2 deliverable: execution, once hardware exists.

---

## 19. TPU Strategy

**Status in v0.1: adapter interface and metadata schema only.**

**Decision — an out-of-process TPU adapter, not an in-process backend.**
*Why:* TPU execution goes through JAX/XLA in Python, on a host runtime with a fundamentally different execution model — no kernel launches, no explicit device allocation, ahead-of-time compilation of a whole computation. Forcing this into the in-process C++ backend interface would either distort the interface or misrepresent TPU semantics. The adapter runs as a separate process, speaks the same `RunRequest`/`RunResult` JSON contract, and writes the same reproducibility package.

**Decision — JAX as the primary TPU frontend** (PyTorch/XLA as a secondary cross-check in v0.2+). JAX's compilation model is more explicit, making compile-vs-execute separation cleaner to measure.

**Mandatory separation of TPU timings** — collapsing these is the standard way TPU benchmarks mislead:

```text
trace/lowering time
XLA compilation time          ← often dominant, always reported separately
first execution (post-compile)
steady-state execution
host↔device transfer
host-side overhead
```

**TPU-specific metadata:** TPU generation, chip count, topology/slice shape, host runtime version, JAX/libtpu versions, XLA flags, donated-buffer semantics, sharding specification.

**Comparability policy:** a GPU-vs-TPU claim from one GEMM shape is not a claim about architectures. It is a claim about one shape, one precision, one compiler, on one day. Figures must say so.

---

## 20. Testing Architecture

| Tier | Scope | Runs where | Trigger |
|---|---|---|---|
| **T0 static** | format, lint, `mypy --strict`, clang-tidy, schema validity | Any runner | Every PR |
| **T1 unit (Python)** | config validation, sweep expansion, statistics, store integrity, provenance rules, report generation | CPU runner | Every PR |
| **T2 unit (C++)** | timing utilities, result serialisation, schema round-trip, error paths, sample buffer, comparator | CPU runner | Every PR |
| **T3 correctness (CPU)** | GEMM-0 vs BLAS across the full §14.3 matrix | CPU runner | Every PR |
| **T4 correctness (CUDA)** | GEMM-1..7 vs GEMM-0 and cuBLAS, full shape/dtype matrix | **Self-hosted GPU** | Merge to main, nightly |
| **T5 integration** | full path: config → run → store → analysis → figure → report, CPU backend | CPU runner | Every PR |
| **T6 integration (GPU)** | same, CUDA backend, small sizes | Self-hosted GPU | Nightly |
| **T7 property-based** | schema round-trips, statistical function invariants, provenance-propagation invariants | CPU runner | Every PR |
| **T8 failure paths** | malformed config, OOM, missing device, correctness failure, checksum mismatch, throttling injection | CPU + GPU | Every PR (CPU) |

**Property tests that matter most** (hypothesis-driven):
- Any config that validates must serialise, deserialise, and re-validate identically.
- Aggregates recomputed from stored raw samples must equal the stored aggregates exactly.
- A derived metric must never carry provenance `MEASURED`. (Invariant enforced by test, not by discipline.)
- Provenance class must propagate correctly through every combination operation — combining `MEASURED` with `VENDOR_REPORTED` must yield `DERIVED`, never `MEASURED`.
- No figure builder may emit a figure whose underlying values lack a provenance class.

**Explicitly not tested by CI: performance.** No performance assertion runs on any shared runner. Performance regression tracking is a nightly job on dedicated hardware, and it *reports*, it does not *gate* (§21).

---

## 21. CI/CD Architecture

**Decision — GitHub Actions.** Repository already on GitHub; self-hosted runner support is what the GPU tiers need; no competing requirement justifies another system.

### 21.1 Pipelines

| Workflow | Runner | Trigger | Contents |
|---|---|---|---|
| `pr-checks` | GitHub-hosted Ubuntu | every PR | T0, T1, T2, T3, T5, T7, T8(CPU); CPU build; docs build; **schema back-compat check** |
| `cuda-build` | GitHub-hosted, CUDA toolkit, **build only** | every PR | Compiles CUDA sources — catches syntax/API errors without needing a GPU |
| `gpu-nightly` | **self-hosted NVIDIA** | nightly + on merge to main | CUDA build, T4, T6, small regression suite (reporting only) |
| `perf-tracking` | self-hosted NVIDIA | nightly | Runs a fixed experiment set, ingests to the catalogue, publishes a trend report. **Never fails a build on a performance change.** |
| `rocm-build` | GitHub-hosted, ROCm image, build only | every PR (v0.2) | Compile-only until hardware exists |
| `tpu-suite` | manual / scheduled | on demand (v0.2) | TPU access is metered and slow; per-PR TPU CI is not justifiable |
| `release` | GitHub-hosted | tag | Version consistency, changelog, container publish, docs deploy |

### 21.2 Rationale for the CPU/GPU split

This directly reflects the verified environment: **the standard development environment has no GPU.** Therefore:

- Everything that *can* be tested without a GPU *must* be, so that the vast majority of development is unblocked by hardware availability.
- CUDA code must at minimum *compile* in PR CI, because a compile error discovered a day later on a nightly GPU run is an expensive feedback loop.
- Correctness on GPU is gated at merge, not at PR, because self-hosted GPU capacity is finite and PR-time GPU queueing would stall the project.

### 21.3 Why performance never gates CI

Shared runners are virtualised, noisy-neighboured, thermally uncontrolled, and of unspecified topology. A performance number from such a machine fails PHOENIX's own validity criteria (§32). Gating on it would either produce constant false failures or — worse — train the team to ignore failures. Performance is tracked on dedicated hardware and reported to humans who apply judgement.

### 21.4 Schema back-compatibility gate

A CI job replays a corpus of stored historical result files against the current schema. Any change that makes a historical file unreadable fails the PR unless the change is accompanied by a schema major-version bump and a documented migration. This is the mechanism that enforces Rule 6 and §37 mechanically rather than by intention.

---

## 22. Container Strategy

Three images: `phoenix-cpu`, `phoenix-cuda`, `phoenix-rocm`.

**The layer distinction that must not be blurred:**

```text
HOST (not containerised, must be recorded)
  ├── kernel
  ├── NVIDIA/AMD kernel driver          ← the container CANNOT control this
  └── container runtime + GPU toolkit
CONTAINER (fully controlled and digest-pinned)
  ├── CUDA/ROCm user-space toolkit
  ├── math libraries (cuBLAS, rocBLAS)
  ├── compilers
  ├── Python environment
  └── PHOENIX
```

A container digest therefore pins the *user-space* environment only. The environment manifest records the host driver separately, and the environment fingerprint is a hash over **both**. A common and serious error in reproducible-benchmarking claims is asserting "reproducible because containerised" while the host driver — which materially affects performance — floats freely. PHOENIX does not make that claim.

Images are digest-pinned in every reproducibility package. Base images are pinned by digest, not tag. Builds are as deterministic as practical (pinned apt snapshots where feasible), and where full determinism is not achievable that is documented rather than glossed.

---

## 23. Data and Database Architecture

**Decision — DuckDB + Parquet + JSON + YAML. No server.**

*Why:* a research measurement corpus is write-once, read-many, analytical, single-user, and must remain readable in ten years. Parquet is a stable open columnar format readable without PHOENIX. DuckDB queries it in-process with full SQL and zero operational burden. PostgreSQL was considered and rejected: a server process, migrations, and backups are operational cost with no analytical benefit at this scale, and it would make the corpus dependent on running infrastructure. SQLite was considered: fine for metadata, poor for columnar analytical scans over millions of samples.

**Critical property:** DuckDB is a *catalogue and query engine over the Parquet files*, **not a copy of them**. The Parquet files under `results/raw/` are the system of record. The DuckDB file is disposable and fully rebuildable by re-ingesting. This means a corrupted or lost database costs nothing, and it is impossible for the database to drift from the raw data.

### 23.1 Entities

```text
hardware_device        (id, revision) ─┐
software_environment   (fingerprint)  ─┤
experiment_definition  (id, version, commit) ─┤
                                              ├─> benchmark_run (run_id)
                                                     │
                                                     ├─> raw_sample     (Tier 1)
                                                     ├─> telemetry_sample
                                                     ├─> aggregate      (Tier 2)
                                                     ├─> derived_metric (Tier 3)
                                                     ├─> reported_value (Tier 4)
                                                     ├─> profiling_run ─> profiler_counter
                                                     ├─> artifact       (path, checksum)
                                                     └─> validity_event

hypothesis (id) ──> experiment_definition
research_paper (id) ──> claim ──> hypothesis     (literature informs hypotheses)
figure (id) ──> reported_value                   (every figure traceable to values)
```

`figure → reported_value → aggregate → raw_sample` is a complete provenance chain. **Any published figure can be traced to the individual iteration timings that produced it.** That chain is the single most valuable property of this architecture and it is why the four tiers are separated.

### 23.2 Migration strategy

Schema version stamped in every file. Migrations are explicit, versioned, forward-only scripts that write *new* files and never modify originals. Historical data is readable via versioned readers; when semantics change incompatibly, old data is marked with its original schema version and the analysis layer refuses to mix incompatible versions silently.

---

## 24. Research-Paper Database

One YAML file per paper in `research/papers/`, schema-validated. **Not a bibliography — an evidence database.**

```yaml
schema_version: "1.0.0"
paper_id: "PAP-0001"
title: "…"
authors: ["…"]
year: null
venue: {name: "…", type: CONFERENCE, tier_notes: "…"}
identifiers: {doi: "…", arxiv: "…", url: "…"}
artifacts: {code: "…", data: "…", reproducible: UNKNOWN}

categories: [photonic_computing, optical_neural_networks]

hardware_studied:
  - {kind: PHOTONIC, description: "…", fabrication: "…", scale: "…"}

methodology:
  type: EXPERIMENTAL              # EXPERIMENTAL | SIMULATION | ANALYTICAL | SURVEY
  measured_on_physical_hardware: true
  simulator: null
  workloads: ["…"]
  precision: "…"
  baseline_compared_against: "…"
  baseline_fairness_assessment: |
    Was the baseline tuned? Same precision? Same batch size? Same problem size?
    Was the comparison end-to-end or component-only?

claims:
  - claim_id: "PAP-0001-C1"
    statement: "…"
    quantity: {name: "energy per MAC", value: null, unit: "…"}
    evidence_class: MEASURED       # MEASURED | SIMULATED | ANALYTICAL | PROJECTED
    scope: "optical core only"     # ← what the number DOES and does NOT include
    excludes: ["laser", "DAC", "ADC", "memory", "control electronics"]
    uncertainty_reported: false
    confidence_in_claim: MEDIUM

limitations:
  stated_by_authors: ["…"]
  identified_by_phoenix: ["…"]

phoenix_relevance:
  relevance: HIGH
  informs_hypotheses: ["HYP-004"]
  reproducible_by_phoenix: PARTIAL
  follow_up_experiments: ["EXP-0xx"]

provenance:
  added_by: "…"
  added_on: "…"
  read_status: FULL_READ           # FULL_READ | SKIMMED | ABSTRACT_ONLY
  verified_by: "…"
```

**The `claims[].scope` and `claims[].excludes` fields are the most important in this schema.** The dominant failure mode in photonic-computing literature comparison is exactly the one §4 of the research context identifies: an optical-core energy figure compared against a full-system GPU figure. Recording, per claim, *what the number includes and excludes* is what will let PHOENIX later assemble an honest cross-paper comparison instead of an apples-to-oranges table.

`read_status` is mandatory and honest. A claim extracted from an abstract is marked as such. **A paper may never be cited in PHOENIX output at `ABSTRACT_ONLY` status.**

`research/papers.csv` is a generated export. Editing it is a no-op that will be overwritten — the YAML is the source of truth.

---

## 25. Reproducibility Architecture

Every run emits:

```text
results/raw/<run_id>/
├── MANIFEST.json          # every file + BLAKE3 checksum + schema versions
├── config.yaml            # the exact config, as authored
├── config.normalized.json # post-validation, content-hashed
├── request.json           # the exact bytes crossing the Python→C++ boundary
├── environment.json       # OS, kernel, driver, toolkit, libs, compiler flags, env vars
├── hardware.json          # discovered device state at run time
├── git.json               # commit, branch, dirty flag, diff if dirty
├── build.json             # compiler, flags, CUDA architectures, lib versions, build ID
├── calibration.json       # timer overhead characterisation
├── results.json           # run record (§10.5)
├── samples.parquet        # Tier 1, immutable
├── telemetry.parquet      # power/clocks/temperature time series
├── logs/                  # structured JSONL, both planes
└── profiling/             # present only for profiled runs
```

**Environment fingerprint** = BLAKE3 over the normalised environment manifest (OS, kernel, driver, toolkit, library versions, compiler + flags, target architectures, container digest, CPU model, GPU model + UUID). Two runs with the same fingerprint are comparable by default; differing fingerprints require an explicit override that stamps every downstream artefact.

**Dirty-tree policy:** running with uncommitted changes is permitted (research requires it) but the full diff is embedded in `git.json` and the run is marked `provisional`. A provisional run may **never** be the source of a value in a published figure.

**Reproduction command:** `phoenix reproduce results/raw/<run_id>` re-executes from the stored package, then diffs the new environment against the stored one and reports every difference before running — so the researcher knows what changed before interpreting why the result did.

---

## 26. Documentation Architecture

| Document | Audience | Rule |
|---|---|---|
| README | Everyone | States what PHOENIX is and — equally prominently — what it is not |
| Getting Started | New user | Must work end-to-end on a CPU-only machine, no GPU |
| Architecture | Contributor | This document's descendants |
| **Methodology** | Reviewer | **Every benchmark has a methodology page. No exceptions.** |
| Profiling guide | Contributor | |
| Reproducibility guide | Reviewer | How to re-run and what "same" means |
| Experiment catalogue | Researcher | Generated from `experiments/`; every EXP-xxx with hypothesis, status, results link |
| Research notes | Researcher | Hypothesis records (§31 of master prompt) |
| Hardware coverage | User | Which devices have records, at what confidence |
| ADR index | Contributor | Generated from `docs/adr/` |
| API reference | Contributor | mkdocstrings + Doxygen |
| Release notes | Everyone | Includes schema changes and their comparability impact |

**Two hard rules:**
1. **Every published figure carries its experiment ID and run IDs in the caption**, and every figure is traceable through `figure → reported_value → aggregate → raw_sample`.
2. **Every figure legend distinguishes provenance classes visually** — measured and analytical values never share an undifferentiated line style. A roofline plot showing a theoretical roof and an attained point must make the distinction visible without reading the caption.

---

## 27. First 10 Experiments

Each is a scientific unit with a hypothesis stated *before* execution. **No results appear here. Where a result would go, the text says `NOT YET MEASURED`.**

Common to all: raw samples retained; correctness gate before measurement; environment manifest captured; reproducibility package emitted; median with bootstrap CI as the primary statistic; distribution published.

---

### EXP-001 — CPU Reference GEMM

**Hypothesis:** A naive triple-loop FP64 CPU GEMM will be compute-bound at small sizes and memory-bound at large sizes, with a transition detectable in the measured scaling curve.
**Objective:** Establish the ground-truth reference and validate the entire measurement pipeline on hardware that is always available.
**Workload:** GEMM, square 128→4096, FP64 and FP32.
**Hardware:** Host CPU (`hardware/cpu/` record required first).
**Independent variables:** matrix size, dtype. **Controls:** single-threaded; CPU frequency scaling recorded; no other load.
**Measured:** host wall time per iteration, correctness vs an independent BLAS.
**Derived:** FLOP/s, arithmetic intensity, roofline position.
**Correctness:** bitwise-reproducible run-to-run; matches reference BLAS within FP64 tolerance.
**Profiling:** none (out of v0.1 CPU scope).
**Invalidation:** CV above gate; frequency scaling observed mid-run; background load detected.
**Can conclude:** that the pipeline works end-to-end, and how this specific CPU implementation scales.
**Cannot conclude:** anything about CPU architecture generally, or anything comparative with GPUs.
**Result:** NOT YET MEASURED.

---

### EXP-002 — Naive CUDA GEMM

**Hypothesis:** A one-thread-per-output-element CUDA GEMM will be global-memory-bandwidth-bound at all tested sizes, with attained bandwidth far below vendor-reported peak due to absent data reuse.
**Objective:** Establish the GPU floor and demonstrate the measurement path on device.
**Workload:** square 128→4096 plus rectangular and non-power-of-two shapes; FP32.
**Variables:** size, shape. **Controls:** fixed clocks where possible (recorded either way); exclusive device; identical inputs to EXP-001.
**Measured:** device kernel time, host iteration time, API overhead, power, clocks, temperature.
**Derived:** FLOP/s, attained bandwidth, arithmetic intensity, fraction of vendor-reported peak (labelled as a ratio against a `VENDOR_REPORTED` value, never as an efficiency claim about the hardware).
**Correctness:** vs EXP-001 CPU reference and vs cuBLAS.
**Profiling:** Nsight Compute in a separate run — memory throughput, transaction efficiency, occupancy, warp stall reasons.
**Invalidation:** correctness failure; throttling; measured time within 10× of timer-calibration floor (small sizes are expected to trip this — that finding is itself a result).
**Can conclude:** how this kernel behaves on this GPU with this toolkit.
**Cannot conclude:** anything about GPU capability — this kernel is deliberately bad.
**Result:** NOT YET MEASURED.

---

### EXP-003 — Shared-Memory Tiled CUDA GEMM

**Hypothesis:** Shared-memory tiling raises data reuse, reducing global-memory traffic per FLOP, shifting the kernel's roofline position rightward and increasing attained FLOP/s relative to EXP-002 for sizes ≥ 1024.
**Variables:** matrix size, tile size (16/32). **Controls:** identical to EXP-002 in every other respect.
**Measured:** as EXP-002, plus shared-memory bank-conflict counters (profiled run).
**Derived:** as EXP-002, plus speedup vs EXP-002 with CI, plus measured reduction in global-memory traffic.
**Correctness:** all shapes including dimensions not a multiple of tile size — the boundary-condition case is the point.
**Invalidation:** as EXP-002, plus disagreement between the measured traffic reduction and the tiling theory (which would indicate the kernel is not doing what is believed).
**Can conclude:** the measured effect of tiling on this kernel/GPU/size range.
**Cannot conclude:** that tiling is optimal, or that the observed change is entirely attributable to reuse — that requires the profiler evidence.
**Result:** NOT YET MEASURED.

---

### EXP-004 — Register-Tiled CUDA GEMM

**Hypothesis:** Per-thread register micro-tiling further raises arithmetic intensity, but increased register pressure reduces occupancy; there exists a micro-tile size that maximises attained throughput, and it is not the largest one.
**Variables:** micro-tile dimensions, matrix size. **Controls:** as EXP-003.
**Measured:** as EXP-003, plus registers-per-thread and achieved occupancy from the profiled run.
**Derived:** throughput vs occupancy relationship — the trade-off curve is the deliverable.
**Invalidation:** register spilling detected but not recorded; occupancy data unavailable.
**Can conclude:** where the occupancy/ILP trade-off sits for this kernel on this GPU.
**Cannot conclude:** a general rule about register tiling.
**Result:** NOT YET MEASURED.

---

### EXP-005 — cuBLAS and cuBLASLt Baselines

**Hypothesis:** Vendor libraries will attain substantially higher throughput than GEMM-1..4, and cuBLASLt will differ from cuBLAS in the mixed-precision cases due to different algorithm selection.
**Objective:** Establish the vendor reference; quantify — with a number, not an adjective — the gap PHOENIX kernels must explain.
**Variables:** size, dtype, layout. **Controls:** identical workload, identical measurement protocol as EXP-002..004.
**Measured:** as above; **plus the selected algorithm / heuristic ID where the API exposes it** — a cuBLASLt result without recording which algorithm was chosen is not reproducible.
**Note:** cuBLAS and cuBLASLt are recorded as two distinct baselines, never merged.
**Invalidation:** algorithm selection not recorded; autotuning cache state not controlled between repetitions.
**Can conclude:** vendor-library performance on this GPU under this protocol.
**Cannot conclude:** the hardware's maximum capability.
**Result:** NOT YET MEASURED.

---

### EXP-006 — Reduced Precision and Tensor Core Path

**Hypothesis:** FP16/BF16 inputs with FP32 accumulation on Tensor Cores will attain substantially higher throughput than FP32 CUDA-core paths, and BF16 will show smaller numerical error than FP16 at large K due to its wider exponent range.
**Objective:** Characterise the throughput/accuracy trade-off *jointly*. Throughput alone is not the result.
**Variables:** dtype combination (FP32, TF32, FP16→FP32, BF16→FP32), K (the reduction length that drives error accumulation), size.
**Controls:** identical shapes, identical inputs, identical protocol.
**Measured:** timing; **and numerical error vs the FP64 CPU reference for every configuration.**
**Derived:** throughput, and error-vs-throughput trade-off curves.
**Correctness:** dtype-appropriate tolerance (§14.2). A tolerance failure here is a finding about precision, not necessarily a bug — the two are distinguished by cross-checking against cuBLAS at the same dtype.
**Invalidation:** Tensor Core utilisation counter zero (the kernel did not use the intended hardware path); accumulate dtype not recorded.
**Can conclude:** the measured throughput/accuracy trade-off on this GPU for GEMM.
**Cannot conclude:** that reduced precision is acceptable for any model — that requires end-to-end model accuracy, which is v0.3.
**Result:** NOT YET MEASURED.

---

### EXP-007 — Nsight Kernel Characterisation

**Hypothesis:** For each PHOENIX kernel stage, a dominant bottleneck is identifiable from profiler counters, and the identified bottleneck explains the difference in measured throughput between consecutive stages.
**Objective:** Move from *what* to *why*. This experiment is what converts EXP-002..006 from numbers into understanding.
**Workload:** one representative size per kernel stage.
**Measured:** occupancy, registers, shared-memory usage/conflicts, memory throughput, transaction efficiency, warp stall reason distribution, Tensor Core utilisation, roofline coordinates.
**Deliverable:** a per-stage bottleneck analysis stating what was hypothesised, what the counters show, and **what alternative explanations were ruled out and how**.
**Invalidation:** incomplete counter collection; profiler version differing across stages; profiled timings leaking into the performance dataset (structurally prevented, but asserted in test).
**Can conclude:** the bottleneck for each kernel at the profiled configuration.
**Cannot conclude:** that the bottleneck is the same at other sizes — that requires profiling at those sizes.
**Result:** NOT YET MEASURED.

---

### EXP-008 — Matrix-Size Scaling Study

**Hypothesis:** Attained throughput increases with size then saturates; the saturation point differs per implementation; small sizes are dominated by launch overhead rather than compute, and this is detectable by comparing device time against the timer-calibration floor.
**Objective:** Map the size-dependence of every implementation on one axis — the closest v0.1 comes to a "which is better when" answer, and the direct methodological ancestor of the later photonic break-even analysis.
**Variables:** size, swept finely (including non-powers-of-two and tile-boundary neighbours); implementation.
**Measured:** timing across all implementations at all sizes, plus launch overhead as a separate quantity.
**Derived:** throughput-vs-size curves per implementation; overhead-dominated regime boundary; crossover points where implementation ranking changes.
**Invalidation:** insufficient size resolution near a suspected crossover; cache-residency change across the sweep not documented.
**Can conclude:** the size-dependent ranking of these implementations on this GPU.
**Cannot conclude:** anything about other GPUs or other workloads.
**Result:** NOT YET MEASURED.
**Note:** this experiment establishes the methodology template that v0.6 will reuse for `E_GPU(N)` vs `E_photonic(N)` break-even analysis. Getting it right here is disproportionately valuable.

---

### EXP-009 — AMD HIP Port (v0.2 execution; v0.1 defines it)

**Hypothesis:** HIP source translation of GEMM-1..4 will compile and produce numerically correct results on AMD hardware, but the performance ranking of the stages may differ from NVIDIA due to differing wavefront size, LDS capacity, and matrix-core semantics.
**Status in v0.1:** definition, interface, container, and methodology only. **No AMD hardware access is confirmed.**
**Can conclude (when run):** that the workload is semantically portable, and how these specific kernels behave on that specific AMD GPU.
**Cannot conclude:** that NVIDIA or AMD is "faster" — see §18's comparability policy, in particular the unequal-tuning-effort confound.
**Result:** NOT YET MEASURED. NOT YET EXECUTABLE.

---

### EXP-010 — TPU Matrix Multiplication (v0.2 execution; v0.1 defines it)

**Hypothesis:** XLA compilation time will dominate first execution by a large factor, steady-state execution will differ substantially from first execution, and reporting a single "TPU GEMM time" without that separation would misrepresent the platform.
**Status in v0.1:** adapter interface and metadata schema only. **No TPU access is confirmed.**
**Measured (when run):** trace/lowering, compile, first execution, steady state, transfer, host overhead — separately.
**Can conclude (when run):** the TPU's timing structure for this shape under this compiler.
**Cannot conclude:** GPU-vs-TPU architectural superiority from a single GEMM.
**Result:** NOT YET MEASURED. NOT YET EXECUTABLE.

---

### Experiment dependency graph

```text
EXP-001 (CPU ref, validates whole pipeline)
   │
   ├──> EXP-002 (naive CUDA) ──> EXP-003 (tiled) ──> EXP-004 (register tiled)
   │                                   │                     │
   │                                   └─────────┬───────────┘
   │                                             v
   ├──> EXP-005 (cuBLAS/Lt baseline) ──────> EXP-007 (Nsight characterisation)
   │              │                                  │
   │              └──> EXP-006 (precision/TC) ───────┘
   │                              │
   └──────────────────────────────┴──> EXP-008 (scaling; consumes all of the above)
                                              │
                                   ┌──────────┴──────────┐
                                   v                     v
                              EXP-009 (HIP)         EXP-010 (TPU)
                                 [v0.2]                [v0.2]
```

---

## 28. Risk Register

| ID | Risk | Prob. | Impact | Detection | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R-01 | **No GPU in the primary dev environment** (confirmed) | Certain | High | Already observed | CPU-first architecture; compile-only CUDA CI; self-hosted GPU runner for correctness; GPU work batched | Lead |
| R-02 | CUDA/driver version drift breaks comparability | High | High | Environment fingerprint mismatch on ingest | Fingerprint gate; analysis refuses cross-fingerprint aggregation by default | Infra |
| R-03 | Thermal throttling silently distorts results | High | High | Clock/temp telemetry + throttle bitmask per sample | Cooldown between repetitions; throttle-flag invalidation; pre/post state diff | Bench |
| R-04 | Benchmark noise exceeds the effect being measured | High | High | CV gate; bootstrap CI width; A-vs-A control | Repetitions as separate processes; CI-based comparison rule; A-vs-A control mandatory before any A-vs-B claim | Bench |
| R-05 | Power telemetry semantics misunderstood (sampling window, what the sensor includes) | High | High | Characterisation experiment vs external meter | Treat NVML semantics as `UNKNOWN` until characterised; no energy claim published before then | Bench |
| R-06 | Timing floor: small-size results dominated by launch overhead | High | Medium | Timer calibration + 10× floor flag | Explicit `MEASUREMENT_FLOOR_RISK` flag; report overhead separately | Bench |
| R-07 | AMD hardware never becomes available | Medium | Medium | Schedule | Interface-only in v0.1; no support claimed; compile-only CI | Lead |
| R-08 | TPU access never becomes available | Medium | Medium | Schedule | Out-of-process adapter isolates the dependency; no support claimed | Lead |
| R-09 | Schema evolution breaks historical data | Medium | High | Back-compat replay job in CI | Versioned schemas, forward-only migrations, back-compat gate | Infra |
| R-10 | Premature abstraction obscures hardware behaviour | Medium | Medium | Review; if a researcher cannot see what the hardware did, the abstraction failed | Thin backend interface; direct access to raw device timings preserved | Lead |
| R-11 | Over-optimising a kernel without profiling | Medium | Medium | Review gate | No optimisation commit without a linked profiling artefact | Review |
| R-12 | **AI-generated plausible-but-wrong technical assumptions** | High | High | Every vendor fact requires a primary-source citation in the hardware DB | ⚠ markers; `provenance` mandatory; unsourced value cannot be stored | All |
| R-13 | Invalid cross-vendor comparison published | Medium | Very High | Review; comparability stamp required on any cross-fingerprint figure | Hard rule §18/§19; figure generation refuses without declaration | Review |
| R-14 | Correctness bug hidden by loose tolerance | Medium | Very High | Adversarial structured inputs; tolerance constants reviewed | Tolerance never loosened to pass; structured index-encoded test inputs | Review |
| R-15 | Scope creep into photonics/compiler before baseline is done | High | High | v0.1 DoD checklist | Non-goals §3; gate photonic work behind v0.1 completion | Lead |
| R-16 | Results directory grows unmanageably | Medium | Low | Disk monitoring | Parquet compression; archival policy; results git-ignored | Infra |
| R-17 | Single-maintainer bus factor | High | High | — | Documentation-first; schemas as contracts; ADRs record reasoning | Lead |
| R-18 | cuBLASLt autotuning state makes runs irreproducible | Medium | High | Repetition variance; algorithm ID logging | Record selected algorithm; control cache state between repetitions | Bench |
| R-19 | **Hybrid P/E-core CPU produces bimodal timings** that look like a code property but are a scheduler artefact | Certain on dev host | High | Bimodal distribution in raw samples; per-sample core-ID capture | Capture core ID per sample; pin affinity to a single core type on the measurement host; never publish a CPU baseline from a hybrid part without affinity control | Bench |
| R-20 | **WSL2 virtualisation layer distorts host timing** by an uncharacterised amount | Certain on dev host | Medium | Comparison against a native Linux host | Dev-host runs marked `PROVISIONAL`; characterise or avoid (§37 item 11); never publish from WSL2 | Bench |
| R-21 | **Laptop thermal envelope invalidates sustained sweeps** | Certain on dev host | Medium | `THERMAL_THROTTLE` flag rate | Short pipeline-validation runs only on the dev host; real sweeps on the measurement host | Bench |
| R-22 | **No measurement host exists yet** — neither CPU-controlled nor GPU | High | Very High | Schedule | Weeks 1–4 need none; procurement/access is the Week 4 decision point, not the Week 5 surprise | Lead |

---

## 29. Major Engineering Trade-offs

**T1 — Two languages instead of one.**
Chosen: C++ compute plane, Python control plane. Cost: a binding boundary, two toolchains, two test stacks, schema duplication. Benefit: measurement-grade timing that Python cannot provide, plus rapid iteration on orchestration and analysis where C++ would be miserable. Pure Python (CuPy) was rejected because interpreter jitter is comparable to the quantities being measured at small sizes — the exact regime where the interesting overhead findings live. Pure C++ was rejected because the analysis and plotting layer would take an order of magnitude more effort for no scientific gain.

**T2 — JSON across the binding boundary instead of rich bound types.**
Cost: serialisation, two schema definitions. Benefit: an auditable, loggable, byte-reproducible boundary; ABI decoupling. Serialisation cost is once per run and therefore irrelevant. This trade is unambiguously worth it.

**T3 — Four data tiers instead of one results table.**
Cost: more schema, more code, more concepts to learn. Benefit: figures traceable to individual iterations; derived values recomputable when a formula changes; structural impossibility of a `VENDOR_REPORTED` number being presented as `MEASURED`. This is the design's core value proposition — it is where complexity is deliberately spent.

**T4 — No performance gating in CI.**
Cost: a performance regression can merge and be caught only the next night. Benefit: no false failures from noisy shared runners, and no culture of ignoring red builds. Accepted deliberately.

**T5 — Interface-only AMD and TPU support in v0.1.**
Cost: unexercised code paths; interfaces will need revision when hardware arrives. Benefit: honesty (no claimed support that has never run), and the interface design work forces the CUDA backend to avoid CUDA-shaped assumptions — a benefit that accrues even if AMD/TPU hardware never materialises.

**T6 — Retaining every kernel stage forever.**
Cost: maintenance of deliberately slow code, larger test matrix. Benefit: Rule 5 — every optimisation claim has its baseline permanently available and re-measurable on new hardware. Deleting GEMM-1 once GEMM-7 exists would destroy the ability to reproduce the progression on a future GPU.

**T7 — DuckDB as a rebuildable catalogue, not the system of record.**
Cost: an ingest step; queries need the catalogue built. Benefit: the corpus survives any database corruption, and remains readable by anything that reads Parquet in ten years.

---

## 30. v0.1 Definition of Done

v0.1 ships when **every** criterion below is satisfied and demonstrated.

### Functional

- [ ] `git clone` → documented setup → working install on a CPU-only Linux host (native Linux **or** WSL2; both are supported and the manifest records which), with no GPU required for any step in Getting Started.
- [ ] `phoenix discover` emits a complete, schema-valid environment + hardware manifest on CPU-only and on GPU hosts.
- [ ] `phoenix validate <config>` rejects malformed configs with actionable messages; every rejection path has a test.
- [ ] `phoenix run EXP-001` completes end-to-end on CPU and writes a complete reproducibility package. **On the development host this validates the pipeline, not the CPU.** Runs from a hybrid-core, thermally-constrained, or virtualised host are marked `PROVISIONAL` and are not admissible as a published baseline (ADR-032). A publishable EXP-001 requires a controlled measurement host.
- [ ] `phoenix run EXP-002..008` complete on an NVIDIA GPU host.
- [ ] `phoenix analyze` produces aggregates, derived metrics, and roofline positions from stored raw data only.
- [ ] `phoenix report EXP-xxx` produces a report with figures generated **exclusively** from raw data — verified by a test that mutates a raw sample and asserts the figure changes.
- [ ] `phoenix reproduce <run_id>` re-executes and reports every environment difference.

### Scientific

- [ ] All GEMM implementations pass the full §14.3 correctness matrix, including adversarial structured inputs.
- [ ] Timer calibration is performed and recorded in every run; the measurement floor is enforced.
- [ ] Median + bootstrap CI + full distribution produced for every reported quantity.
- [ ] A-vs-A control has been run and its result documented, establishing the noise floor for comparisons.
- [ ] No figure exists anywhere in the repository that was not generated from stored raw data.
- [ ] Every numeric value in the hardware database has a `provenance` class and a source citation, or is explicitly `null`/`UNKNOWN`.
- [ ] Power telemetry semantics are either characterised, or every energy figure is marked `NOT YET MEASURED`. **No energy number ships uncharacterised.**
- [ ] EXP-001 through EXP-008 have written analyses that separate observation, interpretation, and speculation.

### Engineering

- [ ] PR CI green: format, lint, mypy strict, clang-tidy, unit tests both planes, CPU correctness, integration, property tests, schema back-compat, CUDA compile-only.
- [ ] Nightly GPU CI green: CUDA correctness matrix, GPU integration.
- [ ] Three containers build and are digest-pinned.
- [ ] All schemas at v1.0.0, published, with back-compat replay corpus in CI.
- [ ] Documentation complete per §26, including a methodology page per benchmark.
- [ ] ADR register complete (§44).
- [ ] ≥ 50 papers in the literature database at `FULL_READ` or `SKIMMED` status, each with extracted claims carrying `evidence_class`, `scope`, and `excludes`.

### Negative criteria (must all be true)

- [ ] No fabricated number anywhere in the repository, including in documentation, examples, and test fixtures.
- [ ] No cross-vendor performance claim.
- [ ] No photonic or neuromorphic implementation.
- [ ] No performance number originating from shared CI hardware in any published artefact.
- [ ] No hand-edited data file under `results/`.

---

## 31. 12-Week Implementation Plan

Ordering is deliberately changed from the master prompt's suggested sequence in one respect, with technical justification given at Week 8.

Each week: **Objectives / Deliverables / Dependencies / Tests / Acceptance / Risks.**

### Week 1 — Foundations
**Objectives:** repository skeleton, dual build system, schema-first workflow, CI skeleton.
**Deliverables:** directory tree (§8); CMake with CPU target; `pyproject.toml` + uv lock; all six JSON Schemas at v1.0.0-draft; nanobind hello-world proving the boundary; `pr-checks` workflow running T0.
**Dependencies:** none.
**Tests:** schema validity; binding import; build on clean container.
**Acceptance:** clean clone builds and imports `phoenix_core` on a CPU-only machine.
**Risks:** over-designing schemas before use — mitigated by marking them draft until Week 4.

### Week 2 — Discovery and provenance primitives
**Objectives:** know exactly what machine we are on; make provenance a real type.
**Deliverables:** `phoenix discover` (CPU, OS, kernel, compiler, Python, and — when present — GPU/driver/toolkit/NVML); environment fingerprint (BLAKE3); `ProvenanceValue` type in both planes; hardware YAML loader + validator; CPU hardware record for the dev host with full citations.
**Dependencies:** W1.
**Tests:** discovery on CPU-only host; graceful, explicit degradation when NVML absent; provenance round-trip property tests.
**Acceptance:** a complete environment manifest is produced on the GPU-less dev container, with GPU fields explicitly `UNKNOWN` rather than absent.
**Risks:** platform-specific discovery gaps — record `UNKNOWN`, never guess.

### Week 3 — Measurement framework
**Objectives:** the executor, timing, and sample capture — without any real workload.
**Deliverables:** `Backend` concept; `BenchmarkExecutor` (warmup → correctness gate → measure); host timing; timer calibration; sample buffer; result serialisation; a synthetic `sleep` backend for testing the harness itself.
**Dependencies:** W1–W2.
**Tests:** T2 unit; calibration reproducibility; sample-buffer property tests; executor tested against the synthetic backend with known-duration work.
**Acceptance:** the harness measures a known-duration synthetic workload and recovers the known value within calibrated overhead.
**Risks:** designing the executor around CUDA — mitigated by building it against the synthetic backend first, so no CUDA assumption can leak in.

### Week 4 — Storage, CPU GEMM, correctness
**Objectives:** first real measurements and the immutable store.
**Deliverables:** GEMM-0 CPU reference (FP64) + blocked + OpenMP + BLAS-backed; comparator and tolerance policy (§14.2); Parquet writer; MANIFEST + checksums; DuckDB ingest; reproducibility package; **EXP-001 executed**.
**Dependencies:** W3.
**Tests:** T3 full correctness matrix incl. adversarial inputs; store immutability; checksum-tamper detection; T5 integration.
**Acceptance:** EXP-001 runs end-to-end on CPU and produces a complete, checksum-verified package. Schemas promoted to v1.0.0.
**Risks:** schema churn once real data appears — this is why schemas were draft until now.

### Week 5 — CUDA backend and GEMM-1
**Objectives:** first GPU execution. **Requires GPU host access from this point.**
**Deliverables:** CUDA backend (device mgmt, memory, streams, events, checked-call wrapper); NVML telemetry thread; GEMM-1 naive; CUDA timer calibration; `cuda-build` (compile-only) and `gpu-nightly` workflows; **EXP-002 executed**.
**Dependencies:** W4 + physical GPU.
**Tests:** T4 correctness on GPU; T6 integration; event-vs-host timing consistency.
**Acceptance:** EXP-002 completes with correctness passing and a full telemetry time series.
**Risks:** **GPU access is the critical path.** Mitigation: everything through W4 is GPU-free, so a GPU delay does not idle the project; batch GPU work when access is available.

### Week 6 — GEMM-2, GEMM-3
**Deliverables:** shared-memory tiled kernel; register-tiled kernel; boundary-condition handling; **EXP-003, EXP-004 executed**.
**Dependencies:** W5.
**Tests:** correctness at tile-boundary-adjacent dimensions; non-multiple-of-tile shapes.
**Acceptance:** both correct on the full shape matrix; measured comparison against GEMM-1 with CIs; changes explained, not merely reported.
**Risks:** boundary bugs that random inputs hide — mitigated by structured index-encoded adversarial inputs.

### Week 7 — Vendor baselines and Tensor Cores
**Deliverables:** GEMM-4 vectorised; cuBLAS and cuBLASLt wrappers (recording selected algorithm); GEMM-5 Tensor Core path; dtype matrix incl. accumulate types; **EXP-005, EXP-006 executed**.
**Dependencies:** W6.
**Tests:** dtype-appropriate tolerances; error-vs-FP64-reference across all dtypes; Tensor Core utilisation non-zero.
**Acceptance:** vendor baselines recorded with algorithm IDs; precision/accuracy trade-off characterised jointly with throughput.
**Risks:** cuBLASLt autotuning irreproducibility (R-18) — record algorithm selection and control cache state.

### Week 8 — Profiling infrastructure
**Deliverables:** Nsight Systems and Nsight Compute wrappers; artefact naming, storage, checksumming, indexing; counter extraction to Parquet; **structural enforcement of `timing_valid_for_reporting = false`**; **EXP-007 executed** across all kernel stages.
**Dependencies:** W7.
**Tests:** contamination test — assert a profiled run's timings cannot enter the performance dataset; artefact integrity.
**Acceptance:** every kernel stage has a bottleneck analysis with counters and ruled-out alternatives.
**Risks:** profiler version/permission issues on the GPU host — resolve early in W5 rather than discovering here.

> **Justified deviation from the suggested ordering.** The master prompt places power/energy abstraction in Week 9 alongside scaling. This design instead treats **power-telemetry characterisation as a blocking prerequisite in Week 9 before any energy figure is derived**, and permits energy work to slip past v0.1 entirely if characterisation is incomplete. Reason: NVML's sampling window, averaging behaviour, and what the sensor physically includes are all `UNKNOWN` (R-05). Deriving `E = P × T` from an uncharacterised power sensor produces a number that looks authoritative and may be wrong by a large and unknown factor. Since energy is the metric on which the entire eventual photonic comparison rests, an uncharacterised energy pipeline is the most dangerous thing v0.1 could ship. It is therefore gated, not scheduled.

### Week 9 — Scaling, statistics, power characterisation
**Deliverables:** statistical layer (aggregates, bootstrap CIs, outlier flagging, sufficiency check); roofline subsystem; **EXP-008 executed**; A-vs-A noise-floor control; power-telemetry characterisation study.
**Dependencies:** W8.
**Tests:** statistical functions property-tested; aggregate-recomputation equality; roofline correctness on synthetic cases.
**Acceptance:** EXP-008 scaling curves generated purely from raw data; noise floor documented; power semantics either characterised or explicitly marked `NOT YET MEASURED` project-wide.
**Risks:** characterisation may require an external power meter that is unavailable → energy stays `NOT YET MEASURED`, which is an acceptable v0.1 outcome and must not be quietly worked around.

### Week 10 — HIP interface and container hardening
**Deliverables:** `HipBackend` satisfying the concept (compiles, no hardware execution); HIP ports of GEMM-1..3 (compile-verified); ROCm container; `rocm-build` compile-only CI; EXP-009 definition; §18 comparability policy documented.
**Dependencies:** W9.
**Tests:** compile-only; interface conformance test that the concept is satisfied.
**Acceptance:** HIP path compiles; **documentation states explicitly that AMD is unexecuted**.
**Risks:** temptation to claim AMD support — prevented by the negative DoD criteria.

### Week 11 — TPU adapter and literature database
**Deliverables:** out-of-process TPU adapter contract + stub; TPU metadata schema; EXP-010 definition; literature database schema, loader, and validator; **≥ 50 papers ingested with extracted claims**; `papers.csv` export generator.
**Dependencies:** W10.
**Tests:** paper schema validation; claim-extraction round-trip; `read_status` enforcement (abstract-only papers cannot be cited).
**Acceptance:** literature database queryable via DuckDB; 50-paper target met with honest `read_status` on each.
**Risks:** literature work is large and easily deferred — it is scheduled explicitly and is a DoD criterion for exactly that reason.

### Week 12 — Release hardening
**Deliverables:** full documentation; ADR register; reproducibility verification (independent re-run of every experiment from its package); back-compat replay corpus; CHANGELOG; CITATION.cff; v0.1.0 tag.
**Dependencies:** W1–W11.
**Tests:** full suite; reproduce-from-package for every experiment; DoD checklist walked item by item.
**Acceptance:** every §30 box ticked, including all negative criteria.
**Risks:** DoD items discovered incomplete late — mitigated by walking the checklist at the end of Weeks 4, 8, and 11 rather than only at 12.

**Critical path:** GPU host access from Week 5. Weeks 1–4 and the literature work in Week 11 are GPU-free and can absorb a GPU-availability delay without idling the project.

---

## 32. Invalid Benchmark Conditions

Detected automatically where possible; recorded as `invalidation_reasons[]`; **the run's data is always retained**.

| Code | Condition | Detection | Default action |
|---|---|---|---|
| `CORRECTNESS_FAILED` | Result outside tolerance | Comparator | INVALID |
| `THERMAL_THROTTLE` | Vendor throttle flag set during measurement | Telemetry bitmask | INVALID |
| `POWER_THROTTLE` | Power-cap throttling | Telemetry bitmask | INVALID |
| `CLOCK_INSTABILITY` | Clock variance beyond threshold during measurement | Telemetry | FLAG |
| `ENVIRONMENT_DRIFT` | Pre/post state differ beyond threshold | State diff | FLAG |
| `BACKGROUND_LOAD` | Other processes on the device | Device query | INVALID if `require_clean_environment` |
| `INSUFFICIENT_SAMPLES` | Valid samples below `min_valid_samples` | Count | INVALID |
| `HIGH_VARIANCE` | CV above `max_cv` | Statistics | FLAG (INVALID if configured) |
| `MEASUREMENT_FLOOR_RISK` | Duration within 10× of calibrated overhead | Calibration | FLAG |
| `INSUFFICIENT_WARMUP` | Warmup adequacy check failed | §15.3 | FLAG |
| `PRECISION_TARGET_NOT_MET` | CI half-width target not reached at iteration cap | §15.5 | FLAG |
| `POWER_UNAVAILABLE` | Telemetry required but absent | Runtime | INVALID if `power.required` |
| `PROFILING_INCOMPLETE` | Counter collection partial | Profiler exit | INVALID for the profiling artefact only |
| `UNSUPPORTED_PRECISION` | Backend cannot honour requested dtype | Capability check | Rejected before execution |
| `CHECKSUM_MISMATCH` | Stored file altered | Ingest verification | TAMPERED — excluded from all analysis |
| `DIRTY_TREE` | Uncommitted changes | Git probe | PROVISIONAL — usable, never publishable |
| `ENVIRONMENT_FINGERPRINT_MISMATCH` | Aggregating across environments | Analysis | Requires explicit override + stamp |

**Invalid runs are never deleted.** A pattern of `THERMAL_THROTTLE` across a machine is a finding about that machine, and deleting the evidence destroys it.

---

## 33. Roofline Analysis

```text
Arithmetic Intensity (AI) = FLOPs / Bytes moved from device memory
Attainable FLOP/s = min(Peak FLOP/s, AI × Peak Bandwidth)
```

Four distinct lines/points, **visually distinguished by provenance class**:

| Element | Provenance | Source |
|---|---|---|
| Theoretical compute roof | `VENDOR_REPORTED` | Hardware DB, with `conditions` shown in the caption |
| **Empirical compute ceiling** | `MEASURED` | Best attained across all PHOENIX kernels + vendor libraries |
| Theoretical bandwidth roof | `VENDOR_REPORTED` | Hardware DB |
| **Empirical bandwidth ceiling** | `MEASURED` | Dedicated bandwidth microbenchmark |
| Kernel position | `MEASURED` (FLOP/s) + `DERIVED` (AI) | Run data + profiler traffic counters |

**Two rules:**
1. **Theoretical peak is never drawn in the same style as an attained value.** Vendor roofs are dashed and labelled `VENDOR_REPORTED`; measured ceilings are solid.
2. **AI is computed from profiler-measured memory traffic where available, not from the analytical minimum.** The analytical AI assumes perfect caching; the measured AI reflects what actually moved. Both are stored, clearly distinguished, because the gap between them *is* the cache-efficiency result.

---

## 34. Architecture Decision Register

| ADR | Decision | Status | Reason | Revisit trigger |
|---|---|---|---|---|
| ADR-001 | C++20 compute plane / Python 3.11 control plane | Accepted | Timing fidelity + analysis velocity | Python timing becomes adequate; or nvcc C++20 gaps block work |
| ADR-002 | nanobind for bindings | Accepted | Lower overhead, smaller, modern | Ecosystem gap blocks a needed feature |
| ADR-003 | JSON-string API across the binding boundary | Accepted | Auditable, loggable, ABI-decoupled | Serialisation ever becomes measurable in a run's cost |
| ADR-004 | Provenance class mandatory on every numeric value | **Accepted — foundational** | Prevents category mixing; enables future photonic comparison | Never |
| ADR-005 | Four data tiers (raw / aggregate / derived / reported) | Accepted | Figures traceable to iterations; derived values recomputable | Never |
| ADR-006 | Raw data immutable + checksummed | Accepted | Rule 6 | Never |
| ADR-007 | DuckDB + Parquet, catalogue rebuildable from raw | Accepted | Local-first, archival, zero ops | Corpus outgrows single-node analysis |
| ADR-008 | YAML authored → JSON normalised | Accepted | Comments in authoring; determinism in storage | — |
| ADR-009 | GEMM-only workload in v0.1 | Accepted | Depth before breadth | v0.1 DoD met |
| ADR-010 | Every kernel stage retained permanently | Accepted | Rule 5 baselines | Never |
| ADR-011 | CUDA Runtime API, not Driver API | Accepted | Sufficient; more readable | Explicit context/module control needed |
| ADR-012 | nvcc only; no Clang-CUDA in v0.1 | Accepted | Avoid uncontrolled codegen variable | Deliberate compiler-comparison experiment |
| ADR-013 | Median primary statistic + mandatory dispersion | Accepted | Skewed bounded-below distributions | — |
| ADR-014 | Bootstrap (BCa) CIs, not parametric | Accepted | Non-normal distributions | — |
| ADR-015 | Profiled runs structurally excluded from perf data | Accepted | Profiler perturbs execution | Zero-overhead profiling exists |
| ADR-016 | Repetitions are separate processes | Accepted | Captures process-level variance | — |
| ADR-017 | Outliers flagged, never deleted | Accepted | P4 | Never |
| ADR-018 | Tolerance from `C·eps·sqrt(K)`, never hand-tuned to pass | Accepted | Prevents hiding bugs | Numerical justification in review |
| ADR-019 | HIP interface-only in v0.1 | Accepted | No hardware; no false claims | AMD hardware acquired |
| ADR-020 | TPU as out-of-process adapter | Accepted | Genuinely different execution model | — |
| ADR-021 | GitHub Actions; self-hosted GPU runner | Accepted | Repo already on GitHub | GPU runner unavailable |
| ADR-022 | No performance gating in CI | Accepted | Shared runners fail PHOENIX validity criteria | Dedicated, controlled CI hardware |
| ADR-023 | Container pins user space only; host driver recorded separately | Accepted | Honest reproducibility claim | — |
| ADR-024 | Energy gated behind power-telemetry characterisation | Accepted | Uncharacterised energy is the most dangerous possible v0.1 output | Characterisation complete |
| ADR-025 | Backend interface designed against the TPU case, not the CUDA case | Accepted | CUDA-shaped abstractions break on other accelerators; the reverse holds | — |
| ADR-026 | One YAML file per paper; claims carry `scope` + `excludes` | Accepted | Enables honest cross-paper energy comparison later | — |
| ADR-027 | `prepare()` separated from `execute_once()` | Accepted | Makes compile cost first-class and visible | — |
| ADR-028 | Schema back-compat replay gate in CI | Accepted | Mechanically enforces Rule 6 | — |
| ADR-029 | CUTLASS excluded from v0.1 | Accepted | Understanding before borrowing | v0.2 comparison study |
| ADR-030 | A-vs-A control mandatory before any A-vs-B claim | Accepted | Establishes the noise floor a claim must exceed | Never |
| ADR-031 | **WSL2 / Ubuntu 26.04 is the canonical development platform**; Windows-native (MSVC) is not supported | Accepted | The entire §5 stack, §22 containers, and §21 CI are Linux-native. Ubuntu and Docker Desktop are already installed; the Windows side has no C++ toolchain at all. Windows-native would require rewriting §5, §22, and §30 for no scientific gain. | A native Linux workstation becomes the primary development machine, at which point WSL2 becomes one supported option rather than the canonical one |
| ADR-032 | **The development host is not a measurement host.** Runs from it are `PROVISIONAL` and unpublishable | **Accepted — blocking** | Hybrid P/E cores (R-19), WSL2 virtualisation (R-20), and a laptop thermal envelope (R-21) each independently violate §32's validity criteria. Executing EXP-001 here validates the pipeline; it does not measure a CPU. | A controlled measurement host is provisioned and characterised |
| ADR-033 | Project interpreter pinned by `uv` independently of the system Python | Accepted | System Python in WSL2 is 3.14.4, above §5's tested range; pinning decouples the project from distro drift | §5's tested range is extended after validation |
| ADR-034 | GEMM correctness scales error by `(\|A\|·\|B\|)`, not `\|C\|`; `C` unchanged | Accepted | Dividing by `\|C\|` measures cancellation in the input data, not implementation error; found by EXP-001's first execution | Mixed-precision accumulate paths (EXP-006) |

---

## 35. v0.2 → v1.0 Evolution

| Version | Theme | Adds | Depends on which v0.1 interface staying stable |
|---|---|---|---|
| **v0.2** | Multi-vendor execution | AMD execution, TPU execution, multi-GPU, CUTLASS comparison, CUPTI | `Backend` concept; result schema; comparability policy |
| **v0.3** | Heterogeneous runtime | Multi-backend execution in one run; device selection; transfer accounting | `Backend`; run record; timing taxonomy |
| **v0.4** | Workload characterisation | CNN, transformer, attention, LLM inference; operator-level measurement; workload feature extraction | Workload schema generalised beyond GEMM |
| **v0.5** | Compiler / mapping | Graph IR; cost model **trained on v0.1–v0.4 measured data**; operator→backend mapping | Derived-metric schema; measurement corpus |
| **v0.6** | Photonic simulation | MZI mesh simulator; optical loss, phase error, thermal/wavelength drift; ADC/DAC; **full-system energy model** | Provenance (`SIMULATED` vs `MEASURED`); energy pipeline; break-even methodology from EXP-008 |
| **v0.7** | Photonic hardware | Physical device integration where available | `Backend` concept; reproducibility package |
| **v0.8** | Neuromorphic | SNN, event-driven workloads, sparse computation | Workload schema; metric schema (events/joule) |
| **v1.0** | Heterogeneous platform | Unified runtime answering the EDP-under-accuracy-constraint question | All of the above |

### Interfaces that must remain stable

These are the load-bearing contracts. Breaking any of them invalidates historical comparability:

1. **`Backend` concept** — every future accelerator, including photonic, satisfies it or the abstraction failed.
2. **Provenance value type** — the moment `SIMULATED` photonic energy must be compared against `MEASURED` GPU energy, this type is what keeps the comparison honest. It is the single most important thing v0.1 builds.
3. **Four-tier data model** — raw data recorded in 2026 must be re-analysable in 2030 under a revised formula.
4. **Result schema (with migrations)** — comparability across the project's whole lifetime.
5. **Reproducibility package format** — a v0.1 run must be reconstructible from a v1.0 codebase.
6. **Timing taxonomy** — `device_kernel_time` must mean the same thing on a photonic backend as on CUDA, or every cross-substrate latency comparison is meaningless.
7. **Experiment identity (ID + version + commit)** — the link from a published figure back to its raw iterations.

**The critical evolution risk:** v0.6 will compare `SIMULATED` photonic results against `MEASURED` GPU results. That comparison is only defensible because the provenance system makes the asymmetry explicit, visible in every figure, and impossible to accidentally erase. If v0.1's provenance system is weak, every photonic conclusion PHOENIX ever publishes is weak.

---

## 36. Final Architectural Decisions (summary)

1. **Languages:** C++20 (compute), Python 3.11 (control), CUDA C++ (kernels), nanobind (boundary).
2. **Boundary:** one JSON crossing per run; Python never in a measured region.
3. **Build:** CMake ≥ 3.24, uv, pinned dependencies by SHA/digest.
4. **Data:** Parquet raw (immutable, checksummed) + DuckDB catalogue (rebuildable) + YAML/JSON contracts.
5. **Provenance:** mandatory machine-readable class on every numeric value, schema-enforced.
6. **Tiers:** raw → aggregate → derived → reported; every figure traceable to individual iterations.
7. **Workload:** GEMM only; eight implementation stages, all retained permanently.
8. **Correctness:** FP64 CPU reference ground truth; `C·eps·sqrt(K)` tolerance; adversarial structured inputs; tolerances never loosened to pass.
9. **Measurement:** device events + host wall; repetitions as separate processes; timer calibration and measurement floor enforced.
10. **Statistics:** median primary, dispersion mandatory, BCa bootstrap CIs, A-vs-A control before any comparison claim.
11. **Profiling:** Nsight Systems then Nsight Compute; profiled timings structurally excluded from performance data.
12. **Vendors:** CUDA executes; HIP and TPU are interfaces only, and this is stated plainly rather than implied away.
13. **CI:** GitHub Actions; CPU everything per PR; CUDA compile-only per PR; GPU correctness nightly; performance never gates.
14. **Energy:** gated behind power-telemetry characterisation. Uncharacterised energy does not ship.
15. **Non-goals:** no photonics, no compiler, no scheduler, no cross-vendor claim in v0.1.

---

## 37. Unresolved / Requires Hardware Access

Genuinely undecidable without hardware, credentials, or vendor documentation. Everything else in this document is decided.

1. **Which specific NVIDIA GPU(s) will PHOENIX run on?** Determines compute capability, available precisions (FP8 availability in particular), the target architecture list, and which vendor hardware records must be authored first. **Blocks:** hardware DB priority, `CMAKE_CUDA_ARCHITECTURES`, EXP-006 dtype coverage.
2. **NVML power-telemetry semantics on the target GPU** — actual sampling interval, averaging window, and what the sensor physically includes (board vs die vs board-plus-memory). Requires characterisation against an external meter. **Blocks:** every energy figure (ADR-024).
3. **Is an external power meter available?** If not, energy remains `NOT YET MEASURED` in v0.1. This is an acceptable outcome and must not be worked around with estimates.
4. **Self-hosted GPU CI runner availability and administration.** **Blocks:** the entire nightly GPU tier; without it, GPU correctness becomes a manual process and the schedule must be re-planned.
5. **Can GPU clocks be locked and exclusive compute mode set?** Requires elevated privileges on the GPU host. Determines whether clock stability is controlled or merely observed — which changes the achievable variance and therefore the comparison methodology.
6. **AMD hardware access — will it exist at all?** **Blocks:** EXP-009 execution and all of v0.2's AMD scope.
7. **TPU access — Cloud TPU credentials, generation, slice size, budget.** **Blocks:** EXP-010 execution.
8. **Current stable CUDA toolkit and driver versions ⚠** — must be verified against NVIDIA's current release at implementation time rather than assumed from this document.
9. **Thermal and electrical environment of the GPU host** — ambient temperature stability and whether cooling is adequate for sustained runs. Directly determines whether long sweeps are valid or throttling-dominated.
10. **Do institutional constraints exist on publishing benchmark results** for specific vendor hardware (some vendor EULAs restrict benchmark publication). **Blocks:** publication planning, not implementation.
11. **What is WSL2's effect on host-side timing fidelity?** Clock source, scheduling, and memory behaviour interpose a VM layer of `UNKNOWN` magnitude between PHOENIX's timers and the hardware. Requires a side-by-side characterisation against a native Linux host. **Blocks:** any publishable measurement taken under WSL2 (ADR-032). Does not block development.
12. **Will a controlled CPU measurement host exist?** Distinct from item 1, which concerns the GPU. A publishable CPU baseline (EXP-001) needs a machine with homogeneous cores, a stable thermal envelope, controllable frequency scaling, and no hypervisor. **Blocks:** publication of every CPU result; **does not block** Weeks 1–4, which only need the pipeline to run.
13. **Can CPU affinity and frequency scaling be controlled on whatever measurement host is chosen?** Determines whether R-19's hybrid-core artefact is eliminated or merely observed and recorded.

---

## 38. Implementation Gate

Per §47 of the master prompt, this document ends at the gate.

**Delivered:** final technology stack (§5), architecture diagram (§6), repository tree (§8), all schemas (§9–§11), the first-10-experiment matrix (§27), the ADR register (§34), the 12-week plan (§31), and the genuinely unresolved items (§37).

**No implementation code has been written.**

Implementation begins only on an explicit instruction of the form:

> **"Start PHOENIX v0.1 implementation."**

Recommended first instruction after approval, given that the development environment has no GPU: **Weeks 1–4**, which are entirely GPU-free and deliver a working end-to-end measurement pipeline on CPU — a complete, demonstrable, scientifically valid vertical slice that de-risks every later GPU week.
