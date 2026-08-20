# H100 SXM5 80GB — Analytical Roofline

> **PURELY THEORETICAL. This device has never been executed on by PHOENIX.**
>
> No NVIDIA GPU exists in this project's environment (verified: no `nvidia-smi`, no `nvcc`, no discrete GPU — see ADR-032 and TDD §37 item 1). Every figure in this report is `VENDOR_REPORTED` (from NVIDIA's own published specifications, cross-checked across independent secondary sources — see the hardware record's own header for exactly which ones and their confidence) or `DERIVED`/`ANALYTICAL` from those specifications by the roofline model (TDD §33). **Nothing in this document is a measurement, a benchmark result, or evidence that PHOENIX runs on this or any GPU.**

Generated from `hardware/gpu/nvidia-h100-sxm5-80gb.yaml` by `phoenix.analysis.roofline_report` — every number below is computed here, not hand-typed.

## Peak compute roofs (VENDOR_REPORTED)

| Precision | Peak | Conditions | Confidence |
|---|---:|---|---|
| FP64 | 34 TFLOP/s | non-Tensor-Core CUDA-core FP64 path, boost clock, dense (structured sparsity is not applicable to this path) | HIGH |
| FP64_TENSOR_CORE | 67 TFLOP/s | 4th-gen Tensor Core FP64 matrix path, boost clock, dense (NVIDIA does not market a structured-sparsity variant for FP64 Tensor Core) | HIGH |
| FP32 | 67 TFLOP/s | non-Tensor-Core CUDA-core FP32 path, boost clock | HIGH |
| TF32_TENSOR_CORE_DENSE | 495 TFLOP/s | 4th-gen Tensor Core TF32 path, dense (NO structured sparsity), boost clock | MEDIUM |
| TF32_TENSOR_CORE_SPARSE | 989 TFLOP/s | 4th-gen Tensor Core TF32 path, WITH 2:4 structured sparsity (~2x the dense figure), boost clock | MEDIUM |
| BF16_TENSOR_CORE_DENSE | 990 TFLOP/s | 4th-gen Tensor Core BF16 path, dense (no sparsity), boost clock | HIGH |
| BF16_TENSOR_CORE_SPARSE | 1979 TFLOP/s | 4th-gen Tensor Core BF16 path, WITH 2:4 structured sparsity (2x dense), boost clock | MEDIUM |
| FP16_TENSOR_CORE_DENSE | 990 TFLOP/s | 4th-gen Tensor Core FP16 path, dense (no sparsity), boost clock | HIGH |
| FP16_TENSOR_CORE_SPARSE | 1979 TFLOP/s | 4th-gen Tensor Core FP16 path, WITH 2:4 structured sparsity (2x dense), boost clock | MEDIUM |
| FP8_TENSOR_CORE_DENSE | 1979 TFLOP/s | 4th-gen Tensor Core FP8 path (E4M3/E5M2), dense (no sparsity), boost clock | MEDIUM |
| FP8_TENSOR_CORE_SPARSE | 3958 TFLOP/s | 4th-gen Tensor Core FP8 path, WITH 2:4 structured sparsity (2x dense), boost clock | MEDIUM |
| INT8_TENSOR_CORE_DENSE | 1979 TOPS | 4th-gen Tensor Core INT8 path, dense (no sparsity), boost clock | HIGH |
| INT8_TENSOR_CORE_SPARSE | 3958 TOPS | 4th-gen Tensor Core INT8 path, WITH 2:4 structured sparsity (2x dense), boost clock | MEDIUM |

**Device memory bandwidth (theoretical peak, VENDOR_REPORTED):** 3.35 TB/s — theoretical peak HBM3 bandwidth, SXM5 module

**Empirical bandwidth ceiling (MEASURED):** `NOT YET MEASURED` — no hardware, no microbenchmark, and none is claimed.

## FP64

Roofline crossover: **10.15 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 10.67 | compute-bound | 34,000.0 |
| 256 | 21.33 | compute-bound | 34,000.0 |
| 512 | 42.67 | compute-bound | 34,000.0 |
| 1024 | 85.33 | compute-bound | 34,000.0 |
| 2048 | 170.67 | compute-bound | 34,000.0 |
| 4096 | 341.33 | compute-bound | 34,000.0 |
| 8192 | 682.67 | compute-bound | 34,000.0 |
| 16384 | 1365.33 | compute-bound | 34,000.0 |

## FP64_TENSOR_CORE

Roofline crossover: **20.00 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 10.67 | memory-bound | 35,733.3 |
| 256 | 21.33 | compute-bound | 67,000.0 |
| 512 | 42.67 | compute-bound | 67,000.0 |
| 1024 | 85.33 | compute-bound | 67,000.0 |
| 2048 | 170.67 | compute-bound | 67,000.0 |
| 4096 | 341.33 | compute-bound | 67,000.0 |
| 8192 | 682.67 | compute-bound | 67,000.0 |
| 16384 | 1365.33 | compute-bound | 67,000.0 |

## FP32

Roofline crossover: **20.00 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 21.33 | compute-bound | 67,000.0 |
| 256 | 42.67 | compute-bound | 67,000.0 |
| 512 | 85.33 | compute-bound | 67,000.0 |
| 1024 | 170.67 | compute-bound | 67,000.0 |
| 2048 | 341.33 | compute-bound | 67,000.0 |
| 4096 | 682.67 | compute-bound | 67,000.0 |
| 8192 | 1365.33 | compute-bound | 67,000.0 |
| 16384 | 2730.67 | compute-bound | 67,000.0 |

## TF32_TENSOR_CORE_DENSE

Roofline crossover: **147.76 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 21.33 | memory-bound | 71,466.7 |
| 256 | 42.67 | memory-bound | 142,933.3 |
| 512 | 85.33 | memory-bound | 285,866.7 |
| 1024 | 170.67 | compute-bound | 495,000.0 |
| 2048 | 341.33 | compute-bound | 495,000.0 |
| 4096 | 682.67 | compute-bound | 495,000.0 |
| 8192 | 1365.33 | compute-bound | 495,000.0 |
| 16384 | 2730.67 | compute-bound | 495,000.0 |

## TF32_TENSOR_CORE_SPARSE

Roofline crossover: **295.22 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 21.33 | memory-bound | 71,466.7 |
| 256 | 42.67 | memory-bound | 142,933.3 |
| 512 | 85.33 | memory-bound | 285,866.7 |
| 1024 | 170.67 | memory-bound | 571,733.3 |
| 2048 | 341.33 | compute-bound | 989,000.0 |
| 4096 | 682.67 | compute-bound | 989,000.0 |
| 8192 | 1365.33 | compute-bound | 989,000.0 |
| 16384 | 2730.67 | compute-bound | 989,000.0 |

## BF16_TENSOR_CORE_DENSE

Roofline crossover: **295.52 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 42.67 | memory-bound | 142,933.3 |
| 256 | 85.33 | memory-bound | 285,866.7 |
| 512 | 170.67 | memory-bound | 571,733.3 |
| 1024 | 341.33 | compute-bound | 990,000.0 |
| 2048 | 682.67 | compute-bound | 990,000.0 |
| 4096 | 1365.33 | compute-bound | 990,000.0 |
| 8192 | 2730.67 | compute-bound | 990,000.0 |
| 16384 | 5461.33 | compute-bound | 990,000.0 |

## BF16_TENSOR_CORE_SPARSE

Roofline crossover: **590.75 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 42.67 | memory-bound | 142,933.3 |
| 256 | 85.33 | memory-bound | 285,866.7 |
| 512 | 170.67 | memory-bound | 571,733.3 |
| 1024 | 341.33 | memory-bound | 1,143,466.7 |
| 2048 | 682.67 | compute-bound | 1,979,000.0 |
| 4096 | 1365.33 | compute-bound | 1,979,000.0 |
| 8192 | 2730.67 | compute-bound | 1,979,000.0 |
| 16384 | 5461.33 | compute-bound | 1,979,000.0 |

## FP16_TENSOR_CORE_DENSE

Roofline crossover: **295.52 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 42.67 | memory-bound | 142,933.3 |
| 256 | 85.33 | memory-bound | 285,866.7 |
| 512 | 170.67 | memory-bound | 571,733.3 |
| 1024 | 341.33 | compute-bound | 990,000.0 |
| 2048 | 682.67 | compute-bound | 990,000.0 |
| 4096 | 1365.33 | compute-bound | 990,000.0 |
| 8192 | 2730.67 | compute-bound | 990,000.0 |
| 16384 | 5461.33 | compute-bound | 990,000.0 |

## FP16_TENSOR_CORE_SPARSE

Roofline crossover: **590.75 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 42.67 | memory-bound | 142,933.3 |
| 256 | 85.33 | memory-bound | 285,866.7 |
| 512 | 170.67 | memory-bound | 571,733.3 |
| 1024 | 341.33 | memory-bound | 1,143,466.7 |
| 2048 | 682.67 | compute-bound | 1,979,000.0 |
| 4096 | 1365.33 | compute-bound | 1,979,000.0 |
| 8192 | 2730.67 | compute-bound | 1,979,000.0 |
| 16384 | 5461.33 | compute-bound | 1,979,000.0 |

## FP8_TENSOR_CORE_DENSE

Roofline crossover: **590.75 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 85.33 | memory-bound | 285,866.7 |
| 256 | 170.67 | memory-bound | 571,733.3 |
| 512 | 341.33 | memory-bound | 1,143,466.7 |
| 1024 | 682.67 | compute-bound | 1,979,000.0 |
| 2048 | 1365.33 | compute-bound | 1,979,000.0 |
| 4096 | 2730.67 | compute-bound | 1,979,000.0 |
| 8192 | 5461.33 | compute-bound | 1,979,000.0 |
| 16384 | 10922.67 | compute-bound | 1,979,000.0 |

## FP8_TENSOR_CORE_SPARSE

Roofline crossover: **1181.49 FLOP/byte** — GEMMs below this arithmetic intensity are memory-bound under the no-reuse traffic model; above it, compute-bound.

| M=N=K | Arithmetic intensity (FLOP/B) | Regime | Attainable (GFLOP/s) |
|---:|---:|---|---:|
| 128 | 85.33 | memory-bound | 285,866.7 |
| 256 | 170.67 | memory-bound | 571,733.3 |
| 512 | 341.33 | memory-bound | 1,143,466.7 |
| 1024 | 682.67 | memory-bound | 2,286,933.3 |
| 2048 | 1365.33 | compute-bound | 3,958,000.0 |
| 4096 | 2730.67 | compute-bound | 3,958,000.0 |
| 8192 | 5461.33 | compute-bound | 3,958,000.0 |
| 16384 | 10922.67 | compute-bound | 3,958,000.0 |

## Assumptions (apply to every table above)

- Memory traffic uses the **no-reuse model**: every element of A, B, and C is charged exactly once against device memory, crediting zero cache reuse. This is a lower bound on real traffic and therefore an *upper* bound on true arithmetic intensity — a real kernel's AI, if it ever runs, will be lower than shown here for small sizes where cache matters, converging toward these figures only at sizes that exceed cache regardless.
- A fused multiply-add counts as 2 FLOPs.
- Peak figures assume boost clock and the sparsity condition named in each precision's own row above; see the hardware record for full source citations and confidence levels.
- Ignores kernel launch overhead, host-device transfer, and anything outside the GEMM itself.

