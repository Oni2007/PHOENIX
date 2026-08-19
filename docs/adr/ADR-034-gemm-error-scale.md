# ADR-034 — The GEMM correctness gate scales error by `(|A|·|B|)`, not by `|C|`

**Status:** Accepted
**Date:** 2026-08-19
**Supersedes part of:** TDD §14.2
**Found by:** the first execution of EXP-001

## Context

TDD §14.2 fixes the tolerance as `C · eps(dtype) · sqrt(K)` but did not state what
that relative tolerance is *relative to*. The initial implementation divided the
absolute error by `|C_ij|` — the magnitude of the output element.

The first run of EXP-001 produced a clean, unambiguous signal:

| dtype | runs | correctness | observed `max_rel_err` / tolerance |
|---|---:|---|---:|
| fp64 | 36 | all pass, error exactly `0.0` | 0× |
| fp32 | 36 | **all fail** | 54× – 2176× |

Three things ruled out a kernel bug:

1. **All three fp32 implementations produced byte-identical output** at every size
   (`max_rel_err` agreed to four significant figures). Three different loop
   structures — naive, blocked, and OpenMP-parallel — do not share a bug.
2. **Every fp64 run was bit-exact** against the reference. A memory or indexing
   fault would not be precision-selective.
3. **The failure ratio was non-monotonic in K** (54 → 1968 → 2176 → 612), which is
   the signature of an unlucky denominator, not of accumulated rounding error.

The fault was in the comparator. Dividing by `|C_ij|` measures **catastrophic
cancellation**, which is a property of the *input data*: for random inputs, a
handful of the 262 144 output elements land near zero, and any absolute error at
those elements produces an enormous relative error. That says nothing about the
implementation.

## Decision

The correctness gate scales the error by the **sum of absolute products**:

```text
|Ĉ - C|_ij  ≤  C · eps(dtype) · sqrt(K) · (|A|·|B|)_ij
```

`(|A|·|B|)_ij` is the standard backward-error denominator for a dot product. It is
computed in the same pass as the FP64 reference, so correctness costs one extra
GEMM rather than two.

Cancellation is **still reported**, as `max_cancellation` (`err / |ref|`), so the
property remains visible in the data. It is simply not the pass/fail criterion.

## Consequence

After the change, all 72 EXP-001 runs pass, and fp32 error lands at **≈ 2.4e-7 ≈
2·eps(fp32)**, nearly flat in K — which is exactly what correct fp32 accumulation
should produce.

## What was explicitly NOT done

**The tolerance constant `C` was not raised.** `C` remains 8.0. TDD §14.2 states
that a tolerance is never loosened to make a failing kernel pass, and that a
failure arising only through the tolerance model must be *investigated*, not
accommodated. The investigation found the model wrong, not the constant too tight.
Raising `C` to ~2200 would have made the same tests pass while destroying the
gate's ability to detect a real fault.

## Known residual

With the `(|A|·|B|)` denominator, the `sqrt(K)` term is now **conservative by about
90×** at the sizes tested (tolerance 2.2e-5 vs observed 2.4e-7). The gate would
still catch an error two orders of magnitude above fp32 rounding, so it is not
toothless — but it is looser than it needs to be.

Tightening `C`, or replacing `sqrt(K)` with a constant under this denominator, is a
decision that should be made **from measured data across more sizes and dtypes**,
not guessed now. Recorded rather than silently left.

## Revisit trigger

Tensor Core / mixed-precision paths arrive in week 7 (EXP-006), where the
accumulate dtype differs from the input dtype and the error model needs revisiting
regardless.
