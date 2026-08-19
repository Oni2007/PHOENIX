# FIND-001 — fp32 GEMM correctness failure was a comparator defect, not a kernel defect

**Type:** Finding (methodology)
**Date:** 2026-08-19
**Experiment:** EXP-001, first execution
**Status:** Resolved — see [ADR-034](../../docs/adr/ADR-034-gemm-error-scale.md)

## Observation

36 of 72 EXP-001 runs returned `CORRECTNESS_FAILED`. The failures partitioned
perfectly by dtype: every fp32 run failed, every fp64 run passed with error
exactly zero.

## Interpretation

Separated from the observation deliberately.

The uniform partition by dtype, the byte-identical agreement of three
independently-structured fp32 implementations, and the non-monotonic failure ratio
in K together indicate a defect in the *error model*, not in any kernel. The
comparator divided by `|C_ij|`, and so measured catastrophic cancellation in the
input data rather than implementation error.

## Ruled out, and how

| Alternative explanation | How it was ruled out |
|---|---|
| A bug in one fp32 kernel | All three implementations produced byte-identical output |
| An indexing/boundary fault | fp64 passes bit-exactly on the same shapes; the adversarial index-encoded input tests pass |
| Genuine fp32 unsuitability at these K | Post-fix error is ≈2·eps(fp32), flat in K — textbook behaviour |
| Tolerance constant too tight | The failure ratio reached 2176×; no defensible `C` covers that, and raising `C` would disable the gate |

## Speculation

Kept separate from both of the above, and untested.

The same defect would likely have gone unnoticed until week 7, when FP16/BF16 with
FP32 accumulation makes cancellation effects larger still. Finding it at the CPU
reference stage — the cheapest possible place — is an argument for the TDD's
insistence that the reference ladder be built before the interesting hardware.

## Follow-up

- Tighten the tolerance model under the new denominator once multi-size, multi-dtype
  data exists (ADR-034, "known residual").
- Revisit for mixed-precision accumulate paths in EXP-006.
