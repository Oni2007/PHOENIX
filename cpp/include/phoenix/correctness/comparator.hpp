#pragma once
#include <span>
#include <string>

#include "phoenix/schema/request.hpp"

namespace phoenix {

// Tolerance is DERIVED from the dtype's machine epsilon and the reduction length,
// never hand-picked (TDD 14.2):
//
//     tolerance_rel(dtype, K) = C * eps(dtype) * sqrt(K)
//
// sqrt(K) rather than K reflects random-walk error accumulation over the reduction
// for randomly-signed rounding errors. THAT IS AN ASSUMPTION AND IS RECORDED AS ONE.
double tolerance_rel(DType dtype, std::int64_t K, double C);

struct ComparisonResult {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;     // scaled by the error-scale below, not by |ref|
    double max_cancellation = 0.0;  // err / |ref|: reported, never used as the gate
    bool passed = false;
    double tolerance_rel = 0.0;
    std::string rule;
};

// The scale-aware comparison. `error_scale` is the magnitude against which a GEMM's
// rounding error must be judged: the standard backward-error bound is
//
//     |C_hat - C|_ij  <=  gamma_K * (|A| * |B|)_ij
//
// so (|A|*|B|)_ij -- NOT |C_ij| -- is the correct denominator. Dividing by |C_ij|
// measures catastrophic cancellation at output elements that happen to land near
// zero, which is a property of the INPUT DATA, not of the implementation. Doing so
// makes a correct fp32 kernel look broken by two to three orders of magnitude.
//
// If `error_scale` is empty the comparator falls back to |ref| with an absolute
// floor -- adequate for backends whose output is not a reduction.
ComparisonResult compare(std::span<const double> candidate, std::span<const double> reference,
                         std::span<const double> error_scale, DType dtype, std::int64_t K,
                         double C);

// Convenience overload: no scale available.
ComparisonResult compare(std::span<const double> candidate, std::span<const double> reference,
                         DType dtype, std::int64_t K, double C);

}  // namespace phoenix
