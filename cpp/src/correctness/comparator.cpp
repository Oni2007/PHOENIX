#include "phoenix/correctness/comparator.hpp"

#include <cmath>
#include <limits>

#include "phoenix/core/error.hpp"

namespace phoenix {

double tolerance_rel(DType dtype, std::int64_t K, double C) {
    return C * dtype_epsilon(dtype) * std::sqrt(static_cast<double>(K));
}

ComparisonResult compare(std::span<const double> candidate, std::span<const double> reference,
                         std::span<const double> error_scale, DType dtype, std::int64_t K,
                         double C) {
    if (candidate.size() != reference.size()) {
        throw ExecutionError("comparator: size mismatch between candidate and reference");
    }
    if (!error_scale.empty() && error_scale.size() != reference.size()) {
        throw ExecutionError("comparator: error_scale size does not match reference");
    }

    ComparisonResult r;
    r.tolerance_rel = tolerance_rel(dtype, K, C);
    r.rule = error_scale.empty() ? "C*eps(dtype)*sqrt(K), scaled by |ref| with absolute floor"
                                 : "C*eps(dtype)*sqrt(K), scaled by (|A|*|B|)_ij";

    const double tiny = std::numeric_limits<double>::min();

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const double c = candidate[i];
        const double ref = reference[i];
        if (std::isnan(c) || std::isnan(ref)) {
            r.max_abs_err = std::numeric_limits<double>::infinity();
            r.max_rel_err = std::numeric_limits<double>::infinity();
            r.passed = false;
            return r;
        }
        const double abs_err = std::fabs(c - ref);
        r.max_abs_err = std::max(r.max_abs_err, abs_err);

        const double denom = error_scale.empty()
                                 ? std::max(std::fabs(ref), r.tolerance_rel)
                                 : std::max(error_scale[i], tiny);
        r.max_rel_err = std::max(r.max_rel_err, abs_err / denom);

        // Reported so that cancellation remains visible as a property of the data.
        // It is NOT the pass/fail criterion.
        const double cancel_denom = std::max(std::fabs(ref), tiny);
        r.max_cancellation = std::max(r.max_cancellation, abs_err / cancel_denom);
    }
    r.passed = r.max_rel_err <= r.tolerance_rel;
    return r;
}

ComparisonResult compare(std::span<const double> candidate, std::span<const double> reference,
                         DType dtype, std::int64_t K, double C) {
    return compare(candidate, reference, {}, dtype, K, C);
}

}  // namespace phoenix
