#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "phoenix/correctness/comparator.hpp"

using namespace phoenix;

TEST(Tolerance, ScalesWithSqrtOfReductionLength) {
    const double t1 = tolerance_rel(DType::FP32, 100, 8.0);
    const double t2 = tolerance_rel(DType::FP32, 400, 8.0);
    // 4x the reduction length must give 2x the tolerance, not 4x.
    EXPECT_NEAR(t2 / t1, 2.0, 1e-9);
}

TEST(Tolerance, Fp32IsLooserThanFp64) {
    EXPECT_GT(tolerance_rel(DType::FP32, 1024, 8.0), tolerance_rel(DType::FP64, 1024, 8.0));
}

TEST(Comparator, IdenticalBuffersPass) {
    std::vector<double> a{1.0, 2.0, 3.0};
    const auto r = compare(a, a, DType::FP64, 16, 8.0);
    EXPECT_TRUE(r.passed);
    EXPECT_EQ(r.max_abs_err, 0.0);
    EXPECT_NE(r.rule.find("C*eps(dtype)*sqrt(K)"), std::string::npos);
}

TEST(Comparator, GrossErrorFails) {
    std::vector<double> a{1.0, 2.0, 3.0};
    std::vector<double> b{1.0, 2.0, 3.5};
    const auto r = compare(a, b, DType::FP64, 16, 8.0);
    EXPECT_FALSE(r.passed);
    EXPECT_NEAR(r.max_abs_err, 0.5, 1e-12);
}

TEST(Comparator, NaNFailsRatherThanComparingEqual) {
    std::vector<double> a{std::nan("")};
    std::vector<double> b{1.0};
    const auto r = compare(a, b, DType::FP64, 16, 8.0);
    EXPECT_FALSE(r.passed);
}

TEST(Comparator, SizeMismatchThrows) {
    std::vector<double> a{1.0};
    std::vector<double> b{1.0, 2.0};
    EXPECT_THROW((void)compare(a, b, DType::FP64, 4, 8.0), std::exception);
}

// ── Regression tests for the error-model fix ─────────────────────────────────
//
// Found by EXP-001: every fp32 run failed at 54x-2176x tolerance while every fp64
// run passed exactly, and all three fp32 implementations produced byte-identical
// output. Three independent loop structures do not share a bug, so the fault was
// in the comparator, not the kernels: it divided by |C_ij|, which measures
// catastrophic cancellation in the INPUT DATA rather than implementation error.

TEST(Comparator, NearZeroReferenceDoesNotFailAScaleAwareComparison) {
    // A tiny true value produced by cancellation of large intermediates. The
    // absolute error is entirely consistent with fp32; only the |ref|-relative
    // view makes it look catastrophic.
    std::vector<double> ref{1.0e-7};
    std::vector<double> got{1.6e-7};
    std::vector<double> scale{100.0};  // (|A|*|B|)_ij: the intermediates were large

    const auto unscaled = compare(got, ref, DType::FP32, 512, 8.0);
    EXPECT_FALSE(unscaled.passed) << "dividing by |ref| flags a correct result";

    const auto scaled = compare(got, ref, scale, DType::FP32, 512, 8.0);
    EXPECT_TRUE(scaled.passed) << "scaled by (|A|*|B|) the same result is correct";
}

TEST(Comparator, CancellationIsStillReportedNotHidden) {
    std::vector<double> ref{1.0e-7};
    std::vector<double> got{1.6e-7};
    std::vector<double> scale{100.0};
    const auto r = compare(got, ref, scale, DType::FP32, 512, 8.0);
    EXPECT_TRUE(r.passed);
    // The cancellation is large and remains visible as a property of the data.
    EXPECT_GT(r.max_cancellation, 0.5);
}

TEST(Comparator, ScaleAwareComparisonStillCatchesARealError) {
    // The fix must not make the gate toothless: an error large relative to the
    // magnitude of the intermediates must still fail.
    std::vector<double> ref{1.0};
    std::vector<double> got{1.5};
    std::vector<double> scale{1.0};
    const auto r = compare(got, ref, scale, DType::FP32, 512, 8.0);
    EXPECT_FALSE(r.passed);
}

TEST(Comparator, RejectsMismatchedScaleLength) {
    std::vector<double> ref{1.0, 2.0};
    std::vector<double> got{1.0, 2.0};
    std::vector<double> scale{1.0};
    EXPECT_THROW((void)compare(got, ref, scale, DType::FP32, 4, 8.0), std::exception);
}
