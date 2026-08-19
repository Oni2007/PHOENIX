#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "../backends/cpu/gemm_kernels.hpp"
#include "phoenix/correctness/comparator.hpp"

using namespace phoenix;
using namespace phoenix::cpu;

namespace {

struct Shape {
    std::int64_t M, N, K;
};

std::vector<double> random_matrix(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// Structured input where each element encodes its own index. Random-uniform inputs
// hide boundary bugs, because an off-by-one that reads a neighbouring element of a
// random matrix produces a plausible number. This makes index errors visible.
std::vector<double> index_encoded(std::int64_t rows, std::int64_t cols) {
    std::vector<double> v(static_cast<std::size_t>(rows * cols));
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < cols; ++j) {
            v[static_cast<std::size_t>(i * cols + j)] =
                static_cast<double>(i + 1) + static_cast<double>(j + 1) / 1024.0;
        }
    }
    return v;
}

}  // namespace

class GemmShapes : public ::testing::TestWithParam<Shape> {};

TEST_P(GemmShapes, NaiveMatchesReference) {
    const auto s = GetParam();
    auto A = random_matrix(static_cast<std::size_t>(s.M * s.K), 11);
    auto B = random_matrix(static_cast<std::size_t>(s.K * s.N), 22);
    std::vector<double> ref(static_cast<std::size_t>(s.M * s.N), 0.0);
    std::vector<double> got(static_cast<std::size_t>(s.M * s.N), 0.0);

    gemm_reference_fp64(s.M, s.N, s.K, 1.0, A, B, 0.0, ref);
    gemm_naive<double>(s.M, s.N, s.K, 1.0, A, B, 0.0, got);

    const auto r = compare(got, ref, DType::FP64, s.K, 8.0);
    EXPECT_TRUE(r.passed) << "max_rel_err=" << r.max_rel_err << " tol=" << r.tolerance_rel;
}

TEST_P(GemmShapes, BlockedMatchesReference) {
    const auto s = GetParam();
    auto A = random_matrix(static_cast<std::size_t>(s.M * s.K), 33);
    auto B = random_matrix(static_cast<std::size_t>(s.K * s.N), 44);
    std::vector<double> ref(static_cast<std::size_t>(s.M * s.N), 0.0);
    std::vector<double> got(static_cast<std::size_t>(s.M * s.N), 0.0);

    gemm_reference_fp64(s.M, s.N, s.K, 1.0, A, B, 0.0, ref);
    gemm_blocked<double>(s.M, s.N, s.K, 1.0, A, B, 0.0, got);

    const auto r = compare(got, ref, DType::FP64, s.K, 8.0);
    EXPECT_TRUE(r.passed) << "max_rel_err=" << r.max_rel_err << " tol=" << r.tolerance_rel;
}

TEST_P(GemmShapes, OmpMatchesReference) {
    const auto s = GetParam();
    auto A = random_matrix(static_cast<std::size_t>(s.M * s.K), 55);
    auto B = random_matrix(static_cast<std::size_t>(s.K * s.N), 66);
    std::vector<double> ref(static_cast<std::size_t>(s.M * s.N), 0.0);
    std::vector<double> got(static_cast<std::size_t>(s.M * s.N), 0.0);

    gemm_reference_fp64(s.M, s.N, s.K, 1.0, A, B, 0.0, ref);
    gemm_omp<double>(s.M, s.N, s.K, 1.0, A, B, 0.0, got);

    const auto r = compare(got, ref, DType::FP64, s.K, 8.0);
    EXPECT_TRUE(r.passed) << "max_rel_err=" << r.max_rel_err << " tol=" << r.tolerance_rel;
}

TEST_P(GemmShapes, BlockedSurvivesAdversarialIndexEncodedInput) {
    const auto s = GetParam();
    auto A = index_encoded(s.M, s.K);
    auto B = index_encoded(s.K, s.N);
    std::vector<double> ref(static_cast<std::size_t>(s.M * s.N), 0.0);
    std::vector<double> got(static_cast<std::size_t>(s.M * s.N), 0.0);

    gemm_reference_fp64(s.M, s.N, s.K, 1.0, A, B, 0.0, ref);
    gemm_blocked<double>(s.M, s.N, s.K, 1.0, A, B, 0.0, got);

    const auto r = compare(got, ref, DType::FP64, s.K, 8.0);
    EXPECT_TRUE(r.passed) << "max_rel_err=" << r.max_rel_err;
}

INSTANTIATE_TEST_SUITE_P(ShapeMatrix, GemmShapes,
                         ::testing::Values(
                             // square
                             Shape{1, 1, 1}, Shape{16, 16, 16}, Shape{64, 64, 64},
                             // rectangular
                             Shape{32, 8, 16}, Shape{8, 32, 16},
                             // tall-skinny and short-wide
                             Shape{64, 2, 128}, Shape{2, 64, 128},
                             // K >> M,N
                             Shape{4, 4, 257},
                             // odd / prime / off-by-one around the 64 block boundary
                             Shape{63, 65, 67}, Shape{65, 63, 61}, Shape{17, 19, 23},
                             // not a multiple of the block size
                             Shape{100, 100, 100}));

TEST(GemmAlphaBeta, ReferenceHonoursAlphaAndBeta) {
    const std::int64_t M = 8, N = 8, K = 8;
    auto A = random_matrix(static_cast<std::size_t>(M * K), 7);
    auto B = random_matrix(static_cast<std::size_t>(K * N), 8);

    std::vector<double> base(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_fp64(M, N, K, 1.0, A, B, 0.0, base);

    std::vector<double> scaled(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_fp64(M, N, K, 2.0, A, B, 0.0, scaled);

    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_NEAR(scaled[i], 2.0 * base[i], 1e-12);
    }
}

TEST(GemmDeterminism, ReferenceIsBitwiseReproducible) {
    // Bitwise reproducibility is REQUIRED of the CPU reference (TDD 14.4).
    const std::int64_t M = 24, N = 24, K = 24;
    auto A = random_matrix(static_cast<std::size_t>(M * K), 99);
    auto B = random_matrix(static_cast<std::size_t>(K * N), 98);
    std::vector<double> r1(static_cast<std::size_t>(M * N), 0.0);
    std::vector<double> r2(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_fp64(M, N, K, 1.0, A, B, 0.0, r1);
    gemm_reference_fp64(M, N, K, 1.0, A, B, 0.0, r2);
    for (std::size_t i = 0; i < r1.size(); ++i) {
        EXPECT_EQ(r1[i], r2[i]) << "reference must be bit-identical run to run";
    }
}

TEST(GemmPrecision, Fp32AccumulatesMoreErrorThanFp64AtLargeK) {
    // A finding about precision, not a bug: recorded as such.
    const std::int64_t M = 4, N = 4, K = 4096;
    auto A = random_matrix(static_cast<std::size_t>(M * K), 5);
    auto B = random_matrix(static_cast<std::size_t>(K * N), 6);
    std::vector<float> A32(A.size()), B32(B.size());
    for (std::size_t i = 0; i < A.size(); ++i) A32[i] = static_cast<float>(A[i]);
    for (std::size_t i = 0; i < B.size(); ++i) B32[i] = static_cast<float>(B[i]);

    std::vector<double> ref(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_fp64(M, N, K, 1.0, A, B, 0.0, ref);

    std::vector<float> c32(static_cast<std::size_t>(M * N), 0.0F);
    gemm_naive<float>(M, N, K, 1.0F, A32, B32, 0.0F, c32);
    std::vector<double> got(c32.begin(), c32.end());

    const auto r32 = compare(got, ref, DType::FP32, K, 8.0);
    EXPECT_TRUE(r32.passed) << "fp32 must still pass at its OWN tolerance; max_rel_err="
                            << r32.max_rel_err << " tol=" << r32.tolerance_rel;

    const auto r64 = compare(got, ref, DType::FP64, K, 8.0);
    EXPECT_FALSE(r64.passed) << "fp32 output must NOT pass at fp64 tolerance";
}

TEST(GemmErrorScale, ScaleIsTheSumOfAbsoluteProducts) {
    // (|A|*|B|)_ij must be computed from magnitudes, so it is >= |C_ij| always.
    const std::int64_t M = 8, N = 8, K = 64;
    auto A = random_matrix(static_cast<std::size_t>(M * K), 1);
    auto B = random_matrix(static_cast<std::size_t>(K * N), 2);
    std::vector<double> C(static_cast<std::size_t>(M * N), 0.0);
    std::vector<double> scale(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_and_scale_fp64(M, N, K, 1.0, A, B, 0.0, C, scale);
    for (std::size_t i = 0; i < C.size(); ++i) {
        EXPECT_GE(scale[i], std::fabs(C[i]) - 1e-12)
            << "the error scale must dominate the result magnitude";
        EXPECT_GT(scale[i], 0.0);
    }
}

TEST(GemmErrorScale, ReferenceMatchesTheSinglePassVersion) {
    const std::int64_t M = 12, N = 10, K = 33;
    auto A = random_matrix(static_cast<std::size_t>(M * K), 3);
    auto B = random_matrix(static_cast<std::size_t>(K * N), 4);
    std::vector<double> c1(static_cast<std::size_t>(M * N), 0.0);
    std::vector<double> c2(static_cast<std::size_t>(M * N), 0.0);
    std::vector<double> scale(static_cast<std::size_t>(M * N), 0.0);
    gemm_reference_fp64(M, N, K, 1.0, A, B, 0.0, c1);
    gemm_reference_and_scale_fp64(M, N, K, 1.0, A, B, 0.0, c2, scale);
    for (std::size_t i = 0; i < c1.size(); ++i) EXPECT_EQ(c1[i], c2[i]);
}
