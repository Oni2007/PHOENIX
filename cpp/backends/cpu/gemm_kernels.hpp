#pragma once
#include <cstdint>
#include <vector>

namespace phoenix::cpu {

// GEMM-0: the ground truth. A naive triple loop that accumulates in FP64
// REGARDLESS of the tested dtype (TDD 14.1). Computing the reference in double
// is what separates implementation error from precision-induced error -- two
// things that are constantly conflated in accelerator work.
void gemm_reference_fp64(std::int64_t M, std::int64_t N, std::int64_t K, double alpha,
                         const std::vector<double>& A, const std::vector<double>& B, double beta,
                         std::vector<double>& C);

// Computes the FP64 reference and, in the same pass, the error scale
// (|A|*|B|)_ij against which rounding error must be judged.
void gemm_reference_and_scale_fp64(std::int64_t M, std::int64_t N, std::int64_t K,
                                   double alpha, const std::vector<double>& A,
                                   const std::vector<double>& B, double beta,
                                   std::vector<double>& C, std::vector<double>& scale);

// Implementations under test. Each is retained permanently: earlier stages are the
// baselines that make later claims meaningful.
template <typename T>
void gemm_naive(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
                const std::vector<T>& B, T beta, std::vector<T>& C);

template <typename T>
void gemm_blocked(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
                  const std::vector<T>& B, T beta, std::vector<T>& C, std::int64_t block = 64);

template <typename T>
void gemm_omp(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
              const std::vector<T>& B, T beta, std::vector<T>& C, std::int64_t block = 64);

extern template void gemm_naive<double>(std::int64_t, std::int64_t, std::int64_t, double,
                                        const std::vector<double>&, const std::vector<double>&,
                                        double, std::vector<double>&);
extern template void gemm_naive<float>(std::int64_t, std::int64_t, std::int64_t, float,
                                       const std::vector<float>&, const std::vector<float>&, float,
                                       std::vector<float>&);
extern template void gemm_blocked<double>(std::int64_t, std::int64_t, std::int64_t, double,
                                          const std::vector<double>&, const std::vector<double>&,
                                          double, std::vector<double>&, std::int64_t);
extern template void gemm_blocked<float>(std::int64_t, std::int64_t, std::int64_t, float,
                                         const std::vector<float>&, const std::vector<float>&,
                                         float, std::vector<float>&, std::int64_t);
extern template void gemm_omp<double>(std::int64_t, std::int64_t, std::int64_t, double,
                                      const std::vector<double>&, const std::vector<double>&,
                                      double, std::vector<double>&, std::int64_t);
extern template void gemm_omp<float>(std::int64_t, std::int64_t, std::int64_t, float,
                                     const std::vector<float>&, const std::vector<float>&, float,
                                     std::vector<float>&, std::int64_t);

}  // namespace phoenix::cpu
