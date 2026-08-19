#include "gemm_kernels.hpp"

#include <algorithm>
#include <cmath>

namespace phoenix::cpu {

void gemm_reference_fp64(std::int64_t M, std::int64_t N, std::int64_t K, double alpha,
                         const std::vector<double>& A, const std::vector<double>& B, double beta,
                         std::vector<double>& C) {
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (std::int64_t k = 0; k < K; ++k) {
                acc += A[static_cast<std::size_t>(i * K + k)] *
                       B[static_cast<std::size_t>(k * N + j)];
            }
            const auto idx = static_cast<std::size_t>(i * N + j);
            C[idx] = alpha * acc + beta * C[idx];
        }
    }
}

void gemm_reference_and_scale_fp64(std::int64_t M, std::int64_t N, std::int64_t K,
                                   double alpha, const std::vector<double>& A,
                                   const std::vector<double>& B, double beta,
                                   std::vector<double>& C, std::vector<double>& scale) {
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t j = 0; j < N; ++j) {
            double acc = 0.0;
            double abs_acc = 0.0;
            for (std::int64_t k = 0; k < K; ++k) {
                const double a = A[static_cast<std::size_t>(i * K + k)];
                const double b = B[static_cast<std::size_t>(k * N + j)];
                acc += a * b;
                abs_acc += std::fabs(a) * std::fabs(b);
            }
            const auto idx = static_cast<std::size_t>(i * N + j);
            C[idx] = alpha * acc + beta * C[idx];
            scale[idx] = std::fabs(alpha) * abs_acc;
        }
    }
}

template <typename T>
void gemm_naive(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
                const std::vector<T>& B, T beta, std::vector<T>& C) {
    for (std::int64_t i = 0; i < M; ++i) {
        for (std::int64_t j = 0; j < N; ++j) {
            T acc = T{0};
            for (std::int64_t k = 0; k < K; ++k) {
                acc += A[static_cast<std::size_t>(i * K + k)] *
                       B[static_cast<std::size_t>(k * N + j)];
            }
            const auto idx = static_cast<std::size_t>(i * N + j);
            C[idx] = alpha * acc + beta * C[idx];
        }
    }
}

template <typename T>
void gemm_blocked(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
                  const std::vector<T>& B, T beta, std::vector<T>& C, std::int64_t block) {
    // Scale C by beta first so the blocked accumulation can add into it freely.
    for (std::int64_t i = 0; i < M * N; ++i) C[static_cast<std::size_t>(i)] *= beta;

    for (std::int64_t ii = 0; ii < M; ii += block) {
        const std::int64_t i_max = std::min(ii + block, M);
        for (std::int64_t kk = 0; kk < K; kk += block) {
            const std::int64_t k_max = std::min(kk + block, K);
            for (std::int64_t jj = 0; jj < N; jj += block) {
                const std::int64_t j_max = std::min(jj + block, N);
                for (std::int64_t i = ii; i < i_max; ++i) {
                    for (std::int64_t k = kk; k < k_max; ++k) {
                        const T a = alpha * A[static_cast<std::size_t>(i * K + k)];
                        for (std::int64_t j = jj; j < j_max; ++j) {
                            C[static_cast<std::size_t>(i * N + j)] +=
                                a * B[static_cast<std::size_t>(k * N + j)];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void gemm_omp(std::int64_t M, std::int64_t N, std::int64_t K, T alpha, const std::vector<T>& A,
              const std::vector<T>& B, T beta, std::vector<T>& C, std::int64_t block) {
    for (std::int64_t i = 0; i < M * N; ++i) C[static_cast<std::size_t>(i)] *= beta;

#ifdef PHOENIX_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t ii = 0; ii < M; ii += block) {
        const std::int64_t i_max = std::min(ii + block, M);
        for (std::int64_t kk = 0; kk < K; kk += block) {
            const std::int64_t k_max = std::min(kk + block, K);
            for (std::int64_t jj = 0; jj < N; jj += block) {
                const std::int64_t j_max = std::min(jj + block, N);
                for (std::int64_t i = ii; i < i_max; ++i) {
                    for (std::int64_t k = kk; k < k_max; ++k) {
                        const T a = alpha * A[static_cast<std::size_t>(i * K + k)];
                        for (std::int64_t j = jj; j < j_max; ++j) {
                            C[static_cast<std::size_t>(i * N + j)] +=
                                a * B[static_cast<std::size_t>(k * N + j)];
                        }
                    }
                }
            }
        }
    }
}

template void gemm_naive<double>(std::int64_t, std::int64_t, std::int64_t, double,
                                 const std::vector<double>&, const std::vector<double>&, double,
                                 std::vector<double>&);
template void gemm_naive<float>(std::int64_t, std::int64_t, std::int64_t, float,
                                const std::vector<float>&, const std::vector<float>&, float,
                                std::vector<float>&);
template void gemm_blocked<double>(std::int64_t, std::int64_t, std::int64_t, double,
                                   const std::vector<double>&, const std::vector<double>&, double,
                                   std::vector<double>&, std::int64_t);
template void gemm_blocked<float>(std::int64_t, std::int64_t, std::int64_t, float,
                                  const std::vector<float>&, const std::vector<float>&, float,
                                  std::vector<float>&, std::int64_t);
template void gemm_omp<double>(std::int64_t, std::int64_t, std::int64_t, double,
                               const std::vector<double>&, const std::vector<double>&, double,
                               std::vector<double>&, std::int64_t);
template void gemm_omp<float>(std::int64_t, std::int64_t, std::int64_t, float,
                              const std::vector<float>&, const std::vector<float>&, float,
                              std::vector<float>&, std::int64_t);

}  // namespace phoenix::cpu
