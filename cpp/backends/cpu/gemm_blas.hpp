#pragma once
#include <cstdint>
#include <vector>

namespace phoenix::cpu {

// The independent vendor-BLAS cross-check the TDD's correctness hierarchy calls
// for (§14.1): our own GEMM-0 triple loop is algorithmically independent of
// naive/blocked/omp, but it is still PHOENIX's own code. A shared bug in the
// project's understanding of the math would not be caught by checking PHOENIX
// against PHOENIX. An externally-maintained, independently-implemented BLAS is.
//
// Gated on availability rather than assumed: this development host has no root
// access and therefore no system BLAS (verified: absent from the ld cache). CI
// runners DO have root, install libopenblas-dev, and exercise this path for real.
// Locally, callers must check blas_available() first; calling the gemm functions
// when it is false is a programming error, not a graceful degradation path.
bool blas_available() noexcept;

void gemm_blas_fp64(std::int64_t M, std::int64_t N, std::int64_t K, double alpha,
                    const std::vector<double>& A, const std::vector<double>& B, double beta,
                    std::vector<double>& C);

void gemm_blas_fp32(std::int64_t M, std::int64_t N, std::int64_t K, float alpha,
                    const std::vector<float>& A, const std::vector<float>& B, float beta,
                    std::vector<float>& C);

}  // namespace phoenix::cpu
