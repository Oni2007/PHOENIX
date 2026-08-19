#include "gemm_blas.hpp"

#include "phoenix/core/error.hpp"

namespace phoenix::cpu {

bool blas_available() noexcept {
#ifdef PHOENIX_HAVE_BLAS
    return true;
#else
    return false;
#endif
}

#ifdef PHOENIX_HAVE_BLAS

// The standard CBLAS ABI (netlib CBLAS / OpenBLAS's cblas_* entry points).
// Declared locally rather than depending on a system cblas.h: the integer
// constants below (101/102/111/112) are the fixed values of the CBLAS_LAYOUT and
// CBLAS_TRANSPOSE enums as specified by the reference CBLAS, not PHOENIX-specific
// choices, so this stays correct against any conforming CBLAS implementation.
extern "C" {
void cblas_dgemm(int layout, int transa, int transb, int m, int n, int k, double alpha,
                 const double* a, int lda, const double* b, int ldb, double beta, double* c,
                 int ldc);
void cblas_sgemm(int layout, int transa, int transb, int m, int n, int k, float alpha,
                 const float* a, int lda, const float* b, int ldb, float beta, float* c,
                 int ldc);
}

namespace {
constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
}  // namespace

void gemm_blas_fp64(std::int64_t M, std::int64_t N, std::int64_t K, double alpha,
                    const std::vector<double>& A, const std::vector<double>& B, double beta,
                    std::vector<double>& C) {
    cblas_dgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, static_cast<int>(M),
               static_cast<int>(N), static_cast<int>(K), alpha, A.data(), static_cast<int>(K),
               B.data(), static_cast<int>(N), beta, C.data(), static_cast<int>(N));
}

void gemm_blas_fp32(std::int64_t M, std::int64_t N, std::int64_t K, float alpha,
                    const std::vector<float>& A, const std::vector<float>& B, float beta,
                    std::vector<float>& C) {
    cblas_sgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, static_cast<int>(M),
               static_cast<int>(N), static_cast<int>(K), alpha, A.data(), static_cast<int>(K),
               B.data(), static_cast<int>(N), beta, C.data(), static_cast<int>(N));
}

#else  // !PHOENIX_HAVE_BLAS

void gemm_blas_fp64(std::int64_t, std::int64_t, std::int64_t, double, const std::vector<double>&,
                    const std::vector<double>&, double, std::vector<double>&) {
    throw ConfigurationError(
        "gemm_blas_fp64 called but this build has no BLAS (blas_available() was not "
        "checked first, or was checked and ignored)");
}

void gemm_blas_fp32(std::int64_t, std::int64_t, std::int64_t, float, const std::vector<float>&,
                    const std::vector<float>&, float, std::vector<float>&) {
    throw ConfigurationError(
        "gemm_blas_fp32 called but this build has no BLAS (blas_available() was not "
        "checked first, or was checked and ignored)");
}

#endif

}  // namespace phoenix::cpu
