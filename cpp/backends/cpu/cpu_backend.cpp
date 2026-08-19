#include "cpu_backend.hpp"

#include <random>
#include <thread>

#include "gemm_kernels.hpp"
#include "phoenix/core/error.hpp"
#include "phoenix/timing/host_timer.hpp"

namespace phoenix::cpu {

BackendIdentity CpuBackend::identity() { return {"cpu", "0.1.0"}; }

std::vector<DeviceInfo> CpuBackend::probe() {
    DeviceInfo d;
    d.backend = "cpu";
    d.index = 0;
    d.name = "host CPU";
    d.notes = "concurrency=" + std::to_string(std::thread::hardware_concurrency());
    return {d};
}

BackendCapabilities CpuBackend::capabilities() {
    BackendCapabilities c;
    c.supported_dtypes = {DType::FP64, DType::FP32};
    c.device_side_timing = false;  // no device clock; device_time_ns stays null
    c.power_telemetry = false;     // no characterised power source on this backend
    c.alignment_bytes = 64;
    return c;
}

void CpuBackend::prepare(const RunRequest& r) {
    req_ = r;
    impl_ = r.implementation;
    fp32_ = (r.workload.dtype_compute == DType::FP32);

    const auto M = r.workload.M, N = r.workload.N, K = r.workload.K;
    const auto sa = static_cast<std::size_t>(M * K);
    const auto sb = static_cast<std::size_t>(K * N);
    const auto sc = static_cast<std::size_t>(M * N);

    // Deterministic inputs from the recorded seed: the same numbers regardless of
    // dtype, so an fp32 run and an fp64 run are comparable rather than merely similar.
    std::mt19937_64 rng(r.workload.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    a64_.resize(sa);
    b64_.resize(sb);
    for (auto& v : a64_) v = dist(rng);
    for (auto& v : b64_) v = dist(rng);
    c64_.assign(sc, 0.0);

    if (fp32_) {
        a32_.resize(sa);
        b32_.resize(sb);
        for (std::size_t i = 0; i < sa; ++i) a32_[i] = static_cast<float>(a64_[i]);
        for (std::size_t i = 0; i < sb; ++i) b32_[i] = static_cast<float>(b64_[i]);
        c32_.assign(sc, 0.0F);
    }

    // The trusted reference, always FP64, computed from the fp64 inputs.
    if (r.correctness.enabled) {
        reference_.assign(sc, 0.0);
        error_scale_.assign(sc, 0.0);
        gemm_reference_and_scale_fp64(M, N, K, r.workload.alpha, a64_, b64_, r.workload.beta,
                                      reference_, error_scale_);
    }
    output_.assign(sc, 0.0);
}

void CpuBackend::run_kernel() {
    const auto M = req_.workload.M, N = req_.workload.N, K = req_.workload.K;
    if (fp32_) {
        const auto alpha = static_cast<float>(req_.workload.alpha);
        const auto beta = static_cast<float>(req_.workload.beta);
        if (impl_ == "gemm_naive") {
            gemm_naive<float>(M, N, K, alpha, a32_, b32_, beta, c32_);
        } else if (impl_ == "gemm_blocked") {
            gemm_blocked<float>(M, N, K, alpha, a32_, b32_, beta, c32_);
        } else if (impl_ == "gemm_omp") {
            gemm_omp<float>(M, N, K, alpha, a32_, b32_, beta, c32_);
        } else {
            throw ConfigurationError("unknown cpu implementation: " + impl_);
        }
    } else {
        const double alpha = req_.workload.alpha;
        const double beta = req_.workload.beta;
        if (impl_ == "gemm_reference" || impl_ == "gemm_naive") {
            if (impl_ == "gemm_reference") {
                gemm_reference_fp64(M, N, K, alpha, a64_, b64_, beta, c64_);
            } else {
                gemm_naive<double>(M, N, K, alpha, a64_, b64_, beta, c64_);
            }
        } else if (impl_ == "gemm_blocked") {
            gemm_blocked<double>(M, N, K, alpha, a64_, b64_, beta, c64_);
        } else if (impl_ == "gemm_omp") {
            gemm_omp<double>(M, N, K, alpha, a64_, b64_, beta, c64_);
        } else {
            throw ConfigurationError("unknown cpu implementation: " + impl_);
        }
    }
}

void CpuBackend::warmup() { run_kernel(); }

ExecutionRecord CpuBackend::execute_once() {
    // beta=0 semantics mean C is fully overwritten each call; for beta!=0 the
    // accumulation would compound across iterations, so C is reset first.
    if (req_.workload.beta != 0.0) {
        const auto sc = static_cast<std::size_t>(req_.workload.M * req_.workload.N);
        if (fp32_) {
            c32_.assign(sc, 0.0F);
        } else {
            c64_.assign(sc, 0.0);
        }
    }
    HostTimer t;
    t.start();
    run_kernel();
    ExecutionRecord rec;
    rec.host_wall_ns = t.stop_ns();
    rec.device_time_ns = -1;  // null: there is no device clock on this backend
    return rec;
}

void CpuBackend::synchronize() {}

std::span<const double> CpuBackend::read_output() {
    if (fp32_) {
        output_.resize(c32_.size());
        for (std::size_t i = 0; i < c32_.size(); ++i) output_[i] = static_cast<double>(c32_[i]);
    } else {
        output_ = c64_;
    }
    return {output_.data(), output_.size()};
}

std::span<const double> CpuBackend::read_reference() {
    return {reference_.data(), reference_.size()};
}

std::span<const double> CpuBackend::read_error_scale() {
    return {error_scale_.data(), error_scale_.size()};
}

void CpuBackend::teardown() {}

}  // namespace phoenix::cpu
