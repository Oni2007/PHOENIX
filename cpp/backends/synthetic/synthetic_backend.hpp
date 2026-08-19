#pragma once
#include <span>
#include <vector>

#include "phoenix/backend/backend.hpp"

namespace phoenix::synthetic {

// A backend whose execution time is KNOWN by construction. It exists so the
// measurement harness can be tested against a known answer, and so the executor
// could be built and validated before any real kernel existed -- which is what
// stops CUDA-shaped assumptions from leaking into the executor (TDD week 3).
//
// The "workload" is a busy-wait of a requested duration, encoded in K nanoseconds.
class SyntheticBackend {
public:
    static BackendIdentity identity();
    std::vector<DeviceInfo> probe();
    BackendCapabilities capabilities();
    void prepare(const RunRequest& r);
    void warmup();
    ExecutionRecord execute_once();
    void synchronize();
    std::span<const double> read_output();
    std::span<const double> read_reference();
    std::span<const double> read_error_scale();
    void teardown();

private:
    std::int64_t target_ns_ = 0;
    std::vector<double> output_{0.0};
    std::vector<double> reference_{0.0};
};

static_assert(Backend<SyntheticBackend>, "SyntheticBackend must satisfy the Backend contract");

}  // namespace phoenix::synthetic
