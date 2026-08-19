#pragma once
#include <span>
#include <string>
#include <vector>

#include "phoenix/backend/backend.hpp"

namespace phoenix::cpu {

class CpuBackend {
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
    void run_kernel();

    RunRequest req_{};
    std::string impl_;
    bool fp32_ = false;

    std::vector<double> a64_, b64_, c64_;
    std::vector<float> a32_, b32_, c32_;
    std::vector<double> reference_;
    std::vector<double> error_scale_;
    std::vector<double> output_;
};

static_assert(Backend<CpuBackend>, "CpuBackend must satisfy the Backend contract");

}  // namespace phoenix::cpu
