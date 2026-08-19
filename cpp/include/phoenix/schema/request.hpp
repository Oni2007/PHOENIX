#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace phoenix {

enum class DType { FP64, FP32 };  // v0.1 CPU scope. FP16/BF16/TF32 arrive with CUDA.

std::string_view to_string(DType d) noexcept;
DType dtype_from_string(std::string_view s);
double dtype_epsilon(DType d) noexcept;

struct GemmWorkload {
    std::int64_t M = 0, N = 0, K = 0;
    DType dtype_a = DType::FP64;
    DType dtype_b = DType::FP64;
    DType dtype_c = DType::FP64;
    DType dtype_compute = DType::FP64;
    double alpha = 1.0;
    double beta = 0.0;
    std::uint64_t seed = 0;
};

struct MeasurementSpec {
    int warmup_iterations = 20;
    int measure_iterations = 100;
    int repetition_index = 0;
    bool retain_warmup_samples = true;
    int min_valid_samples = 1;
    double max_cv = 0.0;  // 0 disables the gate
};

struct CorrectnessSpec {
    bool enabled = true;
    std::string reference = "cpu_reference";
    double tolerance_c = 8.0;  // C in C*eps(dtype)*sqrt(K); reviewed, never raised to pass
};

struct RunRequest {
    std::string schema_version;
    std::string experiment_id;
    std::string backend;
    std::string implementation;
    GemmWorkload workload;
    MeasurementSpec measurement;
    CorrectnessSpec correctness;

    static RunRequest from_json(const std::string& json_text);
};

struct CorrectnessOutcome {
    bool checked = false;
    bool passed = false;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double max_cancellation = 0.0;
    double tolerance_rel = 0.0;
    std::string tolerance_rule;
    std::string reference;
};

struct RunResult {
    std::string schema_version = "1.0.0";
    std::string status;
    std::vector<std::string> invalidation_reasons;
    std::string backend;
    std::string implementation;
    std::int64_t prepare_ns = 0;
    std::int64_t first_execution_ns = 0;
    int warmup_samples = 0;
    int measure_samples = 0;
    CorrectnessOutcome correctness;
    std::string error_message;
    std::string error_context;
    bool timing_valid_for_reporting = true;

    [[nodiscard]] std::string to_json() const;
};

}  // namespace phoenix
