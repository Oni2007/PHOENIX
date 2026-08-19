#include "phoenix/schema/request.hpp"

#include <limits>

#include <nlohmann/json.hpp>

#include "phoenix/core/error.hpp"

using json = nlohmann::json;

namespace phoenix {

std::string_view to_string(DType d) noexcept {
    switch (d) {
        case DType::FP64: return "fp64";
        case DType::FP32: return "fp32";
    }
    return "fp64";
}

DType dtype_from_string(std::string_view s) {
    if (s == "fp64" || s == "float64") return DType::FP64;
    if (s == "fp32" || s == "float32") return DType::FP32;
    throw ConfigurationError("unsupported dtype in v0.1 CPU scope: " + std::string(s));
}

double dtype_epsilon(DType d) noexcept {
    switch (d) {
        case DType::FP64: return std::numeric_limits<double>::epsilon();
        case DType::FP32: return static_cast<double>(std::numeric_limits<float>::epsilon());
    }
    return std::numeric_limits<double>::epsilon();
}

namespace {

const json& require(const json& j, const char* key) {
    if (!j.contains(key)) {
        throw ConfigurationError(std::string("request missing required field: ") + key);
    }
    return j.at(key);
}

}  // namespace

RunRequest RunRequest::from_json(const std::string& json_text) {
    json j;
    try {
        j = json::parse(json_text);
    } catch (const json::exception& e) {
        throw ConfigurationError(std::string("request is not valid JSON: ") + e.what());
    }

    RunRequest r;
    r.schema_version = require(j, "schema_version").get<std::string>();
    if (r.schema_version != "1.0.0") {
        throw ConfigurationError("unsupported request schema_version: " + r.schema_version);
    }
    r.experiment_id = require(j, "experiment_id").get<std::string>();
    r.backend = require(j, "backend").get<std::string>();
    r.implementation = require(j, "implementation").get<std::string>();

    const auto& w = require(j, "workload");
    if (w.value("kind", std::string("GEMM")) != "GEMM") {
        throw ConfigurationError("v0.1 supports only workload kind GEMM");
    }
    r.workload.M = require(w, "M").get<std::int64_t>();
    r.workload.N = require(w, "N").get<std::int64_t>();
    r.workload.K = require(w, "K").get<std::int64_t>();
    if (r.workload.M <= 0 || r.workload.N <= 0 || r.workload.K <= 0) {
        throw ConfigurationError("GEMM dimensions must all be positive");
    }
    r.workload.dtype_a = dtype_from_string(w.value("dtype_a", "fp64"));
    r.workload.dtype_b = dtype_from_string(w.value("dtype_b", "fp64"));
    r.workload.dtype_c = dtype_from_string(w.value("dtype_c", "fp64"));
    r.workload.dtype_compute = dtype_from_string(w.value("dtype_compute", "fp64"));
    r.workload.alpha = w.value("alpha", 1.0);
    r.workload.beta = w.value("beta", 0.0);
    r.workload.seed = w.value("seed", static_cast<std::uint64_t>(0));

    if (j.contains("measurement")) {
        const auto& m = j.at("measurement");
        r.measurement.warmup_iterations = m.value("warmup_iterations", 20);
        r.measurement.measure_iterations = m.value("measure_iterations", 100);
        r.measurement.repetition_index = m.value("repetition_index", 0);
        r.measurement.retain_warmup_samples = m.value("retain_warmup_samples", true);
        r.measurement.min_valid_samples = m.value("min_valid_samples", 1);
        r.measurement.max_cv = m.value("max_cv", 0.0);
    }
    if (r.measurement.measure_iterations <= 0) {
        throw ConfigurationError("measure_iterations must be positive");
    }
    if (r.measurement.warmup_iterations < 0) {
        throw ConfigurationError("warmup_iterations must not be negative");
    }

    if (j.contains("correctness")) {
        const auto& c = j.at("correctness");
        r.correctness.enabled = c.value("enabled", true);
        r.correctness.reference = c.value("reference", std::string("cpu_reference"));
        r.correctness.tolerance_c = c.value("tolerance_c", 8.0);
        if (r.correctness.tolerance_c <= 0.0) {
            throw ConfigurationError("correctness.tolerance_c must be positive");
        }
    }
    return r;
}

std::string RunResult::to_json() const {
    json j;
    j["schema_version"] = schema_version;
    j["status"] = status;
    j["invalidation_reasons"] = invalidation_reasons;
    j["backend"] = backend;
    j["implementation"] = implementation;
    j["prepare_ns"] = prepare_ns;
    j["first_execution_ns"] = first_execution_ns;
    j["warmup_samples"] = warmup_samples;
    j["measure_samples"] = measure_samples;
    j["timing_valid_for_reporting"] = timing_valid_for_reporting;

    json c;
    c["checked"] = correctness.checked;
    if (correctness.checked) {
        c["passed"] = correctness.passed;
        c["max_abs_err"] = correctness.max_abs_err;
        c["max_rel_err"] = correctness.max_rel_err;
        c["max_cancellation"] = correctness.max_cancellation;
        c["tolerance_rel"] = correctness.tolerance_rel;
        c["tolerance_rule"] = correctness.tolerance_rule;
        c["reference"] = correctness.reference;
    } else {
        // Never invent a correctness verdict for a check that did not run.
        c["passed"] = nullptr;
        c["max_abs_err"] = nullptr;
        c["max_rel_err"] = nullptr;
        c["max_cancellation"] = nullptr;
        c["tolerance_rel"] = nullptr;
        c["tolerance_rule"] = nullptr;
        c["reference"] = nullptr;
    }
    j["correctness"] = c;

    if (!error_message.empty()) {
        j["error"] = {{"message", error_message}, {"context", error_context}};
    } else {
        j["error"] = nullptr;
    }
    return j.dump();
}

}  // namespace phoenix
