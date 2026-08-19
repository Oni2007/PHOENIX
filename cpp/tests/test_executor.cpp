#include <gtest/gtest.h>

#include "phoenix/backend/backend.hpp"
#include "phoenix/bench/executor.hpp"
#include "phoenix/core/error.hpp"

using namespace phoenix;

namespace {

std::string request_json(const char* backend, const char* impl, std::int64_t K, int warmup,
                         int measure, bool correctness) {
    return std::string(R"({"schema_version":"1.0.0","experiment_id":"EXP-TEST","backend":")") +
           backend + R"(","implementation":")" + impl +
           R"(","workload":{"kind":"GEMM","M":8,"N":8,"K":)" + std::to_string(K) +
           R"(},"measurement":{"warmup_iterations":)" + std::to_string(warmup) +
           R"(,"measure_iterations":)" + std::to_string(measure) +
           R"(},"correctness":{"enabled":)" + (correctness ? "true" : "false") + "}}";
}

}  // namespace

TEST(Executor, RecoversAKnownDurationFromTheSyntheticBackend) {
    // The harness is validated against work of known duration before it is trusted
    // with work of unknown duration.
    const std::int64_t target_ns = 2'000'000;  // 2 ms
    auto backend = make_backend("synthetic");
    const auto cal = calibrate_timers(200);
    BenchmarkExecutor ex(*backend, cal);
    SampleBuffer samples;

    const auto req = RunRequest::from_json(request_json("synthetic", "busy", target_ns, 2, 10, false));
    const auto result = ex.run(req, samples);

    EXPECT_EQ(result.status, "COMPLETED");
    EXPECT_EQ(result.measure_samples, 10);
    ASSERT_EQ(samples.size(), 12U);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples.phase[i] == static_cast<std::int32_t>(Phase::Measure)) {
            EXPECT_GE(samples.host_wall_ns[i], target_ns)
                << "measured duration must not be below the known floor";
            EXPECT_LT(samples.host_wall_ns[i], target_ns * 20)
                << "measured duration implausibly far above the known value";
        }
    }
}

TEST(Executor, RetainsWarmupSamplesFlaggedRatherThanDiscardingThem) {
    auto backend = make_backend("synthetic");
    BenchmarkExecutor ex(*backend, calibrate_timers(100));
    SampleBuffer samples;
    const auto req = RunRequest::from_json(request_json("synthetic", "busy", 100000, 5, 5, false));
    const auto result = ex.run(req, samples);

    EXPECT_EQ(result.warmup_samples, 5);
    int warmup_rows = 0;
    for (auto p : samples.phase) {
        if (p == static_cast<std::int32_t>(Phase::Warmup)) ++warmup_rows;
    }
    EXPECT_EQ(warmup_rows, 5) << "warmup samples are retained, not discarded";
}

TEST(Executor, DeviceTimeIsNullOnBackendsWithoutADeviceClock) {
    auto backend = make_backend("cpu");
    BenchmarkExecutor ex(*backend, calibrate_timers(100));
    SampleBuffer samples;
    const auto req = RunRequest::from_json(request_json("cpu", "gemm_naive", 8, 1, 3, false));
    (void)ex.run(req, samples);
    for (auto d : samples.device_time_ns) {
        EXPECT_EQ(d, SampleBuffer::kNullI64) << "CPU has no device clock; must be null, not 0";
    }
}

TEST(Executor, CorrectnessGateRunsAndPasses) {
    auto backend = make_backend("cpu");
    BenchmarkExecutor ex(*backend, calibrate_timers(100));
    SampleBuffer samples;
    const auto req = RunRequest::from_json(request_json("cpu", "gemm_blocked", 32, 1, 3, true));
    const auto result = ex.run(req, samples);
    EXPECT_TRUE(result.correctness.checked);
    EXPECT_TRUE(result.correctness.passed);
    EXPECT_EQ(result.status, "COMPLETED");
}

TEST(Executor, FlagsMeasurementFloorRiskOnTrivialWork) {
    // A 1x1x1 GEMM is mostly harness overhead. The run is not silently reported as
    // a fast GEMM; it is flagged.
    auto backend = make_backend("cpu");
    const auto cal = calibrate_timers(500);
    BenchmarkExecutor ex(*backend, cal);
    SampleBuffer samples;
    const auto req = RunRequest::from_json(
        R"({"schema_version":"1.0.0","experiment_id":"E","backend":"cpu",)"
        R"("implementation":"gemm_naive","workload":{"M":1,"N":1,"K":1},)"
        R"("measurement":{"warmup_iterations":0,"measure_iterations":5},)"
        R"("correctness":{"enabled":false}})");
    const auto result = ex.run(req, samples);
    bool flagged = false;
    for (const auto& r : result.invalidation_reasons) {
        if (r == invalidation::kMeasurementFloorRisk) flagged = true;
    }
    EXPECT_TRUE(flagged);
}

TEST(Executor, RejectsUnknownBackendRatherThanFallingBack) {
    EXPECT_THROW((void)make_backend("cuda"), ConfigurationError);
    EXPECT_THROW((void)make_backend("hip"), ConfigurationError);
    EXPECT_THROW((void)make_backend("tpu"), ConfigurationError);
}
