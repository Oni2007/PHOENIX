#include "phoenix/bench/executor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "phoenix/core/error.hpp"
#include "phoenix/correctness/comparator.hpp"

namespace phoenix {

void BenchmarkExecutor::check_capabilities(const RunRequest& request) const {
    const auto caps = backend_.capabilities();
    // Refuse what the backend cannot honour rather than silently substituting
    // behaviour (e.g. quietly promoting fp32 to fp64). TDD 12.1.
    if (!caps.supports(request.workload.dtype_compute)) {
        throw ConfigurationError("backend does not support requested compute dtype: " +
                                 std::string(to_string(request.workload.dtype_compute)));
    }
}

RunResult BenchmarkExecutor::run(const RunRequest& request, SampleBuffer& samples) {
    RunResult result;
    result.backend = request.backend;
    result.implementation = request.implementation;
    ValidityReport validity;

    check_capabilities(request);

    const auto total = static_cast<std::size_t>(request.measurement.warmup_iterations +
                                                request.measurement.measure_iterations);
    samples.reserve(total);

    // prepare: allocation, buffer fill, reference computation
    HostTimer prep;
    prep.start();
    backend_.prepare(request);
    result.prepare_ns = prep.stop_ns();

    const std::int64_t run_start_ns = HostTimer::now_ns();
    std::int32_t index = 0;

    // warmup: recorded, flagged, excluded from aggregation, never discarded.
    // Retaining it is what makes convergence to steady state analysable.
    for (int i = 0; i < request.measurement.warmup_iterations; ++i) {
        HostTimer t;
        t.start();
        const auto rec = backend_.execute_once();
        backend_.synchronize();
        const auto host_ns = t.stop_ns();
        if (i == 0) result.first_execution_ns = host_ns;
        if (request.measurement.retain_warmup_samples) {
            samples.push(index, request.measurement.repetition_index, Phase::Warmup, host_ns,
                         rec.device_time_ns, SampleBuffer::kNullI64, false,
                         SampleBuffer::kNullF64, SampleBuffer::kNullF64,
                         HostTimer::now_ns() - run_start_ns, true);
        }
        ++index;
        result.warmup_samples++;
    }

    // correctness gate: OUTSIDE the timed loop, before any measurement
    if (request.correctness.enabled) {
        backend_.execute_once();
        backend_.synchronize();
        const auto out = backend_.read_output();
        const auto ref = backend_.read_reference();
        const auto scale = backend_.read_error_scale();
        const auto cmp = compare(out, ref, scale, request.workload.dtype_compute,
                                 request.workload.K, request.correctness.tolerance_c);
        result.correctness.checked = true;
        result.correctness.passed = cmp.passed;
        result.correctness.max_abs_err = cmp.max_abs_err;
        result.correctness.max_rel_err = cmp.max_rel_err;
        result.correctness.max_cancellation = cmp.max_cancellation;
        result.correctness.tolerance_rel = cmp.tolerance_rel;
        result.correctness.tolerance_rule = cmp.rule;
        result.correctness.reference = request.correctness.reference;
        if (!cmp.passed) validity.fail(invalidation::kCorrectnessFailed);
    }

    // steady-state measurement
    std::vector<double> measured;
    measured.reserve(static_cast<std::size_t>(request.measurement.measure_iterations));
    for (int i = 0; i < request.measurement.measure_iterations; ++i) {
        HostTimer t;
        t.start();
        const auto rec = backend_.execute_once();
        backend_.synchronize();
        const auto host_ns = t.stop_ns();
        samples.push(index, request.measurement.repetition_index, Phase::Measure, host_ns,
                     rec.device_time_ns, SampleBuffer::kNullI64, false, SampleBuffer::kNullF64,
                     SampleBuffer::kNullF64, HostTimer::now_ns() - run_start_ns, true);
        measured.push_back(static_cast<double>(host_ns));
        ++index;
        result.measure_samples++;
    }

    backend_.teardown();

    // Validity gates. The run's data is written regardless of the verdict:
    // an invalid run is data about the measurement environment.
    if (result.measure_samples < request.measurement.min_valid_samples) {
        validity.fail(invalidation::kInsufficientSamples);
    }
    if (!measured.empty()) {
        std::vector<double> sorted = measured;
        const auto mid = static_cast<std::ptrdiff_t>(sorted.size() / 2);
        std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
        const double median = sorted[sorted.size() / 2];
        if (calibration_.at_floor_risk(static_cast<std::int64_t>(median))) {
            validity.flag(invalidation::kMeasurementFloorRisk);
        }
        const double mean = std::accumulate(measured.begin(), measured.end(), 0.0) /
                            static_cast<double>(measured.size());
        double ss = 0.0;
        for (double v : measured) ss += (v - mean) * (v - mean);
        const double sd =
            measured.size() > 1 ? std::sqrt(ss / static_cast<double>(measured.size() - 1)) : 0.0;
        const double cv = mean > 0.0 ? sd / mean : 0.0;
        if (request.measurement.max_cv > 0.0 && cv > request.measurement.max_cv) {
            validity.flag(invalidation::kHighVariance);
        }
    }

    result.invalidation_reasons = validity.reasons;
    result.status = validity.valid ? std::string("COMPLETED") : std::string("INVALID");
    return result;
}

}  // namespace phoenix
