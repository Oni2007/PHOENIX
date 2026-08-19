#pragma once
#include "phoenix/backend/backend.hpp"
#include "phoenix/bench/sample_buffer.hpp"
#include "phoenix/core/status.hpp"
#include "phoenix/schema/request.hpp"
#include "phoenix/timing/host_timer.hpp"

namespace phoenix {

// Owns the measurement protocol (TDD §15.2), and nothing else:
//   prepare -> warmup -> correctness gate -> steady-state measurement -> flush
//
// The correctness check happens OUTSIDE the timed loop. No I/O happens inside it.
class BenchmarkExecutor {
public:
    BenchmarkExecutor(IBackend& backend, TimerCalibration calibration)
        : backend_(backend), calibration_(calibration) {}

    RunResult run(const RunRequest& request, SampleBuffer& samples);

private:
    void check_capabilities(const RunRequest& request) const;

    IBackend& backend_;
    TimerCalibration calibration_;
};

}  // namespace phoenix
