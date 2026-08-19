#include "synthetic_backend.hpp"

#include "phoenix/timing/host_timer.hpp"

namespace phoenix::synthetic {

BackendIdentity SyntheticBackend::identity() { return {"synthetic", "0.1.0"}; }

std::vector<DeviceInfo> SyntheticBackend::probe() {
    DeviceInfo d;
    d.backend = "synthetic";
    d.index = 0;
    d.name = "synthetic known-duration workload";
    d.notes = "K encodes the target busy-wait duration in nanoseconds";
    return {d};
}

BackendCapabilities SyntheticBackend::capabilities() {
    BackendCapabilities c;
    c.supported_dtypes = {DType::FP64, DType::FP32};
    c.device_side_timing = false;
    c.power_telemetry = false;
    return c;
}

void SyntheticBackend::prepare(const RunRequest& r) { target_ns_ = r.workload.K; }

void SyntheticBackend::warmup() { execute_once(); }

ExecutionRecord SyntheticBackend::execute_once() {
    HostTimer t;
    t.start();
    // Busy-wait rather than sleep: sleeping yields the core and measures the
    // scheduler's wake-up latency instead of the harness.
    while (t.stop_ns() < target_ns_) {
    }
    ExecutionRecord rec;
    rec.host_wall_ns = t.stop_ns();
    rec.device_time_ns = -1;
    return rec;
}

void SyntheticBackend::synchronize() {}
std::span<const double> SyntheticBackend::read_output() { return {output_.data(), output_.size()}; }
std::span<const double> SyntheticBackend::read_reference() {
    return {reference_.data(), reference_.size()};
}
std::span<const double> SyntheticBackend::read_error_scale() {
    return {};  // not a reduction: the comparator falls back to |ref|
}
void SyntheticBackend::teardown() {}

}  // namespace phoenix::synthetic
