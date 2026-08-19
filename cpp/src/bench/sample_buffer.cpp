#include "phoenix/bench/sample_buffer.hpp"

namespace phoenix {

void SampleBuffer::reserve(std::size_t n) {
    sample_index.reserve(n);
    repetition_index.reserve(n);
    phase.reserve(n);
    host_wall_ns.reserve(n);
    device_time_ns.reserve(n);
    api_overhead_ns.reserve(n);
    correctness_checked.reserve(n);
    max_abs_err.reserve(n);
    max_rel_err.reserve(n);
    timestamp_ns.reserve(n);
    valid.reserve(n);
}

void SampleBuffer::push(std::int32_t si, std::int32_t ri, Phase ph, std::int64_t host_ns,
                        std::int64_t dev_ns, std::int64_t api_ns, bool checked, double abs_err,
                        double rel_err, std::int64_t ts_ns, bool is_valid) {
    sample_index.push_back(si);
    repetition_index.push_back(ri);
    phase.push_back(static_cast<std::int32_t>(ph));
    host_wall_ns.push_back(host_ns);
    device_time_ns.push_back(dev_ns);
    api_overhead_ns.push_back(api_ns);
    correctness_checked.push_back(checked ? 1U : 0U);
    max_abs_err.push_back(abs_err);
    max_rel_err.push_back(rel_err);
    timestamp_ns.push_back(ts_ns);
    valid.push_back(is_valid ? 1U : 0U);
}

void SampleBuffer::clear() {
    sample_index.clear();
    repetition_index.clear();
    phase.clear();
    host_wall_ns.clear();
    device_time_ns.clear();
    api_overhead_ns.clear();
    correctness_checked.clear();
    max_abs_err.clear();
    max_rel_err.clear();
    timestamp_ns.clear();
    valid.clear();
}

}  // namespace phoenix
