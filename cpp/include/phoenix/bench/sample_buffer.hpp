#pragma once
#include <cstdint>
#include <limits>
#include <vector>

namespace phoenix {

enum class Phase : std::int32_t { Warmup = 0, Measure = 1 };

// Struct-of-arrays: one column per Tier-1 field (TDD §10.1). SoA is what Parquet
// wants, and it lets every column be handed to Python zero-copy.
//
// Null encoding across the boundary: -1 for integer columns, NaN for float columns.
// The Python side converts these to real Arrow nulls on write.
class SampleBuffer {
public:
    static constexpr std::int64_t kNullI64 = -1;
    static inline const double kNullF64 = std::numeric_limits<double>::quiet_NaN();

    void reserve(std::size_t n);

    void push(std::int32_t sample_index, std::int32_t repetition_index, Phase phase,
              std::int64_t host_wall_ns, std::int64_t device_time_ns,
              std::int64_t api_overhead_ns, bool correctness_checked, double max_abs_err,
              double max_rel_err, std::int64_t timestamp_ns, bool valid);

    [[nodiscard]] std::size_t size() const noexcept { return sample_index.size(); }
    void clear();

    std::vector<std::int32_t> sample_index;
    std::vector<std::int32_t> repetition_index;
    std::vector<std::int32_t> phase;
    std::vector<std::int64_t> host_wall_ns;
    std::vector<std::int64_t> device_time_ns;
    std::vector<std::int64_t> api_overhead_ns;
    std::vector<std::uint8_t> correctness_checked;
    std::vector<double> max_abs_err;
    std::vector<double> max_rel_err;
    std::vector<std::int64_t> timestamp_ns;
    std::vector<std::uint8_t> valid;
};

}  // namespace phoenix
