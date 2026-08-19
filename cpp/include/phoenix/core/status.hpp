#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace phoenix {

enum class RunStatus { Completed, Failed, Invalid };

std::string_view to_string(RunStatus s) noexcept;

// §32 invalidation codes. Invalid runs are FLAGGED and RETAINED, never deleted:
// a pattern of throttling across a machine is a finding about that machine.
namespace invalidation {
inline constexpr std::string_view kCorrectnessFailed      = "CORRECTNESS_FAILED";
inline constexpr std::string_view kInsufficientSamples    = "INSUFFICIENT_SAMPLES";
inline constexpr std::string_view kHighVariance           = "HIGH_VARIANCE";
inline constexpr std::string_view kMeasurementFloorRisk   = "MEASUREMENT_FLOOR_RISK";
inline constexpr std::string_view kInsufficientWarmup     = "INSUFFICIENT_WARMUP";
inline constexpr std::string_view kUnsupportedPrecision   = "UNSUPPORTED_PRECISION";
inline constexpr std::string_view kPowerUnavailable       = "POWER_UNAVAILABLE";
}  // namespace invalidation

struct ValidityReport {
    bool valid = true;
    std::vector<std::string> reasons;

    void fail(std::string_view code) {
        valid = false;
        reasons.emplace_back(code);
    }
    void flag(std::string_view code) { reasons.emplace_back(code); }
};

}  // namespace phoenix
