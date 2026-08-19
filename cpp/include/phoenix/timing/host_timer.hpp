#pragma once
#include <chrono>
#include <cstdint>

namespace phoenix {

// The only clock used for host-side measurement. steady_clock is monotonic and
// unaffected by wall-clock adjustment.
class HostTimer {
public:
    using clock = std::chrono::steady_clock;

    void start() noexcept { t0_ = clock::now(); }
    [[nodiscard]] std::int64_t stop_ns() const noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0_).count();
    }
    [[nodiscard]] static std::int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count();
    }

private:
    clock::time_point t0_{};
};

// TDD §12.3. Measured before the first measurement of any session and recorded in
// the run's package. Any duration within `floor_multiple` x overhead is flagged
// MEASUREMENT_FLOOR_RISK — this is what prevents reporting a tiny GEMM "time" that
// is mostly harness overhead.
struct TimerCalibration {
    std::int64_t clock_resolution_ns = 0;
    std::int64_t timer_overhead_ns = 0;
    std::int64_t empty_loop_ns = 0;
    int samples = 0;
    double floor_multiple = 10.0;

    [[nodiscard]] std::int64_t measurement_floor_ns() const noexcept {
        return static_cast<std::int64_t>(floor_multiple *
                                         static_cast<double>(timer_overhead_ns));
    }
    [[nodiscard]] bool at_floor_risk(std::int64_t duration_ns) const noexcept {
        return duration_ns < measurement_floor_ns();
    }
};

TimerCalibration calibrate_timers(int samples = 2000);

}  // namespace phoenix
