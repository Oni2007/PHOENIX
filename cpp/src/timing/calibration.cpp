#include "phoenix/timing/host_timer.hpp"

#include <algorithm>
#include <vector>

namespace phoenix {

TimerCalibration calibrate_timers(int samples) {
    TimerCalibration cal;
    cal.samples = samples;

    // Clock resolution: smallest non-zero delta the clock will report.
    std::int64_t resolution = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < samples; ++i) {
        const auto a = HostTimer::clock::now();
        HostTimer::clock::time_point b;
        do {
            b = HostTimer::clock::now();
        } while (b == a);
        const auto d =
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
        resolution = std::min(resolution, d);
    }
    cal.clock_resolution_ns = resolution;

    // Timer overhead: cost of a start/stop pair around nothing. Median, so a
    // single scheduler interruption does not set the floor for the whole session.
    std::vector<std::int64_t> overheads;
    overheads.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        HostTimer t;
        t.start();
        overheads.push_back(t.stop_ns());
    }
    std::ranges::nth_element(overheads, overheads.begin() + samples / 2);
    cal.timer_overhead_ns = overheads[static_cast<std::size_t>(samples / 2)];

    // Empty measured-loop cost for the same iteration count a real run would use.
    HostTimer loop;
    loop.start();
    volatile int sink = 0;
    for (int i = 0; i < samples; ++i) sink = sink + 1;
    (void)sink;
    cal.empty_loop_ns = loop.stop_ns() / std::max(1, samples);

    return cal;
}

}  // namespace phoenix
