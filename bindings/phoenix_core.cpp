// The one boundary between the Python control plane and the C++ compute plane.
//
// Six entry points, and JSON strings rather than rich bound types: every crossing
// is loggable verbatim into the reproducibility package, and the exact request that
// produced a result is recoverable byte-for-byte (TDD 7.2 / ADR-003).
#include <mutex>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "phoenix/backend/backend.hpp"
#include "phoenix/bench/executor.hpp"
#include "phoenix/bench/sample_buffer.hpp"
#include "phoenix/core/error.hpp"
#include "phoenix/schema/request.hpp"
#include "phoenix/timing/host_timer.hpp"

namespace nb = nanobind;
using namespace phoenix;

// execute() holds a process-wide lock. Concurrent benchmark execution destroys
// measurement validity, so the constraint is enforced rather than documented.
std::mutex g_execute_mutex;

struct RunOutput {
    std::string result_json;
    std::string calibration_json;
    SampleBuffer samples;
};

namespace {

template <typename T>
nb::ndarray<nb::numpy, T> as_array(std::vector<T>& v, nb::handle owner) {
    // Zero-copy view; `owner` ties the array's lifetime to the RunOutput.
    return nb::ndarray<nb::numpy, T>(v.data(), {v.size()}, owner);
}

std::string calibration_to_json(const TimerCalibration& c) {
    return std::string("{") + "\"clock_resolution_ns\":" +
           std::to_string(c.clock_resolution_ns) + ",\"timer_overhead_ns\":" +
           std::to_string(c.timer_overhead_ns) + ",\"empty_loop_ns\":" +
           std::to_string(c.empty_loop_ns) + ",\"samples\":" + std::to_string(c.samples) +
           ",\"floor_multiple\":" + std::to_string(c.floor_multiple) +
           ",\"measurement_floor_ns\":" + std::to_string(c.measurement_floor_ns()) +
           ",\"provenance\":\"MEASURED\"}";
}

}  // namespace

NB_MODULE(phoenix_core, m) {
    m.doc() = "PHOENIX C++ compute plane";

    nb::class_<RunOutput>(m, "RunOutput")
        .def(nb::init<>())
        .def_ro("result_json", &RunOutput::result_json)
        .def_ro("calibration_json", &RunOutput::calibration_json)
        .def("sample_index",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.sample_index, self);
             })
        .def("repetition_index",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.repetition_index, self);
             })
        .def("phase",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.phase, self);
             })
        .def("host_wall_ns",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.host_wall_ns, self);
             })
        .def("device_time_ns",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.device_time_ns, self);
             })
        .def("api_overhead_ns",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.api_overhead_ns, self);
             })
        .def("correctness_checked",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.correctness_checked, self);
             })
        .def("max_abs_err",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.max_abs_err, self);
             })
        .def("max_rel_err",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.max_rel_err, self);
             })
        .def("timestamp_ns",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.timestamp_ns, self);
             })
        .def("valid",
             [](nb::object self) {
                 auto& r = nb::cast<RunOutput&>(self);
                 return as_array(r.samples.valid, self);
             })
        .def("__len__", [](RunOutput& r) { return r.samples.size(); });

    m.def("abi_version", [] { return std::string("1.0"); });

    m.def("build_info", [] {
        std::string compiler = "unknown";
#if defined(__clang__)
        compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        compiler = "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
                   std::to_string(__GNUC_PATCHLEVEL__);
#endif
        std::string openmp = "false";
#ifdef PHOENIX_HAVE_OPENMP
        openmp = "true";
#endif
        return std::string("{\"compiler\":\"") + compiler + "\",\"cxx_standard\":\"" +
               std::to_string(__cplusplus) + "\",\"openmp\":" + openmp +
               ",\"build_type\":\"" + PHOENIX_BUILD_TYPE + "\"}";
    });

    m.def("enumerate_backends", [] {
        std::vector<std::string> out;
        for (const auto& b : enumerate_backends()) out.push_back(b.name + ":" + b.version);
        return out;
    });

    m.def("probe_device", [](const std::string& backend, int index) {
        auto b = make_backend(backend);
        auto devices = b->probe();
        if (index < 0 || index >= static_cast<int>(devices.size())) {
            throw std::runtime_error("device index out of range for backend " + backend);
        }
        const auto& d = devices[static_cast<std::size_t>(index)];
        return std::string("{\"backend\":\"") + d.backend +
               "\",\"index\":" + std::to_string(d.index) + ",\"name\":\"" + d.name +
               "\",\"notes\":\"" + d.notes + "\"}";
    });

    m.def("validate_request", [](const std::string& request_json) {
        try {
            const auto r = RunRequest::from_json(request_json);
            auto b = make_backend(r.backend);
            if (!b->capabilities().supports(r.workload.dtype_compute)) {
                return std::string("{\"valid\":false,\"error\":\"backend does not support dtype\"}");
            }
            return std::string("{\"valid\":true,\"error\":null}");
        } catch (const PhoenixError& e) {
            return std::string("{\"valid\":false,\"error\":\"") + e.what() + "\"}";
        }
    });

    m.def(
        "calibrate",
        [](int samples) {
            nb::gil_scoped_release release;
            return calibration_to_json(calibrate_timers(samples));
        },
        nb::arg("samples") = 2000);

    m.def(
        "execute",
        [](const std::string& request_json, int calibration_samples) {
            RunOutput out;
            {
                // The GIL is released for the whole measured region. Python is never
                // on the critical path of a measurement.
                nb::gil_scoped_release release;
                std::scoped_lock lock(g_execute_mutex);

                const auto cal = calibrate_timers(calibration_samples);
                out.calibration_json = calibration_to_json(cal);

                const auto request = RunRequest::from_json(request_json);
                auto backend = make_backend(request.backend);
                BenchmarkExecutor executor(*backend, cal);
                try {
                    const auto result = executor.run(request, out.samples);
                    out.result_json = result.to_json();
                } catch (const PhoenixError& e) {
                    // Partial samples are retained and the failure is recorded as
                    // data, not thrown away.
                    RunResult failed;
                    failed.status = "FAILED";
                    failed.backend = request.backend;
                    failed.implementation = request.implementation;
                    failed.error_message = e.what();
                    failed.error_context = e.context();
                    out.result_json = failed.to_json();
                }
            }
            return out;
        },
        nb::arg("request_json"), nb::arg("calibration_samples") = 2000);
}
