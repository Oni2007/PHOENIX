#pragma once
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "phoenix/schema/request.hpp"

namespace phoenix {

struct BackendIdentity {
    std::string name;
    std::string version;
};

struct DeviceInfo {
    std::string backend;
    int index = 0;
    std::string name;
    std::string notes;
};

struct BackendCapabilities {
    std::vector<DType> supported_dtypes;
    bool device_side_timing = false;
    bool power_telemetry = false;
    std::size_t alignment_bytes = 64;

    [[nodiscard]] bool supports(DType d) const noexcept;
};

struct ExecutionRecord {
    std::int64_t host_wall_ns = 0;
    std::int64_t device_time_ns = -1;  // -1 == null; no device clock on CPU
};

// ── The contract every backend satisfies (TDD §12.1) ──────────────────────────
//
// Designed against the TPU execution model, not CUDA (P10 / ADR-025). `prepare()`
// is separated from `execute_once()` because on a compile-then-execute device the
// compilation cost is one of the most important things to report, and a CUDA-shaped
// interface would hide it. On CPU, prepare() allocates and fills buffers.
template <typename T>
concept Backend = requires(T b, const RunRequest& r) {
    { T::identity() } -> std::same_as<BackendIdentity>;
    { b.probe() } -> std::same_as<std::vector<DeviceInfo>>;
    { b.capabilities() } -> std::same_as<BackendCapabilities>;
    { b.prepare(r) } -> std::same_as<void>;
    { b.warmup() } -> std::same_as<void>;
    { b.execute_once() } -> std::same_as<ExecutionRecord>;
    { b.synchronize() } -> std::same_as<void>;
    { b.read_output() } -> std::same_as<std::span<const double>>;
    { b.read_reference() } -> std::same_as<std::span<const double>>;
    { b.read_error_scale() } -> std::same_as<std::span<const double>>;
    { b.teardown() } -> std::same_as<void>;
};

// Runtime-polymorphic face of the same contract, so the executor and registry can
// hold a backend without being templated on it.
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual BackendIdentity identity() const = 0;
    virtual std::vector<DeviceInfo> probe() = 0;
    virtual BackendCapabilities capabilities() = 0;
    virtual void prepare(const RunRequest& r) = 0;
    virtual void warmup() = 0;
    virtual ExecutionRecord execute_once() = 0;
    virtual void synchronize() = 0;
    virtual std::span<const double> read_output() = 0;
    virtual std::span<const double> read_reference() = 0;
    virtual std::span<const double> read_error_scale() = 0;
    virtual void teardown() = 0;
};

// Adapter: any type satisfying the concept becomes an IBackend. The static_assert
// means a backend that does not satisfy the contract fails to COMPILE rather than
// failing at runtime in the middle of an experiment.
template <Backend T>
class BackendAdapter final : public IBackend {
public:
    BackendIdentity identity() const override { return T::identity(); }
    std::vector<DeviceInfo> probe() override { return impl_.probe(); }
    BackendCapabilities capabilities() override { return impl_.capabilities(); }
    void prepare(const RunRequest& r) override { impl_.prepare(r); }
    void warmup() override { impl_.warmup(); }
    ExecutionRecord execute_once() override { return impl_.execute_once(); }
    void synchronize() override { impl_.synchronize(); }
    std::span<const double> read_output() override { return impl_.read_output(); }
    std::span<const double> read_reference() override { return impl_.read_reference(); }
    std::span<const double> read_error_scale() override {
        return impl_.read_error_scale();
    }
    void teardown() override { impl_.teardown(); }

private:
    T impl_{};
};

std::unique_ptr<IBackend> make_backend(const std::string& name);
std::vector<BackendIdentity> enumerate_backends();

}  // namespace phoenix
