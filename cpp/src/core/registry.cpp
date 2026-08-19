#include "../../backends/cpu/cpu_backend.hpp"
#include "../../backends/synthetic/synthetic_backend.hpp"
#include "phoenix/backend/backend.hpp"
#include "phoenix/core/error.hpp"

namespace phoenix {

bool BackendCapabilities::supports(DType d) const noexcept {
    for (auto s : supported_dtypes) {
        if (s == d) return true;
    }
    return false;
}

std::unique_ptr<IBackend> make_backend(const std::string& name) {
    if (name == "cpu") return std::make_unique<BackendAdapter<cpu::CpuBackend>>();
    if (name == "synthetic") return std::make_unique<BackendAdapter<synthetic::SyntheticBackend>>();
    // hip and tpu are deliberately absent: v0.1 claims no support for hardware
    // that has never executed a run.
    throw ConfigurationError("unknown or unimplemented backend: " + name);
}

std::vector<BackendIdentity> enumerate_backends() {
    return {cpu::CpuBackend::identity(), synthetic::SyntheticBackend::identity()};
}

}  // namespace phoenix
