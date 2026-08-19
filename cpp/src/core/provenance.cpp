#include "phoenix/core/provenance.hpp"

namespace phoenix {

std::string_view to_string(Provenance p) noexcept {
    switch (p) {
        case Provenance::Measured: return "MEASURED";
        case Provenance::VendorReported: return "VENDOR_REPORTED";
        case Provenance::Simulated: return "SIMULATED";
        case Provenance::Analytical: return "ANALYTICAL";
        case Provenance::Hypothetical: return "HYPOTHETICAL";
        case Provenance::Derived: return "DERIVED";
    }
    return "UNKNOWN";
}

std::optional<Provenance> provenance_from_string(std::string_view s) noexcept {
    if (s == "MEASURED") return Provenance::Measured;
    if (s == "VENDOR_REPORTED") return Provenance::VendorReported;
    if (s == "SIMULATED") return Provenance::Simulated;
    if (s == "ANALYTICAL") return Provenance::Analytical;
    if (s == "HYPOTHETICAL") return Provenance::Hypothetical;
    if (s == "DERIVED") return Provenance::Derived;
    return std::nullopt;
}

}  // namespace phoenix
