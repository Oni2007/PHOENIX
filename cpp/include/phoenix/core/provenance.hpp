// Provenance is a first-class type, not documentation (TDD P3 / ADR-004).
// It is impossible to store a number in PHOENIX without declaring where it came from.
#pragma once
#include <optional>
#include <string>
#include <string_view>

namespace phoenix {

enum class Provenance {
    Measured,
    VendorReported,
    Simulated,
    Analytical,
    Hypothetical,
    Derived,
};

std::string_view to_string(Provenance p) noexcept;
std::optional<Provenance> provenance_from_string(std::string_view s) noexcept;

// A numeric value that cannot exist without a provenance class.
// `value` is nullopt when the quantity is UNKNOWN / NOT YET MEASURED — null is a
// legal, first-class value precisely so nobody is pressured into inventing a number.
struct ProvenanceValue {
    std::optional<double> value;
    std::string unit;
    Provenance provenance;

    static ProvenanceValue unknown(std::string unit, Provenance p) {
        return ProvenanceValue{std::nullopt, std::move(unit), p};
    }
    [[nodiscard]] bool is_known() const noexcept { return value.has_value(); }
};

}  // namespace phoenix
