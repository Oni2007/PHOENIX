#include <gtest/gtest.h>

#include "phoenix/core/provenance.hpp"
#include "phoenix/core/status.hpp"

using namespace phoenix;

TEST(Provenance, RoundTripsThroughStrings) {
    const Provenance all[] = {Provenance::Measured,      Provenance::VendorReported,
                              Provenance::Simulated,     Provenance::Analytical,
                              Provenance::Hypothetical,  Provenance::Derived};
    for (auto p : all) {
        const auto s = to_string(p);
        const auto back = provenance_from_string(s);
        ASSERT_TRUE(back.has_value()) << s;
        EXPECT_EQ(*back, p);
    }
}

TEST(Provenance, UnknownStringIsRejectedRatherThanDefaulted) {
    EXPECT_FALSE(provenance_from_string("PROBABLY_MEASURED").has_value());
    EXPECT_FALSE(provenance_from_string("").has_value());
}

TEST(Provenance, NullValueIsLegalAndCarriesItsClass) {
    // Rule 1: the schema must never pressure an author into inventing a number.
    auto v = ProvenanceValue::unknown("TFLOP/s", Provenance::VendorReported);
    EXPECT_FALSE(v.is_known());
    EXPECT_EQ(v.provenance, Provenance::VendorReported);
}

TEST(Validity, FailingMarksInvalidAndRecordsReason) {
    ValidityReport r;
    EXPECT_TRUE(r.valid);
    r.flag(invalidation::kMeasurementFloorRisk);
    EXPECT_TRUE(r.valid) << "a flag must not invalidate a run";
    r.fail(invalidation::kCorrectnessFailed);
    EXPECT_FALSE(r.valid);
    EXPECT_EQ(r.reasons.size(), 2U);
}
