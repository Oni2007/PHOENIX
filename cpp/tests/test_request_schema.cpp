#include <gtest/gtest.h>

#include "phoenix/core/error.hpp"
#include "phoenix/schema/request.hpp"

using namespace phoenix;

namespace {
std::string minimal(const std::string& extra = "") {
    return std::string(R"({"schema_version":"1.0.0","experiment_id":"EXP-001",)") +
           R"("backend":"cpu","implementation":"gemm_naive",)" +
           R"("workload":{"kind":"GEMM","M":4,"N":4,"K":4})" + extra + "}";
}
}  // namespace

TEST(RequestSchema, ParsesMinimalRequest) {
    const auto r = RunRequest::from_json(minimal());
    EXPECT_EQ(r.experiment_id, "EXP-001");
    EXPECT_EQ(r.workload.M, 4);
    EXPECT_EQ(r.workload.dtype_compute, DType::FP64);
}

TEST(RequestSchema, RejectsMalformedJson) {
    EXPECT_THROW((void)RunRequest::from_json("{not json"), ConfigurationError);
}

TEST(RequestSchema, RejectsUnknownSchemaVersion) {
    EXPECT_THROW((void)RunRequest::from_json(
                     R"({"schema_version":"9.9.9","experiment_id":"e","backend":"cpu",)"
                     R"("implementation":"gemm_naive","workload":{"M":1,"N":1,"K":1}})"),
                 ConfigurationError);
}

TEST(RequestSchema, RejectsMissingRequiredField) {
    EXPECT_THROW((void)RunRequest::from_json(
                     R"({"schema_version":"1.0.0","backend":"cpu",)"
                     R"("implementation":"gemm_naive","workload":{"M":1,"N":1,"K":1}})"),
                 ConfigurationError);
}

TEST(RequestSchema, RejectsNonPositiveDimensions) {
    EXPECT_THROW((void)RunRequest::from_json(
                     R"({"schema_version":"1.0.0","experiment_id":"e","backend":"cpu",)"
                     R"("implementation":"gemm_naive","workload":{"M":0,"N":1,"K":1}})"),
                 ConfigurationError);
}

TEST(RequestSchema, RejectsPrecisionOutsideV01Scope) {
    EXPECT_THROW((void)dtype_from_string("fp8_e4m3"), ConfigurationError);
    EXPECT_THROW((void)dtype_from_string("bf16"), ConfigurationError);
}

TEST(ResultSchema, UncheckedCorrectnessSerialisesAsNullNotAsPass) {
    // An unrun check must never serialise as a passing verdict.
    RunResult r;
    r.status = "COMPLETED";
    const auto j = r.to_json();
    EXPECT_NE(j.find("\"checked\":false"), std::string::npos);
    EXPECT_NE(j.find("\"passed\":null"), std::string::npos);
    EXPECT_EQ(j.find("\"passed\":true"), std::string::npos);
}
