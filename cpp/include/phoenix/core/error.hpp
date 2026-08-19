#pragma once
#include <source_location>
#include <stdexcept>
#include <string>

namespace phoenix {

// Three-tier error model (TDD §7.4). No exception crosses the binding boundary;
// the binding layer converts these into a structured result or a Python exception.
class PhoenixError : public std::runtime_error {
public:
    PhoenixError(std::string msg, std::source_location loc = std::source_location::current())
        : std::runtime_error(std::move(msg)), loc_(loc) {}
    [[nodiscard]] const std::source_location& where() const noexcept { return loc_; }
    [[nodiscard]] std::string context() const {
        return std::string(loc_.file_name()) + ":" + std::to_string(loc_.line()) + " in " +
               loc_.function_name();
    }

private:
    std::source_location loc_;
};

// Request was invalid before execution; the run never starts.
class ConfigurationError : public PhoenixError {
public:
    using PhoenixError::PhoenixError;
};

// Backend / allocation failure during the run. Partial samples are retained.
class ExecutionError : public PhoenixError {
public:
    using PhoenixError::PhoenixError;
};

}  // namespace phoenix
