#include "phoenix/core/status.hpp"

namespace phoenix {

std::string_view to_string(RunStatus s) noexcept {
    switch (s) {
        case RunStatus::Completed: return "COMPLETED";
        case RunStatus::Failed: return "FAILED";
        case RunStatus::Invalid: return "INVALID";
    }
    return "FAILED";
}

}  // namespace phoenix
