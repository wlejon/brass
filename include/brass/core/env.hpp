#pragma once

#include <string>

namespace brass {

// The environment variable's value, empty when it is unset. MSVC deprecates
// getenv (C4996, an error under /WX), so this is the one place that reads it.
std::string env_value(const char* name);

} // namespace brass
