#pragma once

#include <string_view>

namespace brass {

constexpr int version_major() noexcept { return 0; }
constexpr int version_minor() noexcept { return 1; }
constexpr int version_patch() noexcept { return 0; }
constexpr std::string_view version_string() noexcept { return "0.1.0"; }

} // namespace brass
