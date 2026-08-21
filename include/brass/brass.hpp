#pragma once

#include <brass/core/arena.hpp>
#include <brass/core/bitset.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/core/span.hpp>
#include <brass/core/string_pool.hpp>

#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>

#include <string_view>

namespace brass {

constexpr int version_major() noexcept { return 0; }
constexpr int version_minor() noexcept { return 1; }
constexpr int version_patch() noexcept { return 0; }
constexpr std::string_view version_string() noexcept { return "0.1.0"; }

} // namespace brass
