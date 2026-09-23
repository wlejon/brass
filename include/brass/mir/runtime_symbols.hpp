#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

// The runtime-symbol registry: what calls to a declared external function
// mean to the optimizer. A frontend that owns a runtime function's contract
// declares its roles on the module (Module::add_symbol_role, printed as
// `extern @sym role...`); passes ask the declaration and never match a
// symbol's name, so the same passes serve any frontend that declares its
// runtime.
namespace brass {

class Instruction;
class Module;

enum class SymbolRole : uint8_t {
    // `allocator`: returns a fresh object no other pointer refers to and
    // touches no memory the caller can see.
    Allocator,
    // `pure`: no side effects, reads no memory a store can change, never
    // reaches a GC point and cannot trap; its result depends only on its
    // arguments (and the calling thread), so calls may be merged, hoisted and
    // executed speculatively.
    Pure,
    // `array_new`: (length) -> a fresh runtime array of `length` elements.
    ArrayNew,
    // `array_get`: (array, index) -> the element at `index`.
    ArrayGet,
    // `array_set`: (array, index, value, ...) stores `value` at `index`.
    ArraySet,
    // `frem`: (f64, f64) -> f64, the remainder with C fmod semantics.
    FloatRem,
};

inline constexpr SymbolRole kAllSymbolRoles[] = {
    SymbolRole::Allocator, SymbolRole::Pure, SymbolRole::ArrayNew,
    SymbolRole::ArrayGet, SymbolRole::ArraySet, SymbolRole::FloatRem,
};

// The keyword the printer writes and the parser reads for `role`.
std::string_view symbol_role_name(SymbolRole role) noexcept;
std::optional<SymbolRole> parse_symbol_role(std::string_view text) noexcept;

// The module `inst` belongs to, or null for a detached instruction.
const Module* module_of(const Instruction& inst) noexcept;

// True when `inst` is a plain direct call (Opcode::call) whose callee its
// module declares with `role`.
bool callee_has_role(const Instruction& inst, SymbolRole role) noexcept;

} // namespace brass
