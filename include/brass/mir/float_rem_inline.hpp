#pragma once

#include <brass/mir/function.hpp>
#include <cstddef>

namespace brass {

// Gives every call to a `frem` runtime symbol (C fmod semantics) an inline
// integer path: when both operands are integers a 64-bit integer holds and
// the divisor is neither 0 nor -1, the remainder is the integer `srem`, with
// the dividend's sign put on a zero result; every other operand pair takes
// the call. Runs after f64 demotion, which turns the `frem` of a loop whose
// numbers it proves integral into a plain `smod` and leaves the rest.
// Returns how many calls it expanded.
size_t inline_float_rem(Function& fn);

} // namespace brass
