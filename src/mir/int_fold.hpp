#pragma once

#include <brass/mir/opcodes.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>
#include <optional>

// The one integer constant folder the MIR optimizer uses. Every pass that
// evaluates an integer operation at compile time goes through here so the
// answer matches what the interpreter and the JIT compute at run time.
//
// Values travel as int64_t in the canonical form the rest of MIR uses: a
// 32-bit value is sign-extended. Results come back in the same form.
namespace brass::int_fold {

// Bit width the folder evaluates an operation of this operand type at:
// 32 for i32, 64 for i64 and pointers, 0 for anything it does not fold.
unsigned width_of(Type t) noexcept;

// Brings `v` to the canonical form of a `width`-bit value.
int64_t canonical(int64_t v, unsigned width) noexcept;

// Two-operand integer operations and comparisons at `width` bits.
// Comparisons yield 0 or 1. Returns nullopt when the operation has no single
// defined result to fold to: division or remainder by zero. Signed division
// of the minimum value by -1 wraps to the minimum value (remainder 0), as
// docs/semantics.md defines it.
std::optional<int64_t> binary(Opcode op, unsigned width, int64_t a, int64_t b) noexcept;

// neg, not, clz, ctz, popcnt at `width` bits.
std::optional<int64_t> unary(Opcode op, unsigned width, int64_t a) noexcept;

// sext_i64, zext_i64 and trunc_i32 of a constant of type `src`.
std::optional<int64_t> convert(Opcode op, Type src, int64_t a) noexcept;

// True when evaluating `op` with divisor `divisor` can trap, so the
// instruction must not be executed speculatively.
bool division_may_trap(Opcode op, unsigned width, int64_t divisor) noexcept;

} // namespace brass::int_fold
