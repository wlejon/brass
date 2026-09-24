#pragma once

// i8 / i16 integer semantics for the interpreters (docs/semantics.md,
// "Narrow integers"). An iN value is an N-bit two's-complement pattern.
// The interpreters hold it zero-extended in an i32 RuntimeValue, but read
// only its low N bits, so a value that arrives with other high bits (a
// native return, a deopt slot) means the same thing.

#include <brass/interpreter/value.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>

namespace brass {

// 8 or 16 for i8 / i16, otherwise 0.
constexpr unsigned narrow_int_bits(Type t) noexcept {
    return t == Type::i8() ? 8u : t == Type::i16() ? 16u : 0u;
}

constexpr uint32_t narrow_zext(uint64_t v, unsigned bits) noexcept {
    return static_cast<uint32_t>(v & ((uint64_t{1} << bits) - 1));
}

constexpr int32_t narrow_sext(uint64_t v, unsigned bits) noexcept {
    const uint32_t z = narrow_zext(v, bits);
    const uint32_t sign = uint32_t{1} << (bits - 1);
    return static_cast<int32_t>((z ^ sign) - sign);
}

// Evaluates integer opcode `op` whose operands (or result) are narrow:
// `bits` is the operand width (the result width too, unless the result
// is an i32 comparison or overflow flag). Returns false when `op` is not
// an integer operation this covers. Division by zero throws, as at i32.
bool eval_narrow_int(Opcode op, unsigned bits, const RuntimeValue* operands, size_t count, RuntimeValue& out);

// The Interpreter's step for an integer instruction on narrow operands:
// true when `inst` was one and its result is set in `frame`.
template <class Frame>
bool interp_narrow_step(const Instruction& inst, Frame& frame) {
    const size_t n = inst.operand_count();
    if (n == 0 || n > 2 || !inst.operand(0) || !inst.result()) return false;
    const unsigned bits = narrow_int_bits(inst.operand(0)->type());
    if (!bits) return false;
    RuntimeValue ops[2] = {frame.get_value(inst.operand(0)),
                           n > 1 ? frame.get_value(inst.operand(1)) : RuntimeValue::from_i32(0)};
    RuntimeValue res;
    if (!eval_narrow_int(inst.opcode(), bits, ops, n, res)) return false;
    frame.set_value(inst.result(), res);
    return true;
}

} // namespace brass
