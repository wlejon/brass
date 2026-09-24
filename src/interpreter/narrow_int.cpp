#include <brass/interpreter/narrow_int.hpp>
#include <stdexcept>
#include <string>

namespace brass {

bool eval_narrow_int(Opcode op, unsigned bits, const RuntimeValue* operands, size_t count, RuntimeValue& out) {
    if (bits != 8 && bits != 16) return false;
    const uint64_t raw_a = count > 0 ? operands[0].raw_bits() : 0;
    const uint64_t raw_b = count > 1 ? operands[1].raw_bits() : 0;
    const uint32_t a = narrow_zext(raw_a, bits);
    const uint32_t b = narrow_zext(raw_b, bits);
    const int32_t sa = narrow_sext(raw_a, bits);
    const int32_t sb = narrow_sext(raw_b, bits);
    const int64_t smin = -(int64_t{1} << (bits - 1));
    const int64_t smax = (int64_t{1} << (bits - 1)) - 1;
    const int64_t umax = (int64_t{1} << bits) - 1;

    auto value = [&](uint64_t r) { out = RuntimeValue::from_i32(static_cast<int32_t>(narrow_zext(r, bits))); };
    auto flag = [&](bool f) { out = RuntimeValue::from_i32(f ? 1 : 0); };
    auto divisor = [&](const char* what) {
        if (b == 0) {
            throw std::runtime_error(std::string("Interpreter error: Division by zero (i") + std::to_string(bits) +
                                     " " + what + ")");
        }
    };
    // Shift amounts of the width or more give an unspecified result
    // (docs/semantics.md); the interpreters shift the widened value.
    const uint32_t amt = b & 31u;

    switch (op) {
        case Opcode::add: value(uint64_t{a} + b); return true;
        case Opcode::sub: value(uint64_t{a} - b); return true;
        case Opcode::mul: value(uint64_t{a} * b); return true;
        case Opcode::neg: value(0 - uint64_t{a}); return true;
        case Opcode::and_: value(a & b); return true;
        case Opcode::or_: value(a | b); return true;
        case Opcode::xor_: value(a ^ b); return true;
        case Opcode::not_: value(~uint64_t{a}); return true;
        case Opcode::shl: value(uint64_t{a} << amt); return true;
        case Opcode::lshr: value(a >> amt); return true;
        case Opcode::ashr: value(static_cast<uint64_t>(static_cast<int64_t>(sa >> amt))); return true;
        case Opcode::sdiv: divisor("sdiv"); value(static_cast<uint64_t>(int64_t{sa} / sb)); return true;
        case Opcode::smod: divisor("smod"); value(static_cast<uint64_t>(int64_t{sa} % sb)); return true;
        case Opcode::udiv: divisor("udiv"); value(a / b); return true;
        case Opcode::umod: divisor("umod"); value(a % b); return true;
        case Opcode::clz: {
            unsigned n = 0;
            for (int i = static_cast<int>(bits) - 1; i >= 0 && !((a >> i) & 1u); --i) ++n;
            value(n);
            return true;
        }
        case Opcode::ctz: {
            unsigned n = 0;
            while (n < bits && !((a >> n) & 1u)) ++n;
            value(n);
            return true;
        }
        case Opcode::popcnt: {
            unsigned n = 0;
            for (uint32_t x = a; x; x &= x - 1) ++n;
            value(n);
            return true;
        }
        case Opcode::eq: flag(a == b); return true;
        case Opcode::ne: flag(a != b); return true;
        case Opcode::slt: flag(sa < sb); return true;
        case Opcode::sle: flag(sa <= sb); return true;
        case Opcode::sgt: flag(sa > sb); return true;
        case Opcode::sge: flag(sa >= sb); return true;
        case Opcode::ult: flag(a < b); return true;
        case Opcode::ule: flag(a <= b); return true;
        case Opcode::ugt: flag(a > b); return true;
        case Opcode::uge: flag(a >= b); return true;
        case Opcode::sadd_overflow: { const int64_t r = int64_t{sa} + sb; flag(r < smin || r > smax); return true; }
        case Opcode::ssub_overflow: { const int64_t r = int64_t{sa} - sb; flag(r < smin || r > smax); return true; }
        case Opcode::smul_overflow: { const int64_t r = int64_t{sa} * sb; flag(r < smin || r > smax); return true; }
        case Opcode::uadd_overflow: flag(int64_t{a} + b > umax); return true;
        case Opcode::usub_overflow: flag(a < b); return true;
        case Opcode::umul_overflow: flag(int64_t{a} * b > umax); return true;
        default: return false;
    }
}

} // namespace brass
