#include "int_fold.hpp"
#include <bit>

namespace brass::int_fold {

unsigned width_of(Type t) noexcept {
    if (t == Type::i32()) return 32;
    if (t == Type::i64() || t.is_pointer_or_gcref()) return 64;
    return 0;
}

int64_t canonical(int64_t v, unsigned width) noexcept {
    if (width == 32) return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(v))));
    return v;
}

namespace {

uint64_t mask_of(unsigned width) noexcept {
    return width >= 64 ? ~0ULL : ((1ULL << width) - 1ULL);
}

int64_t from_bits(uint64_t bits, unsigned width) noexcept {
    return canonical(static_cast<int64_t>(bits & mask_of(width)), width);
}

int64_t min_signed(unsigned width) noexcept {
    return width == 32 ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
}

} // namespace

bool division_may_trap(Opcode op, unsigned width, int64_t divisor) noexcept {
    switch (op) {
        case Opcode::sdiv:
        case Opcode::smod:
            return canonical(divisor, width) == 0 || canonical(divisor, width) == -1;
        case Opcode::udiv:
        case Opcode::umod:
            return canonical(divisor, width) == 0;
        default:
            return false;
    }
}

std::optional<int64_t> binary(Opcode op, unsigned width, int64_t a_in, int64_t b_in) noexcept {
    if (width != 32 && width != 64) return std::nullopt;
    const int64_t a = canonical(a_in, width);
    const int64_t b = canonical(b_in, width);
    const uint64_t m = mask_of(width);
    const uint64_t ua = static_cast<uint64_t>(a) & m;
    const uint64_t ub = static_cast<uint64_t>(b) & m;
    const uint64_t shift = ub & static_cast<uint64_t>(width - 1);

    switch (op) {
        case Opcode::add: return from_bits(ua + ub, width);
        case Opcode::sub: return from_bits(ua - ub, width);
        case Opcode::mul: return from_bits(ua * ub, width);
        case Opcode::and_: return from_bits(ua & ub, width);
        case Opcode::or_: return from_bits(ua | ub, width);
        case Opcode::xor_: return from_bits(ua ^ ub, width);
        case Opcode::shl: return from_bits(ua << shift, width);
        case Opcode::lshr: return from_bits(ua >> shift, width);
        case Opcode::ashr: {
            // Arithmetic shift of the sign-extended value is exact at any width.
            return canonical(a >> shift, width);
        }
        case Opcode::sdiv:
        case Opcode::smod:
            if (b == 0 || (a == min_signed(width) && b == -1)) return std::nullopt;
            return canonical(op == Opcode::sdiv ? a / b : a % b, width);
        case Opcode::udiv:
        case Opcode::umod:
            if (ub == 0) return std::nullopt;
            return from_bits(op == Opcode::udiv ? ua / ub : ua % ub, width);
        case Opcode::eq: return a == b ? 1 : 0;
        case Opcode::ne: return a != b ? 1 : 0;
        case Opcode::slt: return a < b ? 1 : 0;
        case Opcode::sle: return a <= b ? 1 : 0;
        case Opcode::sgt: return a > b ? 1 : 0;
        case Opcode::sge: return a >= b ? 1 : 0;
        case Opcode::ult: return ua < ub ? 1 : 0;
        case Opcode::ule: return ua <= ub ? 1 : 0;
        case Opcode::ugt: return ua > ub ? 1 : 0;
        case Opcode::uge: return ua >= ub ? 1 : 0;
        default: return std::nullopt;
    }
}

std::optional<int64_t> unary(Opcode op, unsigned width, int64_t a_in) noexcept {
    if (width != 32 && width != 64) return std::nullopt;
    const uint64_t ua = static_cast<uint64_t>(a_in) & mask_of(width);
    switch (op) {
        case Opcode::neg: return from_bits(0ULL - ua, width);
        case Opcode::not_: return from_bits(~ua, width);
        case Opcode::clz:
            return width == 32 ? static_cast<int64_t>(std::countl_zero(static_cast<uint32_t>(ua)))
                               : static_cast<int64_t>(std::countl_zero(ua));
        case Opcode::ctz:
            return width == 32 ? static_cast<int64_t>(std::countr_zero(static_cast<uint32_t>(ua)))
                               : static_cast<int64_t>(std::countr_zero(ua));
        case Opcode::popcnt: return static_cast<int64_t>(std::popcount(ua));
        default: return std::nullopt;
    }
}

std::optional<int64_t> convert(Opcode op, Type src, int64_t a) noexcept {
    switch (op) {
        case Opcode::sext_i64:
            if (src == Type::i32()) return canonical(a, 32);
            return std::nullopt;
        case Opcode::zext_i64:
            if (src == Type::i32()) return static_cast<int64_t>(static_cast<uint64_t>(a) & 0xFFFFFFFFULL);
            return std::nullopt;
        case Opcode::trunc_i32:
            if (src == Type::i64() || src == Type::i32()) return canonical(a, 32);
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

} // namespace brass::int_fold
