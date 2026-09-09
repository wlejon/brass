#include "sccp_lattice.hpp"
#include <bit>
#include <limits>
#include <cstring>

namespace brass {

LatticeValue evaluate_unary(Opcode op, Type res_type, const LatticeValue& val) {
    if (val.is_top()) {
        return LatticeValue::make_top();
    }
    if (val.is_bottom()) {
        return LatticeValue::make_bottom(res_type);
    }

    switch (op) {
        case Opcode::neg: {
            if (res_type == Type::i32()) {
                uint32_t u = static_cast<uint32_t>(val.as_i32());
                return LatticeValue::make_i32(static_cast<int32_t>(0u - u));
            }
            if (res_type == Type::i64()) {
                uint64_t u = static_cast<uint64_t>(val.as_i64());
                return LatticeValue::make_i64(static_cast<int64_t>(0ULL - u));
            }
            if (res_type == Type::f32()) {
                return LatticeValue::make_f32(-val.as_f32());
            }
            if (res_type == Type::f64()) {
                return LatticeValue::make_f64(-val.as_f64());
            }
            break;
        }

        case Opcode::not_: {
            if (res_type == Type::i32()) {
                return LatticeValue::make_i32(~val.as_i32());
            }
            if (res_type == Type::i64()) {
                return LatticeValue::make_i64(~val.as_i64());
            }
            break;
        }

        case Opcode::clz: {
            if (res_type == Type::i32()) {
                uint32_t u = static_cast<uint32_t>(val.as_i32());
                int32_t c = (u == 0) ? 32 : static_cast<int32_t>(std::countl_zero(u));
                return LatticeValue::make_i32(c);
            }
            if (res_type == Type::i64()) {
                uint64_t u = static_cast<uint64_t>(val.as_i64());
                int64_t c = (u == 0) ? 64 : static_cast<int64_t>(std::countl_zero(u));
                return LatticeValue::make_i64(c);
            }
            break;
        }

        case Opcode::ctz: {
            if (res_type == Type::i32()) {
                uint32_t u = static_cast<uint32_t>(val.as_i32());
                int32_t c = (u == 0) ? 32 : static_cast<int32_t>(std::countr_zero(u));
                return LatticeValue::make_i32(c);
            }
            if (res_type == Type::i64()) {
                uint64_t u = static_cast<uint64_t>(val.as_i64());
                int64_t c = (u == 0) ? 64 : static_cast<int64_t>(std::countr_zero(u));
                return LatticeValue::make_i64(c);
            }
            break;
        }

        case Opcode::popcnt: {
            if (res_type == Type::i32()) {
                uint32_t u = static_cast<uint32_t>(val.as_i32());
                return LatticeValue::make_i32(static_cast<int32_t>(std::popcount(u)));
            }
            if (res_type == Type::i64()) {
                uint64_t u = static_cast<uint64_t>(val.as_i64());
                return LatticeValue::make_i64(static_cast<int64_t>(std::popcount(u)));
            }
            break;
        }

        default:
            break;
    }

    return LatticeValue::make_bottom(res_type);
}

LatticeValue evaluate_binary(Opcode op, Type res_type, const LatticeValue& lhs, const LatticeValue& rhs) {
    bool is_int = (res_type == Type::i32() || res_type == Type::i64() || res_type == Type::ptr());

    // Short circuits for pure integer operations
    if (is_int && op == Opcode::mul) {
        if ((lhs.is_constant() && lhs.is_int_zero()) || (rhs.is_constant() && rhs.is_int_zero())) {
            return (res_type == Type::i32()) ? LatticeValue::make_i32(0) : LatticeValue::make_i64(0);
        }
    }
    if (is_int && op == Opcode::and_) {
        if ((lhs.is_constant() && lhs.is_int_zero()) || (rhs.is_constant() && rhs.is_int_zero())) {
            return (res_type == Type::i32()) ? LatticeValue::make_i32(0) : LatticeValue::make_i64(0);
        }
    }
    if (is_int && op == Opcode::or_) {
        if (lhs.is_constant()) {
            if (res_type == Type::i32() && lhs.as_i32() == -1) return lhs;
            if (res_type == Type::i64() && lhs.as_i64() == -1) return lhs;
        }
        if (rhs.is_constant()) {
            if (res_type == Type::i32() && rhs.as_i32() == -1) return rhs;
            if (res_type == Type::i64() && rhs.as_i64() == -1) return rhs;
        }
    }

    if (lhs.is_top() || rhs.is_top()) {
        return LatticeValue::make_top();
    }
    if (lhs.is_bottom() || rhs.is_bottom()) {
        return LatticeValue::make_bottom(res_type);
    }

    // Both are Constant
    if (lhs.type().is_float() || rhs.type().is_float() || res_type.is_float()) {
        if (lhs.type() == Type::f32() || rhs.type() == Type::f32() || res_type == Type::f32()) {
            float f1 = lhs.as_f32();
            float f2 = rhs.as_f32();
            switch (op) {
                case Opcode::add: return LatticeValue::make_f32(f1 + f2);
                case Opcode::sub: return LatticeValue::make_f32(f1 - f2);
                case Opcode::mul: return LatticeValue::make_f32(f1 * f2);
                case Opcode::sdiv: return LatticeValue::make_f32(f1 / f2);
                case Opcode::eq: return LatticeValue::make_i32(f1 == f2 ? 1 : 0);
                case Opcode::ne: return LatticeValue::make_i32(f1 != f2 ? 1 : 0);
                case Opcode::slt: return LatticeValue::make_i32(f1 < f2 ? 1 : 0);
                case Opcode::sle: return LatticeValue::make_i32(f1 <= f2 ? 1 : 0);
                case Opcode::sgt: return LatticeValue::make_i32(f1 > f2 ? 1 : 0);
                case Opcode::sge: return LatticeValue::make_i32(f1 >= f2 ? 1 : 0);
                default: break;
            }
        } else {
            double d1 = lhs.as_f64();
            double d2 = rhs.as_f64();
            switch (op) {
                case Opcode::add: return LatticeValue::make_f64(d1 + d2);
                case Opcode::sub: return LatticeValue::make_f64(d1 - d2);
                case Opcode::mul: return LatticeValue::make_f64(d1 * d2);
                case Opcode::sdiv: return LatticeValue::make_f64(d1 / d2);
                case Opcode::eq: return LatticeValue::make_i32(d1 == d2 ? 1 : 0);
                case Opcode::ne: return LatticeValue::make_i32(d1 != d2 ? 1 : 0);
                case Opcode::slt: return LatticeValue::make_i32(d1 < d2 ? 1 : 0);
                case Opcode::sle: return LatticeValue::make_i32(d1 <= d2 ? 1 : 0);
                case Opcode::sgt: return LatticeValue::make_i32(d1 > d2 ? 1 : 0);
                case Opcode::sge: return LatticeValue::make_i32(d1 >= d2 ? 1 : 0);
                default: break;
            }
        }
        return LatticeValue::make_bottom(res_type);
    }

    if (res_type == Type::i32()) {
        int32_t a = lhs.as_i32();
        int32_t b = rhs.as_i32();
        uint32_t ua = static_cast<uint32_t>(a);
        uint32_t ub = static_cast<uint32_t>(b);

        switch (op) {
            case Opcode::add:
                return LatticeValue::make_i32(static_cast<int32_t>(ua + ub));
            case Opcode::sub:
                return LatticeValue::make_i32(static_cast<int32_t>(ua - ub));
            case Opcode::mul:
                return LatticeValue::make_i32(static_cast<int32_t>(ua * ub));
            case Opcode::sdiv:
                if (b == 0 || (a == std::numeric_limits<int32_t>::min() && b == -1)) {
                    return LatticeValue::make_bottom(res_type);
                }
                return LatticeValue::make_i32(a / b);
            case Opcode::udiv:
                if (ub == 0) return LatticeValue::make_bottom(res_type);
                return LatticeValue::make_i32(static_cast<int32_t>(ua / ub));
            case Opcode::smod:
                if (b == 0 || (a == std::numeric_limits<int32_t>::min() && b == -1)) {
                    return LatticeValue::make_bottom(res_type);
                }
                return LatticeValue::make_i32(a % b);
            case Opcode::umod:
                if (ub == 0) return LatticeValue::make_bottom(res_type);
                return LatticeValue::make_i32(static_cast<int32_t>(ua % ub));
            case Opcode::and_:
                return LatticeValue::make_i32(a & b);
            case Opcode::or_:
                return LatticeValue::make_i32(a | b);
            case Opcode::xor_:
                return LatticeValue::make_i32(a ^ b);
            case Opcode::shl:
                return LatticeValue::make_i32(static_cast<int32_t>(ua << (ub & 31u)));
            case Opcode::lshr:
                return LatticeValue::make_i32(static_cast<int32_t>(ua >> (ub & 31u)));
            case Opcode::ashr:
                return LatticeValue::make_i32(a >> (ub & 31u));
            case Opcode::eq:
                return LatticeValue::make_i32(a == b ? 1 : 0);
            case Opcode::ne:
                return LatticeValue::make_i32(a != b ? 1 : 0);
            case Opcode::slt:
                return LatticeValue::make_i32(a < b ? 1 : 0);
            case Opcode::sle:
                return LatticeValue::make_i32(a <= b ? 1 : 0);
            case Opcode::sgt:
                return LatticeValue::make_i32(a > b ? 1 : 0);
            case Opcode::sge:
                return LatticeValue::make_i32(a >= b ? 1 : 0);
            case Opcode::ult:
                return LatticeValue::make_i32(ua < ub ? 1 : 0);
            case Opcode::ule:
                return LatticeValue::make_i32(ua <= ub ? 1 : 0);
            case Opcode::ugt:
                return LatticeValue::make_i32(ua > ub ? 1 : 0);
            case Opcode::uge:
                return LatticeValue::make_i32(ua >= ub ? 1 : 0);
            default:
                break;
        }
    } else {
        // 64-bit integers or pointers
        int64_t a = lhs.as_i64();
        int64_t b = rhs.as_i64();
        uint64_t ua = static_cast<uint64_t>(a);
        uint64_t ub = static_cast<uint64_t>(b);

        switch (op) {
            case Opcode::add:
                return LatticeValue::make_i64(static_cast<int64_t>(ua + ub));
            case Opcode::sub:
                return LatticeValue::make_i64(static_cast<int64_t>(ua - ub));
            case Opcode::mul:
                return LatticeValue::make_i64(static_cast<int64_t>(ua * ub));
            case Opcode::sdiv:
                if (b == 0 || (a == std::numeric_limits<int64_t>::min() && b == -1)) {
                    return LatticeValue::make_bottom(res_type);
                }
                return LatticeValue::make_i64(a / b);
            case Opcode::udiv:
                if (ub == 0) return LatticeValue::make_bottom(res_type);
                return LatticeValue::make_i64(static_cast<int64_t>(ua / ub));
            case Opcode::smod:
                if (b == 0 || (a == std::numeric_limits<int64_t>::min() && b == -1)) {
                    return LatticeValue::make_bottom(res_type);
                }
                return LatticeValue::make_i64(a % b);
            case Opcode::umod:
                if (ub == 0) return LatticeValue::make_bottom(res_type);
                return LatticeValue::make_i64(static_cast<int64_t>(ua % ub));
            case Opcode::and_:
                return LatticeValue::make_i64(a & b);
            case Opcode::or_:
                return LatticeValue::make_i64(a | b);
            case Opcode::xor_:
                return LatticeValue::make_i64(a ^ b);
            case Opcode::shl:
                return LatticeValue::make_i64(static_cast<int64_t>(ua << (ub & 63u)));
            case Opcode::lshr:
                return LatticeValue::make_i64(static_cast<int64_t>(ua >> (ub & 63u)));
            case Opcode::ashr:
                return LatticeValue::make_i64(a >> (ub & 63u));
            case Opcode::eq:
                return LatticeValue::make_i32(a == b ? 1 : 0);
            case Opcode::ne:
                return LatticeValue::make_i32(a != b ? 1 : 0);
            case Opcode::slt:
                return LatticeValue::make_i32(a < b ? 1 : 0);
            case Opcode::sle:
                return LatticeValue::make_i32(a <= b ? 1 : 0);
            case Opcode::sgt:
                return LatticeValue::make_i32(a > b ? 1 : 0);
            case Opcode::sge:
                return LatticeValue::make_i32(a >= b ? 1 : 0);
            case Opcode::ult:
                return LatticeValue::make_i32(ua < ub ? 1 : 0);
            case Opcode::ule:
                return LatticeValue::make_i32(ua <= ub ? 1 : 0);
            case Opcode::ugt:
                return LatticeValue::make_i32(ua > ub ? 1 : 0);
            case Opcode::uge:
                return LatticeValue::make_i32(ua >= ub ? 1 : 0);
            default:
                break;
        }
    }

    return LatticeValue::make_bottom(res_type);
}

LatticeValue evaluate_conversion(Opcode op, Type res_type, const LatticeValue& val) {
    if (val.is_top()) {
        return LatticeValue::make_top();
    }
    if (val.is_bottom()) {
        return LatticeValue::make_bottom(res_type);
    }

    switch (op) {
        case Opcode::sext_i64:
            return LatticeValue::make_i64(static_cast<int64_t>(val.as_i32()));
        case Opcode::zext_i64:
            return LatticeValue::make_i64(static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(val.as_i32()))));
        case Opcode::trunc_i32:
            return LatticeValue::make_i32(static_cast<int32_t>(val.as_i64()));
        case Opcode::fptosi_i32:
            return LatticeValue::make_i32(static_cast<int32_t>(val.as_f64()));
        case Opcode::fptosi_i64:
            return LatticeValue::make_i64(static_cast<int64_t>(val.as_f64()));
        case Opcode::sitofp_f64_i32:
            return LatticeValue::make_f64(static_cast<double>(val.as_i32()));
        case Opcode::sitofp_f64_i64:
            return LatticeValue::make_f64(static_cast<double>(val.as_i64()));
        case Opcode::bitcast_i64_f64: {
            double d = val.as_f64();
            int64_t i = 0;
            std::memcpy(&i, &d, sizeof(int64_t));
            return LatticeValue::make_i64(i);
        }
        case Opcode::bitcast_f64_i64: {
            int64_t i = val.as_i64();
            double d = 0.0;
            std::memcpy(&d, &i, sizeof(double));
            return LatticeValue::make_f64(d);
        }
        default:
            break;
    }

    return LatticeValue::make_bottom(res_type);
}

LatticeValue evaluate_select(const LatticeValue& cond, const LatticeValue& tv, const LatticeValue& fv) {
    if (cond.is_constant()) {
        return cond.is_int_zero() ? fv : tv;
    }
    if (cond.is_bottom()) {
        return tv.meet(fv);
    }
    // cond is Top
    if (tv == fv) {
        return tv;
    }
    return LatticeValue::make_top();
}

} // namespace brass
