#include "sccp_lattice.hpp"
#include "int_fold.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <cstring>

namespace brass {

namespace {

// Integer lattice constants keep 32-bit values sign-extended, which is the
// representation int_fold works in.
int64_t int_bits(const LatticeValue& v) {
    return v.type() == Type::i32() ? static_cast<int64_t>(v.as_i32()) : v.as_i64();
}

LatticeValue make_int(unsigned width, int64_t v) {
    return width == 32 ? LatticeValue::make_i32(static_cast<int32_t>(v)) : LatticeValue::make_i64(v);
}

} // namespace

LatticeValue evaluate_unary(Opcode op, Type res_type, const LatticeValue& val) {
    if (val.is_top()) {
        return LatticeValue::make_top();
    }
    if (val.is_bottom()) {
        return LatticeValue::make_bottom(res_type);
    }

    switch (op) {
        case Opcode::neg:
            if (res_type == Type::f32()) return LatticeValue::make_f32(-val.as_f32());
            if (res_type == Type::f64()) return LatticeValue::make_f64(-val.as_f64());
            [[fallthrough]];
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt: {
            const unsigned width = (res_type == Type::i32() || res_type == Type::i64()) ? int_fold::width_of(res_type) : 0;
            if (auto r = int_fold::unary(op, width, int_bits(val))) return make_int(width, *r);
            break;
        }

        case Opcode::sqrt_f32:
            return LatticeValue::make_f32(std::sqrt(val.as_f32()));
        case Opcode::sqrt_f64:
            return LatticeValue::make_f64(std::sqrt(val.as_f64()));
        case Opcode::floor_f32:
            return LatticeValue::make_f32(std::floor(val.as_f32()));
        case Opcode::floor_f64:
            return LatticeValue::make_f64(std::floor(val.as_f64()));
        case Opcode::ceil_f32:
            return LatticeValue::make_f32(std::ceil(val.as_f32()));
        case Opcode::ceil_f64:
            return LatticeValue::make_f64(std::ceil(val.as_f64()));
        case Opcode::round_f32:
            return LatticeValue::make_f32(std::round(val.as_f32()));
        case Opcode::round_f64:
            return LatticeValue::make_f64(std::round(val.as_f64()));
        case Opcode::fabs_f32:
            return LatticeValue::make_f32(std::fabs(val.as_f32()));
        case Opcode::fabs_f64:
            return LatticeValue::make_f64(std::fabs(val.as_f64()));

        default:
            break;
    }

    return LatticeValue::make_bottom(res_type);
}

LatticeValue evaluate_binary(Opcode op, Type res_type, const LatticeValue& lhs, const LatticeValue& rhs) {
    // Pointer arithmetic is never folded: its result must stay a pointer,
    // and an integer constant cannot stand in for one.
    if (res_type.is_pointer_or_gcref()) {
        if (lhs.is_top() || rhs.is_top()) return LatticeValue::make_top();
        return LatticeValue::make_bottom(res_type);
    }
    bool is_int = (res_type == Type::i32() || res_type == Type::i64());

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
                case Opcode::fmin_f32: return LatticeValue::make_f32(std::fmin(f1, f2));
                case Opcode::fmax_f32: return LatticeValue::make_f32(std::fmax(f1, f2));
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
                case Opcode::fmin_f64: return LatticeValue::make_f64(std::fmin(d1, d2));
                case Opcode::fmax_f64: return LatticeValue::make_f64(std::fmax(d1, d2));
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

    // Integer operations evaluate at the operands' width; comparisons yield i32.
    Type op_type = (lhs.type() != Type::void_type()) ? lhs.type() : res_type;
    const unsigned width = (op_type == Type::i32()) ? 32U : 64U;
    if (auto r = int_fold::binary(op, width, int_bits(lhs), int_bits(rhs))) {
        if (is_comparison(op)) return LatticeValue::make_i32(static_cast<int32_t>(*r));
        return make_int(width, *r);
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
        case Opcode::zext_i64:
        case Opcode::trunc_i32: {
            if (auto r = int_fold::convert(op, val.type(), int_bits(val))) {
                return op == Opcode::trunc_i32 ? LatticeValue::make_i32(static_cast<int32_t>(*r))
                                               : LatticeValue::make_i64(*r);
            }
            break;
        }
        case Opcode::fptosi_i32: {
            double d = val.as_f64();
            if (std::isnan(d) || d < static_cast<double>(INT32_MIN) || d > static_cast<double>(INT32_MAX)) {
                return LatticeValue::make_bottom(res_type);
            }
            return LatticeValue::make_i32(static_cast<int32_t>(d));
        }
        case Opcode::fptosi_i32_f32: {
            float f = val.as_f32();
            if (std::isnan(f) || f < static_cast<float>(INT32_MIN) || f >= 2147483648.0f) {
                return LatticeValue::make_bottom(res_type);
            }
            return LatticeValue::make_i32(static_cast<int32_t>(f));
        }
        case Opcode::fptosi_i64: {
            double d = val.as_f64();
            if (std::isnan(d) || d < -9223372036854775808.0 || d >= 9223372036854775808.0) {
                return LatticeValue::make_bottom(res_type);
            }
            return LatticeValue::make_i64(static_cast<int64_t>(d));
        }
        case Opcode::fptosi_i64_f32: {
            float f = val.as_f32();
            if (std::isnan(f) || f < -9223372036854775808.0f || f >= 9223372036854775808.0f) {
                return LatticeValue::make_bottom(res_type);
            }
            return LatticeValue::make_i64(static_cast<int64_t>(f));
        }
        case Opcode::sitofp_f64_i32:
            return LatticeValue::make_f64(static_cast<double>(val.as_i32()));
        case Opcode::sitofp_f32_i32:
            return LatticeValue::make_f32(static_cast<float>(val.as_i32()));
        case Opcode::sitofp_f64_i64:
            return LatticeValue::make_f64(static_cast<double>(val.as_i64()));
        case Opcode::sitofp_f32_i64:
            return LatticeValue::make_f32(static_cast<float>(val.as_i64()));
        case Opcode::fptrunc_f32_f64:
            return LatticeValue::make_f32(static_cast<float>(val.as_f64()));
        case Opcode::fpext_f64_f32:
            return LatticeValue::make_f64(static_cast<double>(val.as_f32()));
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
