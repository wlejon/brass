#include <brass/interpreter/value.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace brass {

RuntimeValue val_add(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_f64(lhs.as_f64() + rhs.as_f64());
    }
    if (lhs.is_f32() || rhs.is_f32()) {
        return RuntimeValue::from_f32(lhs.as_f32() + rhs.as_f32());
    }
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32();
        uint32_t b = rhs.as_u32();
        return RuntimeValue::from_u32(a + b);
    }
    if (lhs.is_ptr() || lhs.is_gcref()) {
        uint64_t a = lhs.raw_bits();
        uint64_t b = rhs.raw_bits();
        return lhs.is_ptr() ? RuntimeValue::from_ptr(static_cast<uintptr_t>(a + b))
                            : RuntimeValue::from_gcref(static_cast<uintptr_t>(a + b));
    }
    uint64_t a = lhs.as_u64();
    uint64_t b = rhs.as_u64();
    return RuntimeValue::from_u64(a + b);
}

RuntimeValue val_sub(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_f64(lhs.as_f64() - rhs.as_f64());
    }
    if (lhs.is_f32() || rhs.is_f32()) {
        return RuntimeValue::from_f32(lhs.as_f32() - rhs.as_f32());
    }
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32();
        uint32_t b = rhs.as_u32();
        return RuntimeValue::from_u32(a - b);
    }
    if (lhs.is_ptr() || lhs.is_gcref()) {
        uint64_t a = lhs.raw_bits();
        uint64_t b = rhs.raw_bits();
        return lhs.is_ptr() ? RuntimeValue::from_ptr(static_cast<uintptr_t>(a - b))
                            : RuntimeValue::from_gcref(static_cast<uintptr_t>(a - b));
    }
    uint64_t a = lhs.as_u64();
    uint64_t b = rhs.as_u64();
    return RuntimeValue::from_u64(a - b);
}

RuntimeValue val_mul(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_f64(lhs.as_f64() * rhs.as_f64());
    }
    if (lhs.is_f32() || rhs.is_f32()) {
        return RuntimeValue::from_f32(lhs.as_f32() * rhs.as_f32());
    }
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32();
        uint32_t b = rhs.as_u32();
        return RuntimeValue::from_u32(a * b);
    }
    uint64_t a = lhs.as_u64();
    uint64_t b = rhs.as_u64();
    return RuntimeValue::from_u64(a * b);
}

RuntimeValue val_sdiv(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_f64(lhs.as_f64() / rhs.as_f64());
    }
    if (lhs.is_f32() || rhs.is_f32()) {
        return RuntimeValue::from_f32(lhs.as_f32() / rhs.as_f32());
    }
    if (lhs.is_i32()) {
        int32_t b = rhs.as_i32();
        if (b == 0) {
            throw std::runtime_error("Interpreter error: Division by zero (i32 sdiv)");
        }
        int32_t a = lhs.as_i32();
        if (a == std::numeric_limits<int32_t>::min() && b == -1) {
            return RuntimeValue::from_i32(std::numeric_limits<int32_t>::min());
        }
        return RuntimeValue::from_i32(a / b);
    }
    int64_t b = rhs.as_i64();
    if (b == 0) {
        throw std::runtime_error("Interpreter error: Division by zero (i64 sdiv)");
    }
    int64_t a = lhs.as_i64();
    if (a == std::numeric_limits<int64_t>::min() && b == -1) {
        return RuntimeValue::from_i64(std::numeric_limits<int64_t>::min());
    }
    return RuntimeValue::from_i64(a / b);
}

RuntimeValue val_udiv(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t b = rhs.as_u32();
        if (b == 0) {
            throw std::runtime_error("Interpreter error: Division by zero (i32 udiv)");
        }
        return RuntimeValue::from_u32(lhs.as_u32() / b);
    }
    uint64_t b = rhs.as_u64();
    if (b == 0) {
        throw std::runtime_error("Interpreter error: Division by zero (i64 udiv)");
    }
    return RuntimeValue::from_u64(lhs.as_u64() / b);
}

RuntimeValue val_smod(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_f64(std::fmod(lhs.as_f64(), rhs.as_f64()));
    }
    if (lhs.is_i32()) {
        int32_t b = rhs.as_i32();
        if (b == 0) {
            throw std::runtime_error("Interpreter error: Modulo by zero (i32 smod)");
        }
        int32_t a = lhs.as_i32();
        if (a == std::numeric_limits<int32_t>::min() && b == -1) {
            return RuntimeValue::from_i32(0);
        }
        return RuntimeValue::from_i32(a % b);
    }
    int64_t b = rhs.as_i64();
    if (b == 0) {
        throw std::runtime_error("Interpreter error: Modulo by zero (i64 smod)");
    }
    int64_t a = lhs.as_i64();
    if (a == std::numeric_limits<int64_t>::min() && b == -1) {
        return RuntimeValue::from_i64(0);
    }
    return RuntimeValue::from_i64(a % b);
}

RuntimeValue val_umod(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t b = rhs.as_u32();
        if (b == 0) {
            throw std::runtime_error("Interpreter error: Modulo by zero (i32 umod)");
        }
        return RuntimeValue::from_u32(lhs.as_u32() % b);
    }
    uint64_t b = rhs.as_u64();
    if (b == 0) {
        throw std::runtime_error("Interpreter error: Modulo by zero (i64 umod)");
    }
    return RuntimeValue::from_u64(lhs.as_u64() % b);
}

RuntimeValue val_neg(RuntimeValue val) {
    if (val.is_f64()) {
        return RuntimeValue::from_f64(-val.as_f64());
    }
    if (val.is_f32()) {
        return RuntimeValue::from_f32(-val.as_f32());
    }
    if (val.is_i32()) {
        uint32_t a = val.as_u32();
        return RuntimeValue::from_u32(~a + 1U);
    }
    uint64_t a = val.as_u64();
    return RuntimeValue::from_u64(~a + 1ULL);
}

RuntimeValue val_and(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_u32(lhs.as_u32() & rhs.as_u32());
    }
    return RuntimeValue::from_u64(lhs.as_u64() & rhs.as_u64());
}

RuntimeValue val_or(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_u32(lhs.as_u32() | rhs.as_u32());
    }
    return RuntimeValue::from_u64(lhs.as_u64() | rhs.as_u64());
}

RuntimeValue val_xor(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_u32(lhs.as_u32() ^ rhs.as_u32());
    }
    return RuntimeValue::from_u64(lhs.as_u64() ^ rhs.as_u64());
}

RuntimeValue val_shl(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t shift = rhs.as_u32() & 31U;
        return RuntimeValue::from_u32(lhs.as_u32() << shift);
    }
    uint64_t shift = rhs.as_u64() & 63ULL;
    return RuntimeValue::from_u64(lhs.as_u64() << shift);
}

RuntimeValue val_lshr(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t shift = rhs.as_u32() & 31U;
        return RuntimeValue::from_u32(lhs.as_u32() >> shift);
    }
    uint64_t shift = rhs.as_u64() & 63ULL;
    return RuntimeValue::from_u64(lhs.as_u64() >> shift);
}

RuntimeValue val_ashr(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t shift = rhs.as_u32() & 31U;
        return RuntimeValue::from_i32(lhs.as_i32() >> shift);
    }
    uint64_t shift = rhs.as_u64() & 63ULL;
    return RuntimeValue::from_i64(lhs.as_i64() >> shift);
}

RuntimeValue val_not(RuntimeValue val) {
    if (val.is_i32()) {
        return RuntimeValue::from_u32(~val.as_u32());
    }
    return RuntimeValue::from_u64(~val.as_u64());
}

RuntimeValue val_clz(RuntimeValue val) {
    if (val.is_i32()) {
        return RuntimeValue::from_i32(static_cast<int32_t>(std::countl_zero(val.as_u32())));
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(std::countl_zero(val.as_u64())));
}

RuntimeValue val_ctz(RuntimeValue val) {
    if (val.is_i32()) {
        return RuntimeValue::from_i32(static_cast<int32_t>(std::countr_zero(val.as_u32())));
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(std::countr_zero(val.as_u64())));
}

RuntimeValue val_popcnt(RuntimeValue val) {
    if (val.is_i32()) {
        return RuntimeValue::from_i32(static_cast<int32_t>(std::popcount(val.as_u32())));
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(std::popcount(val.as_u64())));
}

RuntimeValue val_eq(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() == rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32() && rhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() == rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.raw_bits() == rhs.raw_bits() ? 1 : 0);
}

RuntimeValue val_ne(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() != rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32() && rhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() != rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.raw_bits() != rhs.raw_bits() ? 1 : 0);
}

RuntimeValue val_slt(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() < rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() < rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_i64() < rhs.as_i64() ? 1 : 0);
}

RuntimeValue val_ult(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_u32() < rhs.as_u32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_u64() < rhs.as_u64() ? 1 : 0);
}

RuntimeValue val_sle(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() <= rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() <= rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_i64() <= rhs.as_i64() ? 1 : 0);
}

RuntimeValue val_ule(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_u32() <= rhs.as_u32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_u64() <= rhs.as_u64() ? 1 : 0);
}

RuntimeValue val_sgt(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() > rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() > rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_i64() > rhs.as_i64() ? 1 : 0);
}

RuntimeValue val_ugt(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_u32() > rhs.as_u32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_u64() > rhs.as_u64() ? 1 : 0);
}

RuntimeValue val_sge(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_f64() || rhs.is_f64()) {
        return RuntimeValue::from_i32(lhs.as_f64() >= rhs.as_f64() ? 1 : 0);
    }
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_i32() >= rhs.as_i32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_i64() >= rhs.as_i64() ? 1 : 0);
}

RuntimeValue val_uge(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        return RuntimeValue::from_i32(lhs.as_u32() >= rhs.as_u32() ? 1 : 0);
    }
    return RuntimeValue::from_i32(lhs.as_u64() >= rhs.as_u64() ? 1 : 0);
}

RuntimeValue val_sadd_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        int32_t a = lhs.as_i32(), b = rhs.as_i32();
        int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
        bool ovf = (sum < INT32_MIN || sum > INT32_MAX);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    int64_t a = lhs.as_i64(), b = rhs.as_i64();
    bool ovf = ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b));
    return RuntimeValue::from_i32(ovf ? 1 : 0);
}

RuntimeValue val_ssub_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        int32_t a = lhs.as_i32(), b = rhs.as_i32();
        int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
        bool ovf = (diff < INT32_MIN || diff > INT32_MAX);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    int64_t a = lhs.as_i64(), b = rhs.as_i64();
    bool ovf = ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b));
    return RuntimeValue::from_i32(ovf ? 1 : 0);
}

RuntimeValue val_smul_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        int32_t a = lhs.as_i32(), b = rhs.as_i32();
        int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
        bool ovf = (prod < INT32_MIN || prod > INT32_MAX);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    int64_t a = lhs.as_i64(), b = rhs.as_i64();
#if defined(__GNUC__) || defined(__clang__)
    int64_t res = 0;
    bool ovf = __builtin_mul_overflow(a, b, &res);
    return RuntimeValue::from_i32(ovf ? 1 : 0);
#else
    if (a == 0 || b == 0) return RuntimeValue::from_i32(0);
    if (a == -1 && b == INT64_MIN) return RuntimeValue::from_i32(1);
    if (b == -1 && a == INT64_MIN) return RuntimeValue::from_i32(1);
    if (a > 0 && b > 0 && a > INT64_MAX / b) return RuntimeValue::from_i32(1);
    if (a > 0 && b < 0 && b < INT64_MIN / a) return RuntimeValue::from_i32(1);
    if (a < 0 && b > 0 && a < INT64_MIN / b) return RuntimeValue::from_i32(1);
    if (a < 0 && b < 0 && a < INT64_MAX / b) return RuntimeValue::from_i32(1);
    return RuntimeValue::from_i32(0);
#endif
}

RuntimeValue val_uadd_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32(), b = rhs.as_u32();
        bool ovf = (a + b < a);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    uint64_t a = lhs.as_u64(), b = rhs.as_u64();
    bool ovf = (a + b < a);
    return RuntimeValue::from_i32(ovf ? 1 : 0);
}

RuntimeValue val_usub_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32(), b = rhs.as_u32();
        bool ovf = (a < b);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    uint64_t a = lhs.as_u64(), b = rhs.as_u64();
    bool ovf = (a < b);
    return RuntimeValue::from_i32(ovf ? 1 : 0);
}

RuntimeValue val_umul_overflow(RuntimeValue lhs, RuntimeValue rhs) {
    if (lhs.is_i32()) {
        uint32_t a = lhs.as_u32(), b = rhs.as_u32();
        uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
        bool ovf = (prod > UINT32_MAX);
        return RuntimeValue::from_i32(ovf ? 1 : 0);
    }
    uint64_t a = lhs.as_u64(), b = rhs.as_u64();
    if (a == 0 || b == 0) return RuntimeValue::from_i32(0);
    uint64_t prod = a * b;
    bool ovf = (prod / a != b);
    return RuntimeValue::from_i32(ovf ? 1 : 0);
}

std::string to_string(const RuntimeValue& val) {
    std::ostringstream ss;
    switch (val.kind()) {
        case RuntimeValueKind::Void:
            return "void";
        case RuntimeValueKind::I32:
            return std::to_string(val.as_i32());
        case RuntimeValueKind::I64:
            return std::to_string(val.as_i64());
        case RuntimeValueKind::F64: {
            double d = val.as_f64();
            if (std::isnan(d)) return "nan";
            if (std::isinf(d)) return (d < 0) ? "-inf" : "inf";
            ss << std::defaultfloat << d;
            std::string s = ss.str();
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
                s += ".0";
            }
            return s;
        }
        case RuntimeValueKind::F32: {
            float f = val.as_f32();
            if (std::isnan(f)) return "nan";
            if (std::isinf(f)) return (f < 0) ? "-inf" : "inf";
            ss << std::defaultfloat << f;
            std::string s = ss.str();
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
                s += ".0";
            }
            return s;
        }
        case RuntimeValueKind::Ptr: {
            ss << "0x" << std::hex << val.as_ptr();
            return ss.str();
        }
        case RuntimeValueKind::GCRef: {
            if (val.is_null()) {
                return "gcref(null)";
            }
            ss << "gcref(0x" << std::hex << val.as_gcref() << ")";
            return ss.str();
        }
        case RuntimeValueKind::F32x4: {
            ss << "<";
            for (size_t i = 0; i < 4; ++i) {
                if (i > 0) ss << ", ";
                float f = val.f32_lane(i);
                if (std::isnan(f)) ss << "nan";
                else if (std::isinf(f)) ss << ((f < 0) ? "-inf" : "inf");
                else {
                    std::ostringstream elem;
                    elem << std::defaultfloat << f;
                    std::string s = elem.str();
                    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
                    ss << s;
                }
            }
            ss << ">";
            return ss.str();
        }
        case RuntimeValueKind::F64x2: {
            ss << "<";
            for (size_t i = 0; i < 2; ++i) {
                if (i > 0) ss << ", ";
                double d = val.f64_lane(i);
                if (std::isnan(d)) ss << "nan";
                else if (std::isinf(d)) ss << ((d < 0) ? "-inf" : "inf");
                else {
                    std::ostringstream elem;
                    elem << std::defaultfloat << d;
                    std::string s = elem.str();
                    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
                    ss << s;
                }
            }
            ss << ">";
            return ss.str();
        }
        case RuntimeValueKind::I32x4: {
            ss << "<" << val.i32_lane(0) << ", " << val.i32_lane(1) << ", "
               << val.i32_lane(2) << ", " << val.i32_lane(3) << ">";
            return ss.str();
        }
        case RuntimeValueKind::I64x2: {
            ss << "<" << val.i64_lane(0) << ", " << val.i64_lane(1) << ">";
            return ss.str();
        }
    }
    return "<unknown>";
}

std::ostream& operator<<(std::ostream& os, const RuntimeValue& val) {
    return os << to_string(val);
}

} // namespace brass
