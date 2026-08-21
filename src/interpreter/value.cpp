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
    }
    return "<unknown>";
}

std::ostream& operator<<(std::ostream& os, const RuntimeValue& val) {
    return os << to_string(val);
}

} // namespace brass
