#pragma once

#include <brass/mir/types.hpp>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <iosfwd>
#include <bit>

namespace brass {

enum class RuntimeValueKind : uint8_t {
    Void = 0,
    I32,
    I64,
    F64,
    Ptr,
    GCRef
};

class RuntimeValue {
public:
    constexpr RuntimeValue() noexcept : kind_(RuntimeValueKind::Void), raw_bits_(0) {}

    static constexpr RuntimeValue from_i32(int32_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I32;
        v.raw_bits_ = static_cast<uint64_t>(static_cast<uint32_t>(val));
        return v;
    }

    static constexpr RuntimeValue from_u32(uint32_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I32;
        v.raw_bits_ = static_cast<uint64_t>(val);
        return v;
    }

    static constexpr RuntimeValue from_i64(int64_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I64;
        v.raw_bits_ = static_cast<uint64_t>(val);
        return v;
    }

    static constexpr RuntimeValue from_u64(uint64_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I64;
        v.raw_bits_ = val;
        return v;
    }

    static RuntimeValue from_f64(double val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::F64;
        std::memcpy(&v.raw_bits_, &val, sizeof(double));
        return v;
    }

    static constexpr RuntimeValue from_ptr(uintptr_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::Ptr;
        v.raw_bits_ = static_cast<uint64_t>(val);
        return v;
    }

    static constexpr RuntimeValue from_ptr(const void* val) noexcept {
        return from_ptr(reinterpret_cast<uintptr_t>(val));
    }

    static constexpr RuntimeValue from_gcref(uintptr_t val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::GCRef;
        v.raw_bits_ = static_cast<uint64_t>(val);
        return v;
    }

    static constexpr RuntimeValue from_void() noexcept {
        return RuntimeValue();
    }

    static RuntimeValue from_bits(Type type, uint64_t bits) noexcept {
        RuntimeValue v;
        v.raw_bits_ = bits;
        switch (type.kind()) {
            case TypeKind::I32: v.kind_ = RuntimeValueKind::I32; break;
            case TypeKind::I64: v.kind_ = RuntimeValueKind::I64; break;
            case TypeKind::F64: v.kind_ = RuntimeValueKind::F64; break;
            case TypeKind::Ptr: v.kind_ = RuntimeValueKind::Ptr; break;
            case TypeKind::GCRef: v.kind_ = RuntimeValueKind::GCRef; break;
            case TypeKind::Void: v.kind_ = RuntimeValueKind::Void; break;
        }
        return v;
    }

    constexpr RuntimeValueKind kind() const noexcept { return kind_; }

    constexpr Type type() const noexcept {
        switch (kind_) {
            case RuntimeValueKind::I32: return Type::i32();
            case RuntimeValueKind::I64: return Type::i64();
            case RuntimeValueKind::F64: return Type::f64();
            case RuntimeValueKind::Ptr: return Type::ptr();
            case RuntimeValueKind::GCRef: return Type::gcref();
            case RuntimeValueKind::Void: return Type::void_type();
        }
        return Type::void_type();
    }

    constexpr bool is_void() const noexcept { return kind_ == RuntimeValueKind::Void; }
    constexpr bool is_i32() const noexcept { return kind_ == RuntimeValueKind::I32; }
    constexpr bool is_i64() const noexcept { return kind_ == RuntimeValueKind::I64; }
    constexpr bool is_f64() const noexcept { return kind_ == RuntimeValueKind::F64; }
    constexpr bool is_ptr() const noexcept { return kind_ == RuntimeValueKind::Ptr; }
    constexpr bool is_gcref() const noexcept { return kind_ == RuntimeValueKind::GCRef; }
    constexpr bool is_integer() const noexcept { return is_i32() || is_i64(); }
    constexpr bool is_pointer_or_gcref() const noexcept { return is_ptr() || is_gcref(); }
    constexpr bool is_null() const noexcept { return raw_bits_ == 0; }

    int32_t as_i32() const noexcept {
        return static_cast<int32_t>(static_cast<uint32_t>(raw_bits_ & 0xFFFFFFFFULL));
    }

    uint32_t as_u32() const noexcept {
        return static_cast<uint32_t>(raw_bits_ & 0xFFFFFFFFULL);
    }

    int64_t as_i64() const noexcept {
        return static_cast<int64_t>(raw_bits_);
    }

    uint64_t as_u64() const noexcept {
        return raw_bits_;
    }

    double as_f64() const noexcept {
        double d = 0.0;
        std::memcpy(&d, &raw_bits_, sizeof(double));
        return d;
    }

    uintptr_t as_ptr() const noexcept {
        return static_cast<uintptr_t>(raw_bits_);
    }

    template <typename T>
    T* as_native_ptr() const noexcept {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(raw_bits_));
    }

    uintptr_t as_gcref() const noexcept {
        return static_cast<uintptr_t>(raw_bits_);
    }

    void set_gcref(uintptr_t addr) noexcept {
        kind_ = RuntimeValueKind::GCRef;
        raw_bits_ = static_cast<uint64_t>(addr);
    }

    uint64_t raw_bits() const noexcept { return raw_bits_; }
    uint64_t& raw_bits_ref() noexcept { return raw_bits_; }

    constexpr bool operator==(const RuntimeValue& other) const noexcept {
        return kind_ == other.kind_ && raw_bits_ == other.raw_bits_;
    }

    constexpr bool operator!=(const RuntimeValue& other) const noexcept {
        return !(*this == other);
    }

private:
    RuntimeValueKind kind_ = RuntimeValueKind::Void;
    uint64_t raw_bits_ = 0;
};

// Conversions
inline RuntimeValue val_sext_i64(RuntimeValue v) noexcept {
    return RuntimeValue::from_i64(static_cast<int64_t>(v.as_i32()));
}

inline RuntimeValue val_zext_i64(RuntimeValue v) noexcept {
    return RuntimeValue::from_u64(static_cast<uint64_t>(v.as_u32()));
}

inline RuntimeValue val_trunc_i32(RuntimeValue v) noexcept {
    return RuntimeValue::from_i32(static_cast<int32_t>(v.as_i64()));
}

inline RuntimeValue val_fptosi_i32(RuntimeValue v) noexcept {
    return RuntimeValue::from_i32(static_cast<int32_t>(v.as_f64()));
}

inline RuntimeValue val_fptosi_i64(RuntimeValue v) noexcept {
    return RuntimeValue::from_i64(static_cast<int64_t>(v.as_f64()));
}

inline RuntimeValue val_sitofp_f64_i32(RuntimeValue v) noexcept {
    return RuntimeValue::from_f64(static_cast<double>(v.as_i32()));
}

inline RuntimeValue val_sitofp_f64_i64(RuntimeValue v) noexcept {
    return RuntimeValue::from_f64(static_cast<double>(v.as_i64()));
}

inline RuntimeValue val_bitcast_i64_f64(RuntimeValue v) noexcept {
    return RuntimeValue::from_i64(static_cast<int64_t>(v.raw_bits()));
}

inline RuntimeValue val_bitcast_f64_i64(RuntimeValue v) noexcept {
    RuntimeValue res;
    double d = 0.0;
    uint64_t b = v.raw_bits();
    std::memcpy(&d, &b, sizeof(double));
    return RuntimeValue::from_f64(d);
}

// Arithmetic & Bitwise
RuntimeValue val_add(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_sub(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_mul(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_sdiv(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_udiv(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_smod(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_umod(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_neg(RuntimeValue val);

RuntimeValue val_and(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_or(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_xor(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_shl(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_lshr(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ashr(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_not(RuntimeValue val);
RuntimeValue val_clz(RuntimeValue val);
RuntimeValue val_ctz(RuntimeValue val);
RuntimeValue val_popcnt(RuntimeValue val);

// Comparisons (returns i32: 1 for true, 0 for false)
RuntimeValue val_eq(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ne(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_slt(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ult(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_sle(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ule(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_sgt(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ugt(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_sge(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_uge(RuntimeValue lhs, RuntimeValue rhs);

// Overflow-checked Arithmetic (returns i32: 1 on overflow, 0 on no overflow)
RuntimeValue val_sadd_overflow(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_ssub_overflow(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_smul_overflow(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_uadd_overflow(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_usub_overflow(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_umul_overflow(RuntimeValue lhs, RuntimeValue rhs);

std::string to_string(const RuntimeValue& val);
std::ostream& operator<<(std::ostream& os, const RuntimeValue& val);

namespace interpreter {
    using Value = brass::RuntimeValue;
}

} // namespace brass
