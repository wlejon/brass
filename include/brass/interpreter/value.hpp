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
    F32,
    F64,
    Ptr,
    GCRef,
    F32x4,
    F64x2,
    I32x4,
    I64x2
};

class RuntimeValue {
public:
    constexpr RuntimeValue() noexcept : kind_(RuntimeValueKind::Void), raw_bits_(0), v128_bytes_{} {}

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

    static RuntimeValue from_f32(float val) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::F32;
        std::memcpy(&v.raw_bits_, &val, sizeof(float));
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

    static RuntimeValue from_f32x4(float f0, float f1, float f2, float f3) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::F32x4;
        float arr[4] = {f0, f1, f2, f3};
        std::memcpy(v.v128_bytes_, arr, 16);
        return v;
    }

    static RuntimeValue from_f64x2(double d0, double d1) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::F64x2;
        double arr[2] = {d0, d1};
        std::memcpy(v.v128_bytes_, arr, 16);
        return v;
    }

    static RuntimeValue from_i32x4(int32_t i0, int32_t i1, int32_t i2, int32_t i3) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I32x4;
        int32_t arr[4] = {i0, i1, i2, i3};
        std::memcpy(v.v128_bytes_, arr, 16);
        return v;
    }

    static RuntimeValue from_i64x2(int64_t l0, int64_t l1) noexcept {
        RuntimeValue v;
        v.kind_ = RuntimeValueKind::I64x2;
        int64_t arr[2] = {l0, l1};
        std::memcpy(v.v128_bytes_, arr, 16);
        return v;
    }

    static RuntimeValue from_v128(Type type, const void* bytes) noexcept {
        RuntimeValue v;
        switch (type.kind()) {
            case TypeKind::F32x4: v.kind_ = RuntimeValueKind::F32x4; break;
            case TypeKind::F64x2: v.kind_ = RuntimeValueKind::F64x2; break;
            case TypeKind::I32x4: v.kind_ = RuntimeValueKind::I32x4; break;
            case TypeKind::I64x2: v.kind_ = RuntimeValueKind::I64x2; break;
            default: v.kind_ = RuntimeValueKind::I32x4; break;
        }
        if (bytes) {
            std::memcpy(v.v128_bytes_, bytes, 16);
        }
        return v;
    }

    static RuntimeValue from_bits(Type type, uint64_t bits) noexcept {
        RuntimeValue v;
        v.raw_bits_ = bits;
        switch (type.kind()) {
            case TypeKind::I32: v.kind_ = RuntimeValueKind::I32; break;
            case TypeKind::I64: v.kind_ = RuntimeValueKind::I64; break;
            case TypeKind::F32: v.kind_ = RuntimeValueKind::F32; break;
            case TypeKind::F64: v.kind_ = RuntimeValueKind::F64; break;
            case TypeKind::Ptr: v.kind_ = RuntimeValueKind::Ptr; break;
            case TypeKind::GCRef: v.kind_ = RuntimeValueKind::GCRef; break;
            case TypeKind::Void: v.kind_ = RuntimeValueKind::Void; break;
            case TypeKind::F32x4: v.kind_ = RuntimeValueKind::F32x4; std::memcpy(v.v128_bytes_, &bits, 8); break;
            case TypeKind::F64x2: v.kind_ = RuntimeValueKind::F64x2; std::memcpy(v.v128_bytes_, &bits, 8); break;
            case TypeKind::I32x4: v.kind_ = RuntimeValueKind::I32x4; std::memcpy(v.v128_bytes_, &bits, 8); break;
            case TypeKind::I64x2: v.kind_ = RuntimeValueKind::I64x2; std::memcpy(v.v128_bytes_, &bits, 8); break;
        }
        return v;
    }

    constexpr RuntimeValueKind kind() const noexcept { return kind_; }

    constexpr Type type() const noexcept {
        switch (kind_) {
            case RuntimeValueKind::I32: return Type::i32();
            case RuntimeValueKind::I64: return Type::i64();
            case RuntimeValueKind::F32: return Type::f32();
            case RuntimeValueKind::F64: return Type::f64();
            case RuntimeValueKind::Ptr: return Type::ptr();
            case RuntimeValueKind::GCRef: return Type::gcref();
            case RuntimeValueKind::Void: return Type::void_type();
            case RuntimeValueKind::F32x4: return Type::f32x4();
            case RuntimeValueKind::F64x2: return Type::f64x2();
            case RuntimeValueKind::I32x4: return Type::i32x4();
            case RuntimeValueKind::I64x2: return Type::i64x2();
        }
        return Type::void_type();
    }

    constexpr bool is_void() const noexcept { return kind_ == RuntimeValueKind::Void; }
    constexpr bool is_i32() const noexcept { return kind_ == RuntimeValueKind::I32; }
    constexpr bool is_i64() const noexcept { return kind_ == RuntimeValueKind::I64; }
    constexpr bool is_f32() const noexcept { return kind_ == RuntimeValueKind::F32; }
    constexpr bool is_f64() const noexcept { return kind_ == RuntimeValueKind::F64; }
    constexpr bool is_ptr() const noexcept { return kind_ == RuntimeValueKind::Ptr; }
    constexpr bool is_gcref() const noexcept { return kind_ == RuntimeValueKind::GCRef; }
    constexpr bool is_integer() const noexcept { return is_i32() || is_i64(); }
    constexpr bool is_pointer_or_gcref() const noexcept { return is_ptr() || is_gcref(); }
    constexpr bool is_vector() const noexcept {
        return kind_ == RuntimeValueKind::F32x4 || kind_ == RuntimeValueKind::F64x2 ||
               kind_ == RuntimeValueKind::I32x4 || kind_ == RuntimeValueKind::I64x2;
    }
    constexpr bool is_f32x4() const noexcept { return kind_ == RuntimeValueKind::F32x4; }
    constexpr bool is_f64x2() const noexcept { return kind_ == RuntimeValueKind::F64x2; }
    constexpr bool is_i32x4() const noexcept { return kind_ == RuntimeValueKind::I32x4; }
    constexpr bool is_i64x2() const noexcept { return kind_ == RuntimeValueKind::I64x2; }
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

    float as_f32() const noexcept {
        float f = 0.0f;
        std::memcpy(&f, &raw_bits_, sizeof(float));
        return f;
    }

    double as_f64() const noexcept {
        double d = 0.0;
        std::memcpy(&d, &raw_bits_, sizeof(double));
        return d;
    }

    float f32_lane(size_t idx) const noexcept {
        float f = 0.0f;
        if (idx < 4) {
            std::memcpy(&f, v128_bytes_ + idx * sizeof(float), sizeof(float));
        }
        return f;
    }

    double f64_lane(size_t idx) const noexcept {
        double d = 0.0;
        if (idx < 2) {
            std::memcpy(&d, v128_bytes_ + idx * sizeof(double), sizeof(double));
        }
        return d;
    }

    int32_t i32_lane(size_t idx) const noexcept {
        int32_t v = 0;
        if (idx < 4) {
            std::memcpy(&v, v128_bytes_ + idx * sizeof(int32_t), sizeof(int32_t));
        }
        return v;
    }

    int64_t i64_lane(size_t idx) const noexcept {
        int64_t v = 0;
        if (idx < 2) {
            std::memcpy(&v, v128_bytes_ + idx * sizeof(int64_t), sizeof(int64_t));
        }
        return v;
    }

    const uint8_t* v128_bytes() const noexcept { return v128_bytes_; }
    uint8_t* v128_bytes_mut() noexcept { return v128_bytes_; }

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

    bool operator==(const RuntimeValue& other) const noexcept {
        if (kind_ != other.kind_) return false;
        if (is_vector()) {
            return std::memcmp(v128_bytes_, other.v128_bytes_, 16) == 0;
        }
        return raw_bits_ == other.raw_bits_;
    }

    bool operator!=(const RuntimeValue& other) const noexcept {
        return !(*this == other);
    }

private:
    RuntimeValueKind kind_ = RuntimeValueKind::Void;
    uint64_t raw_bits_ = 0;
    alignas(16) uint8_t v128_bytes_[16];
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

// Vector Operations
RuntimeValue val_vadd(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vsub(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vmul(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vdiv(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vneg(RuntimeValue val);
RuntimeValue val_vmin(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vmax(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vsqrt(RuntimeValue val);

RuntimeValue val_vand(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vor(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vxor(RuntimeValue lhs, RuntimeValue rhs);
RuntimeValue val_vnot(RuntimeValue val);

RuntimeValue val_vbroadcast(Type vec_type, RuntimeValue val);
RuntimeValue val_vextract_lane(RuntimeValue val, uint32_t lane);
RuntimeValue val_vinsert_lane(RuntimeValue vec, RuntimeValue scalar, uint32_t lane);
RuntimeValue val_vshuffle(RuntimeValue v1, RuntimeValue v2, uint32_t mask);
RuntimeValue val_vzero(Type vec_type);

std::string to_string(const RuntimeValue& val);
std::ostream& operator<<(std::ostream& os, const RuntimeValue& val);

namespace interpreter {
    using Value = brass::RuntimeValue;
}

} // namespace brass
