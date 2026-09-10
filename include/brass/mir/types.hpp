#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <string>
#include <iosfwd>

namespace brass {

enum class TypeKind : uint8_t {
    I32,
    I64,
    F32,
    F64,
    Ptr,
    GCRef,
    Void,
    F32x4,
    F64x2,
    I32x4,
    I64x2,
    F32x8,
    F64x4,
    I32x8,
    I64x4
};

class Type {
public:
    constexpr Type() noexcept : kind_(TypeKind::Void) {}
    constexpr explicit Type(TypeKind kind) noexcept : kind_(kind) {}

    static constexpr Type i32() noexcept { return Type(TypeKind::I32); }
    static constexpr Type i64() noexcept { return Type(TypeKind::I64); }
    static constexpr Type f32() noexcept { return Type(TypeKind::F32); }
    static constexpr Type f64() noexcept { return Type(TypeKind::F64); }
    static constexpr Type ptr() noexcept { return Type(TypeKind::Ptr); }
    static constexpr Type gcref() noexcept { return Type(TypeKind::GCRef); }
    static constexpr Type void_type() noexcept { return Type(TypeKind::Void); }

    static constexpr Type f32x4() noexcept { return Type(TypeKind::F32x4); }
    static constexpr Type f64x2() noexcept { return Type(TypeKind::F64x2); }
    static constexpr Type i32x4() noexcept { return Type(TypeKind::I32x4); }
    static constexpr Type i64x2() noexcept { return Type(TypeKind::I64x2); }

    static constexpr Type f32x8() noexcept { return Type(TypeKind::F32x8); }
    static constexpr Type f64x4() noexcept { return Type(TypeKind::F64x4); }
    static constexpr Type i32x8() noexcept { return Type(TypeKind::I32x8); }
    static constexpr Type i64x4() noexcept { return Type(TypeKind::I64x4); }

    constexpr TypeKind kind() const noexcept { return kind_; }

    constexpr size_t size_in_bytes() const noexcept {
        switch (kind_) {
            case TypeKind::I32:
            case TypeKind::F32: return 4;
            case TypeKind::I64:
            case TypeKind::F64:
            case TypeKind::Ptr:
            case TypeKind::GCRef: return 8;
            case TypeKind::Void: return 0;
            case TypeKind::F32x4:
            case TypeKind::F64x2:
            case TypeKind::I32x4:
            case TypeKind::I64x2: return 16;
            case TypeKind::F32x8:
            case TypeKind::F64x4:
            case TypeKind::I32x8:
            case TypeKind::I64x4: return 32;
        }
        return 0;
    }

    constexpr bool is_integer() const noexcept {
        return kind_ == TypeKind::I32 || kind_ == TypeKind::I64;
    }

    constexpr bool is_i32() const noexcept {
        return kind_ == TypeKind::I32;
    }

    constexpr bool is_i64() const noexcept {
        return kind_ == TypeKind::I64;
    }

    constexpr bool is_float() const noexcept {
        return kind_ == TypeKind::F32 || kind_ == TypeKind::F64;
    }

    constexpr bool is_numeric() const noexcept {
        return is_integer() || is_float();
    }

    constexpr bool is_pointer() const noexcept {
        return kind_ == TypeKind::Ptr;
    }

    constexpr bool is_gcref() const noexcept {
        return kind_ == TypeKind::GCRef;
    }

    constexpr bool is_pointer_or_gcref() const noexcept {
        return is_pointer() || is_gcref();
    }

    constexpr bool is_void() const noexcept {
        return kind_ == TypeKind::Void;
    }

    constexpr bool is_vector() const noexcept {
        return is_v128() || is_v256();
    }

    constexpr bool is_v128() const noexcept {
        return kind_ == TypeKind::F32x4 || kind_ == TypeKind::F64x2 ||
               kind_ == TypeKind::I32x4 || kind_ == TypeKind::I64x2;
    }

    constexpr bool is_v256() const noexcept {
        return kind_ == TypeKind::F32x8 || kind_ == TypeKind::F64x4 ||
               kind_ == TypeKind::I32x8 || kind_ == TypeKind::I64x4;
    }

    constexpr uint32_t vector_lanes() const noexcept {
        switch (kind_) {
            case TypeKind::F32x4:
            case TypeKind::I32x4:
                return 4;
            case TypeKind::F64x2:
            case TypeKind::I64x2:
                return 2;
            case TypeKind::F32x8:
            case TypeKind::I32x8:
                return 8;
            case TypeKind::F64x4:
            case TypeKind::I64x4:
                return 4;
            default:
                return 0;
        }
    }

    constexpr Type element_type() const noexcept {
        switch (kind_) {
            case TypeKind::F32x4: return Type(TypeKind::F32);
            case TypeKind::F64x2: return Type(TypeKind::F64);
            case TypeKind::I32x4: return Type(TypeKind::I32);
            case TypeKind::I64x2: return Type(TypeKind::I64);
            case TypeKind::F32x8: return Type(TypeKind::F32);
            case TypeKind::F64x4: return Type(TypeKind::F64);
            case TypeKind::I32x8: return Type(TypeKind::I32);
            case TypeKind::I64x4: return Type(TypeKind::I64);
            default: return *this;
        }
    }

    std::string_view name() const noexcept;

    constexpr bool operator==(const Type& other) const noexcept = default;
    constexpr bool operator!=(const Type& other) const noexcept = default;

private:
    TypeKind kind_;
};

std::string to_string(Type t);
std::ostream& operator<<(std::ostream& os, Type t);

} // namespace brass
