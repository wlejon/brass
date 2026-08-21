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
    F64,
    Ptr,
    GCRef,
    Void
};

class Type {
public:
    constexpr Type() noexcept : kind_(TypeKind::Void) {}
    constexpr explicit Type(TypeKind kind) noexcept : kind_(kind) {}

    static constexpr Type i32() noexcept { return Type(TypeKind::I32); }
    static constexpr Type i64() noexcept { return Type(TypeKind::I64); }
    static constexpr Type f64() noexcept { return Type(TypeKind::F64); }
    static constexpr Type ptr() noexcept { return Type(TypeKind::Ptr); }
    static constexpr Type gcref() noexcept { return Type(TypeKind::GCRef); }
    static constexpr Type void_type() noexcept { return Type(TypeKind::Void); }

    constexpr TypeKind kind() const noexcept { return kind_; }

    constexpr size_t size_in_bytes() const noexcept {
        switch (kind_) {
            case TypeKind::I32: return 4;
            case TypeKind::I64:
            case TypeKind::F64:
            case TypeKind::Ptr:
            case TypeKind::GCRef: return 8;
            case TypeKind::Void: return 0;
        }
        return 0;
    }

    constexpr bool is_integer() const noexcept {
        return kind_ == TypeKind::I32 || kind_ == TypeKind::I64;
    }

    constexpr bool is_float() const noexcept {
        return kind_ == TypeKind::F64;
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

    std::string_view name() const noexcept;

    constexpr bool operator==(const Type& other) const noexcept = default;
    constexpr bool operator!=(const Type& other) const noexcept = default;

private:
    TypeKind kind_;
};

std::string to_string(Type t);
std::ostream& operator<<(std::ostream& os, Type t);

} // namespace brass
