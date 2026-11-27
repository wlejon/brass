#include <brass/mir/types.hpp>
#include <brass/mir/half_float.hpp>
#include <ostream>

namespace brass {

extern "C" {
float brass_f16_to_f32(uint32_t bits) noexcept { return f16_bits_to_f32(static_cast<uint16_t>(bits)); }
uint32_t brass_f32_to_f16(float val) noexcept { return f32_to_f16_bits(val); }
float brass_bf16_to_f32(uint32_t bits) noexcept { return bf16_bits_to_f32(static_cast<uint16_t>(bits)); }
uint32_t brass_f32_to_bf16(float val) noexcept { return f32_to_bf16_bits(val); }
}

std::string_view Type::name() const noexcept {
    switch (kind_) {
        case TypeKind::I8: return "i8";
        case TypeKind::I16: return "i16";
        case TypeKind::I32: return "i32";
        case TypeKind::I64: return "i64";
        case TypeKind::F16: return "f16";
        case TypeKind::BF16: return "bf16";
        case TypeKind::F32: return "f32";
        case TypeKind::F64: return "f64";
        case TypeKind::Ptr: return "ptr";
        case TypeKind::GCRef: return "gcref";
        case TypeKind::Void: return "void";
        case TypeKind::F32x4: return "f32x4";
        case TypeKind::F64x2: return "f64x2";
        case TypeKind::I32x4: return "i32x4";
        case TypeKind::I64x2: return "i64x2";
        case TypeKind::F32x8: return "f32x8";
        case TypeKind::F64x4: return "f64x4";
        case TypeKind::I32x8: return "i32x8";
        case TypeKind::I64x4: return "i64x4";
        case TypeKind::Tagged: return "tagged";
    }
    return "unknown";
}

std::string to_string(Type t) {
    return std::string(t.name());
}

std::ostream& operator<<(std::ostream& os, Type t) {
    return os << t.name();
}

} // namespace brass
