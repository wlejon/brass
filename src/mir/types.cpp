#include <brass/mir/types.hpp>
#include <ostream>

namespace brass {

std::string_view Type::name() const noexcept {
    switch (kind_) {
        case TypeKind::I32: return "i32";
        case TypeKind::I64: return "i64";
        case TypeKind::F64: return "f64";
        case TypeKind::Ptr: return "ptr";
        case TypeKind::GCRef: return "gcref";
        case TypeKind::Void: return "void";
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
