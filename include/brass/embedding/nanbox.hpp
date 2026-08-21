#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace brass {

enum class HostValueType : uint8_t {
    Double,
    Int32,
    Bool,
    Null,
    Undefined,
    GCRef,
    Pointer
};

class HostValue {
public:
    // Tag constants (IEEE-754 Quiet NaN payload encoding)
    // Canonical Quiet NaN base: 0x7FF8000000000000ULL
    // Tags occupy the high 16 bits (bits 63..48).
    static constexpr uint64_t TAG_BASE      = 0x7FF8000000000000ULL;
    static constexpr uint64_t TAG_MASK      = 0xFFFF000000000000ULL;
    static constexpr uint64_t PAYLOAD_MASK  = 0x0000FFFFFFFFFFFFULL;

    static constexpr uint64_t TAG_INT32     = 0x7FF9000000000000ULL; // TAG_BASE | (1ULL << 48)
    static constexpr uint64_t TAG_BOOL      = 0x7FFA000000000000ULL; // TAG_BASE | (2ULL << 48)
    static constexpr uint64_t TAG_NULL      = 0x7FFB000000000000ULL; // TAG_BASE | (3ULL << 48)
    static constexpr uint64_t TAG_UNDEFINED = 0x7FFC000000000000ULL; // TAG_BASE | (4ULL << 48)
    static constexpr uint64_t TAG_GCREF     = 0x7FFD000000000000ULL; // TAG_BASE | (5ULL << 48)
    static constexpr uint64_t TAG_POINTER   = 0x7FFE000000000000ULL; // TAG_BASE | (6ULL << 48)

    constexpr HostValue() noexcept : raw_(TAG_UNDEFINED) {}
    constexpr explicit HostValue(uint64_t raw) noexcept : raw_(raw) {}

    // Factory methods
    static inline HostValue from_f64(double val) noexcept {
        uint64_t bits = 0;
        std::memcpy(&bits, &val, sizeof(bits));
        // Canonicalize NaN values to canonical double NaN to prevent tag collisions
        if (std::isnan(val)) {
            bits = TAG_BASE;
        }
        return HostValue(bits);
    }

    static inline HostValue from_double(double val) noexcept {
        return from_f64(val);
    }

    static constexpr HostValue from_i32(int32_t val) noexcept {
        return HostValue(TAG_INT32 | static_cast<uint64_t>(static_cast<uint32_t>(val)));
    }

    static constexpr HostValue from_int32(int32_t val) noexcept {
        return from_i32(val);
    }

    static constexpr HostValue from_bool(bool val) noexcept {
        return HostValue(TAG_BOOL | (val ? 1ULL : 0ULL));
    }

    static constexpr HostValue null_val() noexcept {
        return HostValue(TAG_NULL);
    }

    static constexpr HostValue undefined_val() noexcept {
        return HostValue(TAG_UNDEFINED);
    }

    static constexpr HostValue from_gcref(uintptr_t ptr) noexcept {
        return HostValue(TAG_GCREF | (static_cast<uint64_t>(ptr) & PAYLOAD_MASK));
    }

    static inline HostValue from_gcref(const void* ptr) noexcept {
        return from_gcref(reinterpret_cast<uintptr_t>(ptr));
    }

    static inline HostValue from_pointer(const void* ptr) noexcept {
        return HostValue(TAG_POINTER | (reinterpret_cast<uintptr_t>(ptr) & PAYLOAD_MASK));
    }

    static constexpr HostValue from_raw(uint64_t raw) noexcept {
        return HostValue(raw);
    }

    // Predicates
    [[nodiscard]] constexpr bool is_f64() const noexcept {
        uint64_t tag = raw_ & TAG_MASK;
        return tag < TAG_INT32 || tag > TAG_POINTER;
    }

    [[nodiscard]] constexpr bool is_double() const noexcept {
        return is_f64();
    }

    [[nodiscard]] constexpr bool is_i32() const noexcept {
        return (raw_ & TAG_MASK) == TAG_INT32;
    }

    [[nodiscard]] constexpr bool is_int32() const noexcept {
        return is_i32();
    }

    [[nodiscard]] constexpr bool is_number() const noexcept {
        return is_f64() || is_i32();
    }

    [[nodiscard]] constexpr bool is_bool() const noexcept {
        return (raw_ & TAG_MASK) == TAG_BOOL;
    }

    [[nodiscard]] constexpr bool is_null() const noexcept {
        return (raw_ & TAG_MASK) == TAG_NULL;
    }

    [[nodiscard]] constexpr bool is_undefined() const noexcept {
        return (raw_ & TAG_MASK) == TAG_UNDEFINED;
    }

    [[nodiscard]] constexpr bool is_gcref() const noexcept {
        return (raw_ & TAG_MASK) == TAG_GCREF;
    }

    [[nodiscard]] constexpr bool is_object() const noexcept {
        return is_gcref();
    }

    [[nodiscard]] constexpr bool is_pointer() const noexcept {
        return (raw_ & TAG_MASK) == TAG_POINTER;
    }

    [[nodiscard]] constexpr HostValueType type() const noexcept {
        if (is_i32()) return HostValueType::Int32;
        if (is_bool()) return HostValueType::Bool;
        if (is_null()) return HostValueType::Null;
        if (is_undefined()) return HostValueType::Undefined;
        if (is_gcref()) return HostValueType::GCRef;
        if (is_pointer()) return HostValueType::Pointer;
        return HostValueType::Double;
    }

    // Extraction
    [[nodiscard]] inline double as_f64() const noexcept {
        if (is_i32()) {
            return static_cast<double>(as_i32());
        }
        double val = 0.0;
        std::memcpy(&val, &raw_, sizeof(val));
        return val;
    }

    [[nodiscard]] inline double as_double() const noexcept {
        return as_f64();
    }

    [[nodiscard]] constexpr int32_t as_i32() const noexcept {
        return static_cast<int32_t>(static_cast<uint32_t>(raw_ & 0xFFFFFFFFULL));
    }

    [[nodiscard]] constexpr int32_t as_int32() const noexcept {
        return as_i32();
    }

    [[nodiscard]] constexpr bool as_bool() const noexcept {
        return (raw_ & 1ULL) != 0;
    }

    [[nodiscard]] constexpr uintptr_t as_gcref() const noexcept {
        return static_cast<uintptr_t>(raw_ & PAYLOAD_MASK);
    }

    template <typename T = void>
    [[nodiscard]] inline T* as_gcref_ptr() const noexcept {
        return reinterpret_cast<T*>(as_gcref());
    }

    template <typename T = void>
    [[nodiscard]] inline T* as_pointer() const noexcept {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(raw_ & PAYLOAD_MASK));
    }

    [[nodiscard]] constexpr uint64_t raw() const noexcept {
        return raw_;
    }

    [[nodiscard]] constexpr uint64_t tag() const noexcept {
        return raw_ & TAG_MASK;
    }

    [[nodiscard]] constexpr uint64_t payload() const noexcept {
        return raw_ & PAYLOAD_MASK;
    }

    // In-place pointer update for moving GC relocation
    constexpr void update_gcref(uintptr_t new_ptr) noexcept {
        raw_ = TAG_GCREF | (static_cast<uint64_t>(new_ptr) & PAYLOAD_MASK);
    }

    // Comparison operators
    constexpr bool operator==(const HostValue& other) const noexcept {
        if (raw_ == other.raw_) {
            return true;
        }
        if (is_f64() && other.is_f64()) {
            return as_f64() == other.as_f64();
        }
        return false;
    }

    constexpr bool operator!=(const HostValue& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] inline std::string to_string() const {
        if (is_null()) return "null";
        if (is_undefined()) return "undefined";
        if (is_bool()) return as_bool() ? "true" : "false";
        if (is_i32()) return std::to_string(as_i32());
        if (is_f64()) {
            std::ostringstream ss;
            ss << as_f64();
            return ss.str();
        }
        if (is_gcref()) {
            std::ostringstream ss;
            ss << "gcref(0x" << std::hex << as_gcref() << ")";
            return ss.str();
        }
        if (is_pointer()) {
            std::ostringstream ss;
            ss << "ptr(0x" << std::hex << reinterpret_cast<uintptr_t>(as_pointer()) << ")";
            return ss.str();
        }
        return "unknown";
    }

private:
    uint64_t raw_;
};

inline std::ostream& operator<<(std::ostream& os, HostValueType t) {
    switch (t) {
        case HostValueType::Double: return os << "Double";
        case HostValueType::Int32: return os << "Int32";
        case HostValueType::Bool: return os << "Bool";
        case HostValueType::Null: return os << "Null";
        case HostValueType::Undefined: return os << "Undefined";
        case HostValueType::GCRef: return os << "GCRef";
        case HostValueType::Pointer: return os << "Pointer";
        default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const HostValue& val) {
    return os << val.to_string();
}

} // namespace brass
