#pragma once

#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <functional>
#include <iosfwd>
#include <cstring>

namespace brass::runtime {

enum class DeoptReason : uint32_t {
    None = 0,
    Generic = 1,
    TypeCheckFailed = 2,
    Overflow = 3,
    ShapeCheckFailed = 4,
    BoundsCheckFailed = 5,
    DivisionByZero = 6,
    NullCheckFailed = 7,
    Custom = 8
};

std::string_view to_string(DeoptReason reason) noexcept;
std::ostream& operator<<(std::ostream& os, DeoptReason reason);

enum class DeoptValueKind : uint8_t {
    Int32 = 0,
    Int64 = 1,
    Float64 = 2,
    Pointer = 3,
    GcRef = 4
};

std::string_view to_string(DeoptValueKind kind) noexcept;
std::ostream& operator<<(std::ostream& os, DeoptValueKind kind);

struct DeoptValue {
    DeoptValueKind kind = DeoptValueKind::Int64;
    uint64_t raw = 0;

    constexpr DeoptValue() noexcept = default;
    constexpr DeoptValue(DeoptValueKind k, uint64_t v) noexcept : kind(k), raw(v) {}

    static constexpr DeoptValue i32(int32_t v) noexcept {
        return DeoptValue(DeoptValueKind::Int32, static_cast<uint64_t>(static_cast<uint32_t>(v)));
    }

    static constexpr DeoptValue i64(int64_t v) noexcept {
        return DeoptValue(DeoptValueKind::Int64, static_cast<uint64_t>(v));
    }

    static DeoptValue f64(double v) noexcept {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(double));
        return DeoptValue(DeoptValueKind::Float64, bits);
    }

    static constexpr DeoptValue ptr(uintptr_t v) noexcept {
        return DeoptValue(DeoptValueKind::Pointer, static_cast<uint64_t>(v));
    }

    static constexpr DeoptValue gcref(uintptr_t v) noexcept {
        return DeoptValue(DeoptValueKind::GcRef, static_cast<uint64_t>(v));
    }

    int32_t as_i32() const noexcept {
        return static_cast<int32_t>(static_cast<uint32_t>(raw & 0xFFFFFFFFULL));
    }

    int64_t as_i64() const noexcept {
        return static_cast<int64_t>(raw);
    }

    double as_f64() const noexcept {
        double d = 0.0;
        std::memcpy(&d, &raw, sizeof(double));
        return d;
    }

    uintptr_t as_ptr() const noexcept {
        return static_cast<uintptr_t>(raw);
    }

    uintptr_t as_gcref() const noexcept {
        return static_cast<uintptr_t>(raw);
    }

    RuntimeValue to_runtime_value() const;
    static DeoptValue from_runtime_value(const RuntimeValue& rv);

    constexpr bool operator==(const DeoptValue& other) const noexcept = default;
    constexpr bool operator!=(const DeoptValue& other) const noexcept = default;
};

class DeoptFrame {
public:
    static constexpr size_t kMaxSlots = 64;

    uint32_t resume_id = 0;
    DeoptReason reason = DeoptReason::Generic;
    void* target_fn = nullptr;
    std::string exit_symbol;
    size_t count = 0;
    std::array<uint64_t, kMaxSlots> slots{};
    std::array<DeoptValueKind, kMaxSlots> kinds{};

    DeoptFrame() noexcept {
        slots.fill(0);
        kinds.fill(DeoptValueKind::Int64);
    }

    void clear() noexcept {
        resume_id = 0;
        reason = DeoptReason::None;
        target_fn = nullptr;
        exit_symbol.clear();
        count = 0;
        slots.fill(0);
        kinds.fill(DeoptValueKind::Int64);
    }

    void push_value(uint64_t val, DeoptValueKind kind = DeoptValueKind::Int64) noexcept {
        if (count < kMaxSlots) {
            slots[count] = val;
            kinds[count] = kind;
            count++;
        }
    }

    void push_deopt_value(const DeoptValue& dv) noexcept {
        push_value(dv.raw, dv.kind);
    }

    void set_value(size_t idx, const DeoptValue& dv) noexcept {
        if (idx < kMaxSlots) {
            slots[idx] = dv.raw;
            kinds[idx] = dv.kind;
            if (idx >= count) {
                count = idx + 1;
            }
        }
    }

    DeoptValue get_value(size_t idx) const noexcept {
        if (idx < count) {
            return DeoptValue(kinds[idx], slots[idx]);
        }
        return DeoptValue();
    }

    std::vector<DeoptValue> values() const {
        std::vector<DeoptValue> res;
        res.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            res.push_back(DeoptValue(kinds[i], slots[i]));
        }
        return res;
    }

    std::vector<RuntimeValue> to_runtime_values() const;
};
} // namespace brass::runtime

extern "C" {
    brass::runtime::DeoptFrame* brass_get_thread_deopt_frame();
    void brass_set_thread_deopt_frame(const brass::runtime::DeoptFrame* frame);
    void* brass_deopt_exit(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots);
}

namespace brass::runtime {

using ::brass_get_thread_deopt_frame;
using ::brass_set_thread_deopt_frame;
using ::brass_deopt_exit;

using DeoptHandlerFn = std::function<void*(const DeoptFrame&)>;

DeoptFrame* get_thread_deopt_frame() noexcept;
void set_thread_deopt_frame(const DeoptFrame* frame) noexcept;
void register_deopt_handler(DeoptHandlerFn handler);
DeoptHandlerFn get_deopt_handler();

} // namespace brass::runtime
