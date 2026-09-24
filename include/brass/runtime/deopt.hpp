#pragma once

#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <functional>
#include <exception>
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
    GcRef = 4,
    Float32 = 5,
    Boolean = 6
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

    static DeoptValue f32(float v) noexcept {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(float));
        return DeoptValue(DeoptValueKind::Float32, static_cast<uint64_t>(bits));
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

    static constexpr DeoptValue boolean(bool v) noexcept {
        return DeoptValue(DeoptValueKind::Boolean, v ? 1ULL : 0ULL);
    }

    int32_t as_i32() const noexcept {
        return static_cast<int32_t>(static_cast<uint32_t>(raw & 0xFFFFFFFFULL));
    }

    int64_t as_i64() const noexcept {
        return static_cast<int64_t>(raw);
    }

    float as_f32() const noexcept {
        uint32_t bits = static_cast<uint32_t>(raw & 0xFFFFFFFFULL);
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }

    double as_f64() const noexcept {
        double d = 0.0;
        std::memcpy(&d, &raw, sizeof(double));
        return d;
    }

    bool as_bool() const noexcept {
        return raw != 0;
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

// The materialized state of a failed guard. The state map has no size limit:
// `slots` and `kinds` grow with it (`count` always equals their size).
class DeoptFrame {
public:
    uint32_t resume_id = 0;
    DeoptReason reason = DeoptReason::Generic;
    void* target_fn = nullptr;
    // Entry point of the optimized code that deoptimized (null when the
    // frame came from an interpreter or an old-style exit).
    void* code_entry = nullptr;
    std::string exit_symbol;
    size_t count = 0;
    std::vector<uint64_t> slots;
    std::vector<DeoptValueKind> kinds;

    DeoptFrame() = default;

    void clear() noexcept {
        resume_id = 0;
        reason = DeoptReason::None;
        target_fn = nullptr;
        code_entry = nullptr;
        exit_symbol.clear();
        count = 0;
        slots.clear();
        kinds.clear();
    }

    void push_value(uint64_t val, DeoptValueKind kind = DeoptValueKind::Int64) {
        slots.push_back(val);
        kinds.push_back(kind);
        count = slots.size();
    }

    void push_deopt_value(const DeoptValue& dv) {
        push_value(dv.raw, dv.kind);
    }

    void set_value(size_t idx, const DeoptValue& dv) {
        if (idx >= slots.size()) {
            slots.resize(idx + 1, 0);
            kinds.resize(idx + 1, DeoptValueKind::Int64);
        }
        slots[idx] = dv.raw;
        kinds[idx] = dv.kind;
        count = slots.size();
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

// What optimized x64 code builds on its stack when a guard fails: this
// header, then `count` 8-byte slots, then `count` DeoptValueKind bytes.
struct DeoptExitRecord {
    void* code_entry;
    uint32_t resume_id;
    uint32_t reason;
    uint32_t count;
    uint32_t flags;

    static constexpr uint32_t kHasExitSymbol = 1u;
    static constexpr size_t kSlotsOffset = 24;
    // Or'd into brass_deopt_exit(_typed)'s `reason` by code that calls an
    // exit stub after it (the AArch64 guard exit): an unhandled deopt then
    // resumes in the stub instead of being fatal.
    static constexpr uint32_t kReasonHasExitStub = 0x80000000u;

    const uint64_t* slots() const noexcept {
        return reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(this) + kSlotsOffset);
    }
    const uint8_t* kinds() const noexcept {
        return reinterpret_cast<const uint8_t*>(slots() + count);
    }
};
static_assert(sizeof(DeoptExitRecord) == DeoptExitRecord::kSlotsOffset, "DeoptExitRecord header layout");
static_assert(offsetof(DeoptExitRecord, code_entry) == 0 && offsetof(DeoptExitRecord, resume_id) == 8 &&
              offsetof(DeoptExitRecord, reason) == 12 && offsetof(DeoptExitRecord, count) == 16 &&
              offsetof(DeoptExitRecord, flags) == 20, "DeoptExitRecord field offsets are baked into JIT code");
} // namespace brass::runtime

extern "C" {
    brass::runtime::DeoptFrame* brass_get_thread_deopt_frame();
    // Slot storage of this thread's deopt frame (what exit stubs receive).
    const uint64_t* brass_get_thread_deopt_slots();
    void brass_set_thread_deopt_frame(const brass::runtime::DeoptFrame* frame);
    void* brass_deopt_exit(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots);
    void* brass_deopt_exit_typed(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots, const uint8_t* raw_kinds);
    // Deopt entry of optimized x64 code. Publishes the thread deopt frame and
    // returns a pointer to the 8-byte result the optimized frame must return
    // when the deopt was handled (a registered resumer for `code_entry`, or a
    // registered handler when the guard has no exit symbol); null otherwise.
    const uint64_t* brass_deopt_exit_record(const brass::runtime::DeoptExitRecord* record);
}

namespace brass::runtime {

using ::brass_get_thread_deopt_frame;
using ::brass_set_thread_deopt_frame;
using ::brass_deopt_exit;
using ::brass_deopt_exit_typed;

using DeoptHandlerFn = std::function<void*(const DeoptFrame&)>;

DeoptFrame* get_thread_deopt_frame() noexcept;
void set_thread_deopt_frame(const DeoptFrame* frame) noexcept;
// The calling thread's deopt handler: it handles guard failures of code
// running on this thread only. Whoever installs one for a scope (an OSR
// call) restores the previous one when the scope ends.
void register_deopt_handler(DeoptHandlerFn handler);
DeoptHandlerFn get_deopt_handler();

// Called by a deopt handler, before it returns, when the failing code is not
// its own (an OSR call's handler seeing a guard of code outside its module).
// The deopt entry then does what it does with no handler installed: the
// native code takes its exit stub, and a guard with none aborts. The
// handler's return value is ignored.
void deopt_handler_decline() noexcept;

// Called by a deopt handler whose Tier-0 continuation threw a MIR exception
// (`value`, its raw bits) that must reach the native frames above the
// deoptimized one as a native throw would: once the handler returns, the
// deopt entry raises `value` as a brass exception to the first landing pad in
// the native frames between the deoptimized frame (not included) and
// `stack_limit` (the address of a local of the C++ frame that called the
// native code). If none of them has one, `fallback` (the Tier-0 exception)
// is rethrown unchanged. The handler's return value is ignored. A deopt
// resumer (below) may call it the same way; `stack_limit` UINTPTR_MAX
// searches every native frame above, as a native throw does.
void deopt_handler_throw_native(uint64_t value, uintptr_t stack_limit, std::exception_ptr fallback) noexcept;

// Per-code deopt continuation: given the materialized frame, finishes the
// function in a lower tier and returns its result bits. Registered by the
// tier-2 installer for each entry point it publishes; takes precedence over
// the thread's handler.
using DeoptResumerFn = std::function<uint64_t(const DeoptFrame&)>;
void register_deopt_resumer(void* code_entry, DeoptResumerFn resumer);
void unregister_deopt_resumer(void* code_entry);
bool has_deopt_resumer(void* code_entry);

} // namespace brass::runtime
