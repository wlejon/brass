#include <brass/runtime/deopt.hpp>
#include <ostream>
#include <mutex>

namespace brass::runtime {

std::string_view to_string(DeoptReason reason) noexcept {
    switch (reason) {
        case DeoptReason::None: return "None";
        case DeoptReason::Generic: return "Generic";
        case DeoptReason::TypeCheckFailed: return "TypeCheckFailed";
        case DeoptReason::Overflow: return "Overflow";
        case DeoptReason::ShapeCheckFailed: return "ShapeCheckFailed";
        case DeoptReason::BoundsCheckFailed: return "BoundsCheckFailed";
        case DeoptReason::DivisionByZero: return "DivisionByZero";
        case DeoptReason::NullCheckFailed: return "NullCheckFailed";
        case DeoptReason::Custom: return "Custom";
        default: return "Unknown";
    }
}

std::ostream& operator<<(std::ostream& os, DeoptReason reason) {
    return os << to_string(reason);
}

std::string_view to_string(DeoptValueKind kind) noexcept {
    switch (kind) {
        case DeoptValueKind::Int32: return "Int32";
        case DeoptValueKind::Int64: return "Int64";
        case DeoptValueKind::Float64: return "Float64";
        case DeoptValueKind::Pointer: return "Pointer";
        case DeoptValueKind::GcRef: return "GcRef";
        default: return "Unknown";
    }
}

std::ostream& operator<<(std::ostream& os, DeoptValueKind kind) {
    return os << to_string(kind);
}

RuntimeValue DeoptValue::to_runtime_value() const {
    switch (kind) {
        case DeoptValueKind::Int32:
            return RuntimeValue::from_i32(as_i32());
        case DeoptValueKind::Int64:
            return RuntimeValue::from_i64(as_i64());
        case DeoptValueKind::Float64:
            return RuntimeValue::from_f64(as_f64());
        case DeoptValueKind::Pointer:
            return RuntimeValue::from_ptr(as_ptr());
        case DeoptValueKind::GcRef:
            return RuntimeValue::from_gcref(as_gcref());
        default:
            return RuntimeValue::from_i64(as_i64());
    }
}

DeoptValue DeoptValue::from_runtime_value(const RuntimeValue& rv) {
    if (rv.is_i32()) {
        return DeoptValue::i32(rv.as_i32());
    } else if (rv.is_i64()) {
        return DeoptValue::i64(rv.as_i64());
    } else if (rv.is_f64()) {
        return DeoptValue::f64(rv.as_f64());
    } else if (rv.is_ptr()) {
        return DeoptValue::ptr(rv.as_ptr());
    } else if (rv.is_gcref()) {
        return DeoptValue::gcref(rv.as_gcref());
    }
    return DeoptValue::i64(0);
}

std::vector<RuntimeValue> DeoptFrame::to_runtime_values() const {
    std::vector<RuntimeValue> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(get_value(i).to_runtime_value());
    }
    return result;
}

static thread_local DeoptFrame g_thread_deopt_frame;
static DeoptHandlerFn g_deopt_handler = nullptr;
static std::mutex g_deopt_handler_mutex;

DeoptFrame* get_thread_deopt_frame() noexcept {
    return &g_thread_deopt_frame;
}

void set_thread_deopt_frame(const DeoptFrame* frame) noexcept {
    if (frame) {
        g_thread_deopt_frame = *frame;
    } else {
        g_thread_deopt_frame.clear();
    }
}

void register_deopt_handler(DeoptHandlerFn handler) {
    std::lock_guard<std::mutex> lock(g_deopt_handler_mutex);
    g_deopt_handler = std::move(handler);
}

DeoptHandlerFn get_deopt_handler() {
    std::lock_guard<std::mutex> lock(g_deopt_handler_mutex);
    return g_deopt_handler;
}

} // namespace brass::runtime

extern "C" {

brass::runtime::DeoptFrame* brass_get_thread_deopt_frame() {
    return brass::runtime::get_thread_deopt_frame();
}

void brass_set_thread_deopt_frame(const brass::runtime::DeoptFrame* frame) {
    brass::runtime::set_thread_deopt_frame(frame);
}

void* brass_deopt_exit(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots) {
    auto* frame = brass::runtime::get_thread_deopt_frame();
    frame->clear();
    frame->resume_id = resume_id;
    frame->reason = static_cast<brass::runtime::DeoptReason>(reason);
    frame->count = (count > brass::runtime::DeoptFrame::kMaxSlots) ? brass::runtime::DeoptFrame::kMaxSlots : count;

    if (raw_slots) {
        for (size_t i = 0; i < frame->count; ++i) {
            frame->slots[i] = raw_slots[i];
            frame->kinds[i] = brass::runtime::DeoptValueKind::Int64;
        }
    }

    auto handler = brass::runtime::get_deopt_handler();
    if (handler) {
        return handler(*frame);
    }
    return nullptr;
}

}
