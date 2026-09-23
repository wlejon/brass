#include <brass/runtime/deopt.hpp>
#include <ostream>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

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
        case DeoptValueKind::Float32: return "Float32";
        case DeoptValueKind::Float64: return "Float64";
        case DeoptValueKind::Pointer: return "Pointer";
        case DeoptValueKind::GcRef: return "GcRef";
        case DeoptValueKind::Boolean: return "Boolean";
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
        case DeoptValueKind::Float32:
            return RuntimeValue::from_f32(as_f32());
        case DeoptValueKind::Float64:
            return RuntimeValue::from_f64(as_f64());
        case DeoptValueKind::Pointer:
            return RuntimeValue::from_ptr(as_ptr());
        case DeoptValueKind::GcRef:
            return RuntimeValue::from_gcref(as_gcref());
        case DeoptValueKind::Boolean:
            return RuntimeValue::from_i32(as_bool() ? 1 : 0);
        default:
            return RuntimeValue::from_i64(as_i64());
    }
}

DeoptValue DeoptValue::from_runtime_value(const RuntimeValue& rv) {
    if (rv.is_i32()) {
        return DeoptValue::i32(rv.as_i32());
    } else if (rv.is_i64()) {
        return DeoptValue::i64(rv.as_i64());
    } else if (rv.is_f32()) {
        return DeoptValue::f32(rv.as_f32());
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

namespace {
std::mutex g_resumer_mutex;
std::unordered_map<void*, std::shared_ptr<const DeoptResumerFn>>& resumers() {
    static auto* map = new std::unordered_map<void*, std::shared_ptr<const DeoptResumerFn>>();
    return *map;
}
std::shared_ptr<const DeoptResumerFn> find_resumer(void* code_entry) {
    if (!code_entry) return nullptr;
    std::lock_guard<std::mutex> lock(g_resumer_mutex);
    auto it = resumers().find(code_entry);
    return it != resumers().end() ? it->second : nullptr;
}
thread_local uint64_t t_deopt_result = 0;
} // namespace

void register_deopt_resumer(void* code_entry, DeoptResumerFn resumer) {
    if (!code_entry || !resumer) {
        throw std::invalid_argument("register_deopt_resumer: null code entry or resumer");
    }
    std::lock_guard<std::mutex> lock(g_resumer_mutex);
    resumers()[code_entry] = std::make_shared<const DeoptResumerFn>(std::move(resumer));
}

void unregister_deopt_resumer(void* code_entry) {
    std::lock_guard<std::mutex> lock(g_resumer_mutex);
    resumers().erase(code_entry);
}

bool has_deopt_resumer(void* code_entry) {
    return find_resumer(code_entry) != nullptr;
}

} // namespace brass::runtime

extern "C" {

brass::runtime::DeoptFrame* brass_get_thread_deopt_frame() {
    return brass::runtime::get_thread_deopt_frame();
}

void brass_set_thread_deopt_frame(const brass::runtime::DeoptFrame* frame) {
    brass::runtime::set_thread_deopt_frame(frame);
}

void* brass_deopt_exit_typed(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots, const uint8_t* raw_kinds) {
    using brass::runtime::DeoptExitRecord;
    const bool has_exit_stub = (reason & DeoptExitRecord::kReasonHasExitStub) != 0;
    reason &= ~DeoptExitRecord::kReasonHasExitStub;
    auto* frame = brass::runtime::get_thread_deopt_frame();
    frame->clear();
    frame->resume_id = resume_id;
    frame->reason = static_cast<brass::runtime::DeoptReason>(reason);
    if (count > 0 && !raw_slots) {
        std::fprintf(stderr, "brass_deopt_exit: %u state values but no slot buffer\n", count);
        std::abort();
    }
    frame->slots.assign(raw_slots, raw_slots + count);
    frame->kinds.resize(count, brass::runtime::DeoptValueKind::Int64);
    if (raw_kinds) {
        for (size_t i = 0; i < count; ++i) {
            frame->kinds[i] = static_cast<brass::runtime::DeoptValueKind>(raw_kinds[i]);
        }
    }
    frame->count = count;

    auto handler = brass::runtime::get_deopt_handler();
    if (handler) {
        return handler(*frame);
    }
    if (has_exit_stub) return nullptr; // the caller resumes in its exit stub
    // Nothing can resume this frame: returning would hand the optimized
    // code's caller a made-up 0.
    std::fprintf(stderr, "brass_deopt_exit: guard failed (resume id %u, reason %u) with no deopt handler, "
                         "resumer or exit stub to resume in\n", resume_id, reason);
    std::abort();
}

void* brass_deopt_exit(uint32_t resume_id, uint32_t reason, uint32_t count, const uint64_t* raw_slots) {
    return brass_deopt_exit_typed(resume_id, reason, count, raw_slots, nullptr);
}

const uint64_t* brass_get_thread_deopt_slots() {
    return brass::runtime::get_thread_deopt_frame()->slots.data();
}

const uint64_t* brass_deopt_exit_record(const brass::runtime::DeoptExitRecord* record) {
    using namespace brass::runtime;
    if (!record) {
        std::fprintf(stderr, "brass_deopt_exit_record: null record\n");
        std::abort();
    }
    auto* frame = get_thread_deopt_frame();
    frame->clear();
    frame->resume_id = record->resume_id;
    frame->reason = static_cast<DeoptReason>(record->reason);
    frame->code_entry = record->code_entry;
    const uint64_t* slots = record->slots();
    const uint8_t* kinds = record->kinds();
    frame->slots.assign(slots, slots + record->count);
    frame->kinds.resize(record->count);
    for (uint32_t i = 0; i < record->count; ++i) {
        frame->kinds[i] = static_cast<DeoptValueKind>(kinds[i]);
    }
    frame->count = record->count;

    if (auto resumer = find_resumer(record->code_entry)) {
        // The resumer may run code that deoptimizes again and overwrites the
        // thread frame: it works on a copy.
        DeoptFrame snapshot = *frame;
        uint64_t result = (*resumer)(snapshot);
        t_deopt_result = result;
        return &t_deopt_result;
    }
    const bool has_exit_symbol = (record->flags & DeoptExitRecord::kHasExitSymbol) != 0;
    auto handler = get_deopt_handler();
    if (handler) {
        void* r = handler(*frame);
        if (!has_exit_symbol) {
            t_deopt_result = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r));
            return &t_deopt_result;
        }
    }
    if (has_exit_symbol) return nullptr; // the caller resumes in its exit stub
    // Nothing can resume this frame: returning null would make the optimized
    // code return a made-up 0.
    std::fprintf(stderr, "brass_deopt_exit_record: guard failed (resume id %u, reason %u) with no deopt handler, "
                         "resumer or exit stub to resume in\n", record->resume_id, record->reason);
    std::abort();
}

}
