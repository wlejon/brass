#include <brass/runtime/coroutine.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/embedding/nanbox.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/exception.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#if defined(_MSC_VER)
#define BRASS_CORO_NOINLINE __declspec(noinline)
#else
#define BRASS_CORO_NOINLINE __attribute__((noinline))
#endif

namespace brass::runtime {

namespace {

static MicrotaskQueue g_microtask_queue;
static std::vector<BrassCoroFrame*> g_active_coro_frames;

} // namespace

MicrotaskQueue& get_global_microtask_queue() {
    return g_microtask_queue;
}

void MicrotaskQueue::enqueue(Task task) {
    tasks_.push_back(std::move(task));
}

void MicrotaskQueue::enqueue_coro(BrassCoroFrame* frame, uint64_t input_val) {
    if (!frame) return;
    tasks_.push_back([frame, input_val]() {
        if (!frame->is_done) {
            brass_coro_resume(reinterpret_cast<uintptr_t>(frame), input_val);
        }
    });
}

void MicrotaskQueue::run_all() {
    size_t idx = 0;
    while (idx < tasks_.size()) {
        auto task = std::move(tasks_[idx++]);
        if (task) {
            task();
        }
    }
    tasks_.clear();
}

void Promise::fulfill(uint64_t val) {
    if (state_ != PromiseState::Pending) return;
    state_ = PromiseState::Fulfilled;
    value_ = val;

    for (auto& cb : callbacks_) {
        if (cb) cb(val);
    }
    callbacks_.clear();

    for (auto* frame : awaiting_frames_) {
        if (frame && !frame->is_done) {
            get_global_microtask_queue().enqueue_coro(frame, val);
        }
    }
    awaiting_frames_.clear();
}

void Promise::reject(uint64_t reason) {
    if (state_ != PromiseState::Pending) return;
    state_ = PromiseState::Rejected;
    value_ = reason;
}

void Promise::then(std::function<void(uint64_t)> on_fulfilled) {
    if (state_ == PromiseState::Fulfilled) {
        on_fulfilled(value_);
    } else if (state_ == PromiseState::Pending) {
        callbacks_.push_back(std::move(on_fulfilled));
    }
}

void Promise::await_in(BrassCoroFrame* frame) {
    if (!frame) return;
    if (state_ == PromiseState::Fulfilled) {
        get_global_microtask_queue().enqueue_coro(frame, value_);
    } else if (state_ == PromiseState::Pending) {
        awaiting_frames_.push_back(frame);
    }
}

void register_active_coro_frame(BrassCoroFrame* frame) {
    if (!frame) return;
    g_active_coro_frames.push_back(frame);
}

void unregister_active_coro_frame(BrassCoroFrame* frame) {
    if (!frame) return;
    auto it = std::find(g_active_coro_frames.begin(), g_active_coro_frames.end(), frame);
    if (it != g_active_coro_frames.end()) {
        g_active_coro_frames.erase(it);
    }
}

bool is_active_coro_frame(uintptr_t frame) {
    if (!frame) return false;
    for (auto* f : g_active_coro_frames) {
        if (reinterpret_cast<uintptr_t>(f) == frame) return true;
    }
    return false;
}

void visit_active_coro_frames(const std::function<void(uintptr_t*)>& visitor) {
    for (auto*& frame : g_active_coro_frames) {
        if (frame && !frame->is_done) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(frame);
            visitor(&addr);
            frame = reinterpret_cast<BrassCoroFrame*>(addr);
        }
    }
}

void append_active_coro_roots(std::vector<uintptr_t*>& roots) {
    for (auto*& frame : g_active_coro_frames) {
        if (frame && !frame->is_done) {
            roots.push_back(reinterpret_cast<uintptr_t*>(&frame));
        }
    }
}

namespace {

// The coroutine frame, from the thread's heap. A collection this allocation
// triggers must update the gcref slots of the generated frames from
// (caller_fp, caller_ip) upward, as brass_gc_alloc's does: the caller holds
// gcrefs across brass_coro_create. Both 0 (an interpreter, whose roots reach
// the heap through its scope and provider) has no generated frame to report;
// the heap's other roots (interpreter scopes and providers, native frames
// under re-entered Tier-0 code, suspended coroutines) are gathered as always.
// A caller frame that no stack maps describe cannot have its gcrefs found:
// a collection there is fatal, as it is for brass_gc_alloc.
[[noreturn]] void coro_fatal_no_maps() {
    std::fprintf(stderr, "brass: fatal: brass_coro_create needs a collection but no stack maps are "
                         "active and the calling code has none registered, so live gcrefs in native "
                         "frames cannot be found; call brass_set_active_stack_maps with the running "
                         "code's maps\n");
    std::fflush(stderr);
    std::abort();
}

uintptr_t allocate_coro_frame(size_t total_size, uint64_t frame_mask,
                              uintptr_t caller_fp, uintptr_t caller_ip) {
    if (brass::host_heap() != nullptr) {
        // The frame is an object of the host's heap; its slots are traced
        // through frame_mask like any other, and while suspended it is also
        // a root through the active-frame registry.
        return brass::host_heap_allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME,
                                         caller_fp, caller_ip);
    }
    const bool has_caller = caller_fp != 0 && caller_ip != 0;
    const ModuleStackMap* maps = has_caller ? brass::brass_stack_maps_for_caller(caller_ip) : nullptr;
    if (GenerationalGC* gen_gc = brass::brass_get_active_generational_gc()) {
        if (!gen_gc->can_allocate_fast(total_size) && has_caller) {
            if (!maps) coro_fatal_no_maps();
            return brass::brass_runtime_gc_alloc(gen_gc, *maps, total_size, frame_mask,
                                                 TYPE_TAG_CORO_FRAME, caller_fp, caller_ip);
        }
        return gen_gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
    }
    if (HostGC* host_gc = brass::get_active_host_gc()) {
        // As host_gc_alloc_bridge: collect with the caller's frames first.
        if (caller_fp != 0 && caller_ip != 0 && !host_gc->can_allocate_fast(total_size)) {
            std::vector<uintptr_t*> ptr_roots;
            std::vector<HostValue*> val_roots;
            host_gc->collect(ptr_roots, val_roots, caller_fp, caller_ip);
        }
        return host_gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
    }
    if (MiniCheneyGC* gc = brass::brass_get_active_gc()) {
        if (!gc->can_allocate_fast(total_size) && has_caller) {
            if (!maps) coro_fatal_no_maps();
            return brass::brass_runtime_gc_alloc(gc, *maps, total_size, frame_mask,
                                                 TYPE_TAG_CORO_FRAME, caller_fp, caller_ip);
        }
        return gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
    }
    return reinterpret_cast<uintptr_t>(std::calloc(1, total_size));
}

} // namespace

} // namespace brass::runtime

extern "C" {

using namespace brass;
using namespace brass::runtime;

uintptr_t brass_coro_create_at(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask,
                               uintptr_t caller_fp, uintptr_t caller_ip) {
    size_t extra_slots = (slot_count > 1) ? (slot_count - 1) : 0;
    size_t total_size = sizeof(BrassCoroFrame) + extra_slots * sizeof(uint64_t);

    constexpr size_t slot_shift = CORO_OFFSET_SLOTS / sizeof(uint64_t);
    uint64_t frame_mask = (pointer_mask << slot_shift);
    if (pointer_mask & (1ULL << 63)) {
        frame_mask |= (1ULL << 63);
    }

    auto* frame = reinterpret_cast<BrassCoroFrame*>(
        allocate_coro_frame(total_size, frame_mask, caller_fp, caller_ip));

    if (!frame) return 0;

    frame->state_id = 0;
    frame->is_done = 0;
    frame->fn_ptr = fn_ptr;
    frame->yielded_val = 0;
    frame->resume_arg = 0;
    frame->slot_count = slot_count;
    frame->flags = 0;

    register_active_coro_frame(frame);
    return reinterpret_cast<uintptr_t>(frame);
}

// Generated code calls brass_coro_create with no frame argument, so it takes
// its caller's frame as brass_gc_alloc does. MSVC: the stub in
// gc_msvc_x64.asm, which calls brass_coro_create_at.
#if !defined(_MSC_VER)
uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask) {
    void* frame = __builtin_frame_address(0);
    uintptr_t caller_fp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    return brass_coro_create_at(fn_ptr, slot_count, pointer_mask, caller_fp, caller_ip);
}
#endif

namespace {

// The body of both resume entries; `caller_*` name the frame that called the
// entry (brass_capture_caller_frame, taken there). Every scope it opens lives
// in this frame, so an exception leaving it has already destroyed them.
uint64_t coro_resume_impl(uintptr_t coro_frame, uint64_t input_val, bool have_caller, uintptr_t caller_rbp,
                          uintptr_t caller_ip) {
    if (!coro_frame) return 0;
    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);

    if (frame->is_done != 0) {
        unregister_active_coro_frame(frame);
        return frame->yielded_val;
    }

    frame->resume_arg = input_val;

    if (frame->fn_ptr == nullptr) {
        // Not a jump into non-code: nothing can run this frame. (A
        // FastInterpreter's unlowered coroutine keeps its state in that
        // interpreter; only it resumes the handle.)
        std::fprintf(stderr, "brass: fatal: brass_coro_resume of coroutine frame %p, which has no body "
                             "(a FastInterpreter coroutine of a body not lowered by CoroTransformPass "
                             "is resumed only by that interpreter)\n",
                     static_cast<void*>(frame));
        std::fflush(stderr);
        std::abort();
    }

    // The body (generated code, or a MIR body run in Tier 0) may collect.
    // The generated frames that called this function hold gcrefs across the
    // call: record them (native_frames.hpp), since a walk from the body
    // cannot rely on following frame pointers through this C++ frame
    // everywhere. The frame itself may move; `coro_frame` is a root so the
    // writes below reach the live copy. (A C++ caller has no generated frame
    // to record; its captured frame reports nothing.)
    NativeFramesScope native_frames(have_caller ? caller_rbp : 0, have_caller ? caller_ip : 0);
    ThreadRootsScope frame_root([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &coro_frame);
    uint64_t result;
    if (mir_coro_body(frame) != nullptr) {
        result = resume_mir_coro_body(coro_frame);
    } else {
        // A native body is generated code entered from this C++ frame: a
        // throw in it searches for pads no further than here and leaves as
        // a C++ exception, which unwinds the scopes above (an SEH exception
        // raised past this frame would skip their destructors, /EHs).
        GeneratedCodeEntryScope entry;
        using CoroFn = uint64_t (*)(BrassCoroFrame*);
        result = reinterpret_cast<CoroFn>(frame->fn_ptr)(frame);
    }
    frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);
    frame->yielded_val = result;
    if (frame->is_done != 0) {
        unregister_active_coro_frame(frame);
    }
    return result;
}

} // namespace

// The C++ entry (hosts, interpreters, the microtask queue): an exception the
// body throws reaches the caller as a C++ exception.
BRASS_CORO_NOINLINE uint64_t brass_coro_resume(uintptr_t coro_frame, uint64_t input_val) {
    uintptr_t caller_rbp = 0, caller_ip = 0;
    const bool have_caller = brass_capture_caller_frame(caller_rbp, caller_ip);
    return coro_resume_impl(coro_frame, input_val, have_caller, caller_rbp, caller_ip);
}

// The entry generated code calls (the JIT's "brass_coro_resume"). Its caller's
// landing pads see only native throws (the personality ignores C++
// exceptions), so an exception the body throws is caught here, once
// coro_resume_impl's scopes are gone, and raised again natively from this
// frame, which holds nothing to unwind.
BRASS_CORO_NOINLINE uint64_t brass_coro_resume_from_generated(uintptr_t coro_frame, uint64_t input_val) {
    uintptr_t caller_rbp = 0, caller_ip = 0;
    const bool have_caller = brass_capture_caller_frame(caller_rbp, caller_ip);
    HostValue pending{};
    try {
        return coro_resume_impl(coro_frame, input_val, have_caller, caller_rbp, caller_ip);
    } catch (const BrassException& ex) {
        pending = ex.value();
    } catch (const InterpreterThrownException& ex) {
        pending = HostValue::from_raw(ex.value().raw_bits());
    }
    brass_throw(pending);
}

uint32_t brass_coro_is_done(uintptr_t coro_frame) {
    if (!coro_frame) return 1;
    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);
    return frame->is_done;
}

void brass_coro_destroy(uintptr_t coro_frame) {
    if (!coro_frame) return;
    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);
    frame->is_done = 1;
    unregister_active_coro_frame(frame);
}

} // extern "C"
