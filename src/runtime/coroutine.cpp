#include <brass/runtime/coroutine.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/host_heap.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/embedding/nanbox.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

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

} // namespace brass::runtime

extern "C" {

using namespace brass;
using namespace brass::runtime;

uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask) {
    size_t extra_slots = (slot_count > 1) ? (slot_count - 1) : 0;
    size_t total_size = sizeof(BrassCoroFrame) + extra_slots * sizeof(uint64_t);

    constexpr size_t slot_shift = CORO_OFFSET_SLOTS / sizeof(uint64_t);
    uint64_t frame_mask = (pointer_mask << slot_shift);
    if (pointer_mask & (1ULL << 63)) {
        frame_mask |= (1ULL << 63);
    }

    BrassCoroFrame* frame = nullptr;

    GenerationalGC* gen_gc = brass::brass_get_active_generational_gc();
    if (brass::host_heap() != nullptr) {
        // The frame is an object of the host's heap; its slots are traced
        // through frame_mask like any other, and while suspended it is also
        // a root through the active-frame registry below.
        frame = reinterpret_cast<BrassCoroFrame*>(
            brass::host_heap_allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME));
    } else if (gen_gc != nullptr) {
        uintptr_t payload = gen_gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
        frame = reinterpret_cast<BrassCoroFrame*>(payload);
    } else {
        HostGC* host_gc = brass::get_active_host_gc();
        if (host_gc != nullptr) {
            uintptr_t payload = host_gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
            frame = reinterpret_cast<BrassCoroFrame*>(payload);
        } else {
            MiniCheneyGC* gc = brass::brass_get_active_gc();
            if (gc != nullptr) {
                uintptr_t payload = gc->allocate(total_size, frame_mask, TYPE_TAG_CORO_FRAME);
                frame = reinterpret_cast<BrassCoroFrame*>(payload);
            } else {
                frame = static_cast<BrassCoroFrame*>(std::calloc(1, total_size));
            }
        }
    }

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

uint64_t brass_coro_resume(uintptr_t coro_frame, uint64_t input_val) {
    if (!coro_frame) return 0;
    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);

    if (frame->is_done != 0) {
        unregister_active_coro_frame(frame);
        return frame->yielded_val;
    }

    frame->resume_arg = input_val;

    if (frame->fn_ptr != nullptr) {
        using CoroFn = uint64_t (*)(BrassCoroFrame*);
        auto fn = reinterpret_cast<CoroFn>(frame->fn_ptr);
        uint64_t result = fn(frame);
        frame->yielded_val = result;
        if (frame->is_done != 0) {
            unregister_active_coro_frame(frame);
        }
        return result;
    }

    return 0;
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
