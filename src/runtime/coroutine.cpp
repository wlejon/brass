#include <brass/runtime/coroutine.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/object.hpp>
#include <brass/embedding/nanbox.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace brass::runtime {

namespace {

static MicrotaskQueue g_microtask_queue;
static std::vector<BrassCoroFrame*> g_active_coro_frames;
static ShapeRegistry g_iter_shape_registry;

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
        if (frame) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(frame);
            visitor(&addr);
            frame = reinterpret_cast<BrassCoroFrame*>(addr);
        }
    }
}

static void* (*g_coro_symbol_resolver)(const char*) = nullptr;

void set_coro_symbol_resolver(void* (*resolver)(const char*)) {
    g_coro_symbol_resolver = resolver;
}

} // namespace brass::runtime

extern "C" {

using namespace brass;
using namespace brass::runtime;

uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask) {
    size_t extra_slots = (slot_count > 1) ? (slot_count - 1) : 0;
    size_t total_size = sizeof(BrassCoroFrame) + extra_slots * sizeof(uint64_t);

    MiniCheneyGC* gc = brass::brass_get_active_gc();
    BrassCoroFrame* frame = nullptr;

    if (gc != nullptr) {
        uintptr_t payload = gc->allocate(total_size, pointer_mask, TYPE_TAG_CORO_FRAME);
        frame = reinterpret_cast<BrassCoroFrame*>(payload);
    } else {
        frame = static_cast<BrassCoroFrame*>(std::calloc(1, total_size));
    }

    if (!frame) return 0;

    void* actual_fn = fn_ptr;
    if (g_coro_symbol_resolver && fn_ptr) {
        void* res = g_coro_symbol_resolver(reinterpret_cast<const char*>(fn_ptr));
        if (res) actual_fn = res;
    }

    frame->state_id = 0;
    frame->is_done = 0;
    frame->fn_ptr = actual_fn;
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
        return frame->yielded_val;
    }

    frame->resume_arg = input_val;

    if (g_coro_symbol_resolver && frame->fn_ptr != nullptr) {
        void* res = g_coro_symbol_resolver(reinterpret_cast<const char*>(frame->fn_ptr));
        if (res) frame->fn_ptr = res;
    }

    if (frame->fn_ptr != nullptr) {
        using CoroFn = uint64_t (*)(BrassCoroFrame*);
        auto fn = reinterpret_cast<CoroFn>(frame->fn_ptr);
        uint64_t result = fn(frame);
        frame->yielded_val = result;
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

uint64_t bronze_iter_open(uint64_t gen_or_obj) {
    // Returns the iterator handle (coroutine frame or generator object)
    return gen_or_obj;
}

uint64_t bronze_iter_step(uint64_t iter_handle) {
    if (!iter_handle) return HostValue::undefined_val().raw();

    uint64_t yielded = brass_coro_resume(iter_handle, 0);
    uint32_t done = brass_coro_is_done(iter_handle);

    DynamicObject* obj = DynamicObject::create();
    if (!obj) {
        return yielded;
    }

    obj->set_property("value", HostValue(yielded), g_iter_shape_registry);
    obj->set_property("done", HostValue::from_bool(done != 0), g_iter_shape_registry);

    return HostValue::from_gcref(reinterpret_cast<uintptr_t>(obj)).raw();
}

uint64_t bronze_create_async_machine(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask, uint64_t env) {
    uintptr_t frame_addr = brass_coro_create(fn_ptr, std::max(slot_count, 1U), pointer_mask);
    if (!frame_addr) return 0;

    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(frame_addr);
    frame->slots[0] = env;

    return static_cast<uint64_t>(frame_addr);
}

uint64_t bronze_async_start(uint64_t coro_frame, uint64_t arg) {
    if (!coro_frame) return 0;
    uint64_t res = brass_coro_resume(coro_frame, arg);
    get_global_microtask_queue().run_all();

    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);
    if (frame) {
        if (frame->is_done && frame->yielded_val != 0) {
            return frame->yielded_val;
        }
        if (frame->yielded_val == 0 && res != 0) {
            frame->yielded_val = res;
            frame->is_done = 1;
        }
    }
    return res;
}

uint64_t bronze_async_await(uint64_t coro_frame, uint64_t val) {
    if (!coro_frame) return val;
    BrassCoroFrame* frame = reinterpret_cast<BrassCoroFrame*>(coro_frame);
    get_global_microtask_queue().enqueue_coro(frame, val);
    return val;
}

}
