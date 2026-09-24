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
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#if defined(_MSC_VER)
#define BRASS_CORO_NOINLINE __declspec(noinline)
#else
#define BRASS_CORO_NOINLINE __attribute__((noinline))
#endif

namespace brass::runtime {

namespace {

static MicrotaskQueue g_microtask_queue;

// The set of live registries, and every CoroFrameCell's state and owner
// (leaked: heaps may die during static destruction). Recursive: a visitor
// may register frames.
std::recursive_mutex& registries_mutex() {
    static auto* m = new std::recursive_mutex();
    return *m;
}
std::vector<CoroFrameRegistry*>& live_registries() {
    static auto* v = new std::vector<CoroFrameRegistry*>();
    return *v;
}

} // namespace

// One registered frame. `addr` is the root slot a collection of the owning
// heap updates (under the registry's lock); `state` and `owner` are guarded
// by registries_mutex(). Holders outside the heap (CoroFrameRef) keep the
// cell alive after it leaves its registry.
struct CoroFrameCell {
    enum class State : uint8_t { Live, Finished, HeapGone };
    uintptr_t addr = 0;
    CoroFrameRegistry* owner = nullptr;
    State state = State::Live;
};

// One heap's frames. Cells do not move, so a collection may update an entry
// through the slot it was handed while other heaps' registries change.
class CoroFrameRegistry {
public:
    explicit CoroFrameRegistry(CoroFrameHolds h) : holds(std::move(h)) {
        std::lock_guard<std::recursive_mutex> lock(registries_mutex());
        live_registries().push_back(this);
    }
    ~CoroFrameRegistry() {
        std::lock_guard<std::recursive_mutex> lock(registries_mutex());
        auto& all = live_registries();
        all.erase(std::remove(all.begin(), all.end(), this), all.end());
        // The heap's memory goes with it: a frame still registered here is
        // unfinished, and every reference to it is now stale.
        std::lock_guard<std::recursive_mutex> own(mutex);
        for (auto& cell : frames) {
            cell->state = CoroFrameCell::State::HeapGone;
            cell->owner = nullptr;
        }
        frames.clear();
    }
    CoroFrameRegistry(const CoroFrameRegistry&) = delete;
    CoroFrameRegistry& operator=(const CoroFrameRegistry&) = delete;

    // Guards `frames` and every cell's `addr`. Held by the heap's collection
    // for as long as it updates root slots (CoroRootsLock).
    std::recursive_mutex mutex;
    std::list<std::shared_ptr<CoroFrameCell>> frames;
    // Whether the heap holds a frame at an address (coroutine.hpp). Guarded
    // by `mutex`.
    CoroFrameHolds holds;
};

std::shared_ptr<CoroFrameRegistry> make_coro_frame_registry(CoroFrameHolds holds) {
    if (!holds) throw std::logic_error("make_coro_frame_registry: a heap's registry needs its holds predicate");
    return std::make_shared<CoroFrameRegistry>(std::move(holds));
}

void set_coro_frame_registry_holds(CoroFrameRegistry& registry, CoroFrameHolds holds) {
    if (!holds) throw std::logic_error("set_coro_frame_registry_holds: a heap's registry needs its holds predicate");
    std::lock_guard<std::recursive_mutex> own(registry.mutex);
    registry.holds = std::move(holds);
}

CoroRootsLock::CoroRootsLock(CoroFrameRegistry& registry) : registry_(&registry) {
    registry_->mutex.lock();
}

CoroRootsLock::~CoroRootsLock() {
    registry_->mutex.unlock();
}

namespace {

// The installed HostHeap's frames, and frames allocated outside any heap
// (never collected, so never reported as roots).
CoroFrameRegistry& host_heap_frames() {
    static auto* r = new CoroFrameRegistry([](uintptr_t addr) {
        const HostHeap* heap = brass::host_heap();
        return heap != nullptr && heap->contains(addr);
    });
    return *r;
}
// Every frame allocated outside any heap: they are never freed, so an
// address here stays a frame. Guarded by unmanaged_frames()' lock.
std::unordered_set<uintptr_t>& unmanaged_frame_addrs() {
    static auto* s = new std::unordered_set<uintptr_t>();
    return *s;
}
CoroFrameRegistry& unmanaged_frames() {
    static auto* r = new CoroFrameRegistry([](uintptr_t addr) {
        return unmanaged_frame_addrs().count(addr) != 0;
    });
    return *r;
}

// The frame `handle` names, once some live heap is known to hold it (its
// registry's `holds`); otherwise a hard error, before any access through it.
BrassCoroFrame* checked_coro_frame(uintptr_t handle, const char* what) {
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    for (CoroFrameRegistry* r : live_registries()) {
        std::lock_guard<std::recursive_mutex> own(r->mutex);
        if (r->holds(handle)) return reinterpret_cast<BrassCoroFrame*>(handle);
    }
    // A registered frame is unfinished and live wherever it is (a HostHeap
    // that cannot answer `contains` still has its registered frames).
    for (CoroFrameRegistry* r : live_registries()) {
        std::lock_guard<std::recursive_mutex> own(r->mutex);
        for (auto& cell : r->frames) {
            if (cell->addr == handle) return reinterpret_cast<BrassCoroFrame*>(handle);
        }
    }
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "%s: coroutine handle %p names no frame a live heap holds (a stale handle: its heap "
                  "was torn down, a collection moved the frame, or it is not a coroutine frame)",
                  what, reinterpret_cast<void*>(handle));
    throw std::logic_error(msg);
}

// The registry of the heap allocate_coro_frame allocates from on this thread
// (the same order of choice).
CoroFrameRegistry& current_coro_registry() {
    if (brass::host_heap() != nullptr) return host_heap_frames();
    if (GenerationalGC* gen_gc = brass::brass_get_active_generational_gc()) return gen_gc->coro_frames();
    if (HostGC* host_gc = brass::get_active_host_gc()) return host_gc->coro_frames();
    if (MiniCheneyGC* gc = brass::brass_get_active_gc()) return gc->coro_frames();
    return unmanaged_frames();
}

// The cell of `frame` in `r`, or null. Caller holds registries_mutex().
std::shared_ptr<CoroFrameCell> find_in(CoroFrameRegistry& r, uintptr_t frame) {
    std::lock_guard<std::recursive_mutex> own(r.mutex);
    for (auto& cell : r.frames) {
        if (cell->addr == frame) return cell;
    }
    return nullptr;
}

// Caller holds registries_mutex().
bool erase_from(CoroFrameRegistry& r, uintptr_t frame) {
    std::lock_guard<std::recursive_mutex> own(r.mutex);
    for (auto it = r.frames.begin(); it != r.frames.end(); ++it) {
        if ((*it)->addr == frame) {
            (*it)->state = CoroFrameCell::State::Finished;
            (*it)->owner = nullptr;
            r.frames.erase(it);
            return true;
        }
    }
    return false;
}

// The cell of a registered frame, the thread's own heap first. Caller holds
// registries_mutex().
std::shared_ptr<CoroFrameCell> find_registered(CoroFrameRegistry& mine, uintptr_t frame) {
    if (auto cell = find_in(mine, frame)) return cell;
    for (CoroFrameRegistry* r : live_registries()) {
        if (r == &mine) continue;
        if (auto cell = find_in(*r, frame)) return cell;
    }
    return nullptr;
}

} // namespace

MicrotaskQueue& get_global_microtask_queue() {
    return g_microtask_queue;
}

void MicrotaskQueue::enqueue(Task task) {
    tasks_.push_back(std::move(task));
}

CoroFrameRef coro_frame_ref(BrassCoroFrame* frame) {
    if (!frame) return nullptr;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(frame);
    CoroFrameRegistry& mine = current_coro_registry(); // may create it: before the lock
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    if (auto cell = find_registered(mine, addr)) return cell;
    // Every unregistered frame is finished (it finished, threw, or was
    // destroyed); anything else is not a coroutine frame.
    if (checked_coro_frame(addr, "coroutine frame reference")->is_done == 0) {
        throw std::logic_error("coroutine frame reference: not a registered coroutine frame");
    }
    auto done = std::make_shared<CoroFrameCell>();
    done->addr = addr;
    done->state = CoroFrameCell::State::Finished;
    return done;
}

BrassCoroFrame* resolve_coro_frame_ref(const CoroFrameRef& ref) {
    if (!ref) return nullptr;
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    switch (ref->state) {
        case CoroFrameCell::State::Finished:
            return nullptr;
        case CoroFrameCell::State::HeapGone:
            throw std::logic_error("coroutine frame reference: the frame's heap was torn down while it "
                                   "was unfinished (a stale coroutine handle)");
        case CoroFrameCell::State::Live:
            break;
    }
    std::lock_guard<std::recursive_mutex> own(ref->owner->mutex);
    auto* frame = reinterpret_cast<BrassCoroFrame*>(ref->addr);
    return frame->is_done ? nullptr : frame;
}

void MicrotaskQueue::enqueue_coro(BrassCoroFrame* frame, uint64_t input_val) {
    if (!frame) return;
    enqueue_coro(coro_frame_ref(frame), input_val);
}

void MicrotaskQueue::enqueue_coro(CoroFrameRef frame, uint64_t input_val) {
    if (!frame) return;
    tasks_.push_back([frame = std::move(frame), input_val]() {
        // The frame's address now: a collection may have moved it.
        if (BrassCoroFrame* live = resolve_coro_frame_ref(frame)) {
            brass_coro_resume(reinterpret_cast<uintptr_t>(live), input_val);
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

    // Each waiter is resolved when the queue runs it (a finished one is
    // skipped there, one whose heap died is a hard error there).
    auto waiters = std::move(awaiting_frames_);
    awaiting_frames_.clear();
    for (auto& frame : waiters) {
        get_global_microtask_queue().enqueue_coro(std::move(frame), val);
    }
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
        awaiting_frames_.push_back(coro_frame_ref(frame));
    }
}

void register_active_coro_frame(BrassCoroFrame* frame) {
    if (!frame) return;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(frame);
    CoroFrameRegistry& r = current_coro_registry(); // may create it: before the lock
    auto cell = std::make_shared<CoroFrameCell>();
    cell->addr = addr;
    cell->owner = &r;
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    std::lock_guard<std::recursive_mutex> own(r.mutex);
    r.frames.push_back(std::move(cell));
}

void unregister_active_coro_frame(BrassCoroFrame* frame) {
    if (!frame) return;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(frame);
    // The thread's own heap first. Another heap's entries are read under
    // that registry's lock, which its collection holds while it updates them.
    CoroFrameRegistry& mine = current_coro_registry(); // may create it: before the lock
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    if (erase_from(mine, addr)) return;
    for (CoroFrameRegistry* r : live_registries()) {
        if (r != &mine && erase_from(*r, addr)) return;
    }
}

bool is_active_coro_frame(uintptr_t frame) {
    if (!frame) return false;
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    for (CoroFrameRegistry* r : live_registries()) {
        if (find_in(*r, frame)) return true;
    }
    return false;
}

void visit_active_coro_frames(const std::function<void(uintptr_t*)>& visitor) {
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    for (CoroFrameRegistry* r : live_registries()) {
        std::lock_guard<std::recursive_mutex> own(r->mutex);
        for (auto& cell : r->frames) {
            auto* frame = reinterpret_cast<BrassCoroFrame*>(cell->addr);
            if (frame && !frame->is_done) {
                visitor(&cell->addr);
            }
        }
    }
}

void append_active_coro_roots(CoroFrameRegistry& registry, std::vector<uintptr_t*>& roots) {
    std::lock_guard<std::recursive_mutex> own(registry.mutex);
    for (auto& cell : registry.frames) {
        auto* frame = reinterpret_cast<BrassCoroFrame*>(cell->addr);
        if (frame && !frame->is_done) {
            roots.push_back(&cell->addr);
        }
    }
}

void append_host_heap_coro_roots(std::vector<uintptr_t*>& roots) {
    // The host updates these slots after this returns: only while its
    // collection holds the registry's lock may it be handed them.
    if (!brass::in_host_heap_collection()) {
        throw std::logic_error("brass_enumerate_thread_roots: called outside a HostHeapCollectionScope; a host's "
                               "collection holds one until it has updated the slots reported");
    }
    append_active_coro_roots(host_heap_frames(), roots);
}

void lock_host_heap_coro_roots() {
    host_heap_frames().mutex.lock();
}

void unlock_host_heap_coro_roots() noexcept {
    host_heap_frames().mutex.unlock();
}

void finish_thrown_coro_frame(BrassCoroFrame* frame) noexcept {
    if (!frame) return;
    frame->is_done = 1;
    frame->state_id = ~0U;
    frame->yielded_val = 0;
    unregister_active_coro_frame(frame);
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
    const uintptr_t frame = reinterpret_cast<uintptr_t>(std::calloc(1, total_size));
    if (frame) {
        CoroFrameRegistry& r = unmanaged_frames();
        std::lock_guard<std::recursive_mutex> own(r.mutex);
        unmanaged_frame_addrs().insert(frame);
    }
    return frame;
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
// gc_msvc_x64.asm / gc_msvc_arm64.asm, which calls brass_coro_create_at.
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
    BrassCoroFrame* frame = checked_coro_frame(coro_frame, "brass_coro_resume");

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
    try {
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
    } catch (...) {
        // `coro_frame` is still a root here: it names the live copy.
        finish_thrown_coro_frame(reinterpret_cast<BrassCoroFrame*>(coro_frame));
        throw;
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
    return checked_coro_frame(coro_frame, "brass_coro_is_done")->is_done;
}

void brass_coro_destroy(uintptr_t coro_frame) {
    if (!coro_frame) return;
    // Only a registered frame is written. Every frame leaves its registry
    // finished (is_done set), so an unregistered handle a live heap holds is
    // already done; one no live heap holds is stale (a hard error).
    CoroFrameRegistry& mine = current_coro_registry(); // may create it: before the lock
    std::lock_guard<std::recursive_mutex> lock(registries_mutex());
    auto cell = find_registered(mine, coro_frame);
    if (!cell) {
        (void)checked_coro_frame(coro_frame, "brass_coro_destroy");
        return;
    }
    CoroFrameRegistry* owner = cell->owner;
    std::lock_guard<std::recursive_mutex> own(owner->mutex);
    reinterpret_cast<BrassCoroFrame*>(coro_frame)->is_done = 1;
    erase_from(*owner, coro_frame);
}

} // extern "C"
