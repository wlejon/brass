#pragma once

#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include <string_view>

namespace brass {
class Function;
}

namespace brass::runtime {

struct BrassCoroFrame {
    uint32_t state_id;      // 0 = initial, 1..N = suspend states, ~0U = done
    uint32_t is_done;       // 1 if finished, 0 if active/suspended
    void* fn_ptr;           // The body: see CORO_FLAG_MIR_BODY
    uint64_t yielded_val;   // Last yielded value
    uint64_t resume_arg;    // Input value passed into resume
    uint32_t slot_count;    // Total slots allocated
    uint32_t flags;         // CORO_FLAG_*
    uint64_t slots[1];      // Spilled SSA slots (flexible / variable sized)
};

// A frame's body is one of two kinds, and every tier resumes either:
//  - flag clear: fn_ptr is generated code, `uint64_t (*)(BrassCoroFrame*)`
//    (a frame generated code created);
//  - flag set: fn_ptr is the `const Function*` of a lowered MIR body (a
//    frame an interpreter created). brass_coro_resume runs it in Tier 0
//    (resume_mir_coro_body); an interpreter runs it itself.
// A frame with neither (fn_ptr null) has no body any tier can run.
static constexpr uint32_t CORO_FLAG_MIR_BODY = 1u;

inline const Function* mir_coro_body(const BrassCoroFrame* frame) noexcept {
    return (frame->flags & CORO_FLAG_MIR_BODY) ? static_cast<const Function*>(frame->fn_ptr) : nullptr;
}

// Creates a frame whose body is the lowered MIR function `body` (no
// generated frame calls this: an interpreter's roots reach the heap
// through its own scope and provider).
uintptr_t create_mir_coro_frame(const Function& body, uint32_t slot_count, uint64_t pointer_mask);

// Runs one step of a MIR-bodied frame on this thread, in the innermost
// interpreter running here (Interpreter, then FastInterpreter), else in a
// fresh Interpreter allocating from the thread's active heap. `frame_addr`
// must be a root of the caller: a collection the body triggers updates it.
// Returns the body's result bits (brass_coro_resume stores them).
uint64_t resume_mir_coro_body(uintptr_t& frame_addr);

static constexpr size_t CORO_FRAME_HEADER_SIZE = 40; // 5 * 8 bytes
static constexpr uint32_t TYPE_TAG_CORO_FRAME  = 200;

static constexpr int32_t CORO_OFFSET_STATE_ID   = 0;
static constexpr int32_t CORO_OFFSET_IS_DONE    = 4;
static constexpr int32_t CORO_OFFSET_FN_PTR     = 8;
static constexpr int32_t CORO_OFFSET_YIELD_VAL  = 16;
static constexpr int32_t CORO_OFFSET_RESUME_ARG = 24;
static constexpr int32_t CORO_OFFSET_SLOT_COUNT = 32;
static constexpr int32_t CORO_OFFSET_FLAGS      = 36;
static constexpr int32_t CORO_OFFSET_SLOTS      = 40;

// A counted reference to a registered coroutine frame, for holders outside
// any heap (the microtask queue, a Promise's waiters). It follows the frame
// when a collection moves it, notes when the frame finishes or is destroyed,
// and notes when the frame's heap is torn down: using it then is a hard error
// (std::logic_error), never a read or write of the dead heap's memory.
struct CoroFrameCell;
using CoroFrameRef = std::shared_ptr<CoroFrameCell>;

// The reference to `frame`, a live frame. A finished frame (no longer
// registered) yields a reference that resolves to nullptr. Throws
// std::logic_error for a frame no registry knows that is not finished, and
// for the address of a frame whose heap was torn down.
CoroFrameRef coro_frame_ref(BrassCoroFrame* frame);
// The frame's current address, or nullptr once it finished or was destroyed.
// Throws std::logic_error when the frame's heap was torn down.
BrassCoroFrame* resolve_coro_frame_ref(const CoroFrameRef& ref);

// Microtask Queue for async/await scheduling
class MicrotaskQueue {
public:
    using Task = std::function<void()>;

    void enqueue(Task task);
    // Resumes the frame when the queue runs, unless it finished by then. A
    // frame whose heap was torn down in between is a hard error at run_all.
    void enqueue_coro(BrassCoroFrame* frame, uint64_t input_val);
    void enqueue_coro(CoroFrameRef frame, uint64_t input_val);
    void run_all();
    bool empty() const noexcept { return tasks_.empty(); }
    size_t size() const noexcept { return tasks_.size(); }
    void clear() noexcept { tasks_.clear(); }

private:
    std::vector<Task> tasks_;
};

MicrotaskQueue& get_global_microtask_queue();

// Promise state and bridge
enum class PromiseState : uint8_t {
    Pending,
    Fulfilled,
    Rejected
};

class Promise {
public:
    Promise() = default;

    PromiseState state() const noexcept { return state_; }
    uint64_t value() const noexcept { return value_; }

    void fulfill(uint64_t val);
    void reject(uint64_t reason);
    void then(std::function<void(uint64_t)> on_fulfilled);
    void await_in(BrassCoroFrame* frame);

private:
    PromiseState state_ = PromiseState::Pending;
    uint64_t value_ = 0;
    std::vector<std::function<void(uint64_t)>> callbacks_;
    std::vector<CoroFrameRef> awaiting_frames_;
};

// Suspended coroutine frames are roots of the heap they were allocated in.
// Each gc::Heap owns a registry of its frames (frames allocated on a thread
// with no heap share one and are never freed), and a heap's entries go with
// it. A
// frame leaves its registry when it finishes, when its body throws, or on
// brass_coro_destroy. The set of registries is guarded by one lock, and each
// registry's entries by its own: a heap's collection updates its entries in
// place while it holds a CoroRootsLock on its registry, so a thread that
// looks through the registries (unregister, is_active_coro_frame) never reads
// an entry a collection is writing. Lock order: the set, then a registry;
// a collection holding its registry's lock takes no other.
//
// A raw frame handle (brass_coro_resume, brass_coro_is_done,
// brass_coro_destroy) is checked before anything reads or writes through it:
// it must name a frame some live heap holds, as that heap's `holds` predicate
// answers (an object of its current space with the coroutine frame type tag;
// frames allocated outside any heap are never freed and are recorded). A
// handle into a heap that was torn down, into the space a collection left, or
// to memory that is no coroutine frame is a hard error (std::logic_error),
// never a read of that memory. The check needs no record of retired
// addresses: a frame a collection moved to memory a torn-down heap once used
// is held by its live heap and passes. (A stale handle that happens to equal
// the address of another heap's live frame names that frame: it is memory a
// live heap owns, and the handle's holder broke the rooting contract.)
class CoroFrameRegistry;
// Whether `addr` is the address of a coroutine frame the owning heap holds
// now. Called under the registry's lock, which the heap's collection holds.
using CoroFrameHolds = std::function<bool(uintptr_t addr)>;
std::shared_ptr<CoroFrameRegistry> make_coro_frame_registry(CoroFrameHolds holds);

// Held by a heap's collection from gathering its coroutine roots
// (append_active_coro_roots) until the last root slot is updated.
class CoroRootsLock {
public:
    explicit CoroRootsLock(CoroFrameRegistry& registry);
    ~CoroRootsLock();
    CoroRootsLock(const CoroRootsLock&) = delete;
    CoroRootsLock& operator=(const CoroRootsLock&) = delete;

private:
    CoroFrameRegistry* registry_;
};

// Registers `frame` in the registry of the heap the thread allocates
// coroutine frames from (allocate_coro_frame's choice, coroutine.cpp).
void register_active_coro_frame(BrassCoroFrame* frame);
void unregister_active_coro_frame(BrassCoroFrame* frame);
bool is_active_coro_frame(uintptr_t frame);
// Every registry's unfinished frames.
void visit_active_coro_frames(const std::function<void(uintptr_t*)>& visitor);
// The unfinished frames of one heap, as root slots its collection updates
// (under a CoroRootsLock on `registry`, held until the slots are updated).
void append_active_coro_roots(CoroFrameRegistry& registry, std::vector<uintptr_t*>& roots);

// A body that throws is finished, as a generator that threw is: its frame is
// done (a later resume returns 0 without running it) and no longer a root.
// Every resume path (brass_coro_resume, both interpreters) calls this while
// the exception passes.
void finish_thrown_coro_frame(BrassCoroFrame* frame) noexcept;

} // namespace brass::runtime

extern "C" {
// The entry generated code calls: allocates the frame with the calling
// frame's gcrefs among a collection's roots (as brass_gc_alloc).
uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask);
// The same with the generated frame named: its frame pointer and the return
// address into it (both 0: none, e.g. an interpreter's call).
uintptr_t brass_coro_create_at(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask,
                               uintptr_t caller_fp, uintptr_t caller_ip);
// Resumes the frame from C++: an exception the body throws leaves as a C++
// exception.
uint64_t brass_coro_resume(uintptr_t coro_frame, uint64_t input_val);
// The same, for a generated caller (the JIT binds "brass_coro_resume" here):
// an exception the body throws is raised again natively (brass_throw), so the
// caller's landing pads see it.
uint64_t brass_coro_resume_from_generated(uintptr_t coro_frame, uint64_t input_val);
uint32_t brass_coro_is_done(uintptr_t coro_frame);
// Marks a registered frame finished and unregisters it; a finished frame is
// left as it is. A handle no live heap holds (its heap was torn down, or it
// is no coroutine frame) is a hard error and is not written through, as for
// brass_coro_resume and brass_coro_is_done.
void brass_coro_destroy(uintptr_t coro_frame);
}
