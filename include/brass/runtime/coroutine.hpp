#pragma once

#include <brass/interpreter/value.hpp>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <string_view>

namespace brass {
class Function;
}

namespace brass::runtime {

class FunctionDispatchTable;

// A coroutine frame: a heap object (TYPE_TAG_CORO_FRAME) holding a lowered
// body's state between resumes. The header is ABI: every tier and a MIR
// producer address it through the CORO_OFFSET_* constants.
struct BrassCoroFrame {
    uint32_t state_id;      // 0 = initial, 1..N = suspend states, ~0U = done
    uint32_t is_done;       // 1 if finished, 0 if active/suspended
    void* fn_ptr;           // The body: see CORO_FLAG_BODY
    uint64_t yielded_val;   // Last yielded (or returned) value
    uint64_t resume_arg;    // The value the latest resume passed in
    uint32_t slot_count;    // Total slots allocated
    uint32_t flags;         // CORO_FLAG_*
    uint32_t resume_mode;   // The mode the latest resume passed in (CoroResumeMode or the producer's own)
    uint32_t reserved;
    uint64_t awaiter;       // The frame waiting on this one (a gcref, traced), or 0
    uint64_t slots[1];      // Spilled SSA slots (flexible / variable sized)
};

// A frame's body is one of two kinds, and every tier resumes either:
//  - flag clear: fn_ptr is generated code, `uint64_t (*)(BrassCoroFrame*)`
//    (a frame generated code created through the fixed-code entry,
//    brass_coro_create); it always resumes in that code;
//  - flag set: fn_ptr is a `const CoroBody*` (every frame an interpreter or
//    the program's tiered code creates); it resumes in the current best tier
//    of that body (resume_coro_body).
// A frame with neither (fn_ptr null) has no body any tier can run.
static constexpr uint32_t CORO_FLAG_BODY = 1u;

// How a resume asks the body to continue. The runtime only carries the
// word (BrassCoroFrame::resume_mode); a body reads it after a suspend and
// acts on it. The three a generator protocol needs are named here; a
// producer may define more.
enum class CoroResumeMode : uint32_t {
    Next = 0,    // continue with the resume value
    Throw = 1,   // throw the resume value at the suspend point
    Return = 2,  // return the resume value (running finally code on the way)
};

// A lowered coroutine body as its frames see it: the frame shape (how many
// slots, which of them the collector traces) and how to run it. One per
// (body, program); created on first use and never freed, so a frame's
// pointer to it stays valid for the frame's life, and a collection can
// always trace the frame. When the body's module is destroyed the
// descriptor forgets its Function (resuming a frame of it then is a hard
// error); when its program is destroyed it forgets the program and
// resumes in Tier 0.
struct CoroBody {
    std::string name;
    uint32_t slot_count = 1;
    // Bit i set: slot i holds a reference (gcref or tagged), traced as a
    // tagged word (the collector keeps a slot's high 16 bits). Unbounded.
    std::vector<uint64_t> ref_bits;
    // The lowered MIR body, null once its module is destroyed.
    std::atomic<const Function*> mir{nullptr};
    // The program whose tiers run it (its handle of `name`), or null: Tier 0.
    std::atomic<FunctionDispatchTable*> table{nullptr};

    bool slot_is_ref(uint32_t slot) const noexcept {
        return slot / 64 < ref_bits.size() && ((ref_bits[slot / 64] >> (slot % 64)) & 1) != 0;
    }
};

// The descriptor of the lowered body `body` in program `table` (null: no
// program; its frames always resume in Tier 0). Throws std::logic_error
// when `body` is not a lowered coroutine body.
const CoroBody& coro_body_of(const Function& body, FunctionDispatchTable* table);
// Called when `fn`'s module, or program `table`, goes away (see CoroBody).
void forget_coro_body_function(const Function* fn) noexcept;
void forget_coro_body_program(const FunctionDispatchTable* table) noexcept;

inline const CoroBody* coro_body(const BrassCoroFrame* frame) noexcept {
    return (frame->flags & CORO_FLAG_BODY) ? static_cast<const CoroBody*>(frame->fn_ptr) : nullptr;
}
// The lowered MIR body of a frame with a descriptor (null otherwise, or
// once its module is gone).
inline const Function* mir_coro_body(const BrassCoroFrame* frame) noexcept {
    const CoroBody* body = coro_body(frame);
    return body ? body->mir.load(std::memory_order_acquire) : nullptr;
}

// Creates a frame of `body` (at least `min_slots` slots) with no generated
// caller: an interpreter's roots reach the heap through its own scope and
// provider.
uintptr_t create_coro_frame(const CoroBody& body, uint32_t min_slots = 0);
// The same for the lowered MIR function `body` outside any program, with
// at least `slot_count` slots and the slots of `pointer_mask` traced too.
uintptr_t create_mir_coro_frame(const Function& body, uint32_t slot_count, uint64_t pointer_mask);

// Runs one step of a frame with a descriptor on this thread, in its body's
// current best tier: the program's native entry for it once tiered up,
// otherwise Tier 0 (the innermost interpreter running here, else a fresh
// one), counting the resume toward the body's tier-up. `frame_addr` must be
// a root of the caller: a collection the body triggers updates it. Returns
// the body's result bits (brass_coro_resume stores them).
uint64_t resume_coro_body(uintptr_t& frame_addr);
// The write barrier of a resume storing `value` in the frame's resume_arg
// (the body may keep it in a slot of an old frame): every resume path that
// writes resume_arg itself calls it.
void coro_resume_value_barrier(uintptr_t frame, uint64_t value) noexcept;
// The native code a resume of `frame` should run now, counting the resume
// toward tier-up; null when the body runs in Tier 0. For an interpreter
// that resumes a frame itself unless its body has native code.
void* coro_body_native_entry(const BrassCoroFrame* frame);

static constexpr size_t CORO_FRAME_HEADER_SIZE = 56; // 7 * 8 bytes
static constexpr uint32_t TYPE_TAG_CORO_FRAME  = 200;

static constexpr int32_t CORO_OFFSET_STATE_ID    = 0;
static constexpr int32_t CORO_OFFSET_IS_DONE     = 4;
static constexpr int32_t CORO_OFFSET_FN_PTR      = 8;
static constexpr int32_t CORO_OFFSET_YIELD_VAL   = 16;
static constexpr int32_t CORO_OFFSET_RESUME_ARG  = 24;
static constexpr int32_t CORO_OFFSET_SLOT_COUNT  = 32;
static constexpr int32_t CORO_OFFSET_FLAGS       = 36;
static constexpr int32_t CORO_OFFSET_RESUME_MODE = 40;
static constexpr int32_t CORO_OFFSET_AWAITER     = 48;
static constexpr int32_t CORO_OFFSET_SLOTS       = 56;
static_assert(CORO_OFFSET_SLOTS == static_cast<int32_t>(CORO_FRAME_HEADER_SIZE));

// Async stacks. A frame's awaiter is the frame that waits on it (set by the
// producer: an async function awaiting another's result, a generator
// delegating to another). While a frame runs, it is this thread's current
// coroutine; the chain from it through the awaiters is its async stack.
// Each entry: the frame, its body's name ("" for a fixed-code body) and its
// state (the suspend it last stopped at).
struct CoroStackEntry {
    uintptr_t frame = 0;
    std::string_view name;
    uint32_t state_id = 0;
};
// The innermost frame being resumed on this thread, or 0.
uintptr_t current_coro_frame() noexcept;
// Held by every resume path while it runs a body: `*frame_root` (a root
// slot of the resumer, so it follows the frame when it moves) is the
// thread's current frame until the scope ends.
class RunningCoroScope {
public:
    explicit RunningCoroScope(uintptr_t* frame_root);
    ~RunningCoroScope();
    RunningCoroScope(const RunningCoroScope&) = delete;
    RunningCoroScope& operator=(const RunningCoroScope&) = delete;
};
// `frame`, then its awaiter, then that one's, ... (a cycle ends the walk).
std::vector<CoroStackEntry> coro_async_stack(uintptr_t frame);
// The current frame's async stack (empty when no frame runs).
std::vector<CoroStackEntry> current_async_stack();

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
// The fixed-code entry generated code calls: allocates a frame whose body is
// always the code at `fn_ptr`, slot i traced when bit i of `pointer_mask`
// is set (any bit from 56 up traces every slot from 56 on; a body needing
// more precision uses brass_coro_create_body), with the calling frame's gcrefs among a
// collection's roots (as brass_gc_alloc).
uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask);
// The same with the generated frame named: its frame pointer and the return
// address into it (both 0: none, e.g. an interpreter's call).
uintptr_t brass_coro_create_at(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask,
                               uintptr_t caller_fp, uintptr_t caller_ip);
// The entry a program's tiered code calls: a frame of the body `body` (a
// `const CoroBody*`), resumed in that body's best tier, with its shape.
uintptr_t brass_coro_create_body(const void* body);
uintptr_t brass_coro_create_body_at(const void* body, uintptr_t caller_fp, uintptr_t caller_ip);
// Resumes the frame from C++ in mode Next: an exception the body throws
// leaves as a C++ exception.
uint64_t brass_coro_resume(uintptr_t coro_frame, uint64_t input_val);
// The same with a resume mode (CoroResumeMode, or the producer's own).
uint64_t brass_coro_resume_with(uintptr_t coro_frame, uint64_t input_val, uint32_t mode);
// The entry generated code calls (the JIT binds "brass_coro_resume" here):
// an exception the body throws is raised again natively (brass_throw), so the
// caller's landing pads see it.
uint64_t brass_coro_resume_from_generated(uintptr_t coro_frame, uint64_t input_val, uint32_t mode);
uint32_t brass_coro_is_done(uintptr_t coro_frame);
// The frame's awaiter link (CoroStackEntry), with the store's write
// barrier; 0 clears it. The awaiter is kept alive by the frame.
void brass_coro_set_awaiter(uintptr_t coro_frame, uintptr_t awaiter);
uintptr_t brass_coro_awaiter(uintptr_t coro_frame);
// Marks a registered frame finished and unregisters it; a finished frame is
// left as it is. A handle no live heap holds (its heap was torn down, or it
// is no coroutine frame) is a hard error and is not written through, as for
// brass_coro_resume and brass_coro_is_done.
void brass_coro_destroy(uintptr_t coro_frame);
}
