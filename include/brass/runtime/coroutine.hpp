#pragma once

#include <brass/gc/mini_cheney.hpp>
#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <cstddef>
#include <functional>
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

// Microtask Queue for async/await scheduling
class MicrotaskQueue {
public:
    using Task = std::function<void()>;

    void enqueue(Task task);
    void enqueue_coro(BrassCoroFrame* frame, uint64_t input_val);
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
    std::vector<BrassCoroFrame*> awaiting_frames_;
};

// Root tracking for Cheney moving GC
void register_active_coro_frame(BrassCoroFrame* frame);
void unregister_active_coro_frame(BrassCoroFrame* frame);
bool is_active_coro_frame(uintptr_t frame);
void visit_active_coro_frames(const std::function<void(uintptr_t*)>& visitor);
void append_active_coro_roots(std::vector<uintptr_t*>& roots);

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
void brass_coro_destroy(uintptr_t coro_frame);
}
