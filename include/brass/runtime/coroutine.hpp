#pragma once

#include <brass/gc/mini_cheney.hpp>
#include <brass/interpreter/value.hpp>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>
#include <string_view>

namespace brass::runtime {

struct BrassCoroFrame {
    uint32_t state_id;      // 0 = initial, 1..N = suspend states, ~0U = done
    uint32_t is_done;       // 1 if finished, 0 if active/suspended
    void* fn_ptr;           // Pointer to coroutine state machine function
    uint64_t yielded_val;   // Last yielded value
    uint64_t resume_arg;    // Input value passed into resume
    uint32_t slot_count;    // Total slots allocated
    uint32_t flags;         // Additional flags
    uint64_t slots[1];      // Spilled SSA slots (flexible / variable sized)
};

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
void set_coro_symbol_resolver(void* (*resolver)(const char*));

// Bronze IL Iterator & Async Helpers
uint64_t bronze_iter_open(uint64_t gen_or_obj);
uint64_t bronze_iter_step(uint64_t iter_handle);
uint64_t bronze_create_async_machine(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask, uint64_t env);
uint64_t bronze_async_start(uint64_t coro_frame, uint64_t arg);
uint64_t bronze_async_await(uint64_t coro_frame, uint64_t val);

} // namespace brass::runtime

extern "C" {
uintptr_t brass_coro_create(void* fn_ptr, uint32_t slot_count, uint64_t pointer_mask);
uint64_t brass_coro_resume(uintptr_t coro_frame, uint64_t input_val);
uint32_t brass_coro_is_done(uintptr_t coro_frame);
void brass_coro_destroy(uintptr_t coro_frame);
}
