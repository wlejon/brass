#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct CoroTransformStats {
    size_t coroutines_transformed = 0;
    size_t suspend_points_transformed = 0;
    size_t variables_spilled = 0;
};

struct CoroTransformOptions {
    uint32_t first_slot_index = 0;
    CoroTransformStats* stats = nullptr;
};

// Frame shape of a lowered coroutine body, as `coro_create` must allocate it.
struct CoroFrameLayout {
    uint32_t slot_count = 0;   // slots the body addresses (at least 1)
    uint64_t pointer_mask = 0; // bit i set: slot i holds a pointer/gcref
};

// True when `fn` has the lowered coroutine-body ABI that every tier executes:
// no `coro_suspend` left and exactly one pointer/gcref parameter (the frame).
bool is_lowered_coro_body(const Function& fn);

// The frame slots a lowered body loads/stores through its frame parameter.
CoroFrameLayout compute_coro_frame_layout(const Function& fn);

// Coroutine State Machine Transformation Pass
// Transforms coroutine bodies (functions containing `coro_suspend`, and every
// `coro_create` target in the module) into stackless resumable state machines
// that take only the frame:
// - `coro_create @f(a0, ..., an)` puts argument i in frame slot i; the body
//   loads its original parameters from those slots on entry
// - Live SSA values across suspends are spilled to GC-tracked frame slots
//   (after the argument slots)
// - Resume blocks are created and entry dispatch switch is injected
// - Frame completion marks `is_done = 1` and `state_id = ~0U`
class CoroTransformPass {
public:
    explicit CoroTransformPass(const CoroTransformOptions& options = {})
        : options_(options) {}

    // `force` lowers `fn` even when it has no `coro_suspend` (a coroutine
    // body that runs to completion on its first resume).
    //
    // `create_arg_count` is the number of arguments the module's
    // `coro_create @fn(...)` pass (run_on_module supplies it). With it, the
    // body's parameters are exactly those arguments when the counts match,
    // or a leading pointer/gcref frame parameter followed by them when the
    // body has one more; anything else is a std::logic_error. Without it
    // (-1: no coro_create names `fn`), a leading pointer/gcref parameter is
    // taken to be the frame, as bodies built with an explicit frame
    // parameter declare it.
    bool run_on_function(Function& fn, bool force = false, int64_t create_arg_count = -1);
    bool run_on_module(Module& mod);

    const CoroTransformOptions& options() const noexcept { return options_; }

private:
    CoroTransformOptions options_;
};

} // namespace brass
