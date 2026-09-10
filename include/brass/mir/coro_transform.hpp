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

// Coroutine State Machine Transformation Pass
// Transforms functions containing `coro_suspend` into stackless resumable state machines:
// - Live SSA values across suspends are spilled to GC-tracked frame slots
// - Resume blocks are created and entry dispatch switch is injected
// - Frame completion marks `is_done = 1` and `state_id = ~0U`
class CoroTransformPass {
public:
    explicit CoroTransformPass(const CoroTransformOptions& options = {})
        : options_(options) {}

    bool run_on_function(Function& fn);
    bool run_on_module(Module& mod);

    const CoroTransformOptions& options() const noexcept { return options_; }

private:
    CoroTransformOptions options_;
};

} // namespace brass
