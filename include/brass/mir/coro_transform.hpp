#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

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
    // Bit i (word i / 64, bit i % 64) set: slot i holds a reference (a
    // pointer, gcref or tagged value). Unbounded.
    std::vector<uint64_t> ref_bits;
    // The first 64 slots of ref_bits (what the fixed-code entry,
    // brass_coro_create, takes), and whether it describes them all.
    uint64_t pointer_mask = 0;
    bool fits_pointer_mask = true;
};

// True when `fn` has the lowered coroutine-body ABI that every tier executes:
// no `coro_suspend` left and exactly one pointer/gcref parameter (the frame).
bool is_lowered_coro_body(const Function& fn);

// The frame slots a lowered body loads/stores through its frame parameter.
CoroFrameLayout compute_coro_frame_layout(const Function& fn);

// Frame slots are 8 bytes; a value of type `t` occupies this many
// consecutive ones (2 for a v128, 4 for a v256). Frame accesses of vector
// slots are unaligned full-width loads/stores.
uint32_t coro_slot_count(Type t);

// Slots `coro_create`'s arguments fill: argument i starts at the sum of
// coro_slot_count over arguments 0..i-1.
uint32_t coro_create_slot_count(const Instruction& create);

// Coroutine State Machine Transformation Pass
// Transforms coroutine bodies (functions containing `coro_suspend`, and every
// `coro_create` target in the module) into stackless resumable state machines
// that take only the frame:
// - `coro_create @f(a0, ..., an)` puts its arguments in consecutive frame
//   slots (coro_slot_count each); the body loads its original parameters
//   from those slots on entry
// - yielded, resume and return values must fit 8 bytes (std::logic_error
//   otherwise; the verifier rejects wide coro_suspend/coro_resume values)
// - Live SSA values across suspends are spilled to GC-tracked frame slots
//   (after the argument slots), a reference's store followed by its
//   write barrier (the frame may be old by then); any number of them
// - A body may declare a leading frame parameter and read the frame's
//   header through it (CORO_OFFSET_RESUME_MODE after a suspend is the mode
//   the resume passed); it is the lowered body's frame
// - Resume blocks are created and entry dispatch switch is injected
// - Frame completion marks `is_done = 1` and `state_id = ~0U`
// Lowers every coroutine body of `mod` that is not lowered yet (a body
// already lowered is left alone), so a module can be handed to any tier:
// what the tiered pipeline, the JIT engine and the embedding compiler run
// before they execute a module. Returns whether anything changed.
bool lower_coroutines(Module& mod);

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
