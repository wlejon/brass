#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

// The one pass manager. Every optimizer entry point - the production
// pipeline, optimize_module / optimize_function, the loop pipeline, IPO, the
// tier-2 JIT pipeline, brass-opt and the fuzzer - builds a Pipeline (a
// declared list of named steps) and runs it here. A step name appears at most
// once in a pipeline: a pass meant to run twice is added twice under two
// names ("cfg_simplify", "cfg_simplify 2"), so a repeat is always a decision
// visible in the list, never an accident of two drivers.
namespace brass {

enum class PassScope : uint8_t {
    Module,    // runs once on the whole module
    Function,  // runs on each function the step's filter admits
};

enum class FunctionFilter : uint8_t {
    All,
    // Every function but the `__wrapper_*` ABI wrappers.
    NonWrapper,
    // Not a coroutine state machine or an ABI wrapper: the
    // functions loop transforms may restructure (loop_eligible below).
    LoopEligible,
};

struct PassStep {
    std::string name;
    PassScope scope = PassScope::Function;
    FunctionFilter filter = FunctionFilter::All;
    std::function<bool(Module&)> run_module;
    std::function<bool(Function&)> run_function;
    // Optional: runs on each function the step changed (a cleanup the step
    // needs, such as DCE after vectorization). Part of the step, reported
    // under its name.
    std::function<bool(Function&)> follow_up;
    // A function step that may run on many functions at once ("Threads",
    // below): it reads and writes only the function it is given, plus what
    // its `prepare` recorded. Off by default: a step not known to keep that
    // contract runs on one function at a time, in module order.
    bool concurrent = false;
    // Optional, for a concurrent step that reads other functions: runs on the
    // module, on the calling thread, before any step of the step's run of
    // concurrent steps starts, and records what the step will read of them.
    // Not run by run_pipeline(Function&); the step must then manage without.
    std::function<void(Module&)> prepare;
};

class Pipeline {
public:
    // Appends `step`. Throws std::logic_error when the pipeline already has
    // a step of that name, or the step has no callable for its scope.
    Pipeline& add(PassStep step);
    // Appends every step of `other`, with the same duplicate check.
    Pipeline& append(const Pipeline& other);
    // Puts `step` first.
    Pipeline& prepend(PassStep step);
    // A copy without the steps named in `names`; a name also matches its
    // numbered repeats ("cfg_simplify" drops "cfg_simplify 2").
    Pipeline without(const std::vector<std::string>& names) const;

    const std::vector<PassStep>& steps() const noexcept { return steps_; }
    std::vector<std::string> names() const;
    bool contains(std::string_view name) const noexcept;
    bool empty() const noexcept { return steps_.empty(); }

    // Set by pipelines that run loop transforms: running one over a whole
    // module marks it (Module::has_loop_optimizations), so backends do not
    // optimize its loops a second time. A run over one function does not.
    void set_marks_loop_optimized(bool marks) noexcept { marks_loop_optimized_ = marks; }
    bool marks_loop_optimized() const noexcept { return marks_loop_optimized_; }

private:
    std::vector<PassStep> steps_;
    bool marks_loop_optimized_ = false;
};

struct PassPipelineHooks {
    // Called with the step name just before it runs.
    std::function<void(std::string_view)> before_pass;
    // Called with the step name after it ran; returning false stops the
    // pipeline.
    std::function<bool(std::string_view)> after_pass;
};

struct PipelineResult {
    bool completed = true;  // false: a hook stopped the pipeline
    bool changed = false;   // some step reported a change
};

struct PipelineRunOptions {
    // Threads the function steps may use: 0 is the machine's hardware
    // threads, 1 runs everything on the calling thread. The output does not
    // depend on it ("Threads", below). A small module uses fewer than it may.
    size_t threads = 0;
    // A thread per this many blocks of the functions a stage runs on, at
    // most `threads`: below it, starting a thread costs more than it saves.
    size_t min_blocks_per_thread = 2000;
};

// Runs `pipeline` on `mod`. Function steps visit the functions the module had
// when the pipeline started (functions a step adds, such as
// auto-parallelization's kernels, are not optimized by later steps), with CFG
// predecessors rebuilt around every call.
//
// Threads. The steps fall into stages, run one after another: a module
// step; a function step that is not `concurrent`, run on each function in
// module order; or a maximal run of consecutive
// concurrent function steps, where each function runs the whole run's steps in
// pipeline order, functions in parallel. Within that run a step on one
// function sees every other function as it was when the run began (its
// `prepare` recorded that, and nothing else of them may be read), so the
// order functions run in, and the number of threads, cannot change the
// module the stage leaves. Each function runs inside a FunctionWorkScope
// (module.hpp): the IR and names its steps make go to its own arena and
// pool, not the module's, which are not thread-safe. With `before_pass` or
// `after_pass` hooks, a concurrent run is taken a step at a time instead
// (all functions through one step, then the hooks, then the next step): the
// same module results, with the hooks seeing it between steps.
//
// What a concurrent step must not do, and what keeps a step serial: add,
// remove or rename functions or declarations (Module::create_function,
// add_symbol_role, define_string_symbol ...); read another function's body
// or attributes that a step of the same run may change, other than through
// `prepare`; write shared statistics or caches (a step given a stats sink is
// built serial); depend on the addresses of IR objects for anything but
// identity (an order of pointers changes with where the arenas lie).
// That last rule binds every step and analysis, serial ones too: the arenas
// of a parallel run lie elsewhere than a serial run's, and under ASLR every
// run's do, so a walk of a pointer-keyed hash map that decides an order
// (LoopAnalysis's loops and lower_coroutines' reloads did) makes a build
// differ from run to run. Order by block or value order, or by id.
PipelineResult run_pipeline(Module& mod, const Pipeline& pipeline, const PassPipelineHooks& hooks = {},
                            const PipelineRunOptions& run = {});

// Runs `pipeline` on one function. Every step must be function-scoped;
// a module step is a std::logic_error.
PipelineResult run_pipeline(Function& fn, const Pipeline& pipeline, const PassPipelineHooks& hooks = {});

// The loop transforms' eligibility rule (FunctionFilter::LoopEligible).
bool loop_eligible(const Function& fn);

} // namespace brass
