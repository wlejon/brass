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

// Runs `pipeline` on `mod`, step by step. Function steps visit the functions
// the module had when the pipeline started (functions a step adds, such as
// auto-parallelization's kernels, are not optimized by later steps), with
// CFG predecessors rebuilt around every call.
PipelineResult run_pipeline(Module& mod, const Pipeline& pipeline, const PassPipelineHooks& hooks = {});

// Runs `pipeline` on one function. Every step must be function-scoped;
// a module step is a std::logic_error.
PipelineResult run_pipeline(Function& fn, const Pipeline& pipeline, const PassPipelineHooks& hooks = {});

// The loop transforms' eligibility rule (FunctionFilter::LoopEligible).
bool loop_eligible(const Function& fn);

} // namespace brass
