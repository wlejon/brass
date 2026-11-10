#include <brass/mir/pass_manager.hpp>
#include <brass/core/parallel_for.hpp>
#include <brass/core/phase_times.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <stdexcept>
#include <unordered_set>

namespace brass {

namespace {

bool matches(std::string_view step, std::string_view name) {
    if (step == name) return true;
    // "cfg_simplify 2" is a numbered repeat of "cfg_simplify".
    return step.size() > name.size() && step.substr(0, name.size()) == name && step[name.size()] == ' ';
}

void check_step(const PassStep& step) {
    if (step.name.empty()) throw std::logic_error("pass step without a name");
    const bool callable = step.scope == PassScope::Module ? static_cast<bool>(step.run_module)
                                                          : static_cast<bool>(step.run_function);
    if (!callable) throw std::logic_error("pass step '" + step.name + "' has nothing to run for its scope");
}

bool admitted(const PassStep& step, const Function& fn) {
    switch (step.filter) {
        case FunctionFilter::All: return true;
        case FunctionFilter::NonWrapper: return !fn.name().starts_with("__wrapper_");
        case FunctionFilter::LoopEligible: return loop_eligible(fn);
    }
    return true;
}

bool run_on_function(const PassStep& step, Function& fn, uint8_t& preds_valid) {
    if (!admitted(step, fn)) return false;
    if (!preds_valid) {
        fn.rebuild_cfg_predecessors();
        preds_valid = 1;
    }
    bool changed = step.run_function(fn);
    if (changed) {
        fn.rebuild_cfg_predecessors();
        if (step.follow_up) {
            if (step.follow_up(fn)) fn.rebuild_cfg_predecessors();
        }
    }
    return changed;
}

} // namespace

Pipeline& Pipeline::add(PassStep step) {
    check_step(step);
    if (contains(step.name)) {
        throw std::logic_error("pipeline already runs a step named '" + step.name +
                               "'; name a deliberate repeat (e.g. '" + step.name + " 2')");
    }
    steps_.push_back(std::move(step));
    return *this;
}

Pipeline& Pipeline::append(const Pipeline& other) {
    for (const PassStep& step : other.steps_) add(step);
    marks_loop_optimized_ |= other.marks_loop_optimized_;
    return *this;
}

Pipeline& Pipeline::prepend(PassStep step) {
    check_step(step);
    if (contains(step.name)) {
        throw std::logic_error("pipeline already runs a step named '" + step.name + "'");
    }
    steps_.insert(steps_.begin(), std::move(step));
    return *this;
}

Pipeline Pipeline::without(const std::vector<std::string>& names) const {
    Pipeline out;
    out.marks_loop_optimized_ = marks_loop_optimized_;
    for (const PassStep& step : steps_) {
        bool drop = false;
        for (const std::string& n : names) drop |= matches(step.name, n);
        if (!drop) out.steps_.push_back(step);
    }
    return out;
}

std::vector<std::string> Pipeline::names() const {
    std::vector<std::string> out;
    out.reserve(steps_.size());
    for (const PassStep& step : steps_) out.push_back(step.name);
    return out;
}

bool Pipeline::contains(std::string_view name) const noexcept {
    for (const PassStep& step : steps_) {
        if (step.name == name) return true;
    }
    return false;
}

bool loop_eligible(const Function& fn) {
    return fn.resume_points().empty() && !fn.name().starts_with("__wrapper_");
}

namespace {

// Indices of the snapshot's functions still in the module (a module step may
// have removed some).
std::vector<size_t> live_function_indices(const Module& mod, const std::vector<Function*>& fns) {
    const std::unordered_set<const Function*> live(mod.functions().begin(), mod.functions().end());
    std::vector<size_t> out;
    out.reserve(fns.size());
    for (size_t i = 0; i < fns.size(); ++i) {
        if (fns[i] && live.count(fns[i])) out.push_back(i);
    }
    return out;
}

// Runs steps [first, last) of a concurrent stage on every live function:
// each function through all of them, functions on up to `threads` threads.
// The changed flags land per function and are or-ed in function order.
bool run_concurrent(const std::vector<PassStep>& steps, size_t first, size_t last,
                    const std::vector<Function*>& fns, const std::vector<size_t>& live,
                    std::vector<uint8_t>& preds_valid, const PipelineRunOptions& run) {
    std::vector<size_t> weights(live.size());
    size_t total = 0;
    for (size_t i = 0; i < live.size(); ++i) {
        weights[i] = fns[live[i]]->block_count() + 1;
        total += weights[i];
    }
    const size_t n = threads_for_work(run.threads, live.size(), total, run.min_blocks_per_thread);
    std::vector<uint8_t> changed(live.size(), 0);
    parallel_for_each(heaviest_first(weights), n, [&](size_t i) {
        const size_t fn_idx = live[i];
        Function& fn = *fns[fn_idx];
        FunctionWorkScope work(fn);
        bool any = false;
        for (size_t s = first; s < last; ++s) {
            PhaseTimeScope timing(steps[s].name);
            any |= run_on_function(steps[s], fn, preds_valid[fn_idx]);
        }
        changed[i] = any ? 1 : 0;
    });
    bool out = false;
    for (uint8_t c : changed) out |= c != 0;
    return out;
}

} // namespace

PipelineResult run_pipeline(Module& mod, const Pipeline& pipeline, const PassPipelineHooks& hooks,
                            const PipelineRunOptions& run) {
    PipelineResult result;
    if (pipeline.marks_loop_optimized()) mod.set_has_loop_optimizations(true);
    const std::vector<Function*> fns = mod.functions();
    std::vector<uint8_t> preds_valid(fns.size(), 0);
    const std::vector<PassStep>& steps = pipeline.steps();
    const bool stepwise = static_cast<bool>(hooks.before_pass) || static_cast<bool>(hooks.after_pass);
    // Runs the hooks around one step; false: a hook stopped the pipeline.
    auto after = [&](const PassStep& step) { return !hooks.after_pass || hooks.after_pass(step.name); };

    for (size_t i = 0; i < steps.size();) {
        const PassStep& step = steps[i];
        if (step.scope == PassScope::Module || !step.concurrent) {
            if (hooks.before_pass) hooks.before_pass(step.name);
            {
                PhaseTimeScope timing(step.name);
                if (step.scope == PassScope::Module) {
                    const bool mod_changed = step.run_module(mod);
                    result.changed |= mod_changed;
                    if (mod_changed) preds_valid.assign(fns.size(), 0);
                } else {
                    for (size_t fn_idx : live_function_indices(mod, fns)) {
                        result.changed |= run_on_function(step, *fns[fn_idx], preds_valid[fn_idx]);
                    }
                }
            }
            if (!after(step)) {
                result.completed = false;
                return result;
            }
            ++i;
            continue;
        }
        // A maximal run of concurrent function steps: one stage.
        size_t end = i;
        while (end < steps.size() && steps[end].scope == PassScope::Function && steps[end].concurrent) ++end;
        for (size_t s = i; s < end; ++s) {
            if (steps[s].prepare) steps[s].prepare(mod);
        }
        const std::vector<size_t> stage_live = live_function_indices(mod, fns);
        if (!stepwise) {
            result.changed |= run_concurrent(steps, i, end, fns, stage_live, preds_valid, run);
            i = end;
            continue;
        }
        for (; i < end; ++i) {
            if (hooks.before_pass) hooks.before_pass(steps[i].name);
            result.changed |= run_concurrent(steps, i, i + 1, fns, stage_live, preds_valid, run);
            if (!after(steps[i])) {
                result.completed = false;
                return result;
            }
        }
    }
    return result;
}

PipelineResult run_pipeline(Function& fn, const Pipeline& pipeline, const PassPipelineHooks& hooks) {
    for (const PassStep& step : pipeline.steps()) {
        if (step.scope != PassScope::Function) {
            throw std::logic_error("pipeline step '" + step.name + "' is module-scoped and cannot run on one function");
        }
    }
    PipelineResult result;
    uint8_t preds_valid = 0;
    for (const PassStep& step : pipeline.steps()) {
        if (hooks.before_pass) hooks.before_pass(step.name);
        {
            PhaseTimeScope timing(step.name);
            result.changed |= run_on_function(step, fn, preds_valid);
        }
        if (hooks.after_pass && !hooks.after_pass(step.name)) {
            result.completed = false;
            return result;
        }
    }
    return result;
}

} // namespace brass
