#include <brass/mir/pass_manager.hpp>
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

bool run_on_function(const PassStep& step, Function& fn) {
    if (!admitted(step, fn)) return false;
    fn.rebuild_cfg_predecessors();
    bool changed = step.run_function(fn);
    fn.rebuild_cfg_predecessors();
    if (changed && step.follow_up) {
        step.follow_up(fn);
        fn.rebuild_cfg_predecessors();
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
    if (!fn.resume_points().empty() || fn.name().starts_with("__wrapper_")) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::osr_entry) return false;
        }
    }
    return true;
}

PipelineResult run_pipeline(Module& mod, const Pipeline& pipeline, const PassPipelineHooks& hooks) {
    PipelineResult result;
    if (pipeline.marks_loop_optimized()) mod.set_has_loop_optimizations(true);
    const std::vector<Function*> fns = mod.functions();
    for (const PassStep& step : pipeline.steps()) {
        if (hooks.before_pass) hooks.before_pass(step.name);
        if (step.scope == PassScope::Module) {
            result.changed |= step.run_module(mod);
        } else {
            // A module step may have removed functions since the snapshot.
            const std::unordered_set<const Function*> live(mod.functions().begin(), mod.functions().end());
            for (Function* fn : fns) {
                if (fn && live.count(fn)) result.changed |= run_on_function(step, *fn);
            }
        }
        if (hooks.after_pass && !hooks.after_pass(step.name)) {
            result.completed = false;
            return result;
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
    for (const PassStep& step : pipeline.steps()) {
        if (hooks.before_pass) hooks.before_pass(step.name);
        result.changed |= run_on_function(step, fn);
        if (hooks.after_pass && !hooks.after_pass(step.name)) {
            result.completed = false;
            return result;
        }
    }
    return result;
}

} // namespace brass
