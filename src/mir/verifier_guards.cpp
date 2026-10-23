#include "verifier_guards.hpp"
#include <brass/mir/block_liveness.hpp>
#include <brass/mir/opcodes.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

void verify_guard_resume_state(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    // A guard's resume id names the one Tier-0 guard a deopt resumes at.
    // Guards may share an id only as copies of one guard (code duplication
    // in an optimized clone): same exit label and same state value types.
    {
        std::unordered_map<uint32_t, const Instruction*> first_guard;
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (!inst || inst->opcode() != Opcode::guard) continue;
                auto [it, fresh] = first_guard.emplace(inst->resume_id(), inst);
                if (fresh) continue;
                const Instruction* g = it->second;
                bool same = g->symbol() == inst->symbol() && g->state_map().size() == inst->state_map().size();
                for (size_t i = 0; same && i < g->state_map().size(); ++i) {
                    const Value* a = g->state_map()[i];
                    const Value* c = inst->state_map()[i];
                    same = a && c && a->type() == c->type();
                }
                if (!same) {
                    report_error(fn_prefix + "Two different guards have resume id " +
                                 std::to_string(inst->resume_id()) + " (exit labels '" + std::string(g->symbol()) +
                                 "' and '" + std::string(inst->symbol()) + "'); each guard needs its own id.");
                }
            }
        }
    }

    std::vector<const Instruction*> resuming;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::guard && !fn.guard_exit_stub(*inst) &&
                fn.get_resume_target(inst->resume_id())) {
                resuming.push_back(inst);
            }
        }
    }
    if (resuming.empty()) return;

    std::vector<const BasicBlock*> targets;
    targets.reserve(resuming.size());
    for (const Instruction* guard : resuming) targets.push_back(fn.get_resume_target(guard->resume_id()));
    const auto live_in = block_live_ins(fn, targets);
    for (const Instruction* guard : resuming) {
        const BasicBlock* target = fn.get_resume_target(guard->resume_id());
        auto it = live_in.find(target);
        if (it == live_in.end()) continue;
        const std::unordered_set<const Value*> state(guard->state_map().begin(), guard->state_map().end());
        std::vector<uint32_t> missing;
        for (const Value* v : it->second) {
            if (!state.count(v)) missing.push_back(v->id());
        }
        if (missing.empty()) continue;
        std::sort(missing.begin(), missing.end());
        std::string ids;
        for (uint32_t id : missing) ids += (ids.empty() ? "%" : ", %") + std::to_string(id);
        report_error(fn_prefix + "Guard (resume id " + std::to_string(guard->resume_id()) + ") resumes at '" +
                     std::string(target->name()) + "', which reads " + ids +
                     " (value ids) that is neither a parameter of the block nor one of the guard's state values: "
                     "a lower tier resuming there after a deopt would not have it. Add it to the guard's state.");
    }
}

} // namespace brass
