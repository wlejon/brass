#include "verifier_guards.hpp"
#include <brass/mir/block_liveness.hpp>
#include <brass/mir/opcodes.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace brass {

void verify_guard_resume_state(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
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

    const auto live_in = block_live_ins(fn);
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
