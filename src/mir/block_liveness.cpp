// Block live-in sets (block_liveness.hpp).
#include <brass/mir/block_liveness.hpp>
#include <brass/mir/uses.hpp>

#include <vector>

namespace brass {

std::unordered_map<const BasicBlock*, std::unordered_set<const Value*>> block_live_ins(const Function& fn) {
    using ValueSet = std::unordered_set<const Value*>;
    struct BlockInfo {
        ValueSet use;
        ValueSet def;
        std::vector<const BasicBlock*> succs;
    };
    std::unordered_map<const BasicBlock*, BlockInfo> info;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        BlockInfo& bi = info[bb];
        for (const Value* p : bb->params()) bi.def.insert(p);
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            for_each_use(*inst, [&](Value* v) {
                // Only SSA values are live; a constant is defined like any
                // other instruction, so it is one too.
                if (!bi.def.count(v)) bi.use.insert(v);
            });
            if (inst->result()) bi.def.insert(inst->result());
        }
        for (const BasicBlock* s : bb->successors()) {
            if (s) bi.succs.push_back(s);
        }
    }
    std::unordered_map<const BasicBlock*, ValueSet> live_in;
    for (auto& [bb, bi] : info) live_in[bb] = bi.use;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = fn.blocks().rbegin(); it != fn.blocks().rend(); ++it) {
            const BasicBlock* bb = *it;
            if (!bb) continue;
            BlockInfo& bi = info[bb];
            ValueSet& in = live_in[bb];
            for (const BasicBlock* s : bi.succs) {
                for (const Value* v : live_in[s]) {
                    if (!bi.def.count(v) && in.insert(v).second) changed = true;
                }
            }
        }
    }
    return live_in;
}

} // namespace brass
