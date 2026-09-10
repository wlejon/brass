#include <brass/mir/osr.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <unordered_set>
#include <queue>
#include <algorithm>

namespace brass {

OsrTarget analyze_osr_target(Function& fn, BasicBlock* loop_header) {
    OsrTarget target;
    if (!loop_header) return target;

    fn.rebuild_cfg_predecessors();
    target.loop_header = loop_header;
    target.loop_header_id = loop_header->id();

    DominatorTree dom(fn);
    std::unordered_set<const BasicBlock*> loop_blocks;
    loop_blocks.insert(loop_header);

    // 1. Identify latches: predecessors of loop_header dominated by loop_header
    std::vector<BasicBlock*> latches;
    for (BasicBlock* pred : loop_header->predecessors()) {
        if (pred && dom.dominates(loop_header, pred)) {
            latches.push_back(pred);
        }
    }

    // 2. Discover natural loop blocks
    std::queue<BasicBlock*> q;
    for (BasicBlock* latch : latches) {
        if (latch != loop_header && loop_blocks.insert(latch).second) {
            q.push(latch);
        }
    }

    while (!q.empty()) {
        BasicBlock* curr = q.front();
        q.pop();
        for (BasicBlock* pred : curr->predecessors()) {
            if (pred && loop_blocks.insert(pred).second) {
                q.push(pred);
            }
        }
    }

    // 3. Collect live-ins:
    // First: all block parameters of loop_header (loop-carried phis)
    for (size_t i = 0; i < loop_header->param_count(); ++i) {
        Value* p = loop_header->param(i);
        if (p) {
            uint32_t slot = static_cast<uint32_t>(target.live_ins.size());
            target.live_ins.push_back(p);
            target.val_to_slot[p] = slot;
            target.live_in_details.push_back(OsrLiveIn{
                p, slot, true, static_cast<uint32_t>(i)
            });
        }
    }

    // Second: all values used inside the loop that are defined outside the loop
    std::unordered_set<const Value*> seen(target.live_ins.begin(), target.live_ins.end());
    std::vector<Value*> external_live_ins;

    auto check_val = [&](Value* v) {
        if (!v || seen.count(v)) return;
        bool defined_outside = false;
        if (v->is_block_param()) {
            if (loop_blocks.find(v->defining_block()) == loop_blocks.end()) {
                defined_outside = true;
            }
        } else if (v->is_instruction()) {
            const Instruction* def = v->defining_instruction();
            if (def && loop_blocks.find(def->parent()) == loop_blocks.end()) {
                defined_outside = true;
            }
        }
        if (defined_outside) {
            seen.insert(v);
            external_live_ins.push_back(v);
        }
    };

    for (const BasicBlock* bb : loop_blocks) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (Value* op : inst->operands()) {
                check_val(op);
            }
            for (Value* arg : inst->branch_target().args) {
                check_val(arg);
            }
            for (Value* arg : inst->true_target().args) {
                check_val(arg);
            }
            for (Value* arg : inst->false_target().args) {
                check_val(arg);
            }
            for (const auto& sc : inst->switch_cases()) {
                for (Value* arg : sc.target.args) {
                    check_val(arg);
                }
            }
            for (Value* sv : inst->state_map()) {
                check_val(sv);
            }
        }
    }

    // Sort external live-ins by SSA id for deterministic order
    std::sort(external_live_ins.begin(), external_live_ins.end(),
              [](const Value* a, const Value* b) {
                  return a->id() < b->id();
              });

    for (Value* v : external_live_ins) {
        uint32_t slot = static_cast<uint32_t>(target.live_ins.size());
        target.live_ins.push_back(v);
        target.val_to_slot[v] = slot;
        target.live_in_details.push_back(OsrLiveIn{
            v, slot, false, 0
        });
    }

    return target;
}

std::vector<OsrTarget> find_all_osr_targets(Function& fn) {
    std::vector<OsrTarget> targets;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);

    for (const auto& loop : loops.post_order_loops()) {
        if (loop && loop->header()) {
            targets.push_back(analyze_osr_target(fn, loop->header()));
        }
    }
    return targets;
}

} // namespace brass
