#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/module.hpp>
#include <queue>
#include <algorithm>

namespace brass {

void LoopInfo::add_latch(BasicBlock* latch) {
    if (latch && std::find(latches_.begin(), latches_.end(), latch) == latches_.end()) {
        latches_.push_back(latch);
    }
}

void LoopInfo::add_block(BasicBlock* bb) {
    if (bb && block_set_.insert(bb).second) {
        blocks_.push_back(bb);
    }
}

bool LoopInfo::contains(const BasicBlock* bb) const noexcept {
    if (!bb) return false;
    return block_set_.find(bb) != block_set_.end();
}

bool LoopInfo::contains(const Instruction* inst) const noexcept {
    if (!inst) return false;
    return contains(inst->parent());
}

bool LoopInfo::is_loop_invariant(const Value* val) const noexcept {
    if (!val) return true;
    if (val->is_block_param()) {
        return !contains(val->defining_block());
    }
    if (val->is_instruction()) {
        const Instruction* def_inst = val->defining_instruction();
        return !def_inst || !contains(def_inst->parent());
    }
    return true;
}

size_t LoopInfo::depth() const noexcept {
    return 1 + (parent_ ? parent_->depth() : 0);
}

LoopAnalysis::LoopAnalysis(Function& fn, const DominatorTree& dom) {
    discover_loops(fn, dom);
}

void LoopAnalysis::discover_loops(Function& fn, const DominatorTree& dom) {
    std::unordered_map<BasicBlock*, std::unique_ptr<LoopInfo>> header_to_loop;

    // 1. Find all back-edges Tail -> Header where Header dominates Tail
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || !dom.is_reachable(bb)) continue;

        for (BasicBlock* succ : bb->successors()) {
            if (!succ || !dom.is_reachable(succ)) continue;

            if (dom.dominates(succ, bb)) {
                // succ is loop header, bb is loop latch
                BasicBlock* header = succ;
                BasicBlock* latch = bb;

                auto it = header_to_loop.find(header);
                if (it == header_to_loop.end()) {
                    auto loop = std::make_unique<LoopInfo>(header);
                    loop->add_block(header);
                    loop->add_latch(latch);
                    header_to_loop[header] = std::move(loop);
                } else {
                    it->second->add_latch(latch);
                }

                LoopInfo* cur_loop = header_to_loop[header].get();

                // 2. Discover all natural loop blocks by backward search from latch to header
                std::queue<BasicBlock*> q;
                if (latch != header) {
                    cur_loop->add_block(latch);
                    q.push(latch);
                }

                while (!q.empty()) {
                    BasicBlock* curr = q.front();
                    q.pop();

                    for (BasicBlock* pred : curr->predecessors()) {
                        if (!pred || !dom.is_reachable(pred)) continue;
                        if (!cur_loop->contains(pred)) {
                            cur_loop->add_block(pred);
                            q.push(pred);
                        }
                    }
                }
            }
        }
    }

    // 3. Collect all discovered loops
    std::vector<std::unique_ptr<LoopInfo>> all_loops;
    all_loops.reserve(header_to_loop.size());
    for (auto& [hdr, loop] : header_to_loop) {
        all_loops.push_back(std::move(loop));
    }

    // 4. Build loop hierarchy
    // Sort loops by block count ascending (smaller inner loops first)
    std::sort(all_loops.begin(), all_loops.end(), [](const auto& a, const auto& b) {
        return a->blocks().size() < b->blocks().size();
    });

    std::vector<bool> is_sub_loop(all_loops.size(), false);

    for (size_t i = 0; i < all_loops.size(); ++i) {
        for (size_t j = i + 1; j < all_loops.size(); ++j) {
            // Check if all_loops[i] is contained in all_loops[j]
            bool contained = true;
            for (BasicBlock* b : all_loops[i]->blocks()) {
                if (!all_loops[j]->contains(b)) {
                    contained = false;
                    break;
                }
            }
            if (contained) {
                // all_loops[j] is the smallest enclosing parent for all_loops[i]
                is_sub_loop[i] = true;
                break;
            }
        }
    }

    // Now organize into parents and sub-loops
    // For each loop i that is a sub-loop, find its direct parent (the smallest loop j > i containing it)
    for (size_t i = 0; i < all_loops.size(); ++i) {
        if (!is_sub_loop[i]) continue;
        for (size_t j = i + 1; j < all_loops.size(); ++j) {
            bool contained = true;
            for (BasicBlock* b : all_loops[i]->blocks()) {
                if (!all_loops[j]->contains(b)) {
                    contained = false;
                    break;
                }
            }
            if (contained) {
                all_loops[i]->set_parent(all_loops[j].get());
                break;
            }
        }
    }

    // Put sub-loops into their parent's sub_loops list
    // Iterate from largest to smallest to populate top-level and sub-loops
    for (size_t i = 0; i < all_loops.size(); ++i) {
        if (!is_sub_loop[i]) {
            top_level_loops_.push_back(std::move(all_loops[i]));
        }
    }

    for (size_t i = 0; i < all_loops.size(); ++i) {
        if (all_loops[i] && all_loops[i]->parent()) {
            all_loops[i]->parent()->add_sub_loop(std::move(all_loops[i]));
        }
    }
}

std::vector<LoopInfo*> LoopAnalysis::post_order_loops() const {
    std::vector<LoopInfo*> result;

    auto collect_post_order = [&](auto& self, LoopInfo* loop) -> void {
        if (!loop) return;
        for (const auto& sub : loop->sub_loops()) {
            self(self, sub.get());
        }
        result.push_back(loop);
    };

    for (const auto& top : top_level_loops_) {
        collect_post_order(collect_post_order, top.get());
    }

    return result;
}

BasicBlock* LoopAnalysis::ensure_preheader(Function& fn, LoopInfo& loop) {
    BasicBlock* header = loop.header();
    if (!header) return nullptr;

    // Check outside predecessors (predecessors of header not in loop)
    std::vector<BasicBlock*> outside_preds;
    for (BasicBlock* pred : header->predecessors()) {
        if (pred && !loop.contains(pred)) {
            outside_preds.push_back(pred);
        }
    }

    if (outside_preds.empty()) {
        // Loop header is unreachable or has no outside predecessors
        return nullptr;
    }

    // If there is exactly 1 outside predecessor, check if it's already a dedicated preheader
    if (outside_preds.size() == 1) {
        BasicBlock* sole_pred = outside_preds[0];
        if (sole_pred != fn.entry_block() || fn.entry_block()->successors().size() == 1) {
            if (sole_pred->successors().size() == 1 && sole_pred->terminator() && sole_pred->terminator()->opcode() == Opcode::br) {
                loop.set_preheader(sole_pred);
                return sole_pred;
            }
        }
    }

    // Create a new dedicated preheader basic block
    std::string ph_name = std::string(header->name()) + "_preheader";
    BasicBlock* preheader = fn.parent() ? fn.parent()->arena().make<BasicBlock>(fn.next_block_id(), fn.parent()->string_pool().intern(ph_name))
                                       : new BasicBlock(fn.next_block_id(), ph_name);
    preheader->set_parent(&fn);

    // Insert preheader immediately before header in fn.blocks()
    auto& blocks = fn.blocks();
    auto it = std::find(blocks.begin(), blocks.end(), header);
    if (it != blocks.end()) {
        blocks.insert(it, preheader);
    } else {
        blocks.push_back(preheader);
    }

    // Create block parameters in preheader matching header's parameters
    std::vector<Value*> ph_params;
    ph_params.reserve(header->param_count());
    for (size_t i = 0; i < header->param_count(); ++i) {
        Type ptype = header->param(i)->type();
        Value* p = fn.parent() ? fn.parent()->arena().make<Value>(fn.next_value_id(), ptype, ValueKind::BlockParam)
                               : new Value(fn.next_value_id(), ptype, ValueKind::BlockParam);
        preheader->add_param(p);
        ph_params.push_back(p);
    }

    // Redirect all outside predecessors to preheader
    for (BasicBlock* pred : outside_preds) {
        Instruction* term = pred->terminator();
        if (!term) continue;

        if (term->opcode() == Opcode::br) {
            if (term->branch_target().block == header) {
                term->branch_target().block = preheader;
            }
        } else if (term->opcode() == Opcode::br_if) {
            if (term->true_target().block == header) {
                term->true_target().block = preheader;
            }
            if (term->false_target().block == header) {
                term->false_target().block = preheader;
            }
        }
    }

    // Create unconditional branch in preheader to header passing ph_params
    Instruction* br_inst = fn.parent() ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
                                       : new Instruction(Opcode::br, Type::void_type());
    br_inst->set_branch_target(BranchTarget(header, std::move(ph_params)));
    preheader->append_instruction(br_inst);

    fn.rebuild_cfg_predecessors();
    loop.set_preheader(preheader);

    return preheader;
}

} // namespace brass
