#include <brass/codegen/block_layout.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/branch_probability.hpp>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace brass::codegen {

namespace {

bool is_block_cold(const LirFunction& fn, const LirBlock& block, const BlockLayoutOptions& opts) {
    // Entry block is never cold
    if (!fn.blocks.empty() && fn.blocks[0].get() == &block) {
        return false;
    }

    if (opts.block_freq && opts.mir_function) {
        const BasicBlock* mb = opts.mir_function->get_block_by_name(block.name);
        if (mb && opts.block_freq->get_block_count(mb) == 0) {
            return true;
        }
    }

    // Check resume entries
    for (const auto& rp : fn.resume_entries) {
        if (rp.second == block.id) {
            return true;
        }
    }

    // Check name heuristics
    if (block.name.find("cold") != std::string::npos ||
        block.name.find("deopt") != std::string::npos ||
        block.name.find("resume") != std::string::npos) {
        return true;
    }

    // Check instructions
    for (const auto& inst : block.instructions) {
        if (!inst) continue;
        if (inst->opcode == LirOpcode::Safepoint && inst->deopt_reason != 0) {
            return true;
        }
        if (inst->opcode == LirOpcode::GuardExit) {
            return true;
        }
        if (!inst->exit_symbol.empty() || inst->callee_symbol == "brass_deopt_exit") {
            return true;
        }
    }

    return false;
}

struct BlockBranchTargets {
    std::vector<uint32_t> targets;
    uint32_t fallthrough_target = UINT32_MAX;
    bool single_unconditional_jump = false;
};

BlockBranchTargets get_branch_targets(const LirBlock& block) {
    BlockBranchTargets res;
    size_t n = block.instructions.size();
    if (n == 0) return res;

    // Check if ends with Jmp
    const auto& last = *block.instructions[n - 1];
    if (last.opcode == LirOpcode::Jmp && !last.uses.empty() && last.uses[0].is_label()) {
        uint32_t jmp_target = last.uses[0].label_id;

        // Search backwards for an earlier Jcc in the block
        int jcc_idx = -1;
        for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
            if (block.instructions[static_cast<size_t>(i)]->opcode == LirOpcode::Jcc) {
                jcc_idx = i;
                break;
            }
            if (block.instructions[static_cast<size_t>(i)]->opcode == LirOpcode::Jmp ||
                block.instructions[static_cast<size_t>(i)]->opcode == LirOpcode::Ret) {
                break;
            }
        }

        if (jcc_idx >= 0) {
            const auto& jcc = *block.instructions[static_cast<size_t>(jcc_idx)];
            if (!jcc.uses.empty() && jcc.uses[0].is_label()) {
                uint32_t jcc_target = jcc.uses[0].label_id;
                res.targets.push_back(jcc_target);
                res.targets.push_back(jmp_target);
                res.fallthrough_target = jmp_target;
                return res;
            }
        }

        res.targets.push_back(jmp_target);
        res.fallthrough_target = jmp_target;
        res.single_unconditional_jump = true;
        return res;
    }

    if (last.opcode == LirOpcode::Jcc && !last.uses.empty() && last.uses[0].is_label()) {
        res.targets.push_back(last.uses[0].label_id);
        return res;
    }

    // Fallback to CFG successors if present
    for (const LirBlock* succ : block.successors) {
        if (succ) {
            res.targets.push_back(succ->id);
        }
    }

    return res;
}

} // namespace

void optimize_block_layout(LirFunction& fn, const BlockLayoutOptions& opts) {
    if (fn.blocks.size() <= 1) return;

    // Ensure every block (except return/halt) has an explicit jump if it implicitly fell through
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        auto& b = fn.blocks[i];
        if (!b || b->instructions.empty()) continue;
        const auto& last = *b->instructions.back();
        if (last.opcode != LirOpcode::Jmp && last.opcode != LirOpcode::Ret && last.opcode != LirOpcode::GuardExit) {
            if (i + 1 < fn.blocks.size() && fn.blocks[i + 1]) {
                auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
                jmp->add_use(LirOperand::label(fn.blocks[i + 1]->id));
                b->append_inst(std::move(jmp));
            }
        }
    }

    std::unordered_map<uint32_t, LirBlock*> id_to_block;
    std::unordered_set<uint32_t> cold_blocks;

    for (const auto& b : fn.blocks) {
        if (!b) continue;
        id_to_block[b->id] = b.get();
        if (is_block_cold(fn, *b, opts)) {
            cold_blocks.insert(b->id);
        }
    }

    std::vector<LirBlock*> layout;
    std::unordered_set<uint32_t> placed;

    // Entry block must remain first (blocks[0])
    LirBlock* entry = fn.blocks[0].get();
    layout.push_back(entry);
    placed.insert(entry->id);

    LirBlock* curr = entry;

    size_t non_cold_count = 0;
    for (const auto& b : fn.blocks) {
        if (b && !cold_blocks.count(b->id)) {
            non_cold_count++;
        }
    }

    // Trace building loop for hot/regular blocks
    while (layout.size() < non_cold_count) {
        BlockBranchTargets bt = get_branch_targets(*curr);

        LirBlock* best_succ = nullptr;
        int best_score = -10000;

        for (uint32_t succ_id : bt.targets) {
            if (placed.count(succ_id)) continue;
            auto it = id_to_block.find(succ_id);
            if (it == id_to_block.end()) continue;
            LirBlock* s = it->second;

            int score = 0;
            if (cold_blocks.count(s->id)) {
                score -= 5000;
            } else {
                if (opts.branch_prob && opts.mir_function) {
                    const BasicBlock* curr_mb = opts.mir_function->get_block_by_name(curr->name);
                    const BasicBlock* s_mb = opts.mir_function->get_block_by_name(s->name);
                    if (curr_mb && s_mb) {
                        double prob = opts.branch_prob->get_edge_probability(curr_mb, s_mb);
                        uint64_t edge_cnt = opts.branch_prob->get_edge_count(curr_mb, s_mb);
                        score += static_cast<int>(prob * 10000.0) + (edge_cnt > 0 ? 500 : 0);
                    }
                }

                score += static_cast<int>(s->loop_depth) * 100;
                if (s->loop_depth > curr->loop_depth) {
                    score += 80;
                } else if (s->loop_depth == curr->loop_depth) {
                    score += 50;
                } else {
                    score -= 100;
                }

                if (s->id == bt.fallthrough_target && s->loop_depth >= curr->loop_depth) {
                    score += 40;
                }
                if (bt.single_unconditional_jump) {
                    score += 60;
                }
            }

            if (score > best_score) {
                best_score = score;
                best_succ = s;
            }
        }

        if (best_succ && !cold_blocks.count(best_succ->id)) {
            curr = best_succ;
            layout.push_back(curr);
            placed.insert(curr->id);
        } else {
            // End of current trace: pick best unplaced block to start new trace
            LirBlock* next_trace_start = nullptr;
            int max_depth = -1;

            for (const auto& b : fn.blocks) {
                if (!b || placed.count(b->id) || cold_blocks.count(b->id)) continue;
                int b_score = static_cast<int>(b->loop_depth) * 10;
                if (opts.block_freq && opts.mir_function) {
                    const BasicBlock* mb = opts.mir_function->get_block_by_name(b->name);
                    if (mb) {
                        b_score += static_cast<int>(std::min<uint64_t>(1000, opts.block_freq->get_block_count(mb)));
                    }
                }
                if (b_score > max_depth) {
                    max_depth = b_score;
                    next_trace_start = b.get();
                }
            }

            if (next_trace_start) {
                curr = next_trace_start;
                layout.push_back(curr);
                placed.insert(curr->id);
            } else {
                break;
            }
        }
    }

    // Place cold blocks (and any other remaining unplaced blocks) at the end
    if (opts.cold_block_at_end) {
        for (const auto& b : fn.blocks) {
            if (b && !placed.count(b->id)) {
                layout.push_back(b.get());
                placed.insert(b->id);
            }
        }
    }

    // Rebuild fn.blocks in the new layout order
    std::unordered_map<uint32_t, std::unique_ptr<LirBlock>> block_storage;
    for (auto& b : fn.blocks) {
        if (b) {
            uint32_t id = b->id;
            block_storage[id] = std::move(b);
        }
    }

    fn.blocks.clear();
    fn.blocks.reserve(layout.size());
    for (LirBlock* b : layout) {
        if (b && block_storage.count(b->id)) {
            fn.blocks.push_back(std::move(block_storage[b->id]));
        }
    }

    // Eliminate redundant unconditional jumps where target is immediate fall-through successor
    for (size_t i = 0; i + 1 < fn.blocks.size(); ++i) {
        auto& b = fn.blocks[i];
        if (!b || b->instructions.empty()) continue;
        uint32_t next_id = fn.blocks[i + 1]->id;
        const auto& last = *b->instructions.back();
        if (last.opcode == LirOpcode::Jmp && !last.uses.empty() &&
            last.uses[0].is_label() && last.uses[0].label_id == next_id) {
            b->instructions.pop_back();
        }
    }
}

void optimize_block_layout(LirFunction& fn) {
    optimize_block_layout(fn, BlockLayoutOptions());
}

} // namespace brass::codegen
