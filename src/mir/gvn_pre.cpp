#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/critical_edge.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/memory_ssa.hpp>
#include "gvn_pre_dataflow.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace brass {

std::string GvnPreStats::format_report() const {
    std::ostringstream ss;
    ss << "[gvn-pre] Expressions hoisted: " << expressions_hoisted << "\n"
       << "[gvn-pre] Expressions eliminated: " << expressions_eliminated << "\n"
       << "[gvn-pre] Critical edges split: " << critical_edges_split << "\n"
       << "[gvn-pre] Block parameters inserted: " << block_params_inserted;
    return ss.str();
}

void GvnPreStats::dump(std::ostream& os) const {
    os << format_report() << "\n";
}

namespace {

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;

            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) {
                    inst->set_operand(i, new_val);
                }
            }

            if (inst->opcode() == Opcode::br) {
                for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                    if (inst->branch_target().args[i] == old_val) {
                        inst->branch_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::br_if) {
                for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                    if (inst->true_target().args[i] == old_val) {
                        inst->true_target().args[i] = new_val;
                    }
                }
                for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                    if (inst->false_target().args[i] == old_val) {
                        inst->false_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::switch_) {
                for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                    if (inst->default_target().args[i] == old_val) {
                        inst->default_target().args[i] = new_val;
                    }
                }
                for (auto& sc : inst->switch_cases()) {
                    for (size_t i = 0; i < sc.target.args.size(); ++i) {
                        if (sc.target.args[i] == old_val) {
                            sc.target.args[i] = new_val;
                        }
                    }
                }
            } else if (inst->opcode() == Opcode::invoke) {
                for (size_t i = 0; i < inst->normal_target().args.size(); ++i) {
                    if (inst->normal_target().args[i] == old_val) {
                        inst->normal_target().args[i] = new_val;
                    }
                }
                for (size_t i = 0; i < inst->unwind_target().args.size(); ++i) {
                    if (inst->unwind_target().args[i] == old_val) {
                        inst->unwind_target().args[i] = new_val;
                    }
                }
            }

            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) {
                    inst->state_map()[i] = new_val;
                }
            }
        }
    }
}

void append_branch_arg_to_target(Instruction* term, BasicBlock* target_bb, Value* arg) {
    if (!term || !target_bb || !arg) return;

    if (term->opcode() == Opcode::br) {
        if (term->branch_target().block == target_bb) {
            term->branch_target().args.push_back(arg);
        }
    } else if (term->opcode() == Opcode::br_if) {
        if (term->true_target().block == target_bb) {
            term->true_target().args.push_back(arg);
        }
        if (term->false_target().block == target_bb) {
            term->false_target().args.push_back(arg);
        }
    } else if (term->opcode() == Opcode::switch_) {
        if (term->default_target().block == target_bb) {
            term->default_target().args.push_back(arg);
        }
        for (auto& sc : term->switch_cases()) {
            if (sc.target.block == target_bb) {
                sc.target.args.push_back(arg);
            }
        }
    } else if (term->opcode() == Opcode::invoke) {
        if (term->normal_target().block == target_bb) {
            term->normal_target().args.push_back(arg);
        }
        if (term->unwind_target().block == target_bb) {
            term->unwind_target().args.push_back(arg);
        }
    }
}

Value* ensure_operand_available(Builder& b, Value* op, BasicBlock* bb, const DominatorTree& dom) {
    if (!op || !op->is_instruction()) return op;
    Instruction* def = op->defining_instruction();
    if (!def) return op;
    if (def->opcode() == Opcode::iconst_i32 ||
        def->opcode() == Opcode::iconst_i64 ||
        def->opcode() == Opcode::fconst_f64) {
        if (!def->parent() || !dom.dominates(def->parent(), bb)) {
            if (def->opcode() == Opcode::iconst_i32) {
                return b.build_iconst_i32(static_cast<int32_t>(def->imm_i64()));
            } else if (def->opcode() == Opcode::iconst_i64) {
                return b.build_iconst_i64(def->imm_i64());
            } else if (def->opcode() == Opcode::fconst_f64) {
                return b.build_fconst_f64(def->imm_f64());
            }
        }
    }
    return op;
}

Value* insert_hoisted_expression(
    Builder& b,
    const PreExpression& expr,
    const Instruction* exemplar,
    BasicBlock* bb,
    const DominatorTree& dom
) {
    Instruction* term = bb->terminator();
    if (term) {
        b.position_before(term);
    } else {
        b.position_at_end(bb);
    }

    Value* op0 = ensure_operand_available(b, const_cast<Value*>(expr.op0), bb, dom);
    Value* op1 = ensure_operand_available(b, const_cast<Value*>(expr.op1), bb, dom);
    Value* op2 = ensure_operand_available(b, const_cast<Value*>(expr.op2), bb, dom);

    Instruction* inst = b.arena().make<Instruction>(expr.opcode, expr.type);
    uint32_t val_id = b.current_function()->next_value_id();
    Value* res = b.arena().make<Value>(val_id, expr.type, ValueKind::InstructionResult);
    res->set_defining_instruction(inst);
    inst->set_result(res);

    if (op0) inst->add_operand(op0);
    if (op1) inst->add_operand(op1);
    if (op2) inst->add_operand(op2);
    if (expr.opcode == Opcode::fconst_f64) {
        double f = 0.0;
        std::memcpy(&f, &expr.imm_bits, sizeof(double));
        inst->set_imm_f64(f);
    } else {
        inst->set_imm_i64(static_cast<int64_t>(expr.imm_bits));
    }
    inst->set_offset(expr.offset);
    inst->set_scale(expr.scale);
    inst->set_memory_type(expr.memory_type);
    inst->set_symbol(expr.symbol);

    if (exemplar) {
        inst->set_loc(exemplar->loc());
    }

    b.insert(inst);
    return res;
}

} // namespace

bool gvn_pre_function(Function& fn) {
    GvnPreOptions options;
    return gvn_pre_function(fn, options);
}

bool gvn_pre_function(Function& fn, const GvnPreOptions& options) {
    fn.rebuild_cfg_predecessors();
    bool overall_changed = false;

    // 1. Critical Edge Splitting
    if (options.enable_critical_edge_splitting) {
        CriticalEdgeStats ce_stats;
        bool ce_changed = split_critical_edges(fn, &ce_stats);
        if (ce_changed) {
            overall_changed = true;
            if (options.stats) {
                options.stats->critical_edges_split += ce_stats.critical_edges_split;
            }
        }
    }

    if (!options.enable_pre) {
        return overall_changed;
    }

    // 2. Iterative GVN-PRE optimization loop
    for (size_t iter = 0; iter < options.max_iterations; ++iter) {
        bool iter_changed = false;
        fn.rebuild_cfg_predecessors();

        DominatorTree dom(fn);
        AliasAnalysis aa(fn);

        std::unordered_map<const Value*, const Value*> value_leaders;

        // Populate value leaders for constant instructions so equivalent constants share a leader
        struct ConstKey {
            Opcode op;
            uint64_t bits;
            bool operator==(const ConstKey& other) const noexcept {
                return op == other.op && bits == other.bits;
            }
        };
        struct ConstKeyHash {
            size_t operator()(const ConstKey& k) const noexcept {
                return (static_cast<size_t>(k.op) * 31) ^ static_cast<size_t>(k.bits);
            }
        };
        std::unordered_map<ConstKey, const Value*, ConstKeyHash> const_leaders;

        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst || !inst->produces_value()) continue;
                Opcode op = inst->opcode();
                if (op == Opcode::iconst_i32 || op == Opcode::iconst_i64 || op == Opcode::fconst_f64) {
                    uint64_t bits = 0;
                    if (op == Opcode::fconst_f64) {
                        double f = inst->imm_f64();
                        std::memcpy(&bits, &f, sizeof(double));
                    } else {
                        bits = static_cast<uint64_t>(inst->imm_i64());
                    }
                    ConstKey key{op, bits};
                    auto it = const_leaders.find(key);
                    if (it == const_leaders.end()) {
                        const_leaders[key] = inst->result();
                        value_leaders[inst->result()] = inst->result();
                    } else {
                        value_leaders[inst->result()] = it->second;
                    }
                }
            }
        }

        // One walk indexes every candidate's evaluations, every operand
        // definition and every memory writer; each expression below is then
        // analysed from its own events (gvn_pre_dataflow.hpp).
        PreFunctionIndex index;
        index.build(fn, value_leaders, options.enable_load_pre);

        // The natural loops — a backedge latch -> header with a preheader, and
        // the blocks that reach the latch without passing the header — in the
        // (latch, header) order the LICM step visits them. They do not depend
        // on the expression, so they are built once per iteration.
        struct NaturalLoop {
            BasicBlock* header;
            BasicBlock* latch;
            BasicBlock* preheader;
            std::unordered_set<BasicBlock*> blocks;
        };
        std::vector<NaturalLoop> loops;
        for (BasicBlock* latch : fn.blocks()) {
            if (!latch || !dom.is_reachable(latch)) continue;
            for (BasicBlock* header : latch->successors()) {
                if (!header || !dom.is_reachable(header)) continue;
                if (!dom.dominates(header, latch)) continue; // Not a backedge

                // Find loop preheader: a predecessor of `header` that dominates `header` and is not dominated by header
                BasicBlock* preheader = nullptr;
                for (BasicBlock* pred : header->predecessors()) {
                    if (pred && pred != latch && !dom.dominates(header, pred) && dom.dominates(pred, header)) {
                        preheader = pred;
                        break;
                    }
                }
                if (!preheader) continue;

                NaturalLoop loop{header, latch, preheader, {}};
                loop.blocks.insert(header);
                if (latch != header) {
                    loop.blocks.insert(latch);
                    std::vector<BasicBlock*> worklist;
                    worklist.push_back(latch);
                    while (!worklist.empty()) {
                        BasicBlock* cur = worklist.back();
                        worklist.pop_back();
                        for (BasicBlock* pred : cur->predecessors()) {
                            if (pred && loop.blocks.insert(pred).second) {
                                worklist.push_back(pred);
                            }
                        }
                    }
                }
                loops.push_back(std::move(loop));
            }
        }

        // Blocks on some CFG cycle (a non-trivial strongly connected
        // component, so irreducible cycles count too). An expression with a
        // single evaluation outside every cycle has nothing this pass can do
        // for it: hoisting out of a loop needs an evaluation in one, and a
        // join is only partially redundant when the expression is both
        // anticipated below it and available above it — one evaluation is
        // both only from inside a cycle. A load is the exception, since a
        // must-alias store can make it available; that is filtered on the
        // store types the index recorded.
        std::unordered_set<const BasicBlock*> cyclic_blocks;
        {
            std::unordered_map<const BasicBlock*, uint32_t> dfs_index;
            std::unordered_map<const BasicBlock*, uint32_t> low_link;
            std::unordered_set<const BasicBlock*> on_stack;
            std::vector<const BasicBlock*> scc_stack;
            struct Frame {
                const BasicBlock* block;
                size_t next_succ;
            };
            uint32_t next_index = 0;
            for (const BasicBlock* root : fn.blocks()) {
                if (!root || dfs_index.count(root)) continue;
                std::vector<Frame> frames;
                frames.push_back(Frame{root, 0});
                dfs_index[root] = low_link[root] = next_index++;
                scc_stack.push_back(root);
                on_stack.insert(root);
                while (!frames.empty()) {
                    Frame& frame = frames.back();
                    const BasicBlock* bb = frame.block;
                    auto succs = bb->successors();
                    if (frame.next_succ < succs.size()) {
                        const BasicBlock* succ = succs[frame.next_succ++];
                        if (!succ) continue;
                        if (!dfs_index.count(succ)) {
                            dfs_index[succ] = low_link[succ] = next_index++;
                            scc_stack.push_back(succ);
                            on_stack.insert(succ);
                            frames.push_back(Frame{succ, 0});
                        } else if (on_stack.count(succ)) {
                            low_link[bb] = std::min(low_link[bb], dfs_index[succ]);
                        }
                        continue;
                    }
                    if (low_link[bb] == dfs_index[bb]) {
                        // bb is the root of an SCC: pop it. A component of
                        // one block is a cycle only if the block loops to
                        // itself.
                        std::vector<const BasicBlock*> component;
                        while (true) {
                            const BasicBlock* member = scc_stack.back();
                            scc_stack.pop_back();
                            on_stack.erase(member);
                            component.push_back(member);
                            if (member == bb) break;
                        }
                        bool cyclic = component.size() > 1;
                        if (!cyclic) {
                            for (const BasicBlock* succ : bb->successors()) {
                                if (succ == bb) { cyclic = true; break; }
                            }
                        }
                        if (cyclic) cyclic_blocks.insert(component.begin(), component.end());
                    }
                    frames.pop_back();
                    if (!frames.empty()) {
                        const BasicBlock* parent = frames.back().block;
                        low_link[parent] = std::min(low_link[parent], low_link[bb]);
                    }
                }
            }
        }

        PreDataflow dataflow(fn, dom, aa, index);

        for (const PreExpression& expr : index.candidates) {
            const Instruction* exemplar = index.exemplars.at(expr);
            const std::vector<Instruction*>& evals = index.evaluations.at(expr);
            if (evals.size() == 1) {
                // A lone evaluation can only be hoisted out of a cycle or fed
                // by a store of its type; anything else has nothing to gain
                // from the dataflow, and skipping it is most of the pass on
                // straight-line code.
                const BasicBlock* eval_block = index.block_of(evals.front());
                const bool in_cycle = eval_block && cyclic_blocks.count(eval_block) != 0;
                const bool store_may_supply = expr.is_load() && index.has_store_of_type(expr.memory_type);
                if (!in_cycle && !store_may_supply) continue;
            }
            dataflow.analyze_expression(expr, exemplar);

            // A. Check for Loop Invariant Code Motion (LICM):
            // If an expression is computed inside a loop, all its operands dominate the loop preheader,
            // and no instruction in the loop clobbers it, hoist it to the preheader.
            for (const NaturalLoop& loop : loops) {
                const std::unordered_set<BasicBlock*>& loop_set = loop.blocks;
                BasicBlock* preheader = loop.preheader;

                bool has_loop_eval = false;
                for (BasicBlock* b_block : loop_set) {
                    const auto& b_info = dataflow.get_local_info(b_block);
                    if (!b_info.evaluations.empty()) {
                        has_loop_eval = true;
                        break;
                    }
                }
                if (!has_loop_eval) continue;

                // Expression operands must dominate the preheader
                if (!dataflow.can_evaluate_at_end(preheader, expr, exemplar)) continue;

                // If expression is memory load, the entire loop must be transparent
                if (expr.is_load()) {
                    bool loop_transparent = true;
                    for (BasicBlock* b_block : loop_set) {
                        if (!dataflow.get_local_info(b_block).transp) {
                            loop_transparent = false;
                            break;
                        }
                    }
                    if (!loop_transparent) continue;
                }

                // Hoist expression to preheader if not already available at exit
                Value* hoisted_val = dataflow.available_at_exit(preheader);
                if (!hoisted_val) {
                    Builder b(fn);
                    hoisted_val = insert_hoisted_expression(b, expr, exemplar, preheader, dom);
                    if (options.stats) options.stats->expressions_hoisted++;
                }

                // Eliminate all evaluations in the loop
                for (BasicBlock* b_block : loop_set) {
                    const auto& b_info = dataflow.get_local_info(b_block);
                    std::vector<Instruction*> to_remove = b_info.evaluations;
                    for (Instruction* inst : to_remove) {
                        if (inst && inst->parent()) {
                            replace_all_uses(fn, inst->result(), hoisted_val);
                            inst->parent()->remove_instruction(inst);
                            if (options.stats) options.stats->expressions_eliminated++;
                        }
                    }
                }

                iter_changed = true;
                break;
            }

            if (iter_changed) break;

            // B. Check Join/Merge blocks for Partial Redundancy
            for (BasicBlock* bb : fn.blocks()) {
                if (!bb) continue;
                if (bb->predecessors().size() <= 1) continue;

                // Loop headers must not be treated as join/merge blocks for PRE.
                // Hoisting out of loops is handled by LICM (Part A).
                bool is_loop_header = false;
                for (BasicBlock* pred : bb->predecessors()) {
                    if (pred && dom.dominates(bb, pred)) {
                        is_loop_header = true;
                        break;
                    }
                }
                if (is_loop_header) continue;

                // Must be anticipated at entry of bb
                if (!dataflow.is_anticipated_at_entry(bb)) {
                    const auto& info = dataflow.get_local_info(bb);
                    if (!info.ant_loc) continue;
                }

                // Check availability across predecessors
                std::vector<BasicBlock*> missing_preds;
                std::vector<BasicBlock*> avail_preds;

                for (BasicBlock* pred : bb->predecessors()) {
                    if (!pred) continue;
                    Value* avail_v = dataflow.available_at_exit(pred);
                    if (avail_v) {
                        avail_preds.push_back(pred);
                    } else {
                        missing_preds.push_back(pred);
                    }
                }

                // Partial redundancy requires some available and some missing
                if (avail_preds.empty() || missing_preds.empty()) {
                    continue;
                }

                // Check if all missing predecessors can safely evaluate expr
                bool can_hoist_all = true;
                for (BasicBlock* pred : missing_preds) {
                    if (!dataflow.can_evaluate_at_end(pred, expr, exemplar)) {
                        can_hoist_all = false;
                        break;
                    }
                }
                if (!can_hoist_all) continue;

                // Hoist computation to all missing predecessors
                Builder b(fn);
                std::unordered_map<BasicBlock*, Value*> incoming_values;

                for (BasicBlock* pred : avail_preds) {
                    incoming_values[pred] = dataflow.available_at_exit(pred);
                }

                for (BasicBlock* pred : missing_preds) {
                    Value* v_new = insert_hoisted_expression(b, expr, exemplar, pred, dom);
                    incoming_values[pred] = v_new;
                    if (options.stats) options.stats->expressions_hoisted++;
                }

                // Check if all incoming values are identical
                bool all_identical = true;
                Value* first_val = incoming_values[bb->predecessors()[0]];
                for (BasicBlock* pred : bb->predecessors()) {
                    if (incoming_values[pred] != first_val) {
                        all_identical = false;
                        break;
                    }
                }

                Value* unified_val = nullptr;
                if (all_identical) {
                    unified_val = first_val;
                } else {
                    // Inject a block parameter at bb
                    unified_val = b.add_block_param(bb, expr.type);
                    if (options.stats) options.stats->block_params_inserted++;

                    // Update all predecessor branch targets to pass the incoming value
                    for (BasicBlock* pred : bb->predecessors()) {
                        Instruction* term = pred->terminator();
                        if (term) {
                            append_branch_arg_to_target(term, bb, incoming_values[pred]);
                        }
                    }
                }

                // Eliminate evaluations of expr in bb and dominated blocks
                const auto& info = dataflow.get_local_info(bb);
                std::vector<Instruction*> to_remove = info.evaluations;
                for (Instruction* inst : to_remove) {
                    if (inst && inst->parent()) {
                        replace_all_uses(fn, inst->result(), unified_val);
                        inst->parent()->remove_instruction(inst);
                        if (options.stats) options.stats->expressions_eliminated++;
                    }
                }

                iter_changed = true;
                break;
            }

            if (iter_changed) break;
        }

        if (iter_changed) {
            overall_changed = true;
            fn.rebuild_cfg_predecessors();
        } else {
            break;
        }
    }

    return overall_changed;
}

bool gvn_pre_module(Module& mod) {
    GvnPreOptions options;
    return gvn_pre_module(mod, options);
}

bool gvn_pre_module(Module& mod, const GvnPreOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= gvn_pre_function(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
