#include <brass/mir/loop_opt.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/select_opt.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/pgo/profile_data.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include "int_fold.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <functional>
#include <optional>
#include <cstring>

namespace brass {

bool ivsr_pass(Function& fn, LoopInfo& loop, DominatorTree& dom);
void replace_all_uses(Function& fn, Value* old_val, Value* new_val);

namespace {

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

bool is_pure_instruction(const Instruction* inst) {
    if (!inst) return false;
    // Pure, but tied to its block: hoisting or merging a derived gcref could
    // keep it live across a GC point (gc_refs.hpp).
    if (is_derived_gcref(inst->result())) return false;
    Opcode op = inst->opcode();
    switch (op) {
        case Opcode::iconst_i32: case Opcode::iconst_i64: case Opcode::fconst_f64:
        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32:
        case Opcode::fptosi_i32: case Opcode::fptosi_i64: case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64:
        case Opcode::bitcast_i64_f64: case Opcode::bitcast_f64_i64:
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::neg:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
        case Opcode::shl: case Opcode::lshr: case Opcode::ashr: case Opcode::not_:
        case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
        case Opcode::eq: case Opcode::ne: case Opcode::slt: case Opcode::ult:
        case Opcode::sle: case Opcode::ule: case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::select: return true;
        case Opcode::call: return inst->symbol() == "bronze_tls_block_addr";
        case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod: {
            // LICM executes pure instructions speculatively, so a division is
            // pure only when its constant divisor rules out every trap.
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            int64_t divisor = 0;
            if (!get_const_int(inst->operand(1), divisor)) return false;
            const unsigned width = int_fold::width_of(inst->type());
            return width != 0 && !int_fold::division_may_trap(op, width, divisor);
        }
        default: return false;
    }
}

} // namespace

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) inst->set_operand(i, new_val);
            }
            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                if (inst->branch_target().args[i] == old_val) inst->branch_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                if (inst->true_target().args[i] == old_val) inst->true_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                if (inst->false_target().args[i] == old_val) inst->false_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                if (inst->default_target().args[i] == old_val) inst->default_target().args[i] = new_val;
            }
            for (auto& sc : inst->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    if (sc.target.args[i] == old_val) sc.target.args[i] = new_val;
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) inst->state_map()[i] = new_val;
            }
        }
    }
}

namespace {

std::unordered_map<const Value*, uint32_t> compute_use_counts(const Function& fn) {
    std::unordered_map<const Value*, uint32_t> counts;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (const Value* op : inst->operands()) if (op) counts[op]++;
            for (const Value* arg : inst->branch_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->true_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->false_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->default_target().args) if (arg) counts[arg]++;
            for (const auto& sc : inst->switch_cases()) {
                for (const Value* arg : sc.target.args) if (arg) counts[arg]++;
            }
            for (const Value* sv : inst->state_map()) if (sv) counts[sv]++;
        }
    }
    return counts;
}

// Drops parameter `p_i` of `bb` and the matching argument on every edge into
// it, whatever kind of terminator the edge comes from.
void remove_block_param(BasicBlock* bb, size_t p_i) {
    Function* fn = bb->parent();
    if (!fn) return;
    auto& params = bb->params();
    params.erase(params.begin() + static_cast<std::ptrdiff_t>(p_i));
    for (size_t k = p_i; k < params.size(); ++k) {
        params[k]->set_block_param(bb, static_cast<uint32_t>(k));
    }
    auto drop_arg = [&](BranchTarget& bt) {
        if (bt.block == bb && p_i < bt.args.size()) {
            bt.args.erase(bt.args.begin() + static_cast<std::ptrdiff_t>(p_i));
        }
    };
    // Scan every terminator rather than the cached predecessor list, which
    // earlier edits in the same pass may have left stale.
    for (BasicBlock* pred : fn->blocks()) {
        if (!pred) continue;
        Instruction* term = pred->terminator();
        if (!term) continue;
        drop_arg(term->branch_target());
        drop_arg(term->true_target());
        drop_arg(term->false_target());
        for (auto& sc : term->switch_cases()) drop_arg(sc.target);
    }
}

bool constant_folding_pass(Function& fn) {
    bool changed = false;
    Builder b(*fn.parent());
    b.set_function(&fn);

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();
            if (!is_pure_instruction(cur) || !cur->produces_value()) {
                cur = next;
                continue;
            }

            Opcode op = cur->opcode();
            if (cur->operand_count() == 2) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);
                int64_t c0 = 0, c1 = 0;
                bool has_c0 = get_const_int(op0, c0);
                bool has_c1 = get_const_int(op1, c1);
                Type res_type = cur->type();
                Value* replacement = nullptr;
                // Identities below only hold for integer arithmetic, and only a
                // result of the same type as the replacement may stand in for it.
                const unsigned width = op0 ? int_fold::width_of(op0->type()) : 0;
                const bool int_result = res_type == Type::i32() || res_type == Type::i64();
                auto same_type = [&](const Value* v) { return v && v->type() == res_type; };
                auto build_int = [&](int64_t v) -> Value* {
                    b.position_before(cur);
                    return (res_type == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(v)) : b.build_iconst_i64(v);
                };

                if (has_c0 && has_c1) {
                    if (int_result && width != 0) {
                        if (auto folded = int_fold::binary(op, width, c0, c1)) replacement = build_int(*folded);
                    }
                } else if (has_c1 && width != 0) {
                    if ((op == Opcode::add || op == Opcode::sub || op == Opcode::or_ || op == Opcode::xor_ ||
                         op == Opcode::shl || op == Opcode::ashr || op == Opcode::lshr) && c1 == 0) {
                        if (same_type(op0)) replacement = op0;
                    } else if (op == Opcode::mul && c1 == 1) {
                        if (same_type(op0)) replacement = op0;
                    } else if (op == Opcode::mul && c1 == 0 && int_result) {
                        replacement = build_int(0);
                    }
                } else if (has_c0 && op1 && int_fold::width_of(op1->type()) != 0) {
                    if ((op == Opcode::add || op == Opcode::or_ || op == Opcode::xor_) && c0 == 0) {
                        if (same_type(op1)) replacement = op1;
                    } else if (op == Opcode::mul && c0 == 1) {
                        if (same_type(op1)) replacement = op1;
                    } else if (op == Opcode::mul && c0 == 0 && int_result) {
                        replacement = build_int(0);
                    }
                } else if (op0 == op1 && (op == Opcode::sub || op == Opcode::xor_)) {
                    if (int_result && width != 0) {
                        replacement = build_int(0);
                    } else if (op == Opcode::sub && res_type == Type::f64() && fn.allow_fp_reassociation()) {
                        b.position_before(cur);
                        replacement = b.build_fconst_f64(0.0);
                    }
                }

                if (replacement) {
                    replace_all_uses(fn, cur->result(), replacement);
                    bb->remove_instruction(cur);
                    changed = true;
                }
            } else if (cur->operand_count() == 1) {
                Value* op0 = cur->operand(0);
                int64_t c0;
                if (get_const_int(op0, c0)) {
                    Value* replacement = nullptr;
                    std::optional<int64_t> folded;
                    if (op == Opcode::sext_i64 || op == Opcode::zext_i64 || op == Opcode::trunc_i32) {
                        folded = int_fold::convert(op, op0->type(), c0);
                    } else if (cur->type() == Type::i32() || cur->type() == Type::i64()) {
                        folded = int_fold::unary(op, int_fold::width_of(cur->type()), c0);
                    }
                    if (folded) {
                        b.position_before(cur);
                        replacement = (cur->type() == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(*folded))
                                                                   : b.build_iconst_i64(*folded);
                    }

                    if (replacement) {
                        replace_all_uses(fn, cur->result(), replacement);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                }
            } else if (cur->operand_count() == 3 && cur->opcode() == Opcode::select) {
                Value* cond = cur->operand(0);
                Value* true_v = cur->operand(1);
                Value* false_v = cur->operand(2);
                Value* replacement = nullptr;
                int64_t cond_c;
                if (get_const_int(cond, cond_c)) {
                    replacement = (cond_c != 0) ? true_v : false_v;
                } else if (true_v == false_v) {
                    replacement = true_v;
                }
                if (replacement) {
                    replace_all_uses(fn, cur->result(), replacement);
                    bb->remove_instruction(cur);
                    changed = true;
                }
            }
            cur = next;
        }
    }
    return changed;
}

struct ExprKey {
    Opcode op;
    Type type;
    const Value* op0 = nullptr;
    const Value* op1 = nullptr;
    const Value* op2 = nullptr;
    uint64_t imm_bits = 0;

    bool operator==(const ExprKey& o) const noexcept {
        return op == o.op && type == o.type && op0 == o.op0 && op1 == o.op1 && op2 == o.op2 && imm_bits == o.imm_bits;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.op);
        h = h * 31 + std::hash<const void*>()(k.op0);
        h = h * 31 + std::hash<const void*>()(k.op1);
        h = h * 31 + std::hash<const void*>()(k.op2);
        h = h * 31 + static_cast<size_t>(k.imm_bits);
        return h;
    }
};

bool cse_pass(Function& fn, DominatorTree& dom) {
    bool changed = false;
    std::unordered_map<ExprKey, Value*, ExprKeyHash> expr_map;

    auto is_commutative = [](Opcode op) {
        return op == Opcode::add || op == Opcode::mul || op == Opcode::and_ ||
               op == Opcode::or_ || op == Opcode::xor_ || op == Opcode::eq || op == Opcode::ne;
    };

    std::function<void(BasicBlock*)> visit_block = [&](BasicBlock* bb) {
        if (!bb) return;
        std::vector<ExprKey> added_keys;

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();
            if (is_pure_instruction(cur) && cur->produces_value()) {
                ExprKey key;
                key.op = cur->opcode();
                key.type = cur->type();
                if (cur->opcode() == Opcode::fconst_f64) {
                    double fval = cur->imm_f64();
                    std::memcpy(&key.imm_bits, &fval, sizeof(double));
                } else {
                    key.imm_bits = static_cast<uint64_t>(cur->imm_i64());
                }
                if (cur->operand_count() >= 1) key.op0 = cur->operand(0);
                if (cur->operand_count() >= 2) key.op1 = cur->operand(1);
                if (cur->operand_count() >= 3) key.op2 = cur->operand(2);

                if (is_commutative(key.op) && key.op0 > key.op1) {
                    std::swap(key.op0, key.op1);
                }

                auto it = expr_map.find(key);
                if (it != expr_map.end()) {
                    replace_all_uses(fn, cur->result(), it->second);
                    bb->remove_instruction(cur);
                    changed = true;
                } else {
                    expr_map[key] = cur->result();
                    added_keys.push_back(key);
                }
            }
            cur = next;
        }

        for (const BasicBlock* child : dom.children(bb)) {
            if (child) visit_block(const_cast<BasicBlock*>(child));
        }

        for (const auto& key : added_keys) {
            expr_map.erase(key);
        }
    };

    if (fn.entry_block()) {
        visit_block(fn.entry_block());
    }

    return changed;
}

bool dead_code_elimination_pass(Function& fn) {
    bool changed = false;
    bool progress = true;

    while (progress) {
        progress = false;
        auto use_counts = compute_use_counts(fn);

        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (!cur->has_side_effects() && cur->produces_value()) {
                    Value* res = cur->result();
                    if (!res || use_counts[res] == 0) {
                        bb->remove_instruction(cur);
                        progress = true;
                        changed = true;
                    }
                }
                cur = next;
            }
        }

        if (progress) use_counts = compute_use_counts(fn);

        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || bb == fn.entry_block()) continue;
            size_t p_i = 0;
            while (p_i < bb->param_count()) {
                Value* p = bb->param(p_i);
                if (p && use_counts[p] == 0) {
                    remove_block_param(bb, p_i);
                    progress = true;
                    changed = true;
                    use_counts = compute_use_counts(fn);
                } else {
                    ++p_i;
                }
            }
        }
    }
    return changed;
}

bool licm_pass(Function& fn, LoopInfo& loop, DominatorTree& dom) {
    (void)dom;
    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;
    Instruction* ph_term = preheader->terminator();
    if (!ph_term) return false;

    bool any_hoisted = false;
    bool hoisted_in_round = true;

    while (hoisted_in_round) {
        hoisted_in_round = false;
        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == preheader) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (is_pure_instruction(cur) && cur->produces_value()) {
                    bool all_operands_invariant = true;
                    for (Value* op : cur->operands()) {
                        if (!op) continue;
                        if (op->is_block_param()) {
                            if (loop.contains(op->defining_block())) {
                                all_operands_invariant = false;
                                break;
                            }
                        } else if (op->is_instruction()) {
                            Instruction* def_inst = op->defining_instruction();
                            if (def_inst && loop.contains(def_inst->parent()) && def_inst->parent() != preheader) {
                                all_operands_invariant = false;
                                break;
                            }
                        }
                    }

                    if (all_operands_invariant) {
                        bb->remove_instruction(cur);
                        preheader->insert_before(cur, ph_term);
                        hoisted_in_round = true;
                        any_hoisted = true;
                    }
                }
                cur = next;
            }
        }
    }
    return any_hoisted;
}

bool eliminate_dead_induction_cycles(Function& fn) {
    bool changed = false;
    bool progress = true;

    while (progress) {
        progress = false;
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || bb == fn.entry_block()) continue;
            size_t p_i = 0;
            while (p_i < bb->param_count()) {
                Value* p = bb->param(p_i);
                if (!p) { ++p_i; continue; }

                std::unordered_set<Instruction*> cycle_insts;
                std::vector<Value*> worklist = {p};
                std::unordered_set<Value*> visited = {p};
                bool is_pure_cycle = true;

                while (!worklist.empty()) {
                    Value* cur_v = worklist.back();
                    worklist.pop_back();

                    for (BasicBlock* u_bb : fn.blocks()) {
                        if (!u_bb) continue;
                        for (Instruction* inst : *u_bb) {
                            if (!inst) continue;
                            for (Value* sv : inst->state_map()) {
                                if (sv == cur_v) { is_pure_cycle = false; break; }
                            }
                            if (!is_pure_cycle) break;

                            for (Value* op : inst->operands()) {
                                if (op == cur_v) {
                                    if (inst->has_side_effects() || !inst->produces_value()) { is_pure_cycle = false; break; }
                                    if (inst->opcode() != Opcode::add && inst->opcode() != Opcode::sub) { is_pure_cycle = false; break; }
                                    cycle_insts.insert(inst);
                                    Value* res = inst->result();
                                    if (res && !visited.count(res)) {
                                        visited.insert(res);
                                        worklist.push_back(res);
                                    }
                                }
                            }
                            if (!is_pure_cycle) break;

                            auto check_bt = [&](const BranchTarget& bt) {
                                for (size_t arg_idx = 0; arg_idx < bt.args.size(); ++arg_idx) {
                                    if (bt.args[arg_idx] == cur_v) {
                                        if (bt.block == bb && arg_idx == p_i) {}
                                        else { is_pure_cycle = false; }
                                    }
                                }
                            };

                            // Every edge kind (br, br_if, switch, invoke) can carry the value.
                            check_bt(inst->branch_target());
                            check_bt(inst->true_target());
                            check_bt(inst->false_target());
                            for (const auto& sc : inst->switch_cases()) check_bt(sc.target);
                            if (!is_pure_cycle) break;
                        }
                        if (!is_pure_cycle) break;
                    }
                    if (!is_pure_cycle) break;
                }

                if (is_pure_cycle && !cycle_insts.empty()) {
                    for (Instruction* inst : cycle_insts) {
                        if (inst && inst->parent()) inst->parent()->remove_instruction(inst);
                    }
                    remove_block_param(bb, p_i);
                    progress = true;
                    changed = true;
                } else {
                    ++p_i;
                }
            }
        }
    }
    return changed;
}
} // namespace

bool optimize_function_loops(Function& fn, const LoopOptOptions& options) {
    if (!fn.resume_points().empty() || fn.name().starts_with("__wrapper_")) {
        return false;
    }
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::osr_entry) {
                return false;
            }
        }
    }
    bool any_changed = false;
    if (options.enable_f64_demote) {
        F64DemoteOptions demote_opts;
        demote_opts.stats = options.demote_stats;
        any_changed |= f64_demote_pass(fn, demote_opts);
    }

    if (options.enable_diamond_select) {
        any_changed |= simplify_cfg_diamonds(fn);
    }

    if (options.enable_sroa) {
        SroaOptions sroa_opts;
        any_changed |= sroa_function(fn, sroa_opts);
    }

    if (options.enable_allocation_sinking || options.enable_partial_escape) {
        AllocationSinkingOptions sink_opts;
        if (options.pea_stats) {
            sink_opts.stats = options.pea_stats;
        } else if (options.stats) {
            sink_opts.stats = &options.stats->pea_stats;
        }
        any_changed |= sink_allocations(fn, sink_opts);
    }

    for (size_t iter = 0; iter < options.max_iterations; ++iter) {
        bool iter_changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);

        if (options.enable_dce) {
            iter_changed |= constant_folding_pass(fn);
            iter_changed |= cse_pass(fn, dom);
            iter_changed |= dead_code_elimination_pass(fn);
            iter_changed |= eliminate_dead_induction_cycles(fn);
            iter_changed |= dead_code_elimination_pass(fn);
        }

        std::vector<LoopInfo*> post_order = loops.post_order_loops();
        if (post_order.empty()) {
            if (iter_changed) any_changed = true;
            break;
        }

        for (LoopInfo* loop : post_order) {
            if (!loop) continue;
            if (options.enable_licm) iter_changed |= licm_pass(fn, *loop, dom);
            if (options.enable_ivsr) iter_changed |= ivsr_pass(fn, *loop, dom);
        }

        if (options.enable_dce) {
            iter_changed |= constant_folding_pass(fn);
            iter_changed |= cse_pass(fn, dom);
            iter_changed |= dead_code_elimination_pass(fn);
            iter_changed |= eliminate_dead_induction_cycles(fn);
            iter_changed |= dead_code_elimination_pass(fn);
        }

        fn.rebuild_cfg_predecessors();
        if (iter_changed) any_changed = true;
        else break;
    }

    if (options.enable_loop_tile) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopTileOptions tile_opts;
        tile_opts.tile_size_i = options.tile_size_i;
        tile_opts.tile_size_j = options.tile_size_j;
        tile_opts.tile_size_k = options.tile_size_k;
        tile_opts.enable_loop_interchange = options.enable_loop_interchange;
        if (loop_tile_pass(fn, dom, tile_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            DominatorTree dom_after(fn);
            if (options.enable_dce) {
                constant_folding_pass(fn);
                cse_pass(fn, dom_after);
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_loop_distribution) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopDistributionOptions dist_opts;
        if (options.stats) dist_opts.stats = &options.stats->distribution_stats;
        if (loop_distribution_pass(fn, dom, dist_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_loop_fusion) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopFusionOptions fuse_opts;
        if (options.stats) fuse_opts.stats = &options.stats->fusion_stats;
        if (loop_fusion_pass(fn, dom, fuse_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_array_contraction) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        ArrayContractionOptions contract_opts;
        contract_opts.fuse_loops_first = options.enable_loop_fusion;
        if (options.stats) contract_opts.stats = &options.stats->contraction_stats;
        if (array_contraction_pass(fn, dom, contract_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_parallel_loops) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        ParallelLoopOptions par_opts;
        par_opts.parallel_threshold = options.parallel_threshold;
        par_opts.parallel_workers = options.parallel_workers;
        par_opts.allow_fp_reassociation = options.enable_fp_reassociation || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
        if (options.parallel_stats) {
            par_opts.stats = options.parallel_stats;
        } else if (options.stats) {
            par_opts.stats = &options.stats->parallel_stats;
        }
        if (auto_parallelize_function(fn, dom, par_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_bce) {
        fn.rebuild_cfg_predecessors();
        RangeAnalysisOptions bce_opts;
        bce_opts.enable_bce = true;
        bce_opts.enable_implied_checks = true;
        bce_opts.dump_stats = options.dump_range_stats;
        if (options.range_stats) {
            bce_opts.stats = options.range_stats;
        } else if (options.stats) {
            bce_opts.stats = &options.stats->bce_stats;
        }
        if (run_bounds_check_elimination(fn, *fn.parent(), bce_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            DominatorTree dom_after(fn);
            if (options.enable_dce) {
                constant_folding_pass(fn);
                cse_pass(fn, dom_after);
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_slp) {
        SlpOptions slp_opts;
        slp_opts.allow_fp_reassociation = options.enable_fp_reassociation || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
        if (slp_vectorize_function(fn, slp_opts)) {
            any_changed = true;
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_vectorize) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopVectorizeOptions vec_opts;
        vec_opts.enable_avx2 = options.enable_avx2;
        vec_opts.vector_width = options.vector_width;
        vec_opts.allow_fp_reassociation = options.enable_fp_reassociation || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
        if (loop_vectorize_pass(fn, dom, vec_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            DominatorTree dom_after(fn);
            if (options.enable_dce) {
                constant_folding_pass(fn);
                cse_pass(fn, dom_after);
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_fma) {
        FmaOptOptions fma_opts;
        fma_opts.stats = options.fma_stats;
        if (fma_opt_pass(fn, fma_opts)) {
            any_changed = true;
            if (options.enable_dce) {
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_unroll) {
        bool should_unroll = true;
        if (options.profile_data) {
            const auto* prof = options.profile_data->find_function(std::string(fn.name()));
            if (prof) {
                mir::BranchProbabilityAnalysis bpa(fn, *prof);
                DominatorTree dom_chk(fn);
                LoopAnalysis la(fn, dom_chk);
                bool has_hot_loop = false;
                for (LoopInfo* loop : la.post_order_loops()) {
                    if (loop && loop->header()) {
                        uint64_t hdr_cnt = bpa.block_frequency_info().get_block_count(loop->header());
                        BasicBlock* ph = loop->preheader();
                        uint64_t ph_cnt = ph ? bpa.block_frequency_info().get_block_count(ph) : 0;
                        uint64_t iters = (ph_cnt > 0) ? (hdr_cnt / ph_cnt) : hdr_cnt;
                        if (iters >= options.min_pgo_unroll_iterations && hdr_cnt >= 10) {
                            has_hot_loop = true;
                            break;
                        }
                    }
                }
                if (!has_hot_loop) should_unroll = false;
            }
        }
        if (should_unroll) {
            fn.rebuild_cfg_predecessors();
            DominatorTree dom(fn);
            bool allow_fp = options.enable_fp_reassociation || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
            if (loop_unroll_pass(fn, dom, {options.unroll_factor, true, allow_fp, true})) {
                any_changed = true;
                fn.rebuild_cfg_predecessors();
                DominatorTree dom_after(fn);
                constant_folding_pass(fn);
                cse_pass(fn, dom_after);
                dead_code_elimination_pass(fn);
            }
        }
    }

    if (options.enable_dce) {
        dead_code_elimination_pass(fn);
    }

    if (options.enable_diamond_select) {
        any_changed |= simplify_cfg_diamonds(fn);
    }

    return any_changed;
}

bool optimize_module_loops(Module& mod, const LoopOptOptions& options) {
    bool changed = false;
    mod.set_has_loop_optimizations(true);
    LoopOptOptions mod_opts = options;
    if (mod.allow_fp_reassociation()) mod_opts.enable_fp_reassociation = true;
    for (Function* fn : mod.functions()) {
        if (fn && !fn->name().starts_with("__wrapper_")) changed |= optimize_function_loops(*fn, mod_opts);
    }
    return changed;
}

bool optimize_function(Function& fn) {
    LoopOptOptions opts;
    return optimize_function(fn, opts);
}

bool optimize_function(Function& fn, const LoopOptOptions& options) {
    bool changed = false;

    // 1. SROA
    if (options.enable_sroa) {
        SroaOptions sroa_opts;
        changed |= sroa_function(fn, sroa_opts);
    }

    // 2. GVN (CSE + RLE + DSE)
    if (options.enable_gvn) {
        GvnOptions gvn_opts;
        changed |= gvn_function(fn, gvn_opts);
    }

    // 2b. SCCP & Guard Elim & CFG Simplify
    if (options.enable_sccp) {
        SccpOptions sccp_opts;
        sccp_opts.enable_guard_elim = options.enable_guard_elim;
        changed |= sccp_function(fn, sccp_opts);
    }
    if (options.enable_cfg_simplify) {
        CfgSimplifyOptions cfg_opts;
        changed |= cfg_simplify_function(fn, cfg_opts);
    }

    // 2c. Loop Unswitch
    if (options.enable_loop_unswitch) {
        bool should_unswitch = true;
        if (options.profile_data) {
            const auto* prof = options.profile_data->find_function(std::string(fn.name()));
            if (prof) {
                mir::BranchProbabilityAnalysis bpa(fn, *prof);
                DominatorTree dom_chk(fn);
                LoopAnalysis la(fn, dom_chk);
                bool has_hot_loop = false;
                for (LoopInfo* loop : la.post_order_loops()) {
                    if (loop && loop->header()) {
                        uint64_t hdr_cnt = bpa.block_frequency_info().get_block_count(loop->header());
                        if (hdr_cnt >= options.min_pgo_unswitch_count) {
                            has_hot_loop = true;
                            break;
                        }
                    }
                }
                if (!has_hot_loop) should_unswitch = false;
            }
        }
        if (should_unswitch) {
            LoopUnswitchStats* ustats = options.stats ? &options.stats->unswitch_stats : nullptr;
            if (unswitch_loops_in_function(fn, options.unswitch_options, ustats)) {
                changed = true;
                if (options.enable_cfg_simplify) {
                    CfgSimplifyOptions cfg_opts;
                    cfg_simplify_function(fn, cfg_opts);
                }
            }
        }
    }

    // 2d. Jump Threading
    if (options.enable_jump_threading) {
        JumpThreadingStats* jstats = options.stats ? &options.stats->jump_threading_stats : nullptr;
        if (run_jump_threading(fn, options.jump_threading_options, jstats)) {
            changed = true;
            if (options.enable_cfg_simplify) {
                CfgSimplifyOptions cfg_opts;
                cfg_simplify_function(fn, cfg_opts);
            }
        }
    }

    // 2e. Partial Escape Analysis & Allocation Sinking
    if (options.enable_allocation_sinking || options.enable_partial_escape) {
        AllocationSinkingOptions sink_opts;
        if (options.pea_stats) {
            sink_opts.stats = options.pea_stats;
        } else if (options.stats) {
            sink_opts.stats = &options.stats->pea_stats;
        }
        if (sink_allocations(fn, sink_opts)) {
            changed = true;
            if (options.enable_cfg_simplify) {
                CfgSimplifyOptions cfg_opts;
                cfg_simplify_function(fn, cfg_opts);
            }
        }
    }

    // 3. Loop optimizations, LICM, IVSR, DCE & Vectorization
    changed |= optimize_function_loops(fn, options);

    return changed;
}

bool optimize_module(Module& mod) {
    LoopOptOptions opts;
    return optimize_module(mod, opts);
}

bool optimize_module(Module& mod, const LoopOptOptions& options) {
    bool changed = false;
    mod.set_has_loop_optimizations(true);
    LoopOptOptions mod_opts = options;
    if (mod.allow_fp_reassociation()) mod_opts.enable_fp_reassociation = true;
    for (Function* fn : mod.functions()) {
        if (fn) changed |= optimize_function(*fn, mod_opts);
    }
    return changed;
}

} // namespace brass
