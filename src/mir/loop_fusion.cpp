#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace brass {

std::string LoopFusionStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Loop Fusion Statistics ===\n"
       << "  Loops fused: " << loops_fused << "\n"
       << "  Candidates checked: " << candidates_checked << "\n"
       << "  Rejected (non-adjacent): " << rejected_non_adjacent << "\n"
       << "  Rejected (domain mismatch): " << rejected_domain_mismatch << "\n"
       << "  Rejected (dependencies): " << rejected_dependencies << "\n";
    return ss.str();
}

namespace {

static bool get_const_int(const Value* val, int64_t& out_val) {
    while (val && val->is_instruction()) {
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
        if (def->opcode() == Opcode::fconst_f64) {
            out_val = static_cast<int64_t>(def->imm_f64());
            return true;
        }
        if ((def->opcode() == Opcode::sitofp_f64_i64 || def->opcode() == Opcode::sitofp_f64_i32 ||
             def->opcode() == Opcode::fptosi_i64 || def->opcode() == Opcode::fptosi_i32 ||
             def->opcode() == Opcode::bitcast_i64_f64 || def->opcode() == Opcode::bitcast_f64_i64 ||
             def->opcode() == Opcode::sext_i64 || def->opcode() == Opcode::zext_i64) && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        return false;
    }
    return false;
}

static void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
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
        }
    }
}

struct FusibleLoopInfo {
    LoopInfo* loop = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* exit_bb = nullptr;
    BranchTarget* ph_bt = nullptr;
    BranchTarget* latch_bt = nullptr;

    size_t iv_index = 0;
    Value* iv_param = nullptr;
    Type iv_type = Type::i64();
    Value* init_val = nullptr;
    Value* limit_val = nullptr;
    int64_t step = 1;
    Opcode cmp_opcode = Opcode::slt;
    bool exit_on_false = true;
};

static bool extract_fusible_loop(Function& fn, LoopInfo& loop, FusibleLoopInfo& info) {
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return false;

    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || !latch_term || !hdr_term) return false;

    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
        ph_bt = &ph_term->branch_target();
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
        latch_bt = &latch_term->branch_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    if (hdr_term->opcode() != Opcode::br_if) return false;

    Value* cond_val = hdr_term->operand(0);
    if (!cond_val || !cond_val->is_instruction()) return false;
    Instruction* cmp_inst = cond_val->defining_instruction();
    if (!cmp_inst || !is_comparison(cmp_inst->opcode()) || cmp_inst->parent() != header) return false;

    BasicBlock* body_bb = nullptr;
    BasicBlock* exit_bb = nullptr;
    bool exit_on_false = true;

    if (loop.contains(hdr_term->true_target().block) && !loop.contains(hdr_term->false_target().block)) {
        body_bb = hdr_term->true_target().block;
        exit_bb = hdr_term->false_target().block;
        exit_on_false = true;
    } else if (!loop.contains(hdr_term->true_target().block) && loop.contains(hdr_term->false_target().block)) {
        body_bb = hdr_term->false_target().block;
        exit_bb = hdr_term->true_target().block;
        exit_on_false = false;
    } else {
        return false;
    }

    if (!body_bb || !exit_bb) return false;

    Opcode cmp_op = cmp_inst->opcode();
    if (cmp_op != Opcode::slt && cmp_op != Opcode::ult &&
        cmp_op != Opcode::sle && cmp_op != Opcode::ule) {
        return false;
    }

    Value* cmp_lhs = cmp_inst->operand(0);
    Value* cmp_rhs = cmp_inst->operand(1);

    bool found_iv = false;
    size_t iv_idx = 0;
    int64_t step_val = 1;
    Value* limit = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        Value* effective_cmp_lhs = cmp_lhs;
        if (effective_cmp_lhs && effective_cmp_lhs->is_instruction()) {
            Instruction* d = effective_cmp_lhs->defining_instruction();
            if (d && (d->opcode() == Opcode::sitofp_f64_i64 || d->opcode() == Opcode::sitofp_f64_i32 ||
                      d->opcode() == Opcode::fptosi_i64 || d->opcode() == Opcode::fptosi_i32 ||
                      d->opcode() == Opcode::bitcast_i64_f64 || d->opcode() == Opcode::bitcast_f64_i64 ||
                      d->opcode() == Opcode::sext_i64 || d->opcode() == Opcode::zext_i64)) {
                if (d->operand_count() >= 1) effective_cmp_lhs = d->operand(0);
            }
        }
        if ((param == cmp_lhs || param == effective_cmp_lhs) && loop.is_loop_invariant(cmp_rhs)) {
            Value* latch_next = latch_bt->args[i];
            if (latch_next && latch_next->is_instruction()) {
                Instruction* def = latch_next->defining_instruction();
                if (def && def->opcode() == Opcode::add) {
                    Value* step_op = (def->operand(0) == param) ? def->operand(1) : ((def->operand(1) == param) ? def->operand(0) : nullptr);
                    int64_t c = 0;
                    if (step_op && get_const_int(step_op, c) && c > 0) {
                        found_iv = true;
                        iv_idx = i;
                        step_val = c;
                        limit = cmp_rhs;
                        break;
                    }
                }
            }
        }
    }

    if (!found_iv || !limit) return false;

    info.loop = &loop;
    info.preheader = preheader;
    info.header = header;
    info.body = body_bb;
    info.latch = latch;
    info.exit_bb = exit_bb;
    info.ph_bt = ph_bt;
    info.latch_bt = latch_bt;
    info.iv_index = iv_idx;
    info.iv_param = header->param(iv_idx);
    info.iv_type = info.iv_param->type();
    info.init_val = ph_bt->args[iv_idx];
    info.limit_val = limit;
    info.step = step_val;
    info.cmp_opcode = cmp_op;
    info.exit_on_false = exit_on_false;

    return true;
}

static bool check_adjacency(const FusibleLoopInfo& l1, const FusibleLoopInfo& l2) {
    if (l1.loop == l2.loop) return false;
    if (l1.loop->parent() != l2.loop->parent()) return false;

    BasicBlock* curr = l1.exit_bb;
    std::unordered_set<const BasicBlock*> visited;

    while (curr && curr != l2.preheader && curr != l2.header) {
        if (!visited.insert(curr).second) return false;
        // The bridge block must not branch or contain side-effects
        Instruction* term = curr->terminator();
        if (!term || term->opcode() != Opcode::br) return false;

        for (Instruction* inst = curr->head(); inst != term; inst = inst->next()) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (is_call(op) || has_side_effects(op)) return false;
        }

        curr = term->branch_target().block;
    }

    if (curr != l2.preheader && curr != l2.header) return false;

    // Loop 2 preheader must have only 1 predecessor (the path from loop 1)
    if (l2.preheader && l2.preheader->predecessors().size() > 1) {
        return false;
    }

    return true;
}

static bool check_domain_congruence(const FusibleLoopInfo& l1, const FusibleLoopInfo& l2) {
    if (l1.step != l2.step) return false;
    if (l1.cmp_opcode != l2.cmp_opcode) return false;
    if (l1.exit_on_false != l2.exit_on_false) return false;

    // Check initial values
    if (l1.init_val != l2.init_val) {
        int64_t c1 = 0, c2 = 0;
        bool has_c1 = get_const_int(l1.init_val, c1);
        bool has_c2 = get_const_int(l2.init_val, c2);
        if (!has_c1 || !has_c2 || c1 != c2) return false;
    }

    // Check limit values
    if (l1.limit_val != l2.limit_val) {
        int64_t c1 = 0, c2 = 0;
        bool has_c1 = get_const_int(l1.limit_val, c1);
        bool has_c2 = get_const_int(l2.limit_val, c2);
        if (!has_c1 || !has_c2 || c1 != c2) return false;
    }

    return true;
}

struct MemAccess {
    Instruction* inst = nullptr;
    Value* base = nullptr;
    Value* index = nullptr;
    int64_t const_offset = 0;
    bool is_store = false;
    bool is_iv_indexed = false;
};

static Value* unwrap_boxed_index(Value* val) {
    while (val && val->is_instruction()) {
        Instruction* def = val->defining_instruction();
        if (!def) break;
        if (def->opcode() == Opcode::call && def->symbol() == "box_f64" && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        if ((def->opcode() == Opcode::bitcast_i64_f64 || def->opcode() == Opcode::bitcast_f64_i64 ||
             def->opcode() == Opcode::sitofp_f64_i64 || def->opcode() == Opcode::sitofp_f64_i32 ||
             def->opcode() == Opcode::fptosi_i64 || def->opcode() == Opcode::fptosi_i32 ||
             def->opcode() == Opcode::sext_i64 || def->opcode() == Opcode::zext_i64) && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        break;
    }
    return val;
}

static void collect_memory_accesses(const FusibleLoopInfo& info, std::vector<MemAccess>& accesses) {
    for (BasicBlock* bb : info.loop->blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();

            if (op == Opcode::store_indexed) {
                MemAccess acc;
                acc.inst = inst;
                acc.base = inst->operand(0);
                acc.index = inst->operand(1);
                acc.is_store = true;
                acc.is_iv_indexed = (acc.index == info.iv_param);
                accesses.push_back(acc);
            } else if (op == Opcode::load_indexed) {
                MemAccess acc;
                acc.inst = inst;
                acc.base = inst->operand(0);
                acc.index = inst->operand(1);
                acc.is_store = false;
                acc.is_iv_indexed = (acc.index == info.iv_param);
                accesses.push_back(acc);
            } else if (op == Opcode::call && inst->symbol() == "bronze_elem_set") {
                MemAccess acc;
                acc.inst = inst;
                acc.base = inst->operand(0);
                acc.index = unwrap_boxed_index(inst->operand(1));
                acc.is_store = true;
                acc.is_iv_indexed = (acc.index == info.iv_param);
                accesses.push_back(acc);
            } else if (op == Opcode::call && inst->symbol() == "bronze_elem_get") {
                MemAccess acc;
                acc.inst = inst;
                acc.base = inst->operand(0);
                acc.index = unwrap_boxed_index(inst->operand(1));
                acc.is_store = false;
                acc.is_iv_indexed = (acc.index == info.iv_param);
                accesses.push_back(acc);
            }
        }
    }
}

static bool check_dependencies(const FusibleLoopInfo& l1, const FusibleLoopInfo& l2) {
    // 1. Check unknown external calls with side effects
    for (BasicBlock* bb : l1.loop->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->is_terminator()) continue;
            Opcode op = inst->opcode();
            if (is_call(op)) {
                std::string_view sym = inst->symbol();
                if (sym != "bronze_elem_get" && sym != "bronze_elem_set" &&
                    sym != "bronze_prop_set" && sym != "bronze_prop_get" &&
                    sym != "box_f64" && sym != "unbox_f64" &&
                    !is_allocation_callee(sym)) {
                    // Unknown call in loop 1 could have side effects
                    return false;
                }
            }
        }
    }
    for (BasicBlock* bb : l2.loop->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->is_terminator()) continue;
            Opcode op = inst->opcode();
            if (is_call(op)) {
                std::string_view sym = inst->symbol();
                if (sym != "bronze_elem_get" && sym != "bronze_elem_set" &&
                    sym != "bronze_prop_set" && sym != "bronze_prop_get" &&
                    sym != "box_f64" && sym != "unbox_f64" &&
                    !is_allocation_callee(sym)) {
                    return false;
                }
            }
        }
    }

    // 2. Check memory conflicts between loop 1 and loop 2
    std::vector<MemAccess> acc1, acc2;
    collect_memory_accesses(l1, acc1);
    collect_memory_accesses(l2, acc2);

    for (const auto& a1 : acc1) {
        for (const auto& a2 : acc2) {
            if (a1.base == a2.base) {
                // If accesses touch the same array, check indices
                if (!a1.is_iv_indexed || !a2.is_iv_indexed) {
                    // Non-IV or non-congruent indexing on shared array is unsafe
                    return false;
                }
                // Both are indexed by their respective primary IV: distance is 0.
                // In fused loop: loop 1 executes before loop 2 in iteration i.
                // A store in loop 1 followed by load in loop 2 is a valid forward dep.
            }
        }
    }

    // 3. Ensure loop 2 does not use a loop-carried non-invariant value computed inside loop 1
    std::unordered_set<const Value*> l1_body_defs;
    for (BasicBlock* bb : l1.loop->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->result()) {
                l1_body_defs.insert(inst->result());
            }
        }
    }

    for (BasicBlock* bb : l2.loop->blocks()) {
        for (Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op_val = inst->operand(i);
                if (op_val && l1_body_defs.count(op_val) > 0) {
                    // Direct SSA use of a per-iteration value from loop 1 across iterations
                    return false;
                }
            }
        }
    }

    return true;
}

} // namespace

bool can_fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    (void)dom;
    if (options.stats) options.stats->candidates_checked++;

    FusibleLoopInfo info1, info2;
    if (!extract_fusible_loop(fn, loop1, info1) || !extract_fusible_loop(fn, loop2, info2)) {
        return false;
    }

    if (!check_adjacency(info1, info2)) {
        if (options.stats) options.stats->rejected_non_adjacent++;
        return false;
    }

    if (!check_domain_congruence(info1, info2)) {
        if (options.stats) options.stats->rejected_domain_mismatch++;
        return false;
    }

    if (!check_dependencies(info1, info2)) {
        if (options.stats) options.stats->rejected_dependencies++;
        return false;
    }

    return true;
}

bool fuse_loops(
    Function& fn,
    LoopInfo& loop1,
    LoopInfo& loop2,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    if (!can_fuse_loops(fn, loop1, loop2, dom, options)) {
        return false;
    }

    FusibleLoopInfo info1, info2;
    extract_fusible_loop(fn, loop1, info1);
    extract_fusible_loop(fn, loop2, info2);

    BasicBlock* hdr1 = info1.header;
    BasicBlock* hdr2 = info2.header;
    BasicBlock* body1 = info1.body;
    BasicBlock* body2 = info2.body;
    BasicBlock* latch1 = info1.latch;
    BasicBlock* latch2 = info2.latch;

    // 1. Extend hdr1 with hdr2's non-IV parameters
    std::unordered_map<Value*, Value*> param_map;
    std::vector<size_t> non_iv_indices;

    Builder b(fn);
    for (size_t i = 0; i < hdr2->param_count(); ++i) {
        if (i == info2.iv_index) continue;
        Value* p2 = hdr2->param(i);
        Value* new_p = b.add_block_param(hdr1, p2->type());
        param_map[p2] = new_p;
        non_iv_indices.push_back(i);
    }

    // 2. Extend preheader branch arguments to hdr1
    if (info1.ph_bt) {
        for (size_t idx : non_iv_indices) {
            Value* init_val = (info2.ph_bt && idx < info2.ph_bt->args.size()) ? info2.ph_bt->args[idx] : nullptr;
            info1.ph_bt->args.push_back(init_val);
        }
    }

    // 2b. Move any invariant/bridge instructions from info2.preheader (and bridge blocks) to info1.preheader
    Instruction* ph1_term = info1.preheader ? info1.preheader->terminator() : nullptr;
    if (ph1_term) {
        BasicBlock* bridge = info1.exit_bb;
        std::unordered_set<BasicBlock*> visited_bridge;
        while (bridge && bridge != info2.header) {
            if (!visited_bridge.insert(bridge).second) break;
            std::vector<Instruction*> bridge_insts;
            for (Instruction* inst = bridge->head(); inst != nullptr; inst = inst->next()) {
                if (!inst->is_terminator()) {
                    bridge_insts.push_back(inst);
                }
            }
            for (Instruction* inst : bridge_insts) {
                bridge->remove_instruction(inst);
                info1.preheader->insert_before(inst, ph1_term);
            }
            Instruction* term = bridge->terminator();
            if (term && term->opcode() == Opcode::br) {
                bridge = term->branch_target().block;
            } else {
                break;
            }
        }
        if (info2.preheader && info2.preheader != info1.exit_bb && visited_bridge.count(info2.preheader) == 0) {
            std::vector<Instruction*> ph2_insts;
            for (Instruction* inst = info2.preheader->head(); inst != nullptr; inst = inst->next()) {
                if (!inst->is_terminator()) {
                    ph2_insts.push_back(inst);
                }
            }
            for (Instruction* inst : ph2_insts) {
                info2.preheader->remove_instruction(inst);
                info1.preheader->insert_before(inst, ph1_term);
            }
        }
    }

    // 2c. Move any non-terminator instructions in hdr2 (other than condition definition) into hdr1
    Instruction* hdr1_term = hdr1->terminator();
    Value* hdr2_cond = hdr2->terminator() ? hdr2->terminator()->operand(0) : nullptr;
    Instruction* hdr2_cond_inst = (hdr2_cond && hdr2_cond->is_instruction()) ? hdr2_cond->defining_instruction() : nullptr;

    std::vector<Instruction*> hdr2_to_move;
    for (Instruction* inst = hdr2->head(); inst != nullptr; inst = inst->next()) {
        if (!inst->is_terminator() && inst != hdr2_cond_inst) {
            hdr2_to_move.push_back(inst);
        }
    }
    for (Instruction* inst : hdr2_to_move) {
        hdr2->remove_instruction(inst);
        for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
            Value* op_val = inst->operand(op_i);
            if (op_val == info2.iv_param) {
                inst->set_operand(op_i, info1.iv_param);
            } else if (param_map.count(op_val) > 0) {
                inst->set_operand(op_i, param_map[op_val]);
            }
        }
        hdr1->insert_before(inst, hdr1_term);
    }

    // 3. Move non-terminator instructions of body2 into body1
    Instruction* latch1_term = latch1->terminator();
    std::vector<Instruction*> to_move;
    for (Instruction* inst = body2->head(); inst != nullptr; inst = inst->next()) {
        if (!inst->is_terminator()) {
            to_move.push_back(inst);
        }
    }

    for (Instruction* inst : to_move) {
        body2->remove_instruction(inst);
        // Remap operands: iv2 -> iv1, params of hdr2 -> new params of hdr1
        for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
            Value* op_val = inst->operand(op_i);
            if (op_val == info2.iv_param) {
                inst->set_operand(op_i, info1.iv_param);
            } else if (param_map.count(op_val) > 0) {
                inst->set_operand(op_i, param_map[op_val]);
            }
        }
        body1->insert_before(inst, latch1_term);
    }

    // 4. Update latch1 backedge arguments
    Instruction* latch2_term = latch2->terminator();
    BranchTarget& l1_bt = latch1_term->branch_target();
    const BranchTarget& l2_bt = latch2_term->branch_target();

    for (size_t idx : non_iv_indices) {
        Value* next_val = (idx < l2_bt.args.size()) ? l2_bt.args[idx] : nullptr;
        if (next_val == info2.iv_param) {
            next_val = info1.iv_param;
        } else if (param_map.count(next_val) > 0) {
            next_val = param_map[next_val];
        }
        l1_bt.args.push_back(next_val);
    }

    // 5. Update hdr1 exit branch to target info2.exit_bb
    hdr1_term = hdr1->terminator();
    Instruction* hdr2_term = hdr2->terminator();
    BranchTarget& l1_exit_target = info1.exit_on_false ? hdr1_term->false_target() : hdr1_term->true_target();
    const BranchTarget& l2_exit_target = info2.exit_on_false ? hdr2_term->false_target() : hdr2_term->true_target();

    std::vector<Value*> orig_l1_exit_args = l1_exit_target.args;

    l1_exit_target.block = l2_exit_target.block;
    l1_exit_target.args.clear();
    for (Value* arg : l2_exit_target.args) {
        if (arg == info2.iv_param) {
            arg = info1.iv_param;
        } else if (param_map.count(arg) > 0) {
            arg = param_map[arg];
        }
        l1_exit_target.args.push_back(arg);
    }

    // If info1.exit_bb had parameters (e.g. reduction accumulators), forward them to info2.exit_bb
    if (info1.exit_bb && info1.exit_bb != info2.header && info1.exit_bb != info2.exit_bb) {
        for (size_t k = 0; k < info1.exit_bb->params().size(); ++k) {
            Value* p = info1.exit_bb->params()[k];
            Value* arg_val = (k < orig_l1_exit_args.size()) ? orig_l1_exit_args[k] : nullptr;
            Value* exit_p = b.add_block_param(info2.exit_bb, p->type());
            replace_all_uses(fn, p, exit_p);
            l1_exit_target.args.push_back(arg_val);
        }
    }

    // Replace any external uses of hdr2 params
    for (auto& [old_p, new_p] : param_map) {
        replace_all_uses(fn, old_p, new_p);
    }

    // 6. Remove dead blocks of loop 2
    if (info1.exit_bb != info2.header && info1.exit_bb != info2.exit_bb && info1.exit_bb != hdr1) {
        fn.remove_block(info1.exit_bb);
    }
    if (info2.preheader && info2.preheader != hdr1 && info2.preheader != info2.exit_bb) {
        fn.remove_block(info2.preheader);
    }
    if (body2 != body1) fn.remove_block(body2);
    if (latch2 != body2 && latch2 != latch1) fn.remove_block(latch2);
    fn.remove_block(hdr2);

    fn.rebuild_cfg_predecessors();

    if (options.stats) options.stats->loops_fused++;
    return true;
}

bool loop_fusion_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopFusionOptions& options
) {
    bool any_fused = false;
    bool changed = true;

    while (changed) {
        changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree current_dom(fn);
        LoopAnalysis la(fn, current_dom);

        const auto& loops = la.top_level_loops();
        for (size_t i = 0; i < loops.size(); ++i) {
            for (size_t j = 0; j < loops.size(); ++j) {
                if (i == j) continue;
                if (can_fuse_loops(fn, *loops[i], *loops[j], current_dom, options)) {
                    if (fuse_loops(fn, *loops[i], *loops[j], current_dom, options)) {
                        changed = true;
                        any_fused = true;
                        break;
                    }
                }
            }
            if (changed) break;
        }
    }

    return any_fused;
}

} // namespace brass
