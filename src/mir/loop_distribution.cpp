#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iostream>

namespace brass {

std::string LoopDistributionStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Loop Distribution Statistics ===\n"
       << "  Loops distributed: " << loops_distributed << "\n"
       << "  Candidates checked: " << candidates_checked << "\n"
       << "  Rejected (pure vectorizable): " << rejected_pure_vectorizable << "\n"
       << "  Rejected (pure scalar): " << rejected_pure_scalar << "\n"
       << "  Rejected (dependency cycles): " << rejected_cycles << "\n";
    return ss.str();
}

namespace {

static bool get_const_int(const Value* val, int64_t& out_val) {
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

static bool is_supported_vector_arithmetic(Opcode op) {
    switch (op) {
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::neg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::not_:
        case Opcode::select:
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
            return true;
        default:
            return false;
    }
}

static bool is_unvectorizable_opcode(Opcode op) {
    if (is_call(op)) return true;
    switch (op) {
        case Opcode::safepoint:
        case Opcode::guard:
        case Opcode::resume_point:
        case Opcode::throw_:
        case Opcode::invoke:
        case Opcode::landing_pad:
        case Opcode::resume:
        case Opcode::osr_entry:
        case Opcode::coro_create:
        case Opcode::coro_suspend:
        case Opcode::coro_resume:
        case Opcode::coro_destroy:
            return true;
        default:
            return false;
    }
}

struct DistributableLoopInfo {
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
    Instruction* iv_inc_inst = nullptr;
};

static bool extract_distributable_loop(Function& fn, LoopInfo& loop, DistributableLoopInfo& info) {
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
    if (loop.blocks().size() > 2) return false; // Single body + header

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
    Instruction* iv_inc = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
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
                        iv_inc = def;
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
    info.iv_inc_inst = iv_inc;

    return true;
}

} // namespace

bool can_distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    (void)dom;
    if (options.stats) options.stats->candidates_checked++;

    DistributableLoopInfo info;
    if (!extract_distributable_loop(fn, loop, info)) return false;

    // Inspect instructions in the body
    size_t vectorizable_count = 0;
    size_t unvectorizable_count = 0;

    for (Instruction* inst = info.body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator() || inst == info.iv_inc_inst) continue;
        Opcode op = inst->opcode();

        if (is_unvectorizable_opcode(op)) {
            unvectorizable_count++;
        } else if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
            Value* base = inst->operand(0);
            Value* idx = inst->operand(1);
            if (info.loop->is_loop_invariant(base) && idx == info.iv_param) {
                vectorizable_count++;
            } else {
                unvectorizable_count++;
            }
        } else if (is_supported_vector_arithmetic(op)) {
            vectorizable_count++;
        } else {
            unvectorizable_count++;
        }
    }

    if (unvectorizable_count == 0) {
        if (options.stats) options.stats->rejected_pure_vectorizable++;
        return false;
    }
    if (vectorizable_count == 0) {
        if (options.stats) options.stats->rejected_pure_scalar++;
        return false;
    }

    // Check dependency graph: Partition 2 = unvectorizable and its downstream dependencies.
    // Partition 1 = remaining instructions.
    // Partition 1 must not depend on Partition 2.
    std::unordered_set<Instruction*> part2;
    for (Instruction* inst = info.body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator() || inst == info.iv_inc_inst) continue;
        Opcode op = inst->opcode();
        if (is_unvectorizable_opcode(op)) {
            part2.insert(inst);
        } else if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
            Value* base = inst->operand(0);
            Value* idx = inst->operand(1);
            if (!info.loop->is_loop_invariant(base) || idx != info.iv_param) {
                part2.insert(inst);
            }
        } else if (!is_supported_vector_arithmetic(op)) {
            part2.insert(inst);
        }
    }

    // Grow part2 forward through SSA uses
    bool changed = true;
    while (changed) {
        changed = false;
        for (Instruction* inst = info.body->head(); inst != nullptr; inst = inst->next()) {
            if (inst->is_terminator() || inst == info.iv_inc_inst || part2.count(inst) > 0) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op_val = inst->operand(i);
                if (op_val && op_val->is_instruction()) {
                    Instruction* def = op_val->defining_instruction();
                    if (def && part2.count(def) > 0) {
                        part2.insert(inst);
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    // Check if any vectorizable instruction in Partition 1 remains, with at least one store
    size_t part1_count = 0;
    size_t part1_stores = 0;
    for (Instruction* inst = info.body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator() || inst == info.iv_inc_inst) continue;
        if (part2.count(inst) == 0) {
            part1_count++;
            if (inst->opcode() == Opcode::store_indexed || inst->opcode() == Opcode::store) {
                part1_stores++;
            }
        }
    }

    if (part1_count == 0 || part1_stores == 0) {
        if (options.stats) options.stats->rejected_cycles++;
        return false;
    }

    return true;
}

bool distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    if (!can_distribute_loop(fn, loop, dom, options)) return false;

    DistributableLoopInfo info;
    extract_distributable_loop(fn, loop, info);

    BasicBlock* hdr = info.header;
    BasicBlock* body = info.body;
    BasicBlock* latch = info.latch;
    BasicBlock* ph = info.preheader;
    BasicBlock* orig_exit = info.exit_bb;

    // Partition instructions
    std::unordered_set<Instruction*> part2_set;
    for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator() || inst == info.iv_inc_inst) continue;
        Opcode op = inst->opcode();
        if (is_unvectorizable_opcode(op)) {
            part2_set.insert(inst);
        } else if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
            Value* base = inst->operand(0);
            Value* idx = inst->operand(1);
            if (!info.loop->is_loop_invariant(base) || idx != info.iv_param) {
                part2_set.insert(inst);
            }
        } else if (!is_supported_vector_arithmetic(op)) {
            part2_set.insert(inst);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
            if (inst->is_terminator() || inst == info.iv_inc_inst || part2_set.count(inst) > 0) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op_val = inst->operand(i);
                if (op_val && op_val->is_instruction()) {
                    Instruction* def = op_val->defining_instruction();
                    if (def && part2_set.count(def) > 0) {
                        part2_set.insert(inst);
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    std::vector<Instruction*> part1_insts;
    std::vector<Instruction*> part2_insts;

    for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator() || inst == info.iv_inc_inst) continue;
        if (part2_set.count(inst) > 0) {
            part2_insts.push_back(inst);
        } else {
            part1_insts.push_back(inst);
        }
    }

    // Create Loop 2 blocks:
    // ph -> hdr1 -> body1 -> latch1 -> exit1
    // exit1 -> hdr2 -> body2 -> latch2 -> orig_exit
    Builder b(*fn.parent());
    b.set_function(&fn);

    BasicBlock* exit1 = b.create_block("dist_exit1");
    BasicBlock* hdr2 = b.create_block("dist_hdr2");
    BasicBlock* body2 = b.create_block("dist_body2");

    fn.append_block(exit1);
    fn.append_block(hdr2);
    fn.append_block(body2);

    // Setup hdr2 parameters
    Value* iv2 = b.add_block_param(hdr2, info.iv_type);

    // Any values computed in part1 and used in part2:
    // If stored to an array, reload from array.
    // If not stored to an array, allocate buffer in preheader, store in loop 1, reload in loop 2.
    std::unordered_map<Value*, Value*> array_reloads;

    // Find array stores in part1: map value -> array base
    std::unordered_map<Value*, Value*> stored_arrays;
    for (Instruction* inst : part1_insts) {
        if (inst->opcode() == Opcode::store_indexed && inst->operand(1) == info.iv_param) {
            stored_arrays[inst->operand(2)] = inst->operand(0);
        }
    }

    // For any instruction in part2 using a def from part1:
    b.position_at_end(body2);
    for (Instruction* inst : part2_insts) {
        for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
            Value* op_val = inst->operand(op_i);
            if (op_val == info.iv_param) {
                inst->set_operand(op_i, iv2);
                continue;
            }
            if (!op_val || !op_val->is_instruction()) continue;
            Instruction* def = op_val->defining_instruction();
            if (def && def->parent() == body && part2_set.count(def) == 0) {
                // op_val is defined in part1 and used in part2
                if (array_reloads.count(op_val) == 0) {
                    Value* reloaded = nullptr;
                    uint8_t scale = static_cast<uint8_t>(op_val->type().size_in_bytes());
                    auto it_arr = stored_arrays.find(op_val);
                    if (it_arr != stored_arrays.end()) {
                        Value* base = it_arr->second;
                        reloaded = b.build_load_indexed(op_val->type(), base, iv2, scale);
                    } else {
                        // Create temporary buffer in preheader
                        Builder ph_b(*fn.parent());
                        ph_b.set_function(&fn);
                        ph_b.position_before(ph->terminator());
                        Value* elem_size = ph_b.build_iconst_i64(static_cast<int64_t>(scale));
                        Value* total_bytes = ph_b.build_mul(info.limit_val, elem_size);
                        Value* tmp_buf = ph_b.build_call("brass_gc_alloc", Type::gcref(), {total_bytes});

                        // Store in loop 1
                        Builder b1(*fn.parent());
                        b1.set_function(&fn);
                        b1.position_before(latch->terminator());
                        b1.build_store_indexed(op_val->type(), tmp_buf, info.iv_param, scale, op_val);

                        // Reload in loop 2
                        reloaded = b.build_load_indexed(op_val->type(), tmp_buf, iv2, scale);
                    }
                    array_reloads[op_val] = reloaded;
                }
                inst->set_operand(op_i, array_reloads[op_val]);
            }
        }
    }

    // Move part2 instructions into body2
    for (Instruction* inst : part2_insts) {
        body->remove_instruction(inst);
        body2->append_instruction(inst);
    }

    // Connect body2: compute next iv2 and branch back to hdr2
    b.position_at_end(body2);
    Value* step_val = (info.iv_type == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(info.step)) : b.build_iconst_i64(info.step);
    Value* next_iv2 = b.build_add(iv2, step_val);
    b.build_br(hdr2, {next_iv2});

    // Setup hdr2 terminator
    b.position_at_end(hdr2);
    Value* cond2 = nullptr;
    switch (info.cmp_opcode) {
        case Opcode::slt: cond2 = b.build_slt(iv2, info.limit_val); break;
        case Opcode::ult: cond2 = b.build_ult(iv2, info.limit_val); break;
        case Opcode::sle: cond2 = b.build_sle(iv2, info.limit_val); break;
        case Opcode::ule: cond2 = b.build_ule(iv2, info.limit_val); break;
        default: cond2 = b.build_slt(iv2, info.limit_val); break;
    }
    if (info.exit_on_false) {
        b.build_br_if(cond2, body2, {}, orig_exit, {});
    } else {
        b.build_br_if(cond2, orig_exit, {}, body2, {});
    }

    // Setup exit1: branches to hdr2 with init_val
    b.position_at_end(exit1);
    b.build_br(hdr2, {info.init_val});

    // Update hdr1 exit branch to target exit1
    Instruction* hdr_term = hdr->terminator();
    BranchTarget& exit_tgt = info.exit_on_false ? hdr_term->false_target() : hdr_term->true_target();
    exit_tgt.block = exit1;
    exit_tgt.args.clear();

    fn.rebuild_cfg_predecessors();

    if (options.stats) options.stats->loops_distributed++;
    return true;
}

bool loop_distribution_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    bool any_distributed = false;
    bool changed = true;

    while (changed) {
        changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree current_dom(fn);
        LoopAnalysis la(fn, current_dom);

        for (LoopInfo* loop : la.post_order_loops()) {
            if (!loop) continue;
            if (can_distribute_loop(fn, *loop, current_dom, options)) {
                if (distribute_loop(fn, *loop, current_dom, options)) {
                    changed = true;
                    any_distributed = true;
                    break;
                }
            }
        }
    }

    return any_distributed;
}

} // namespace brass
