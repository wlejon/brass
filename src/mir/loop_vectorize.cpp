#include <brass/mir/loop_vectorize.hpp>
#include "loop_vectorize_analysis.hpp"
#include <brass/mir/builder.hpp>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace brass {

namespace {

static Value* build_const_step(Builder& b, Type t, int64_t val) {
    if (t == Type::i32()) {
        return b.build_iconst_i32(static_cast<int32_t>(val));
    }
    return b.build_iconst_i64(val);
}

} // namespace

bool vectorize_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopVectorizeOptions& options
) {
    VectorizableLoopInfo vli;
    if (!analyze_vectorizable_loop(fn, loop, dom, vli, options)) {
        return false;
    }

    uint32_t W = vli.vector_width;
    BasicBlock* header = vli.header;
    BasicBlock* body = vli.body;
    BasicBlock* preheader = vli.preheader;
    BasicBlock* exit_bb = vli.exit_bb;

    Builder b(*fn.parent());
    b.set_function(&fn);

    // =========================================================================
    // 1. Create Blocks
    // =========================================================================
    std::string base_name = std::string(header->name());
    BasicBlock* vec_hdr = b.create_block(base_name + "_vec_hdr");
    BasicBlock* vec_body = b.create_block(base_name + "_vec_body");
    BasicBlock* vec_exit = b.create_block(base_name + "_vec_exit");
    BasicBlock* rem_hdr = b.create_block(base_name + "_rem_hdr");
    BasicBlock* rem_body = b.create_block(base_name + "_rem_body");

    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), header);
    fn_blocks.insert(it, vec_hdr);
    fn_blocks.push_back(vec_body);
    fn_blocks.push_back(vec_exit);
    fn_blocks.push_back(rem_hdr);
    fn_blocks.push_back(rem_body);

    vec_hdr->set_parent(&fn);
    vec_body->set_parent(&fn);
    vec_exit->set_parent(&fn);
    rem_hdr->set_parent(&fn);
    rem_body->set_parent(&fn);

    // =========================================================================
    // 2. Set Up Vector Header Parameters
    // =========================================================================
    // Parameters in vec_hdr:
    // For primary IV: iv_type
    // For reduction: vec_type
    // For invariants: original param type
    std::vector<Value*> vec_hdr_params;
    Value* vec_iv_param = nullptr;
    Value* vec_acc_param = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == vli.primary_iv_index) {
            vec_iv_param = b.add_block_param(vec_hdr, vli.iv_type);
            vec_hdr_params.push_back(vec_iv_param);
        } else if (vli.has_reduction && i == vli.reduction_param_index) {
            vec_acc_param = b.add_block_param(vec_hdr, vli.vec_type);
            vec_hdr_params.push_back(vec_acc_param);
        } else {
            Value* p = b.add_block_param(vec_hdr, header->param(i)->type());
            vec_hdr_params.push_back(p);
        }
    }

    // =========================================================================
    // 3. Populate Preheader Branch to vec_hdr
    // =========================================================================
    Instruction* ph_term = preheader->terminator();
    b.position_before(ph_term);

    std::vector<Value*> vec_ph_args;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == vli.primary_iv_index) {
            vec_ph_args.push_back(vli.init_iv);
        } else if (vli.has_reduction && i == vli.reduction_param_index) {
            Value* zero_acc = b.build_vzero(vli.vec_type);
            vec_ph_args.push_back(zero_acc);
        } else {
            if (ph_term->opcode() == Opcode::br) {
                vec_ph_args.push_back(ph_term->branch_target().args[i]);
            } else if (ph_term->opcode() == Opcode::br_if) {
                if (ph_term->true_target().block == header) {
                    vec_ph_args.push_back(ph_term->true_target().args[i]);
                } else {
                    vec_ph_args.push_back(ph_term->false_target().args[i]);
                }
            }
        }
    }

    if (ph_term->opcode() == Opcode::br) {
        ph_term->set_branch_target(BranchTarget(vec_hdr, vec_ph_args));
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) {
            ph_term->set_true_target(BranchTarget(vec_hdr, vec_ph_args));
        } else {
            ph_term->set_false_target(BranchTarget(vec_hdr, vec_ph_args));
        }
    }

    // =========================================================================
    // 4. Vector Header Condition & Branch
    // =========================================================================
    b.position_at_end(vec_hdr);
    Value* w_minus_1 = build_const_step(b, vli.iv_type, static_cast<int64_t>(W - 1));
    Value* vec_limit = b.build_sub(vli.limit_val, w_minus_1);

    Value* vec_cond = nullptr;
    switch (vli.cmp_opcode) {
        case Opcode::slt: vec_cond = b.build_slt(vec_iv_param, vec_limit); break;
        case Opcode::ult: vec_cond = b.build_ult(vec_iv_param, vec_limit); break;
        case Opcode::sle: vec_cond = b.build_sle(vec_iv_param, vec_limit); break;
        case Opcode::ule: vec_cond = b.build_ule(vec_iv_param, vec_limit); break;
        default: vec_cond = b.build_slt(vec_iv_param, vec_limit); break;
    }

    b.build_br_if(vec_cond, vec_body, {}, vec_exit, vec_hdr->params());

    // =========================================================================
    // 5. Vector Body Transformation
    // =========================================================================
    b.position_at_end(vec_body);
    std::unordered_map<const Value*, Value*> vec_map;

    for (size_t i = 0; i < header->param_count(); ++i) {
        vec_map[header->param(i)] = vec_hdr_params[i];
    }

    Value* next_vec_acc = vec_acc_param;

    Value* next_iv = nullptr;
    Instruction* step_inst = nullptr;
    Instruction* latch_term = vli.latch->terminator();
    if (latch_term && latch_term->opcode() == Opcode::br) {
        if (vli.primary_iv_index < latch_term->branch_target().args.size()) {
            next_iv = latch_term->branch_target().args[vli.primary_iv_index];
            if (next_iv && next_iv->is_instruction()) {
                step_inst = next_iv->defining_instruction();
            }
        }
    }

    for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator()) break;

        if (inst == step_inst || (next_iv && inst->result() == next_iv)) {
            continue;
        }
        if (inst->opcode() == Opcode::iconst_i64 || inst->opcode() == Opcode::iconst_i32) {
            continue;
        }

        if (vli.has_reduction && inst == vli.reduction_update_inst) {
            Value* acc_op = (inst->operand(0) == header->param(vli.reduction_param_index))
                                ? inst->operand(0)
                                : inst->operand(1);
            Value* val_op = (inst->operand(0) == acc_op) ? inst->operand(1) : inst->operand(0);

            Value* vec_val = vec_map[val_op];
            if (!vec_val) {
                if (val_op->type() == vli.elem_type) {
                    vec_val = b.build_vbroadcast(vli.vec_type, val_op);
                } else {
                    continue;
                }
            }
            next_vec_acc = b.build_vadd(vec_map[acc_op], vec_val);
            vec_map[inst->result()] = next_vec_acc;
            continue;
        }

        if (inst->opcode() == Opcode::load_indexed) {
            Value* base = inst->operand(0);
            int32_t offset = inst->offset();
            uint8_t shift_amount = (vli.vector_width == 4) ? 2 : 3;
            Value* shift_val = build_const_step(b, vli.iv_type, shift_amount);
            Value* byte_off = b.build_shl(vec_iv_param, shift_val);
            Value* addr = b.build_add(base, byte_off);
            Value* vloaded = b.build_vload(vli.vec_type, addr, offset);
            vec_map[inst->result()] = vloaded;
            continue;
        }

        if (inst->opcode() == Opcode::store_indexed) {
            Value* base = inst->operand(0);
            int32_t offset = inst->offset();
            Value* val_to_store = inst->operand(2);
            Value* vec_val = vec_map[val_to_store];
            if (!vec_val) {
                if (val_to_store->type() == vli.elem_type) {
                    vec_val = b.build_vbroadcast(vli.vec_type, val_to_store);
                } else {
                    continue;
                }
            }
            uint8_t shift_amount = (vli.vector_width == 4) ? 2 : 3;
            Value* shift_val = build_const_step(b, vli.iv_type, shift_amount);
            Value* byte_off = b.build_shl(vec_iv_param, shift_val);
            Value* addr = b.build_add(base, byte_off);
            b.build_vstore(vli.vec_type, addr, offset, vec_val);
            continue;
        }

        // Only vectorize element-type arithmetic
        if (inst->type() != vli.elem_type) {
            continue;
        }

        std::vector<Value*> vec_operands;
        bool can_vectorize = true;
        for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
            Value* op = inst->operand(op_i);
            auto it_v = vec_map.find(op);
            if (it_v != vec_map.end()) {
                vec_operands.push_back(it_v->second);
            } else if (op->type() == vli.elem_type) {
                vec_operands.push_back(b.build_vbroadcast(vli.vec_type, op));
            } else {
                can_vectorize = false;
                break;
            }
        }
        if (!can_vectorize) continue;

        Value* v_res = nullptr;
        Opcode op = inst->opcode();
        switch (op) {
            case Opcode::add: v_res = b.build_vadd(vec_operands[0], vec_operands[1]); break;
            case Opcode::sub: v_res = b.build_vsub(vec_operands[0], vec_operands[1]); break;
            case Opcode::mul: v_res = b.build_vmul(vec_operands[0], vec_operands[1]); break;
            case Opcode::sdiv:
            case Opcode::udiv: v_res = b.build_vdiv(vec_operands[0], vec_operands[1]); break;
            case Opcode::neg: v_res = b.build_vneg(vec_operands[0]); break;
            case Opcode::and_: v_res = b.build_vand(vec_operands[0], vec_operands[1]); break;
            case Opcode::or_: v_res = b.build_vor(vec_operands[0], vec_operands[1]); break;
            case Opcode::xor_: v_res = b.build_vxor(vec_operands[0], vec_operands[1]); break;
            case Opcode::not_: v_res = b.build_vnot(vec_operands[0]); break;
            default: break;
        }

        if (v_res && inst->produces_value()) {
            vec_map[inst->result()] = v_res;
        }
    }

    // Step primary IV by W
    Value* w_step = build_const_step(b, vli.iv_type, static_cast<int64_t>(W));
    Value* next_vec_iv = b.build_add(vec_iv_param, w_step);

    std::vector<Value*> vec_latch_args;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == vli.primary_iv_index) {
            vec_latch_args.push_back(next_vec_iv);
        } else if (vli.has_reduction && i == vli.reduction_param_index) {
            vec_latch_args.push_back(next_vec_acc);
        } else {
            vec_latch_args.push_back(vec_hdr_params[i]);
        }
    }

    b.build_br(vec_hdr, vec_latch_args);

    // =========================================================================
    // 6. Vector Exit Block (Horizontal Reduction & Branch to Remainder)
    // =========================================================================
    b.position_at_end(vec_exit);

    for (size_t i = 0; i < vec_hdr->param_count(); ++i) {
        b.add_block_param(vec_exit, vec_hdr->param(i)->type());
    }

    Value* exit_iv = vec_exit->param(vli.primary_iv_index);
    Value* final_scalar_acc = nullptr;

    if (vli.has_reduction) {
        Value* exit_acc = vec_exit->param(vli.reduction_param_index);
        if (W == 4) {
            Value* l0 = b.build_vextract_lane(exit_acc, 0);
            Value* l1 = b.build_vextract_lane(exit_acc, 1);
            Value* l2 = b.build_vextract_lane(exit_acc, 2);
            Value* l3 = b.build_vextract_lane(exit_acc, 3);
            Value* s01 = b.build_add(l0, l1);
            Value* s23 = b.build_add(l2, l3);
            Value* hsum = b.build_add(s01, s23);
            final_scalar_acc = b.build_add(hsum, vli.reduction_init_val);
        } else {
            Value* l0 = b.build_vextract_lane(exit_acc, 0);
            Value* l1 = b.build_vextract_lane(exit_acc, 1);
            Value* hsum = b.build_add(l0, l1);
            final_scalar_acc = b.build_add(hsum, vli.reduction_init_val);
        }
    }

    std::vector<Value*> rem_init_args;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == vli.primary_iv_index) {
            rem_init_args.push_back(exit_iv);
        } else if (vli.has_reduction && i == vli.reduction_param_index) {
            rem_init_args.push_back(final_scalar_acc);
        } else {
            rem_init_args.push_back(vec_exit->param(i));
        }
    }

    b.build_br(rem_hdr, rem_init_args);

    // =========================================================================
    // 7. Remainder Header & Body
    // =========================================================================
    b.position_at_end(rem_hdr);
    for (size_t i = 0; i < header->param_count(); ++i) {
        b.add_block_param(rem_hdr, header->param(i)->type());
    }

    Value* rem_iv = rem_hdr->param(vli.primary_iv_index);
    Value* rem_cond = nullptr;
    switch (vli.cmp_opcode) {
        case Opcode::slt: rem_cond = b.build_slt(rem_iv, vli.limit_val); break;
        case Opcode::ult: rem_cond = b.build_ult(rem_iv, vli.limit_val); break;
        case Opcode::sle: rem_cond = b.build_sle(rem_iv, vli.limit_val); break;
        case Opcode::ule: rem_cond = b.build_ule(rem_iv, vli.limit_val); break;
        default: rem_cond = b.build_slt(rem_iv, vli.limit_val); break;
    }

    // Remainder Body
    b.position_at_end(rem_body);
    std::unordered_map<const Value*, Value*> rem_map;
    for (size_t i = 0; i < header->param_count(); ++i) {
        rem_map[header->param(i)] = rem_hdr->param(i);
    }

    for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator()) break;

        Instruction* cloned = fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type());
        cloned->set_imm_i64(inst->imm_i64());
        cloned->set_imm_f64(inst->imm_f64());
        cloned->set_scale(inst->scale());
        cloned->set_offset(inst->offset());
        cloned->set_memory_type(inst->memory_type());
        if (!inst->symbol().empty()) cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));

        for (Value* op : inst->operands()) {
            if (!op) continue;
            auto it_v = rem_map.find(op);
            cloned->add_operand(it_v != rem_map.end() ? it_v->second : op);
        }

        if (inst->produces_value()) {
            Value* res = fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
            res->set_defining_instruction(cloned);
            cloned->set_result(res);
            rem_map[inst->result()] = res;
        }

        rem_body->append_instruction(cloned);
    }

    Value* step_one = build_const_step(b, vli.iv_type, 1);
    Value* next_rem_iv = b.build_add(rem_iv, step_one);

    std::vector<Value*> rem_latch_args;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == vli.primary_iv_index) {
            rem_latch_args.push_back(next_rem_iv);
        } else if (vli.has_reduction && i == vli.reduction_param_index) {
            rem_latch_args.push_back(rem_map[vli.reduction_update_inst->result()]);
        } else {
            rem_latch_args.push_back(rem_hdr->param(i));
        }
    }

    b.build_br(rem_hdr, rem_latch_args);

    // Remainder Header Terminator
    b.position_at_end(rem_hdr);
    Instruction* old_hdr_term = header->terminator();
    const BranchTarget& old_exit_target = vli.exit_on_false ? old_hdr_term->false_target() : old_hdr_term->true_target();

    std::vector<Value*> final_exit_args;
    if (!old_exit_target.args.empty()) {
        for (Value* arg : old_exit_target.args) {
            if (arg && arg->is_block_param() && arg->defining_block() == header) {
                final_exit_args.push_back(rem_hdr->param(arg->param_index()));
            } else {
                final_exit_args.push_back(arg);
            }
        }
    }

    b.build_br_if(rem_cond, rem_body, {}, exit_bb, final_exit_args);

    // =========================================================================
    // 8. Remove Old Loop Blocks
    // =========================================================================
    fn.remove_block(header);
    if (body != header) {
        fn.remove_block(body);
    }
    if (vli.latch != header && vli.latch != body) {
        fn.remove_block(vli.latch);
    }

    fn.rebuild_cfg_predecessors();
    return true;
}

bool vectorize_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom
) {
    return vectorize_loop(fn, loop, dom, LoopVectorizeOptions());
}

bool loop_vectorize_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopVectorizeOptions& options
) {
    LoopAnalysis loops(fn, dom);
    std::vector<LoopInfo*> post_order = loops.post_order_loops();

    bool any_changed = false;
    for (LoopInfo* loop : post_order) {
        if (loop) {
            if (vectorize_loop(fn, *loop, dom, options)) {
                any_changed = true;
                break;
            }
        }
    }
    return any_changed;
}

bool loop_vectorize_pass(
    Function& fn,
    const DominatorTree& dom
) {
    return loop_vectorize_pass(fn, dom, LoopVectorizeOptions());
}

} // namespace brass