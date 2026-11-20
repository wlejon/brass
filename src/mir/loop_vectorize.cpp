#include <brass/mir/loop_vectorize.hpp>
#include "loop_vectorize_analysis.hpp"
#include <brass/mir/builder.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

namespace brass {

namespace {

static Value* build_const_step(Builder& b, Type t, int64_t val) {
    if (t == Type::i32()) {
        return b.build_iconst_i32(static_cast<int32_t>(val));
    }
    return b.build_iconst_i64(val);
}

static Type vec_type_for_lane(Type elem, uint32_t w) {
    if (elem == Type::f32()) return (w == 8) ? Type::f32x8() : (w == 4) ? Type::f32x4() : Type::void_type();
    if (elem == Type::i32()) return (w == 8) ? Type::i32x8() : (w == 4) ? Type::i32x4() : Type::void_type();
    if (elem == Type::f64()) return (w == 4) ? Type::f64x4() : (w == 2) ? Type::f64x2() : Type::void_type();
    if (elem == Type::i64()) return (w == 4) ? Type::i64x4() : (w == 2) ? Type::i64x2() : Type::void_type();
    return Type::void_type();
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
    BasicBlock* vec_preheader = b.create_block(base_name + "_vec_preheader");
    BasicBlock* vec_hdr = b.create_block(base_name + "_vec_hdr");
    BasicBlock* vec_body = b.create_block(base_name + "_vec_body");
    BasicBlock* vec_exit = b.create_block(base_name + "_vec_exit");
    BasicBlock* rem_hdr = b.create_block(base_name + "_rem_hdr");
    BasicBlock* rem_body = b.create_block(base_name + "_rem_body");

    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), header);
    fn_blocks.insert(it, vec_preheader);
    it = std::find(fn_blocks.begin(), fn_blocks.end(), header);
    fn_blocks.insert(it, vec_hdr);
    fn_blocks.push_back(vec_body);
    fn_blocks.push_back(vec_exit);
    fn_blocks.push_back(rem_hdr);
    fn_blocks.push_back(rem_body);

    vec_preheader->set_parent(&fn);
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

    for (size_t i = 0; i < vec_hdr->param_count(); ++i) {
        b.add_block_param(vec_exit, vec_hdr->param(i)->type());
    }

    // =========================================================================
    // 3. Populate Preheader Branch to vec_hdr with Trip Count Guard
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

    // Trip count guard: ensure at least one full vector width W iteration can execute without signed overflow
    Value* w_minus_1_step = build_const_step(b, vli.iv_type, static_cast<int64_t>(W - 1));
    Value* last_vec_iv = b.build_add(vli.init_iv, w_minus_1_step);
    Value* tc_guard = nullptr;
    if (vli.cmp_opcode == Opcode::ult || vli.cmp_opcode == Opcode::ule) {
        Value* no_ovf = b.build_ule(vli.init_iv, last_vec_iv);
        Value* in_limit = (vli.cmp_opcode == Opcode::ule)
            ? b.build_ule(last_vec_iv, vli.limit_val)
            : b.build_ult(last_vec_iv, vli.limit_val);
        tc_guard = b.build_and(no_ovf, in_limit);
    } else {
        Value* no_ovf = b.build_sle(vli.init_iv, last_vec_iv);
        Value* in_limit = (vli.cmp_opcode == Opcode::sle)
            ? b.build_sle(last_vec_iv, vli.limit_val)
            : b.build_slt(last_vec_iv, vli.limit_val);
        tc_guard = b.build_and(no_ovf, in_limit);
    }

    // Runtime alias checks for memory operations that may alias
    auto widen_i64 = [&](Value* v) -> Value* {
        return (v->type() == Type::i32()) ? b.build_sext_i64(v) : v;
    };
    for (const auto& check : vli.alias_checks) {
        uint8_t shift1 = (check.op1.elem_type.size_in_bytes() == 4) ? 2 : 3;
        uint8_t shift2 = (check.op2.elem_type.size_in_bytes() == 4) ? 2 : 3;
        Value* s1 = b.build_iconst_i64(shift1);
        Value* s2 = b.build_iconst_i64(shift2);
        Value* init64 = widen_i64(vli.init_iv);
        Value* lim64 = widen_i64(vli.limit_val);

        Value* off_init1 = b.build_add(b.build_shl(init64, s1), b.build_iconst_i64(check.op1.offset));
        Value* off_limit1 = b.build_add(b.build_shl(lim64, s1), b.build_iconst_i64(check.op1.offset));
        Value* ptr_start1 = b.build_add(check.op1.base, off_init1);
        Value* ptr_end1 = b.build_add(check.op1.base, off_limit1);

        Value* off_init2 = b.build_add(b.build_shl(init64, s2), b.build_iconst_i64(check.op2.offset));
        Value* off_limit2 = b.build_add(b.build_shl(lim64, s2), b.build_iconst_i64(check.op2.offset));
        Value* ptr_start2 = b.build_add(check.op2.base, off_init2);
        Value* ptr_end2 = b.build_add(check.op2.base, off_limit2);

        Value* cond1 = b.build_ule(ptr_end1, ptr_start2);
        Value* cond2 = b.build_ule(ptr_end2, ptr_start1);
        Value* no_overlap = b.build_or(cond1, cond2);
        tc_guard = b.build_and(tc_guard, no_overlap);
    }

    Instruction* vec_pre_term = nullptr;
    if (ph_term->opcode() == Opcode::br) {
        ph_term->parent()->remove_instruction(ph_term);
        b.position_at_end(preheader);
        b.build_br_if(tc_guard, vec_preheader, {}, vec_exit, vec_ph_args);
        b.position_at_end(vec_preheader);
        vec_pre_term = b.build_br(vec_hdr, vec_ph_args);
    } else if (ph_term->opcode() == Opcode::br_if) {
        BasicBlock* vec_guard = b.create_block(base_name + "_vec_guard");
        auto& fn_blks = fn.blocks();
        auto it_vp = std::find(fn_blks.begin(), fn_blks.end(), vec_preheader);
        fn_blks.insert(it_vp, vec_guard);
        vec_guard->set_parent(&fn);

        if (ph_term->true_target().block == header) {
            ph_term->set_true_target(BranchTarget(vec_guard, vec_ph_args));
        } else {
            ph_term->set_false_target(BranchTarget(vec_guard, vec_ph_args));
        }

        b.position_at_end(vec_guard);
        std::vector<Value*> guard_params;
        for (Value* a : vec_ph_args) {
            guard_params.push_back(b.add_block_param(vec_guard, a->type()));
        }
        b.build_br_if(tc_guard, vec_preheader, {}, vec_exit, guard_params);
        b.position_at_end(vec_preheader);
        vec_pre_term = b.build_br(vec_hdr, guard_params);
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
    Builder pre_b(*fn.parent());
    pre_b.set_function(&fn);
    std::unordered_set<const Value*> hoisted_scalars;
    std::unordered_map<const Value*, std::unordered_map<uint8_t, Value*>> preheader_bcasts;

    auto get_or_build_broadcast = [&](Type vtype, Value* sc) -> Value* {
        bool sc_before_hdr = !sc->defining_block() ||
                             (sc->defining_block() != vec_hdr && sc->defining_block() != vec_body);
        if (vec_pre_term && sc_before_hdr && (hoisted_scalars.count(sc) || loop.is_loop_invariant(sc))) {
            auto& inner = preheader_bcasts[sc];
            uint8_t tk = static_cast<uint8_t>(vtype.kind());
            auto it_bc = inner.find(tk);
            if (it_bc != inner.end()) return it_bc->second;
            pre_b.position_before(vec_pre_term);
            Value* bc = pre_b.build_vbroadcast(vtype, sc);
            inner[tk] = bc;
            return bc;
        }
        return b.build_vbroadcast(vtype, sc);
    };

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
        // Constants are rematerialized in vec_preheader (the scalar body does
        // not dominate the vector body, and hoisting avoids per-iteration broadcasts).
        if (inst->opcode() == Opcode::iconst_i64) {
            Builder& cb = vec_pre_term ? pre_b : b;
            if (vec_pre_term) cb.position_before(vec_pre_term);
            Value* c = cb.build_iconst_i64(inst->imm_i64());
            vec_map[inst->result()] = c;
            if (vec_pre_term) hoisted_scalars.insert(c);
            continue;
        }
        if (inst->opcode() == Opcode::iconst_i32) {
            Builder& cb = vec_pre_term ? pre_b : b;
            if (vec_pre_term) cb.position_before(vec_pre_term);
            Value* c = cb.build_iconst_i32(inst->imm_i32());
            vec_map[inst->result()] = c;
            if (vec_pre_term) hoisted_scalars.insert(c);
            continue;
        }
        if (inst->opcode() == Opcode::fconst_f64) {
            Builder& cb = vec_pre_term ? pre_b : b;
            if (vec_pre_term) cb.position_before(vec_pre_term);
            Value* c = (inst->type() == Type::f32())
                ? cb.build_fconst_f32(static_cast<float>(inst->imm_f64()))
                : cb.build_fconst_f64(inst->imm_f64());
            vec_map[inst->result()] = c;
            if (vec_pre_term) hoisted_scalars.insert(c);
            continue;
        }

        if (vli.has_reduction && inst == vli.reduction_update_inst) {
            if (inst->opcode() == Opcode::fma_f32 || inst->opcode() == Opcode::fma_f64) {
                auto to_vec = [&](Value* v) -> Value* {
                    Value* mv = vec_map[v];
                    if (!mv) return get_or_build_broadcast(vli.vec_type, v);
                    if (mv->type() == vli.elem_type) return get_or_build_broadcast(vli.vec_type, mv);
                    return mv;
                };
                Value* va = to_vec(inst->operand(0));
                Value* vb = to_vec(inst->operand(1));
                Value* acc_op = inst->operand(2);
                Value* current_acc = vec_map[acc_op];
                if (current_acc && current_acc->type() == vli.elem_type) {
                    current_acc = get_or_build_broadcast(vli.vec_type, current_acc);
                }
                next_vec_acc = b.build_vfma(va, vb, current_acc ? current_acc : vec_acc_param);
                vec_map[inst->result()] = next_vec_acc;
                continue;
            }
            Value* acc_op = (inst->operand(0) == header->param(vli.reduction_param_index))
                                ? inst->operand(0)
                                : inst->operand(1);
            Value* val_op = (inst->operand(0) == acc_op) ? inst->operand(1) : inst->operand(0);

            Value* vec_val = vec_map[val_op];
            if (!vec_val) {
                if (val_op->type() != vli.elem_type) {
                    throw std::logic_error("loop_vectorize: reduction operand has no vector form");
                }
                vec_val = get_or_build_broadcast(vli.vec_type, val_op);
            } else if (vec_val->type() == vli.elem_type) {
                vec_val = get_or_build_broadcast(vli.vec_type, vec_val);
            }
            Value* current_acc = vec_map[acc_op];
            if (current_acc && current_acc->type() == vli.elem_type) {
                current_acc = get_or_build_broadcast(vli.vec_type, current_acc);
            }
            next_vec_acc = b.build_vadd(current_acc ? current_acc : vec_acc_param, vec_val);
            vec_map[inst->result()] = next_vec_acc;
            continue;
        }

        if (inst->opcode() == Opcode::load_indexed) {
            Value* base = inst->operand(0);
            int32_t offset = inst->offset();
            uint8_t shift_amount = (vli.elem_type.size_in_bytes() == 4) ? 2 : 3;
            Value* iv64 = widen_i64(vec_iv_param);
            Value* byte_off = b.build_shl(iv64, b.build_iconst_i64(shift_amount));
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
                if (val_to_store->type() != vli.elem_type) {
                    throw std::logic_error("loop_vectorize: stored value has no vector form");
                }
                vec_val = get_or_build_broadcast(vli.vec_type, val_to_store);
            } else if (vec_val->type() == vli.elem_type) {
                vec_val = get_or_build_broadcast(vli.vec_type, vec_val);
            }
            uint8_t shift_amount = (vli.elem_type.size_in_bytes() == 4) ? 2 : 3;
            Value* iv64 = widen_i64(vec_iv_param);
            Value* byte_off = b.build_shl(iv64, b.build_iconst_i64(shift_amount));
            Value* addr = b.build_add(base, byte_off);
            b.build_vstore(vli.vec_type, addr, offset, vec_val);
            continue;
        }

        // Only vectorize lane-width-compatible operations
        Type out_vec_type = vec_type_for_lane(inst->type(), W);
        if (!out_vec_type.is_vector() || inst->type().size_in_bytes() != vli.elem_type.size_in_bytes()) {
            continue;
        }

        std::vector<Value*> vec_operands;
        bool can_vectorize = true;
        for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
            Value* op = inst->operand(op_i);
            Type op_elem = op ? op->type() : Type::void_type();
            Type op_vec = vec_type_for_lane(op_elem, W);
            if (!op_vec.is_vector() || op_elem.size_in_bytes() != vli.elem_type.size_in_bytes()) {
                can_vectorize = false;
                break;
            }
            auto it_v = vec_map.find(op);
            if (it_v != vec_map.end()) {
                Value* v_op = it_v->second;
                if (v_op->type() == op_vec) {
                    vec_operands.push_back(v_op);
                } else if (v_op->type() == op_elem) {
                    vec_operands.push_back(get_or_build_broadcast(op_vec, v_op));
                } else {
                    can_vectorize = false;
                    break;
                }
            } else if (loop.is_loop_invariant(op)) {
                vec_operands.push_back(get_or_build_broadcast(op_vec, op));
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
            case Opcode::fma_f32:
            case Opcode::fma_f64: v_res = b.build_vfma(vec_operands[0], vec_operands[1], vec_operands[2]); break;
            case Opcode::sdiv:
            case Opcode::udiv: v_res = b.build_vdiv(vec_operands[0], vec_operands[1]); break;
            case Opcode::neg: v_res = b.build_vneg(vec_operands[0]); break;
            case Opcode::fmin_f32:
            case Opcode::fmin_f64: v_res = b.build_vmin(vec_operands[0], vec_operands[1]); break;
            case Opcode::fmax_f32:
            case Opcode::fmax_f64: v_res = b.build_vmax(vec_operands[0], vec_operands[1]); break;
            case Opcode::floor_f32:
            case Opcode::floor_f64: v_res = b.build_vfloor(vec_operands[0]); break;
            case Opcode::ceil_f32:
            case Opcode::ceil_f64: v_res = b.build_vceil(vec_operands[0]); break;
            case Opcode::round_f32:
            case Opcode::round_f64: v_res = b.build_vround(vec_operands[0]); break;
            case Opcode::fptosi_i32_f32:
            case Opcode::sitofp_f32_i32: v_res = b.build_vconvert(out_vec_type, vec_operands[0]); break;
            case Opcode::bitcast_f32_i32:
            case Opcode::bitcast_i32_f32:
            case Opcode::bitcast_f64_i64:
            case Opcode::bitcast_i64_f64: v_res = b.build_vbitcast(out_vec_type, vec_operands[0]); break;
            case Opcode::shl: v_res = b.build_vshl(vec_operands[0], vec_operands[1]); break;
            case Opcode::lshr: v_res = b.build_vlshr(vec_operands[0], vec_operands[1]); break;
            case Opcode::ashr: v_res = b.build_vashr(vec_operands[0], vec_operands[1]); break;
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

    Value* exit_iv = vec_exit->param(vli.primary_iv_index);
    Value* final_scalar_acc = nullptr;

    if (vli.has_reduction) {
        Value* exit_acc = vec_exit->param(vli.reduction_param_index);
        auto reduce_add = [&](Value* x, Value* y) -> Value* {
            if (vli.reduction_type.is_float()) {
                return b.build_fadd(x, y);
            }
            return b.build_add(x, y);
        };

        if (W == 8) {
            Value* l0 = b.build_vextract_lane(exit_acc, 0);
            Value* l1 = b.build_vextract_lane(exit_acc, 1);
            Value* l2 = b.build_vextract_lane(exit_acc, 2);
            Value* l3 = b.build_vextract_lane(exit_acc, 3);
            Value* l4 = b.build_vextract_lane(exit_acc, 4);
            Value* l5 = b.build_vextract_lane(exit_acc, 5);
            Value* l6 = b.build_vextract_lane(exit_acc, 6);
            Value* l7 = b.build_vextract_lane(exit_acc, 7);
            Value* s01 = reduce_add(l0, l1);
            Value* s23 = reduce_add(l2, l3);
            Value* s45 = reduce_add(l4, l5);
            Value* s67 = reduce_add(l6, l7);
            Value* s0123 = reduce_add(s01, s23);
            Value* s4567 = reduce_add(s45, s67);
            Value* hsum = reduce_add(s0123, s4567);
            final_scalar_acc = reduce_add(hsum, vli.reduction_init_val);
        } else if (W == 4) {
            Value* l0 = b.build_vextract_lane(exit_acc, 0);
            Value* l1 = b.build_vextract_lane(exit_acc, 1);
            Value* l2 = b.build_vextract_lane(exit_acc, 2);
            Value* l3 = b.build_vextract_lane(exit_acc, 3);
            Value* s01 = reduce_add(l0, l1);
            Value* s23 = reduce_add(l2, l3);
            Value* hsum = reduce_add(s01, s23);
            final_scalar_acc = reduce_add(hsum, vli.reduction_init_val);
        } else {
            Value* l0 = b.build_vextract_lane(exit_acc, 0);
            Value* l1 = b.build_vextract_lane(exit_acc, 1);
            Value* hsum = reduce_add(l0, l1);
            final_scalar_acc = reduce_add(hsum, vli.reduction_init_val);
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
        cloned->copy_attributes_from(*inst);
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

    std::unordered_map<Value*, Value*> exit_rewrite_map;
    std::vector<Value*> final_exit_args;

    if (!old_exit_target.args.empty()) {
        for (size_t arg_i = 0; arg_i < old_exit_target.args.size(); ++arg_i) {
            Value* arg = old_exit_target.args[arg_i];
            if (arg && arg->is_block_param() && arg->defining_block() == header) {
                final_exit_args.push_back(rem_hdr->param(arg->param_index()));
                if (arg_i < exit_bb->param_count()) {
                    exit_rewrite_map[arg] = exit_bb->param(arg_i);
                }
            } else {
                final_exit_args.push_back(arg);
            }
        }
    }

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* hp = header->param(i);
        if (exit_rewrite_map.count(hp)) continue;

        bool is_used_outside = false;
        for (BasicBlock* bb : fn.blocks()) {
            if (bb == vec_preheader || bb == vec_hdr || bb == vec_body || bb == vec_exit ||
                bb == rem_hdr || bb == rem_body || loop.contains(bb)) {
                continue;
            }
            for (Instruction* inst : *bb) {
                for (Value* op : inst->operands()) {
                    if (op == hp) { is_used_outside = true; break; }
                }
                if (is_used_outside) break;
                for (Value* a : inst->branch_target().args) {
                    if (a == hp) { is_used_outside = true; break; }
                }
                if (is_used_outside) break;
                for (Value* a : inst->true_target().args) {
                    if (a == hp) { is_used_outside = true; break; }
                }
                if (is_used_outside) break;
                for (Value* a : inst->false_target().args) {
                    if (a == hp) { is_used_outside = true; break; }
                }
                if (is_used_outside) break;
                for (const auto& sc : inst->switch_cases()) {
                    for (Value* a : sc.target.args) {
                        if (a == hp) { is_used_outside = true; break; }
                    }
                    if (is_used_outside) break;
                }
                if (is_used_outside) break;
            }
            if (is_used_outside) break;
        }

        if (is_used_outside) {
            Value* new_param = b.add_block_param(exit_bb, hp->type());
            exit_rewrite_map[hp] = new_param;
            final_exit_args.push_back(rem_hdr->param(i));
        }
    }

    b.build_br_if(rem_cond, rem_body, {}, exit_bb, final_exit_args);

    if (!exit_rewrite_map.empty()) {
        auto update_val = [&](Value*& v) {
            if (v && exit_rewrite_map.count(v)) {
                v = exit_rewrite_map[v];
            }
        };

        for (BasicBlock* bb : fn.blocks()) {
            if (bb == vec_preheader || bb == vec_hdr || bb == vec_body || bb == vec_exit ||
                bb == rem_hdr || bb == rem_body || loop.contains(bb)) {
                continue;
            }
            for (Instruction* inst : *bb) {
                for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                    Value* op = inst->operand(op_i);
                    if (op && exit_rewrite_map.count(op)) {
                        inst->set_operand(op_i, exit_rewrite_map[op]);
                    }
                }
                for (Value*& a : inst->branch_target().args) update_val(a);
                for (Value*& a : inst->true_target().args) update_val(a);
                for (Value*& a : inst->false_target().args) update_val(a);
                for (auto& sc : inst->switch_cases()) {
                    for (Value*& a : sc.target.args) update_val(a);
                }
            }
        }
    }

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