#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>

namespace brass {

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

Value* make_smart_const_int(Builder& b, Type t, int64_t val) {
    if (t == Type::i32()) {
        return b.build_iconst_i32(static_cast<int32_t>(val));
    }
    return b.build_iconst_i64(val);
}

Value* make_smart_mul(Builder& b, Type t, Value* val, int64_t mul_factor) {
    if (mul_factor == 1) return val;
    if (mul_factor == 0) return make_smart_const_int(b, t, 0);
    int64_t c;
    if (get_const_int(val, c)) {
        return make_smart_const_int(b, t, c * mul_factor);
    }
    Value* factor_val = make_smart_const_int(b, t, mul_factor);
    return b.build_mul(val, factor_val);
}

Value* make_smart_add(Builder& b, Type t, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) {
        return make_smart_const_int(b, t, c0 + c1);
    }
    if (has_c0 && c0 == 0) return rhs;
    if (has_c1 && c1 == 0) return lhs;
    return b.build_add(lhs, rhs);
}

Value* get_invariant_val(Builder& b, Type t, Value* val) {
    if (!val) return nullptr;
    int64_t c;
    if (get_const_int(val, c)) {
        return make_smart_const_int(b, t, c);
    }
    return val;
}

enum class ParamRole {
    BasicIV,
    DerivedIV,
    Invariant,
    ReductionAcc,
    SerialReductionAcc
};

struct ParamAnalysis {
    ParamRole role = ParamRole::Invariant;
    Value* header_param = nullptr;
    size_t param_index = 0;
    Value* ph_init_val = nullptr;
    Value* latch_next_val = nullptr;
    Value* step_val = nullptr;
    Instruction* update_inst = nullptr;
    bool is_sub = false;
    Type type = Type::void_type();
};

struct CountedLoopAnalysis {
    bool is_counted = false;
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* exit_bb = nullptr;

    size_t primary_iv_index = 0;
    Opcode cmp_opcode = Opcode::slt;
    Value* limit_val = nullptr;
    bool exit_on_false = true; // br_if cond, body, exit

    std::vector<ParamAnalysis> params;
    std::vector<size_t> reduction_indices;
};

static bool value_depends_on_param(
    const Value* val,
    const Value* param,
    const LoopInfo& loop,
    std::unordered_set<const Value*>& visited
) {
    if (!val) return false;
    if (val == param) return true;
    if (!val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def || !loop.contains(def->parent())) return false;
    if (visited.count(val)) return false;
    visited.insert(val);
    for (size_t op_i = 0; op_i < def->operand_count(); ++op_i) {
        if (value_depends_on_param(def->operand(op_i), param, loop, visited)) {
            return true;
        }
    }
    return false;
}

bool analyze_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    CountedLoopAnalysis& cla,
    const LoopUnrollOptions& options
) {
    (void)fn;
    (void)dom;
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return false;
    if (header->name().find("_unroll_hdr") != std::string_view::npos ||
        header->name().find("_rem_hdr") != std::string_view::npos) {
        return false;
    }
    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || !latch_term || !hdr_term) return false;

    // Check preheader branch
    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
        ph_bt = &ph_term->branch_target();
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    // Check latch branch
    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
        latch_bt = &latch_term->branch_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    // Check header conditional branch
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

    // For single-body inner loops (e.g. header + body or header == latch)
    if (loop.blocks().size() > 2) {
        // Only unroll simple 1-block or 2-block loops for now
        return false;
    }

    // Verify all instructions in loop have no side effects, no calls, no safepoints
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->is_call() || (inst->has_side_effects() && inst != latch_term && inst != hdr_term)) {
                return false;
            }
        }
    }

    // Analyze parameters
    cla.params.resize(header->param_count());
    bool found_primary_iv = false;

    Value* cmp_lhs = cmp_inst->operand(0);
    Value* cmp_rhs = cmp_inst->operand(1);

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        Value* ph_init = ph_bt->args[i];
        Value* latch_next = latch_bt->args[i];

        ParamAnalysis& pa = cla.params[i];
        pa.header_param = param;
        pa.param_index = i;
        pa.ph_init_val = ph_init;
        pa.latch_next_val = latch_next;
        pa.type = param->type();

        if (latch_next == param) {
            pa.role = ParamRole::Invariant;
            continue;
        }

        if (latch_next && latch_next->is_instruction()) {
            Instruction* def = latch_next->defining_instruction();
            if (def && loop.contains(def->parent())) {
                if (def->opcode() == Opcode::add) {
                    Value* op0 = def->operand(0);
                    Value* op1 = def->operand(1);
                    if (op0 == param && loop.is_loop_invariant(op1)) {
                        pa.role = ParamRole::DerivedIV;
                        pa.step_val = op1;
                        pa.update_inst = def;
                    } else if (op1 == param && loop.is_loop_invariant(op0)) {
                        pa.role = ParamRole::DerivedIV;
                        pa.step_val = op0;
                        pa.update_inst = def;
                    } else if (op0 == param || op1 == param) {
                        Value* term = (op0 == param) ? op1 : op0;
                        if (term != param) {
                            std::unordered_set<const Value*> visited;
                            if (!value_depends_on_param(term, param, loop, visited)) {
                                if (pa.type == Type::f64() && !options.enable_fp_reduction_jam) {
                                    pa.role = ParamRole::SerialReductionAcc;
                                    pa.update_inst = def;
                                    cla.reduction_indices.push_back(i);
                                } else if (options.enable_reduction_jam) {
                                    pa.role = ParamRole::ReductionAcc;
                                    pa.update_inst = def;
                                    cla.reduction_indices.push_back(i);
                                }
                            }
                        }
                    }
                } else if (def->opcode() == Opcode::sub) {
                    if (def->operand(0) == param && loop.is_loop_invariant(def->operand(1))) {
                        pa.role = ParamRole::DerivedIV;
                        pa.step_val = def->operand(1);
                        pa.update_inst = def;
                        pa.is_sub = true;
                    }
                }
            }
        }

        // Check if this param is the primary IV used in header condition
        if (!found_primary_iv && (param == cmp_lhs || param == cmp_rhs)) {
            if (pa.role == ParamRole::DerivedIV && pa.step_val != nullptr) {
                pa.role = ParamRole::BasicIV;
                cla.primary_iv_index = i;
                found_primary_iv = true;

                if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
                    cla.cmp_opcode = cmp_inst->opcode();
                    cla.limit_val = cmp_rhs;
                } else if (param == cmp_rhs && loop.is_loop_invariant(cmp_lhs)) {
                    // Reversed comparison
                    cla.cmp_opcode = cmp_inst->opcode();
                    cla.limit_val = cmp_lhs;
                }
            }
        }
    }

    // Ensure all parameters with loop changes are recognized
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.latch_next_val != pa.header_param && pa.role == ParamRole::Invariant) {
            return false;
        }
    }

    if (!found_primary_iv || !cla.limit_val) return false;

    // Verify that the primary IV and all derived IVs are integer types (Type::i32() or Type::i64())
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
            if (pa.type != Type::i32() && pa.type != Type::i64()) {
                return false;
            }
        }
    }

    // Only handle standard ascending `<` or `<=` comparisons (e.g. slt, ult, sle, ule)
    if (cla.cmp_opcode != Opcode::slt && cla.cmp_opcode != Opcode::ult &&
        cla.cmp_opcode != Opcode::sle && cla.cmp_opcode != Opcode::ule) {
        return false;
    }

    // Do not unroll loops containing non-pure control calls or safepoints
    for (Instruction* inst = body_bb->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator()) continue;
        Opcode op = inst->opcode();
        if (op == Opcode::call || op == Opcode::call_indirect || op == Opcode::patchable_call ||
            op == Opcode::safepoint || op == Opcode::guard || op == Opcode::resume_point) {
            return false;
        }
    }

    cla.is_counted = true;
    cla.header = header;
    cla.body = body_bb;
    cla.latch = latch;
    cla.preheader = preheader;
    cla.exit_bb = exit_bb;
    cla.exit_on_false = exit_on_false;

    return true;
}

} // namespace

bool unroll_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopUnrollOptions& options
) {
    CountedLoopAnalysis cla;
    if (!analyze_loop(fn, loop, dom, cla, options)) {
        return false;
    }

    size_t F = options.unroll_factor;
    if (F != 2 && F != 4 && F != 8) {
        F = 4;
    }

    BasicBlock* header = cla.header;
    BasicBlock* body = cla.body;
    BasicBlock* preheader = cla.preheader;
    BasicBlock* exit_bb = cla.exit_bb;

    const ParamAnalysis& primary_iv = cla.params[cla.primary_iv_index];
    Type iv_type = primary_iv.type;

    Builder b(*fn.parent());
    b.set_function(&fn);

    // =========================================================================
    // 1. Create Blocks for Unrolled Loop, Reduction, and Remainder Loop
    // =========================================================================
    std::string base_name = std::string(header->name());
    BasicBlock* unroll_hdr = b.create_block(base_name + "_unroll_hdr");
    BasicBlock* unroll_body = b.create_block(base_name + "_unroll_body");
    BasicBlock* unroll_exit = b.create_block(base_name + "_unroll_exit");

    BasicBlock* rem_hdr = b.create_block(base_name + "_rem_hdr");
    BasicBlock* rem_body = b.create_block(base_name + "_rem_body");

    // Insert unroll_hdr before header
    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), header);
    fn_blocks.insert(it, unroll_hdr);
    fn_blocks.push_back(unroll_body);
    fn_blocks.push_back(unroll_exit);
    fn_blocks.push_back(rem_hdr);
    fn_blocks.push_back(rem_body);

    unroll_hdr->set_parent(&fn);
    unroll_body->set_parent(&fn);
    unroll_exit->set_parent(&fn);
    rem_hdr->set_parent(&fn);
    rem_body->set_parent(&fn);

    // =========================================================================
    // 2. Set Up Unroll Header Parameters (Multiple parallel accumulators)
    // =========================================================================
    // Param mapping:
    // For non-reduction params: 1 param in unroll_hdr
    // For reduction params: F params (acc0, acc1, ..., acc_{F-1}) in unroll_hdr
    std::vector<Value*> unroll_hdr_params;
    std::vector<std::vector<Value*>> reduction_acc_params(header->param_count());

    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            reduction_acc_params[i].resize(F);
            for (size_t k = 0; k < F; ++k) {
                Value* acc_p = b.add_block_param(unroll_hdr, pa.type);
                reduction_acc_params[i][k] = acc_p;
            }
        } else {
            Value* p = b.add_block_param(unroll_hdr, pa.type);
            unroll_hdr_params.push_back(p);
        }
    }

    // =========================================================================
    // 3. Populate Preheader Branch with F initial accumulator values
    // =========================================================================
    Instruction* ph_term = preheader->terminator();
    b.position_before(ph_term);

    std::vector<Value*> unroll_ph_args;
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            // acc0 gets original initial value
            unroll_ph_args.push_back(pa.ph_init_val);
            // acc1..acc_{F-1} get identity 0.0 or 0
            for (size_t k = 1; k < F; ++k) {
                if (pa.type == Type::f64()) {
                    unroll_ph_args.push_back(b.build_fconst_f64(0.0));
                } else if (pa.type == Type::i32()) {
                    unroll_ph_args.push_back(b.build_iconst_i32(0));
                } else {
                    unroll_ph_args.push_back(b.build_iconst_i64(0));
                }
            }
        } else {
            unroll_ph_args.push_back(pa.ph_init_val);
        }
    }

    // Replace preheader branch target
    ph_term->set_branch_target(BranchTarget(unroll_hdr, std::move(unroll_ph_args)));

    // =========================================================================
    // 4. Unroll Header: Evaluate Main Loop Trip Condition
    // =========================================================================
    b.position_at_end(unroll_hdr);

    // Primary IV in unroll_hdr:
    size_t iv_unroll_idx = 0;
    for (size_t i = 0; i < cla.primary_iv_index; ++i) {
        if (cla.params[i].role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            // skipped from unroll_hdr_params
        } else {
            iv_unroll_idx++;
        }
    }
    Value* cur_iv = unroll_hdr_params[iv_unroll_idx];

    // Compute check_val = cur_iv + (F - 1) * iv_step
    Value* iv_step = get_invariant_val(b, iv_type, primary_iv.step_val);
    Value* f_minus_1_step = make_smart_mul(b, iv_type, iv_step, static_cast<int64_t>(F - 1));
    Value* check_iv = make_smart_add(b, iv_type, cur_iv, f_minus_1_step);
    Value* limit_val = get_invariant_val(b, iv_type, cla.limit_val);

    Value* unroll_cond = nullptr;
    switch (cla.cmp_opcode) {
        case Opcode::slt: unroll_cond = b.build_slt(check_iv, limit_val); break;
        case Opcode::ult: unroll_cond = b.build_ult(check_iv, limit_val); break;
        case Opcode::sle: unroll_cond = b.build_sle(check_iv, limit_val); break;
        case Opcode::ule: unroll_cond = b.build_ule(check_iv, limit_val); break;
        case Opcode::sgt: unroll_cond = b.build_sgt(check_iv, limit_val); break;
        case Opcode::ugt: unroll_cond = b.build_ugt(check_iv, limit_val); break;
        case Opcode::sge: unroll_cond = b.build_sge(check_iv, limit_val); break;
        case Opcode::uge: unroll_cond = b.build_uge(check_iv, limit_val); break;
        default: unroll_cond = b.build_slt(check_iv, limit_val); break;
    }

    // Branch to unroll_body (on true) or unroll_exit (on false)
    std::vector<Value*> unroll_exit_args = unroll_hdr->params();
    b.build_br_if(unroll_cond, unroll_body, {}, unroll_exit, unroll_exit_args);

    // =========================================================================
    // 5. Unroll Body: Emit F Copies of the Body Instructions
    // =========================================================================
    b.position_at_end(unroll_body);

    std::vector<Value*> cur_acc_vals(header->param_count());
    std::vector<std::vector<Value*>> next_acc_vals(header->param_count());
    std::unordered_map<size_t, Value*> serial_acc_map;

    size_t init_non_red_idx = 0;
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            next_acc_vals[i] = reduction_acc_params[i];
        } else {
            Value* p = unroll_hdr_params[init_non_red_idx++];
            if (pa.role == ParamRole::SerialReductionAcc) {
                serial_acc_map[i] = p;
            }
        }
    }

    bool has_serial_reduction = false;
    for (size_t red_i : cla.reduction_indices) {
        if (cla.params[red_i].role == ParamRole::SerialReductionAcc) {
            has_serial_reduction = true;
            break;
        }
    }

    std::vector<std::unordered_map<const Value*, Value*>> all_iter_maps(F);

    if (has_serial_reduction) {
        // Pass 1: Emit all independent operand computations for all F iterations
        for (size_t k = 0; k < F; ++k) {
            auto& iter_val_map = all_iter_maps[k];
            size_t non_red_idx = 0;

            for (size_t i = 0; i < header->param_count(); ++i) {
                const ParamAnalysis& pa = cla.params[i];
                Value* orig_param = header->param(i);

                if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
                    iter_val_map[orig_param] = next_acc_vals[i][k];
                } else if (pa.role == ParamRole::SerialReductionAcc) {
                    non_red_idx++;
                } else if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
                    Value* base_p = unroll_hdr_params[non_red_idx++];
                    if (k == 0) {
                        iter_val_map[orig_param] = base_p;
                    } else {
                        Value* step_v = get_invariant_val(b, pa.type, pa.step_val);
                        Value* k_step = make_smart_mul(b, pa.type, step_v, static_cast<int64_t>(k));
                        Value* k_val = pa.is_sub ? b.build_sub(base_p, k_step) : make_smart_add(b, pa.type, base_p, k_step);
                        iter_val_map[orig_param] = k_val;
                    }
                } else {
                    iter_val_map[orig_param] = unroll_hdr_params[non_red_idx++];
                }
            }

            for (Instruction* inst = body->head(); inst != nullptr; inst = inst->next()) {
                if (inst->is_terminator()) break;

                bool is_serial_update = false;
                for (size_t red_i : cla.reduction_indices) {
                    if (cla.params[red_i].role == ParamRole::SerialReductionAcc && inst == cla.params[red_i].update_inst) {
                        is_serial_update = true;
                        break;
                    }
                }
                if (is_serial_update) continue;

                Instruction* cloned = fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type());
                cloned->set_imm_i64(inst->imm_i64());
                cloned->set_imm_f64(inst->imm_f64());
                cloned->set_scale(inst->scale());
                cloned->set_offset(inst->offset());
                cloned->set_memory_type(inst->memory_type());
                if (!inst->symbol().empty()) cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));

                for (Value* op : inst->operands()) {
                    if (!op) continue;
                    auto it_v = iter_val_map.find(op);
                    if (it_v != iter_val_map.end()) {
                        cloned->add_operand(it_v->second);
                    } else {
                        cloned->add_operand(op);
                    }
                }

                if (inst->produces_value()) {
                    Value* res = fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
                    res->set_defining_instruction(cloned);
                    cloned->set_result(res);
                    iter_val_map[inst->result()] = res;
                }

                unroll_body->append_instruction(cloned);
            }
        }

        // Pass 2: Emit the serial reduction update chain in strict program order
        for (size_t k = 0; k < F; ++k) {
            auto& iter_val_map = all_iter_maps[k];
            for (size_t red_i : cla.reduction_indices) {
                if (cla.params[red_i].role != ParamRole::SerialReductionAcc) continue;
                Instruction* inst = cla.params[red_i].update_inst;
                Value* orig_param = header->param(red_i);

                iter_val_map[orig_param] = serial_acc_map[red_i];

                Instruction* cloned = fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type());
                cloned->set_imm_i64(inst->imm_i64());
                cloned->set_imm_f64(inst->imm_f64());
                cloned->set_scale(inst->scale());
                cloned->set_offset(inst->offset());
                cloned->set_memory_type(inst->memory_type());
                if (!inst->symbol().empty()) cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));

                for (Value* op : inst->operands()) {
                    if (!op) continue;
                    auto it_v = iter_val_map.find(op);
                    if (it_v != iter_val_map.end()) {
                        cloned->add_operand(it_v->second);
                    } else {
                        cloned->add_operand(op);
                    }
                }

                Value* res = fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
                res->set_defining_instruction(cloned);
                cloned->set_result(res);
                iter_val_map[inst->result()] = res;
                serial_acc_map[red_i] = res;

                unroll_body->append_instruction(cloned);
            }
        }
    } else {
        for (size_t k = 0; k < F; ++k) {
            std::unordered_map<const Value*, Value*> iter_val_map;

            // Map header parameters for iteration k
            size_t non_red_idx = 0;
            for (size_t i = 0; i < header->param_count(); ++i) {
                const ParamAnalysis& pa = cla.params[i];
                Value* orig_param = header->param(i);

                if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
                    // Map accumulator to its current accumulator value
                    iter_val_map[orig_param] = next_acc_vals[i][k];
                } else if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
                    Value* base_p = unroll_hdr_params[non_red_idx++];
                    if (k == 0) {
                        iter_val_map[orig_param] = base_p;
                    } else {
                        Value* step_v = get_invariant_val(b, pa.type, pa.step_val);
                        Value* k_step = make_smart_mul(b, pa.type, step_v, static_cast<int64_t>(k));
                        Value* k_val = pa.is_sub ? b.build_sub(base_p, k_step) : make_smart_add(b, pa.type, base_p, k_step);
                        iter_val_map[orig_param] = k_val;
                    }
                } else {
                    // Invariant
                    iter_val_map[orig_param] = unroll_hdr_params[non_red_idx++];
                }
            }

            // Clone non-terminator body instructions
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
                    auto it_v = iter_val_map.find(op);
                    if (it_v != iter_val_map.end()) {
                        cloned->add_operand(it_v->second);
                    } else {
                        cloned->add_operand(op);
                    }
                }

                if (inst->produces_value()) {
                    Value* res = fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
                    res->set_defining_instruction(cloned);
                    cloned->set_result(res);
                    iter_val_map[inst->result()] = res;

                    // Check if this instruction was the reduction accumulator update
                    for (size_t red_i : cla.reduction_indices) {
                        if (inst == cla.params[red_i].update_inst) {
                            if (cla.params[red_i].role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
                                next_acc_vals[red_i][k] = res;
                            }
                        }
                    }
                }

                unroll_body->append_instruction(cloned);
            }
        }
    }

    // Step primary and derived IVs by F in latch jump
    std::vector<Value*> next_unroll_latch_args;
    size_t non_red_idx = 0;

    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            for (size_t k = 0; k < F; ++k) {
                next_unroll_latch_args.push_back(next_acc_vals[i][k]);
            }
        } else if (pa.role == ParamRole::SerialReductionAcc) {
            next_unroll_latch_args.push_back(serial_acc_map[i]);
            non_red_idx++;
        } else if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
            Value* base_p = unroll_hdr_params[non_red_idx++];
            Value* step_v = get_invariant_val(b, pa.type, pa.step_val);
            Value* f_step = make_smart_mul(b, pa.type, step_v, static_cast<int64_t>(F));
            Value* next_v = pa.is_sub ? b.build_sub(base_p, f_step) : make_smart_add(b, pa.type, base_p, f_step);
            next_unroll_latch_args.push_back(next_v);
        } else {
            next_unroll_latch_args.push_back(unroll_hdr_params[non_red_idx++]);
        }
    }

    b.build_br(unroll_hdr, next_unroll_latch_args);

    // =========================================================================
    // 6. Unroll Exit: Tree-Reduce Accumulators & Prepare Remainder Loop
    // =========================================================================
    b.position_at_end(unroll_exit);

    // unroll_exit receives params matching unroll_hdr
    for (size_t i = 0; i < unroll_hdr->param_count(); ++i) {
        b.add_block_param(unroll_exit, unroll_hdr->param(i)->type());
    }

    // Unpack unroll_exit params into non-reduction and reduction values
    std::vector<Value*> exit_non_red_params;
    std::vector<std::vector<Value*>> exit_acc_params(header->param_count());

    size_t exit_param_idx = 0;
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            exit_acc_params[i].resize(F);
            for (size_t k = 0; k < F; ++k) {
                exit_acc_params[i][k] = unroll_exit->param(exit_param_idx++);
            }
        } else {
            exit_non_red_params.push_back(unroll_exit->param(exit_param_idx++));
        }
    }

    // Tree-reduce accumulators in unroll_exit
    std::vector<Value*> reduced_acc_finals(header->param_count(), nullptr);
    for (size_t red_i : cla.reduction_indices) {
        if (cla.params[red_i].role != ParamRole::ReductionAcc || !options.enable_reduction_jam) {
            continue;
        }
        const auto& accs = exit_acc_params[red_i];
        if (F == 4) {
            Value* s01 = b.build_add(accs[0], accs[1]);
            Value* s23 = b.build_add(accs[2], accs[3]);
            reduced_acc_finals[red_i] = b.build_add(s01, s23);
        } else if (F == 2) {
            reduced_acc_finals[red_i] = b.build_add(accs[0], accs[1]);
        } else if (F == 8) {
            Value* s01 = b.build_add(accs[0], accs[1]);
            Value* s23 = b.build_add(accs[2], accs[3]);
            Value* s45 = b.build_add(accs[4], accs[5]);
            Value* s67 = b.build_add(accs[6], accs[7]);
            Value* s0123 = b.build_add(s01, s23);
            Value* s4567 = b.build_add(s45, s67);
            reduced_acc_finals[red_i] = b.build_add(s0123, s4567);
        } else {
            Value* sum = accs[0];
            for (size_t k = 1; k < accs.size(); ++k) {
                sum = b.build_add(sum, accs[k]);
            }
            reduced_acc_finals[red_i] = sum;
        }
    }

    // Assemble arguments to remainder loop header
    std::vector<Value*> rem_hdr_init_args;
    non_red_idx = 0;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (cla.params[i].role == ParamRole::ReductionAcc && options.enable_reduction_jam) {
            rem_hdr_init_args.push_back(reduced_acc_finals[i]);
        } else {
            rem_hdr_init_args.push_back(exit_non_red_params[non_red_idx++]);
        }
    }

    b.build_br(rem_hdr, rem_hdr_init_args);

    // =========================================================================
    // 7. Remainder Loop Header & Body (Single-iteration cleanup loop)
    // =========================================================================
    b.position_at_end(rem_hdr);

    for (size_t i = 0; i < header->param_count(); ++i) {
        b.add_block_param(rem_hdr, header->param(i)->type());
    }

    Value* rem_iv = rem_hdr->param(cla.primary_iv_index);
    Value* rem_limit = get_invariant_val(b, iv_type, cla.limit_val);
    Value* rem_cond = nullptr;
    switch (cla.cmp_opcode) {
        case Opcode::slt: rem_cond = b.build_slt(rem_iv, rem_limit); break;
        case Opcode::ult: rem_cond = b.build_ult(rem_iv, rem_limit); break;
        case Opcode::sle: rem_cond = b.build_sle(rem_iv, rem_limit); break;
        case Opcode::ule: rem_cond = b.build_ule(rem_iv, rem_limit); break;
        case Opcode::sgt: rem_cond = b.build_sgt(rem_iv, rem_limit); break;
        case Opcode::ugt: rem_cond = b.build_ugt(rem_iv, rem_limit); break;
        case Opcode::sge: rem_cond = b.build_sge(rem_iv, rem_limit); break;
        case Opcode::uge: rem_cond = b.build_uge(rem_iv, rem_limit); break;
        default: rem_cond = b.build_slt(rem_iv, rem_limit); break;
    }

    // Remainder Body
    b.position_at_end(rem_body);

    std::unordered_map<const Value*, Value*> rem_val_map;
    for (size_t i = 0; i < header->param_count(); ++i) {
        rem_val_map[header->param(i)] = rem_hdr->param(i);
    }

    std::vector<Value*> rem_next_latch_args(header->param_count());

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
            auto it_v = rem_val_map.find(op);
            if (it_v != rem_val_map.end()) {
                cloned->add_operand(it_v->second);
            } else {
                cloned->add_operand(op);
            }
        }

        if (inst->produces_value()) {
            Value* res = fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
            res->set_defining_instruction(cloned);
            cloned->set_result(res);
            rem_val_map[inst->result()] = res;
        }

        rem_body->append_instruction(cloned);
    }

    // Remainder latch jump arguments
    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
            Value* p = rem_hdr->param(i);
            Value* step_v = get_invariant_val(b, pa.type, pa.step_val);
            Value* next_v = pa.is_sub ? b.build_sub(p, step_v) : b.build_add(p, step_v);
            rem_next_latch_args[i] = next_v;
        } else if (pa.role == ParamRole::ReductionAcc || pa.role == ParamRole::SerialReductionAcc) {
            rem_next_latch_args[i] = rem_val_map[pa.update_inst->result()];
        } else {
            rem_next_latch_args[i] = rem_hdr->param(i);
        }
    }

    b.build_br(rem_hdr, rem_next_latch_args);

    // Remainder Header Terminator
    b.position_at_end(rem_hdr);
    Instruction* old_hdr_term = header->terminator();
    const BranchTarget& old_exit_target = cla.exit_on_false ? old_hdr_term->false_target() : old_hdr_term->true_target();

    if (!old_exit_target.args.empty()) {
        std::vector<Value*> final_exit_args;
        for (Value* arg : old_exit_target.args) {
            if (arg && arg->is_block_param() && arg->defining_block() == header) {
                final_exit_args.push_back(rem_hdr->param(arg->param_index()));
            } else {
                final_exit_args.push_back(arg);
            }
        }
        b.build_br_if(rem_cond, rem_body, {}, exit_bb, final_exit_args);
    } else {
        bool uses_header_param = false;
        for (BasicBlock* bb : fn.blocks()) {
            if (bb == unroll_hdr || bb == unroll_body || bb == unroll_exit ||
                bb == rem_hdr || bb == rem_body || loop.contains(bb)) {
                continue;
            }
            for (Instruction* inst : *bb) {
                for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                    Value* op = inst->operand(op_i);
                    if (op && op->is_block_param() && op->defining_block() == header) {
                        uses_header_param = true;
                        break;
                    }
                }
                if (uses_header_param) break;
            }
            if (uses_header_param) break;
        }

        if (uses_header_param) {
            std::unordered_map<Value*, Value*> exit_rewrite_map;
            for (size_t i = 0; i < header->param_count(); ++i) {
                Value* new_p = b.add_block_param(exit_bb, header->param(i)->type());
                exit_rewrite_map[header->param(i)] = new_p;
            }

            b.position_at_end(rem_hdr);
            b.build_br_if(rem_cond, rem_body, {}, exit_bb, rem_hdr->params());

            for (BasicBlock* bb : fn.blocks()) {
                if (bb == unroll_hdr || bb == unroll_body || bb == unroll_exit ||
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
                }
            }
        } else {
            b.build_br_if(rem_cond, rem_body, {}, exit_bb, {});
        }
    }

    // =========================================================================
    // 8. Remove Old Loop Blocks
    // =========================================================================
    fn.remove_block(header);
    if (body != header) {
        fn.remove_block(body);
    }
    if (cla.latch != header && cla.latch != body) {
        fn.remove_block(cla.latch);
    }

    fn.rebuild_cfg_predecessors();
    return true;
}

bool loop_unroll_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopUnrollOptions& options
) {
    LoopUnrollOptions effective_opts = options;
    if (fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation())) {
        effective_opts.enable_fp_reduction_jam = true;
    }
    LoopAnalysis loops(fn, dom);
    std::vector<LoopInfo*> post_order = loops.post_order_loops();

    bool any_changed = false;
    for (LoopInfo* loop : post_order) {
        if (!loop) continue;
        if (unroll_loop(fn, *loop, dom, effective_opts)) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            break; // restart loop analysis after transforming a loop
        }
    }
    return any_changed;
}

} // namespace brass
