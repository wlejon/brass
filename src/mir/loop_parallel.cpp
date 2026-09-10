#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/builder.hpp>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace brass {

std::string ParallelLoopStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Parallel Loop Statistics ===\n"
       << "  Loops analyzed: " << loops_analyzed << "\n"
       << "  DOALL loops found: " << doall_loops_found << "\n"
       << "  Reduction loops found: " << reduction_loops_found << "\n"
       << "  Loops transformed: " << parallel_loops_transformed << "\n"
       << "  Rejected (carried dependence): " << loops_rejected_carried_dependence << "\n"
       << "  Rejected (uncontrolled effects): " << loops_rejected_uncontrolled_effects << "\n"
       << "  Rejected (below cost threshold): " << loops_rejected_cost << "\n";
    return ss.str();
}

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

bool is_pure_math_call(std::string_view callee) noexcept {
    return callee == "sqrt" || callee == "fabs" || callee == "floor" ||
           callee == "ceil" || callee == "sin" || callee == "cos";
}

bool is_uncontrolled_side_effect(const Instruction* inst) noexcept {
    if (!inst) return false;
    Opcode op = inst->opcode();
    if (op == Opcode::throw_ || op == Opcode::invoke ||
        op == Opcode::landing_pad || op == Opcode::resume ||
        op == Opcode::osr_entry) {
        return true;
    }
    if (op == Opcode::call) {
        return !is_pure_math_call(inst->symbol());
    }
    if (op == Opcode::call_indirect || op == Opcode::patchable_call) {
        return true;
    }
    return false;
}

static bool decompose_affine_expr(
    const Value* val,
    const Value* iv_val,
    const Value*& out_base,
    int64_t& out_stride,
    int64_t& out_offset
) {
    if (!val) return false;
    if (val == iv_val) {
        out_stride += 1;
        return true;
    }
    int64_t c = 0;
    if (get_const_int(val, c)) {
        out_offset += c;
        return true;
    }
    if (!val->is_instruction()) {
        if (!out_base) {
            out_base = val;
            return true;
        }
        return false;
    }
    const Instruction* inst = val->defining_instruction();
    if (!inst) return false;

    if (inst->opcode() == Opcode::add) {
        return decompose_affine_expr(inst->operand(0), iv_val, out_base, out_stride, out_offset) &&
               decompose_affine_expr(inst->operand(1), iv_val, out_base, out_stride, out_offset);
    }
    if (inst->opcode() == Opcode::sub) {
        int64_t rhs_stride = 0, rhs_offset = 0;
        const Value* rhs_base = nullptr;
        if (!decompose_affine_expr(inst->operand(0), iv_val, out_base, out_stride, out_offset)) return false;
        if (!decompose_affine_expr(inst->operand(1), iv_val, rhs_base, rhs_stride, rhs_offset)) return false;
        if (rhs_base != nullptr) return false;
        out_stride -= rhs_stride;
        out_offset -= rhs_offset;
        return true;
    }
    if (inst->opcode() == Opcode::mul) {
        int64_t k = 0;
        if (inst->operand(0) == iv_val && get_const_int(inst->operand(1), k)) {
            out_stride += k;
            return true;
        }
        if (inst->operand(1) == iv_val && get_const_int(inst->operand(0), k)) {
            out_stride += k;
            return true;
        }
        return false;
    }
    if (inst->opcode() == Opcode::shl) {
        int64_t shift = 0;
        if (inst->operand(0) == iv_val && get_const_int(inst->operand(1), shift)) {
            out_stride += (1LL << shift);
            return true;
        }
        return false;
    }
    if (!out_base) {
        out_base = val;
        return true;
    }
    return false;
}

} // namespace

bool parse_subscript_expression(
    const Instruction* inst,
    const Value* iv_val,
    SubscriptExpr& out_expr
) {
    if (!inst || !iv_val) return false;
    out_expr.is_valid = false;
    Opcode op = inst->opcode();

    if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
        out_expr.base = inst->operand(0);
        out_expr.scale = inst->scale() > 0 ? inst->scale() : 1;
        out_expr.offset = inst->offset();
        out_expr.iv = const_cast<Value*>(iv_val);

        const Value* idx = inst->operand(1);
        const Value* dummy_base = nullptr;
        int64_t idx_stride = 0;
        int64_t idx_offset = 0;

        if (decompose_affine_expr(idx, iv_val, dummy_base, idx_stride, idx_offset) && dummy_base == nullptr) {
            out_expr.stride = idx_stride * static_cast<int64_t>(out_expr.scale);
            out_expr.offset += idx_offset * static_cast<int64_t>(out_expr.scale);
            out_expr.is_valid = true;
            return true;
        }
    } else if (op == Opcode::load || op == Opcode::store) {
        out_expr.scale = 1;
        out_expr.offset = inst->offset();
        out_expr.iv = const_cast<Value*>(iv_val);

        const Value* ptr = inst->operand(0);
        int64_t ptr_stride = 0;
        int64_t ptr_offset = 0;
        const Value* base = nullptr;

        if (decompose_affine_expr(ptr, iv_val, base, ptr_stride, ptr_offset)) {
            out_expr.base = const_cast<Value*>(base);
            out_expr.stride = ptr_stride;
            out_expr.offset += ptr_offset;
            out_expr.is_valid = true;
            return true;
        }
    }

    return false;
}

ParallelDependence check_subscript_dependence(
    const ParallelMemAccess& a1,
    const ParallelMemAccess& a2
) {
    ParallelDependence dep;
    dep.src = a1.inst;
    dep.dst = a2.inst;

    // Read-After-Read is never a data conflict
    if (!a1.is_store && !a2.is_store) {
        dep.kind = DependenceKind::None;
        dep.is_loop_carried = false;
        return dep;
    }

    if (a1.is_store && !a2.is_store) {
        dep.kind = DependenceKind::RAW;
    } else if (!a1.is_store && a2.is_store) {
        dep.kind = DependenceKind::WAR;
    } else {
        dep.kind = DependenceKind::WAW;
    }

    if (!a1.expr.is_valid || !a2.expr.is_valid) {
        dep.dir = DependenceDir::Any;
        dep.is_loop_carried = true;
        dep.has_distance = false;
        return dep;
    }

    // If distinct base pointers, assume disjoint memory regions
    if (a1.expr.base != a2.expr.base) {
        if (a1.expr.base != nullptr && a2.expr.base != nullptr) {
            dep.kind = DependenceKind::None;
            dep.is_loop_carried = false;
            return dep;
        }
    }

    int64_t byte_stride1 = a1.expr.stride;
    int64_t byte_stride2 = a2.expr.stride;

    if (byte_stride1 == byte_stride2 && byte_stride1 != 0) {
        int64_t diff = a2.expr.offset - a1.expr.offset;
        if (diff % byte_stride1 == 0) {
            int64_t dist = diff / byte_stride1;
            dep.distance = dist;
            dep.has_distance = true;

            if (dist == 0) {
                dep.dir = DependenceDir::Equal;
                dep.is_loop_carried = false;
            } else if (dist > 0) {
                dep.dir = DependenceDir::Forward;
                dep.is_loop_carried = true;
            } else {
                dep.dir = DependenceDir::Backward;
                dep.is_loop_carried = true;
            }
        } else {
            // Strides match but offsets are incongruent -> disjoint access streams
            dep.kind = DependenceKind::None;
            dep.is_loop_carried = false;
        }
    } else {
        dep.dir = DependenceDir::Any;
        dep.is_loop_carried = true;
        dep.has_distance = false;
    }

    return dep;
}

bool analyze_parallel_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    ParallelLoopInfo& pli,
    const ParallelLoopOptions& options
) {
    (void)dom;
    pli = ParallelLoopInfo();
    pli.loop = &loop;
    pli.kind = LoopParallelKind::Sequential;

    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) {
        pli.rejection_reason = "loop without single header/latch";
        return false;
    }

    // Do not parallelize already outlined or synthetic kernels
    if (header->name().find("par_k_") != std::string_view::npos ||
        header->name().find("_par_") != std::string_view::npos) {
        pli.rejection_reason = "already parallelized";
        return false;
    }

    BasicBlock* latch = loop.latches()[0];
    if (!latch) {
        pli.rejection_reason = "missing latch";
        return false;
    }

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) {
        pli.rejection_reason = "could not ensure preheader";
        return false;
    }

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || !latch_term || !hdr_term) {
        pli.rejection_reason = "missing terminators";
        return false;
    }

    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
        ph_bt = &ph_term->branch_target();
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) {
        pli.rejection_reason = "invalid preheader branch";
        return false;
    }

    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
        latch_bt = &latch_term->branch_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) {
        pli.rejection_reason = "invalid latch branch";
        return false;
    }

    if (hdr_term->opcode() != Opcode::br_if) {
        pli.rejection_reason = "header terminator not br_if";
        return false;
    }

    Value* cond_val = hdr_term->operand(0);
    if (!cond_val || !cond_val->is_instruction()) {
        pli.rejection_reason = "header condition not instruction";
        return false;
    }
    Instruction* cmp_inst = cond_val->defining_instruction();
    if (!cmp_inst || !is_comparison(cmp_inst->opcode()) || cmp_inst->parent() != header) {
        pli.rejection_reason = "invalid header comparison";
        return false;
    }

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
        pli.rejection_reason = "multi-exit or complex CFG loop";
        return false;
    }

    if (!body_bb || !exit_bb) {
        pli.rejection_reason = "missing body or exit block";
        return false;
    }

    Opcode cmp_op = cmp_inst->opcode();
    if (cmp_op != Opcode::slt && cmp_op != Opcode::ult &&
        cmp_op != Opcode::sle && cmp_op != Opcode::ule) {
        pli.rejection_reason = "unsupported loop condition comparison";
        return false;
    }

    Value* cmp_lhs = cmp_inst->operand(0);
    Value* cmp_rhs = cmp_inst->operand(1);

    // Identify primary induction variable
    bool found_primary_iv = false;
    size_t primary_iv_idx = 0;
    int64_t step_val = 1;
    Value* limit_val = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
            Value* latch_next = latch_bt->args[i];
            if (latch_next && latch_next->is_instruction()) {
                Instruction* def = latch_next->defining_instruction();
                if (def && def->opcode() == Opcode::add) {
                    Value* step_op = (def->operand(0) == param) ? def->operand(1) : ((def->operand(1) == param) ? def->operand(0) : nullptr);
                    int64_t s = 0;
                    if (step_op && get_const_int(step_op, s) && s > 0) {
                        found_primary_iv = true;
                        primary_iv_idx = i;
                        step_val = s;
                        limit_val = cmp_rhs;
                        break;
                    }
                }
            }
        }
    }

    if (!found_primary_iv || !limit_val) {
        pli.rejection_reason = "could not identify countable induction variable";
        return false;
    }

    pli.header = header;
    pli.preheader = preheader;
    pli.body = body_bb;
    pli.latch = latch;
    pli.exit_bb = exit_bb;
    pli.exit_on_false = exit_on_false;
    pli.iv_param_index = primary_iv_idx;
    pli.iv_param = header->param(primary_iv_idx);
    pli.iv_type = pli.iv_param->type();
    pli.init_iv = ph_bt->args[primary_iv_idx];
    pli.limit_val = limit_val;
    pli.step = step_val;
    pli.cmp_opcode = cmp_op;

    int64_t init_c = 0, limit_c = 0;
    if (get_const_int(pli.init_iv, init_c) && get_const_int(limit_val, limit_c)) {
        if (limit_c >= init_c) {
            pli.has_const_trip_count = true;
            pli.const_trip_count = static_cast<uint64_t>((limit_c - init_c + step_val - 1) / step_val);
        }
    }

    // Check for side effects & count instructions
    pli.instruction_count = 0;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            pli.instruction_count++;
            if (is_uncontrolled_side_effect(inst)) {
                pli.rejection_reason = "loop contains uncontrolled side effects or exceptions";
                if (options.stats) options.stats->loops_rejected_uncontrolled_effects++;
                return false;
            }
        }
    }

    // Check for reductions
    bool allow_fp = options.allow_fp_reassociation || fn.allow_fp_reassociation() ||
                    (fn.parent() && fn.parent()->allow_fp_reassociation());

    bool found_reduction = false;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == primary_iv_idx) continue;
        Value* param = header->param(i);
        Value* latch_next = latch_bt->args[i];
        if (latch_next == param) {
            continue; // Invariant loop parameter
        }

        if (latch_next && latch_next->is_instruction()) {
            Instruction* def = latch_next->defining_instruction();
            if (def && loop.contains(def->parent())) {
                runtime::ReductionKind rk = runtime::ReductionKind::None;
                Type t = param->type();

                if (def->opcode() == Opcode::add) {
                    rk = t.is_float() ? runtime::ReductionKind::SumF64 : runtime::ReductionKind::SumI64;
                } else if (def->opcode() == Opcode::mul) {
                    rk = t.is_float() ? runtime::ReductionKind::ProdF64 : runtime::ReductionKind::ProdI64;
                } else if (def->opcode() == Opcode::vmin) {
                    rk = t.is_float() ? runtime::ReductionKind::MinF64 : runtime::ReductionKind::MinI64;
                } else if (def->opcode() == Opcode::vmax) {
                    rk = t.is_float() ? runtime::ReductionKind::MaxF64 : runtime::ReductionKind::MaxI64;
                }

                if (rk != runtime::ReductionKind::None) {
                    if (t.is_float() && !allow_fp) {
                        pli.rejection_reason = "FP reduction requires opt-in FP reassociation";
                        return false;
                    }
                    if (found_reduction) {
                        pli.rejection_reason = "multiple reductions not supported in single parallel loop";
                        return false;
                    }

                    found_reduction = true;
                    pli.has_reduction = true;
                    pli.reduction_kind = rk;
                    pli.reduction_param_index = i;
                    pli.reduction_param = param;
                    pli.reduction_init_val = ph_bt->args[i];
                    pli.reduction_op_inst = def;
                    pli.reduction_type = t;
                }
            }
        }
    }

    // Memory access & dependence analysis
    pli.accesses.clear();
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            Opcode op = inst->opcode();
            if (op == Opcode::load_indexed || op == Opcode::store_indexed ||
                op == Opcode::load || op == Opcode::store) {
                ParallelMemAccess acc;
                acc.inst = inst;
                acc.is_store = (op == Opcode::store_indexed || op == Opcode::store);
                acc.elem_type = inst->memory_type();
                if (parse_subscript_expression(inst, pli.iv_param, acc.expr)) {
                    pli.accesses.push_back(acc);
                } else {
                    pli.rejection_reason = "unmodeled memory access subscript";
                    return false;
                }
            }
        }
    }

    pli.dependences.clear();
    for (size_t i = 0; i < pli.accesses.size(); ++i) {
        for (size_t j = i + 1; j < pli.accesses.size(); ++j) {
            ParallelDependence dep = check_subscript_dependence(pli.accesses[i], pli.accesses[j]);
            if (dep.kind != DependenceKind::None) {
                pli.dependences.push_back(dep);
                if (dep.is_loop_carried) {
                    pli.rejection_reason = "loop-carried memory dependence detected";
                    if (options.stats) options.stats->loops_rejected_carried_dependence++;
                    return false;
                }
            }
        }
    }

    if (found_reduction) {
        pli.kind = LoopParallelKind::Reduction;
        if (options.stats) options.stats->reduction_loops_found++;
    } else {
        pli.kind = LoopParallelKind::DOALL;
        if (options.stats) options.stats->doall_loops_found++;
    }

    return true;
}

} // namespace brass