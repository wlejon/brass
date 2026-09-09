#include "loop_vectorize_analysis.hpp"
#include <brass/mir/opcodes.hpp>
#include <algorithm>

namespace brass {

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

static bool is_supported_loop_arithmetic(Opcode op) {
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
            return true;
        default:
            return false;
    }
}

} // namespace

bool analyze_vectorizable_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    VectorizableLoopInfo& vli,
    const LoopVectorizeOptions& options
) {
    (void)dom;
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return false;

    // Do not re-vectorize already vectorized or unrolled loops
    if (header->name().find("_vec_hdr") != std::string_view::npos ||
        header->name().find("_rem_hdr") != std::string_view::npos ||
        header->name().find("_unroll_hdr") != std::string_view::npos) {
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

    // Check preheader branch target
    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
        ph_bt = &ph_term->branch_target();
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    // Check latch branch target
    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
        latch_bt = &latch_term->branch_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    // Header terminator must be br_if
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
    if (loop.blocks().size() > 2) return false; // Single body block + header (or header==body)

    // Check cmp opcode
    Opcode cmp_op = cmp_inst->opcode();
    if (cmp_op != Opcode::slt && cmp_op != Opcode::ult &&
        cmp_op != Opcode::sle && cmp_op != Opcode::ule) {
        return false;
    }

    Value* cmp_lhs = cmp_inst->operand(0);
    Value* cmp_rhs = cmp_inst->operand(1);

    // Identify Primary IV
    bool found_primary_iv = false;
    size_t primary_iv_idx = 0;
    Value* limit_val = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
            Value* latch_next = latch_bt->args[i];
            if (latch_next && latch_next->is_instruction()) {
                Instruction* def = latch_next->defining_instruction();
                if (def && def->opcode() == Opcode::add) {
                    Value* step = (def->operand(0) == param) ? def->operand(1) : ((def->operand(1) == param) ? def->operand(0) : nullptr);
                    int64_t step_c = 0;
                    if (step && get_const_int(step, step_c) && step_c == 1) {
                        found_primary_iv = true;
                        primary_iv_idx = i;
                        limit_val = cmp_rhs;
                        break;
                    }
                }
            }
        }
    }

    if (!found_primary_iv || !limit_val) return false;

    Value* primary_iv_param = header->param(primary_iv_idx);
    vli.primary_iv_index = primary_iv_idx;
    vli.iv_type = primary_iv_param->type();
    vli.init_iv = ph_bt->args[primary_iv_idx];
    vli.limit_val = limit_val;
    vli.cmp_opcode = cmp_op;
    vli.header = header;
    vli.body = body_bb;
    vli.latch = latch;
    vli.preheader = preheader;
    vli.exit_bb = exit_bb;
    vli.exit_on_false = exit_on_false;

    // Check for reductions in header parameters
    bool allow_fp = options.allow_fp_reassociation || fn.allow_fp_reassociation() ||
                    (fn.parent() && fn.parent()->allow_fp_reassociation());

    bool found_reduction = false;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == primary_iv_idx) continue;
        Value* param = header->param(i);
        Value* latch_next = latch_bt->args[i];
        if (latch_next == param) {
            // Invariant parameter
            continue;
        }

        if (latch_next && latch_next->is_instruction()) {
            Instruction* def = latch_next->defining_instruction();
            if (def && loop.contains(def->parent())) {
                if (def->opcode() == Opcode::add) {
                    Value* op0 = def->operand(0);
                    Value* op1 = def->operand(1);
                    if (op0 == param || op1 == param) {
                        Type t = param->type();
                        if (t.is_float() && !allow_fp) {
                            return false; // FP reduction refused without opt-in FP reassociation
                        }
                        if (found_reduction) {
                            return false; // Multiple reductions not supported currently
                        }
                        found_reduction = true;
                        vli.has_reduction = true;
                        vli.reduction_param_index = i;
                        vli.reduction_type = t;
                        vli.reduction_update_inst = def;
                        vli.reduction_init_val = ph_bt->args[i];
                        continue;
                    }
                }
            }
        }
        return false; // Unknown non-invariant parameter
    }

    // Inspect instructions in loop body
    std::vector<VectorizableMemOp> mem_ops;
    Type determined_elem_type = Type::void_type();

    for (Instruction* inst = body_bb->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator()) continue;
        Opcode op = inst->opcode();

        if (op == Opcode::call || op == Opcode::call_indirect || op == Opcode::patchable_call ||
            op == Opcode::safepoint || op == Opcode::guard || op == Opcode::resume_point) {
            return false;
        }

        if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
            bool is_store = (op == Opcode::store_indexed);
            Value* base = inst->operand(0);
            Value* index = inst->operand(1);
            if (!loop.is_loop_invariant(base)) return false;
            if (index != primary_iv_param) return false;

            Type mem_t = inst->memory_type();
            uint8_t scale = inst->scale();
            if (scale != mem_t.size_in_bytes()) return false;

            if (determined_elem_type.is_void()) {
                determined_elem_type = mem_t;
            } else if (determined_elem_type != mem_t) {
                return false; // Mixed types in loop
            }

            VectorizableMemOp mop;
            mop.inst = inst;
            mop.is_store = is_store;
            mop.base = base;
            mop.index = index;
            mop.scale = scale;
            mop.offset = inst->offset();
            mop.elem_type = mem_t;
            mem_ops.push_back(mop);
            continue;
        }

        if (is_supported_loop_arithmetic(op)) {
            continue;
        }

        if (op == Opcode::iconst_i32 || op == Opcode::iconst_i64 || op == Opcode::fconst_f64) {
            continue;
        }

        return false; // Unsupported opcode in loop body
    }

    if (vli.has_reduction) {
        determined_elem_type = vli.reduction_type;
    }

    if (determined_elem_type.is_void()) {
        return false;
    }

    // Configure vector width & vector type
    if (determined_elem_type == Type::f32()) {
        if (!options.enable_f32x4) return false;
        vli.vector_width = 4;
        vli.elem_type = Type::f32();
        vli.vec_type = Type::f32x4();
    } else if (determined_elem_type == Type::i32()) {
        if (!options.enable_i32x4) return false;
        vli.vector_width = 4;
        vli.elem_type = Type::i32();
        vli.vec_type = Type::i32x4();
    } else if (determined_elem_type == Type::f64()) {
        if (!options.enable_f64x2) return false;
        vli.vector_width = 2;
        vli.elem_type = Type::f64();
        vli.vec_type = Type::f64x2();
    } else {
        return false;
    }

    vli.mem_ops = std::move(mem_ops);
    vli.is_vectorizable = true;
    return true;
}

} // namespace brass