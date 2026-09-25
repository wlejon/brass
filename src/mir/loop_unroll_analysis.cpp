#include "loop_unroll_analysis.hpp"
#include <brass/mir/verifier.hpp>
#include <algorithm>

namespace brass {

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

static Opcode swap_comparison_operands(Opcode op) noexcept {
    switch (op) {
        case Opcode::slt: return Opcode::sgt;
        case Opcode::sle: return Opcode::sge;
        case Opcode::sgt: return Opcode::slt;
        case Opcode::sge: return Opcode::sle;
        case Opcode::ult: return Opcode::ugt;
        case Opcode::ule: return Opcode::uge;
        case Opcode::ugt: return Opcode::ult;
        case Opcode::uge: return Opcode::ule;
        case Opcode::eq:  return Opcode::eq;
        case Opcode::ne:  return Opcode::ne;
        default: return op;
    }
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
        // MIR integer arithmetic wraps.
        return make_smart_const_int(b, t, static_cast<int64_t>(static_cast<uint64_t>(c) * static_cast<uint64_t>(mul_factor)));
    }
    Value* factor_val = make_smart_const_int(b, t, mul_factor);
    return b.build_mul(val, factor_val);
}

Value* make_smart_add(Builder& b, Type t, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) {
        return make_smart_const_int(b, t, static_cast<int64_t>(static_cast<uint64_t>(c0) + static_cast<uint64_t>(c1)));
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

// Reads of `v` by instructions of `loop`, not counting arguments on the
// header's exit edge (the exit sees the final value, which the unroller
// recombines).
static size_t in_loop_uses(const Value* v, const LoopInfo& loop, const BranchTarget* header_exit) {
    size_t n = 0;
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == v) ++n;
            }
            auto count_args = [&](const BranchTarget& t) {
                if (&t == header_exit) return;
                n += static_cast<size_t>(std::count(t.args.begin(), t.args.end(), v));
            };
            switch (inst->opcode()) {
                case Opcode::br: count_args(inst->branch_target()); break;
                case Opcode::br_if: count_args(inst->true_target()); count_args(inst->false_target()); break;
                case Opcode::invoke: count_args(inst->normal_target()); count_args(inst->unwind_target()); break;
                case Opcode::switch_:
                    count_args(inst->default_target());
                    for (const SwitchCase& c : inst->switch_cases()) count_args(c.target);
                    break;
                default: break;
            }
        }
    }
    return n;
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

    if (loop.blocks().size() > 2) {
        return false;
    }

    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->is_call() || (inst->has_side_effects() && inst != latch_term && inst != hdr_term)) {
                return false;
            }
        }
    }

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

        if (!found_primary_iv && (param == cmp_lhs || param == cmp_rhs)) {
            if (pa.role == ParamRole::DerivedIV && pa.step_val != nullptr) {
                pa.role = ParamRole::BasicIV;
                cla.primary_iv_index = i;
                found_primary_iv = true;

                if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
                    cla.cmp_opcode = cmp_inst->opcode();
                    cla.limit_val = cmp_rhs;
                } else if (param == cmp_rhs && loop.is_loop_invariant(cmp_lhs)) {
                    cla.cmp_opcode = swap_comparison_operands(cmp_inst->opcode());
                    cla.limit_val = cmp_lhs;
                }
            }
        }
    }

    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.latch_next_val != pa.header_param && pa.role == ParamRole::Invariant) {
            return false;
        }
    }

    if (!found_primary_iv || !cla.limit_val) return false;

    // An accumulator is only rewritten through its own update: jamming splits
    // it into partial sums, and the serial chain is emitted apart from the
    // rest of the body, so any other reader (another reduction's addend, a
    // comparison, a store) would see the wrong value. The accumulator is read
    // once (by its update) and the update once (by the latch).
    const BranchTarget* header_exit = exit_on_false ? &hdr_term->false_target() : &hdr_term->true_target();
    for (size_t red_i : cla.reduction_indices) {
        const ParamAnalysis& pa = cla.params[red_i];
        if (in_loop_uses(pa.header_param, loop, header_exit) != 1 ||
            in_loop_uses(pa.update_inst->result(), loop, header_exit) != 1) {
            return false;
        }
    }

    for (size_t i = 0; i < header->param_count(); ++i) {
        const ParamAnalysis& pa = cla.params[i];
        if (pa.role == ParamRole::BasicIV || pa.role == ParamRole::DerivedIV) {
            if (pa.type != Type::i32() && pa.type != Type::i64()) {
                return false;
            }
        }
    }

    if (cla.cmp_opcode != Opcode::slt && cla.cmp_opcode != Opcode::ult &&
        cla.cmp_opcode != Opcode::sle && cla.cmp_opcode != Opcode::ule &&
        cla.cmp_opcode != Opcode::sgt && cla.cmp_opcode != Opcode::ugt &&
        cla.cmp_opcode != Opcode::sge && cla.cmp_opcode != Opcode::uge) {
        return false;
    }

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

} // namespace brass
