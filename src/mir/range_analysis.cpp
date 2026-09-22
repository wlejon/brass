#include <brass/mir/range_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <iostream>
#include <sstream>
#include <cmath>
#include <bit>
#include <optional>

namespace brass {

namespace {

#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 int128_t;
#endif

inline int64_t sat_add(int64_t a, int64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) + b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
#else
    if (b > 0 && a > INT64_MAX - b) return INT64_MAX;
    if (b < 0 && a < INT64_MIN - b) return INT64_MIN;
    return a + b;
#endif
}

inline int64_t sat_sub(int64_t a, int64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) - b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
#else
    if (b < 0 && a > INT64_MAX + b) return INT64_MAX;
    if (b > 0 && a < INT64_MIN + b) return INT64_MIN;
    return a - b;
#endif
}

inline int64_t sat_mul(int64_t a, int64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) * b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
#else
    if (a == 0 || b == 0) return 0;
    if (a == -1 && b == INT64_MIN) return INT64_MAX;
    if (b == -1 && a == INT64_MIN) return INT64_MAX;
    if (a > 0 && b > 0 && a > INT64_MAX / b) return INT64_MAX;
    if (a > 0 && b < 0 && b < INT64_MIN / a) return INT64_MIN;
    if (a < 0 && b > 0 && a < INT64_MIN / b) return INT64_MIN;
    if (a < 0 && b < 0 && a < INT64_MAX / b) return INT64_MAX;
    return a * b;
#endif
}

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::patchable_const_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64 || def->opcode() == Opcode::patchable_const_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

// Ranges hold every value in its canonical int64 form (32-bit values
// sign-extended). Interval arithmetic runs at 64 bits, so a 32-bit result that
// leaves the 32-bit range has wrapped at run time: all that is then known is
// that it is some 32-bit value. Clipping to the 32-bit range instead would
// claim values that never occur.
ValueRange fit_to_type(const ValueRange& r, Type t) noexcept {
    if (r.is_empty()) return r;
    if (t == Type::i32()) {
        const ValueRange i32_range = ValueRange::range(INT32_MIN, INT32_MAX);
        return r.is_subrange_of(i32_range) ? r : i32_range;
    }
    if (t == Type::i64()) return r;
    // Other types (pointers, narrow integers whose representation passes do
    // not agree on, floats) carry no integer facts.
    return ValueRange::full();
}

ValueRange full_range_of(Type t) noexcept {
    return t == Type::i32() ? ValueRange::range(INT32_MIN, INT32_MAX) : ValueRange::full();
}

bool is_int_value(const Value* v) noexcept {
    return v && (v->type() == Type::i32() || v->type() == Type::i64());
}

Opcode invert_comparison_opcode(Opcode op) noexcept {
    switch (op) {
        case Opcode::slt: return Opcode::sge;
        case Opcode::sle: return Opcode::sgt;
        case Opcode::sgt: return Opcode::sle;
        case Opcode::sge: return Opcode::slt;
        case Opcode::ult: return Opcode::uge;
        case Opcode::ule: return Opcode::ugt;
        case Opcode::ugt: return Opcode::ule;
        case Opcode::uge: return Opcode::ult;
        case Opcode::eq:  return Opcode::ne;
        case Opcode::ne:  return Opcode::eq;
        default: return op;
    }
}

ValueRange evaluate_comparison(Opcode op, const ValueRange& r0, const ValueRange& r1) noexcept {
    if (r0.is_empty() || r1.is_empty()) return ValueRange::empty();
    if (op == Opcode::eq) {
        if (r0.is_constant() && r1.is_constant()) {
            return ValueRange::constant(r0.min_val == r1.min_val ? 1 : 0);
        }
        ValueRange inter = r0;
        inter.intersect_with(r1);
        if (inter.is_empty()) return ValueRange::constant(0);
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::ne) {
        if (r0.is_constant() && r1.is_constant()) {
            return ValueRange::constant(r0.min_val != r1.min_val ? 1 : 0);
        }
        ValueRange inter = r0;
        inter.intersect_with(r1);
        if (inter.is_empty()) return ValueRange::constant(1);
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::slt) {
        if (r0.max_val < r1.min_val) return ValueRange::constant(1);
        if (r0.min_val >= r1.max_val) return ValueRange::constant(0);
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::sle) {
        if (r0.max_val <= r1.min_val) return ValueRange::constant(1);
        if (r0.min_val > r1.max_val) return ValueRange::constant(0);
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::sgt) {
        return evaluate_comparison(Opcode::slt, r1, r0);
    }
    if (op == Opcode::sge) {
        return evaluate_comparison(Opcode::sle, r1, r0);
    }
    if (op == Opcode::ult) {
        if (r0.is_non_negative() && r1.is_non_negative()) {
            return evaluate_comparison(Opcode::slt, r0, r1);
        }
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::ule) {
        if (r0.is_non_negative() && r1.is_non_negative()) {
            return evaluate_comparison(Opcode::sle, r0, r1);
        }
        return ValueRange::range(0, 1);
    }
    if (op == Opcode::ugt) {
        return evaluate_comparison(Opcode::ult, r1, r0);
    }
    if (op == Opcode::uge) {
        return evaluate_comparison(Opcode::ule, r1, r0);
    }
    return ValueRange::range(0, 1);
}

} // namespace

RangeAnalysis::RangeAnalysis(Function& fn) {
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    run_analysis(fn, dom, loops);
}

RangeAnalysis::RangeAnalysis(Function& fn, const DominatorTree& dom, const LoopAnalysis& loops) {
    run_analysis(fn, dom, loops);
}

ValueRange RangeAnalysis::get_range(const Value* v) const {
    if (!v) return ValueRange::full();
    int64_t c = 0;
    if (get_const_int(v, c)) {
        return ValueRange::constant(c);
    }
    auto it = global_ranges_.find(v);
    if (it != global_ranges_.end()) {
        return it->second;
    }
    if (v->type() == Type::i32()) {
        return ValueRange::range(INT32_MIN, INT32_MAX);
    }
    return ValueRange::full();
}

ValueRange RangeAnalysis::get_range_at(const Value* v, const BasicBlock* bb) const {
    if (!v) return ValueRange::full();
    int64_t c = 0;
    if (get_const_int(v, c)) {
        return ValueRange::constant(c);
    }
    if (bb) {
        auto d_it = block_deltas_.find(bb);
        if (d_it != block_deltas_.end()) {
            // A block pass 3 reached: the in-scope range is the nearest write
            // on the dominator chain, else the snapshot pass 3 started from.
            for (const BlockDelta* delta = &d_it->second;;) {
                const auto& changes = delta->changes;
                for (auto c = changes.rbegin(); c != changes.rend(); ++c) {
                    if (c->first == v) return c->second;
                }
                if (!delta->idom) break;
                auto up = block_deltas_.find(delta->idom);
                if (up == block_deltas_.end()) break;
                delta = &up->second;
            }
            auto s_it = pass3_initial_ranges_.find(v);
            if (s_it != pass3_initial_ranges_.end()) return s_it->second;
            return get_range(v);
        }
        auto b_it = block_ranges_.find(bb);
        if (b_it != block_ranges_.end()) {
            auto v_it = b_it->second.find(v);
            if (v_it != b_it->second.end()) {
                return v_it->second;
            }
        }
    }
    return get_range(v);
}

ValueRange RangeAnalysis::get_context_range(
    const Value* v,
    const std::unordered_map<const Value*, ValueRange>& context_ranges
) const {
    if (!v) return ValueRange::full();
    int64_t c = 0;
    if (get_const_int(v, c)) {
        return ValueRange::constant(c);
    }
    auto it = context_ranges.find(v);
    if (it != context_ranges.end()) {
        return it->second;
    }
    return get_range(v);
}

ValueRange RangeAnalysis::evaluate_instruction(
    const Instruction* inst,
    const std::unordered_map<const Value*, ValueRange>& context_ranges
) const {
    if (!inst) return ValueRange::full();
    Opcode op = inst->opcode();
    switch (op) {
        case Opcode::iconst_i32:
        case Opcode::patchable_const_i32:
            return ValueRange::constant(inst->imm_i32());
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i64:
            return ValueRange::constant(inst->imm_i64());

        case Opcode::add: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::add(r0, r1);
        }
        case Opcode::sub: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::sub(r0, r1);
        }
        case Opcode::mul: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::mul(r0, r1);
        }
        case Opcode::and_: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::and_(r0, r1);
        }
        case Opcode::or_: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::or_(r0, r1);
        }
        case Opcode::xor_: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::xor_(r0, r1);
        }
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            // The hardware uses the shift amount modulo the operand width.
            if (!r1.is_constant()) return full_range_of(inst->type());
            const int64_t width_mask = inst->type() == Type::i32() ? 31 : 63;
            r1 = ValueRange::constant(r1.min_val & width_mask);
            if (op == Opcode::shl) return ValueRange::shl(r0, r1);
            if (op == Opcode::lshr) return ValueRange::lshr(r0, r1);
            return ValueRange::ashr(r0, r1);
        }
        case Opcode::zext_i64: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            return ValueRange::zext(r0);
        }
        case Opcode::sext_i64: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            return ValueRange::sext(r0);
        }
        case Opcode::trunc_i32: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            return ValueRange::trunc(r0);
        }
        case Opcode::select: {
            ValueRange rc = get_context_range(inst->operand(0), context_ranges);
            ValueRange rt = get_context_range(inst->operand(1), context_ranges);
            ValueRange rf = get_context_range(inst->operand(2), context_ranges);
            return ValueRange::select(rc, rt, rf);
        }
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge: {
            // Only integer operands have ranges; float and pointer
            // comparisons are merely known to yield 0 or 1.
            if (!is_int_value(inst->operand(0)) || !is_int_value(inst->operand(1))) {
                return ValueRange::range(0, 1);
            }
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return evaluate_comparison(op, r0, r1);
        }
        default:
            return full_range_of(inst->type());
    }
}

void RangeAnalysis::apply_branch_condition(
    const Value* cond,
    bool is_true_edge,
    std::unordered_map<const Value*, ValueRange>& ranges,
    std::vector<std::pair<const Value*, std::optional<ValueRange>>>& rollback
) {
    if (!cond || !cond->is_instruction()) return;
    const Instruction* cdef = cond->defining_instruction();
    if (!cdef) return;

    if (cdef->opcode() == Opcode::and_ && is_true_edge) {
        apply_branch_condition(cdef->operand(0), true, ranges, rollback);
        apply_branch_condition(cdef->operand(1), true, ranges, rollback);
        return;
    }
    if (cdef->opcode() == Opcode::or_ && !is_true_edge) {
        apply_branch_condition(cdef->operand(0), false, ranges, rollback);
        apply_branch_condition(cdef->operand(1), false, ranges, rollback);
        return;
    }

    if (!is_comparison(cdef->opcode())) return;

    Opcode op = is_true_edge ? cdef->opcode() : invert_comparison_opcode(cdef->opcode());
    auto set_val_range = [&](const Value* val, const ValueRange& r) {
        int64_t c = 0;
        if (get_const_int(val, c)) return;
        auto it = ranges.find(val);
        rollback.emplace_back(val, it != ranges.end() ? std::optional<ValueRange>(it->second) : std::nullopt);
        ranges[val] = r;
    };

    // A comparison yields 0 or 1, so the edge taken fixes its value.
    set_val_range(cond, ValueRange::constant(is_true_edge ? 1 : 0));

    const Value* lhs = cdef->operand(0);
    const Value* rhs = cdef->operand(1);
    if (!is_int_value(lhs) || !is_int_value(rhs)) return;

    ValueRange r_lhs = get_context_range(lhs, ranges);
    ValueRange r_rhs = get_context_range(rhs, ranges);

    switch (op) {
        case Opcode::slt: {
            r_lhs.intersect_with(ValueRange::range(INT64_MIN, sat_sub(r_rhs.max_val, 1)));
            r_rhs.intersect_with(ValueRange::range(sat_add(r_lhs.min_val, 1), INT64_MAX));
            set_val_range(lhs, r_lhs);
            set_val_range(rhs, r_rhs);
            break;
        }
        case Opcode::sle: {
            r_lhs.intersect_with(ValueRange::range(INT64_MIN, r_rhs.max_val));
            r_rhs.intersect_with(ValueRange::range(r_lhs.min_val, INT64_MAX));
            set_val_range(lhs, r_lhs);
            set_val_range(rhs, r_rhs);
            break;
        }
        case Opcode::sgt: {
            r_rhs.intersect_with(ValueRange::range(INT64_MIN, sat_sub(r_lhs.max_val, 1)));
            r_lhs.intersect_with(ValueRange::range(sat_add(r_rhs.min_val, 1), INT64_MAX));
            set_val_range(lhs, r_lhs);
            set_val_range(rhs, r_rhs);
            break;
        }
        case Opcode::sge: {
            r_rhs.intersect_with(ValueRange::range(INT64_MIN, r_lhs.max_val));
            r_lhs.intersect_with(ValueRange::range(r_rhs.min_val, INT64_MAX));
            set_val_range(lhs, r_lhs);
            set_val_range(rhs, r_rhs);
            break;
        }
        // Unsigned order agrees with signed order only among non-negative
        // values: a negative bound is a huge unsigned number and bounds
        // nothing. So an unsigned fact says something only when the side
        // doing the bounding is known non-negative.
        case Opcode::ult: {
            if (r_rhs.min_val >= 0) {
                r_lhs.intersect_with(ValueRange::range(0, sat_sub(r_rhs.max_val, 1)));
                set_val_range(lhs, r_lhs);
                r_rhs.intersect_with(ValueRange::range(1, INT64_MAX));
                set_val_range(rhs, r_rhs);
            }
            break;
        }
        case Opcode::ule: {
            if (r_rhs.min_val >= 0) {
                r_lhs.intersect_with(ValueRange::range(0, r_rhs.max_val));
                set_val_range(lhs, r_lhs);
            }
            break;
        }
        case Opcode::ugt: {
            if (r_lhs.min_val >= 0) {
                r_rhs.intersect_with(ValueRange::range(0, sat_sub(r_lhs.max_val, 1)));
                set_val_range(rhs, r_rhs);
                r_lhs.intersect_with(ValueRange::range(1, INT64_MAX));
                set_val_range(lhs, r_lhs);
            }
            break;
        }
        case Opcode::uge: {
            if (r_lhs.min_val >= 0) {
                r_rhs.intersect_with(ValueRange::range(0, r_lhs.max_val));
                set_val_range(rhs, r_rhs);
            }
            break;
        }
        case Opcode::eq: {
            ValueRange inter = r_lhs;
            inter.intersect_with(r_rhs);
            set_val_range(lhs, inter);
            set_val_range(rhs, inter);
            break;
        }
        case Opcode::ne: {
            if (r_rhs.is_constant()) {
                if (r_lhs.min_val == r_rhs.min_val && r_lhs.max_val > r_rhs.min_val) {
                    r_lhs.min_val = sat_add(r_rhs.min_val, 1);
                    set_val_range(lhs, r_lhs);
                } else if (r_lhs.max_val == r_rhs.min_val && r_lhs.min_val < r_rhs.min_val) {
                    r_lhs.max_val = sat_sub(r_rhs.min_val, 1);
                    set_val_range(lhs, r_lhs);
                }
            } else if (r_lhs.is_constant()) {
                if (r_rhs.min_val == r_lhs.min_val && r_rhs.max_val > r_lhs.min_val) {
                    r_rhs.min_val = sat_add(r_lhs.min_val, 1);
                    set_val_range(rhs, r_rhs);
                } else if (r_rhs.max_val == r_lhs.min_val && r_rhs.min_val < r_lhs.min_val) {
                    r_rhs.max_val = sat_sub(r_lhs.min_val, 1);
                    set_val_range(rhs, r_rhs);
                }
            }
            break;
        }
        default:
            break;
    }
}

void RangeAnalysis::infer_loop_induction_variables(const LoopAnalysis& loops) {
    for (LoopInfo* loop : loops.post_order_loops()) {
        if (!loop) continue;
        BasicBlock* header = loop->header();
        if (!header || loop->latches().size() != 1) continue;
        BasicBlock* latch = loop->latches()[0];
        BasicBlock* preheader = loop->preheader();
        if (!preheader) {
            std::vector<BasicBlock*> outside_preds;
            for (BasicBlock* pred : header->predecessors()) {
                if (pred && !loop->contains(pred)) {
                    outside_preds.push_back(pred);
                }
            }
            if (outside_preds.size() == 1) {
                preheader = outside_preds[0];
            }
        }
        if (!preheader) continue;

        Instruction* ph_term = preheader->terminator();
        Instruction* latch_term = latch->terminator();
        Instruction* hdr_term = header->terminator();
        if (!ph_term || !latch_term || !hdr_term || hdr_term->opcode() != Opcode::br_if) continue;

        BranchTarget* ph_bt = nullptr;
        if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
            ph_bt = &ph_term->branch_target();
        } else if (ph_term->opcode() == Opcode::br_if) {
            if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
            else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
        }
        if (!ph_bt || ph_bt->args.size() != header->param_count()) continue;
        // Both edges of a br_if into the header could carry different values.
        if (ph_term->opcode() == Opcode::br_if &&
            ph_term->true_target().block == header && ph_term->false_target().block == header) continue;

        BranchTarget* latch_bt = nullptr;
        if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
            latch_bt = &latch_term->branch_target();
        }
        if (!latch_bt || latch_bt->args.size() != header->param_count()) continue;

        bool body_is_true = loop->contains(hdr_term->true_target().block);
        bool body_is_false = loop->contains(hdr_term->false_target().block);
        if (body_is_true == body_is_false) continue;

        Value* cond = hdr_term->operand(0);
        if (!cond || !cond->is_instruction()) continue;
        Instruction* cmp = cond->defining_instruction();
        if (cmp && cmp->opcode() == Opcode::and_) {
            // `a && b` holding says `a` holds; `a && b` failing says nothing
            // about `a`, so only a body on the true edge learns from it.
            if (!body_is_true) continue;
            if (cmp->operand(0) && cmp->operand(0)->is_instruction() && is_comparison(cmp->operand(0)->defining_instruction()->opcode())) {
                cmp = cmp->operand(0)->defining_instruction();
            } else if (cmp->operand(1) && cmp->operand(1)->is_instruction() && is_comparison(cmp->operand(1)->defining_instruction()->opcode())) {
                cmp = cmp->operand(1)->defining_instruction();
            }
        }
        if (!cmp || !is_comparison(cmp->opcode())) continue;

        Opcode cmp_op = body_is_true ? cmp->opcode() : invert_comparison_opcode(cmp->opcode());
        Value* cmp_lhs = cmp->operand(0);
        Value* cmp_rhs = cmp->operand(1);
        if (!is_int_value(cmp_lhs) || !is_int_value(cmp_rhs)) continue;

        for (size_t i = 0; i < header->param_count(); ++i) {
            Value* param = header->param(i);
            if (param != cmp_lhs || !loop->is_loop_invariant(cmp_rhs)) continue;
            Value* init_v = ph_bt->args[i];
            Value* step_v = latch_bt->args[i];
            if (!step_v || !step_v->is_instruction()) continue;
            Instruction* sdef = step_v->defining_instruction();
            if (!sdef || sdef->opcode() != Opcode::add) continue;
            Value* sop0 = sdef->operand(0);
            Value* sop1 = sdef->operand(1);
            Value* step_c_v = (sop0 == param) ? sop1 : ((sop1 == param) ? sop0 : nullptr);
            int64_t step_c = 0;
            if (!step_c_v || !get_const_int(step_c_v, step_c) || step_c <= 0) continue;

            const bool strict = cmp_op == Opcode::slt || cmp_op == Opcode::ult;
            const bool non_strict = cmp_op == Opcode::sle || cmp_op == Opcode::ule;
            if (!strict && !non_strict) continue;
            const bool is_unsigned = cmp_op == Opcode::ult || cmp_op == Opcode::ule;

            const ValueRange init_r = get_range(init_v);
            const ValueRange lim_r = get_range(cmp_rhs);
            if (init_r.is_empty() || lim_r.is_empty()) continue;
            // Unsigned bounds agree with signed ones only for non-negative values.
            if (is_unsigned && (init_r.min_val < 0 || lim_r.min_val < 0)) continue;

            // Largest value the body sees, and the largest the step can then
            // produce. If that step could wrap, the variable can come back
            // around below its start and none of this holds.
            const int64_t type_max = param->type() == Type::i32() ? INT32_MAX : INT64_MAX;
            const int64_t body_max = strict ? sat_sub(lim_r.max_val, 1) : lim_r.max_val;
            if (body_max > type_max - step_c) continue;
            const int64_t after_step_max = body_max + step_c;

            // Every value the header sees is the start value or one step past
            // a value the body saw.
            ValueRange header_r = ValueRange::range(init_r.min_val, std::max(init_r.max_val, after_step_max));
            ValueRange body_r = ValueRange::range(init_r.min_val, body_max);

            auto global_it = global_ranges_.find(param);
            if (global_it != global_ranges_.end()) {
                header_r.intersect_with(global_it->second);
                body_r.intersect_with(global_it->second);
            }
            global_ranges_[param] = header_r;
            for (BasicBlock* lbb : loop->blocks()) {
                block_ranges_[lbb][param] = (lbb == header) ? header_r : body_r;
            }
        }
    }
}

void RangeAnalysis::visit_dominator_block(
    const BasicBlock* bb,
    const BasicBlock* idom,
    const DominatorTree& dom,
    std::unordered_map<const Value*, ValueRange>& current_ranges
) {
    if (!bb) return;

    std::vector<std::pair<const Value*, std::optional<ValueRange>>> rollback;
    auto set_range = [&](const Value* val, const ValueRange& r) {
        auto it = current_ranges.find(val);
        if (it != current_ranges.end()) {
            rollback.emplace_back(val, it->second);
            it->second = r;
        } else {
            rollback.emplace_back(val, std::nullopt);
            current_ranges[val] = r;
        }
    };

    // Load any existing block parameter ranges for this block (e.g. from loop IV inference)
    auto b_it = block_ranges_.find(bb);
    if (b_it != block_ranges_.end()) {
        for (const auto& [param_v, r] : b_it->second) {
            // Both the inherited range and the induction range hold here.
            ValueRange refined = r;
            refined.intersect_with(get_context_range(param_v, current_ranges));
            set_range(param_v, refined);
        }
    }

    if (bb->predecessors().size() == 1) {
        BasicBlock* pred = bb->predecessors()[0];
        Instruction* term = pred ? pred->terminator() : nullptr;
        if (term && term->opcode() == Opcode::br_if) {
            Value* cond = term->operand(0);
            bool to_true = (term->true_target().block == bb && term->false_target().block != bb);
            bool to_false = (term->false_target().block == bb && term->true_target().block != bb);
            if (to_true || to_false) {
                apply_branch_condition(cond, to_true, current_ranges, rollback);
            }
        }
    }

    // Evaluate instructions in basic block
    for (const Instruction* inst : *bb) {
        if (!inst) continue;
        if (inst->produces_value()) {
            ValueRange r = fit_to_type(evaluate_instruction(inst, current_ranges), inst->type());
            set_range(inst->result(), r);

            auto it = global_ranges_.find(inst->result());
            if (it != global_ranges_.end()) {
                it->second.union_with(r);
            } else {
                global_ranges_[inst->result()] = r;
            }
        }
    }

    // What this block changed, at the values the changes left in place. Every
    // write above went through `rollback`, so its entries name exactly the
    // values whose in-scope range differs here from the dominator's.
    {
        BlockDelta& delta = block_deltas_[bb];
        delta.idom = idom;
        delta.changes.reserve(rollback.size());
        for (const auto& entry : rollback) {
            auto it = current_ranges.find(entry.first);
            if (it != current_ranges.end()) delta.changes.emplace_back(entry.first, it->second);
        }
    }

    for (const BasicBlock* child : dom.children(bb)) {
        if (child) {
            visit_dominator_block(child, bb, dom, current_ranges);
        }
    }

    for (auto it = rollback.rbegin(); it != rollback.rend(); ++it) {
        if (it->second.has_value()) {
            current_ranges[it->first] = *it->second;
        } else {
            current_ranges.erase(it->first);
        }
    }
}

void RangeAnalysis::run_analysis(Function& fn, const DominatorTree& dom, const LoopAnalysis& loops) {
    global_ranges_.clear();
    block_ranges_.clear();
    block_deltas_.clear();
    pass3_initial_ranges_.clear();

    // Pass 1: Forward evaluation of instructions and block parameters. The
    // iteration starts optimistic, so its ranges are sound only once nothing
    // changes. From the fifth round on, a range may only grow, and a bound
    // that moves jumps straight to its type's limit, so every value settles
    // within a few more rounds; should the cap still be hit, every range
    // falls back to "anything of its type".
    constexpr size_t kWidenAfter = 4;
    constexpr size_t kMaxIterations = 64;
    auto widen = [](ValueRange next, const ValueRange& prev, Type t) {
        next.union_with(prev);
        const ValueRange limits = full_range_of(t);
        if (next.max_val > prev.max_val) next.max_val = limits.max_val;
        if (next.min_val < prev.min_val) next.min_val = limits.min_val;
        return next;
    };
    bool converged = false;
    for (size_t iter = 0; iter < kMaxIterations; ++iter) {
        bool changed = false;
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;

            // Block params
            if (bb == fn.entry_block()) {
                for (size_t i = 0; i < bb->param_count(); ++i) {
                    Value* p = bb->param(i);
                    if (p && global_ranges_.find(p) == global_ranges_.end()) {
                        ValueRange r = (p->type() == Type::i32()) ? ValueRange::range(INT32_MIN, INT32_MAX) : ValueRange::full();
                        global_ranges_[p] = r;
                        changed = true;
                    }
                }
            } else {
                for (size_t i = 0; i < bb->param_count(); ++i) {
                    Value* p = bb->param(i);
                    if (!p) continue;
                    ValueRange param_r = ValueRange::empty();
                    for (const BasicBlock* pred : bb->predecessors()) {
                        if (!pred) continue;
                        const Instruction* term = pred->terminator();
                        if (!term) continue;
                        auto add_arg = [&](const Value* arg) {
                            if (!arg) return;
                            int64_t c = 0;
                            if (get_const_int(arg, c)) {
                                param_r.union_with(ValueRange::constant(c));
                            } else {
                                auto git = global_ranges_.find(arg);
                                if (git != global_ranges_.end()) {
                                    param_r.union_with(git->second);
                                }
                            }
                        };
                        // Any edge kind may pass the argument: br, br_if,
                        // switch cases and default, invoke.
                        auto from_target = [&](const BranchTarget& bt) {
                            if (bt.block == bb && i < bt.args.size()) add_arg(bt.args[i]);
                        };
                        from_target(term->branch_target());
                        from_target(term->true_target());
                        from_target(term->false_target());
                        for (const auto& sc : term->switch_cases()) from_target(sc.target);
                    }
                    param_r = fit_to_type(param_r, p->type());
                    if (!param_r.is_empty()) {
                        auto it = global_ranges_.find(p);
                        if (it == global_ranges_.end() || it->second != param_r) {
                            if (iter >= kWidenAfter && it != global_ranges_.end()) {
                                param_r = widen(param_r, it->second, p->type());
                            }
                            if (it == global_ranges_.end() || it->second != param_r) {
                                global_ranges_[p] = param_r;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (const Instruction* inst : *bb) {
                if (!inst || !inst->produces_value()) continue;
                ValueRange r = fit_to_type(evaluate_instruction(inst, global_ranges_), inst->type());
                auto it = global_ranges_.find(inst->result());
                if (it == global_ranges_.end() || it->second != r) {
                    if (iter >= kWidenAfter && it != global_ranges_.end()) {
                        r = widen(r, it->second, inst->type());
                    }
                    if (it == global_ranges_.end() || it->second != r) {
                        global_ranges_[inst->result()] = r;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        for (auto& [val, r] : global_ranges_) {
            r = full_range_of(val->type());
        }
    }

    // Pass 2: Loop induction variable inference
    infer_loop_induction_variables(loops);

    // Pass 3: Path-sensitive refinement across dominator tree. The walk
    // works on the snapshot itself: every write it makes is rolled back on
    // the way out, so the map is the starting snapshot again when it returns,
    // and get_range_at reads it as such.
    if (fn.entry_block()) {
        pass3_initial_ranges_ = global_ranges_;
        visit_dominator_block(fn.entry_block(), nullptr, dom, pass3_initial_ranges_);
    }
}

void RangeAnalysis::dump(std::ostream& os) const {
    os << "=== Value Range Analysis Dump ===\n";
    for (const auto& [val, r] : global_ranges_) {
        if (val) {
            os << "  v" << val->id() << " : " << r << "\n";
        }
    }
}

} // namespace brass
