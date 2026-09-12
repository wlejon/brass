#include <brass/mir/range_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <iostream>
#include <sstream>
#include <cmath>

namespace brass {

namespace {

inline int64_t sat_add(int64_t a, int64_t b) noexcept {
    __int128_t res = static_cast<__int128_t>(a) + b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
}

inline int64_t sat_sub(int64_t a, int64_t b) noexcept {
    __int128_t res = static_cast<__int128_t>(a) - b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
}

inline int64_t sat_mul(int64_t a, int64_t b) noexcept {
    __int128_t res = static_cast<__int128_t>(a) * b;
    if (res > INT64_MAX) return INT64_MAX;
    if (res < INT64_MIN) return INT64_MIN;
    return static_cast<int64_t>(res);
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

std::string ValueRange::to_string() const {
    if (is_empty()) return "[empty]";
    if (is_constant()) return "[" + std::to_string(min_val) + "]";
    std::ostringstream ss;
    ss << "[";
    if (min_val == INT64_MIN) ss << "-inf";
    else ss << min_val;
    ss << ", ";
    if (max_val == INT64_MAX) ss << "+inf";
    else ss << max_val;
    ss << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const ValueRange& r) {
    os << r.to_string();
    return os;
}

ValueRange ValueRange::add(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    return range(sat_add(a.min_val, b.min_val), sat_add(a.max_val, b.max_val));
}

ValueRange ValueRange::sub(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    return range(sat_sub(a.min_val, b.max_val), sat_sub(a.max_val, b.min_val));
}

ValueRange ValueRange::mul(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    int64_t p1 = sat_mul(a.min_val, b.min_val);
    int64_t p2 = sat_mul(a.min_val, b.max_val);
    int64_t p3 = sat_mul(a.max_val, b.min_val);
    int64_t p4 = sat_mul(a.max_val, b.max_val);
    int64_t min_v = std::min({p1, p2, p3, p4});
    int64_t max_v = std::max({p1, p2, p3, p4});
    return range(min_v, max_v);
}

ValueRange ValueRange::and_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val & b.min_val);
    }
    if (a.is_non_negative() || b.is_non_negative()) {
        int64_t upper = INT64_MAX;
        if (a.is_non_negative() && b.is_non_negative()) {
            upper = std::min(a.max_val, b.max_val);
        } else if (a.is_non_negative()) {
            upper = a.max_val;
        } else {
            upper = b.max_val;
        }
        return range(0, upper);
    }
    return full();
}

ValueRange ValueRange::or_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val | b.min_val);
    }
    if (a.is_non_negative() && b.is_non_negative()) {
        int64_t lower = std::max(a.min_val, b.min_val);
        uint64_t m = static_cast<uint64_t>(a.max_val | b.max_val);
        if (m == 0) return constant(0);
        int clz = __builtin_clzll(m);
        uint64_t bound = (clz == 0) ? UINT64_MAX : ((1ULL << (64 - clz)) - 1ULL);
        int64_t upper = (bound > static_cast<uint64_t>(INT64_MAX)) ? INT64_MAX : static_cast<int64_t>(bound);
        return range(lower, upper);
    }
    return full();
}

ValueRange ValueRange::xor_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val ^ b.min_val);
    }
    if (a.is_non_negative() && b.is_non_negative()) {
        uint64_t m = static_cast<uint64_t>(a.max_val | b.max_val);
        if (m == 0) return constant(0);
        int clz = __builtin_clzll(m);
        uint64_t bound = (clz == 0) ? UINT64_MAX : ((1ULL << (64 - clz)) - 1ULL);
        int64_t upper = (bound > static_cast<uint64_t>(INT64_MAX)) ? INT64_MAX : static_cast<int64_t>(bound);
        return range(0, upper);
    }
    return full();
}

ValueRange ValueRange::shl(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val < 63) {
        int shift = static_cast<int>(b.min_val);
        int64_t factor = 1LL << shift;
        int64_t p1 = sat_mul(a.min_val, factor);
        int64_t p2 = sat_mul(a.max_val, factor);
        return range(std::min(p1, p2), std::max(p1, p2));
    }
    return full();
}

ValueRange ValueRange::lshr(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val <= 64) {
        int shift = static_cast<int>(b.min_val);
        if (shift >= 64) return constant(0);
        if (a.is_non_negative()) {
            return range(a.min_val >> shift, a.max_val >> shift);
        }
        uint64_t max_u = UINT64_MAX >> shift;
        int64_t upper = (max_u > static_cast<uint64_t>(INT64_MAX)) ? INT64_MAX : static_cast<int64_t>(max_u);
        return range(0, upper);
    }
    return full();
}

ValueRange ValueRange::ashr(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val <= 63) {
        int shift = static_cast<int>(b.min_val);
        return range(a.min_val >> shift, a.max_val >> shift);
    }
    return full();
}

ValueRange ValueRange::zext(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    if (a.is_non_negative() && a.max_val <= static_cast<int64_t>(UINT32_MAX)) {
        return range(a.min_val, a.max_val);
    }
    return range(0, static_cast<int64_t>(UINT32_MAX));
}

ValueRange ValueRange::sext(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    int64_t min_v = std::max(static_cast<int64_t>(INT32_MIN), a.min_val);
    int64_t max_v = std::min(static_cast<int64_t>(INT32_MAX), a.max_val);
    if (min_v > max_v) return range(INT32_MIN, INT32_MAX);
    return range(min_v, max_v);
}

ValueRange ValueRange::trunc(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    if (a.min_val >= INT32_MIN && a.max_val <= INT32_MAX) {
        return a;
    }
    return range(INT32_MIN, INT32_MAX);
}

ValueRange ValueRange::select(const ValueRange& cond, const ValueRange& then_r, const ValueRange& else_r) noexcept {
    if (cond.is_empty() || then_r.is_empty() || else_r.is_empty()) return empty();
    if (cond.is_constant()) {
        return (cond.min_val != 0) ? then_r : else_r;
    }
    ValueRange res = then_r;
    res.union_with(else_r);
    return res;
}

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
        case Opcode::shl: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::shl(r0, r1);
        }
        case Opcode::lshr: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return ValueRange::lshr(r0, r1);
        }
        case Opcode::ashr: {
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
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
            ValueRange r0 = get_context_range(inst->operand(0), context_ranges);
            ValueRange r1 = get_context_range(inst->operand(1), context_ranges);
            return evaluate_comparison(op, r0, r1);
        }
        default:
            if (inst->type() == Type::i32()) {
                return ValueRange::range(INT32_MIN, INT32_MAX);
            }
            return ValueRange::full();
    }
}

void RangeAnalysis::apply_branch_condition(
    const Value* cond,
    bool is_true_edge,
    std::unordered_map<const Value*, ValueRange>& ranges
) {
    if (!cond || !cond->is_instruction()) return;
    const Instruction* cdef = cond->defining_instruction();
    if (!cdef) return;

    if (cdef->opcode() == Opcode::and_ && is_true_edge) {
        apply_branch_condition(cdef->operand(0), true, ranges);
        apply_branch_condition(cdef->operand(1), true, ranges);
        return;
    }
    if (cdef->opcode() == Opcode::or_ && !is_true_edge) {
        apply_branch_condition(cdef->operand(0), false, ranges);
        apply_branch_condition(cdef->operand(1), false, ranges);
        return;
    }

    if (!is_comparison(cdef->opcode())) return;

    Opcode op = is_true_edge ? cdef->opcode() : invert_comparison_opcode(cdef->opcode());
    const Value* lhs = cdef->operand(0);
    const Value* rhs = cdef->operand(1);
    if (!lhs || !rhs) return;

    ValueRange r_lhs = get_context_range(lhs, ranges);
    ValueRange r_rhs = get_context_range(rhs, ranges);

    switch (op) {
        case Opcode::slt: {
            r_lhs.intersect_with(ValueRange::range(INT64_MIN, sat_sub(r_rhs.max_val, 1)));
            r_rhs.intersect_with(ValueRange::range(sat_add(r_lhs.min_val, 1), INT64_MAX));
            ranges[lhs] = r_lhs;
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::sle: {
            r_lhs.intersect_with(ValueRange::range(INT64_MIN, r_rhs.max_val));
            r_rhs.intersect_with(ValueRange::range(r_lhs.min_val, INT64_MAX));
            ranges[lhs] = r_lhs;
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::sgt: {
            r_rhs.intersect_with(ValueRange::range(INT64_MIN, sat_sub(r_lhs.max_val, 1)));
            r_lhs.intersect_with(ValueRange::range(sat_add(r_rhs.min_val, 1), INT64_MAX));
            ranges[lhs] = r_lhs;
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::sge: {
            r_rhs.intersect_with(ValueRange::range(INT64_MIN, r_lhs.max_val));
            r_lhs.intersect_with(ValueRange::range(r_rhs.min_val, INT64_MAX));
            ranges[lhs] = r_lhs;
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::ult: {
            if (r_rhs.max_val >= 0) {
                r_lhs.intersect_with(ValueRange::range(0, sat_sub(r_rhs.max_val, 1)));
                ranges[lhs] = r_lhs;
            }
            r_rhs.intersect_with(ValueRange::range(1, INT64_MAX));
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::ule: {
            if (r_rhs.max_val >= 0) {
                r_lhs.intersect_with(ValueRange::range(0, r_rhs.max_val));
                ranges[lhs] = r_lhs;
            }
            ranges[rhs] = r_rhs;
            break;
        }
        case Opcode::ugt: {
            if (r_lhs.max_val >= 0) {
                r_rhs.intersect_with(ValueRange::range(0, sat_sub(r_lhs.max_val, 1)));
                ranges[rhs] = r_rhs;
            }
            r_lhs.intersect_with(ValueRange::range(1, INT64_MAX));
            ranges[lhs] = r_lhs;
            break;
        }
        case Opcode::uge: {
            if (r_lhs.max_val >= 0) {
                r_rhs.intersect_with(ValueRange::range(0, r_lhs.max_val));
                ranges[rhs] = r_rhs;
            }
            ranges[lhs] = r_lhs;
            break;
        }
        case Opcode::eq: {
            ValueRange inter = r_lhs;
            inter.intersect_with(r_rhs);
            ranges[lhs] = inter;
            ranges[rhs] = inter;
            break;
        }
        case Opcode::ne: {
            if (r_rhs.is_constant()) {
                if (r_lhs.min_val == r_rhs.min_val && r_lhs.max_val > r_rhs.min_val) {
                    r_lhs.min_val = sat_add(r_rhs.min_val, 1);
                    ranges[lhs] = r_lhs;
                } else if (r_lhs.max_val == r_rhs.min_val && r_lhs.min_val < r_rhs.min_val) {
                    r_lhs.max_val = sat_sub(r_rhs.min_val, 1);
                    ranges[lhs] = r_lhs;
                }
            } else if (r_lhs.is_constant()) {
                if (r_rhs.min_val == r_lhs.min_val && r_rhs.max_val > r_lhs.min_val) {
                    r_rhs.min_val = sat_add(r_lhs.min_val, 1);
                    ranges[rhs] = r_rhs;
                } else if (r_rhs.max_val == r_lhs.min_val && r_rhs.min_val < r_lhs.min_val) {
                    r_rhs.max_val = sat_sub(r_lhs.min_val, 1);
                    ranges[rhs] = r_rhs;
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

        BranchTarget* latch_bt = nullptr;
        if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
            latch_bt = &latch_term->branch_target();
        }
        if (!latch_bt || latch_bt->args.size() != header->param_count()) continue;

        Value* cond = hdr_term->operand(0);
        if (!cond || !cond->is_instruction()) continue;
        Instruction* cmp = cond->defining_instruction();
        if (!cmp || !is_comparison(cmp->opcode())) continue;

        bool body_is_true = loop->contains(hdr_term->true_target().block);
        bool body_is_false = loop->contains(hdr_term->false_target().block);
        if (body_is_true == body_is_false) continue;

        Opcode cmp_op = body_is_true ? cmp->opcode() : invert_comparison_opcode(cmp->opcode());
        Value* cmp_lhs = cmp->operand(0);
        Value* cmp_rhs = cmp->operand(1);

        for (size_t i = 0; i < header->param_count(); ++i) {
            Value* param = header->param(i);
            if (param == cmp_lhs && loop->is_loop_invariant(cmp_rhs)) {
                Value* init_v = ph_bt->args[i];
                Value* step_v = latch_bt->args[i];
                if (step_v && step_v->is_instruction()) {
                    Instruction* sdef = step_v->defining_instruction();
                    if (sdef && sdef->opcode() == Opcode::add) {
                        Value* sop0 = sdef->operand(0);
                        Value* sop1 = sdef->operand(1);
                        Value* step_c_v = (sop0 == param) ? sop1 : ((sop1 == param) ? sop0 : nullptr);
                        int64_t step_c = 0;
                        if (step_c_v && get_const_int(step_c_v, step_c) && step_c > 0) {
                            ValueRange init_r = get_range(init_v);
                            ValueRange lim_r = get_range(cmp_rhs);
                            int64_t low = init_r.is_empty() ? 0 : init_r.min_val;

                            if (cmp_op == Opcode::slt || cmp_op == Opcode::ult) {
                                int64_t high = (lim_r.max_val < INT64_MAX) ? sat_sub(lim_r.max_val, 1) : INT64_MAX;
                                ValueRange body_r = ValueRange::range(low, high);
                                global_ranges_[param] = body_r;
                                for (BasicBlock* lbb : loop->blocks()) {
                                    if (lbb != header) {
                                        block_ranges_[lbb][param] = body_r;
                                    } else {
                                        block_ranges_[header][param] = ValueRange::range(low, lim_r.max_val);
                                    }
                                }
                            } else if (cmp_op == Opcode::sle || cmp_op == Opcode::ule) {
                                int64_t high = lim_r.max_val;
                                ValueRange body_r = ValueRange::range(low, high);
                                global_ranges_[param] = body_r;
                                for (BasicBlock* lbb : loop->blocks()) {
                                    if (lbb != header) {
                                        block_ranges_[lbb][param] = body_r;
                                    } else {
                                        block_ranges_[header][param] = ValueRange::range(low, sat_add(lim_r.max_val, 1));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void RangeAnalysis::visit_dominator_block(
    const BasicBlock* bb,
    const DominatorTree& dom,
    std::unordered_map<const Value*, ValueRange> current_ranges
) {
    if (!bb) return;

    // Load any existing block parameter ranges for this block (e.g. from loop IV inference)
    auto b_it = block_ranges_.find(bb);
    if (b_it != block_ranges_.end()) {
        for (const auto& [param_v, r] : b_it->second) {
            current_ranges[param_v] = r;
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
                apply_branch_condition(cond, to_true, current_ranges);
            }
        }
    }

    // Evaluate instructions in basic block
    for (const Instruction* inst : *bb) {
        if (!inst) continue;
        if (inst->produces_value()) {
            ValueRange r = evaluate_instruction(inst, current_ranges);
            if (inst->type() == Type::i32()) {
                r.intersect_with(ValueRange::range(INT32_MIN, INT32_MAX));
            }
            current_ranges[inst->result()] = r;

            auto it = global_ranges_.find(inst->result());
            if (it != global_ranges_.end()) {
                it->second.union_with(r);
            } else {
                global_ranges_[inst->result()] = r;
            }
        }
    }

    for (const auto& [val, r] : current_ranges) {
        block_ranges_[bb][val] = r;
    }

    for (const BasicBlock* child : dom.children(bb)) {
        if (child) {
            visit_dominator_block(child, dom, current_ranges);
        }
    }
}

void RangeAnalysis::run_analysis(Function& fn, const DominatorTree& dom, const LoopAnalysis& loops) {
    global_ranges_.clear();
    block_ranges_.clear();

    // Pass 1: Forward evaluation of instructions and block parameters
    for (size_t iter = 0; iter < 16; ++iter) {
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
                        if (term->opcode() == Opcode::br && term->branch_target().block == bb) {
                            if (i < term->branch_target().args.size()) {
                                add_arg(term->branch_target().args[i]);
                            }
                        } else if (term->opcode() == Opcode::br_if) {
                            if (term->true_target().block == bb && i < term->true_target().args.size()) {
                                add_arg(term->true_target().args[i]);
                            }
                            if (term->false_target().block == bb && i < term->false_target().args.size()) {
                                add_arg(term->false_target().args[i]);
                            }
                        }
                    }
                    if (p->type() == Type::i32()) {
                        param_r.intersect_with(ValueRange::range(INT32_MIN, INT32_MAX));
                    }
                    if (!param_r.is_empty()) {
                        auto it = global_ranges_.find(p);
                        if (it == global_ranges_.end() || it->second != param_r) {
                            global_ranges_[p] = param_r;
                            changed = true;
                        }
                    }
                }
            }

            for (const Instruction* inst : *bb) {
                if (!inst || !inst->produces_value()) continue;
                ValueRange r = evaluate_instruction(inst, global_ranges_);
                if (inst->type() == Type::i32()) {
                    r.intersect_with(ValueRange::range(INT32_MIN, INT32_MAX));
                }
                auto it = global_ranges_.find(inst->result());
                if (it == global_ranges_.end() || it->second != r) {
                    global_ranges_[inst->result()] = r;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // Pass 2: Loop induction variable inference
    infer_loop_induction_variables(loops);

    // Pass 3: Path-sensitive refinement across dominator tree
    if (fn.entry_block()) {
        std::unordered_map<const Value*, ValueRange> initial_ranges = global_ranges_;
        visit_dominator_block(fn.entry_block(), dom, initial_ranges);
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
