#include "f64_demote_internal.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

// An f64 value can be carried in an i64 only while the two agree: its
// magnitude stays within 2^53 (beyond that f64 rounds and i64 does not), and
// it is never -0.0 (i64 has no negative zero, and 1/x, Math.sign or Object.is
// can tell the difference). Integer-valuedness alone, which the dataflow in
// f64_demote.cpp establishes, is not enough: an accumulator or a product
// grows past 2^53 long before i64 overflows. This file bounds every candidate
// with interval analysis and removes those it cannot prove.
namespace brass {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kExactLimit = 9007199254740992.0; // 2^53

struct Interval {
    double lo = kInf;   // empty until a value flows in
    double hi = -kInf;

    static Interval all() { return {-kInf, kInf}; }
    static Interval point(double v) { return {v, v}; }
    bool empty() const { return lo > hi; }
    bool contains_zero() const { return lo <= 0.0 && hi >= 0.0; }
    bool operator==(const Interval& o) const { return lo == o.lo && hi == o.hi; }
    bool within(const Interval& o) const { return empty() || (o.lo <= lo && hi <= o.hi); }
};

Interval hull(const Interval& a, const Interval& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
}

double safe_mul(double a, double b) {
    // 0 * inf is NaN; a zero bound times anything is zero.
    if (a == 0.0 || b == 0.0) return 0.0;
    return a * b;
}

Interval mul(const Interval& a, const Interval& b) {
    const double p[4] = {safe_mul(a.lo, b.lo), safe_mul(a.lo, b.hi), safe_mul(a.hi, b.lo), safe_mul(a.hi, b.hi)};
    return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}

double magnitude(const Interval& a) { return std::max(std::fabs(a.lo), std::fabs(a.hi)); }

// The guard a loop header puts on one of its parameters for the iterations
// that stay in the loop: `lower` / `upper` bound the parameter by `limit`.
struct Guard {
    const Value* limit = nullptr;
    bool upper = true;
    bool strict = false;
    const BasicBlock* stay = nullptr;
};

class Bounds {
public:
    Bounds(const Function& fn, const DominatorTree& dom, const LoopAnalysis& loops,
           const std::unordered_set<const Value*>& exact)
        : fn_(fn), dom_(dom), exact_(exact) {
        for (const LoopInfo* loop : loops.post_order_loops()) {
            if (loop) collect_guards(*loop);
        }
    }

    // Returns false if the analysis did not settle; nothing can be trusted then.
    bool solve() {
        constexpr int kMaxRounds = 64;
        constexpr int kWidenAfter = 3;
        for (int round = 0; round < kMaxRounds; ++round) {
            bool changed = false;
            for (const BasicBlock* bb : fn_.blocks()) {
                if (!bb) continue;
                for (size_t i = 0; i < bb->param_count(); ++i) {
                    changed |= update(bb->param(i), param_interval(bb, i), kWidenAfter);
                }
                for (const Instruction* inst : *bb) {
                    if (inst && inst->result() && exact_.count(inst->result())) {
                        changed |= update(inst->result(), instruction_interval(*inst), kWidenAfter);
                    }
                }
            }
            if (!changed) return true;
        }
        return false;
    }

    bool fits_in_i32(const Value* v) const {
        const Interval r = get(v, nullptr);
        return !r.empty() && r.lo >= -2147483648.0 && r.hi <= 2147483647.0;
    }

    // Whether `v` provably agrees with its i64 image everywhere it is computed.
    bool representable(const Value* v) const {
        const Interval r = get(v, nullptr);
        if (r.empty()) return true;  // never computed
        if (magnitude(r) > kExactLimit) return false;
        if (!v->is_instruction()) return true;
        const Instruction& inst = *v->defining_instruction();
        const BasicBlock* bb = inst.parent();
        auto op = [&](size_t i) { return get(inst.operand(i), bb); };
        switch (inst.opcode()) {
            case Opcode::fconst_f64:
                return !std::signbit(inst.imm_f64());
            case Opcode::neg:
                return !op(0).contains_zero();
            case Opcode::mul: {
                const Interval a = op(0);
                const Interval b = op(1);
                return (!a.contains_zero() || b.lo >= 0.0) && (!b.contains_zero() || a.lo >= 0.0);
            }
            case Opcode::smod:
                return op(0).lo >= 0.0 && !op(1).contains_zero();
            case Opcode::call:
                if (callee_has_role(inst, SymbolRole::FloatRem)) return op(0).lo >= 0.0 && !op(1).contains_zero();
                return true;
            case Opcode::sdiv:
                return !op(1).contains_zero() && (!op(0).contains_zero() || op(1).lo > 0.0);
            default:
                return true;
        }
    }

private:
    void collect_guards(const LoopInfo& loop) {
        const BasicBlock* header = loop.header();
        const Instruction* term = header ? header->terminator() : nullptr;
        if (!term || term->opcode() != Opcode::br_if || !term->operand(0) || !term->operand(0)->is_instruction()) return;
        const BasicBlock* t = term->true_target().block;
        const BasicBlock* f = term->false_target().block;
        const bool stay_true = loop.contains(t) && !loop.contains(f);
        const bool stay_false = loop.contains(f) && !loop.contains(t);
        if (!stay_true && !stay_false) return;
        const BasicBlock* stay = stay_true ? t : f;
        // The stay block must be entered only from the header for the guard
        // to hold everywhere it dominates.
        if (stay->predecessors().size() != 1) return;

        const Instruction* cmp = term->operand(0)->defining_instruction();
        if (cmp->operand_count() != 2) return;
        bool less = false;  // lhs < or <= rhs holds when true
        switch (cmp->opcode()) {
            case Opcode::slt: case Opcode::sle: less = true; break;
            case Opcode::sgt: case Opcode::sge: less = false; break;
            default: return;
        }
        const bool strict = stay_true
            ? (cmp->opcode() == Opcode::slt || cmp->opcode() == Opcode::sgt)
            : (cmp->opcode() == Opcode::sle || cmp->opcode() == Opcode::sge);
        if (!stay_true) less = !less;  // staying on false negates the test
        for (int side = 0; side < 2; ++side) {
            const Value* p = cmp->operand(static_cast<size_t>(side));
            const Value* limit = cmp->operand(static_cast<size_t>(1 - side));
            if (!p || !p->is_block_param() || p->defining_block() != header) continue;
            // lhs < rhs bounds lhs above and rhs below.
            const bool upper = (side == 0) ? less : !less;
            guards_[p] = Guard{limit, upper, strict, stay};
        }
    }

    bool is_int_or_exact(const Value* v) const {
        return v && (v->type().is_integer() || exact_.count(v) != 0);
    }

    Interval get(const Value* v, const BasicBlock* use_block) const {
        if (!v) return Interval::all();
        if (v->type() == Type::i32()) {
            if (v->is_instruction() && v->defining_instruction()->opcode() == Opcode::iconst_i32) {
                return Interval::point(static_cast<double>(v->defining_instruction()->imm_i32()));
            }
            Interval r = {-2147483648.0, 2147483647.0};
            auto g = guards_.find(v);
            if (use_block && g != guards_.end() && dom_.dominates(g->second.stay, use_block)) {
                const Interval lim = get(g->second.limit, nullptr);
                const double delta = (g->second.strict && is_int_or_exact(g->second.limit)) ? 1.0 : 0.0;
                if (g->second.upper) r.hi = std::min(r.hi, lim.hi - delta);
                else r.lo = std::max(r.lo, lim.lo + delta);
            }
            if (v->is_block_param() && v->defining_block() && g != guards_.end() && g->second.upper) {
                const Interval lim = get(g->second.limit, nullptr);
                const double delta = (g->second.strict && is_int_or_exact(g->second.limit)) ? 1.0 : 0.0;
                if (lim.hi - delta < 2147483647.0) {
                    const BasicBlock* hdr = v->defining_block();
                    const size_t p_idx = v->param_index();
                    bool non_neg_iv = !hdr->predecessors().empty();
                    for (const BasicBlock* pred : hdr->predecessors()) {
                        const Instruction* term = pred ? pred->terminator() : nullptr;
                        if (!term) { non_neg_iv = false; break; }
                        auto check_bt = [&](const BranchTarget& bt) {
                            if (bt.block != hdr) return;
                            if (p_idx >= bt.args.size() || !bt.args[p_idx] || !bt.args[p_idx]->is_instruction()) {
                                non_neg_iv = false;
                                return;
                            }
                            const Instruction* d = bt.args[p_idx]->defining_instruction();
                            if (d->opcode() == Opcode::iconst_i32 && d->imm_i32() >= 0) return;
                            if (d->opcode() == Opcode::add && dom_.dominates(g->second.stay, pred)) {
                                const Value* s = (d->operand(0) == v) ? d->operand(1) : ((d->operand(1) == v) ? d->operand(0) : nullptr);
                                if (s && s->is_instruction() && s->defining_instruction()->opcode() == Opcode::iconst_i32 &&
                                    s->defining_instruction()->imm_i32() == 1) {
                                    return;
                                }
                            }
                            non_neg_iv = false;
                        };
                        check_bt(term->branch_target());
                        check_bt(term->true_target());
                        check_bt(term->false_target());
                    }
                    if (non_neg_iv) r.lo = std::max(r.lo, 0.0);
                }
            }
            return r;
        }
        if (v->type() == Type::i64()) {
            if (v->is_instruction()) {
                const Instruction* d = v->defining_instruction();
                if (d->opcode() == Opcode::iconst_i64 && std::fabs(static_cast<double>(d->imm_i64())) <= kExactLimit) {
                    return Interval::point(static_cast<double>(d->imm_i64()));
                }
                if (d->opcode() == Opcode::sext_i64) return get(d->operand(0), use_block);
                if (d->opcode() == Opcode::zext_i64) return {0.0, 4294967295.0};
            }
            return Interval::all();
        }
        if (v->type() != Type::f64()) return Interval::all();
        auto it = iv_.find(v);
        Interval r = it != iv_.end() ? it->second : (exact_.count(v) ? Interval{} : Interval::all());
        if (use_block) {
            auto g = guards_.find(v);
            if (g != guards_.end() && dom_.dominates(g->second.stay, use_block)) {
                const Interval lim = get(g->second.limit, nullptr);
                const double delta = (g->second.strict && is_int_or_exact(v) && is_int_or_exact(g->second.limit)) ? 1.0 : 0.0;
                if (g->second.upper) r.hi = std::min(r.hi, lim.hi - delta);
                else r.lo = std::max(r.lo, lim.lo + delta);
            }
        }
        return r;
    }

    Interval param_interval(const BasicBlock* bb, size_t idx) const {
        const Value* p = bb->param(idx);
        if (p->type() != Type::f64() || !exact_.count(p)) return Interval::all();
        Interval r;
        for (const BasicBlock* pred : bb->predecessors()) {
            const Instruction* term = pred ? pred->terminator() : nullptr;
            if (!term) return Interval::all();
            auto visit = [&](const BranchTarget& bt) {
                if (bt.block != bb) return;
                r = hull(r, idx < bt.args.size() ? get(bt.args[idx], pred) : Interval::all());
            };
            visit(term->branch_target());
            visit(term->true_target());
            visit(term->false_target());
            for (const auto& sc : term->switch_cases()) visit(sc.target);
        }
        return r;
    }

    Interval instruction_interval(const Instruction& inst) const {
        const BasicBlock* bb = inst.parent();
        auto op = [&](size_t i) { return get(inst.operand(i), bb); };
        switch (inst.opcode()) {
            case Opcode::fconst_f64: return Interval::point(inst.imm_f64());
            case Opcode::sitofp_f64_i32: return op(0);
            case Opcode::sitofp_f64_i64: return op(0);
            case Opcode::add: return {op(0).lo + op(1).lo, op(0).hi + op(1).hi};
            case Opcode::sub: return {op(0).lo - op(1).hi, op(0).hi - op(1).lo};
            case Opcode::mul: return mul(op(0), op(1));
            case Opcode::neg: return {-op(0).hi, -op(0).lo};
            case Opcode::select: return hull(op(1), op(2));
            case Opcode::smod:
            case Opcode::sdiv:
            case Opcode::call: {
                const double m = magnitude(op(0));
                return {-m, m};
            }
            default: return Interval::all();
        }
    }

    bool update(const Value* v, Interval next, int widen_after) {
        if (!v || !exact_.count(v)) return false;
        if (std::isnan(next.lo) || std::isnan(next.hi)) next = Interval::all();
        Interval& cur = iv_[v];
        if (next == cur) return false;
        if (!next.within(cur) && ++grow_count_[v] > widen_after && !cur.empty()) {
            if (next.hi > cur.hi) next.hi = kInf;
            if (next.lo < cur.lo) next.lo = -kInf;
        }
        cur = next;
        return true;
    }

    const Function& fn_;
    const DominatorTree& dom_;
    const std::unordered_set<const Value*>& exact_;
    std::unordered_map<const Value*, Guard> guards_;
    std::unordered_map<const Value*, Interval> iv_;
    std::unordered_map<const Value*, int> grow_count_;
};

} // namespace

bool canonicalize_loop_bounds(Function& fn, const DominatorTree& dom, const LoopAnalysis& loop_analysis) {
    if (!fn.parent()) return false;
    bool canonicalized_bounds = false;
    std::unordered_set<const Value*> canonical_ivs;

    const auto& po_loops = loop_analysis.post_order_loops();
    for (auto it = po_loops.rbegin(); it != po_loops.rend(); ++it) {
        LoopInfo* loop = *it;
        if (!loop || !loop->header() || loop->latches().size() != 1) continue;
        BasicBlock* header = loop->header();
        BasicBlock* ph = loop->preheader();
        if (!ph) {
            for (BasicBlock* pred : header->predecessors()) {
                if (pred && !loop->contains(pred)) {
                    if (ph) { ph = nullptr; break; }
                    ph = pred;
                }
            }
        }
        if (!ph || is_unprofitable_loop(loop, fn)) continue;
        bool loop_has_fptosi = false;
        for (const BasicBlock* bb : loop->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst && (inst->opcode() == Opcode::fptosi_i32 || inst->opcode() == Opcode::fptosi_i64)) {
                    loop_has_fptosi = true;
                    break;
                }
            }
            if (loop_has_fptosi) break;
        }
        if (!loop_has_fptosi) continue;
        BasicBlock* latch = loop->latches()[0];
        Instruction* ph_term = ph->terminator();
        Instruction* latch_term = latch ? latch->terminator() : nullptr;
        Instruction* hdr_term = header->terminator();
        if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) continue;
        if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header) continue;
        if (!hdr_term || hdr_term->opcode() != Opcode::br_if || !hdr_term->operand(0) || !hdr_term->operand(0)->is_instruction()) continue;

        const bool stay_true = loop->contains(hdr_term->true_target().block);
        const bool stay_false = loop->contains(hdr_term->false_target().block);
        if (stay_true == stay_false) continue;

        Instruction* hdr_cmp = hdr_term->operand(0)->defining_instruction();
        if (!hdr_cmp || hdr_cmp->operand_count() != 2) continue;

        for (size_t i = 0; i < header->param_count(); ++i) {
            Value* p = header->param(i);
            if (!p || p->type() != Type::f64()) continue;
            if (i >= ph_term->branch_target().args.size() || i >= latch_term->branch_target().args.size()) continue;
            Value* init_v = ph_term->branch_target().args[i];
            if (!init_v) continue;

            double init_const = -1.0;
            bool is_valid_init = false;
            if (init_v->is_instruction() && init_v->defining_instruction()->opcode() == Opcode::fconst_f64) {
                init_const = init_v->defining_instruction()->imm_f64();
                is_valid_init = is_safe_integer_f64(init_const) && init_const >= 0.0 && init_const <= 2147483647.0;
            } else if (canonical_ivs.count(init_v) != 0) {
                is_valid_init = true;
            } else if (init_v->is_instruction() && init_v->defining_instruction()->opcode() == Opcode::add) {
                Instruction* idef = init_v->defining_instruction();
                Value* op0 = idef->operand(0);
                Value* op1 = idef->operand(1);
                Value* c_op = canonical_ivs.count(op0) ? op1 : (canonical_ivs.count(op1) ? op0 : nullptr);
                if (c_op && c_op->is_instruction() && c_op->defining_instruction()->opcode() == Opcode::fconst_f64) {
                    const double d = c_op->defining_instruction()->imm_f64();
                    if (d == 0.0 || d == 1.0) is_valid_init = true;
                }
            }
            if (!is_valid_init) continue;

            Value* step_v = latch_term->branch_target().args[i];
            if (!step_v || !step_v->is_instruction() || step_v->defining_instruction()->opcode() != Opcode::add) continue;
            Instruction* sdef = step_v->defining_instruction();
            auto matches_iv = [&](Value* v) -> bool {
                if (v == p) return true;
                if (v && v->is_block_param() && v->defining_block() == latch && latch->predecessors().size() == 1) {
                    BasicBlock* lpred = latch->predecessors()[0];
                    Instruction* lterm = lpred ? lpred->terminator() : nullptr;
                    if (lterm && lterm->opcode() == Opcode::br && lterm->branch_target().block == latch) {
                        const size_t pidx = v->param_index();
                        if (pidx < lterm->branch_target().args.size() && lterm->branch_target().args[pidx] == p) {
                            return true;
                        }
                    }
                }
                return false;
            };
            Value* step_c = matches_iv(sdef->operand(0)) ? sdef->operand(1) : (matches_iv(sdef->operand(1)) ? sdef->operand(0) : nullptr);
            if (!step_c || !step_c->is_instruction() || step_c->defining_instruction()->opcode() != Opcode::fconst_f64 ||
                step_c->defining_instruction()->imm_f64() != 1.0) {
                continue;
            }

            const size_t lim_idx = (hdr_cmp->operand(0) == p) ? 1 : ((hdr_cmp->operand(1) == p) ? 0 : 2);
            if (lim_idx > 1) continue;
            const bool is_p_lt_lim = (lim_idx == 1)
                ? (stay_true ? hdr_cmp->opcode() == Opcode::slt : hdr_cmp->opcode() == Opcode::sge)
                : (stay_true ? hdr_cmp->opcode() == Opcode::sgt : hdr_cmp->opcode() == Opcode::sle);
            if (!is_p_lt_lim) continue;

            Value* limit_v = hdr_cmp->operand(lim_idx);
            if (!limit_v || limit_v->type() != Type::f64() || !loop->is_loop_invariant(limit_v)) continue;
            if (limit_v->is_instruction()) {
                const Instruction* ldef = limit_v->defining_instruction();
                const Opcode lop = ldef->opcode();
                if (lop == Opcode::fconst_f64 || lop == Opcode::sitofp_f64_i32 || lop == Opcode::sitofp_f64_i64) {
                    canonical_ivs.insert(p);
                    continue;
                }
                if (!ldef->parent() || !dom.dominates(ldef->parent(), ph)) continue;
            } else if (limit_v->is_block_param()) {
                if (!limit_v->defining_block() || !dom.dominates(limit_v->defining_block(), ph)) continue;
            }

            Builder ph_b(*fn.parent());
            ph_b.set_function(&fn);
            ph_b.position_before(ph_term);
            Value* ceil_lim = ph_b.build_ceil_f64(limit_v);
            Value* init_f_v = (init_const >= 0.0) ? ph_b.build_fconst_f64(init_const) : init_v;
            Value* max_i32_f = ph_b.build_fconst_f64(2147483647.0);
            Value* gt_init = ph_b.build_sgt(ceil_lim, init_f_v);
            Value* le_max = ph_b.build_sle(ceil_lim, max_i32_f);
            Value* clamped_f = ph_b.build_select(le_max, ceil_lim, max_i32_f);
            Value* safe_f = ph_b.build_select(gt_init, clamped_f, init_f_v);
            Value* lim_i32 = ph_b.build_fptosi_i32(safe_f);
            Value* exact_lim = ph_b.build_sitofp_f64_i32(lim_i32);
            hdr_cmp->set_operand(lim_idx, exact_lim);
            canonical_ivs.insert(p);
            canonicalized_bounds = true;
        }
    }
    return canonicalized_bounds;
}

bool drop_unprovable_exact_values(const Function& fn, const DominatorTree& dom, const LoopAnalysis& loops,
                                  std::unordered_set<const Value*>& exact_ints,
                                  std::unordered_set<const Value*>* i32_bounded) {
    if (exact_ints.empty()) return false;
    Bounds bounds(fn, dom, loops, exact_ints);
    if (!bounds.solve()) {
        exact_ints.clear();
        return true;
    }
    std::vector<const Value*> drop;
    for (const Value* v : exact_ints) {
        if (!bounds.representable(v)) drop.push_back(v);
    }
    for (const Value* v : drop) exact_ints.erase(v);
    if (drop.empty() && i32_bounded) {
        for (const Value* v : exact_ints) {
            if (bounds.fits_in_i32(v)) i32_bounded->insert(v);
        }
    }
    return !drop.empty();
}

} // namespace brass
