#include "f64_demote_internal.hpp"
#include <brass/mir/dominators.hpp>
#include <brass/mir/opcodes.hpp>
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
                return (a.lo >= 0.0 && b.lo >= 0.0) || (!a.contains_zero() && !b.contains_zero());
            }
            case Opcode::smod:
                return op(0).lo >= 0.0 && !op(1).contains_zero();
            case Opcode::call:
                if (inst.symbol() == "bronze_f64_mod") return op(0).lo >= 0.0 && !op(1).contains_zero();
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
        if (!stay_true) less = !less;  // staying on false negates the test
        for (int side = 0; side < 2; ++side) {
            const Value* p = cmp->operand(static_cast<size_t>(side));
            const Value* limit = cmp->operand(static_cast<size_t>(1 - side));
            if (!p || !p->is_block_param() || p->defining_block() != header) continue;
            // lhs < rhs bounds lhs above and rhs below.
            const bool upper = (side == 0) ? less : !less;
            guards_[p] = Guard{limit, upper, stay};
        }
    }

    Interval get(const Value* v, const BasicBlock* use_block) const {
        if (!v) return Interval::all();
        if (v->type() == Type::i32()) return {-2147483648.0, 2147483647.0};
        if (v->type() != Type::f64()) return Interval::all();
        auto it = iv_.find(v);
        Interval r = it != iv_.end() ? it->second : (exact_.count(v) ? Interval{} : Interval::all());
        if (use_block) {
            auto g = guards_.find(v);
            if (g != guards_.end() && dom_.dominates(g->second.stay, use_block)) {
                const Interval lim = get(g->second.limit, nullptr);
                if (g->second.upper) r.hi = std::min(r.hi, lim.hi);
                else r.lo = std::max(r.lo, lim.lo);
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
            case Opcode::sitofp_f64_i32: return {-2147483648.0, 2147483647.0};
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

bool drop_unprovable_exact_values(const Function& fn, const DominatorTree& dom, const LoopAnalysis& loops,
                                  std::unordered_set<const Value*>& exact_ints) {
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
    return !drop.empty();
}

} // namespace brass
