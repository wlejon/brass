#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

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

bool is_int_value(const Value* v) noexcept {
    return v && (v->type() == Type::i32() || v->type() == Type::i64());
}

// An integer comparison `a op b`, as an instruction or as a fact known to hold.
const Instruction* as_int_comparison(const Value* v) {
    if (!v || !v->is_instruction()) return nullptr;
    const Instruction* def = v->defining_instruction();
    if (!def || !is_comparison(def->opcode()) || def->operand_count() != 2) return nullptr;
    if (!is_int_value(def->operand(0)) || !is_int_value(def->operand(1))) return nullptr;
    return def;
}

void fold_branch(Instruction* term, bool take_true) {
    BranchTarget kept = take_true ? term->true_target() : term->false_target();
    term->set_opcode(Opcode::br);
    term->operands().clear();
    term->true_target() = BranchTarget();
    term->false_target() = BranchTarget();
    term->set_branch_target(std::move(kept));
}

// ---------------------------------------------------------------------------
// Comparisons implied by a dominating comparison of the same two values.
//
// A predicate is the set of orderings {less, equal, greater} it accepts,
// read in the signed or the unsigned order (eq and ne mean the same in both).
// If every ordering the known fact allows is one the check accepts, the check
// is true; if none is, it is false. Signed and unsigned orders only agree
// when both values are non-negative, which range analysis must show first.

constexpr uint8_t kLess = 1;
constexpr uint8_t kEqual = 2;
constexpr uint8_t kGreater = 4;

enum class Order : uint8_t { Either, Signed, Unsigned };

struct Predicate {
    uint8_t accepts = 0;
    Order order = Order::Either;
};

std::optional<Predicate> predicate_of(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq:  return Predicate{kEqual, Order::Either};
        case Opcode::ne:  return Predicate{kLess | kGreater, Order::Either};
        case Opcode::slt: return Predicate{kLess, Order::Signed};
        case Opcode::sle: return Predicate{kLess | kEqual, Order::Signed};
        case Opcode::sgt: return Predicate{kGreater, Order::Signed};
        case Opcode::sge: return Predicate{kGreater | kEqual, Order::Signed};
        case Opcode::ult: return Predicate{kLess, Order::Unsigned};
        case Opcode::ule: return Predicate{kLess | kEqual, Order::Unsigned};
        case Opcode::ugt: return Predicate{kGreater, Order::Unsigned};
        case Opcode::uge: return Predicate{kGreater | kEqual, Order::Unsigned};
        default: return std::nullopt;
    }
}

uint8_t swap_sides(uint8_t accepts) noexcept {
    uint8_t out = accepts & kEqual;
    if (accepts & kLess) out |= kGreater;
    if (accepts & kGreater) out |= kLess;
    return out;
}

struct Fact {
    Opcode op;
    const Value* lhs;
    const Value* rhs;
};

// What `fact` says about `check_op(x, y)`, if anything.
std::optional<bool> implied_by(const Fact& fact, Opcode check_op, const Value* x, const Value* y, bool both_non_negative) {
    auto known = predicate_of(fact.op);
    auto check = predicate_of(check_op);
    if (!known || !check) return std::nullopt;
    uint8_t allowed = known->accepts;
    if (fact.lhs == x && fact.rhs == y) {
        // Same orientation.
    } else if (fact.lhs == y && fact.rhs == x) {
        allowed = swap_sides(allowed);
    } else {
        return std::nullopt;
    }
    if (known->order != Order::Either && check->order != Order::Either &&
        known->order != check->order && !both_non_negative) {
        return std::nullopt;
    }
    if ((allowed & ~check->accepts) == 0) return true;
    if ((allowed & check->accepts) == 0) return false;
    return std::nullopt;
}

// The facts that entering a block along an edge controlled by `cond` taking
// its true (or false) side establishes.
void collect_edge_facts(const Value* cond, bool on_true, std::vector<Fact>& out) {
    if (!cond || !cond->is_instruction()) return;
    const Instruction* def = cond->defining_instruction();
    if (!def) return;
    if (const Instruction* cmp = as_int_comparison(cond)) {
        out.push_back(Fact{on_true ? cmp->opcode() : invert_comparison_opcode(cmp->opcode()),
                           cmp->operand(0), cmp->operand(1)});
        return;
    }
    // Comparisons yield 0 or 1, so `and` of them is non-zero only when both
    // hold and `or` is zero only when both fail.
    if (def->opcode() == Opcode::and_ && on_true) {
        collect_edge_facts(def->operand(0), true, out);
        collect_edge_facts(def->operand(1), true, out);
    } else if (def->opcode() == Opcode::or_ && !on_true) {
        collect_edge_facts(def->operand(0), false, out);
        collect_edge_facts(def->operand(1), false, out);
    }
}

class ImpliedCheckFolder {
public:
    ImpliedCheckFolder(Function& fn, Module& mod, const DominatorTree& dom, const RangeAnalysis& ra,
                       const RangeAnalysisOptions& opts)
        : fn_(fn), mod_(mod), dom_(dom), ra_(ra), opts_(opts) {}

    bool run() {
        if (fn_.entry_block()) visit(fn_.entry_block());
        return changed_;
    }

private:
    void visit(BasicBlock* bb) {
        const size_t mark = facts_.size();
        if (bb->predecessors().size() == 1) {
            const BasicBlock* pred = bb->predecessors()[0];
            const Instruction* term = pred ? pred->terminator() : nullptr;
            if (term && term->opcode() == Opcode::br_if && term->true_target().block != term->false_target().block) {
                collect_edge_facts(term->operand(0), term->true_target().block == bb, facts_);
            }
        }

        if (facts_.size() > 0) fold_checks_in(bb);

        for (const BasicBlock* child : dom_.children(bb)) {
            if (child) visit(const_cast<BasicBlock*>(child));
        }
        facts_.resize(mark);
    }

    void fold_checks_in(BasicBlock* bb) {
        for (Instruction* cur = bb->head(); cur; cur = cur->next()) {
            const Instruction* cmp = as_int_comparison(cur->result());
            if (!cmp) continue;
            const Value* x = cmp->operand(0);
            const Value* y = cmp->operand(1);
            const bool both_non_negative =
                ra_.get_range_at(x, bb).min_val >= 0 && ra_.get_range_at(y, bb).min_val >= 0;
            for (auto it = facts_.rbegin(); it != facts_.rend(); ++it) {
                std::optional<bool> outcome = implied_by(*it, cmp->opcode(), x, y, both_non_negative);
                if (!outcome) continue;
                Builder b(mod_);
                b.set_function(&fn_);
                b.position_before(cur);
                replace_all_uses(fn_, cur->result(), b.build_iconst_i32(*outcome ? 1 : 0));
                if (opts_.stats) opts_.stats->checks_implied++;
                changed_ = true;
                break;
            }
        }
    }

    Function& fn_;
    Module& mod_;
    const DominatorTree& dom_;
    const RangeAnalysis& ra_;
    const RangeAnalysisOptions& opts_;
    std::vector<Fact> facts_;
    bool changed_ = false;
};

// ---------------------------------------------------------------------------
// Comparisons, guards and branches whose outcome value ranges decide.

bool eliminate_range_decided_checks(
    Function& fn,
    Module& mod,
    const RangeAnalysis& ra,
    const RangeAnalysisOptions& opts
) {
    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();

            if (as_int_comparison(cur->result())) {
                // The comparison's own range at its block already accounts
                // for signedness, widths and every dominating branch.
                ValueRange r = ra.get_range_at(cur->result(), bb);
                if (r.is_constant() && (r.min_val == 0 || r.min_val == 1)) {
                    Builder b(mod);
                    b.set_function(&fn);
                    b.position_before(cur);
                    Value* folded = b.build_iconst_i32(static_cast<int32_t>(r.min_val));
                    replace_all_uses(fn, cur->result(), folded);

                    Instruction* term = bb->terminator();
                    if (term && term->opcode() == Opcode::br_if && term->operand(0) == folded) {
                        fold_branch(term, r.min_val != 0);
                        if (opts.stats) opts.stats->branches_folded++;
                    }
                    if (opts.stats) opts.stats->bounds_checks_eliminated++;
                    changed = true;
                }
            } else if (cur->opcode() == Opcode::guard) {
                ValueRange r_cond = ra.get_range_at(cur->operand(0), bb);
                if (r_cond.is_constant() && r_cond.min_val != 0) {
                    bb->remove_instruction(cur);
                    if (opts.stats) opts.stats->guards_eliminated++;
                    changed = true;
                }
            } else if (cur->opcode() == Opcode::br_if) {
                ValueRange r_cond = ra.get_range_at(cur->operand(0), bb);
                if (r_cond.is_constant()) {
                    fold_branch(cur, r_cond.min_val != 0);
                    if (opts.stats) opts.stats->branches_folded++;
                    changed = true;
                }
            }

            cur = next;
        }
    }

    return changed;
}

void remove_dead_pure_instructions(Function& fn) {
    bool progress = true;
    while (progress) {
        progress = false;
        auto uses = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (!cur->has_side_effects() && cur->produces_value() && uses[cur->result()] == 0) {
                    bb->remove_instruction(cur);
                    progress = true;
                }
                cur = next;
            }
        }
    }
}

} // namespace

bool run_bounds_check_elimination(Function& fn, Module& mod, const RangeAnalysisOptions& opts) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;
    if (!opts.enable_bce && !opts.enable_implied_checks) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    RangeAnalysis ra(fn, dom, loops);

    bool any_changed = false;
    if (opts.enable_implied_checks) {
        any_changed |= ImpliedCheckFolder(fn, mod, dom, ra, opts).run();
    }
    if (opts.enable_bce) {
        any_changed |= eliminate_range_decided_checks(fn, mod, ra, opts);
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
        cfg_simplify_function(fn);
        fn.rebuild_cfg_predecessors();
        remove_dead_pure_instructions(fn);
        fn.rebuild_cfg_predecessors();
    }

    return any_changed;
}

bool run_bounds_check_elimination(Module& mod, const RangeAnalysisOptions& opts) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= run_bounds_check_elimination(*fn, mod, opts);
        }
    }
    return changed;
}

} // namespace brass
