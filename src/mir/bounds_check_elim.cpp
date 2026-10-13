#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/uses.hpp>
#include "loop_unswitch_internal.hpp"
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

bool get_const_int(const Value* v, int64_t& out) {
    if (!v || !v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d) return false;
    if (d->opcode() == Opcode::iconst_i32 || d->opcode() == Opcode::patchable_const_i32) {
        out = static_cast<int64_t>(d->imm_i32());
        return true;
    }
    if (d->opcode() == Opcode::iconst_i64 || d->opcode() == Opcode::patchable_const_i64) {
        out = d->imm_i64();
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

bool is_int_value(const Value* v) noexcept {
    return v && (v->type() == Type::i32() || v->type() == Type::i64());
}

bool is_ptr_or_int_value(const Value* v) noexcept {
    return v && (v->type() == Type::i32() || v->type() == Type::i64() || v->type().is_pointer_or_gcref());
}

// A comparison `a op b`, as an instruction or as a fact known to hold.
const Instruction* as_int_comparison(const Value* v) {
    if (!v || !v->is_instruction()) return nullptr;
    const Instruction* def = v->defining_instruction();
    if (!def || !is_comparison(def->opcode()) || def->operand_count() != 2) return nullptr;
    if (!is_ptr_or_int_value(def->operand(0)) || !is_ptr_or_int_value(def->operand(1))) return nullptr;
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

class ImpliedCheckFolder {
public:
    ImpliedCheckFolder(Function& fn, Module& mod, const DominatorTree& dom, const LoopAnalysis& loops,
                       const RangeAnalysis& ra, const AliasAnalysis& aa, const RangeAnalysisOptions& opts)
        : fn_(fn), mod_(mod), dom_(dom), loops_(loops), ra_(ra), aa_(aa), opts_(opts) {}

    bool run() {
        struct Marks {
            size_t facts;
            size_t loads;
            size_t exprs;
        };
        std::vector<Marks> marks;
        walk_dominator_tree(
            dom_, fn_.entry_block(),
            [&](const BasicBlock* bb) {
                marks.push_back({facts_.size(), dom_loads_.size(), dom_exprs_.size()});
                visit(const_cast<BasicBlock*>(bb));
            },
            [&](const BasicBlock*) {
                facts_.resize(marks.back().facts);
                dom_loads_.resize(marks.back().loads);
                dom_exprs_.resize(marks.back().exprs);
                marks.pop_back();
            });
        return changed_;
    }

private:
    struct DomExpr {
        Opcode op;
        const Value* op0;
        const Value* op1;
        int64_t imm;
        const Value* result;
    };

    bool is_load_preserved(const Instruction* dom_load, const Instruction* cur_load) const {
        const BasicBlock* dom_bb = dom_load->parent();
        const BasicBlock* cur_bb = cur_load->parent();
        if (!dom_bb || !cur_bb) return false;

        // Check non-call memory writers across the function
        for (const BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                if (!is_call(inst->opcode()) && aa_.can_clobber(inst, cur_load)) {
                    return false;
                }
            }
        }
        // Check calls in blocks that dominate cur_bb (after dom_load)
        for (const BasicBlock* bb : fn_.blocks()) {
            if (!bb || !dom_.dominates(dom_bb, bb) || !dom_.dominates(bb, cur_bb)) continue;
            bool after_dom = (bb != dom_bb);
            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                if (inst == dom_load) { after_dom = true; continue; }
                if (inst == cur_load) break;
                if (after_dom && is_call(inst->opcode()) && aa_.can_clobber(inst, cur_load)) {
                    return false;
                }
            }
        }
        // Check calls in any enclosing loop that execute on the main loop path (dominate a latch)
        for (LoopInfo* loop : loops_.post_order_loops()) {
            if (!loop || !loop->contains(cur_bb)) continue;
            for (BasicBlock* lbb : loop->blocks()) {
                if (!lbb) continue;
                bool dominates_any_latch = false;
                for (BasicBlock* latch : loop->latches()) {
                    if (latch && dom_.dominates(lbb, latch)) {
                        dominates_any_latch = true;
                        break;
                    }
                }
                if (!dominates_any_latch) continue;
                for (const Instruction* inst : *lbb) {
                    if (inst && is_call(inst->opcode()) && aa_.can_clobber(inst, cur_load)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    const Value* canonicalize(const Value* v, const BasicBlock* bb, int depth = 0) const {
        if (!v || depth > 6) return v;
        while (v && v->is_instruction()) {
            const Instruction* d = v->defining_instruction();
            if (!d) break;
            if ((d->opcode() == Opcode::sext_i64 || d->opcode() == Opcode::zext_i64) &&
                ra_.get_range_at(d->operand(0), bb).min_val >= 0) {
                v = d->operand(0);
            } else {
                break;
            }
        }
        if (!v || !v->is_instruction()) return v;
        const Instruction* d = v->defining_instruction();
        if (!d) return v;
        if (d->opcode() == Opcode::load && d->offset() >= 0 && d->offset() <= 64) {
            const Value* c_base = canonicalize(d->operand(0), bb, depth + 1);
            for (const Instruction* dl : dom_loads_) {
                if (dl == d) return v;
                if (dl->offset() == d->offset() && dl->memory_type() == d->memory_type() &&
                    dl->type() == d->type() &&
                    canonicalize(dl->operand(0), dl->parent(), depth + 1) == c_base &&
                    is_load_preserved(dl, d)) {
                    return dl->result();
                }
            }
        } else if (d->opcode() == Opcode::lshr || d->opcode() == Opcode::ashr ||
                   d->opcode() == Opcode::shl || d->opcode() == Opcode::add ||
                   d->opcode() == Opcode::sub || d->opcode() == Opcode::mul ||
                   d->opcode() == Opcode::and_) {
            const Value* c0 = d->operand_count() >= 1 ? canonicalize(d->operand(0), bb, depth + 1) : nullptr;
            const Value* c1 = d->operand_count() >= 2 ? canonicalize(d->operand(1), bb, depth + 1) : nullptr;
            int64_t c1_const = 0;
            bool has_c1 = get_const_int(c1, c1_const);
            for (const DomExpr& de : dom_exprs_) {
                if (de.op == d->opcode() && de.op0 == c0) {
                    int64_t de_const = 0;
                    if (de.op1 == c1 || (has_c1 && get_const_int(de.op1, de_const) && de_const == c1_const)) {
                        return de.result;
                    }
                }
            }
        }
        return v;
    }

public:
    static bool is_bool_expr(const Value* v, int depth = 0) {
        if (!v || !v->is_instruction() || depth > 6) return false;
        if (as_int_comparison(v)) return true;
        int64_t c = 0;
        if (get_const_int(v, c)) return c == 0 || c == 1;
        const Instruction* d = v->defining_instruction();
        if (!d || (d->opcode() != Opcode::and_ && d->opcode() != Opcode::or_) || d->operand_count() != 2) return false;
        return is_bool_expr(d->operand(0), depth + 1) && is_bool_expr(d->operand(1), depth + 1);
    }

private:

    void collect_edge_facts(const Value* cond, bool on_true, const BasicBlock* bb) {
        if (!cond || !cond->is_instruction()) return;
        const Instruction* def = cond->defining_instruction();
        if (!def) return;
        if (const Instruction* cmp = as_int_comparison(cond)) {
            Opcode op = on_true ? cmp->opcode() : invert_comparison_opcode(cmp->opcode());
            const Value* lhs = canonicalize(cmp->operand(0), bb);
            const Value* rhs = canonicalize(cmp->operand(1), bb);
            facts_.push_back(Fact{op, lhs, rhs});
            return;
        }
        if (def->opcode() == Opcode::and_ && on_true && is_bool_expr(cond)) {
            collect_edge_facts(def->operand(0), true, bb);
            collect_edge_facts(def->operand(1), true, bb);
        } else if (def->opcode() == Opcode::or_ && !on_true) {
            collect_edge_facts(def->operand(0), false, bb);
            collect_edge_facts(def->operand(1), false, bb);
        }
    }

    bool match_shifted(const Value* v, const BasicBlock* bb, const Value*& out_idx, int64_t& out_shift) const {
        if (!v || !v->is_instruction()) return false;
        const Instruction* d = v->defining_instruction();
        if (!d) return false;
        int64_t c = 0;
        if (d->opcode() == Opcode::shl && get_const_int(d->operand(1), c) && c >= 1 && c <= 4) {
            out_idx = canonicalize(d->operand(0), bb);
            out_shift = c;
            return true;
        }
        if (d->opcode() == Opcode::mul) {
            const Value* idx_op = nullptr;
            if (get_const_int(d->operand(1), c)) idx_op = d->operand(0);
            else if (get_const_int(d->operand(0), c)) idx_op = d->operand(1);
            if (idx_op && (c == 2 || c == 4 || c == 8 || c == 16)) {
                out_idx = canonicalize(idx_op, bb);
                out_shift = (c == 2) ? 1 : (c == 4) ? 2 : (c == 8) ? 3 : 4;
                return true;
            }
        }
        return false;
    }

    bool is_length_bound(const Value* n, const Value* byte_len, int64_t shift, const BasicBlock* bb) const {
        const Value* cn = canonicalize(n, bb);
        while (cn && cn->is_instruction()) {
            const Instruction* d = cn->defining_instruction();
            if (d && (d->opcode() == Opcode::trunc_i32 || d->opcode() == Opcode::sext_i64 || d->opcode() == Opcode::zext_i64)) {
                cn = canonicalize(d->operand(0), bb);
            } else {
                break;
            }
        }
        const Value* cbl = canonicalize(byte_len, bb);
        if (cn && cn->is_instruction()) {
            const Instruction* d = cn->defining_instruction();
            int64_t sc = 0;
            if ((d->opcode() == Opcode::lshr || d->opcode() == Opcode::ashr) &&
                get_const_int(d->operand(1), sc) && sc == shift &&
                canonicalize(d->operand(0), bb) == cbl) {
                return true;
            }
        }
        if (cbl && cbl->is_instruction()) {
            const Value* idx = nullptr;
            int64_t s = 0;
            if (match_shifted(cbl, bb, idx, s) && s == shift && idx == cn) {
                return true;
            }
        }
        for (const Fact& f : facts_) {
            // Check if fact implies cn <= (cbl >> shift) or (cn << shift) <= cbl
            if (f.rhs && f.rhs->is_instruction()) {
                const Instruction* rd = f.rhs->defining_instruction();
                int64_t sc = 0;
                if ((rd->opcode() == Opcode::lshr || rd->opcode() == Opcode::ashr) &&
                    get_const_int(rd->operand(1), sc) && sc == shift &&
                    canonicalize(rd->operand(0), bb) == cbl) {
                    if (implied_by(f, Opcode::ule, cn, f.rhs, true) == true) return true;
                }
            }
            const Value* sh_idx = nullptr;
            int64_t s = 0;
            if (match_shifted(f.lhs, bb, sh_idx, s) && s == shift && sh_idx == cn) {
                if (implied_by(f, Opcode::ule, f.lhs, cbl, true) == true) return true;
            }
        }
        return false;
    }

    std::optional<bool> check_implied(Opcode check_op, const Value* raw_x, const Value* raw_y, const BasicBlock* bb) const {
        const Value* x = canonicalize(raw_x, bb);
        const Value* y = canonicalize(raw_y, bb);
        const bool x_non_neg = ra_.get_range_at(raw_x, bb).min_val >= 0 || ra_.get_range_at(x, bb).min_val >= 0;
        const bool y_non_neg = ra_.get_range_at(raw_y, bb).min_val >= 0 || ra_.get_range_at(y, bb).min_val >= 0;
        const bool both_non_negative = x_non_neg && y_non_neg;

        int64_t yc = 0;
        if (get_const_int(raw_y, yc)) y = raw_y;

        // Direct fact lookup
        for (auto it = facts_.rbegin(); it != facts_.rend(); ++it) {
            int64_t f_rc = 0;
            Fact f = *it;
            if (get_const_int(raw_y, yc) && get_const_int(f.rhs, f_rc) && yc == f_rc) {
                f.rhs = y;
            }
            if (auto outcome = implied_by(f, check_op, x, y, both_non_negative)) {
                return outcome;
            }
        }

        auto check_pred = predicate_of(check_op);
        if (!check_pred) return std::nullopt;

        // Transitive inequality: x < mid <= y or x <= mid < y
        if (x_non_neg) {
            for (const Fact& f1 : facts_) {
                const Value* mid = nullptr;
                bool strict1 = false;
                if (f1.lhs == x && implied_by(f1, Opcode::ule, x, f1.rhs, true) == true) {
                    mid = f1.rhs;
                    strict1 = (implied_by(f1, Opcode::ult, x, f1.rhs, true) == true);
                } else if (f1.rhs == x && implied_by(f1, Opcode::ule, x, f1.lhs, true) == true) {
                    mid = f1.lhs;
                    strict1 = (implied_by(f1, Opcode::ult, x, f1.lhs, true) == true);
                }
                if (!mid) continue;
                for (const Fact& f2 : facts_) {
                    if (implied_by(f2, Opcode::ule, mid, y, true) == true) {
                        bool strict2 = (implied_by(f2, Opcode::ult, mid, y, true) == true);
                        uint8_t trans_accepts = (strict1 || strict2) ? kLess : (kLess | kEqual);
                        if ((trans_accepts & ~check_pred->accepts) == 0) return true;
                        if ((trans_accepts & check_pred->accepts) == 0) return false;
                    }
                }
            }
        }

        // Scaled/shifted bounds check: (idx << shift) < byte_len when 0 <= idx < N and (N << shift) <= byte_len
        const Value* idx = nullptr;
        int64_t shift = 0;
        if (match_shifted(raw_x, bb, idx, shift) && ra_.get_range_at(idx, bb).min_val >= 0) {
            for (const Fact& f : facts_) {
                const Value* bound_n = nullptr;
                if (f.lhs == idx && implied_by(f, Opcode::ult, idx, f.rhs, true) == true) {
                    bound_n = f.rhs;
                } else if (f.rhs == idx && implied_by(f, Opcode::ult, idx, f.lhs, true) == true) {
                    bound_n = f.lhs;
                }
                if (bound_n && is_length_bound(bound_n, y, shift, bb)) {
                    if (check_pred->order == Order::Signed && !y_non_neg) {
                        // If y is used in lshr(y, shift), y >= (1 << shift) > 0 unless y has sign bit set;
                        // check if bound_n was derived via ashr or y is non-negative.
                        const Value* cn = canonicalize(bound_n, bb);
                        bool signed_safe = false;
                        if (cn && cn->is_instruction() && cn->defining_instruction()->opcode() == Opcode::ashr) {
                            signed_safe = true;
                        } else if (cn && cn->is_instruction() && cn->defining_instruction()->opcode() == Opcode::lshr) {
                            // Standard TypedArray byte_length is non-negative; if loop bound used slt(i, N), N > 0
                            signed_safe = true;
                        }
                        if (!signed_safe) continue;
                    }
                    if ((kLess & ~check_pred->accepts) == 0) return true;
                    if ((kLess & check_pred->accepts) == 0) return false;
                }
            }
        }

        return std::nullopt;
    }

    void visit(BasicBlock* bb) {
        if (bb->predecessors().size() == 1) {
            const BasicBlock* pred = bb->predecessors()[0];
            const Instruction* term = pred ? pred->terminator() : nullptr;
            if (term && term->opcode() == Opcode::br_if && term->true_target().block != term->false_target().block) {
                collect_edge_facts(term->operand(0), term->true_target().block == bb, bb);
            }
        }

        fold_checks_in(bb);
    }

    void fold_checks_in(BasicBlock* bb) {
        for (Instruction* cur = bb->head(); cur; cur = cur->next()) {
            if (cur->opcode() == Opcode::load && cur->offset() >= 0 && cur->offset() <= 64) {
                dom_loads_.push_back(cur);
            } else if (cur->produces_value() &&
                       (cur->opcode() == Opcode::lshr || cur->opcode() == Opcode::ashr ||
                        cur->opcode() == Opcode::shl || cur->opcode() == Opcode::add ||
                        cur->opcode() == Opcode::sub || cur->opcode() == Opcode::mul ||
                        cur->opcode() == Opcode::and_)) {
                const Value* c0 = cur->operand_count() >= 1 ? canonicalize(cur->operand(0), bb) : nullptr;
                const Value* c1 = cur->operand_count() >= 2 ? canonicalize(cur->operand(1), bb) : nullptr;
                dom_exprs_.push_back({cur->opcode(), c0, c1, cur->imm_i64(), cur->result()});
            } else if (cur->opcode() == Opcode::guard) {
                collect_edge_facts(cur->operand(0), true, bb);
                continue;
            }

            if (facts_.empty()) continue;
            const Instruction* cmp = as_int_comparison(cur->result());
            if (!cmp) continue;
            std::optional<bool> outcome = check_implied(cmp->opcode(), cmp->operand(0), cmp->operand(1), bb);
            if (!outcome) continue;
            Builder b(mod_);
            b.set_function(&fn_);
            b.position_before(cur);
            replace_all_uses(fn_, cur->result(), b.build_iconst_i32(*outcome ? 1 : 0));
            if (opts_.stats) opts_.stats->checks_implied++;
            changed_ = true;
        }
    }

    Function& fn_;
    Module& mod_;
    const DominatorTree& dom_;
    const LoopAnalysis& loops_;
    const RangeAnalysis& ra_;
    const AliasAnalysis& aa_;
    const RangeAnalysisOptions& opts_;
    std::vector<Fact> facts_;
    std::vector<const Instruction*> dom_loads_;
    std::vector<DomExpr> dom_exprs_;
    bool changed_ = false;
};

// ---------------------------------------------------------------------------
// Comparisons, guards and branches whose outcome value ranges decide.

bool eliminate_range_decided_checks(
    Function& fn,
    Module& mod,
    const RangeAnalysis& ra,
    const RangeAnalysisOptions& opts,
    bool& branches_folded
) {
    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();

            if (as_int_comparison(cur->result()) && is_int_value(cur->operand(0)) && is_int_value(cur->operand(1))) {
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
                        branches_folded = true;
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
                    branches_folded = true;
                    changed = true;
                }
            }

            cur = next;
        }
    }

    return changed;
}

bool fold_boolean_and_branches(Function& fn, Module& mod, const RangeAnalysisOptions& opts, bool& branches_folded) {
    bool changed = false;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* cur = bb->head(); cur; cur = cur->next()) {
            if ((cur->opcode() == Opcode::and_ || cur->opcode() == Opcode::or_) && cur->operand_count() == 2 && cur->type() == Type::i32()) {
                int64_t c0 = 0, c1 = 0;
                bool has0 = get_const_int(cur->operand(0), c0);
                bool has1 = get_const_int(cur->operand(1), c1);
                Value* rep = nullptr;
                if (has0 && has1) {
                    Builder b(mod);
                    b.set_function(&fn);
                    b.position_before(cur);
                    int32_t res = (cur->opcode() == Opcode::and_) ? static_cast<int32_t>(c0 & c1) : static_cast<int32_t>(c0 | c1);
                    rep = b.build_iconst_i32(res);
                } else if (cur->opcode() == Opcode::and_) {
                    if (has0 && c0 == 1 && ImpliedCheckFolder::is_bool_expr(cur->operand(1))) rep = cur->operand(1);
                    else if (has1 && c1 == 1 && ImpliedCheckFolder::is_bool_expr(cur->operand(0))) rep = cur->operand(0);
                    else if ((has0 && c0 == 0) || (has1 && c1 == 0)) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        rep = b.build_iconst_i32(0);
                    }
                } else if (cur->opcode() == Opcode::or_) {
                    if (has0 && c0 == 0) rep = cur->operand(1);
                    else if (has1 && c1 == 0) rep = cur->operand(0);
                    else if ((has0 && c0 == 1 && ImpliedCheckFolder::is_bool_expr(cur->operand(1))) ||
                             (has1 && c1 == 1 && ImpliedCheckFolder::is_bool_expr(cur->operand(0)))) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        rep = b.build_iconst_i32(1);
                    }
                }
                if (rep && rep->type() == cur->type()) {
                    replace_all_uses(fn, cur->result(), rep);
                    changed = true;
                }
            } else if (cur->opcode() == Opcode::br_if) {
                int64_t c = 0;
                if (get_const_int(cur->operand(0), c)) {
                    fold_branch(cur, c != 0);
                    if (opts.stats) opts.stats->branches_folded++;
                    branches_folded = true;
                    changed = true;
                }
            }
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

    bool any_changed = false;
    constexpr size_t kMaxRounds = 4;
    for (size_t round = 0; round < kMaxRounds; ++round) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);
        RangeAnalysis ra(fn, dom, loops);
        AliasAnalysis aa(fn);

        bool round_changed = false;
        bool branches_folded = false;
        if (opts.enable_implied_checks) {
            round_changed |= ImpliedCheckFolder(fn, mod, dom, loops, ra, aa, opts).run();
        }
        if (opts.enable_bce) {
            round_changed |= eliminate_range_decided_checks(fn, mod, ra, opts, branches_folded);
        }
        if (round_changed) {
            round_changed |= fold_boolean_and_branches(fn, mod, opts, branches_folded);
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            cfg_simplify_function(fn);
            fn.rebuild_cfg_predecessors();
            remove_dead_pure_instructions(fn);
            fn.rebuild_cfg_predecessors();
        }
        if (!branches_folded) break;
    }

    if (opts.enable_bce) {
        LoopUnswitchOptions uo;
        for (size_t round = 0; round < kMaxRounds; ++round) {
            fn.rebuild_cfg_predecessors();
            DominatorTree dom(fn);
            LoopAnalysis loops(fn, dom);
            bool unswitched = false;
            for (LoopInfo* loop : loops.post_order_loops()) {
                if (loop && unswitch_fast_slow_loop(fn, *loop, dom, uo)) {
                    unswitched = true;
                    any_changed = true;
                    break;
                }
            }
            if (!unswitched) break;
        }
        if (any_changed) {
            fn.rebuild_cfg_predecessors();
            remove_dead_pure_instructions(fn);
            fn.rebuild_cfg_predecessors();
        }
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
