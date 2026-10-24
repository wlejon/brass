#include "range_check_version.hpp"
#include "ir_clone.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Loop predication for range checks: an inner counted loop
//
//     for (iv = init; iv < lim; iv++) { ... if (idx ult len) <in bounds> ... }
//
// whose checks' indices are affine in iv (idx = a*iv + k, a and k and len
// loop-invariant) passes every check on every iteration exactly when each
// check passes at iv = init and iv = lim - 1: an affine function's extremes
// over an interval are at its ends, and an index from 0 to below len at both
// ends is from 0 to below len between them. So the preheader tests the ends
// once and branches to a copy of the loop with those checks taken as passed,
// or to the loop as it is, which still checks each one. That is what lets a
// loop over a typed array whose length or extent is only known at run time
// (a stencil, a matrix product with its sizes passed in) run without its
// bounds checks, and then vectorize and strength-reduce as a loop with known
// sizes does.
//
// The preheader computes the ends in wrapping i64 arithmetic, so it also tests
// that no term is large enough for that to wrap: each invariant term within
// +-2^52 (a coefficient is at most 64 and there are at most eight terms), an
// invariant scale of iv within +-2^31, and init and lim within +-2^31. Then
// the ends it computes are the exact integers, and the loop's own index
// arithmetic, wrapping or not, computes the same values modulo 2^64, which
// are exact because they lie in [0, len).

namespace brass {

namespace {

constexpr int64_t kMaxCoef = 64;
constexpr size_t kMaxTerms = 8;
constexpr int64_t kMaxConst = int64_t{1} << 40;
constexpr int64_t kTermBound = int64_t{1} << 52;
constexpr int64_t kIvBound = int64_t{1} << 31;
constexpr size_t kMaxLoopInstructions = 2000;
constexpr const char* kCopySuffix = "_rcv";

bool const_i64(const Value* v, int64_t& out) {
    if (!v || !v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d) return false;
    if (d->opcode() == Opcode::iconst_i64) {
        out = d->imm_i64();
        return true;
    }
    return false;
}

// a*iv + s*iv + sum(coef * term) + c, every value an i64.
struct Affine {
    int64_t iv_coef = 0;
    Value* iv_scale = nullptr;  // an invariant multiplying iv, coefficient 1
    std::vector<std::pair<Value*, int64_t>> terms;
    int64_t c = 0;

    bool add_term(Value* v, int64_t coef) {
        for (auto& t : terms) {
            if (t.first == v) {
                t.second += coef;
                return std::llabs(t.second) <= kMaxCoef;
            }
        }
        terms.emplace_back(v, coef);
        return terms.size() <= kMaxTerms;
    }
    bool plus(const Affine& o, int64_t sign) {
        if (o.iv_scale) {
            if (iv_scale || sign != 1) return false;
            iv_scale = o.iv_scale;
        }
        iv_coef += sign * o.iv_coef;
        for (const auto& t : o.terms) {
            if (!add_term(t.first, sign * t.second)) return false;
        }
        c += sign * o.c;
        return std::llabs(iv_coef) <= kMaxCoef && std::llabs(c) <= kMaxConst;
    }
    bool scale(int64_t k) {
        if (iv_scale && k != 1) return false;
        iv_coef *= k;
        for (auto& t : terms) {
            t.second *= k;
            if (std::llabs(t.second) > kMaxCoef) return false;
        }
        c *= k;
        return std::llabs(iv_coef) <= kMaxCoef && std::llabs(c) <= kMaxConst;
    }
    bool is_iv() const { return iv_coef == 1 && !iv_scale && terms.empty() && c == 0; }
    bool is_invariant() const { return iv_coef == 0 && !iv_scale; }
};

class Matcher {
public:
    Matcher(const LoopInfo& loop, const Value* iv) : loop_(loop), iv_(iv) {}

    bool match(Value* v, Affine& out, int depth = 0) const {
        out = Affine{};
        if (!v || v->type() != Type::i64() || depth > 12) return false;
        if (v == iv_) {
            out.iv_coef = 1;
            return true;
        }
        int64_t k = 0;
        if (const_i64(v, k)) {
            if (std::llabs(k) > kMaxConst) return false;
            out.c = k;
            return true;
        }
        if (loop_.is_loop_invariant(v)) return out.add_term(v, 1);
        if (!v->is_instruction()) return false;
        const Instruction* d = v->defining_instruction();
        if (!d || d->operand_count() != 2) return false;
        Affine a, b;
        switch (d->opcode()) {
            case Opcode::add:
            case Opcode::sub:
                if (!match(d->operand(0), a, depth + 1) || !match(d->operand(1), b, depth + 1)) return false;
                out = a;
                return out.plus(b, d->opcode() == Opcode::add ? 1 : -1);
            case Opcode::mul: {
                for (int i = 0; i < 2; ++i) {
                    Value* other = d->operand(1 - i);
                    if (const_i64(d->operand(i), k)) {
                        if (std::llabs(k) > kMaxCoef || !match(other, out, depth + 1)) return false;
                        return out.scale(k);
                    }
                }
                for (int i = 0; i < 2; ++i) {
                    Value* s = d->operand(i);
                    if (!loop_.is_loop_invariant(s) || s->type() != Type::i64()) continue;
                    if (!match(d->operand(1 - i), a, depth + 1)) return false;
                    if (a.is_iv()) {
                        out.iv_scale = s;
                        return true;
                    }
                    return false;
                }
                return false;
            }
            case Opcode::shl:
                if (!const_i64(d->operand(1), k) || k < 0 || k > 6 || !match(d->operand(0), out, depth + 1)) {
                    return false;
                }
                return out.scale(int64_t{1} << k);
            default:
                return false;
        }
    }

private:
    const LoopInfo& loop_;
    const Value* iv_;
};

struct Check {
    Instruction* term;  // br_if on `ult idx, len`, in bounds on its true edge
    Affine idx;
    Value* len;
};

// A loop value used past the loop, under `exit`, and the exit's parameter
// that carries it there.
struct Escape {
    BasicBlock* exit;
    Value* value;
    Value* param;
};

struct Counted {
    size_t iv_idx = 0;
    Value* iv = nullptr;
    Value* init = nullptr;
    Value* last = nullptr;  // lim, the last value being last - 1 (slt) or last (sle)
    bool inclusive = false;
};

// `for (iv = init; iv < lim (or <=); iv += 1)`, lim loop-invariant.
bool counted_loop(const LoopInfo& loop, BasicBlock* preheader, Counted& out) {
    BasicBlock* header = loop.header();
    if (loop.latches().size() != 1) return false;
    const Instruction* ph_term = preheader->terminator();
    const Instruction* latch_term = loop.latches()[0]->terminator();
    const Instruction* hdr_term = header->terminator();
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;
    if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header) return false;
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return false;
    if (!loop.contains(hdr_term->true_target().block) || loop.contains(hdr_term->false_target().block)) return false;
    const Value* cond = hdr_term->operand(0);
    if (!cond || !cond->is_instruction()) return false;
    const Instruction* cmp = cond->defining_instruction();
    if (!cmp || (cmp->opcode() != Opcode::slt && cmp->opcode() != Opcode::sle)) return false;
    Value* iv = cmp->operand(0);
    Value* lim = cmp->operand(1);
    if (!iv || iv->type() != Type::i64() || !iv->is_block_param() || iv->defining_block() != header) return false;
    if (!lim || lim->type() != Type::i64() || !loop.is_loop_invariant(lim)) return false;
    int64_t k = 0;
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (header->param(i) != iv) continue;
        const Value* next = latch_term->branch_target().args[i];
        const Instruction* step = next && next->is_instruction() ? next->defining_instruction() : nullptr;
        if (!step || step->opcode() != Opcode::add) return false;
        const Value* one = step->operand(0) == iv ? step->operand(1) : step->operand(1) == iv ? step->operand(0) : nullptr;
        if (!const_i64(one, k) || k != 1) return false;
        out.iv_idx = i;
        out.iv = iv;
        out.init = ph_term->branch_target().args[i];
        out.last = lim;
        out.inclusive = cmp->opcode() == Opcode::sle;
        return true;
    }
    return false;
}

// A value the preheader can use: a constant is built again there (a constant
// defined in the loop counts as invariant but does not dominate it).
Value* at_preheader(Builder& b, Value* v) {
    int64_t k = 0;
    if (const_i64(v, k)) return b.build_iconst_i64(k);
    return v;
}

// |v| <= bound, as (v + bound) ule 2 * bound.
Value* within(Builder& b, Value* v, int64_t bound) {
    return b.build_ule(b.build_add(v, b.build_iconst_i64(bound)), b.build_iconst_i64(2 * bound));
}

Value* affine_at(Builder& b, const Affine& a, Value* x) {
    Value* sum = b.build_iconst_i64(a.c);
    for (const auto& [t, coef] : a.terms) {
        Value* tv = at_preheader(b, t);
        sum = b.build_add(sum, coef == 1 ? tv : b.build_mul(tv, b.build_iconst_i64(coef)));
    }
    if (a.iv_coef != 0) sum = b.build_add(sum, b.build_mul(x, b.build_iconst_i64(a.iv_coef)));
    if (a.iv_scale) sum = b.build_add(sum, b.build_mul(x, at_preheader(b, a.iv_scale)));
    return sum;
}

bool version_loop(Function& fn, LoopInfo& loop) {
    BasicBlock* header = loop.header();
    if (!header || header->name().find(kCopySuffix) != std::string_view::npos) return false;
    if (!loop.sub_loops().empty()) return false;
    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;
    Counted cl;
    if (!counted_loop(loop, preheader, cl)) return false;

    size_t instructions = 0;
    for (const BasicBlock* bb : loop.blocks()) {
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (++instructions > kMaxLoopInstructions) return false;
            if (inst->opcode() == Opcode::invoke) return false;
        }
    }

    const Matcher m(loop, cl.iv);
    std::vector<Check> checks;
    for (BasicBlock* bb : loop.blocks()) {
        if (bb == header) continue;
        Instruction* term = bb->terminator();
        if (!term || term->opcode() != Opcode::br_if) continue;
        const Value* cond = term->operand(0);
        const Instruction* cmp = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
        if (!cmp || cmp->opcode() != Opcode::ult || cmp->operand_count() != 2) continue;
        Value* len = cmp->operand(1);
        if (!len || len->type() != Type::i64() || !loop.is_loop_invariant(len)) continue;
        Affine idx;
        if (!m.match(cmp->operand(0), idx) || idx.is_invariant()) continue;
        checks.push_back({term, idx, len});
    }
    if (checks.empty()) return false;

    // The loop's values used past it (a sum read after the loop): each must
    // be under one exit, whose predecessors are all in the loop, so the exit
    // can take it as a parameter from either copy.
    fn.rebuild_cfg_predecessors();
    const DominatorTree dom(fn);
    std::vector<BasicBlock*> exits;
    for (BasicBlock* bb : loop.blocks()) {
        Instruction* term = bb->terminator();
        if (!term) continue;
        for_each_edge(*term, [&](BranchTarget& bt) {
            if (bt.block && !loop.contains(bt.block) &&
                std::find(exits.begin(), exits.end(), bt.block) == exits.end()) {
                exits.push_back(bt.block);
            }
        });
    }
    auto defined_in_loop = [&](const Value* v) {
        if (!v) return false;
        if (v->is_block_param()) return loop.contains(v->defining_block());
        return v->is_instruction() && v->defining_instruction() && loop.contains(v->defining_instruction()->parent());
    };
    std::vector<Escape> escapes;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || loop.contains(bb)) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            bool ok_use = true;
            for_each_use(*inst, [&](Value* v) {
                if (!ok_use || !defined_in_loop(v)) return;
                BasicBlock* under = nullptr;
                for (BasicBlock* e : exits) {
                    if (dom.dominates(e, bb)) under = e;
                }
                if (!under) {
                    ok_use = false;
                    return;
                }
                for (const Escape& x : escapes) {
                    if (x.exit == under && x.value == v) return;
                }
                escapes.push_back({under, v, nullptr});
            });
            if (!ok_use) return false;
        }
    }
    for (const Escape& x : escapes) {
        for (const BasicBlock* p : x.exit->predecessors()) {
            if (!loop.contains(p)) return false;
        }
    }

    // The condition, before the preheader's branch.
    Instruction* ph_term = preheader->terminator();
    Builder b(*fn.parent());
    b.set_function(&fn);
    b.position_before(ph_term);
    Value* init = at_preheader(b, cl.init);
    Value* lim = at_preheader(b, cl.last);
    Value* hi = cl.inclusive ? lim : b.build_sub(lim, b.build_iconst_i64(1));
    Value* ok = b.build_and(within(b, init, kIvBound), within(b, lim, kIvBound));
    std::unordered_set<const Value*> bounded;
    for (const Check& c : checks) {
        for (const auto& t : c.idx.terms) {
            if (bounded.insert(t.first).second) ok = b.build_and(ok, within(b, at_preheader(b, t.first), kTermBound));
        }
        if (c.idx.iv_scale && bounded.insert(c.idx.iv_scale).second) {
            ok = b.build_and(ok, within(b, at_preheader(b, c.idx.iv_scale), kIvBound));
        }
        Value* len = at_preheader(b, c.len);
        for (Value* x : {init, hi}) {
            Value* e = affine_at(b, c.idx, x);
            ok = b.build_and(ok, b.build_and(b.build_sge(e, b.build_iconst_i64(0)), b.build_ult(e, len)));
        }
    }

    // The copy, its checks taken as passed.
    std::vector<BasicBlock*> ordered;
    for (BasicBlock* bb : fn.blocks()) {
        if (bb && loop.contains(bb)) ordered.push_back(bb);
    }
    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : ordered) {
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + kCopySuffix);
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }
    std::vector<std::pair<const Instruction*, Instruction*>> pairs;
    std::unordered_set<const Instruction*> checked;
    for (const Check& c : checks) checked.insert(c.term);
    std::vector<Instruction*> to_fold;
    for (BasicBlock* bb : ordered) {
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            blocks[bb]->append_instruction(shell);
            pairs.emplace_back(inst, shell);
            if (checked.count(inst)) to_fold.push_back(shell);
        }
    }
    for (auto& [orig, copy] : pairs) ir::clone_uses(*orig, *copy, values, blocks);
    for (Instruction* term : to_fold) {
        const BranchTarget kept = term->true_target();
        b.position_before(term);
        b.build_br(kept.block, Span<Value* const>(kept.args.data(), kept.args.size()));
        term->parent()->remove_instruction(term);
    }

    // Each escaping value reaches its exit as a parameter, from the loop or
    // from the copy, and its uses past the exit read that.
    std::vector<std::pair<BasicBlock*, size_t>> exit_params;  // exit, its parameter count before
    for (Escape& x : escapes) {
        auto known = std::find_if(exit_params.begin(), exit_params.end(),
                                  [&](const auto& e) { return e.first == x.exit; });
        if (known == exit_params.end()) exit_params.emplace_back(x.exit, x.exit->param_count());
        x.param = ir::new_block_param(fn, x.exit, x.value->type());
        auto mapped = values.find(x.value);
        Value* copied = mapped != values.end() ? mapped->second : x.value;
        for (BasicBlock* bb : ordered) {
            for (BasicBlock* from : {bb, blocks[bb]}) {
                for_each_edge(*from->terminator(), [&](BranchTarget& bt) {
                    if (bt.block == x.exit) bt.args.push_back(from == bb ? x.value : copied);
                });
            }
        }
    }
    std::unordered_set<const BasicBlock*> copies;
    for (const auto& [orig, copy] : blocks) copies.insert(copy);
    std::vector<std::pair<Value**, Value*>> rewritten;  // slot, what it held
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || loop.contains(bb) || copies.count(bb)) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for_each_use_slot(*inst, [&](Value*& slot) {
                for (const Escape& x : escapes) {
                    if (slot == x.value && dom.dominates(x.exit, bb)) {
                        rewritten.emplace_back(&slot, slot);
                        slot = x.param;
                        return;
                    }
                }
            });
        }
    }

    const std::vector<Value*> args = ph_term->branch_target().args;
    b.position_before(ph_term);
    Instruction* choose = b.build_br_if(ok, blocks[header], Span<Value* const>(args.data(), args.size()), header,
                                        Span<Value* const>(args.data(), args.size()));
    preheader->remove_instruction(ph_term);
    fn.rebuild_cfg_predecessors();
    if (verify_function(fn)) return true;

    // A shape this did not foresee: take the copy back out (the condition's
    // instructions are dead).
    for (auto& [slot, held] : rewritten) *slot = held;
    for (const auto& [exit, count] : exit_params) {
        for (BasicBlock* bb : ordered) {
            for_each_edge(*bb->terminator(), [&](BranchTarget& bt) {
                if (bt.block == exit) bt.args.resize(count);
            });
        }
        exit->params().resize(count);
    }
    b.position_before(choose);
    b.build_br(header, Span<Value* const>(args.data(), args.size()));
    preheader->remove_instruction(choose);
    for (BasicBlock* bb : ordered) fn.remove_block(blocks[bb]);
    fn.rebuild_cfg_predecessors();
    return false;
}

} // namespace

bool version_loops_on_range_checks(Function& fn) {
    if (!fn.parent() || fn.blocks().empty()) return false;
    std::unordered_set<const BasicBlock*> tried;
    bool changed = false;
    for (int round = 0; round < 16; ++round) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);
        bool any = false;
        for (LoopInfo* loop : loops.post_order_loops()) {
            if (!loop || !tried.insert(loop->header()).second) continue;
            if (version_loop(fn, *loop)) {
                any = changed = true;
                break;
            }
        }
        if (!any) break;
    }
    return changed;
}

} // namespace brass
