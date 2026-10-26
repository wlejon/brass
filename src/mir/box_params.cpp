#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// unbox_double_params: an i64 block parameter that only ever holds a
// NaN-boxed double (number_bits.hpp), carried round a loop as boxes and
// unboxed where it is read, becomes an f64 parameter.
namespace brass {

namespace {

bool const_i64(const Value* v, int64_t& out) {
    if (!v || !v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d || d->opcode() != Opcode::iconst_i64) return false;
    out = d->imm_i64();
    return true;
}

bool is_nan_bits(uint64_t bits) {
    return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL && (bits & 0x000FFFFFFFFFFFFFULL) != 0;
}

// `select(ne.f64(f, f), nan, bitcast(f))`: the boxed double f, with `nan`
// the bits a NaN boxes as.
Value* boxed_double(const Value* v, uint64_t& nan) {
    if (!v || !v->is_instruction()) return nullptr;
    const Instruction* sel = v->defining_instruction();
    if (!sel || sel->opcode() != Opcode::select || sel->operand_count() != 3) return nullptr;
    int64_t k = 0;
    if (!const_i64(sel->operand(1), k) || !is_nan_bits(static_cast<uint64_t>(k))) return nullptr;
    const Value* bits = sel->operand(2);
    const Instruction* bc = bits && bits->is_instruction() ? bits->defining_instruction() : nullptr;
    if (!bc || bc->opcode() != Opcode::bitcast_i64_f64) return nullptr;
    Value* f = bc->operand(0);
    const Value* c = sel->operand(0);
    const Instruction* ne = c && c->is_instruction() ? c->defining_instruction() : nullptr;
    if (!ne || ne->opcode() != Opcode::ne || ne->operand(0) != f || ne->operand(1) != f) return nullptr;
    nan = static_cast<uint64_t>(k);
    return f;
}

bool is_unbox(const Instruction& inst, const Value* p) {
    return inst.opcode() == Opcode::bitcast_f64_i64 && inst.operand_count() == 1 && inst.operand(0) == p;
}

struct Candidate {
    Value* param = nullptr;
    // The argument each edge into the block gives it, with that edge.
    std::vector<std::pair<Instruction*, BranchTarget*>> edges;
    Value* unboxed = nullptr;
};

} // namespace

bool unbox_double_params(Function& fn) {
    if (!fn.parent()) return false;
    std::unordered_set<const BasicBlock*> resume_targets;
    for (const auto& rp : fn.resume_points()) {
        if (rp.second) resume_targets.insert(rp.second);
    }

    // The parameters an unbox reads.
    std::unordered_set<const Value*> unboxed;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::bitcast_f64_i64 && inst->operand_count() == 1 &&
                inst->operand(0) && inst->operand(0)->is_block_param()) {
                unboxed.insert(inst->operand(0));
            }
        }
    }
    if (unboxed.empty()) return false;

    std::unordered_map<const Value*, Candidate> cands;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || bb == fn.entry_block() || resume_targets.count(bb)) continue;
        for (Value* p : bb->params()) {
            if (p && p->type() == Type::i64() && unboxed.count(p)) cands[p].param = p;
        }
    }
    // Every edge into a candidate's block, from any terminator; an invoke's
    // targets take the call's results, so its blocks are left alone.
    for (BasicBlock* bb : fn.blocks()) {
        Instruction* term = bb ? bb->terminator() : nullptr;
        if (!term) continue;
        for_each_edge(*term, [&](BranchTarget& bt) {
            for (Value* p : bt.block->params()) {
                auto it = cands.find(p);
                if (it == cands.end()) continue;
                if (term->opcode() == Opcode::invoke || p->param_index() >= bt.args.size()) {
                    it->second.param = nullptr;
                } else {
                    it->second.edges.push_back({term, &bt});
                }
            }
        });
    }

    // One NaN for the function's boxes, so a box made again is the box.
    bool have_nan = false;
    uint64_t nan = 0;
    auto arg_ok = [&](const Value* a) {
        uint64_t k = 0;
        if (boxed_double(a, k)) {
            if (!have_nan) {
                have_nan = true;
                nan = k;
            }
            return k == nan;
        }
        int64_t c = 0;
        if (const_i64(a, c)) return !is_nan_bits(static_cast<uint64_t>(c)) || (have_nan && static_cast<uint64_t>(c) == nan);
        auto it = cands.find(a);
        return it != cands.end() && it->second.param;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& [p, c] : cands) {
            if (!c.param) continue;
            for (auto& [term, bt] : c.edges) {
                if (!arg_ok(bt->args[p->param_index()])) {
                    c.param = nullptr;
                    changed = true;
                    break;
                }
            }
        }
    }
    std::vector<Candidate*> work;
    for (auto& [p, c] : cands) {
        if (c.param) work.push_back(&c);
    }
    if (work.empty()) return false;
    if (!have_nan) nan = 0x7FF8000000000000ULL;
    // A fixed order (the parameters' blocks and positions), so every edge
    // gets its new arguments in the order its block gets the parameters.
    std::sort(work.begin(), work.end(), [](const Candidate* a, const Candidate* b) {
        const uint32_t ba = a->param->defining_block()->id(), bb = b->param->defining_block()->id();
        return ba != bb ? ba < bb : a->param->param_index() < b->param->param_index();
    });

    Builder b(*fn.parent());
    b.set_function(&fn);
    for (Candidate* c : work) c->unboxed = b.add_block_param(c->param->defining_block(), Type::f64());
    for (Candidate* c : work) {
        const size_t idx = c->param->param_index();
        for (auto& [term, bt] : c->edges) {
            Value* a = bt->args[idx];
            uint64_t k = 0;
            Value* f = boxed_double(a, k);
            int64_t bits = 0;
            if (!f && const_i64(a, bits)) {
                double d = 0;
                std::memcpy(&d, &bits, sizeof d);
                b.position_before(term);
                f = b.build_fconst_f64(d);
            }
            if (!f) f = cands.at(a).unboxed;
            bt->args.push_back(f);
        }
    }
    // The unboxes read the f64 parameter; anything else the boxed value,
    // boxed again where the block starts. The i64 parameters go.
    for (Candidate* c : work) {
        Value* p = c->param;
        BasicBlock* bb = p->defining_block();
        std::vector<Instruction*> unboxes;
        for (BasicBlock* ub : fn.blocks()) {
            if (!ub) continue;
            for (Instruction* inst : *ub) {
                if (inst && is_unbox(*inst, p)) unboxes.push_back(inst);
            }
        }
        for (Instruction* u : unboxes) {
            replace_all_uses(fn, u->result(), c->unboxed);
            u->parent()->remove_instruction(u);
        }
        if (has_uses(fn, p)) {
            if (bb->head()) b.position_before(bb->head());
            else b.position_at_end(bb);
            Value* q = c->unboxed;
            Value* rebox = b.build_select(b.build_ne(q, q), b.build_iconst_i64(static_cast<int64_t>(nan)),
                                          b.build_bitcast_i64_f64(q));
            replace_all_uses(fn, p, rebox);
        }
    }
    // Highest index first, so the indices still to remove stay put.
    std::sort(work.begin(), work.end(), [](const Candidate* a, const Candidate* b) {
        return a->param->param_index() > b->param->param_index();
    });
    for (Candidate* c : work) remove_block_param(*c->param->defining_block(), c->param->param_index());
    return true;
}

} // namespace brass
