#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/uses.hpp>
#include "ir_clone.hpp"
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brass {

namespace {

// A basic induction variable: header parameter `param`, `init` on entry,
// `param +/- step` along the latch. Only i64 variables qualify: the scaled
// variable is i64, and i64 wrap-around is what makes `s*i` (scaled step by
// step) equal `s*i` (scaled at once) modulo 2^64.
struct BasicIV {
    Value* param = nullptr;
    Value* init = nullptr;
    Value* step = nullptr;
    bool is_sub = false;
    Instruction* update = nullptr;  // the latch's `param +/- step`
};

// Where a value is read: the instruction, and the operand index, or -1 for
// a deopt state map or an edge argument.
using UseList = std::vector<std::pair<Instruction*, int>>;

std::unordered_map<const Value*, UseList> collect_uses(Function& fn) {
    std::unordered_map<const Value*, UseList> uses;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            const auto& ops = inst->operands();
            for (size_t i = 0; i < ops.size(); ++i) {
                if (ops[i]) uses[ops[i]].emplace_back(inst, static_cast<int>(i));
            }
            for (Value* v : inst->state_map()) {
                if (v) uses[v].emplace_back(inst, -1);
            }
            for_each_edge(*inst, [&](BranchTarget& bt) {
                for (Value* v : bt.args) {
                    if (v) uses[v].emplace_back(inst, -1);
                }
            });
        }
    }
    return uses;
}

bool is_index_slot(const Instruction* inst, int operand) {
    return operand == 1 && (inst->opcode() == Opcode::load_indexed || inst->opcode() == Opcode::store_indexed);
}

struct PairHash {
    size_t operator()(const std::pair<const Value*, uint8_t>& p) const noexcept {
        return std::hash<const void*>()(p.first) ^ (static_cast<size_t>(p.second) << 3);
    }
};

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

Value* build_mul(Builder& b, Value* lhs, int64_t scale) {
    int64_t c = 0;
    if (get_const_int(lhs, c)) {
        // Wrapping multiply, as the i64 `mul` it replaces.
        return b.build_iconst_i64(static_cast<int64_t>(static_cast<uint64_t>(c) * static_cast<uint64_t>(scale)));
    }
    if (scale == 1) return lhs;
    return b.build_mul(lhs, b.build_iconst_i64(scale));
}

Value* build_add(Builder& b, Value* lhs, Value* rhs) {
    int64_t c = 0;
    if (get_const_int(rhs, c) && c == 0) return lhs;
    if (get_const_int(lhs, c) && c == 0 && lhs->type() == rhs->type()) return rhs;
    return b.build_add(lhs, rhs);
}

// The one edge of `term` into `target`, or null when there is none or more
// than one (a second edge would miss the argument a new parameter needs).
BranchTarget* sole_edge_to(Instruction* term, const BasicBlock* target) {
    BranchTarget* found = nullptr;
    size_t count = 0;
    for_each_edge(*term, [&](BranchTarget& bt) {
        if (bt.block == target) {
            found = &bt;
            ++count;
        }
    });
    return count == 1 ? found : nullptr;
}

bool is_i64(const Value* v) { return v && v->type() == Type::i64(); }

// The loop's dedicated preheader when it already has one: the header's only
// outside predecessor, ending in an unconditional `br`. Found without
// creating one, so a loop IVSR leaves alone is left unchanged.
BasicBlock* existing_preheader(const LoopInfo& loop) {
    if (loop.preheader()) return loop.preheader();
    BasicBlock* found = nullptr;
    for (BasicBlock* pred : loop.header()->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (found && found != pred) return nullptr;
        found = pred;
    }
    if (!found || !found->terminator() || found->terminator()->opcode() != Opcode::br) return nullptr;
    return found;
}

// True when, with constant bounds, every value the header compare
// `cmp(i, limit)` sees satisfies 0 <= i and s*i <= INT64_MAX, and s*limit
// fits too: then comparing s*i with s*limit orders exactly as comparing i
// with limit, signed or unsigned. The loop must continue on the compare's
// true edge, so i stops growing once the compare fails.
bool scaled_exit_compare_is_exact(const BasicIV& biv, const Instruction& cmp, Value* limit,
                                  const LoopInfo& loop, const Instruction& header_term, int64_t scale) {
    const Opcode op = cmp.opcode();
    if (op != Opcode::slt && op != Opcode::sle && op != Opcode::ult && op != Opcode::ule) return false;
    if (biv.is_sub) return false;
    int64_t init = 0, step = 0, lim = 0;
    if (!get_const_int(biv.init, init) || !get_const_int(biv.step, step) || !get_const_int(limit, lim)) return false;
    if (init < 0 || step <= 0 || lim < 0) return false;
    if (!loop.contains(header_term.true_target().block) || loop.contains(header_term.false_target().block)) return false;
    // The last value tested is below lim + step (strict) or at most lim + step.
    if (lim > std::numeric_limits<int64_t>::max() - step) return false;
    const int64_t bound = std::max(init, lim + step);
    return bound <= std::numeric_limits<int64_t>::max() / scale;
}

bool ivsr_loop(Function& fn, LoopInfo& loop) {
    BasicBlock* header = loop.header();
    if (!header) return false;
    BasicBlock* preheader = existing_preheader(loop);
    if (!preheader || loop.latches().size() != 1) return false;
    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;
    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    if (!ph_term || !latch_term) return false;
    if (ph_term->opcode() != Opcode::br && ph_term->opcode() != Opcode::br_if) return false;
    if (latch_term->opcode() != Opcode::br && latch_term->opcode() != Opcode::br_if) return false;

    BranchTarget* ph_bt = sole_edge_to(ph_term, header);
    BranchTarget* latch_bt = sole_edge_to(latch_term, header);
    if (!ph_bt || !latch_bt) return false;
    if (ph_bt->args.size() != header->param_count() || latch_bt->args.size() != header->param_count()) return false;

    std::vector<BasicIV> bivs;
    std::unordered_map<const Value*, size_t> param_to_biv;
    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (!is_i64(param) || !is_i64(ph_bt->args[i])) continue;
        Value* latch_arg = latch_bt->args[i];
        if (!latch_arg || !latch_arg->is_instruction()) continue;
        Instruction* def = latch_arg->defining_instruction();
        if (!def || def->type() != Type::i64()) continue;
        BasicIV biv{param, ph_bt->args[i], nullptr, false, def};
        if (def->opcode() == Opcode::add) {
            if (def->operand(0) == param && loop.is_loop_invariant(def->operand(1))) biv.step = def->operand(1);
            else if (def->operand(1) == param && loop.is_loop_invariant(def->operand(0))) biv.step = def->operand(0);
        } else if (def->opcode() == Opcode::sub && def->operand(0) == param && loop.is_loop_invariant(def->operand(1))) {
            biv.step = def->operand(1);
            biv.is_sub = true;
        }
        if (!is_i64(biv.step)) continue;
        param_to_biv[param] = bivs.size();
        bivs.push_back(biv);
    }
    if (bivs.empty()) return false;

    Builder b_ph(*fn.parent());
    b_ph.set_function(&fn);
    b_ph.position_before(ph_term);
    Builder b_latch(*fn.parent());
    b_latch.set_function(&fn);
    b_latch.position_before(latch_term);

    bool changed = false;
    std::unordered_map<std::pair<const Value*, uint8_t>, Value*, PairHash> biv_scaled_map;
    auto scaled_biv = [&](const BasicIV& biv, uint8_t scale) -> Value* {
        if (scale == 1) return biv.param;
        auto key = std::make_pair(static_cast<const Value*>(biv.param), scale);
        auto it = biv_scaled_map.find(key);
        if (it != biv_scaled_map.end()) return it->second;
        Value* init_bytes = build_mul(b_ph, biv.init, scale);
        Value* step_bytes = build_mul(b_ph, biv.step, scale);
        Value* scaled = ir::new_block_param(fn, header, Type::i64());
        ph_bt->args.push_back(init_bytes);
        latch_bt->args.push_back(biv.is_sub ? b_latch.build_sub(scaled, step_bytes) : b_latch.build_add(scaled, step_bytes));
        biv_scaled_map[key] = scaled;
        return scaled;
    };

    // The header's exit compare `cmp(iv, limit)` on a basic variable, when
    // it has one.
    Instruction* hdr_term = header->terminator();
    Instruction* exit_cmp = nullptr;
    if (hdr_term && hdr_term->opcode() == Opcode::br_if) {
        Value* cond_val = hdr_term->operand(0);
        Instruction* c = cond_val && cond_val->is_instruction() ? cond_val->defining_instruction() : nullptr;
        if (c && c->parent() == header && c->operand_count() == 2 && param_to_biv.count(c->operand(0)) &&
            is_i64(c->operand(1)) && loop.is_loop_invariant(c->operand(1))) {
            exit_cmp = c;
        }
    }

    // Whether the variable stays in a register across the loop whatever this
    // pass does with its addresses: something other than an indexed access
    // (or an add feeding only those), its own update, or an exit compare
    // this pass can move onto the variable scaled by `scale`, reads it - an
    // exit edge, a compare against a variable limit, arithmetic.
    std::unordered_map<const Value*, UseList> uses;
    bool uses_ready = false;
    auto stays_live = [&](const BasicIV& biv, uint8_t scale) -> bool {
        if (!uses_ready) {
            uses = collect_uses(fn);
            uses_ready = true;
        }
        auto only_indexes = [&](const Value* v) {
            for (const auto& [u, slot] : uses[v]) {
                if (!is_index_slot(u, slot) || !loop.contains(u->parent())) return false;
            }
            return true;
        };
        for (const auto& [u, slot] : uses[biv.param]) {
            if (u == biv.update) continue;
            if (!loop.contains(u->parent())) return true;
            if (is_index_slot(u, slot)) continue;
            if (u->opcode() == Opcode::add && u->type() == Type::i64() && (slot == 0 || slot == 1) &&
                loop.is_loop_invariant(u->operand(1 - slot)) && only_indexes(u->result())) {
                continue;
            }
            if (u == exit_cmp && slot == 0 &&
                scaled_exit_compare_is_exact(biv, *u, u->operand(1), loop, *hdr_term, scale)) {
                continue;
            }
            return true;
        }
        for (const auto& [u, slot] : uses[biv.update->result()]) {
            if (u != latch_term || slot != -1) return true;
        }
        return false;
    };

    // Pointer variables: `base + s*i` stepped by s*step, one per (base,
    // variable, scale). Used when the step is not a constant: the unroller's
    // copies then address `p + k*s*step` from one register, where the scaled
    // form needs `base + (si + k*s*step)`, an add per copy.
    struct PointerIV {
        const Value* base;
        const Value* param;
        uint8_t scale;
        Value* p;
    };
    std::vector<PointerIV> pointer_ivs;
    auto pointer_iv = [&](Value* base, const BasicIV& biv, uint8_t scale) -> Value* {
        for (const PointerIV& e : pointer_ivs) {
            if (e.base == base && e.param == biv.param && e.scale == scale) return e.p;
        }
        Value* init = build_add(b_ph, base, build_mul(b_ph, biv.init, scale));
        Value* step_bytes = build_mul(b_ph, biv.step, scale);
        Value* p = ir::new_block_param(fn, header, base->type());
        ph_bt->args.push_back(init);
        latch_bt->args.push_back(b_latch.build_add(p, step_bytes));
        pointer_ivs.push_back({base, biv.param, scale, p});
        return p;
    };

    enum class AccessMode { Index, Pointer, Scaled };
    struct Access {
        Instruction* inst;
        const BasicIV* biv;
        Value* inv_offset;
        AccessMode mode;
    };
    std::vector<Access> accesses;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || bb == preheader) continue;
        for (Instruction* inst : *bb) {
            if (!inst || (inst->opcode() != Opcode::load_indexed && inst->opcode() != Opcode::store_indexed)) continue;
            Value* base = inst->operand(0);
            Value* index = inst->operand(1);
            const uint8_t scale = inst->scale();
            if (!base || !index || !loop.is_loop_invariant(base)) continue;
            // base + offset would be a derived gcref live across the loop.
            if (!base->type().is_pointer() && base->type() != Type::i64()) continue;

            const BasicIV* matched = nullptr;
            Value* inv_offset = nullptr;
            if (auto p = param_to_biv.find(index); p != param_to_biv.end()) {
                if (scale <= 1) continue;
                matched = &bivs[p->second];
            } else if (index->is_instruction()) {
                Instruction* idx_def = index->defining_instruction();
                if (!idx_def || idx_def->opcode() != Opcode::add || idx_def->type() != Type::i64()) continue;
                Value* a = idx_def->operand(0);
                Value* c = idx_def->operand(1);
                if (param_to_biv.count(a) && loop.is_loop_invariant(c)) {
                    matched = &bivs[param_to_biv[a]];
                    inv_offset = c;
                } else if (param_to_biv.count(c) && loop.is_loop_invariant(a)) {
                    matched = &bivs[param_to_biv[c]];
                    inv_offset = a;
                }
            }
            if (!matched) continue;

            // How the access is rewritten, decided before any rewriting
            // changes the uses stays_live reads.
            //  - Index: the variable stays live anyway and its scale is an
            //    addressing-mode one, so it indexes the access itself
            //    (`base + i*s + d`) rather than feeding a second, scaled
            //    variable that would be live beside it.
            //  - Pointer: the step is not a constant (see pointer_iv).
            //  - Scaled: `base + si + d` with si the scaled variable, shared
            //    by every array the variable indexes.
            int64_t step_c = 0;
            AccessMode mode = AccessMode::Scaled;
            if (get_const_int(matched->step, step_c)) {
                if ((scale == 2 || scale == 4 || scale == 8) && stays_live(*matched, scale)) mode = AccessMode::Index;
            } else if (!matched->is_sub) {
                mode = AccessMode::Pointer;
            }
            if (mode == AccessMode::Index && !inv_offset) continue;  // already `base + i*s`
            accesses.push_back({inst, matched, inv_offset, mode});
        }
    }

    // A hoisted `base + inv*s`, one per (base, invariant, scale).
    std::vector<std::pair<std::tuple<Value*, Value*, uint8_t>, Value*>> hoisted_bases;
    auto hoisted_base = [&](Value* base, Value* inv, uint8_t scale) -> Value* {
        const auto key = std::make_tuple(base, inv, scale);
        for (const auto& [k, v] : hoisted_bases) {
            if (k == key) return v;
        }
        Value* v = build_add(b_ph, base, build_mul(b_ph, inv, scale));
        hoisted_bases.emplace_back(key, v);
        return v;
    };

    for (const Access& a : accesses) {
        Instruction* inst = a.inst;
        const uint8_t scale = inst->scale();
        // The displacement stays in the access (x64 and AArch64 address
        // it for free), as does a constant invariant part of the index
        // when the sum fits: one base per array rather than one hoisted
        // pointer per constant offset, each held in a register across
        // the loop.
        Value* new_base = inst->operand(0);
        const int32_t offset = inst->offset();
        int64_t new_offset = offset;
        int64_t inv_const = 0;
        if (a.inv_offset && get_const_int(a.inv_offset, inv_const) && inv_const >= INT32_MIN / 8 &&
            inv_const <= INT32_MAX / 8 && offset + inv_const * scale >= INT32_MIN &&
            offset + inv_const * scale <= INT32_MAX) {
            new_offset = offset + inv_const * scale;
        } else if (a.inv_offset) {
            new_base = hoisted_base(new_base, a.inv_offset, scale);
        }
        switch (a.mode) {
            case AccessMode::Index:
                inst->set_operand(0, new_base);
                inst->set_operand(1, a.biv->param);
                break;
            case AccessMode::Pointer: {
                // load_indexed/store_indexed without the index are load/store
                // with the same operands, offset and memory type.
                Value* p = pointer_iv(new_base, *a.biv, scale);
                auto& ops = inst->operands();
                ops.erase(ops.begin() + 1);
                inst->set_opcode(inst->opcode() == Opcode::load_indexed ? Opcode::load : Opcode::store);
                inst->set_operand(0, p);
                inst->set_scale(1);
                break;
            }
            case AccessMode::Scaled:
                inst->set_operand(0, new_base);
                inst->set_operand(1, scaled_biv(*a.biv, scale));
                inst->set_scale(1);
                break;
        }
        inst->set_offset(static_cast<int32_t>(new_offset));
        changed = true;
    }

    // i * s and i << log2(s) are the scaled variable itself.
    for (const auto& [key, scaled] : biv_scaled_map) {
        const Value* biv_param = key.first;
        const int64_t sc = key.second;
        const int64_t shl_k = sc == 8 ? 3 : (sc == 4 ? 2 : (sc == 2 ? 1 : 0));
        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == preheader) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (cur->produces_value() && cur->type() == Type::i64() && cur->operand_count() == 2) {
                    Value* op0 = cur->operand(0);
                    Value* op1 = cur->operand(1);
                    int64_t c0 = 0, c1 = 0;
                    const bool has_c0 = get_const_int(op0, c0);
                    const bool has_c1 = get_const_int(op1, c1);
                    bool same = false;
                    if (cur->opcode() == Opcode::mul) {
                        same = (op0 == biv_param && has_c1 && c1 == sc) || (op1 == biv_param && has_c0 && c0 == sc);
                    } else if (cur->opcode() == Opcode::shl && shl_k > 0) {
                        same = op0 == biv_param && has_c1 && c1 == shl_k;
                    }
                    if (same) {
                        replace_all_uses(fn, cur->result(), scaled);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                }
                cur = next;
            }
        }
    }

    // Move the header's exit compare onto the scaled variable, so the
    // original one can die - only when that cannot change its outcome.
    if (exit_cmp) {
        Value* rhs = exit_cmp->operand(1);
        const BasicIV& biv = bivs[param_to_biv[exit_cmp->operand(0)]];
        for (uint8_t sc : {uint8_t(8), uint8_t(4), uint8_t(2)}) {
            auto it = biv_scaled_map.find(std::make_pair(static_cast<const Value*>(biv.param), sc));
            if (it == biv_scaled_map.end()) continue;
            if (scaled_exit_compare_is_exact(biv, *exit_cmp, rhs, loop, *hdr_term, sc)) {
                exit_cmp->set_operand(0, it->second);
                exit_cmp->set_operand(1, build_mul(b_ph, rhs, sc));
                changed = true;
            }
            break;
        }
    }
    return changed;
}

} // namespace

bool strength_reduce_induction_variables(Function& fn) {
    if (!fn.entry_block() || !fn.parent()) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    bool changed = false;
    for (LoopInfo* loop : loops.post_order_loops()) {
        if (loop) changed |= ivsr_loop(fn, *loop);
    }
    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass
