#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/uses.hpp>
#include "ir_clone.hpp"
#include <algorithm>
#include <limits>
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
};

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
        BasicIV biv{param, ph_bt->args[i], nullptr, false};
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

    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || bb == preheader) continue;
        for (Instruction* inst : *bb) {
            if (!inst || (inst->opcode() != Opcode::load_indexed && inst->opcode() != Opcode::store_indexed)) continue;
            Value* base = inst->operand(0);
            Value* index = inst->operand(1);
            const uint8_t scale = inst->scale();
            const int32_t offset = inst->offset();
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

            Value* new_base = base;
            if (inv_offset) new_base = build_add(b_ph, new_base, build_mul(b_ph, inv_offset, scale));
            if (offset != 0) new_base = build_add(b_ph, new_base, b_ph.build_iconst_i64(offset));
            Value* scaled = scaled_biv(*matched, scale);
            inst->set_operand(0, new_base);
            inst->set_operand(1, scaled);
            inst->set_scale(1);
            inst->set_offset(0);
            changed = true;
        }
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
    Instruction* hdr_term = header->terminator();
    if (hdr_term && hdr_term->opcode() == Opcode::br_if) {
        Value* cond_val = hdr_term->operand(0);
        Instruction* cond_inst = cond_val && cond_val->is_instruction() ? cond_val->defining_instruction() : nullptr;
        if (cond_inst && cond_inst->parent() == header && cond_inst->operand_count() == 2) {
            Value* lhs = cond_inst->operand(0);
            Value* rhs = cond_inst->operand(1);
            auto p = param_to_biv.find(lhs);
            if (p != param_to_biv.end() && is_i64(rhs) && loop.is_loop_invariant(rhs)) {
                const BasicIV& biv = bivs[p->second];
                for (uint8_t sc : {uint8_t(8), uint8_t(4), uint8_t(2)}) {
                    auto it = biv_scaled_map.find(std::make_pair(static_cast<const Value*>(biv.param), sc));
                    if (it == biv_scaled_map.end()) continue;
                    if (scaled_exit_compare_is_exact(biv, *cond_inst, rhs, loop, *hdr_term, sc)) {
                        cond_inst->set_operand(0, it->second);
                        cond_inst->set_operand(1, build_mul(b_ph, rhs, sc));
                        changed = true;
                    }
                    break;
                }
            }
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
