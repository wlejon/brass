#include <brass/mir/sincos_fuse.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

enum class Kind { None, Sin, Cos };

Kind kind_of(const Instruction* inst) {
    if (!inst || inst->opcode() != Opcode::call || inst->operand_count() != 1 || !inst->result() ||
        inst->type() != Type::f64() || !inst->operand(0) || inst->operand(0)->type() != Type::f64()) {
        return Kind::None;
    }
    const bool is_sin = callee_has_role(*inst, SymbolRole::SinF64);
    const bool is_cos = callee_has_role(*inst, SymbolRole::CosF64);
    // A symbol declared both is no sine or cosine the pass can name.
    if (is_sin == is_cos) return Kind::None;
    return is_sin ? Kind::Sin : Kind::Cos;
}

bool is_nan_bits(uint64_t bits) {
    return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
           (bits & 0x000FFFFFFFFFFFFFULL) != 0;
}

bool is_nan_const_i64(const Value* v) {
    const Instruction* d = v && v->is_instruction() ? v->defining_instruction() : nullptr;
    return d && d->opcode() == Opcode::iconst_i64 && is_nan_bits(static_cast<uint64_t>(d->imm_i64()));
}

Value* bitcast_i64_of_f64(const Value* v) {
    const Instruction* d = v && v->is_instruction() ? v->defining_instruction() : nullptr;
    if (!d || d->opcode() != Opcode::bitcast_i64_f64 || d->operand_count() != 1) return nullptr;
    Value* x = d->operand(0);
    return x && x->type() == Type::f64() ? x : nullptr;
}

// Looks through an f64 -> i64 -> f64 roundtrip, including a NaN-canonicalizing
// box (`select(ne(x, x), qnan, bitcast(x))` or its ordered `eq` form). Every
// NaN input produces the same canonical quiet NaN in sin_f64, cos_f64, and
// sincos_f64, and every non-NaN preserves all 64 bits across the roundtrip.
Value* canon_arg(Value* v) {
    for (int depth = 0; depth < 8 && v && v->is_instruction(); ++depth) {
        const Instruction* def = v->defining_instruction();
        if (!def || def->opcode() != Opcode::bitcast_f64_i64 || def->operand_count() != 1) break;
        Value* bits = def->operand(0);
        if (const Instruction* untag = bits && bits->is_instruction() ? bits->defining_instruction() : nullptr;
            untag && untag->opcode() == Opcode::bitcast_i64_tagged && untag->operand_count() == 1) {
            const Value* tagged = untag->operand(0);
            const Instruction* tag = tagged && tagged->is_instruction() ? tagged->defining_instruction() : nullptr;
            if (tag && tag->opcode() == Opcode::bitcast_tagged_i64 && tag->operand_count() == 1) {
                bits = tag->operand(0);
            }
        }
        if (Value* x = bitcast_i64_of_f64(bits)) {
            v = x;
            continue;
        }
        const Instruction* sel = bits && bits->is_instruction() ? bits->defining_instruction() : nullptr;
        if (!sel || sel->opcode() != Opcode::select || sel->operand_count() != 3) break;
        const Value* cond = sel->operand(0);
        const Instruction* cmp = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
        if (!cmp || cmp->operand_count() != 2 || cmp->operand(0) != cmp->operand(1)) break;
        if (cmp->opcode() == Opcode::ne && is_nan_const_i64(sel->operand(1))) {
            if (Value* x = bitcast_i64_of_f64(sel->operand(2)); x && x == cmp->operand(0)) {
                v = x;
                continue;
            }
        } else if (cmp->opcode() == Opcode::eq && is_nan_const_i64(sel->operand(2))) {
            if (Value* x = bitcast_i64_of_f64(sel->operand(1)); x && x == cmp->operand(0)) {
                v = x;
                continue;
            }
        }
        break;
    }
    return v;
}

bool is_cold_block(const BasicBlock* bb) {
    return bb && (bb->is_cold() || bb->name().find("cold") != std::string_view::npos);
}

BasicBlock* common_dominator(const DominatorTree& dom, BasicBlock* a, BasicBlock* b) {
    const BasicBlock* cur = a;
    while (cur && !dom.dominates(cur, b)) cur = dom.immediate_dominator(cur);
    return const_cast<BasicBlock*>(cur);
}

// Insertion point at the end of `ncd`, ahead of its `br_if` and any adjacent
// speculation-site placeholder `guard 1` (which arm_speculation_sites expects
// immediately before `br_if`).
Instruction* ncd_insert_point(BasicBlock* ncd) {
    Instruction* term = ncd ? ncd->terminator() : nullptr;
    if (!term || term->opcode() != Opcode::br_if) return nullptr;
    Instruction* pt = term;
    if (Instruction* prev = pt->prev(); prev && prev->opcode() == Opcode::guard) {
        const Value* cond = prev->operand(0);
        const Instruction* def = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
        if (def && def->opcode() == Opcode::iconst_i32 && def->imm_i32() == 1) {
            pt = prev;
            if (Instruction* sp = pt->prev(); sp && sp->opcode() == Opcode::store) pt = sp;
        }
    }
    return pt;
}

bool value_available_before(const Value* v, const BasicBlock* bb, const Instruction* before_inst,
                            const DominatorTree& dom) {
    if (!v || !bb || !before_inst) return false;
    if (v->is_block_param()) {
        const BasicBlock* def_bb = v->defining_block();
        return def_bb && dom.dominates(def_bb, bb);
    }
    if (!v->is_instruction()) return false;
    const Instruction* def = v->defining_instruction();
    if (!def || !def->parent()) return false;
    const BasicBlock* def_bb = def->parent();
    if (def_bb == bb) {
        for (const Instruction* cur = bb->head(); cur && cur != before_inst; cur = cur->next()) {
            if (cur == def) return true;
        }
        return false;
    }
    if (def->is_terminator()) {
        if (def->opcode() != Opcode::invoke) return false;
        const BasicBlock* normal = def->normal_target().block;
        return normal && normal->predecessors().size() == 1 && dom.dominates(normal, bb);
    }
    return dom.dominates(def_bb, bb);
}

// Whether `s_bb` and `c_bb` both lie on the unique non-cold execution spine
// leaving `ncd` at the same loop depth, with the earlier call's block entered
// right from `ncd`'s guard: every hot execution of `ncd` already runs both
// calls, so computing sincos_f64 at `ncd`'s tail never speculates across an
// uncertain two-way branch or hoists cold code onto a hot path.
bool hot_spine_pair(BasicBlock* ncd, BasicBlock* s_bb, BasicBlock* c_bb, const DominatorTree& dom,
                    const std::unordered_map<const BasicBlock*, size_t>& loop_depths) {
    if (!ncd || !s_bb || !c_bb) return false;
    if (is_cold_block(ncd) || is_cold_block(s_bb) || is_cold_block(c_bb)) return false;
    auto depth_of = [&](const BasicBlock* bb) -> size_t {
        auto it = loop_depths.find(bb);
        return it == loop_depths.end() ? 0 : it->second;
    };
    const size_t ncd_depth = depth_of(ncd);
    if (depth_of(s_bb) != ncd_depth || depth_of(c_bb) != ncd_depth) return false;

    constexpr size_t kMaxSteps = 32;
    const BasicBlock* visited[kMaxSteps + 1] = {ncd};
    size_t visited_count = 1;
    size_t s_step = 0;
    size_t c_step = 0;
    BasicBlock* cur = ncd;
    for (size_t step = 1; step <= kMaxSteps; ++step) {
        Instruction* term = cur->terminator();
        BasicBlock* next = nullptr;
        if (!term) break;
        if (term->opcode() == Opcode::br) {
            BasicBlock* target = term->branch_target().block;
            if (target && !is_cold_block(target)) next = target;
        } else if (term->opcode() == Opcode::br_if) {
            BasicBlock* t = term->true_target().block;
            BasicBlock* f = term->false_target().block;
            if (t && t == f && !is_cold_block(t)) {
                next = t;
            } else if (t && f && is_cold_block(t) != is_cold_block(f)) {
                next = is_cold_block(t) ? f : t;
            }
        }
        if (!next || !dom.dominates(ncd, next) || depth_of(next) != ncd_depth) break;
        bool seen = false;
        for (size_t k = 0; k < visited_count; ++k) {
            if (visited[k] == next) {
                seen = true;
                break;
            }
        }
        if (seen) break;
        visited[visited_count++] = next;
        if (next == s_bb) s_step = step;
        if (next == c_bb) c_step = step;
        if (s_step != 0 && c_step != 0) break;
        cur = next;
    }
    return s_step != 0 && c_step != 0 && std::min(s_step, c_step) <= 2;
}

struct Site {
    Instruction* call;
    size_t pos;  // index within its block
};

struct Group {
    std::vector<Site> sins, coss;
};

}  // namespace

size_t fuse_sincos(Function& fn) {
    Module* mod = fn.parent();
    if (!mod) return 0;
    const std::string_view fused = mod->symbol_with_role(SymbolRole::SinCosF64);
    if (fused.empty()) return 0;

    // Calls grouped by canonical f64 operand, in block order (the map's order
    // is not used for anything that reaches the output: `order` is).
    std::unordered_map<Value*, Group> groups;
    std::vector<Value*> order;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        size_t pos = 0;
        for (Instruction* inst : *bb) {
            const Kind k = kind_of(inst);
            if (k != Kind::None) {
                Value* key = canon_arg(inst->operand(0));
                auto [it, fresh] = groups.try_emplace(key);
                if (fresh) order.push_back(key);
                (k == Kind::Sin ? it->second.sins : it->second.coss).push_back({inst, pos});
            }
            ++pos;
        }
    }

    std::unique_ptr<DominatorTree> dt;
    std::unordered_map<const BasicBlock*, size_t> loop_depths;
    auto ensure_cfg_info = [&]() {
        if (dt) return;
        dt = std::make_unique<DominatorTree>(fn);
        LoopAnalysis loops(fn, *dt);
        for (const LoopInfo* loop : loops.post_order_loops()) {
            const size_t d = loop->depth();
            for (const BasicBlock* bb : loop->blocks()) {
                size_t& cur = loop_depths[bb];
                if (d > cur) cur = d;
            }
        }
    };
    auto depth_of = [&](const BasicBlock* bb) -> size_t {
        auto it = loop_depths.find(bb);
        return it == loop_depths.end() ? 0 : it->second;
    };
    // Whether `a` executes before `b` on every path to `b` without moving the
    // call into a deeper loop than `b`.
    auto before = [&](const Site& a, const Site& b) {
        BasicBlock* abb = a.call->parent();
        BasicBlock* bbb = b.call->parent();
        if (abb == bbb) return a.pos < b.pos;
        ensure_cfg_info();
        return dt->dominates(abb, bbb) && depth_of(abb) <= depth_of(bbb);
    };

    struct Pair {
        Instruction* insert_before;  // dominating call or nearest common dominator's branch
        Value* arg;
        Instruction* sin;
        Instruction* cos;
    };
    std::vector<Pair> pairs;
    for (Value* x : order) {
        Group& g = groups[x];
        if (g.sins.empty() || g.coss.empty()) continue;
        std::vector<bool> cos_used(g.coss.size(), false);
        for (const Site& s : g.sins) {
            bool matched = false;
            for (size_t j = 0; j < g.coss.size(); ++j) {
                if (cos_used[j]) continue;
                const Site& c = g.coss[j];
                Instruction* first = before(s, c) ? s.call : before(c, s) ? c.call : nullptr;
                if (!first) continue;
                cos_used[j] = true;
                pairs.push_back({first, x, s.call, c.call});
                matched = true;
                break;
            }
            if (matched) continue;
            for (size_t j = 0; j < g.coss.size(); ++j) {
                if (cos_used[j]) continue;
                const Site& c = g.coss[j];
                BasicBlock* s_bb = s.call->parent();
                BasicBlock* c_bb = c.call->parent();
                if (s_bb == c_bb) continue;
                ensure_cfg_info();
                if (!dt->is_reachable(s_bb) || !dt->is_reachable(c_bb)) continue;
                BasicBlock* ncd = common_dominator(*dt, s_bb, c_bb);
                Instruction* pt = ncd_insert_point(ncd);
                if (!pt || !value_available_before(x, ncd, pt, *dt)) continue;
                if (!hot_spine_pair(ncd, s_bb, c_bb, *dt, loop_depths)) continue;
                cos_used[j] = true;
                pairs.push_back({pt, x, s.call, c.call});
                break;
            }
        }
    }
    if (pairs.empty()) return 0;

    Builder b(fn);
    BasicBlock* entry = fn.entry_block();
    // One frame slot for every pair, in the entry as every alloca is
    // (allocated once, not per execution; the FastInterpreter allocates an
    // alloca each time it runs one). Each pair reads its two words right
    // after its own call, before any other call can write them, so the pairs
    // never need the slot at once; one slot is one address held across the
    // function rather than one per pair.
    if (Instruction* head = entry->is_empty() ? nullptr : *entry->begin()) {
        b.position_before(head);
    } else {
        b.position_at_end(entry);
    }
    Value* slot = b.build_alloca(16, 8);
    std::unordered_map<Value*, Value*> replaced;
    for (const Pair& p : pairs) {
        Value* x = p.arg;
        for (auto it = replaced.find(x); it != replaced.end(); it = replaced.find(x)) x = it->second;
        b.position_before(p.insert_before);
        b.build_call(fused, Type::void_type(), {x, slot});
        Value* s = b.build_load(Type::f64(), slot, 0);
        Value* c = b.build_load(Type::f64(), slot, 8);
        replaced[p.sin->result()] = s;
        replaced[p.cos->result()] = c;
        replace_all_uses(fn, p.sin->result(), s);
        replace_all_uses(fn, p.cos->result(), c);
        p.sin->parent()->remove_instruction(p.sin);
        p.cos->parent()->remove_instruction(p.cos);
    }
    return pairs.size();
}

}  // namespace brass
