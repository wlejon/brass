#include <brass/mir/sincos_fuse.hpp>
#include <brass/core/env.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/uses.hpp>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

bool fuse_disabled_by_env() {
    static const bool off = [] {
        const std::string v = env_value("BRASS_NO_SINCOS_FUSE");
        return !v.empty() && v[0] != '0';
    }();
    return off;
}

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

struct Site {
    Instruction* call;
    size_t pos;  // index within its block
};

struct Group {
    std::vector<Site> sins, coss;
};

}  // namespace

size_t fuse_sincos(Function& fn) {
    if (fuse_disabled_by_env()) return 0;
    Module* mod = fn.parent();
    if (!mod) return 0;
    const std::string_view fused = mod->symbol_with_role(SymbolRole::SinCosF64);
    if (fused.empty()) return 0;

    // Calls grouped by operand, in block order (the map's order is not used
    // for anything that reaches the output: `order` is).
    std::unordered_map<Value*, Group> groups;
    std::vector<Value*> order;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        size_t pos = 0;
        for (Instruction* inst : *bb) {
            const Kind k = kind_of(inst);
            if (k != Kind::None) {
                auto [it, fresh] = groups.try_emplace(inst->operand(0));
                if (fresh) order.push_back(inst->operand(0));
                (k == Kind::Sin ? it->second.sins : it->second.coss).push_back({inst, pos});
            }
            ++pos;
        }
    }

    std::unique_ptr<DominatorTree> dt;
    // Whether `a` executes before `b` on every path to `b`.
    auto before = [&](const Site& a, const Site& b) {
        if (a.call->parent() == b.call->parent()) return a.pos < b.pos;
        if (!dt) dt = std::make_unique<DominatorTree>(fn);
        return dt->dominates(a.call->parent(), b.call->parent());
    };

    struct Pair {
        Instruction* first;  // the dominating call: the fused call goes here
        Instruction* sin;
        Instruction* cos;
    };
    std::vector<Pair> pairs;
    for (Value* x : order) {
        Group& g = groups[x];
        if (g.sins.empty() || g.coss.empty()) continue;
        std::vector<bool> cos_used(g.coss.size(), false);
        for (const Site& s : g.sins) {
            for (size_t j = 0; j < g.coss.size(); ++j) {
                if (cos_used[j]) continue;
                const Site& c = g.coss[j];
                Instruction* first = before(s, c) ? s.call : before(c, s) ? c.call : nullptr;
                if (!first) continue;
                cos_used[j] = true;
                pairs.push_back({first, s.call, c.call});
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
    for (const Pair& p : pairs) {
        Value* x = p.first->operand(0);
        b.position_before(p.first);
        b.build_call(fused, Type::void_type(), {x, slot});
        Value* s = b.build_load(Type::f64(), slot, 0);
        Value* c = b.build_load(Type::f64(), slot, 8);
        replace_all_uses(fn, p.sin->result(), s);
        replace_all_uses(fn, p.cos->result(), c);
        p.sin->parent()->remove_instruction(p.sin);
        p.cos->parent()->remove_instruction(p.cos);
    }
    return pairs.size();
}

}  // namespace brass
