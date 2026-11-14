#include "gvn_gc_free.hpp"

#include <brass/mir/block.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/interrupt_poll.hpp>

namespace brass {

GcFreePaths::GcFreePaths(const Function& fn) {
    uint32_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        block_number_[bb] = n++;
        BlockInfo bi;
        uint32_t i = 0;
        for (const Instruction* inst = bb->head(); inst; inst = inst->next(), ++i) {
            index_[inst] = i;
            if (may_trigger_gc(*inst)) {
                if (!bi.has_gc) bi.first_gc = i;
                bi.last_gc = i;
                bi.has_gc = true;
            }
        }
        info_[bb] = bi;
    }
    // The targets of a depth-first walk's back edges, which
    // interrupt_poll.cpp polls when the module asks for polls.
    if (!needs_interrupt_polls(fn)) return;
    std::unordered_map<const BasicBlock*, uint8_t> state;  // 1 on the stack, 2 done
    std::vector<std::pair<const BasicBlock*, size_t>> stack;
    auto walk = [&](const BasicBlock* root) {
        if (!root || state.count(root)) return;
        state[root] = 1;
        stack.push_back({root, 0});
        while (!stack.empty()) {
            auto& [bb, next] = stack.back();
            const SuccessorList succs = bb->successors();
            if (next == succs.size()) {
                state[bb] = 2;
                stack.pop_back();
                continue;
            }
            const BasicBlock* succ = succs[next++];
            if (!succ) continue;
            const auto it = state.find(succ);
            if (it == state.end()) {
                state[succ] = 1;
                stack.push_back({succ, 0});
            } else if (it->second == 1) {
                info_[succ].header = true;
            }
        }
    };
    walk(fn.entry_block());
    for (const BasicBlock* bb : fn.blocks()) walk(bb);
}

bool GcFreePaths::between(const Instruction* from, const Instruction* to) {
    if (!from || !to) return false;
    const BasicBlock* a = from->parent();
    const BasicBlock* b = to->parent();
    const auto ia = index_.find(from);
    const auto ib = index_.find(to);
    if (!a || !b || ia == index_.end() || ib == index_.end()) return false;
    const BlockInfo& ai = info_[a];
    const BlockInfo& bi = info_[b];
    if (a == b) {
        // `from` dominates `to` in one block: it comes first, and the
        // stretch is the instructions in between.
        if (ia->second >= ib->second) return false;
        if (!ai.has_gc) return true;
        for (const Instruction* inst = from->next(); inst && inst != to; inst = inst->next()) {
            if (may_trigger_gc(*inst)) return false;
        }
        return true;
    }
    if (ai.has_gc && ai.last_gc > ia->second) return false;
    if (bi.header || (bi.has_gc && bi.first_gc < ib->second)) return false;
    return blocks_between_gc_free(a, b);
}

// Every block on a path from `a` to `b` that does not pass through `a` again
// (which would run `from` afresh): the blocks reachable from a's successors
// without entering `a`, that reach one of b's predecessors without entering
// `a`. Each must be free of GC points; `b` among them (a cycle through `b`
// that avoids `a`) must be free of them entirely.
bool GcFreePaths::blocks_between_gc_free(const BasicBlock* a, const BasicBlock* b) {
    const uint64_t key = (static_cast<uint64_t>(block_number_[a]) << 32) | block_number_[b];
    if (const auto it = cache_.find(key); it != cache_.end()) return it->second;

    const size_t count = block_number_.size();
    std::vector<uint8_t> fwd(count, 0);
    std::vector<const BasicBlock*> work;
    for (const BasicBlock* s : a->successors()) {
        if (s && s != a && !fwd[block_number_[s]]) {
            fwd[block_number_[s]] = 1;
            work.push_back(s);
        }
    }
    while (!work.empty()) {
        const BasicBlock* c = work.back();
        work.pop_back();
        for (const BasicBlock* s : c->successors()) {
            if (!s || s == a || fwd[block_number_[s]]) continue;
            fwd[block_number_[s]] = 1;
            work.push_back(s);
        }
    }
    std::vector<uint8_t> back(count, 0);
    bool ok = true;
    for (const BasicBlock* p : b->predecessors()) {
        if (p && p != a && !back[block_number_[p]]) {
            back[block_number_[p]] = 1;
            work.push_back(p);
        }
    }
    while (!work.empty() && ok) {
        const BasicBlock* c = work.back();
        work.pop_back();
        if (fwd[block_number_[c]] && (info_[c].has_gc || info_[c].header)) ok = false;
        for (const BasicBlock* p : c->predecessors()) {
            if (!p || p == a || back[block_number_[p]]) continue;
            back[block_number_[p]] = 1;
            work.push_back(p);
        }
    }
    cache_[key] = ok;
    return ok;
}

} // namespace brass
