#include "gvn_gc_free.hpp"

#include <brass/mir/block.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/interrupt_poll.hpp>

namespace brass {

void GcFreePaths::build() {
    built_ = true;
    const Function& fn = *fn_;
    uint32_t n = 0;
    block_number_.reserve(fn.current_next_block_id());
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        block_number_.insert(bb, n++);
        // Whether it has a GC point; where they are is taken when a
        // question first names one of its instructions (position()).
        BlockInfo bi;
        for (const Instruction* inst = bb->head(); inst && !bi.has_gc; inst = inst->next()) {
            bi.has_gc = may_trigger_gc(*inst);
        }
        info_.push_back(bi);
    }
    block_count_ = n;
    numbered_.assign(n, 0);
    // The targets of a depth-first walk's back edges, which
    // interrupt_poll.cpp polls when the module asks for polls.
    if (!needs_interrupt_polls(fn)) return;
    std::vector<uint8_t> state(n, 0);  // 1 on the stack, 2 done
    std::vector<std::pair<const BasicBlock*, size_t>> stack;
    auto walk = [&](const BasicBlock* root) {
        if (!root || state[number(root)]) return;
        state[number(root)] = 1;
        stack.push_back({root, 0});
        while (!stack.empty()) {
            auto& [bb, next] = stack.back();
            const SuccessorList succs = bb->successors();
            if (next == succs.size()) {
                state[number(bb)] = 2;
                stack.pop_back();
                continue;
            }
            const BasicBlock* succ = succs[next++];
            if (!succ) continue;
            uint8_t& st = state[number(succ)];
            if (st == 0) {
                st = 1;
                stack.push_back({succ, 0});
            } else if (st == 1) {
                info_[number(succ)].header = true;
            }
        }
    };
    walk(fn.entry_block());
    for (const BasicBlock* bb : fn.blocks()) walk(bb);
}

bool GcFreePaths::position(const Instruction* inst, uint32_t& at) {
    const BasicBlock* bb = inst->parent();
    if (!bb || !block_number_.contains(bb)) return false;
    uint8_t& done = numbered_[number(bb)];
    if (!done) {
        // The block's GC points are placed in the same numbering: GVN may
        // have put constants into the block since build() first looked.
        done = 1;
        BlockInfo& bi = info_[number(bb)];
        bi.has_gc = false;
        bi.first_gc = UINT32_MAX;
        bi.last_gc = 0;
        uint32_t i = 0;
        for (const Instruction* x = bb->head(); x; x = x->next(), ++i) {
            index_[x] = i;
            if (may_trigger_gc(*x)) {
                if (!bi.has_gc) bi.first_gc = i;
                bi.last_gc = i;
                bi.has_gc = true;
            }
        }
    }
    const auto it = index_.find(inst);
    if (it == index_.end()) return false;
    at = it->second;
    return true;
}

bool GcFreePaths::between(const Instruction* from, const Instruction* to) {
    if (!from || !to) return false;
    if (!built_) build();
    const BasicBlock* a = from->parent();
    const BasicBlock* b = to->parent();
    uint32_t ia = 0, ib = 0;
    if (!a || !b || !position(from, ia) || !position(to, ib)) return false;
    const BlockInfo& ai = info_[number(a)];
    const BlockInfo& bi = info_[number(b)];
    if (a == b) {
        // `from` dominates `to` in one block: it comes first, and the
        // stretch is the instructions in between.
        if (ia >= ib) return false;
        if (!ai.has_gc) return true;
        for (const Instruction* inst = from->next(); inst && inst != to; inst = inst->next()) {
            if (may_trigger_gc(*inst)) return false;
        }
        return true;
    }
    if (ai.has_gc && ai.last_gc > ia) return false;
    if (bi.header || (bi.has_gc && bi.first_gc < ib)) return false;
    return blocks_between_gc_free(a, b);
}

// Every block on a path from `a` to `b` that does not pass through `a` again
// (which would run `from` afresh): the blocks reachable from a's successors
// without entering `a`, that reach one of b's predecessors without entering
// `a`. Each must be free of GC points; `b` among them (a cycle through `b`
// that avoids `a`) must be free of them entirely.
//
// Walked backward from `b` alone: `a` dominates `b`, so a block the entry
// reaches that reaches `b` without entering `a` is reached from `a` (the
// path from the entry passes `a`, and its part after the last `a` avoids
// it), and the blocks the entry does not reach are what a forward walk from
// `a` would have left out.
bool GcFreePaths::blocks_between_gc_free(const BasicBlock* a, const BasicBlock* b) {
    const uint32_t na = number(a);
    const uint64_t key = (static_cast<uint64_t>(na) << 32) | number(b);
    if (const auto it = cache_.find(key); it != cache_.end()) return it->second;

    std::vector<uint8_t> back(block_count_, 0);
    std::vector<const BasicBlock*> work;
    bool ok = true;
    for (const BasicBlock* p : b->predecessors()) {
        if (p && p != a && !back[number(p)]) {
            back[number(p)] = 1;
            work.push_back(p);
        }
    }
    while (!work.empty() && ok) {
        const BasicBlock* c = work.back();
        work.pop_back();
        const uint32_t nc = number(c);
        if ((info_[nc].has_gc || info_[nc].header) && dom_->is_reachable(c)) ok = false;
        for (const BasicBlock* p : c->predecessors()) {
            if (!p || p == a || back[number(p)]) continue;
            back[number(p)] = 1;
            work.push_back(p);
        }
    }
    cache_[key] = ok;
    return ok;
}

} // namespace brass
