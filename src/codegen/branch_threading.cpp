#include <brass/codegen/branch_threading.hpp>
#include <algorithm>

namespace brass::codegen {

namespace {

bool is_label_branch(const LirInst& inst, LirOpcode op) noexcept {
    return inst.opcode == op && !inst.uses.empty() && inst.uses[0].is_label();
}

} // namespace

ClosingBranches closing_branches(const LirFunction& fn, size_t index) noexcept {
    ClosingBranches c;
    const auto& ins = fn.blocks[index]->instructions;
    const size_t n = ins.size();
    if (n >= 2 && is_label_branch(*ins[n - 2], LirOpcode::Jcc) && is_label_branch(*ins[n - 1], LirOpcode::Jmp)) {
        c.count = 2;
        c.conditional = true;
        c.taken = ins[n - 2]->uses[0].label_id;
        c.other = ins[n - 1]->uses[0].label_id;
    } else if (n >= 1 && is_label_branch(*ins[n - 1], LirOpcode::Jmp)) {
        c.count = 1;
        c.other = ins[n - 1]->uses[0].label_id;
    } else if (n >= 1 && is_label_branch(*ins[n - 1], LirOpcode::Jcc) && index + 1 < fn.blocks.size()) {
        c.count = 1;
        c.conditional = true;
        c.taken = ins[n - 1]->uses[0].label_id;
        c.other = fn.blocks[index + 1]->id;
    }
    return c;
}

BranchThreading::BranchThreading(const LirFunction& fn) {
    const size_t n = fn.blocks.size();
    uint32_t max_id = 0;
    for (const auto& b : fn.blocks) max_id = std::max(max_id, b->id);
    forward_.resize(size_t{max_id} + 1);
    for (uint32_t id = 0; id <= max_id; ++id) forward_[id] = id;
    skip_.assign(n, 0);
    head_.assign(n, 0);
    if (n == 0) return;
    std::vector<uint32_t> pos(forward_.size(), UINT32_MAX);
    for (size_t i = 0; i < n; ++i) pos[fn.blocks[i]->id] = static_cast<uint32_t>(i);

    // Blocks that are a jump and nothing else, but the entry.
    std::vector<uint32_t> hop(forward_.size(), UINT32_MAX);
    for (size_t i = 1; i < n; ++i) {
        const LirBlock& b = *fn.blocks[i];
        if (b.instructions.size() == 1 && is_label_branch(*b.instructions[0], LirOpcode::Jmp)) {
            hop[b.id] = b.instructions[0]->uses[0].label_id;
        }
    }
    for (uint32_t id = 0; id <= max_id; ++id) {
        uint32_t t = id;
        // A chain is followed for as many steps as there are blocks: a
        // cycle of jumps goes on jumping, to a block of the cycle.
        for (size_t steps = 0; steps < n && t < hop.size() && hop[t] != UINT32_MAX; ++steps) t = hop[t];
        forward_[id] = t;
    }

    // Which blocks something still enters.
    std::vector<uint8_t> entered(forward_.size(), 0);
    auto enter = [&](uint32_t id) { if (id < entered.size()) entered[id] = 1; };
    // A block branched to from itself or after it is a loop's head only where
    // the loop is entered: some predecessor at a lower loop depth. The join of
    // a diamond whose one arm the layout placed at the function's end (a cold
    // miss, an inline cache's slow call) is branched to from after itself as
    // well, and aligning it executed the padding on the arm that falls into
    // it — the hot one. A function liveness found loop-free has no head at
    // all: there every such target is such a join, and padding them put a
    // nop run on the hot path of each inline cache's merge. With no depths
    // computed, every such target is taken for a head, as the layout promises.
    bool depths_known = fn.loop_depths_computed;
    for (const auto& b : fn.blocks) depths_known = depths_known || b->loop_depth > 0;
    auto entered_from_outside = [&](const LirBlock& b) {
        if (!depths_known) return true;
        for (const LirBlock* p : b.predecessors) {
            if (p && p->loop_depth < b.loop_depth) return true;
        }
        return false;
    };
    auto branch_to = [&](uint32_t to, size_t from) {
        enter(to);
        if (to < pos.size() && pos[to] <= from && entered_from_outside(*fn.blocks[pos[to]])) head_[pos[to]] = 1;
    };
    enter(fn.blocks.front()->id);
    for (const auto& r : fn.resume_entries) enter(r.second);
    bool falls = false;  // the block before runs on into this one
    for (size_t i = 0; i < n; ++i) {
        const LirBlock& b = *fn.blocks[i];
        if (falls) enter(b.id);
        const ClosingBranches c = closing_branches(fn, i);
        const size_t body = b.instructions.size() - c.count;
        for (size_t k = 0; k < body; ++k) {
            const LirInst& inst = *b.instructions[k];
            if (inst.unwind_block_id != UINT32_MAX) enter(inst.unwind_block_id);
            for (const auto& u : inst.uses) {
                if (!u.is_label()) continue;
                if (inst.is_branch()) branch_to(u.label_id, i);
                else enter(u.label_id);
            }
        }
        if (c.count > 0) {
            if (c.conditional) branch_to(target(c.taken), i);
            branch_to(target(c.other), i);
            falls = false;
        } else {
            const auto& ins = b.instructions;
            falls = ins.empty() || (ins.back()->opcode != LirOpcode::Ret && ins.back()->opcode != LirOpcode::Trap);
        }
    }
    for (size_t i = 1; i < n; ++i) {
        const LirBlock& b = *fn.blocks[i];
        if (hop[b.id] != UINT32_MAX && !entered[b.id]) skip_[i] = 1;
    }
}

} // namespace brass::codegen
