// Block live-in sets (block_liveness.hpp).
#include <brass/mir/block_liveness.hpp>
#include <brass/mir/uses.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace brass {

namespace {

constexpr uint32_t kNoBlock = UINT32_MAX;

BlockLiveIns compute(const Function& fn, const std::vector<const BasicBlock*>* targets) {
    // Dense block numbering, in the function's block order.
    std::vector<const BasicBlock*> blocks;
    blocks.reserve(fn.blocks().size());
    std::unordered_map<const BasicBlock*, uint32_t> index;
    index.reserve(fn.blocks().size() * 2);
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb || index.count(bb)) continue;
        index.emplace(bb, static_cast<uint32_t>(blocks.size()));
        blocks.push_back(bb);
    }
    const size_t n = blocks.size();
    auto block_of = [&](const BasicBlock* bb) -> uint32_t {
        if (!bb) return kNoBlock;
        auto it = index.find(bb);
        return it == index.end() ? kNoBlock : it->second;
    };

    // Predecessors (from each listed block's own successors) and every use
    // a block reads before any definition of it in the block: a use of a
    // value defined elsewhere (or nowhere in this function). A value defined
    // in the same block is defined before its use in valid SSA.
    std::vector<std::vector<uint32_t>> preds(n);
    std::vector<std::pair<const Value*, uint32_t>> uses;
    std::unordered_map<const Value*, uint32_t> def_block;
    auto def_of = [&](const Value* v) -> uint32_t {
        auto it = def_block.find(v);
        if (it != def_block.end()) return it->second;
        uint32_t d = kNoBlock;
        if (v->is_block_param()) {
            d = block_of(v->defining_block());
        } else if (const Instruction* def = v->defining_instruction()) {
            d = block_of(def->parent());
        }
        def_block.emplace(v, d);
        return d;
    };
    for (uint32_t b = 0; b < n; ++b) {
        const BasicBlock* bb = blocks[b];
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            for_each_use(*inst, [&](Value* v) {
                if (def_of(v) != b) uses.emplace_back(v, b);
            });
        }
        for (const BasicBlock* s : bb->successors()) {
            const uint32_t si = block_of(s);
            if (si != kNoBlock) preds[si].push_back(b);
        }
    }

    std::vector<uint8_t> wanted;
    if (targets) {
        wanted.assign(n, 0);
        for (const BasicBlock* t : *targets) {
            const uint32_t ti = block_of(t);
            if (ti != kNoBlock) wanted[ti] = 1;
        }
    }
    BlockLiveIns live_in;
    live_in.reserve(targets ? targets->size() : n);
    for (uint32_t b = 0; b < n; ++b) {
        if (!targets || wanted[b]) live_in[blocks[b]];
    }

    // Per value: every block from a use back to (not into) its definition.
    std::sort(uses.begin(), uses.end());
    std::vector<uint32_t> stamp(n, 0);
    std::vector<uint32_t> work;
    uint32_t round = 0;
    for (size_t i = 0; i < uses.size();) {
        const Value* v = uses[i].first;
        const uint32_t def = def_of(v);
        ++round;
        auto mark = [&](uint32_t b) {
            if (stamp[b] == round) return;
            stamp[b] = round;
            if (!targets || wanted[b]) live_in[blocks[b]].push_back(v);
            work.push_back(b);
        };
        for (; i < uses.size() && uses[i].first == v; ++i) mark(uses[i].second);
        while (!work.empty()) {
            const uint32_t b = work.back();
            work.pop_back();
            for (uint32_t p : preds[b]) {
                if (p != def) mark(p);
            }
        }
    }
    return live_in;
}

} // namespace

BlockLiveIns block_live_ins(const Function& fn) { return compute(fn, nullptr); }

BlockLiveIns block_live_ins(const Function& fn, const std::vector<const BasicBlock*>& targets) {
    return compute(fn, &targets);
}

} // namespace brass
