// x64 baseline tier: frame layout. Every value lives in a stack slot, but
// values whose live ranges do not overlap share one, so a frame holds about
// as many slots as the function has values live at once rather than one per
// SSA value (deep recursion through baseline code needs small frames).
//
// Liveness is the usual backward dataflow over the CFG; each value's range
// is then the hull, in block layout order, of its definition, its uses and
// the blocks it is live into or out of. The hull over-approximates, which
// only costs sharing. Slots are handed out by a linear scan over the ranges.
//
// Kept out of sharing: block parameters (a predecessor's branch writes them,
// which is outside their range), 128-bit vectors (16-byte slots) and, apart
// from each other, GC references (every GC-ref slot is a stack-map root, so
// it must never hold a non-reference).
#include "baseline_emit_internal.hpp"
#include <algorithm>
#include <functional>
#include <unordered_set>

namespace brass::codegen {

namespace {

struct Range {
    const Value* val = nullptr;
    size_t start = SIZE_MAX;
    size_t end = 0;
    void cover(size_t p) {
        start = std::min(start, p);
        end = std::max(end, p);
    }
};

void for_each_target(const Function& fn, const Instruction& inst,
                     const std::function<void(const BranchTarget&)>& visit) {
    switch (inst.opcode()) {
        case Opcode::br: visit(inst.branch_target()); break;
        case Opcode::br_if: visit(inst.true_target()); visit(inst.false_target()); break;
        case Opcode::switch_:
            visit(inst.default_target());
            for (const auto& sc : inst.switch_cases()) visit(sc.target);
            break;
        case Opcode::guard:
            // Without an exit stub the guard jumps to its resume target with
            // the state map as the target's arguments.
            if (BasicBlock* resume = fn.get_resume_target(inst.resume_id())) {
                visit(BranchTarget(resume, inst.state_map()));
            }
            break;
        default: break;
    }
}

template <typename F>
void for_each_use(const Function& fn, const Instruction& inst, F&& use) {
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        if (inst.operand(i)) use(inst.operand(i));
    }
    for (const Value* v : inst.state_map()) {
        if (v) use(v);
    }
    for_each_target(fn, inst, [&](const BranchTarget& t) {
        for (const Value* v : t.args) {
            if (v) use(v);
        }
    });
}

} // namespace

BaselineFrameLayout layout_baseline_frame(const Function& fn, int32_t start_offset) {
    BaselineFrameLayout layout;
    int32_t offset = start_offset;

    std::vector<const BasicBlock*> blocks;
    std::unordered_map<const BasicBlock*, size_t> block_index;
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        block_index[bb] = blocks.size();
        blocks.push_back(bb);
    }

    // Allocas first: fixed buffers, never shared.
    for (const auto* bb : blocks) {
        for (const auto* inst : *bb) {
            if (!inst || inst->opcode() != Opcode::alloca_) continue;
            int32_t align = inst->offset() > 0 ? inst->offset() : 16;
            if (align < 16) align = 16;
            offset = (offset + align - 1) & ~(align - 1);
            offset += inst->imm_i32();
            layout.alloca_offsets[inst] = offset;
        }
    }

    auto dedicated = [&](const Value* v) {
        if (layout.slot_map.count(v)) return;
        if (bl_is_v128(v->type())) offset = ((offset + 15) & ~15) + 16;
        else offset += 8;
        layout.slot_map[v] = offset;
        if (v->type().is_gcref()) layout.gcref_slots.push_back(offset);
    };

    // Positions: block b spans [begin[b], end[b]]; its k-th instruction is
    // at begin[b] + 1 + k.
    std::vector<size_t> begin(blocks.size()), end(blocks.size());
    std::vector<std::vector<const BasicBlock*>> succs(blocks.size());
    std::vector<std::unordered_set<const Value*>> gen(blocks.size()), kill(blocks.size());
    size_t pos = 0;
    for (size_t b = 0; b < blocks.size(); ++b) {
        const BasicBlock* bb = blocks[b];
        begin[b] = pos++;
        for (const auto* param : bb->params()) kill[b].insert(param);
        for (const auto* inst : *bb) {
            if (!inst) continue;
            ++pos;
            for_each_use(fn, *inst, [&](const Value* v) {
                if (!kill[b].count(v)) gen[b].insert(v);
            });
            for_each_target(fn, *inst, [&](const BranchTarget& t) {
                if (t.block && block_index.count(t.block)) succs[b].push_back(t.block);
            });
            if (inst->produces_value() && inst->result()) kill[b].insert(inst->result());
        }
        end[b] = pos++;
    }

    std::vector<std::unordered_set<const Value*>> live_in(blocks.size()), live_out(blocks.size());
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = blocks.size(); i-- > 0;) {
            std::unordered_set<const Value*> out;
            for (const BasicBlock* s : succs[i]) {
                const size_t si = block_index[s];
                const auto& params = s->params();
                for (const Value* v : live_in[si]) {
                    if (std::find(params.begin(), params.end(), v) == params.end()) out.insert(v);
                }
            }
            std::unordered_set<const Value*> in = gen[i];
            for (const Value* v : out) {
                if (!kill[i].count(v)) in.insert(v);
            }
            if (out.size() != live_out[i].size() || in.size() != live_in[i].size()) changed = true;
            live_out[i] = std::move(out);
            live_in[i] = std::move(in);
        }
    }

    // Ranges of the shareable values.
    std::unordered_map<const Value*, Range> ranges;
    auto shareable = [&](const Value* v) { return ranges.count(v) != 0; };
    for (size_t b = 0; b < blocks.size(); ++b) {
        for (const auto* param : blocks[b]->params()) dedicated(param);
        size_t p = begin[b];
        for (const auto* inst : *blocks[b]) {
            if (!inst) continue;
            ++p;
            if (!inst->produces_value() || !inst->result()) continue;
            const Value* r = inst->result();
            if (bl_is_v128(r->type())) { dedicated(r); continue; }
            Range& range = ranges[r];
            range.val = r;
            range.cover(p);
        }
    }
    for (size_t b = 0; b < blocks.size(); ++b) {
        size_t p = begin[b];
        for (const auto* inst : *blocks[b]) {
            if (!inst) continue;
            ++p;
            for_each_use(fn, *inst, [&](const Value* v) {
                if (shareable(v)) ranges[v].cover(p);
            });
        }
        for (const Value* v : live_in[b]) {
            if (shareable(v)) ranges[v].cover(begin[b]);
        }
        for (const Value* v : live_out[b]) {
            if (shareable(v)) ranges[v].cover(end[b]);
        }
    }

    // Linear scan. A slot is reused only by a range starting strictly after
    // its previous owner's last use, so an instruction's result never shares
    // a slot with one of its own operands.
    std::vector<Range> order;
    order.reserve(ranges.size());
    for (auto& [v, r] : ranges) order.push_back(r);
    std::sort(order.begin(), order.end(), [](const Range& a, const Range& b) {
        return a.start != b.start ? a.start < b.start : a.val->id() < b.val->id();
    });
    struct Active { size_t end; int32_t slot; bool gcref; };
    std::vector<Active> active;
    std::vector<int32_t> free_plain, free_gcref;
    for (const Range& r : order) {
        for (size_t i = 0; i < active.size();) {
            if (active[i].end < r.start) {
                (active[i].gcref ? free_gcref : free_plain).push_back(active[i].slot);
                active[i] = active.back();
                active.pop_back();
            } else {
                ++i;
            }
        }
        const bool gcref = r.val->type().is_gcref();
        auto& pool = gcref ? free_gcref : free_plain;
        int32_t slot;
        if (!pool.empty()) {
            slot = pool.back();
            pool.pop_back();
        } else {
            offset += 8;
            slot = offset;
            if (gcref) layout.gcref_slots.push_back(slot);
        }
        layout.slot_map[r.val] = slot;
        active.push_back({r.end, slot, gcref});
    }

    layout.size = offset;
    return layout;
}

} // namespace brass::codegen
