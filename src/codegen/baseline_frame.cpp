// Baseline tier (x64 and AArch64): frame layout. Every value lives in a stack slot, but
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
//
// A derived gcref (add/sub on a gcref: an interior pointer) is not a root:
// the collector would read an object header at its address. Like tier 2,
// which never has one live at a GC point, the baseline relies on the
// verifier's rule that a derived gcref is block-local and dead at every GC
// point (verifier_gc.cpp), so it lives in an ordinary slot. The layout
// checks that rule on the ranges it computes and rejects a function that
// breaks it.
#include "baseline_frame.hpp"
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/module.hpp>
#include <string>
#include <brass/mir/gc_refs.hpp>
#include <algorithm>
#include <functional>
#include <bit>

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
        case Opcode::invoke: visit(inst.normal_target()); visit(inst.unwind_target()); break;
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

BaselineFrameLayout layout_baseline_frame(const Function& fn, int32_t start_offset, std::string_view stage) {
    BaselineFrameLayout layout;
    int32_t offset = start_offset;

    std::vector<const BasicBlock*> blocks;
    DenseIdMap<BasicBlock, size_t> block_index(fn.current_next_block_id());
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        if (block_index.insert(bb, blocks.size())) blocks.push_back(bb);
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
            if (inst->memory_type().is_tagged()) {
                // The buffer spans [fp - offset, fp - offset + size).
                for (int32_t w = 0; w + 8 <= inst->imm_i32(); w += 8) layout.tagged_slots.push_back(offset - w);
            }
        }
    }

    // The root kind of a value's slot.
    enum class Root { None, GcRef, Tagged };
    auto root_of = [](const Value* v) {
        if (v->type().is_tagged()) return Root::Tagged;
        return v->type().is_gcref() && !is_derived_gcref(v) ? Root::GcRef : Root::None;
    };
    auto rooted = [&](const Value* v) { return root_of(v) == Root::GcRef; };
    auto add_root_slot = [&](Root kind, int32_t slot) {
        if (kind == Root::GcRef) layout.gcref_slots.push_back(slot);
        else if (kind == Root::Tagged) layout.tagged_slots.push_back(slot);
    };

    auto dedicated = [&](const Value* v) {
        if (layout.slot_map.contains(v)) return;
        if (bl_is_v128(v->type())) offset = ((offset + 15) & ~15) + 16;
        else offset += 8;
        layout.slot_map.insert(v, offset);
        add_root_slot(root_of(v), offset);
    };

    // The values whose slots may be shared: instruction results other than
    // 128-bit vectors, numbered densely. Liveness is computed over these
    // alone (a block parameter has a dedicated slot and is never shareable,
    // and the dataflow is per value), as bit rows per block: per-block hash
    // sets of values made a tier-1 compile of a big function mostly
    // allocation.
    DenseIdMap<Value, uint32_t> share_index(fn.current_next_value_id());
    std::vector<Range> ranges;
    for (const BasicBlock* bb : blocks) {
        for (const auto* param : bb->params()) dedicated(param);
        for (const auto* inst : *bb) {
            if (!inst || !inst->produces_value() || !inst->result()) continue;
            const Value* r = inst->result();
            if (bl_is_v128(r->type())) { dedicated(r); continue; }
            if (share_index.insert(r, static_cast<uint32_t>(ranges.size()))) ranges.push_back(Range{r});
        }
    }
    auto shared_index = [&](const Value* v) -> const uint32_t* { return share_index.find(v); };

    // Positions: block b spans [begin[b], end[b]]; its k-th instruction is
    // at begin[b] + 1 + k.
    const size_t nb = blocks.size();
    const size_t words = (ranges.size() + 63) / 64;
    std::vector<uint64_t> gen(nb * words, 0), kill(nb * words, 0), live_in(nb * words, 0), live_out(nb * words, 0);
    auto row = [&](std::vector<uint64_t>& bits, size_t b) { return bits.data() + b * words; };
    auto has = [](const uint64_t* r, uint32_t i) { return (r[i >> 6] >> (i & 63)) & 1; };
    auto put = [](uint64_t* r, uint32_t i) { r[i >> 6] |= uint64_t{1} << (i & 63); };
    std::vector<size_t> begin(nb), end(nb);
    std::vector<std::vector<uint32_t>> succs(nb);
    // The instruction at each position (null at block boundaries).
    std::vector<const Instruction*> at_pos;
    size_t pos = 0;
    for (size_t b = 0; b < nb; ++b) {
        const BasicBlock* bb = blocks[b];
        uint64_t* g = row(gen, b);
        uint64_t* k = row(kill, b);
        begin[b] = pos++;
        at_pos.push_back(nullptr);
        for (const auto* inst : *bb) {
            if (!inst) continue;
            ++pos;
            at_pos.push_back(inst);
            for_each_use(fn, *inst, [&](const Value* v) {
                if (const uint32_t* i = shared_index(v); i && !has(k, *i)) put(g, *i);
            });
            for_each_target(fn, *inst, [&](const BranchTarget& t) {
                if (!t.block) return;
                if (const size_t* si = block_index.find(t.block)) succs[b].push_back(static_cast<uint32_t>(*si));
            });
            if (inst->produces_value() && inst->result()) {
                if (const uint32_t* i = shared_index(inst->result())) put(k, *i);
            }
        }
        end[b] = pos++;
        at_pos.push_back(nullptr);
    }

    // out = union of the successors' live-in; in = gen | (out & ~kill).
    // Both only grow, so a pass that changes no word is the fixpoint.
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = nb; i-- > 0;) {
            uint64_t* out = row(live_out, i);
            uint64_t* in = row(live_in, i);
            const uint64_t* g = row(gen, i);
            const uint64_t* k = row(kill, i);
            for (uint32_t s : succs[i]) {
                const uint64_t* sin = row(live_in, s);
                for (size_t w = 0; w < words; ++w) out[w] |= sin[w];
            }
            for (size_t w = 0; w < words; ++w) {
                const uint64_t next = g[w] | (out[w] & ~k[w]);
                if (next != in[w]) {
                    in[w] = next;
                    changed = true;
                }
            }
        }
    }

    // Ranges of the shareable values.
    for (size_t b = 0; b < nb; ++b) {
        size_t p = begin[b];
        for (const auto* inst : *blocks[b]) {
            if (!inst) continue;
            ++p;
            if (inst->produces_value() && inst->result()) {
                if (const uint32_t* i = shared_index(inst->result())) ranges[*i].cover(p);
            }
            for_each_use(fn, *inst, [&](const Value* v) {
                if (const uint32_t* i = shared_index(v)) ranges[*i].cover(p);
            });
        }
        const uint64_t* in = row(live_in, b);
        const uint64_t* out = row(live_out, b);
        for (size_t w = 0; w < words; ++w) {
            for (uint64_t bits = in[w]; bits; bits &= bits - 1) ranges[w * 64 + std::countr_zero(bits)].cover(begin[b]);
            for (uint64_t bits = out[w]; bits; bits &= bits - 1) ranges[w * 64 + std::countr_zero(bits)].cover(end[b]);
        }
    }

    // A derived gcref sits in an unrooted slot, so the collector would not
    // move it: it must be dead at every GC point and at every block boundary.
    for (const Range& r : ranges) {
        const Value* v = r.val;
        if (!v->type().is_gcref() || rooted(v)) continue;
        for (size_t p = r.start; p <= r.end; ++p) {
            const Instruction* inst = at_pos[p];
            if (!inst) {
                throw_unsupported(stage, "derived gcref live across a block boundary in " +
                                                         std::string(fn.name()));
            }
            if (p != r.start && p != r.end && may_trigger_gc(*inst)) {
                throw_unsupported(stage, "derived gcref live across a GC point in " +
                                                         std::string(fn.name()));
            }
        }
    }

    // Linear scan. A slot is reused only by a range starting strictly after
    // its previous owner's last use, so an instruction's result never shares
    // a slot with one of its own operands.
    std::vector<Range> order = std::move(ranges);
    std::sort(order.begin(), order.end(), [](const Range& a, const Range& b) {
        return a.start != b.start ? a.start < b.start : a.val->id() < b.val->id();
    });
    struct Active { size_t end; int32_t slot; Root kind; };
    std::vector<Active> active;
    std::vector<int32_t> free_slots[3];  // by Root
    for (const Range& r : order) {
        for (size_t i = 0; i < active.size();) {
            if (active[i].end < r.start) {
                free_slots[static_cast<size_t>(active[i].kind)].push_back(active[i].slot);
                active[i] = active.back();
                active.pop_back();
            } else {
                ++i;
            }
        }
        const Root kind = root_of(r.val);
        auto& pool = free_slots[static_cast<size_t>(kind)];
        int32_t slot;
        if (!pool.empty()) {
            slot = pool.back();
            pool.pop_back();
        } else {
            offset += 8;
            slot = offset;
            add_root_slot(kind, slot);
        }
        layout.slot_map.insert(r.val, slot);
        active.push_back({r.end, slot, kind});
    }

    layout.size = offset;
    return layout;
}

} // namespace brass::codegen
