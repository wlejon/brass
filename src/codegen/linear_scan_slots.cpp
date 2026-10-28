#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <bit>

// The allocator's stack slots: which value goes in which slot (slot_of) and
// the zeroing of dead slots and callee-saved registers before calls
// (clear_dead_callee_gprs_at_calls).

namespace brass::codegen {

// One spill slot per value, shared by all its spilled pieces, and by any
// value whose live interval meets none of the slot's other owners': a slot
// only ever holds its value where that value is live (stored at its def or a
// split, read in its spilled pieces), so two values never live at the same
// position can take turns in it. Fewer slots is a smaller frame and fewer
// slots for clear_dead_callee_gprs_at_calls to zero at every call.
int32_t LinearScanAllocator::slot_of(VReg v) {
    int32_t& slot = vreg_slot_[v.id];
    if (slot >= 0) return slot;
    const LiveInterval* mine = liveness_.get_interval(v);
    const bool shareable = v.size <= 8 && mine && !mine->segments.empty();
    // Inclusive at both ends: a value read by an instruction and one written
    // by it are both live there, which keeps a reload and a write-back of the
    // same instruction out of one slot.
    //
    // The one exception is a copy between the two: where one value is copied
    // into the other and dies there, the two meet at the copy alone, and
    // sharing the slot makes that copy a move of the slot onto itself,
    // which is no move (a block argument handed through memory to its
    // parameter).
    auto meets = [this](const LiveInterval& a, const LiveInterval& b) {
        if (a.end_id < b.start_id || b.end_id < a.start_id) return false;
        const std::vector<uint32_t>* copies = nullptr;
        if (auto it = copy_positions_.find(copy_key(a.vreg.id, b.vreg.id)); it != copy_positions_.end()) {
            copies = &it->second;
        }
        for (const auto& x : a.segments) {
            for (const auto& y : b.segments) {
                const uint32_t lo = std::max(x.start, y.start);
                const uint32_t hi = std::min(x.end, y.end);
                if (lo > hi) continue;
                bool at_copy = false;
                if (copies && hi - lo <= 1) {
                    for (uint32_t c : *copies) {
                        if (lo >= c && hi <= c + 1) {
                            at_copy = true;
                            break;
                        }
                    }
                }
                if (!at_copy) return true;
            }
        }
        return false;
    };
    if (shareable) {
        // A bounded search: the slots checked and the owners compared are
        // capped, so a function with thousands of spilled values stays
        // linear and only loses some sharing.
        constexpr size_t kMaxSlotsChecked = 64;
        constexpr size_t kMaxOwners = 16;
        size_t checked = 0;
        // A slot a copy partner already holds comes first: sharing it is
        // what removes the copy.
        std::vector<SlotOwners*> order;
        if (auto h = coalesce_hints_.find(v.id); h != coalesce_hints_.end()) {
            for (VReg partner : h->second) {
                if (partner.id >= vreg_slot_.size() || vreg_slot_[partner.id] < 0) continue;
                if (copy_positions_.find(copy_key(v.id, partner.id)) == copy_positions_.end()) continue;
                for (auto& so : shared_slots_) {
                    if (so.slot == vreg_slot_[partner.id]) order.push_back(&so);
                }
            }
        }
        for (auto it = shared_slots_.rbegin(); it != shared_slots_.rend() && order.size() < kMaxSlotsChecked + 8; ++it) {
            order.push_back(&*it);
        }
        for (SlotOwners* sop : order) {
            if (checked >= kMaxSlotsChecked) break;
            SlotOwners& so = *sop;
            // One register class per slot: the dead-slot clearing reads GPR
            // liveness only, so an XMM value never shares with a GPR one.
            if (so.is_gcref != v.is_gcref || so.size != v.size || so.is_gpr != v.is_gpr() ||
                so.vregs.size() >= kMaxOwners) {
                continue;
            }
            ++checked;
            bool clash = false;
            for (uint32_t o : so.vregs) {
                const LiveInterval* other = liveness_.get_interval(fn_.vreg_table[o].vreg);
                if (!other || meets(*mine, *other)) {
                    clash = true;
                    break;
                }
            }
            if (clash) continue;
            so.vregs.push_back(v.id);
            slot = so.slot;
            return slot;
        }
    }
    slot = allocate_spill_slot(v.is_gcref, v.size);
    if (shareable) shared_slots_.push_back(SlotOwners{slot, v.is_gcref, v.is_gpr(), v.size, {v.id}});
    return slot;
}

void LinearScanAllocator::clear_dead_callee_gprs_at_calls(uint32_t final_callee_gprs) {
    const uint32_t alloc_callee_gprs = final_callee_gprs & ~fn_.reserved_gprs & ~fn_.forced_saved_gprs;
    uint64_t gpr_spill_mask = 0;
    for (size_t id = 0; id < vreg_slot_.size(); ++id) {
        int32_t s = vreg_slot_[id];
        if (s >= 0 && s < 64 && fn_.vreg_table[id].vreg.is_gpr() && fn_.vreg_table[id].vreg.size == 8) {
            gpr_spill_mask |= (uint64_t{1} << s);
        }
    }
    if ((alloc_callee_gprs == 0 && gpr_spill_mask == 0) || !fn_.frame.has_calls) return;

    const size_t num_vregs = fn_.vreg_table.size();
    const size_t num_blocks = fn_.blocks.size();
    // The live vregs as a sparse set: reset per block and walked per call in
    // time proportional to what is live. A flag per vreg, cleared per block
    // and scanned whole at every call, was blocks x vregs plus calls x vregs,
    // a fifth of allocating registers for a big function.
    std::vector<uint32_t> live_list;
    std::vector<uint32_t> live_pos(num_vregs, UINT32_MAX);
    auto live_add = [&](uint32_t id) {
        if (live_pos[id] != UINT32_MAX) return;
        live_pos[id] = static_cast<uint32_t>(live_list.size());
        live_list.push_back(id);
    };
    auto live_remove = [&](uint32_t id) {
        const uint32_t p = live_pos[id];
        if (p == UINT32_MAX) return;
        const uint32_t last = live_list.back();
        live_list[p] = last;
        live_pos[last] = p;
        live_list.pop_back();
        live_pos[id] = UINT32_MAX;
    };
    // What is live at each call, by block and instruction index.
    std::vector<std::vector<uint32_t>> live_callee_by_block(num_blocks);
    std::vector<std::vector<uint64_t>> live_spill_by_block(num_blocks);
    for (size_t bi = 0; bi < num_blocks; ++bi) {
        const auto& block = fn_.blocks[bi];
        for (uint32_t id : live_list) live_pos[id] = UINT32_MAX;
        live_list.clear();
        for (VReg v : liveness_.block_liveness(block.get()).live_out) {
            if (v.is_valid() && v.id < num_vregs && v.is_gpr()) live_add(v.id);
        }
        const size_t n_inst = block->instructions.size();
        std::vector<uint32_t>& live_callee_at = live_callee_by_block[bi];
        std::vector<uint64_t>& live_spill_at = live_spill_by_block[bi];
        live_callee_at.assign(n_inst, 0);
        live_spill_at.assign(n_inst, 0);
        for (size_t idx = n_inst; idx-- > 0;) {
            const LirInst& inst = *block->instructions[idx];
            if (inst.is_call()) {
                uint32_t mask = 0;
                uint64_t smask = 0;
                auto mark_loc = [&](const LirOperand& op) {
                    if (op.is_preg() && op.preg_val.is_valid() && op.preg_val.is_gpr() && op.preg_val.code < 32) {
                        mask |= (1u << op.preg_val.code);
                    } else if (op.is_spill_slot() && op.spill_slot >= 0 && op.spill_slot < 64) {
                        smask |= (uint64_t{1} << op.spill_slot);
                    }
                };
                auto mark_vreg = [&](VReg v) {
                    if (!v.is_valid() || v.id >= num_vregs) return;
                    mark_loc(location_at(fn_.vreg_table[v.id].vreg, inst.id, 8));
                    int32_t s = vreg_slot_[v.id];
                    if (s >= 0 && s < 64) {
                        smask |= (uint64_t{1} << s);
                    }
                };
                for (uint32_t id : live_list) mark_vreg(fn_.vreg_table[id].vreg);
                for (const auto& u : inst.uses) {
                    if (u.is_vreg()) mark_vreg(u.vreg_val);
                    else if (u.is_preg() || u.is_spill_slot()) mark_loc(u);
                    else if (u.is_mem()) {
                        if (u.mem_val.base_vreg.is_valid()) mark_vreg(u.mem_val.base_vreg);
                        if (u.mem_val.index_vreg.is_valid()) mark_vreg(u.mem_val.index_vreg);
                    }
                }
                if (auto mit = moves_before_.find(inst.id); mit != moves_before_.end()) {
                    for (const auto& pm : mit->second) {
                        mark_loc(pm.src);
                        mark_loc(pm.dst);
                    }
                }
                live_callee_at[idx] = mask;
                live_spill_at[idx] = smask;
            }
            for (const auto& d : inst.defs) {
                if (d.is_vreg() && d.vreg_val.is_valid() && d.vreg_val.id < num_vregs) {
                    live_remove(d.vreg_val.id);
                }
            }
            auto mark_use = [&](VReg v) {
                if (v.is_valid() && v.id < num_vregs && v.is_gpr()) live_add(v.id);
            };
            for (const auto& d : inst.defs) {
                if (d.is_mem()) {
                    mark_use(d.mem_val.base_vreg);
                    mark_use(d.mem_val.index_vreg);
                }
            }
            for (const auto& u : inst.uses) {
                if (u.is_vreg()) mark_use(u.vreg_val);
                else if (u.is_mem()) {
                    mark_use(u.mem_val.base_vreg);
                    mark_use(u.mem_val.index_vreg);
                } else if (u.is_label() && inst.opcode == LirOpcode::Jcc) {
                    const auto& exits = liveness_.guard_exits();
                    if (auto eit = exits.find(u.label_id); eit != exits.end()) {
                        for (const auto& gu : eit->second->uses) {
                            if (gu.is_vreg()) mark_use(gu.vreg_val);
                            else if (gu.is_mem()) {
                                mark_use(gu.mem_val.base_vreg);
                                mark_use(gu.mem_val.index_vreg);
                            }
                        }
                    }
                }
            }
        }

    }

    // Which registers and slots may hold a stale word, forward over the CFG:
    // a call clears the dead ones, so what is dirty at a call is what was
    // written since the last call on SOME path to it. Each block's entry is
    // the union of its predecessors' exits; the function's entry, and a block
    // control reaches by no edge the CFG records (a landing pad, a resume
    // target, a guard exit), starts with everything dirty, which is what a
    // block started with when the walk was one block at a time.
    struct Dirty {
        uint32_t callee = 0;
        uint64_t spill = 0;
        bool operator==(const Dirty&) const = default;
    };
    const Dirty all{alloc_callee_gprs, gpr_spill_mask};
    std::unordered_map<uint32_t, size_t> block_index;
    for (size_t bi = 0; bi < num_blocks; ++bi) block_index[fn_.blocks[bi]->id] = bi;
    auto note_into = [&](Dirty& d, const LirOperand& loc) {
        if (loc.is_preg() && loc.preg_val.is_gpr() && loc.preg_val.code < 32) {
            d.callee |= (1u << loc.preg_val.code);
        } else if (loc.is_spill_slot() && loc.spill_slot >= 0 && loc.spill_slot < 64) {
            if (gpr_spill_mask & (uint64_t{1} << loc.spill_slot)) {
                d.spill |= (uint64_t{1} << loc.spill_slot);
                if (!cc_.target().is_aarch64()) {
                    d.callee |= brass::x64::reg_mask(brass::x64::GPR::R14) |
                                brass::x64::reg_mask(brass::x64::GPR::R15);
                }
            }
        }
    };
    auto note_op_into = [&](Dirty& d, const LirOperand& op, uint32_t inst_id) {
        if (op.is_vreg() && op.vreg_val.is_valid()) {
            note_into(d, location_at(op.vreg_val, inst_id, 8));
        } else if (op.is_preg() || op.is_spill_slot()) {
            note_into(d, op);
        } else if (op.is_mem()) {
            if (op.mem_val.base_vreg.is_valid()) note_into(d, location_at(op.mem_val.base_vreg, inst_id, 8));
            if (op.mem_val.index_vreg.is_valid()) note_into(d, location_at(op.mem_val.index_vreg, inst_id, 8));
        }
    };
    auto note_moves_into = [&](Dirty& d, const std::vector<PendingMove>& moves) {
        for (const auto& pm : moves) {
            note_into(d, pm.src);
            note_into(d, pm.dst);
        }
    };
    // The moves a split edge carries are made on the way into its target.
    std::vector<Dirty> edge_writes(num_blocks);
    for (const auto& e : edge_splits_) {
        if (auto it = block_index.find(e.succ->id); it != block_index.end()) note_moves_into(edge_writes[it->second], e.moves);
    }
    std::vector<char> starts_dirty(num_blocks, 0);
    for (size_t bi = 0; bi < num_blocks; ++bi) {
        const LirBlock* block = fn_.blocks[bi].get();
        starts_dirty[bi] = bi == 0 || block->predecessors.empty() || is_entered_implicitly(block) ||
                           is_guard_exit(block);
    }

    // One block's walk from `d`: notes every location an instruction or a
    // pending move touches, and at each call either records what to clear
    // (`emit`) or only applies it.
    auto walk = [&](size_t bi, Dirty d, bool emit) -> Dirty {
        const auto& block = fn_.blocks[bi];
        if (auto mit = moves_at_start_.find(block->id); mit != moves_at_start_.end()) note_moves_into(d, mit->second);
        const auto& live_callee_at = live_callee_by_block[bi];
        const auto& live_spill_at = live_spill_by_block[bi];
        for (size_t idx = 0; idx < block->instructions.size(); ++idx) {
            const LirInst& inst = *block->instructions[idx];
            if (auto mit = moves_before_.find(inst.id); mit != moves_before_.end()) {
                for (const auto& pm : mit->second) {
                    note_op_into(d, pm.src, inst.id);
                    note_op_into(d, pm.dst, inst.id);
                }
            }
            for (const auto& u : inst.uses) note_op_into(d, u, inst.id);
            if (inst.is_call()) {
                if (emit) {
                    uint32_t to_clear = (d.callee & alloc_callee_gprs) & ~live_callee_at[idx];
                    while (to_clear != 0) {
                        uint32_t r = static_cast<uint32_t>(std::countr_zero(to_clear));
                        to_clear &= (to_clear - 1);
                        PReg preg = cc_.target().is_aarch64()
                                        ? PReg::aarch64_gpr(static_cast<aarch64::GPR>(r))
                                        : PReg::gpr(static_cast<x64::GPR>(r));
                        moves_before_[inst.id].push_back(
                            PendingMove{LirOperand::preg(preg, 8), LirOperand::imm32(0), false, 8});
                    }
                    uint64_t spill_to_clear = (d.spill & gpr_spill_mask) & ~live_spill_at[idx];
                    while (spill_to_clear != 0) {
                        uint32_t s = static_cast<uint32_t>(std::countr_zero(spill_to_clear));
                        spill_to_clear &= (spill_to_clear - 1);
                        moves_before_[inst.id].push_back(PendingMove{LirOperand::slot(static_cast<int32_t>(s), 8),
                                                                     LirOperand::imm32(0), false, 8});
                    }
                }
                d.callee = live_callee_at[idx] & alloc_callee_gprs;
                d.spill = live_spill_at[idx] & gpr_spill_mask;
            }
            for (const auto& def : inst.defs) note_op_into(d, def, inst.id);
            if (auto mit = moves_after_.find(inst.id); mit != moves_after_.end()) {
                for (const auto& pm : mit->second) {
                    note_op_into(d, pm.src, inst.id);
                    note_op_into(d, pm.dst, inst.id);
                }
            }
        }
        if (auto mit = moves_at_end_.find(block->id); mit != moves_at_end_.end()) note_moves_into(d, mit->second);
        return d;
    };

    // A block's walk only ORs in what it touches, and a call replaces what
    // is dirty, so its exit is (entry | touched) for a block without a call
    // and, for one with a call, what follows its last call whatever the
    // entry: walk(b, {}) either way. The fixpoint runs on those summaries;
    // re-walking every instruction (and looking up every operand's
    // location) on every round was most of this function's time.
    std::vector<Dirty> touched(num_blocks);
    std::vector<char> has_call(num_blocks, 0);
    for (size_t bi = 0; bi < num_blocks; ++bi) {
        touched[bi] = walk(bi, Dirty{}, false);
        for (const auto& inst : fn_.blocks[bi]->instructions) {
            if (inst->is_call()) {
                has_call[bi] = 1;
                break;
            }
        }
    }
    auto exit_of = [&](size_t bi, Dirty d) -> Dirty {
        if (has_call[bi]) return touched[bi];
        d.callee |= touched[bi].callee;
        d.spill |= touched[bi].spill;
        return d;
    };

    std::vector<Dirty> in(num_blocks), out(num_blocks);
    for (size_t bi = 0; bi < num_blocks; ++bi) in[bi] = starts_dirty[bi] ? all : Dirty{};
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t bi = 0; bi < num_blocks; ++bi) {
            Dirty d = starts_dirty[bi] ? all : Dirty{};
            for (const LirBlock* p : fn_.blocks[bi]->predecessors) {
                auto it = block_index.find(p->id);
                if (it == block_index.end()) {
                    d = all;
                    break;
                }
                d.callee |= out[it->second].callee;
                d.spill |= out[it->second].spill;
            }
            d.callee |= edge_writes[bi].callee;
            d.spill |= edge_writes[bi].spill;
            in[bi] = d;
            const Dirty o = exit_of(bi, d);
            if (!(o == out[bi])) {
                out[bi] = o;
                changed = true;
            }
        }
    }
    for (size_t bi = 0; bi < num_blocks; ++bi) walk(bi, in[bi], true);
}

}  // namespace brass::codegen
