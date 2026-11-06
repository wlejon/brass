#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

// x64 constant slots. A 64-bit constant the allocator spills (a NaN-box tag,
// a payload mask, a limit: bronze hoists one of each to the function entry,
// and the guards in every block read it) was stored to its slot where it was
// made, `movabs r14, C; mov [rbp - x], r14`, and read from there as a folded
// operand. The reads are free; the stores were not: two to twelve of them at
// every entry of a small function, whatever path it took. A slot that only
// ever holds one constant needs no store when its reads take the constant
// from memory that already has it: the function's constant pool, which the
// emitter places after the code (EmitContext::emit_constant_pool) and reads
// rip-relative. Such a read encodes like a slot read, one instruction.

namespace brass::codegen {

namespace {

// Whether every machine instruction the x64 emitter makes of `in` that reads
// a memory operand ends with that operand's displacement: an opcode that
// reads its r/m operand plainly, with no immediate operand anywhere. A
// rip-relative displacement is relative to the end of its instruction, and
// the pool's fixup takes that end to be the displacement's (rip_label_ref).
bool reads_pool_operand(const LirInst& in) {
    switch (in.opcode) {
        case LirOpcode::Mov:
        case LirOpcode::Add:
        case LirOpcode::Sub:
        case LirOpcode::And:
        case LirOpcode::Or:
        case LirOpcode::Xor:
        case LirOpcode::Cmp:
        case LirOpcode::Test:
        case LirOpcode::Cmovcc:
        case LirOpcode::Movsd:
        case LirOpcode::Movq_gx:
            break;
        default:
            return false;
    }
    for (const auto& u : in.uses) {
        if (u.is_imm_int() || u.is_imm_float()) return false;
    }
    return true;
}

// A store of the constant into its slot, which the pool makes redundant:
// `mov [slot], reg` or `mov [slot], imm`.
bool is_slot_store(const LirInst& in, int32_t slot) {
    return in.opcode == LirOpcode::Mov && in.defs.size() == 1 && in.uses.size() == 1 &&
           in.defs[0].is_spill_slot() && in.defs[0].spill_slot == slot && in.defs[0].size == 8 &&
           (in.uses[0].is_preg() || in.uses[0].is_imm_int());
}

} // namespace

void LinearScanAllocator::collect_constant_vregs() {
    constant_vregs_.clear();
    if (!cc_.target().is_x64()) return;
    std::unordered_map<uint32_t, uint32_t> defs;
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            for (const auto& d : inst->defs) {
                if (d.is_vreg()) defs[d.vreg_val.id]++;
            }
            const LirInst& in = *inst;
            if ((in.opcode == LirOpcode::Movabs || in.opcode == LirOpcode::Mov) && !in.is_patchable &&
                in.defs.size() == 1 && in.uses.size() == 1 && in.defs[0].is_vreg() && in.defs[0].size == 8 &&
                in.defs[0].vreg_val.is_gpr() && !in.defs[0].vreg_val.is_gcref && in.uses[0].is_imm_int()) {
                constant_vregs_[in.defs[0].vreg_val.id] = in.uses[0].imm_int;
            }
        }
    }
    for (auto it = constant_vregs_.begin(); it != constant_vregs_.end();) {
        it = defs[it->first] == 1 ? std::next(it) : constant_vregs_.erase(it);
    }
}

void LinearScanAllocator::find_constant_slots() {
    constant_slot_candidates_.clear();
    if (constant_vregs_.empty()) return;
    std::unordered_map<int32_t, uint32_t> owners;
    for (int32_t s : vreg_slot_) {
        if (s >= 0) owners[s]++;
    }
    for (const auto& [id, c] : constant_vregs_) {
        if (id >= vreg_slot_.size()) continue;
        const int32_t s = vreg_slot_[id];
        if (s < 0 || owners[s] != 1) continue;
        if (static_cast<size_t>(s) < fn_.frame.spill_slot_is_gcref.size() && fn_.frame.spill_slot_is_gcref[s]) continue;
        constant_slot_candidates_.emplace_back(s, c);
    }
}

void LinearScanAllocator::pool_constant_slots() {
    if (constant_slot_candidates_.empty()) return;
    std::unordered_map<int32_t, int64_t> slots(constant_slot_candidates_.begin(), constant_slot_candidates_.end());
    std::unordered_set<int32_t> refused;
    // Every write of the slot must be a store of the constant, and every read
    // one the pool can serve. A parallel copy reads each source with a move
    // of its own, and its moves into the slot are dropped like a store.
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            const LirInst& in = *inst;
            const bool pcopy = in.opcode == LirOpcode::ParallelCopy;
            for (size_t k = 0; k < in.defs.size(); ++k) {
                const LirOperand& d = in.defs[k];
                if (!d.is_spill_slot() || !slots.count(d.spill_slot)) continue;
                const bool ok = pcopy ? (d.size == 8 && k < in.uses.size() &&
                                         (in.uses[k].is_preg() || in.uses[k].is_imm_int() ||
                                          (in.uses[k].is_spill_slot() && in.uses[k].spill_slot == d.spill_slot)))
                                      : is_slot_store(in, d.spill_slot);
                if (!ok) refused.insert(d.spill_slot);
            }
            for (const auto& u : in.uses) {
                if (!u.is_spill_slot() || !slots.count(u.spill_slot)) continue;
                if (!pcopy && !reads_pool_operand(in) && !is_slot_store(in, u.spill_slot)) refused.insert(u.spill_slot);
            }
        }
    }
    for (int32_t s : refused) slots.erase(s);
    if (slots.empty()) return;
    for (auto& block : fn_.blocks) {
        auto& ins = block->instructions;
        for (size_t i = 0; i < ins.size(); ++i) {
            LirInst& in = *ins[i];
            if (in.opcode == LirOpcode::ParallelCopy) {
                size_t kept = 0;
                for (size_t k = 0; k < in.defs.size(); ++k) {
                    if (in.defs[k].is_spill_slot() && slots.count(in.defs[k].spill_slot)) continue;
                    in.defs[kept] = in.defs[k];
                    if (k < in.uses.size()) in.uses[kept] = in.uses[k];
                    if (k < in.def_constraints.size()) in.def_constraints[kept] = in.def_constraints[k];
                    if (k < in.use_constraints.size()) in.use_constraints[kept] = in.use_constraints[k];
                    ++kept;
                }
                if (kept != in.defs.size()) {
                    in.defs.resize(kept);
                    in.uses.resize(kept);
                    if (in.def_constraints.size() > kept) in.def_constraints.resize(kept);
                    if (in.use_constraints.size() > kept) in.use_constraints.resize(kept);
                    if (kept == 0) ins[i].reset();
                }
                continue;
            }
            if (in.defs.size() != 1 || !in.defs[0].is_spill_slot() || !slots.count(in.defs[0].spill_slot)) continue;
            // The rewrite makes a spilled constant in a def scratch and
            // stores it from there; the scratch is read by nothing else.
            if (i > 0 && ins[i - 1] && def_scratch_stores_.count(&in) && in.uses[0].is_preg()) {
                const LirInst& prev = *ins[i - 1];
                if ((prev.opcode == LirOpcode::Movabs || prev.opcode == LirOpcode::Mov) && prev.defs.size() == 1 &&
                    prev.defs[0].is_preg() && prev.defs[0].preg_val == in.uses[0].preg_val && prev.uses.size() == 1 &&
                    prev.uses[0].is_imm_int()) {
                    ins[i - 1].reset();
                }
            }
            ins[i].reset();
        }
        ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    }
    for (const auto& [s, c] : slots) fn_.frame.constant_slots.emplace_back(s, c);
}

} // namespace brass::codegen
