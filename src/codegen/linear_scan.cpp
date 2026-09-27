#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace brass::codegen {

LinearScanAllocator::LinearScanAllocator(
    LirFunction& fn,
    LivenessAnalysis& liveness,
    const CallingConvention& cc
)
    : fn_(fn), liveness_(liveness), cc_(cc) {
    init_register_pools();
}

void LinearScanAllocator::init_register_pools() {
    if (cc_.target().is_aarch64()) {
        using namespace brass::aarch64;
        available_gprs_.clear();
        // Caller-saved: X0..X11 (excluding X12, X13, X15 as use scratches, X14 as def_scratch, X16, X17 as emitter scratches, and X18 on Darwin/Windows)
        for (int i = 0; i <= 11; ++i) {
            available_gprs_.push_back(PReg::aarch64_gpr(static_cast<GPR>(i)));
        }
        if (!cc_.target().is_macos() && !cc_.target().is_windows()) {
            available_gprs_.push_back(PReg::aarch64_gpr(GPR::X18));
        }
        // Callee-saved: X19..X28 (excluding FP=X29, LR=X30, SP=31, XZR=32)
        for (int i = 19; i <= 28; ++i) {
            if (fn_.reserved_gprs & (1u << i)) continue;
            available_gprs_.push_back(PReg::aarch64_gpr(static_cast<GPR>(i)));
        }

        available_xmms_.clear();
        // Caller-saved: V0..V7, V16..V25 (excluding V26, V28, V29 as use scratches, V27 as def_scratch, V30, V31 as emitter scratches)
        for (int i = 0; i <= 7; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
        for (int i = 16; i <= 25; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
        // Callee-saved: V8..V15
        for (int i = 8; i <= 15; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
    } else {
        using namespace brass::x64;

        // Available GPRs (excluding RSP=4, RBP=5, scratch registers R10=10, R11=11, R15=15, and def_scratch R14=14)
        // Ordered with caller-saved first, then callee-saved
        std::vector<GPR> gprs = {
            GPR::RAX, GPR::RCX, GPR::RDX, GPR::R8, GPR::R9,
            GPR::RBX, GPR::RSI, GPR::RDI, GPR::R12, GPR::R13
        };

        available_gprs_.clear();
        for (GPR g : gprs) {
            if (fn_.reserved_gprs & reg_mask(g)) continue;
            available_gprs_.push_back(PReg::gpr(g));
        }

        // Available XMMs (excluding scratch XMM12, XMM13, XMM14, and def_scratch XMM15: XMM0..XMM11)
        available_xmms_.clear();
        for (int i = 0; i <= 11; ++i) {
            available_xmms_.push_back(PReg::xmm(static_cast<XMM>(i)));
        }
    }
}

void LinearScanAllocator::build_coalesce_hints() {
    coalesce_hints_.clear();
    fixed_preg_hints_.clear();
    vreg_at_preg_hints_.clear();
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                inst->opcode == LirOpcode::Movaps || inst->opcode == LirOpcode::Vmovaps ||
                inst->opcode == LirOpcode::Vmovups) {
                if (inst->defs.size() >= 1 && inst->uses.size() >= 1) {
                    const auto& def = inst->defs[0];
                    const auto& use = inst->uses[0];
                    if (def.is_vreg() && use.is_vreg()) {
                        VReg dst = def.vreg_val;
                        VReg src = use.vreg_val;
                        if (dst.reg_class == src.reg_class && dst.id != src.id) {
                            coalesce_hints_[dst.id].push_back(src);
                            coalesce_hints_[src.id].push_back(dst);
                        }
                    } else if (def.is_vreg() && use.is_preg()) {
                        fixed_preg_hints_[def.vreg_val.id].push_back(use.preg_val);
                        vreg_at_preg_hints_.emplace((static_cast<uint64_t>(def.vreg_val.id) << 32) | inst->id, use.preg_val);
                    } else if (def.is_preg() && use.is_vreg()) {
                        fixed_preg_hints_[use.vreg_val.id].push_back(def.preg_val);
                        vreg_at_preg_hints_.emplace((static_cast<uint64_t>(use.vreg_val.id) << 32) | inst->id, def.preg_val);
                    }
                }
            } else if (inst->opcode == LirOpcode::ParallelCopy) {
                size_t n = std::min(inst->defs.size(), inst->uses.size());
                for (size_t i = 0; i < n; ++i) {
                    const auto& def = inst->defs[i];
                    const auto& use = inst->uses[i];
                    if (def.is_vreg() && use.is_vreg()) {
                        VReg dst = def.vreg_val;
                        VReg src = use.vreg_val;
                        if (dst.reg_class == src.reg_class && dst.id != src.id) {
                            coalesce_hints_[dst.id].push_back(src);
                            coalesce_hints_[src.id].push_back(dst);
                        }
                    } else if (def.is_vreg() && use.is_preg()) {
                        fixed_preg_hints_[def.vreg_val.id].push_back(use.preg_val);
                        vreg_at_preg_hints_.emplace((static_cast<uint64_t>(def.vreg_val.id) << 32) | inst->id, use.preg_val);
                    } else if (def.is_preg() && use.is_vreg()) {
                        fixed_preg_hints_[use.vreg_val.id].push_back(def.preg_val);
                        vreg_at_preg_hints_.emplace((static_cast<uint64_t>(use.vreg_val.id) << 32) | inst->id, def.preg_val);
                    }
                }
            }
        }
    }
}

void LinearScanAllocator::build_constraint_index() {
    for (int c = 0; c < 2; ++c) {
        constrained_ids_[c].clear();
        constraint_words_[c].clear();
        constraint_or_[c][0].clear();
        constraint_or_[c][1].clear();
    }
    mem_index_vregs_.clear();

    // A call or safepoint takes every caller-saved register of the pools:
    // a value live across one keeps to callee-saved registers there.
    uint32_t caller_saved[2] = {0, 0};
    for (int c = 0; c < 2; ++c) {
        for (PReg reg : (c == 0 ? available_gprs_ : available_xmms_)) {
            if (reg.code < 32 && !is_callee_saved(reg)) caller_saved[c] |= (1u << reg.code);
        }
    }

    auto note_mem_index = [this](const LirOperand& op) {
        if (op.is_mem() && op.mem_val.index_vreg.is_valid() &&
            std::find(mem_index_vregs_.begin(), mem_index_vregs_.end(), op.mem_val.index_vreg) == mem_index_vregs_.end()) {
            mem_index_vregs_.push_back(op.mem_val.index_vreg);
        }
    };
    // A physical-register operand pins that register; otherwise a fixed
    // operand constraint does.
    auto pinned_by = [](const LirOperand& op, const FixedConstraint* constraint, uint32_t& gprs, uint32_t& xmms) {
        PReg reg;
        if (op.is_preg()) {
            reg = op.preg_val;
        } else if (constraint && constraint->has_fixed_preg) {
            reg = constraint->fixed_preg;
        } else {
            return;
        }
        if (!reg.is_valid() || reg.code >= 32) return;
        (reg.reg_class == RegClass::GPR ? gprs : xmms) |= (1u << reg.code);
    };

    std::vector<InstConstraints> held;
    // Instruction ids are assigned in block order by the liveness analysis, so
    // walking the blocks yields the instructions already sorted by id.
    for (const auto& block : fn_.blocks) {
        compute_held_pregs(*block, held);
        size_t idx_in_block = 0;
        for (const auto& inst : block->instructions) {
            const InstConstraints& h = held[idx_in_block++];
            uint32_t pinned[2] = {0, 0};
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                pinned_by(inst->defs[i], i < inst->def_constraints.size() ? &inst->def_constraints[i] : nullptr,
                          pinned[0], pinned[1]);
            }
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                pinned_by(inst->uses[i], i < inst->use_constraints.size() ? &inst->use_constraints[i] : nullptr,
                          pinned[0], pinned[1]);
            }
            const bool is_call = inst->is_call() || inst->opcode == LirOpcode::Safepoint;
            // A register held across the instruction blocks every value
            // covering it, like a clobber: no value's own fixed position
            // there may take it, since that would overwrite the held value.
            const uint32_t blocked[2] = {
                inst->clobbered_gprs | h.clobbered_gprs | (is_call ? caller_saved[0] : 0u),
                inst->clobbered_xmms | h.clobbered_xmms | (is_call ? caller_saved[1] : 0u),
            };
            for (int c = 0; c < 2; ++c) {
                if (blocked[c] == 0 && pinned[c] == 0 && !is_call) continue;
                constrained_ids_[c].push_back(inst->id);
                constraint_words_[c].push_back(ConstraintWord{blocked[c], pinned[c], is_call ? ~0u : 0u});
            }
            for (const auto& op : inst->defs) note_mem_index(op);
            for (const auto& op : inst->uses) note_mem_index(op);
        }
    }

    for (int c = 0; c < 2; ++c) {
        const std::vector<ConstraintWord>& words = constraint_words_[c];
        const size_t n = words.size();
        if (n == 0) continue;
        // Only a vector register can be too wide for the callee-saved ones.
        for (int wide = 0; wide < (c == 1 ? 2 : 1); ++wide) {
            auto& table = constraint_or_[c][wide];
            std::vector<uint32_t> base(n);
            for (size_t i = 0; i < n; ++i) base[i] = words[i].clob | words[i].pin | (wide ? words[i].call : 0u);
            table.push_back(std::move(base));
            for (size_t span = 2; span <= n; span *= 2) {
                const std::vector<uint32_t>& prev = table.back();
                std::vector<uint32_t> level(n - span + 1);
                for (size_t i = 0; i < level.size(); ++i) level[i] = prev[i] | prev[i + span / 2];
                table.push_back(std::move(level));
            }
        }
    }
}

// Blocks by position, each with the loop depth at which a move placed where
// it begins runs: its own depth, or its layout predecessor's when that is
// shallower (the edge into a loop header from its preheader), and a
// range-minimum table over it for choosing split positions.
void LinearScanAllocator::build_block_spans() {
    spans_.clear();
    depth_rmq_.clear();
    uint32_t prev_depth = 0;
    for (const auto& block : fn_.blocks) {
        if (block->instructions.empty()) continue;
        const BlockLiveness& bl = liveness_.block_liveness(block.get());
        BlockSpan s;
        s.start = bl.start_id;
        s.end = bl.end_id;
        s.depth = block->loop_depth;
        s.entry_depth = spans_.empty() ? s.depth : std::min(s.depth, prev_depth);
        prev_depth = s.depth;
        spans_.push_back(s);
    }
    const size_t n = spans_.size();
    if (n == 0) return;
    // Each block's run at its depth or deeper: bounded by the nearest
    // shallower block on each side (a monotonic stack each way).
    {
        std::vector<uint32_t> stack;
        for (size_t i = 0; i < n; ++i) {
            while (!stack.empty() && spans_[stack.back()].depth >= spans_[i].depth) stack.pop_back();
            spans_[i].loop_from = stack.empty() ? spans_[0].start : spans_[stack.back() + 1].start;
            stack.push_back(static_cast<uint32_t>(i));
        }
        stack.clear();
        for (size_t i = n; i-- > 0;) {
            while (!stack.empty() && spans_[stack.back()].depth >= spans_[i].depth) stack.pop_back();
            spans_[i].loop_to = stack.empty() ? spans_[n - 1].end : spans_[stack.back() - 1].end;
            stack.push_back(static_cast<uint32_t>(i));
        }
    }
    std::vector<uint32_t> base(n);
    for (size_t i = 0; i < n; ++i) base[i] = static_cast<uint32_t>(i);
    depth_rmq_.push_back(std::move(base));
    auto better = [this](uint32_t a, uint32_t b) {
        // Least depth; the later block among equals.
        if (spans_[a].entry_depth != spans_[b].entry_depth) return spans_[a].entry_depth < spans_[b].entry_depth ? a : b;
        return std::max(a, b);
    };
    for (size_t span = 2; span <= n; span *= 2) {
        const std::vector<uint32_t>& prev = depth_rmq_.back();
        std::vector<uint32_t> level(n - span + 1);
        for (size_t i = 0; i < level.size(); ++i) level[i] = better(prev[i], prev[i + span / 2]);
        depth_rmq_.push_back(std::move(level));
    }
}

// A physical-register operand is a point constraint only at the instruction
// naming it, but the value it carries lives from the instruction that writes
// the register to the last one that reads it: a call's result in RAX until
// the copy out of it, an argument in RCX from its move to the call, a
// dividend in RAX/RDX until the divide. Instruction selection emits those
// pairs adjacent, but the pre-RA scheduler may move independent instructions
// between them, and an interval assigned the register there overwrites the
// value. For each instruction of the block, out[i].clobbered_* receives the
// physical registers held across it (written before it, read after it).
// Only a write in the block opens a value: the registers read at a block's
// start (the entry block's incoming arguments) are read by its first
// instruction, a ParallelCopy.
void LinearScanAllocator::compute_held_pregs(const LirBlock& block, std::vector<InstConstraints>& out) const {
    const size_t n = block.instructions.size();
    out.assign(n, InstConstraints{});    // Per register (class, code): index of the instruction whose write opened
    // the current value (kNone = no value), and the last index already marked.
    constexpr int kNone = -1;
    int open_def[2][32];
    int marked_to[2][32];
    for (int c = 0; c < 2; ++c) {
        for (int r = 0; r < 32; ++r) {
            open_def[c][r] = kNone;
            marked_to[c][r] = kNone;
        }
    }
    auto cls = [](const PReg& p) { return p.reg_class == RegClass::GPR ? 0 : 1; };
    // xor r, r (zeroing RDX before an unsigned divide) names r as a use but
    // does not read its value.
    auto is_zeroing_idiom = [](const LirInst& inst) {
        switch (inst.opcode) {
            case LirOpcode::Xor: case LirOpcode::Xor32: case LirOpcode::Xorpd:
            case LirOpcode::Xorps: case LirOpcode::Pxor:
                break;
            default:
                return false;
        }
        return inst.uses.size() == 2 && inst.uses[0].is_preg() && inst.uses[1].is_preg() &&
               inst.uses[0].preg_val == inst.uses[1].preg_val;
    };
    for (size_t i = 0; i < n; ++i) {
        const LirInst& inst = *block.instructions[i];
        const bool reads_pregs = !is_zeroing_idiom(inst);
        for (const auto& u : inst.uses) {
            if (!reads_pregs) break;
            if (!u.is_preg() || !u.preg_val.is_valid() || u.preg_val.code >= 32) continue;
            const int c = cls(u.preg_val);
            const int r = u.preg_val.code;
            if (open_def[c][r] == kNone) continue;
            const int from = std::max(open_def[c][r], marked_to[c][r]) + 1;
            for (int k = from; k < static_cast<int>(i); ++k) {
                (c == 0 ? out[k].clobbered_gprs : out[k].clobbered_xmms) |= (1u << r);
            }
            marked_to[c][r] = std::max(marked_to[c][r], static_cast<int>(i));
        }
        for (const auto& d : inst.defs) {
            if (!d.is_preg() || !d.preg_val.is_valid() || d.preg_val.code >= 32) continue;
            // An 8- or 16-bit write keeps the rest of the register, so the
            // value it merges into stays open.
            if (d.size < 4 && open_def[cls(d.preg_val)][d.preg_val.code] != kNone) continue;
            open_def[cls(d.preg_val)][d.preg_val.code] = static_cast<int>(i);
            marked_to[cls(d.preg_val)][d.preg_val.code] = static_cast<int>(i);
        }
        // A clobber ends whatever value the register held; nothing written
        // before it can be read after it.
        for (int r = 0; r < 32; ++r) {
            const bool gpr_clob = (inst.clobbered_gprs >> r) & 1u;
            const bool xmm_clob = (inst.clobbered_xmms >> r) & 1u;
            if (gpr_clob && open_def[0][r] != static_cast<int>(i)) open_def[0][r] = kNone;
            if (xmm_clob && open_def[1][r] != static_cast<int>(i)) open_def[1][r] = kNone;
        }
    }
}

void LinearScanAllocator::allocate() {
    // 0. Build coalescing hints & detect calls
    build_coalesce_hints();
    build_constraint_index();
    build_block_spans();
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint || inst->opcode == LirOpcode::GuardExit) {
                fn_.frame.has_calls = true;
            }
        }
    }

    const size_t num_vregs = fn_.vreg_table.size();
    pieces_.clear();
    vreg_pieces_.assign(num_vregs, {});
    vreg_slot_.assign(num_vregs, -1);
    unsplittable_.assign(num_vregs, 0);
    is_mem_index_.assign(num_vregs, 0);
    for (VReg v : mem_index_vregs_) {
        if (v.id < num_vregs) is_mem_index_[v.id] = 1;
    }
    for (size_t id = 0; id < num_vregs; ++id) {
        if (fn_.vreg_table[id].vreg.is_gcref) unsplittable_[id] = 1;
    }
    def_pos_.assign(num_vregs, UINT32_MAX);
    for (const auto& interval : liveness_.intervals()) {
        if (!interval.vreg.is_valid() || interval.vreg.id >= num_vregs) continue;
        uint32_t def = UINT32_MAX;
        for (const auto& u : interval.use_positions) {
            if (!u.is_def) continue;
            if (def != UINT32_MAX && def != u.inst_id) {
                def = UINT32_MAX;
                break;
            }
            def = u.inst_id;
        }
        def_pos_[interval.vreg.id] = def;
    }
    active_.clear();
    next_seq_ = 0;
    used_callee_gprs_ = 0;
    used_callee_xmms_ = 0;
    next_spill_slot_ = 0;
    moves_before_.clear();
    moves_after_.clear();
    moves_at_start_.clear();
    moves_at_end_.clear();
    edge_splits_.clear();
    emitted_parallel_copy_ = false;

    // 1-2. Split what cannot move between locations, then scan.
    mark_unsplittable_live_ins();
    run_scan();

    // 3. The vreg table and the intervals keep the first piece's location,
    // which is the value's only one unless it was split. After allocation
    // only GC references are looked up there, and those are never split.
    for (auto& interval : liveness_.intervals()) {
        if (!interval.vreg.is_valid() || interval.vreg.id >= num_vregs) continue;
        const auto& list = vreg_pieces_[interval.vreg.id];
        if (list.empty()) continue;
        const AllocPiece& first = pieces_[list.front()];
        interval.assigned_preg = first.spilled ? PReg{} : first.reg;
        interval.assigned_spill_slot = first.spilled ? vreg_slot_[interval.vreg.id] : -1;
        VRegInfo& info = fn_.get_vreg_info(interval.vreg);
        info.assigned_preg = interval.assigned_preg;
        info.assigned_spill_slot = interval.assigned_spill_slot;
        info.is_spilled = first.spilled;
    }

    // 4. Update function frame info
    uint32_t final_callee_gprs = 0;
    uint32_t final_callee_xmms = 0;
    for (const auto& piece : pieces_) {
        if (!piece.spilled && piece.reg.is_valid()) {
            PReg preg = piece.reg;
            if (cc_.target().is_aarch64()) {
                if (preg.is_gpr() && cc_.is_callee_saved(preg.as_aarch64_gpr())) {
                    final_callee_gprs |= aarch64::reg_mask(preg.as_aarch64_gpr());
                } else if (!preg.is_gpr() && cc_.is_callee_saved(preg.as_aarch64_fpr())) {
                    final_callee_xmms |= aarch64::reg_mask(preg.as_aarch64_fpr());
                }
            } else {
                if (preg.is_gpr() && cc_.is_callee_saved(preg.as_gpr())) {
                    final_callee_gprs |= reg_mask(preg.as_gpr());
                } else if (!preg.is_gpr() && cc_.is_callee_saved(preg.as_xmm())) {
                    final_callee_xmms |= reg_mask(preg.as_xmm());
                }
            }
        }
    }

    if (!cc_.target().is_aarch64() && next_spill_slot_ > 0) {
        // If function spills on x64, scratch registers R14 and R15 may be used during rewrite.
        // Ensure they are marked as saved callee-saved GPRs in the frame layout.
        final_callee_gprs |= brass::x64::reg_mask(brass::x64::GPR::R14) | brass::x64::reg_mask(brass::x64::GPR::R15);
    }

    if (cc_.kind() == CallingConvKind::Win64) {
        if (next_spill_slot_ > 0) {
            final_callee_xmms |= brass::x64::reg_mask(brass::x64::XMM::XMM12) |
                                 brass::x64::reg_mask(brass::x64::XMM::XMM13) |
                                 brass::x64::reg_mask(brass::x64::XMM::XMM14) |
                                 brass::x64::reg_mask(brass::x64::XMM::XMM15);
        } else {
            bool uses_xmm_scratch = false;
            for (const auto& block : fn_.blocks) {
                for (const auto& inst : block->instructions) {
                    if (inst->opcode == LirOpcode::Fabs32 || inst->opcode == LirOpcode::Fabs64) {
                        uses_xmm_scratch = true;
                        break;
                    }
                    if (inst->opcode == LirOpcode::ParallelCopy && inst->defs.size() > 1) {
                        uses_xmm_scratch = true;
                        break;
                    }
                    if (!inst->defs.empty() && !inst->uses.empty() && inst->defs[0].is_mem() && inst->uses[0].is_mem()) {
                        uses_xmm_scratch = true;
                        break;
                    }
                }
                if (uses_xmm_scratch) break;
            }
            if (uses_xmm_scratch) {
                final_callee_xmms |= brass::x64::reg_mask(brass::x64::XMM::XMM15);
            }
        }
    }

    fn_.frame.num_spill_slots = next_spill_slot_;
    fn_.frame.saved_callee_gprs = final_callee_gprs | fn_.forced_saved_gprs;
    fn_.frame.saved_callee_xmms = final_callee_xmms;

    // 5. Record live GC references at all call sites and safepoints.
    record_live_gcrefs();

    // 6. Rewrite all instructions in the function, with the moves between
    // the pieces of split values, and give critical edges their blocks.
    collect_resolution_moves();
    rewrite_instructions();
    split_critical_edges();
    if (emitted_parallel_copy_ && cc_.kind() == CallingConvKind::Win64) {
        // A cycle of moves parks a value in XMM14; a memory-to-memory move
        // goes through XMM15. Both are callee-saved on Win64.
        fn_.frame.saved_callee_xmms |= brass::x64::reg_mask(brass::x64::XMM::XMM14) |
                                       brass::x64::reg_mask(brass::x64::XMM::XMM15);
    }

    // A copy the allocator gave one register at both ends is gone: the x64
    // emitter writes nothing for these, and one left between a latch's
    // `jcc` out of the loop and its `jmp` back (the block arguments' copy)
    // kept the emitter from folding the two into one branch.
    if (cc_.target().is_x64()) {
        auto same = [](const LirOperand& d, const LirOperand& s) {
            return d.is_preg() && s.is_preg() && d.preg_val == s.preg_val;
        };
        for (auto& block : fn_.blocks) {
            auto& ins = block->instructions;
            ins.erase(std::remove_if(ins.begin(), ins.end(), [&](const std::unique_ptr<LirInst>& p) {
                LirInst& i = *p;
                if (i.opcode == LirOpcode::ParallelCopy) {
                    if (i.defs.size() != i.uses.size()) return false;
                    size_t kept = 0;
                    for (size_t k = 0; k < i.defs.size(); ++k) {
                        if (same(i.defs[k], i.uses[k])) continue;
                        i.defs[kept] = i.defs[k];
                        i.uses[kept] = i.uses[k];
                        if (k < i.def_constraints.size() && kept < i.def_constraints.size()) {
                            i.def_constraints[kept] = i.def_constraints[k];
                        }
                        if (k < i.use_constraints.size() && kept < i.use_constraints.size()) {
                            i.use_constraints[kept] = i.use_constraints[k];
                        }
                        ++kept;
                    }
                    i.defs.resize(kept);
                    i.uses.resize(kept);
                    if (i.def_constraints.size() > kept) i.def_constraints.resize(kept);
                    if (i.use_constraints.size() > kept) i.use_constraints.resize(kept);
                    return kept == 0;
                }
                if (i.opcode != LirOpcode::Mov && i.opcode != LirOpcode::Movsd && i.opcode != LirOpcode::Movss) {
                    return false;
                }
                return i.defs.size() == 1 && i.uses.size() == 1 && same(i.defs[0], i.uses[0]);
            }), ins.end());
        }
    }
}

LirOperand LinearScanAllocator::location_at(VReg v, uint32_t id, uint8_t size) const {
    if (!v.is_valid() || v.id >= vreg_pieces_.size()) return LirOperand{};
    const auto& list = vreg_pieces_[v.id];
    if (list.empty()) return LirOperand{};
    size_t k = 0;
    if (list.size() > 1) {
        auto it = std::upper_bound(list.begin(), list.end(), id,
            [this](uint32_t pos, uint32_t idx) { return pos < pieces_[idx].from; });
        if (it == list.begin()) return LirOperand{};
        k = static_cast<size_t>(it - list.begin()) - 1;
    }
    const AllocPiece& p = pieces_[list[k]];
    if (id < p.from || id > p.to) return LirOperand{};
    if (p.spilled) return LirOperand::slot(vreg_slot_[v.id], size);
    if (!p.reg.is_valid()) return LirOperand{};
    return LirOperand::preg(p.reg, size);
}

bool LinearScanAllocator::is_callee_saved(PReg reg) const {
    if (cc_.target().is_aarch64()) {
        return reg.is_gpr() ? cc_.is_callee_saved(reg.as_aarch64_gpr()) : cc_.is_callee_saved(reg.as_aarch64_fpr());
    }
    return reg.is_gpr() ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
}

void LinearScanAllocator::mark_callee_saved(PReg reg) {
    if (!is_callee_saved(reg)) return;
    if (cc_.target().is_aarch64()) {
        if (reg.is_gpr()) used_callee_gprs_ |= aarch64::reg_mask(reg.as_aarch64_gpr());
        else used_callee_xmms_ |= aarch64::reg_mask(reg.as_aarch64_fpr());
    } else {
        if (reg.is_gpr()) used_callee_gprs_ |= x64::reg_mask(reg.as_gpr());
        else used_callee_xmms_ |= x64::reg_mask(reg.as_xmm());
    }
}

// One spill slot per value, shared by all its spilled pieces.
int32_t LinearScanAllocator::slot_of(VReg v) {
    int32_t& slot = vreg_slot_[v.id];
    if (slot < 0) slot = allocate_spill_slot(v.is_gcref, v.size);
    return slot;
}

void LinearScanAllocator::record_live_gcrefs() {
    // Backward walk over each block from its live-out set: at a site the
    // running set is exactly what is live after it. The site's own defs (a
    // call's result) are not live across it and uses that end at the site
    // (arguments) are not live after it. Interval position alone cannot
    // answer this: blocks are not in dominance order, so a loop body placed
    // after its exit would see the value's last use before the site.
    const size_t num_vregs = fn_.vreg_table.size();
    auto is_gcref = [&](VReg v) {
        return v.is_valid() && v.id < num_vregs && fn_.vreg_table[v.id].vreg.is_gcref;
    };
    std::vector<uint32_t> gcref_ids;
    for (uint32_t id = 0; id < num_vregs; ++id) {
        if (is_gcref(fn_.vreg_table[id].vreg)) gcref_ids.push_back(id);
    }
    std::vector<char> live(num_vregs, 0);
    for (const auto& block : fn_.blocks) {
        for (uint32_t id : gcref_ids) live[id] = 0;
        for (VReg v : liveness_.block_liveness(block.get()).live_out) {
            if (is_gcref(v)) live[v.id] = 1;
        }
        for (auto it = block->instructions.rbegin(); it != block->instructions.rend(); ++it) {
            LirInst& inst = **it;
            if (inst.is_call() || inst.opcode == LirOpcode::Safepoint) {
                inst.live_gcrefs.clear();
                for (uint32_t id : gcref_ids) {
                    if (live[id]) inst.live_gcrefs.push_back(fn_.vreg_table[id].vreg);
                }
            }
            for (const auto& d : inst.defs) {
                if (d.is_vreg() && is_gcref(d.vreg_val)) live[d.vreg_val.id] = 0;
            }
            auto mark_use = [&](VReg v) { if (is_gcref(v)) live[v.id] = 1; };
            for (const auto& d : inst.defs) {
                if (d.is_mem()) {
                    mark_use(d.mem_val.base_vreg);
                    mark_use(d.mem_val.index_vreg);
                }
            }
            for (const auto& u : inst.uses) {
                if (u.is_vreg()) {
                    mark_use(u.vreg_val);
                } else if (u.is_mem()) {
                    mark_use(u.mem_val.base_vreg);
                    mark_use(u.mem_val.index_vreg);
                }
            }
        }
    }
}

bool LinearScanAllocator::fpr_wider_than_callee_saved(const VReg& vreg) const {
    if (vreg.is_gpr()) return false;
    return vreg.size > (cc_.target().is_aarch64() ? 8u : 16u);
}

// A value wider than 8 bytes takes 2 (16 bytes) or 4 (32 bytes) adjacent
// 8-byte slots, and the slot index it is known by is the one whose address
// is lowest, where the whole value is stored. x64 slots grow downward from
// RBP (slot i at rbp - (i + 1) * 8), so that is the highest of the run;
// AArch64 slots grow upward from FP (AArch64FrameLayout::spill_slot_address),
// so it is the lowest. Getting this backwards stores a vector over the next
// value's slot.
int32_t LinearScanAllocator::allocate_spill_slot(bool is_gcref, uint8_t size) {
    const bool slots_grow_up = cc_.target().is_aarch64();
    if (size == 32) {
        while ((next_spill_slot_ % 4) != 0) {
            next_spill_slot_++;
            fn_.frame.spill_slot_is_gcref.push_back(false);
        }
        for (int i = 0; i < 4; ++i) {
            fn_.frame.spill_slot_is_gcref.push_back(false);
        }
        next_spill_slot_ += 4;
        fn_.frame.num_spill_slots = next_spill_slot_;
        int32_t slot = static_cast<int32_t>(slots_grow_up ? next_spill_slot_ - 4 : next_spill_slot_ - 1);
        return slot;
    }
    if (size == 16) {
        while ((next_spill_slot_ % 2) != 0) {
            next_spill_slot_++;
            fn_.frame.spill_slot_is_gcref.push_back(false);
        }
        fn_.frame.spill_slot_is_gcref.push_back(false);
        fn_.frame.spill_slot_is_gcref.push_back(false);
        next_spill_slot_ += 2;
        fn_.frame.num_spill_slots = next_spill_slot_;
        int32_t slot = static_cast<int32_t>(slots_grow_up ? next_spill_slot_ - 2 : next_spill_slot_ - 1);
        return slot;
    }
    int32_t slot = static_cast<int32_t>(next_spill_slot_++);
    fn_.frame.spill_slot_is_gcref.push_back(is_gcref);
    fn_.frame.num_spill_slots = next_spill_slot_;
    return slot;
}

void run_linear_scan_regalloc(
    LirFunction& fn,
    LivenessAnalysis& liveness,
    const CallingConvention& cc
) {
    LinearScanAllocator allocator(fn, liveness, cc);
    allocator.allocate();
}

} // namespace brass::codegen
