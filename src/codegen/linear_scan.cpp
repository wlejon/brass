#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <set>

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
    using namespace brass::x64;

    // Available GPRs (excluding RSP=4 and RBP=5)
    // Ordered with caller-saved first, then callee-saved
    std::vector<GPR> gprs = {
        GPR::RAX, GPR::RCX, GPR::RDX, GPR::RSI, GPR::RDI,
        GPR::R8, GPR::R9, GPR::R10, GPR::R11,
        GPR::RBX, GPR::R12, GPR::R13, GPR::R14, GPR::R15
    };

    available_gprs_.clear();
    for (GPR g : gprs) {
        available_gprs_.push_back(PReg::gpr(g));
    }

    // Available XMMs (XMM0..XMM15)
    available_xmms_.clear();
    for (int i = 0; i < 16; ++i) {
        available_xmms_.push_back(PReg::xmm(static_cast<XMM>(i)));
    }
}

void LinearScanAllocator::allocate() {
    // 1. Collect all non-empty intervals sorted by start_id
    std::vector<LiveInterval*> unhandled;
    for (auto& interval : liveness_.intervals()) {
        if (interval.start_id <= interval.end_id && interval.vreg.is_valid()) {
            unhandled.push_back(&interval);
        }
    }

    std::sort(unhandled.begin(), unhandled.end(),
        [](const LiveInterval* a, const LiveInterval* b) {
            if (a->start_id != b->start_id) {
                return a->start_id < b->start_id;
            }
            return a->end_id < b->end_id;
        });

    active_.clear();
    used_callee_gprs_ = 0;
    used_callee_xmms_ = 0;
    next_spill_slot_ = 0;

    // 2. Process intervals in order
    for (auto* current : unhandled) {
        expire_old_intervals(current->start_id);

        if (!try_allocate_free_reg(*current)) {
            allocate_blocked_reg(*current);
        }
    }

    // 3. Update function virtual register table
    for (const auto& interval : liveness_.intervals()) {
        if (interval.vreg.is_valid()) {
            VRegInfo& info = fn_.get_vreg_info(interval.vreg);
            info.assigned_preg = interval.assigned_preg;
            info.assigned_spill_slot = interval.assigned_spill_slot;
            info.is_spilled = (interval.assigned_spill_slot >= 0);
        }
    }

    // 4. Update function frame info
    fn_.frame.num_spill_slots = next_spill_slot_;
    fn_.frame.saved_callee_gprs = used_callee_gprs_;
    fn_.frame.saved_callee_xmms = used_callee_xmms_;

    // 5. Rewrite all instructions in the function
    rewrite_instructions();
}

void LinearScanAllocator::expire_old_intervals(uint32_t current_start) {
    auto it = active_.begin();
    while (it != active_.end()) {
        if ((*it)->end_id <= current_start) {
            it = active_.erase(it);
        } else {
            ++it;
        }
    }
}

bool LinearScanAllocator::try_allocate_free_reg(LiveInterval& interval) {
    using namespace brass::x64;

    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;

    // Find registers currently in use by active intervals
    std::set<uint8_t> occupied_regs;
    for (const auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid()) {
            occupied_regs.insert(act->assigned_preg.code);
        }
    }

    // If interval has a fixed constraint, prefer matching it if possible
    for (const auto& pos : interval.use_positions) {
        if (pos.fixed_reg.is_valid() && pos.fixed_reg.reg_class == interval.vreg.reg_class) {
            uint8_t fixed_code = pos.fixed_reg.code;
            if (occupied_regs.find(fixed_code) == occupied_regs.end()) {
                // Fixed register is free
                interval.assigned_preg = pos.fixed_reg;
                if (is_gpr && cc_.is_callee_saved(static_cast<GPR>(fixed_code))) {
                    used_callee_gprs_ |= reg_mask(static_cast<GPR>(fixed_code));
                } else if (!is_gpr && cc_.is_callee_saved(static_cast<XMM>(fixed_code))) {
                    used_callee_xmms_ |= reg_mask(static_cast<XMM>(fixed_code));
                }

                // Insert into active sorted by end_id
                active_.push_back(&interval);
                std::sort(active_.begin(), active_.end(),
                    [](const LiveInterval* a, const LiveInterval* b) {
                        return a->end_id < b->end_id;
                    });
                return true;
            }
        }
    }

    // Prioritize callee-saved if spanning a call, otherwise caller-saved
    std::vector<PReg> candidates;
    if (interval.spans_call) {
        // Try callee-saved first
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
        // Then caller-saved
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (!is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
    } else {
        // Try caller-saved first
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (!is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
        // Then callee-saved
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
    }

    if (!candidates.empty()) {
        PReg chosen = candidates.front();
        interval.assigned_preg = chosen;

        if (is_gpr && cc_.is_callee_saved(chosen.as_gpr())) {
            used_callee_gprs_ |= reg_mask(chosen.as_gpr());
        } else if (!is_gpr && cc_.is_callee_saved(chosen.as_xmm())) {
            used_callee_xmms_ |= reg_mask(chosen.as_xmm());
        }

        active_.push_back(&interval);
        std::sort(active_.begin(), active_.end(),
            [](const LiveInterval* a, const LiveInterval* b) {
                return a->end_id < b->end_id;
            });
        return true;
    }

    return false;
}

void LinearScanAllocator::allocate_blocked_reg(LiveInterval& interval) {
    // Find candidate in active with latest end_id
    LiveInterval* candidate = nullptr;
    for (auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid()) {
            if (!candidate || act->end_id > candidate->end_id) {
                candidate = act;
            }
        }
    }

    if (candidate && candidate->end_id > interval.end_id) {
        // Evict candidate and give its register to interval
        interval.assigned_preg = candidate->assigned_preg;
        candidate->assigned_preg = PReg{};
        candidate->assigned_spill_slot = allocate_spill_slot(candidate->vreg.is_gcref);

        auto it = std::find(active_.begin(), active_.end(), candidate);
        if (it != active_.end()) {
            active_.erase(it);
        }

        active_.push_back(&interval);
        std::sort(active_.begin(), active_.end(),
            [](const LiveInterval* a, const LiveInterval* b) {
                return a->end_id < b->end_id;
            });
    } else {
        // Spill interval
        interval.assigned_spill_slot = allocate_spill_slot(interval.vreg.is_gcref);
        interval.assigned_preg = PReg{};
    }
}

int32_t LinearScanAllocator::allocate_spill_slot(bool is_gcref) {
    int32_t slot = static_cast<int32_t>(next_spill_slot_++);
    fn_.frame.spill_slot_is_gcref.push_back(is_gcref);
    return slot;
}

void LinearScanAllocator::rewrite_instructions() {
    using namespace brass::x64;

    auto resolve_operand = [this](const LirOperand& op) -> LirOperand {
        if (op.is_vreg()) {
            const VRegInfo& info = fn_.get_vreg_info(op.vreg_val);
            if (info.is_spilled) {
                return LirOperand::slot(info.assigned_spill_slot, op.size);
            } else if (info.assigned_preg.is_valid()) {
                return LirOperand::preg(info.assigned_preg, op.size);
            }
        } else if (op.is_mem()) {
            LirMem mem = op.mem_val;
            if (mem.base_vreg.is_valid()) {
                const VRegInfo& b_info = fn_.get_vreg_info(mem.base_vreg);
                if (b_info.assigned_preg.is_valid()) {
                    mem.base_preg = b_info.assigned_preg;
                    mem.base_vreg = VReg{};
                }
            }
            if (mem.index_vreg.is_valid()) {
                const VRegInfo& i_info = fn_.get_vreg_info(mem.index_vreg);
                if (i_info.assigned_preg.is_valid()) {
                    mem.index_preg = i_info.assigned_preg;
                    mem.index_vreg = VReg{};
                }
            }
            return LirOperand::mem_custom(mem, op.size);
        }
        return op;
    };

    for (const auto& block : fn_.blocks) {
        std::vector<std::unique_ptr<LirInst>> rewritten;

        for (auto& inst : block->instructions) {
            // Rewrite defs and uses
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                inst->defs[i] = resolve_operand(inst->defs[i]);
            }
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                inst->uses[i] = resolve_operand(inst->uses[i]);
            }

            // Check if both def and use are memory spill slots for binary/move operations
            // x86 allows at most one memory operand per instruction
            bool has_spill_def = !inst->defs.empty() && inst->defs[0].is_spill_slot();
            bool has_spill_use = false;
            size_t spill_use_idx = 0;
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    has_spill_use = true;
                    spill_use_idx = i;
                    break;
                }
            }

            if (has_spill_def && has_spill_use && inst->opcode != LirOpcode::Safepoint) {
                // Insert a temporary scratch register load before instruction
                bool is_xmm = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Addsd ||
                               inst->opcode == LirOpcode::Subsd || inst->opcode == LirOpcode::Mulsd ||
                               inst->opcode == LirOpcode::Divsd);
                PReg scratch = is_xmm ? PReg::xmm(XMM::XMM15) : PReg::gpr(GPR::R11);
                uint8_t sz = inst->uses[spill_use_idx].size;

                auto load_scratch = std::make_unique<LirInst>(
                    is_xmm ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                load_scratch->add_def(LirOperand::preg(scratch, sz));
                load_scratch->add_use(inst->uses[spill_use_idx]);
                rewritten.push_back(std::move(load_scratch));

                inst->uses[spill_use_idx] = LirOperand::preg(scratch, sz);
                rewritten.push_back(std::move(inst));
            } else {
                rewritten.push_back(std::move(inst));
            }
        }

        block->instructions = std::move(rewritten);
    }
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
