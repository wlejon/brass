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

    // Available GPRs (excluding RSP=4, RBP=5, and scratch registers R10=10, R11=11)
    // Ordered with caller-saved first, then callee-saved
    std::vector<GPR> gprs = {
        GPR::RAX, GPR::RCX, GPR::RDX, GPR::R8, GPR::R9,
        GPR::RBX, GPR::RSI, GPR::RDI, GPR::R12, GPR::R13, GPR::R14, GPR::R15
    };

    available_gprs_.clear();
    for (GPR g : gprs) {
        available_gprs_.push_back(PReg::gpr(g));
    }

    // Available XMMs (excluding scratch XMM14 and XMM15: XMM0..XMM13)
    available_xmms_.clear();
    for (int i = 0; i < 14; ++i) {
        available_xmms_.push_back(PReg::xmm(static_cast<XMM>(i)));
    }
}

void LinearScanAllocator::build_coalesce_hints() {
    coalesce_hints_.clear();
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss) {
                if (inst->defs.size() >= 1 && inst->defs[0].is_vreg() &&
                    inst->uses.size() >= 1 && inst->uses[0].is_vreg()) {
                    VReg dst = inst->defs[0].vreg_val;
                    VReg src = inst->uses[0].vreg_val;
                    if (dst.reg_class == src.reg_class && dst.id != src.id) {
                        coalesce_hints_[dst.id].push_back(src);
                        coalesce_hints_[src.id].push_back(dst);
                    }
                }
            } else if (inst->opcode == LirOpcode::ParallelCopy) {
                size_t n = std::min(inst->defs.size(), inst->uses.size());
                for (size_t i = 0; i < n; ++i) {
                    if (inst->defs[i].is_vreg() && inst->uses[i].is_vreg()) {
                        VReg dst = inst->defs[i].vreg_val;
                        VReg src = inst->uses[i].vreg_val;
                        if (dst.reg_class == src.reg_class && dst.id != src.id) {
                            coalesce_hints_[dst.id].push_back(src);
                            coalesce_hints_[src.id].push_back(dst);
                        }
                    }
                }
            }
        }
    }
}

void LinearScanAllocator::allocate() {
    // 0. Build coalescing hints & detect calls
    build_coalesce_hints();
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint || inst->opcode == LirOpcode::GuardExit) {
                fn_.frame.has_calls = true;
            }
        }
    }

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

        if (current->vreg.is_gcref && current->spans_call) {
            current->assigned_spill_slot = allocate_spill_slot(true);
            current->assigned_preg = PReg{};
            continue;
        }

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

    // 5. Record live GC references at all call sites and safepoints
    for (const auto& block : fn_.blocks) {
        for (auto& inst : block->instructions) {
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint) {
                inst->live_gcrefs.clear();
                for (const auto& interval : liveness_.intervals()) {
                    if (interval.vreg.is_valid() && interval.vreg.is_gcref) {
                        bool has_def_before = false;
                        for (const auto& pos : interval.use_positions) {
                            if (pos.is_def && pos.inst_id < inst->id) {
                                has_def_before = true;
                                break;
                            }
                        }

                        bool has_use_after = false;
                        for (const auto& pos : interval.use_positions) {
                            if (!pos.is_def && pos.inst_id >= inst->id) {
                                has_use_after = true;
                                break;
                            }
                        }

                        if (has_def_before && has_use_after && interval.covers(inst->id)) {
                            inst->live_gcrefs.push_back(interval.vreg);
                        }
                    }
                }
            }
        }
    }

    // 6. Rewrite all instructions in the function
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

std::set<uint8_t> LinearScanAllocator::get_hard_blocked_regs(const LiveInterval& interval) const {
    using namespace brass::x64;

    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;

    std::set<uint8_t> blocked;

    if (interval.spans_call) {
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (!is_callee) {
                blocked.insert(reg.code);
            }
        }
    }

    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (!interval.covers(inst->id)) continue;

            if (is_gpr && inst->clobbered_gprs != 0) {
                for (int i = 0; i < 16; ++i) {
                    if (inst->clobbered_gprs & (1u << i)) {
                        blocked.insert(PReg::gpr(static_cast<GPR>(i)).code);
                    }
                }
            } else if (!is_gpr && inst->clobbered_xmms != 0) {
                for (int i = 0; i < 16; ++i) {
                    if (inst->clobbered_xmms & (1u << i)) {
                        blocked.insert(PReg::xmm(static_cast<XMM>(i)).code);
                    }
                }
            }

            for (size_t i = 0; i < inst->defs.size(); ++i) {
                if (inst->defs[i].is_preg() && inst->defs[i].preg_val.reg_class == interval.vreg.reg_class) {
                    bool is_our_pos = false;
                    for (const auto& pos : interval.use_positions) {
                        if (pos.inst_id == inst->id && pos.fixed_reg == inst->defs[i].preg_val) {
                            is_our_pos = true;
                            break;
                        }
                    }
                    if (!is_our_pos) {
                        blocked.insert(inst->defs[i].preg_val.code);
                    }
                } else if (i < inst->def_constraints.size() && inst->def_constraints[i].has_fixed_preg) {
                    PReg fixed_r = inst->def_constraints[i].fixed_preg;
                    if (fixed_r.reg_class == interval.vreg.reg_class) {
                        bool is_our_pos = false;
                        for (const auto& pos : interval.use_positions) {
                            if (pos.inst_id == inst->id && pos.fixed_reg == fixed_r) {
                                is_our_pos = true;
                                break;
                            }
                        }
                        if (!is_our_pos) {
                            blocked.insert(fixed_r.code);
                        }
                    }
                }
            }

            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_preg() && inst->uses[i].preg_val.reg_class == interval.vreg.reg_class) {
                    bool is_our_pos = false;
                    for (const auto& pos : interval.use_positions) {
                        if (pos.inst_id == inst->id && pos.fixed_reg == inst->uses[i].preg_val) {
                            is_our_pos = true;
                            break;
                        }
                    }
                    if (!is_our_pos) {
                        blocked.insert(inst->uses[i].preg_val.code);
                    }
                } else if (i < inst->use_constraints.size() && inst->use_constraints[i].has_fixed_preg) {
                    PReg fixed_r = inst->use_constraints[i].fixed_preg;
                    if (fixed_r.reg_class == interval.vreg.reg_class) {
                        bool is_our_pos = false;
                        for (const auto& pos : interval.use_positions) {
                            if (pos.inst_id == inst->id && pos.fixed_reg == fixed_r) {
                                is_our_pos = true;
                                break;
                            }
                        }
                        if (!is_our_pos) {
                            blocked.insert(fixed_r.code);
                        }
                    }
                }
            }
        }
    }

    return blocked;
}

std::set<uint8_t> LinearScanAllocator::get_occupied_regs(const LiveInterval& interval) const {
    std::set<uint8_t> occupied_regs = get_hard_blocked_regs(interval);
    for (const auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid()) {
            if (act->overlaps(interval)) {
                occupied_regs.insert(act->assigned_preg.code);
            }
        }
    }
    return occupied_regs;
}

bool LinearScanAllocator::try_allocate_free_reg(LiveInterval& interval) {
    using namespace brass::x64;

    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;

    std::set<uint8_t> occupied_regs = get_occupied_regs(interval);

    // 1. If interval has a fixed constraint, check if that fixed register is valid
    for (const auto& pos : interval.use_positions) {
        if (pos.fixed_reg.is_valid() && pos.fixed_reg.reg_class == interval.vreg.reg_class) {
            uint8_t fixed_code = pos.fixed_reg.code;
            if (occupied_regs.find(fixed_code) == occupied_regs.end()) {
                interval.assigned_preg = pos.fixed_reg;
                if (is_gpr && cc_.is_callee_saved(static_cast<GPR>(fixed_code))) {
                    used_callee_gprs_ |= reg_mask(static_cast<GPR>(fixed_code));
                } else if (!is_gpr && cc_.is_callee_saved(static_cast<XMM>(fixed_code))) {
                    used_callee_xmms_ |= reg_mask(static_cast<XMM>(fixed_code));
                }

                active_.push_back(&interval);
                std::sort(active_.begin(), active_.end(),
                    [](const LiveInterval* a, const LiveInterval* b) {
                        return a->end_id < b->end_id;
                    });
                return true;
            }
        }
    }

    // 2. Try Coalescing Hint from partners
    auto hint_it = coalesce_hints_.find(interval.vreg.id);
    if (hint_it != coalesce_hints_.end()) {
        for (VReg partner_v : hint_it->second) {
            const auto* p_int = liveness_.get_interval(partner_v);
            if (p_int && p_int->assigned_preg.is_valid() && p_int->assigned_preg.reg_class == interval.vreg.reg_class) {
                PReg hint_reg = p_int->assigned_preg;
                if (occupied_regs.find(hint_reg.code) == occupied_regs.end()) {
                    bool is_callee = is_gpr ? cc_.is_callee_saved(hint_reg.as_gpr()) : cc_.is_callee_saved(hint_reg.as_xmm());
                    if (!interval.spans_call || is_callee) {
                        interval.assigned_preg = hint_reg;
                        if (is_gpr && is_callee) {
                            used_callee_gprs_ |= reg_mask(hint_reg.as_gpr());
                        } else if (!is_gpr && is_callee) {
                            used_callee_xmms_ |= reg_mask(hint_reg.as_xmm());
                        }

                        active_.push_back(&interval);
                        std::sort(active_.begin(), active_.end(),
                            [](const LiveInterval* a, const LiveInterval* b) {
                                return a->end_id < b->end_id;
                            });
                        return true;
                    }
                }
            }
        }
    }

    // 3. Register Pool Priority
    std::vector<PReg> candidates;
    if (interval.spans_call) {
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
    } else {
        for (const auto& reg : pool) {
            bool is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
            if (!is_callee && occupied_regs.find(reg.code) == occupied_regs.end()) {
                candidates.push_back(reg);
            }
        }
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
    using namespace brass::x64;

    auto hard_blocked = get_hard_blocked_regs(interval);

    // Find candidate in active with lowest spill_weight (or furthest end_id for equal weights)
    LiveInterval* candidate = nullptr;
    for (auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid()) {
            if (hard_blocked.find(act->assigned_preg.code) != hard_blocked.end()) {
                continue;
            }
            if (!candidate) {
                candidate = act;
            } else if (act->spill_weight < candidate->spill_weight) {
                candidate = act;
            } else if (act->spill_weight == candidate->spill_weight && act->end_id > candidate->end_id) {
                candidate = act;
            }
        }
    }

    if (candidate && candidate->spill_weight < interval.spill_weight) {
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
    } else if (candidate && candidate->end_id > interval.end_id && candidate->spill_weight <= interval.spill_weight) {
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
            // First, check if any memory operands have spilled base or index registers
            for (auto* op_list : {&inst->defs, &inst->uses}) {
                for (auto& op : *op_list) {
                    if (op.is_mem()) {
                        if (op.mem_val.base_vreg.is_valid()) {
                            const VRegInfo& b_info = fn_.get_vreg_info(op.mem_val.base_vreg);
                            if (b_info.is_spilled) {
                                auto load_base = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_base->add_def(LirOperand::preg(PReg::gpr(GPR::R10), 8));
                                load_base->add_use(LirOperand::slot(b_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_base));

                                op.mem_val.base_preg = PReg::gpr(GPR::R10);
                                op.mem_val.base_vreg = VReg{};
                            }
                        }
                        if (op.mem_val.index_vreg.is_valid()) {
                            const VRegInfo& i_info = fn_.get_vreg_info(op.mem_val.index_vreg);
                            if (i_info.is_spilled) {
                                auto load_idx = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_idx->add_def(LirOperand::preg(PReg::gpr(GPR::R11), 8));
                                load_idx->add_use(LirOperand::slot(i_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_idx));

                                op.mem_val.index_preg = PReg::gpr(GPR::R11);
                                op.mem_val.index_vreg = VReg{};
                            }
                        }
                    }
                }
            }

            // Rewrite defs and uses
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                inst->defs[i] = resolve_operand(inst->defs[i]);
            }
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                inst->uses[i] = resolve_operand(inst->uses[i]);
            }

            if (inst->opcode == LirOpcode::Safepoint) {
                rewritten.push_back(std::move(inst));
                continue;
            }

            bool def_is_mem = !inst->defs.empty() && (inst->defs[0].is_mem() || inst->defs[0].is_spill_slot());
            bool use_is_mem = !inst->uses.empty() && (inst->uses[0].is_mem() || inst->uses[0].is_spill_slot());

            // Moves between memory / spill slots:
            if ((inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                 inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss) &&
                def_is_mem && use_is_mem) {
                bool is_xmm = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss);
                PReg scratch = is_xmm ? PReg::xmm(XMM::XMM15) : PReg::gpr(GPR::R11);
                uint8_t sz = inst->uses[0].size;

                auto load_scratch = std::make_unique<LirInst>(
                    is_xmm ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                load_scratch->add_def(LirOperand::preg(scratch, sz));
                load_scratch->add_use(inst->uses[0]);
                rewritten.push_back(std::move(load_scratch));

                auto store_scratch = std::make_unique<LirInst>(
                    is_xmm ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                store_scratch->add_def(inst->defs[0]);
                store_scratch->add_use(LirOperand::preg(scratch, sz));
                rewritten.push_back(std::move(store_scratch));
                continue;
            }

            bool has_spill_def = !inst->defs.empty() && inst->defs[0].is_spill_slot();

            // Other instructions with spill def or spill uses
            LirOperand original_spill_def;
            PReg def_scratch;
            bool is_xmm_def = false;

            if (has_spill_def) {
                original_spill_def = inst->defs[0];
                is_xmm_def = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                              inst->opcode == LirOpcode::Addsd || inst->opcode == LirOpcode::Subsd ||
                              inst->opcode == LirOpcode::Mulsd || inst->opcode == LirOpcode::Divsd ||
                              inst->opcode == LirOpcode::Sqrtsd || inst->opcode == LirOpcode::Xorpd ||
                              inst->opcode == LirOpcode::Cvtsi2sd || inst->opcode == LirOpcode::Cvtsi2sd32 ||
                              inst->opcode == LirOpcode::Movq_xg);
                def_scratch = is_xmm_def ? PReg::xmm(XMM::XMM15) : PReg::gpr(GPR::R11);
                uint8_t sz = original_spill_def.size;

                // If instruction reads from def (e.g. add dst, src), load initial value of def into scratch
                bool reads_def = false;
                for (size_t i = 0; i < inst->uses.size(); ++i) {
                    if (inst->uses[i].is_spill_slot() && inst->uses[i].spill_slot == original_spill_def.spill_slot) {
                        reads_def = true;
                        inst->uses[i] = LirOperand::preg(def_scratch, sz);
                    }
                }

                if (reads_def) {
                    auto load_def = std::make_unique<LirInst>(
                        is_xmm_def ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                    );
                    load_def->add_def(LirOperand::preg(def_scratch, sz));
                    load_def->add_use(original_spill_def);
                    rewritten.push_back(std::move(load_def));
                }

                inst->defs[0] = LirOperand::preg(def_scratch, sz);
            }

            // Handle any remaining spill uses with reserved scratch R10 (GPR) or XMM14 (XMM)
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    bool is_xmm_use = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                                      inst->opcode == LirOpcode::Addsd || inst->opcode == LirOpcode::Subsd ||
                                      inst->opcode == LirOpcode::Mulsd || inst->opcode == LirOpcode::Divsd ||
                                      inst->opcode == LirOpcode::Sqrtsd || inst->opcode == LirOpcode::Ucomisd ||
                                      inst->opcode == LirOpcode::Xorpd || inst->opcode == LirOpcode::Cvttsd2si ||
                                      inst->opcode == LirOpcode::Cvttsd2si32 || inst->opcode == LirOpcode::Movq_gx);
                    PReg use_scratch = is_xmm_use ? PReg::xmm(XMM::XMM14) : PReg::gpr(GPR::R10);
                    uint8_t sz = inst->uses[i].size;

                    bool already_loaded = false;
                    for (size_t j = 0; j < i; ++j) {
                        if (inst->uses[j].is_preg() && inst->uses[j].preg_val == use_scratch) {
                            already_loaded = true;
                            break;
                        }
                    }

                    if (!already_loaded) {
                        auto load_use = std::make_unique<LirInst>(
                            is_xmm_use ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                        );
                        load_use->add_def(LirOperand::preg(use_scratch, sz));
                        load_use->add_use(inst->uses[i]);
                        rewritten.push_back(std::move(load_use));
                    }

                    inst->uses[i] = LirOperand::preg(use_scratch, sz);
                }
            }

            rewritten.push_back(std::move(inst));

            // Write back to spill slot if def was spilled
            if (has_spill_def) {
                uint8_t sz = original_spill_def.size;
                auto store_back = std::make_unique<LirInst>(
                    is_xmm_def ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                store_back->add_def(original_spill_def);
                store_back->add_use(LirOperand::preg(def_scratch, sz));
                rewritten.push_back(std::move(store_back));
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
