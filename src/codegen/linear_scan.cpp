#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <bit>

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
        // Caller-saved: X0..X11 (excluding X12, X13 as use scratches, X14 as def_scratch, X15, X16, X17 as emitter scratches, and X18 on Darwin/Windows)
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
        // Caller-saved: V0..V7, V16..V26 (excluding V27 as def_scratch, V28, V29 as use scratches, V30, V31 as emitter scratches)
        for (int i = 0; i <= 7; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
        for (int i = 16; i <= 26; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
        // Callee-saved: V8..V15
        for (int i = 8; i <= 15; ++i) {
            available_xmms_.push_back(PReg::aarch64_fpr(static_cast<FPR>(i)));
        }
    } else {
        using namespace brass::x64;

        // Available GPRs (excluding RSP=4, RBP=5, and scratch registers R10=10, R11=11)
        // Ordered with caller-saved first, then callee-saved
        std::vector<GPR> gprs = {
            GPR::RAX, GPR::RCX, GPR::RDX, GPR::R8, GPR::R9,
            GPR::RBX, GPR::RSI, GPR::RDI, GPR::R12, GPR::R13, GPR::R14, GPR::R15
        };

        available_gprs_.clear();
        for (GPR g : gprs) {
            if (fn_.reserved_gprs & reg_mask(g)) continue;
            available_gprs_.push_back(PReg::gpr(g));
        }

        // Available XMMs (excluding scratch XMM13, XMM14, and XMM15: XMM0..XMM12)
        available_xmms_.clear();
        for (int i = 0; i <= 12; ++i) {
            available_xmms_.push_back(PReg::xmm(static_cast<XMM>(i)));
        }
    }
}

void LinearScanAllocator::build_coalesce_hints() {
    coalesce_hints_.clear();
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                inst->opcode == LirOpcode::Movaps || inst->opcode == LirOpcode::Vmovaps ||
                inst->opcode == LirOpcode::Vmovups) {
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

void LinearScanAllocator::build_constraint_index() {
    constrained_ids_.clear();
    constraint_or_.clear();
    mem_index_vregs_.clear();

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

    std::vector<InstConstraints> words;
    // Instruction ids are assigned in block order by the liveness analysis, so
    // walking the blocks yields the instructions already sorted by id.
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            uint32_t pinned_gprs = 0, pinned_xmms = 0;
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                pinned_by(inst->defs[i], i < inst->def_constraints.size() ? &inst->def_constraints[i] : nullptr,
                          pinned_gprs, pinned_xmms);
            }
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                pinned_by(inst->uses[i], i < inst->use_constraints.size() ? &inst->use_constraints[i] : nullptr,
                          pinned_gprs, pinned_xmms);
            }
            if (inst->clobbered_gprs != 0 || inst->clobbered_xmms != 0 || pinned_gprs != 0 || pinned_xmms != 0) {
                constrained_ids_.push_back(inst->id);
                words.push_back(InstConstraints{
                    inst->clobbered_gprs,
                    inst->clobbered_xmms,
                    pinned_gprs,
                    pinned_xmms
                });
            }
            for (const auto& op : inst->defs) note_mem_index(op);
            for (const auto& op : inst->uses) note_mem_index(op);
        }
    }

    const size_t n = words.size();
    if (n == 0) return;
    constraint_or_.push_back(std::move(words));
    for (size_t span = 2; span <= n; span *= 2) {
        const std::vector<InstConstraints>& prev = constraint_or_.back();
        std::vector<InstConstraints> level(n - span + 1);
        for (size_t i = 0; i < level.size(); ++i) {
            level[i] = prev[i] | prev[i + span / 2];
        }
        constraint_or_.push_back(std::move(level));
    }
}

// The OR of the constraint words of constrained instructions [lo, hi).
InstConstraints LinearScanAllocator::constraint_or(size_t lo, size_t hi) const noexcept {
    if (lo >= hi) return InstConstraints{};
    const size_t k = static_cast<size_t>(std::bit_width(hi - lo) - 1);
    return constraint_or_[k][lo] | constraint_or_[k][hi - (size_t{1} << k)];
}

void LinearScanAllocator::allocate() {
    // 0. Build coalescing hints & detect calls
    build_coalesce_hints();
    build_constraint_index();
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
            current->assigned_spill_slot = allocate_spill_slot(true, current->vreg.size);
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
    uint32_t final_callee_gprs = 0;
    uint32_t final_callee_xmms = 0;
    for (const auto& interval : liveness_.intervals()) {
        if (interval.vreg.is_valid() && interval.assigned_preg.is_valid()) {
            PReg preg = interval.assigned_preg;
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

    fn_.frame.num_spill_slots = next_spill_slot_;
    fn_.frame.saved_callee_gprs = final_callee_gprs | fn_.forced_saved_gprs;
    fn_.frame.saved_callee_xmms = final_callee_xmms;

    // 5. Record live GC references at all call sites and safepoints. A gcref
    //    is live at a site when it has a definition before the site, a use at
    //    or after it, and its interval covers it — so walk each gcref interval
    //    once over just the sites between its first def and its last use, in
    //    interval order so each site lists its gcrefs in that order.
    std::vector<LirInst*> sites;  // in id order
    for (const auto& block : fn_.blocks) {
        for (auto& inst : block->instructions) {
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint) {
                inst->live_gcrefs.clear();
                sites.push_back(inst.get());
            }
        }
    }
    for (const auto& interval : liveness_.intervals()) {
        if (!interval.vreg.is_valid() || !interval.vreg.is_gcref) continue;
        uint32_t first_def = UINT32_MAX;
        uint32_t last_use = 0;
        bool has_def = false, has_use = false;
        for (const auto& pos : interval.use_positions) {
            if (pos.is_def) {
                has_def = true;
                first_def = std::min(first_def, pos.inst_id);
            } else {
                has_use = true;
                last_use = std::max(last_use, pos.inst_id);
            }
        }
        if (!has_def || !has_use || first_def >= last_use) continue;
        // Sites with first_def < id <= last_use.
        auto it = std::upper_bound(sites.begin(), sites.end(), first_def,
            [](uint32_t id, const LirInst* inst) { return id < inst->id; });
        for (; it != sites.end() && (*it)->id <= last_use; ++it) {
            if (interval.covers((*it)->id)) {
                (*it)->live_gcrefs.push_back(interval.vreg);
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

uint32_t LinearScanAllocator::get_hard_blocked_regs(const LiveInterval& interval) const {
    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;

    uint32_t blocked = 0;
    auto block = [&blocked](PReg reg) { blocked |= (1u << reg.code); };

    if (interval.spans_call) {
        if (interval.vreg.is_gcref) {
            for (const auto& reg : pool) {
                block(reg);
            }
        } else {
            for (const auto& reg : pool) {
                bool is_callee = false;
                if (cc_.target().is_aarch64()) {
                    is_callee = is_gpr ? cc_.is_callee_saved(reg.as_aarch64_gpr()) : cc_.is_callee_saved(reg.as_aarch64_fpr());
                } else {
                    is_callee = is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
                }
                if (!is_callee) {
                    block(reg);
                }
            }
        }
    }

    // Every register clobbered or pinned by an instruction the interval covers
    // blocks it, except a pin that is this interval's own fixed position
    // there: at such an instruction the pins minus its own fixed registers
    // count. Use positions are in id order, so the instructions to except
    // are met in order while walking each segment's constrained range.
    auto both = [&](const InstConstraints& c) -> uint32_t {
        return is_gpr ? (c.clobbered_gprs | c.pinned_gprs) : (c.clobbered_xmms | c.pinned_xmms);
    };
    auto clobbered = [&](const InstConstraints& c) -> uint32_t {
        return is_gpr ? c.clobbered_gprs : c.clobbered_xmms;
    };
    auto pinned = [&](const InstConstraints& c) -> uint32_t {
        return is_gpr ? c.pinned_gprs : c.pinned_xmms;
    };

    auto own_fixed = interval.use_positions.begin();
    const auto own_fixed_end = interval.use_positions.end();
    for (const auto& seg : interval.segments) {
        size_t cur = std::lower_bound(constrained_ids_.begin(), constrained_ids_.end(), seg.start) - constrained_ids_.begin();
        const size_t hi = std::upper_bound(constrained_ids_.begin(), constrained_ids_.end(), seg.end) - constrained_ids_.begin();
        while (own_fixed != own_fixed_end && own_fixed->inst_id < seg.start) ++own_fixed;
        while (own_fixed != own_fixed_end && own_fixed->inst_id <= seg.end) {
            const uint32_t at = own_fixed->inst_id;
            uint32_t own = 0;
            for (; own_fixed != own_fixed_end && own_fixed->inst_id == at; ++own_fixed) {
                if (own_fixed->fixed_reg.is_valid() && own_fixed->fixed_reg.reg_class == interval.vreg.reg_class &&
                    own_fixed->fixed_reg.code < 32) {
                    own |= (1u << own_fixed->fixed_reg.code);
                }
            }
            if (own == 0) continue;
            const size_t idx = std::lower_bound(constrained_ids_.begin() + cur, constrained_ids_.begin() + hi, at) -
                               constrained_ids_.begin();
            if (idx >= hi || constrained_ids_[idx] != at) continue;
            blocked |= both(constraint_or(cur, idx));
            const InstConstraints& w = constraint_or_[0][idx];
            blocked |= clobbered(w);
            blocked |= pinned(w) & ~own;
            cur = idx + 1;
        }
        blocked |= both(constraint_or(cur, hi));
    }

    if (cc_.target().is_x64()) {
        if (is_gpr && std::find(mem_index_vregs_.begin(), mem_index_vregs_.end(), interval.vreg) != mem_index_vregs_.end()) {
            block(PReg::gpr(x64::GPR::R12));
            block(PReg::gpr(x64::GPR::RSP));
        }
    }

    return blocked;
}

uint32_t LinearScanAllocator::get_occupied_regs(const LiveInterval& interval) const {
    uint32_t occupied_regs = get_hard_blocked_regs(interval);
    for (const auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid() &&
            act->assigned_preg.code < 32) {
            if (act->overlaps(interval)) {
                occupied_regs |= (1u << act->assigned_preg.code);
            }
        }
    }
    return occupied_regs;
}

static bool in_mask(uint32_t mask, uint8_t code) noexcept {
    if (code >= 32) return false;
    return (mask >> code) & 1u;
}

bool LinearScanAllocator::try_allocate_free_reg(LiveInterval& interval) {
    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;

    const uint32_t occupied_regs = get_occupied_regs(interval);

    auto check_is_callee = [this, is_gpr](PReg reg) {
        if (cc_.target().is_aarch64()) {
            return is_gpr ? cc_.is_callee_saved(reg.as_aarch64_gpr()) : cc_.is_callee_saved(reg.as_aarch64_fpr());
        } else {
            return is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
        }
    };
    auto mark_callee_saved = [this, is_gpr](PReg reg) {
        if (cc_.target().is_aarch64()) {
            if (is_gpr && cc_.is_callee_saved(reg.as_aarch64_gpr())) {
                used_callee_gprs_ |= aarch64::reg_mask(reg.as_aarch64_gpr());
            } else if (!is_gpr && cc_.is_callee_saved(reg.as_aarch64_fpr())) {
                used_callee_xmms_ |= aarch64::reg_mask(reg.as_aarch64_fpr());
            }
        } else {
            if (is_gpr && cc_.is_callee_saved(reg.as_gpr())) {
                used_callee_gprs_ |= x64::reg_mask(reg.as_gpr());
            } else if (!is_gpr && cc_.is_callee_saved(reg.as_xmm())) {
                used_callee_xmms_ |= x64::reg_mask(reg.as_xmm());
            }
        }
    };

    // 1. If interval has a fixed constraint, check if that fixed register is valid
    for (const auto& pos : interval.use_positions) {
        if (pos.fixed_reg.is_valid() && pos.fixed_reg.reg_class == interval.vreg.reg_class) {
            uint8_t fixed_code = pos.fixed_reg.code;
            if (!in_mask(occupied_regs, fixed_code)) {
                interval.assigned_preg = pos.fixed_reg;
                mark_callee_saved(pos.fixed_reg);

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
                if (!in_mask(occupied_regs, hint_reg.code)) {
                    bool is_callee = check_is_callee(hint_reg);
                    if (!interval.spans_call || is_callee) {
                        interval.assigned_preg = hint_reg;
                        if (is_callee) {
                            mark_callee_saved(hint_reg);
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
            bool is_callee = check_is_callee(reg);
            if (is_callee && !in_mask(occupied_regs, reg.code)) {
                candidates.push_back(reg);
            }
        }
    } else {
        for (const auto& reg : pool) {
            bool is_callee = check_is_callee(reg);
            if (!is_callee && !in_mask(occupied_regs, reg.code)) {
                candidates.push_back(reg);
            }
        }
        for (const auto& reg : pool) {
            bool is_callee = check_is_callee(reg);
            if (is_callee && !in_mask(occupied_regs, reg.code)) {
                candidates.push_back(reg);
            }
        }
    }

    if (!candidates.empty()) {
        PReg chosen = candidates.front();
        interval.assigned_preg = chosen;
        mark_callee_saved(chosen);

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
    bool is_gpr = interval.vreg.is_gpr();
    const auto& pool = is_gpr ? available_gprs_ : available_xmms_;
    const uint32_t hard_blocked = get_hard_blocked_regs(interval);

    auto check_is_callee = [this, is_gpr](PReg reg) {
        if (cc_.target().is_aarch64()) {
            return is_gpr ? cc_.is_callee_saved(reg.as_aarch64_gpr()) : cc_.is_callee_saved(reg.as_aarch64_fpr());
        } else {
            return is_gpr ? cc_.is_callee_saved(reg.as_gpr()) : cc_.is_callee_saved(reg.as_xmm());
        }
    };
    auto mark_callee_saved = [this, is_gpr](PReg reg) {
        if (cc_.target().is_aarch64()) {
            if (is_gpr && cc_.is_callee_saved(reg.as_aarch64_gpr())) {
                used_callee_gprs_ |= aarch64::reg_mask(reg.as_aarch64_gpr());
            } else if (!is_gpr && cc_.is_callee_saved(reg.as_aarch64_fpr())) {
                used_callee_xmms_ |= aarch64::reg_mask(reg.as_aarch64_fpr());
            }
        } else {
            if (is_gpr && cc_.is_callee_saved(reg.as_gpr())) {
                used_callee_gprs_ |= x64::reg_mask(reg.as_gpr());
            } else if (!is_gpr && cc_.is_callee_saved(reg.as_xmm())) {
                used_callee_xmms_ |= x64::reg_mask(reg.as_xmm());
            }
        }
    };

    // The active intervals that conflict with this one, bucketed by the
    // register they hold, in active order — one overlap test per interval
    // rather than one per (register, interval) pair.
    std::vector<LiveInterval*> conflicts_by_reg[32];
    for (auto* act : active_) {
        if (act->vreg.reg_class == interval.vreg.reg_class && act->assigned_preg.is_valid() &&
            act->assigned_preg.code < 32 && act->overlaps(interval)) {
            conflicts_by_reg[act->assigned_preg.code].push_back(act);
        }
    }

    PReg best_reg{};
    std::vector<LiveInterval*> best_conflicts;
    float best_cost = interval.spill_weight;
    uint32_t best_furthest_end = 0;

    for (const auto& reg : pool) {
        if (in_mask(hard_blocked, reg.code)) {
            continue;
        }
        if (interval.spans_call) {
            bool is_callee = check_is_callee(reg);
            if (!is_callee) {
                continue;
            }
        }

        std::vector<LiveInterval*>& conflicts = conflicts_by_reg[reg.code];
        float total_weight = 0.0f;
        uint32_t furthest_end = 0;
        for (const auto* act : conflicts) {
            total_weight += act->spill_weight;
            furthest_end = std::max(furthest_end, act->end_id);
        }

        if (conflicts.empty()) {
            best_reg = reg;
            best_conflicts.clear();
            best_cost = 0.0f;
            break;
        }

        if (total_weight < best_cost || (total_weight == best_cost && furthest_end > best_furthest_end && best_reg.is_valid())) {
            best_cost = total_weight;
            best_reg = reg;
            best_conflicts = std::move(conflicts);
            best_furthest_end = furthest_end;
        }
    }

    if (best_reg.is_valid()) {
        for (auto* c : best_conflicts) {
            c->assigned_preg = PReg{};
            c->assigned_spill_slot = allocate_spill_slot(c->vreg.is_gcref, c->vreg.size);
            auto it = std::find(active_.begin(), active_.end(), c);
            if (it != active_.end()) {
                active_.erase(it);
            }
        }

        interval.assigned_preg = best_reg;
        mark_callee_saved(best_reg);

        active_.push_back(&interval);
        std::sort(active_.begin(), active_.end(),
            [](const LiveInterval* a, const LiveInterval* b) {
                return a->end_id < b->end_id;
            });
    } else {
        // Spill interval
        interval.assigned_spill_slot = allocate_spill_slot(interval.vreg.is_gcref, interval.vreg.size);
        interval.assigned_preg = PReg{};
    }
}

int32_t LinearScanAllocator::allocate_spill_slot(bool is_gcref, uint8_t size) {
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
        int32_t slot = static_cast<int32_t>(next_spill_slot_ - 1);
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
        int32_t slot = static_cast<int32_t>(next_spill_slot_ - 1);
        return slot;
    }
    int32_t slot = static_cast<int32_t>(next_spill_slot_++);
    fn_.frame.spill_slot_is_gcref.push_back(is_gcref);
    fn_.frame.num_spill_slots = next_spill_slot_;
    return slot;
}

void LinearScanAllocator::rewrite_instructions() {
    using namespace brass::x64;

    auto resolve_operand = [this](const LirOperand& op) -> LirOperand {
        if (op.is_vreg()) {
            if (!op.vreg_val.is_valid() || op.vreg_val.id >= fn_.vreg_table.size()) {
                return op;
            }
            const VRegInfo& info = fn_.get_vreg_info(op.vreg_val);
            if (info.is_spilled) {
                return LirOperand::slot(info.assigned_spill_slot, op.size);
            } else if (info.assigned_preg.is_valid()) {
                return LirOperand::preg(info.assigned_preg, op.size);
            }
        } else if (op.is_mem()) {
            LirMem mem = op.mem_val;
            if (mem.base_vreg.is_valid() && mem.base_vreg.id < fn_.vreg_table.size()) {
                const VRegInfo& b_info = fn_.get_vreg_info(mem.base_vreg);
                if (b_info.assigned_preg.is_valid()) {
                    mem.base_preg = b_info.assigned_preg;
                    mem.base_vreg = VReg{};
                }
            }
            if (mem.index_vreg.is_valid() && mem.index_vreg.id < fn_.vreg_table.size()) {
                const VRegInfo& i_info = fn_.get_vreg_info(mem.index_vreg);
                if (i_info.assigned_preg.is_valid()) {
                    mem.index_preg = i_info.assigned_preg;
                    mem.index_vreg = VReg{};
                }
            }
            if (cc_.target().is_x64()) {
                if (mem.index_preg.is_valid() && mem.index_preg == PReg::gpr(x64::GPR::R12) && mem.scale == Scale::One &&
                    mem.base_preg.is_valid() && mem.base_preg != PReg::gpr(x64::GPR::R12)) {
                    std::swap(mem.base_preg, mem.index_preg);
                }
            }
            return LirOperand::mem_custom(mem, op.size);
        }
        return op;
    };

    bool is_aarch64 = cc_.target().is_aarch64();
    PReg base_scratch_reg = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(brass::x64::GPR::R10);
    PReg idx_scratch_reg = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(brass::x64::GPR::R11);

    for (const auto& block : fn_.blocks) {
        std::vector<std::unique_ptr<LirInst>> rewritten;

        for (auto& inst : block->instructions) {
            // First, check if any memory operands have spilled base or index registers
            for (auto* op_list : {&inst->defs, &inst->uses}) {
                for (auto& op : *op_list) {
                    if (op.is_mem()) {
                        if (op.mem_val.base_vreg.is_valid() && op.mem_val.base_vreg.id < fn_.vreg_table.size()) {
                            const VRegInfo& b_info = fn_.get_vreg_info(op.mem_val.base_vreg);
                            if (b_info.is_spilled) {
                                auto load_base = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_base->add_def(LirOperand::preg(base_scratch_reg, 8));
                                load_base->add_use(LirOperand::slot(b_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_base));

                                op.mem_val.base_preg = base_scratch_reg;
                                op.mem_val.base_vreg = VReg{};
                            }
                        }
                        if (op.mem_val.index_vreg.is_valid() && op.mem_val.index_vreg.id < fn_.vreg_table.size()) {
                            const VRegInfo& i_info = fn_.get_vreg_info(op.mem_val.index_vreg);
                            if (i_info.is_spilled) {
                                auto load_idx = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_idx->add_def(LirOperand::preg(idx_scratch_reg, 8));
                                load_idx->add_use(LirOperand::slot(i_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_idx));

                                op.mem_val.index_preg = idx_scratch_reg;
                                op.mem_val.index_vreg = VReg{};
                            }
                        }
                    }
                }
            }

            // Track whether defs and uses were originally XMM operands
            bool orig_def_is_xmm = false;
            if (!inst->defs.empty()) {
                if (inst->defs[0].is_vreg() && inst->defs[0].vreg_val.is_valid() && inst->defs[0].vreg_val.id < fn_.vreg_table.size()) {
                    orig_def_is_xmm = fn_.get_vreg_info(inst->defs[0].vreg_val).vreg.is_xmm();
                } else if (inst->defs[0].is_preg()) {
                    orig_def_is_xmm = inst->defs[0].preg_val.is_xmm();
                }
            }

            std::vector<bool> orig_use_is_xmm(inst->uses.size(), false);
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_vreg() && inst->uses[i].vreg_val.is_valid() && inst->uses[i].vreg_val.id < fn_.vreg_table.size()) {
                    orig_use_is_xmm[i] = fn_.get_vreg_info(inst->uses[i].vreg_val).vreg.is_xmm();
                } else if (inst->uses[i].is_preg()) {
                    orig_use_is_xmm[i] = inst->uses[i].preg_val.is_xmm();
                }
            }

            // Rewrite defs and uses
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                inst->defs[i] = resolve_operand(inst->defs[i]);
            }
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                inst->uses[i] = resolve_operand(inst->uses[i]);
            }

            if (inst->opcode == LirOpcode::Safepoint || inst->opcode == LirOpcode::ParallelCopy) {
                rewritten.push_back(std::move(inst));
                continue;
            }

            bool def_is_mem = !inst->defs.empty() && (inst->defs[0].is_mem() || inst->defs[0].is_spill_slot());
            bool use_is_mem = !inst->uses.empty() && (inst->uses[0].is_mem() || inst->uses[0].is_spill_slot());

            // Moves between memory / spill slots:
            if ((inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                 inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                 inst->opcode == LirOpcode::Movaps) &&
                def_is_mem && use_is_mem) {
                uint8_t sz = inst->uses[0].size;
                bool is_xmm = orig_def_is_xmm || orig_use_is_xmm[0] || (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                               inst->opcode == LirOpcode::Movaps || inst->opcode == LirOpcode::Vmovaps ||
                               inst->opcode == LirOpcode::Vmovups || sz == 16 || sz == 32);
                PReg scratch = is_xmm ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V29) : PReg::xmm(XMM::XMM15))
                                      : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(GPR::R11));
                if (!is_xmm) {
                    PReg gpr_scratch1 = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(GPR::R11);
                    PReg gpr_scratch0 = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(GPR::R10);
                    bool uses_scratch1 = false;
                    if (inst->defs[0].is_mem() && (inst->defs[0].mem_val.base_preg == gpr_scratch1 ||
                                                   inst->defs[0].mem_val.index_preg == gpr_scratch1)) {
                        uses_scratch1 = true;
                    }
                    if (inst->uses[0].is_mem() && (inst->uses[0].mem_val.base_preg == gpr_scratch1 ||
                                                   inst->uses[0].mem_val.index_preg == gpr_scratch1)) {
                        uses_scratch1 = true;
                    }
                    if (uses_scratch1) {
                        scratch = gpr_scratch0;
                    }
                }

                LirOpcode op = is_xmm ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                      : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                auto load_scratch = std::make_unique<LirInst>(op);
                load_scratch->add_def(LirOperand::preg(scratch, sz));
                load_scratch->add_use(inst->uses[0]);
                rewritten.push_back(std::move(load_scratch));

                auto store_scratch = std::make_unique<LirInst>(op);
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
                uint8_t sz = original_spill_def.size;
                is_xmm_def = orig_def_is_xmm || (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                              inst->opcode == LirOpcode::Addsd || inst->opcode == LirOpcode::Subsd ||
                              inst->opcode == LirOpcode::Mulsd || inst->opcode == LirOpcode::Divsd ||
                              inst->opcode == LirOpcode::Sqrtsd || inst->opcode == LirOpcode::Xorpd ||
                              inst->opcode == LirOpcode::Addss || inst->opcode == LirOpcode::Subss ||
                              inst->opcode == LirOpcode::Mulss || inst->opcode == LirOpcode::Divss ||
                              inst->opcode == LirOpcode::Sqrtss || inst->opcode == LirOpcode::Xorps ||
                              inst->opcode == LirOpcode::Cvtsi2sd || inst->opcode == LirOpcode::Cvtsi2sd32 ||
                              inst->opcode == LirOpcode::Movq_xg || inst->opcode == LirOpcode::Movaps ||
                              inst->opcode == LirOpcode::Movups || inst->opcode == LirOpcode::Movd_xg ||
                              inst->opcode == LirOpcode::Addps || inst->opcode == LirOpcode::Subps ||
                              inst->opcode == LirOpcode::Mulps || inst->opcode == LirOpcode::Divps ||
                              inst->opcode == LirOpcode::Minps || inst->opcode == LirOpcode::Maxps ||
                              inst->opcode == LirOpcode::Sqrtps || inst->opcode == LirOpcode::Addpd ||
                              inst->opcode == LirOpcode::Subpd || inst->opcode == LirOpcode::Mulpd ||
                              inst->opcode == LirOpcode::Divpd || inst->opcode == LirOpcode::Minpd ||
                              inst->opcode == LirOpcode::Maxpd || inst->opcode == LirOpcode::Sqrtpd ||
                              inst->opcode == LirOpcode::Paddd || inst->opcode == LirOpcode::Psubd ||
                              inst->opcode == LirOpcode::Pmulld || inst->opcode == LirOpcode::Pminsd ||
                              inst->opcode == LirOpcode::Pmaxsd || inst->opcode == LirOpcode::Paddq ||
                              inst->opcode == LirOpcode::Psubq || inst->opcode == LirOpcode::Pand ||
                              inst->opcode == LirOpcode::Por || inst->opcode == LirOpcode::Pxor ||
                              inst->opcode == LirOpcode::Pandn || inst->opcode == LirOpcode::Pcmpeqd ||
                              inst->opcode == LirOpcode::Pslld || inst->opcode == LirOpcode::Psllq ||
                              inst->opcode == LirOpcode::Shufps || inst->opcode == LirOpcode::Shufpd ||
                              inst->opcode == LirOpcode::Pshufd || inst->opcode == LirOpcode::Movddup ||
                              inst->opcode == LirOpcode::Pinsrd || inst->opcode == LirOpcode::Pinsrq ||
                              inst->opcode == LirOpcode::Insertps ||
                              inst->opcode == LirOpcode::Vfmadd213ss || inst->opcode == LirOpcode::Vfmadd231ss ||
                              inst->opcode == LirOpcode::Vfmadd213sd || inst->opcode == LirOpcode::Vfmadd231sd ||
                              inst->opcode == LirOpcode::Vfmadd213ps || inst->opcode == LirOpcode::Vfmadd231ps ||
                              inst->opcode == LirOpcode::Vfmadd213pd || inst->opcode == LirOpcode::Vfmadd231pd ||
                              inst->opcode == LirOpcode::Vbroadcastss || inst->opcode == LirOpcode::Vbroadcastsd ||
                              sz == 16 || sz == 32);
                if (inst->is_call()) {
                    def_scratch = is_xmm_def ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V0) : PReg::xmm(XMM::XMM0))
                                             : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X0) : PReg::gpr(GPR::RAX));
                } else {
                    def_scratch = is_xmm_def ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V27) : PReg::xmm(XMM::XMM15))
                                             : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X14) : PReg::gpr(GPR::R11));
                }

                // If instruction reads from def (e.g. add dst, src), load initial value of def into scratch
                bool reads_def = false;
                for (size_t i = 0; i < inst->uses.size(); ++i) {
                    if (inst->uses[i].is_spill_slot() && inst->uses[i].spill_slot == original_spill_def.spill_slot) {
                        reads_def = true;
                        inst->uses[i] = LirOperand::preg(def_scratch, sz);
                    }
                }

                if (reads_def) {
                    LirOpcode load_op = is_xmm_def ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                   : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                    auto load_def = std::make_unique<LirInst>(load_op);
                    load_def->add_def(LirOperand::preg(def_scratch, sz));
                    load_def->add_use(original_spill_def);
                    rewritten.push_back(std::move(load_def));
                }

                inst->defs[0] = LirOperand::preg(def_scratch, sz);
            }

            // Handle any remaining spill uses with reserved scratch registers R10/R11 (GPR) or XMM13/XMM14 (XMM)
            std::vector<int32_t> orig_slot_indices(inst->uses.size(), -1);
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    orig_slot_indices[i] = inst->uses[i].spill_slot;
                }
            }

            int gpr_scratch_idx = 0;
            int xmm_scratch_idx = 0;
            PReg gpr_scratches[2] = {
                is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(GPR::R10),
                is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(GPR::R11)
            };
            PReg xmm_scratches[2] = {
                is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V28) : PReg::xmm(XMM::XMM13),
                is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V29) : PReg::xmm(XMM::XMM14)
            };

            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    uint8_t sz = inst->uses[i].size;
                    bool is_xmm_use = (i < orig_use_is_xmm.size()) ? orig_use_is_xmm[i] : false;
                    if (!is_xmm_use) {
                        if (inst->opcode == LirOpcode::Pinsrd || inst->opcode == LirOpcode::Pinsrq) {
                            is_xmm_use = (i == 0);
                        } else if (inst->opcode == LirOpcode::Movd_xg || inst->opcode == LirOpcode::Movq_xg) {
                            is_xmm_use = false;
                        } else if (inst->opcode == LirOpcode::Cvttsd2si || inst->opcode == LirOpcode::Cvttsd2si32 ||
                                   inst->opcode == LirOpcode::Movq_gx || inst->opcode == LirOpcode::Movd_gx ||
                                   inst->opcode == LirOpcode::Pextrd || inst->opcode == LirOpcode::Pextrq ||
                                   inst->opcode == LirOpcode::Extractps) {
                            is_xmm_use = true;
                        } else {
                            is_xmm_use = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                                          inst->opcode == LirOpcode::Addsd || inst->opcode == LirOpcode::Subsd ||
                                          inst->opcode == LirOpcode::Mulsd || inst->opcode == LirOpcode::Divsd ||
                                          inst->opcode == LirOpcode::Sqrtsd || inst->opcode == LirOpcode::Ucomisd ||
                                          inst->opcode == LirOpcode::Addss || inst->opcode == LirOpcode::Subss ||
                                          inst->opcode == LirOpcode::Mulss || inst->opcode == LirOpcode::Divss ||
                                          inst->opcode == LirOpcode::Sqrtss || inst->opcode == LirOpcode::Ucomiss ||
                                          inst->opcode == LirOpcode::Xorpd || inst->opcode == LirOpcode::Movaps ||
                                          inst->opcode == LirOpcode::Movups || inst->opcode == LirOpcode::Addps ||
                                          inst->opcode == LirOpcode::Subps || inst->opcode == LirOpcode::Mulps ||
                                          inst->opcode == LirOpcode::Divps || inst->opcode == LirOpcode::Minps ||
                                          inst->opcode == LirOpcode::Maxps || inst->opcode == LirOpcode::Sqrtps ||
                                          inst->opcode == LirOpcode::Addpd || inst->opcode == LirOpcode::Subpd ||
                                          inst->opcode == LirOpcode::Mulpd || inst->opcode == LirOpcode::Divpd ||
                                          inst->opcode == LirOpcode::Minpd || inst->opcode == LirOpcode::Maxpd ||
                                          inst->opcode == LirOpcode::Sqrtpd || inst->opcode == LirOpcode::Paddd ||
                                          inst->opcode == LirOpcode::Psubd || inst->opcode == LirOpcode::Pmulld ||
                                          inst->opcode == LirOpcode::Pminsd || inst->opcode == LirOpcode::Pmaxsd ||
                                          inst->opcode == LirOpcode::Paddq || inst->opcode == LirOpcode::Psubq ||
                                          inst->opcode == LirOpcode::Pand || inst->opcode == LirOpcode::Por ||
                                          inst->opcode == LirOpcode::Pxor || inst->opcode == LirOpcode::Pandn ||
                                          inst->opcode == LirOpcode::Pcmpeqd || inst->opcode == LirOpcode::Pslld ||
                                          inst->opcode == LirOpcode::Psllq || inst->opcode == LirOpcode::Shufps ||
                                          inst->opcode == LirOpcode::Shufpd || inst->opcode == LirOpcode::Pshufd ||
                                          inst->opcode == LirOpcode::Movddup || inst->opcode == LirOpcode::Insertps ||
                                          inst->opcode == LirOpcode::Xorps ||
                                          inst->opcode == LirOpcode::Vfmadd213ss || inst->opcode == LirOpcode::Vfmadd231ss ||
                                          inst->opcode == LirOpcode::Vfmadd213sd || inst->opcode == LirOpcode::Vfmadd231sd ||
                                          inst->opcode == LirOpcode::Vfmadd213ps || inst->opcode == LirOpcode::Vfmadd231ps ||
                                          inst->opcode == LirOpcode::Vfmadd213pd || inst->opcode == LirOpcode::Vfmadd231pd ||
                                          inst->opcode == LirOpcode::Vbroadcastss || inst->opcode == LirOpcode::Vbroadcastsd ||
                                          sz == 16 || sz == 32);
                        }
                    }
                    int32_t slot = inst->uses[i].spill_slot;

                    // Check if this same slot was already loaded for a previous use
                    PReg use_scratch;
                    bool already_loaded = false;
                    for (size_t j = 0; j < i; ++j) {
                        if (orig_slot_indices[j] == slot && inst->uses[j].is_preg()) {
                            use_scratch = inst->uses[j].preg_val;
                            already_loaded = true;
                            break;
                        }
                    }

                    if (!already_loaded) {
                        if (is_xmm_use) {
                            use_scratch = xmm_scratches[xmm_scratch_idx < 2 ? xmm_scratch_idx++ : 1];
                        } else {
                            use_scratch = gpr_scratches[gpr_scratch_idx < 2 ? gpr_scratch_idx++ : 1];
                        }

                        LirOpcode load_op = is_xmm_use ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                       : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                        auto load_use = std::make_unique<LirInst>(load_op);
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
                LirOpcode store_op = is_xmm_def ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                auto store_back = std::make_unique<LirInst>(store_op);
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
