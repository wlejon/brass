#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/osr.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

static std::pair<Condition, Condition> get_comparison_conditions(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq:  return {Condition::E, Condition::E};
        case Opcode::ne:  return {Condition::NE, Condition::NE};
        case Opcode::slt: return {Condition::L, Condition::B};
        case Opcode::ult: return {Condition::B, Condition::B};
        case Opcode::sle: return {Condition::LE, Condition::BE};
        case Opcode::ule: return {Condition::BE, Condition::BE};
        case Opcode::sgt: return {Condition::G, Condition::A};
        case Opcode::ugt: return {Condition::A, Condition::A};
        case Opcode::sge: return {Condition::GE, Condition::AE};
        case Opcode::uge: return {Condition::AE, Condition::AE};
        default: return {Condition::None, Condition::None};
    }
}

static constexpr Condition swap_relational_condition(Condition cond) noexcept {
    switch (cond) {
        case Condition::E:   return Condition::E;
        case Condition::NE:  return Condition::NE;
        case Condition::L:   return Condition::G;
        case Condition::LE:  return Condition::GE;
        case Condition::G:   return Condition::L;
        case Condition::GE:  return Condition::LE;
        case Condition::B:   return Condition::A;
        case Condition::BE:  return Condition::AE;
        case Condition::A:   return Condition::B;
        case Condition::AE:  return Condition::BE;
        default: return cond;
    }
}

X64ISel::ImmIntInfo X64ISel::get_imm_int_info(const Value* val) const {
    if (!val) return {};
    if (val->is_instruction()) {
        const Instruction* def = val->defining_instruction();
        if (def) {
            if (def->opcode() == Opcode::iconst_i32) {
                return {true, static_cast<int64_t>(def->imm_i32()), true, def};
            }
            if (def->opcode() == Opcode::iconst_i64) {
                int64_t v = def->imm_i64();
                bool fits = (v >= INT32_MIN && v <= INT32_MAX);
                return {true, v, fits, def};
            }
        }
    }
    return {};
}



bool X64ISel::is_value_dead_after(
    const Function& mir_fn,
    const BasicBlock& bb,
    const Instruction* inst,
    const Value* val
) const {
    (void)mir_fn;
    if (!val) return false;
    auto it = use_count_.find(val);
    if (it == use_count_.end() || it->second == 0) return true;
    uint32_t total_uses = it->second;

    uint32_t uses_seen = 0;
    for (const auto* cur : bb) {
        for (const auto* op : cur->operands()) {
            if (op == val) uses_seen++;
        }
        if (cur == inst) break;
    }

    return (uses_seen == total_uses);
}

void X64ISel::analyze_function(const Function& mir_fn) {
    use_count_.clear();
    skipped_insts_.clear();

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            for (const auto* op : inst->operands()) {
                if (op) use_count_[op]++;
            }
            for (const auto* arg : inst->branch_target().args) {
                if (arg) use_count_[arg]++;
            }
            for (const auto* arg : inst->true_target().args) {
                if (arg) use_count_[arg]++;
            }
            for (const auto* arg : inst->false_target().args) {
                if (arg) use_count_[arg]++;
            }
            // Switch case arguments are uses too: without them a constant
            // also folded into an ALU immediate looks fully folded, gets no
            // register, and the case edge copies an unassigned value.
            for (const auto& sc : inst->switch_cases()) {
                for (const auto* arg : sc.target.args) {
                    if (arg) use_count_[arg]++;
                }
            }
            for (const auto* sv : inst->state_map()) {
                if (sv) use_count_[sv]++;
            }
        }
    }

    // 2. Identify fused comparisons in br_if and guard, and LEA fusions in add
    // (add, operand) pairs where the add is lowered as one lea that
    // computes the operand too, so the operand itself is skipped.
    std::vector<std::pair<const Instruction*, const Instruction*>> lea_fusions;
    // The register parts (base, index) of every folded memory address
    // (step 3): each must end up with a register.
    std::vector<const Value*> address_regs;

    auto skip_operand_if_dead = [&](const Value* val) {
        if (val && val->is_instruction() && use_count_[val] == 1) {
            skipped_insts_.insert(val->defining_instruction());
        }
    };

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::br_if || inst->opcode() == Opcode::guard || inst->opcode() == Opcode::select) {
                const Value* cond = inst->operand(0);
                if (cond && cond->is_instruction()) {
                    const Instruction* def_inst = cond->defining_instruction();
                    if (def_inst && def_inst->parent() == bb && comparison_fusible(*def_inst)) {
                        if (use_count_[cond] == 1) {
                            skipped_insts_.insert(def_inst);
                            if (def_inst->opcode() == Opcode::eq || def_inst->opcode() == Opcode::ne) {
                                ImmIntInfo c0 = get_imm_int_info(def_inst->operand(0));
                                ImmIntInfo c1 = get_imm_int_info(def_inst->operand(1));
                                const Value* and_val = (c1.is_imm && c1.val == 0) ? def_inst->operand(0) : ((c0.is_imm && c0.val == 0) ? def_inst->operand(1) : nullptr);
                                if (and_val && and_val->is_instruction()) {
                                    const Instruction* and_inst = and_val->defining_instruction();
                                    if (and_inst && and_inst->parent() == bb && and_inst->opcode() == Opcode::and_ && use_count_[and_val] == 1) {
                                        skipped_insts_.insert(and_inst);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (inst->opcode() == Opcode::add) {
                const Value* op0 = inst->operand(0);
                const Value* op1 = inst->operand(1);
                ImmIntInfo imm0 = get_imm_int_info(op0);
                ImmIntInfo imm1 = get_imm_int_info(op1);
                const Value* reg_val = (imm1.is_imm && imm1.fits_i32) ? op0 : ((imm0.is_imm && imm0.fits_i32) ? op1 : nullptr);
                if (reg_val && reg_val->is_instruction()) {
                    const Instruction* def = reg_val->defining_instruction();
                    if (def && def->parent() == bb && use_count_[reg_val] == 1) {
                        if (def->opcode() == Opcode::mul) {
                            ImmIntInfo m0 = get_imm_int_info(def->operand(0));
                            ImmIntInfo m1 = get_imm_int_info(def->operand(1));
                            int64_t mult = m1.is_imm ? m1.val : (m0.is_imm ? m0.val : 0);
                            if (mult == 2 || mult == 3 || mult == 4 || mult == 5 || mult == 8 || mult == 9) {
                                skipped_insts_.insert(def);
                                lea_fusions.emplace_back(inst, def);
                                if (m0.is_imm) skip_operand_if_dead(def->operand(0));
                                if (m1.is_imm) skip_operand_if_dead(def->operand(1));
                            }
                        } else if (def->opcode() == Opcode::shl) {
                            ImmIntInfo s1 = get_imm_int_info(def->operand(1));
                            if (s1.is_imm && (s1.val == 1 || s1.val == 2 || s1.val == 3)) {
                                skipped_insts_.insert(def);
                                lea_fusions.emplace_back(inst, def);
                                skip_operand_if_dead(def->operand(1));
                            }
                        } else if (def->opcode() == Opcode::add) {
                            ImmIntInfo a0 = get_imm_int_info(def->operand(0));
                            ImmIntInfo a1 = get_imm_int_info(def->operand(1));
                            if (!a0.is_imm && !a1.is_imm) {
                                skipped_insts_.insert(def);
                                lea_fusions.emplace_back(inst, def);
                            }
                        }
                    }
                }
            }
        }
    }

    std::unordered_map<const Value*, uint32_t> folded_uses;

    // A memory operation that folds its address computation reads only the
    // address value itself without a register: that is the one use it
    // folds. The instructions and constants further down the chain are used
    // by the address instruction, not by the memory operation, so counting
    // them once per memory operation would overcount their folded uses and
    // leave a value that still has a real use without a register.
    auto count_address_fold = [&](const MemFold& mf, std::initializer_list<const Value*> operands) {
        address_regs.push_back(mf.base_val);
        address_regs.push_back(mf.index_val);
        for (const Value* op : operands) {
            for (const auto* fi : mf.folded_instructions) {
                if (fi && op && fi->result() == op) {
                    folded_uses[op]++;
                    break;
                }
            }
        }
    };

    // 3. Count folded uses in load/store/alu/branch/guard
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (skipped_insts_.count(inst)) continue;

            if (inst->opcode() == Opcode::load) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                count_address_fold(mf, {inst->operand(0)});
                continue;
            }

            if (inst->opcode() == Opcode::store) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                count_address_fold(mf, {inst->operand(0)});
                const Value* src = inst->operand(1);
                ImmIntInfo imm_src = get_imm_int_info(src);
                if (imm_src.is_imm && imm_src.fits_i32) {
                    folded_uses[src]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::load_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), scale_from_int(inst->scale()), inst->offset());
                count_address_fold(mf, {inst->operand(0), inst->operand(1)});
                continue;
            }

            if (inst->opcode() == Opcode::store_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), scale_from_int(inst->scale()), inst->offset());
                count_address_fold(mf, {inst->operand(0), inst->operand(1)});
                const Value* src = inst->operand(2);
                ImmIntInfo imm_src = get_imm_int_info(src);
                if (imm_src.is_imm && imm_src.fits_i32) {
                    folded_uses[src]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::br_if) {
                const Value* cond = inst->operand(0);
                if (cond && cond->is_instruction()) {
                    const Instruction* def_inst = cond->defining_instruction();
                    if (def_inst && def_inst->parent() == bb && is_comparison(def_inst->opcode()) && skipped_insts_.count(def_inst)) {
                        const Value* lhs = def_inst->operand(0);
                        const Value* rhs = def_inst->operand(1);
                        ImmIntInfo rhs_imm = get_imm_int_info(rhs);
                        ImmIntInfo lhs_imm = get_imm_int_info(lhs);

                        if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                            folded_uses[rhs]++;
                        } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                            folded_uses[lhs]++;
                        } else if (!lhs->type().is_float() && rhs && rhs->is_instruction() &&
                                   can_fuse_load(rhs->defining_instruction(), inst)) {
                            skipped_insts_.insert(rhs->defining_instruction());
                        }
                        continue;
                    }
                }
            }

            // Division folds only a power-of-two divisor (lower_div_mod) and
            // never a load; a shift folds any constant count but needs a
            // register count in CL otherwise.
            if (inst->opcode() == Opcode::udiv || inst->opcode() == Opcode::umod ||
                inst->opcode() == Opcode::sdiv || inst->opcode() == Opcode::smod) {
                const bool is_mod = inst->opcode() == Opcode::umod || inst->opcode() == Opcode::smod;
                if (!inst->type().is_float() && divisor_folds(get_imm_int_info(inst->operand(1)), is_mod)) {
                    folded_uses[inst->operand(1)]++;
                }
                continue;
            }
            if (inst->opcode() == Opcode::shl || inst->opcode() == Opcode::lshr || inst->opcode() == Opcode::ashr) {
                if (get_imm_int_info(inst->operand(1)).is_imm) folded_uses[inst->operand(1)]++;
                continue;
            }
            // Integer multiplication by a constant the lowering strength-
            // reduces or encodes as an immediate reads the other operand
            // from a register, so that operand's load cannot be fused. The
            // choice of constant mirrors lower_binary_alu.
            if (inst->opcode() == Opcode::mul && !inst->type().is_float() && !inst->type().is_vector()) {
                ImmIntInfo imm0 = get_imm_int_info(inst->operand(0));
                ImmIntInfo imm1 = get_imm_int_info(inst->operand(1));
                const Value* imm_val = imm1.is_imm ? inst->operand(1) : (imm0.is_imm ? inst->operand(0) : nullptr);
                const ImmIntInfo& imm = imm1.is_imm ? imm1 : imm0;
                if (imm_val && mul_imm_folds(imm)) {
                    folded_uses[imm_val]++;
                } else if (can_fuse_load(inst->operand(1)->defining_instruction(), inst)) {
                    skipped_insts_.insert(inst->operand(1)->defining_instruction());
                } else if (can_fuse_load(inst->operand(0)->defining_instruction(), inst)) {
                    skipped_insts_.insert(inst->operand(0)->defining_instruction());
                }
                continue;
            }

            bool is_alu = false;
            switch (inst->opcode()) {
                case Opcode::add: case Opcode::sub: case Opcode::mul:
                case Opcode::and_: case Opcode::or_: case Opcode::xor_:
                case Opcode::eq: case Opcode::ne:
                case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
                case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
                    is_alu = true;
                    break;
                default:
                    break;
            }

            if (is_alu && inst->operand_count() >= 2) {
                const Value* op0 = inst->operand(0);
                const Value* op1 = inst->operand(1);
                ImmIntInfo imm0 = get_imm_int_info(op0);
                ImmIntInfo imm1 = get_imm_int_info(op1);

                // Float add/mul keep src0 first (the lhs NaN wins), as
                // lower_binary_alu does: only their rhs load may be fused.
                const bool float_arith = inst->type().is_float() &&
                                         (inst->opcode() == Opcode::add || inst->opcode() == Opcode::mul);
                bool is_comm = !float_arith &&
                               (inst->opcode() == Opcode::add || inst->opcode() == Opcode::mul ||
                                inst->opcode() == Opcode::and_ || inst->opcode() == Opcode::or_ ||
                                inst->opcode() == Opcode::xor_);

                // Subtraction and integer comparison also take a constant
                // first operand as an immediate (mov + sub, swapped cmp) and
                // then read the second from a register, never from memory.
                const bool imm0_foldable = is_comm || inst->opcode() == Opcode::sub || is_comparison(inst->opcode());

                if (imm1.is_imm && imm1.fits_i32) {
                    folded_uses[op1]++;
                } else if (imm0_foldable && imm0.is_imm && imm0.fits_i32) {
                    folded_uses[op0]++;
                } else if (op1 && op1->is_instruction() && can_fuse_load(op1->defining_instruction(), inst)) {
                    skipped_insts_.insert(op1->defining_instruction());
                } else if (is_comm && op0 && op0->is_instruction() && can_fuse_load(op0->defining_instruction(), inst)) {
                    skipped_insts_.insert(op0->defining_instruction());
                }
            }
        }
    }

    // 4. Skip instructions whose uses have all been folded
    for (const auto& [val, fold_count] : folded_uses) {
        auto it = use_count_.find(val);
        if (it != use_count_.end() && it->second == fold_count) {
            if (val->is_instruction()) {
                skipped_insts_.insert(val->defining_instruction());
            }
        }
    }

    // 5. An add that is not lowered itself (folded into a memory operand's
    // address, whose register part is then the add's operand) computes no
    // lea, so the operand it would have absorbed must be computed after all.
    for (const auto& [user, fused] : lea_fusions) {
        if (skipped_insts_.count(user)) skipped_insts_.erase(fused);
    }
    // 6. A folded address reads its base and index from registers, even when
    // they are operands of an add the address absorbed. Such an operand may
    // be a single-use load that step 3 fused into that add as its memory
    // operand (`load [(load tls.deltas) + (load slot)*8]`): whether or not
    // the add is still emitted, the address reads the value itself, so the
    // load has to produce a register after all. At worst the add reads the
    // same memory a second time.
    for (const Value* v : address_regs) {
        if (!v || !v->is_instruction()) continue;
        const Instruction* def = v->defining_instruction();
        if (def && (def->opcode() == Opcode::load || def->opcode() == Opcode::load_indexed)) {
            skipped_insts_.erase(def);
        }
    }
}

X64ISel::X64ISel()
    : target_(Target::host()), cc_(CallingConvention::for_target(Target::host())) {}

X64ISel::X64ISel(const Target& target)
    : target_(target), cc_(CallingConvention::for_target(target)) {}

X64ISel::X64ISel(const Target& target, const CallingConvention& cc)
    : target_(target), cc_(cc) {}

std::unique_ptr<LirFunction> X64ISel::lower(const Function& mir_fn) {
    auto lir = std::make_unique<LirFunction>();
    lir_fn_ = lir.get();
    mir_fn_ = &mir_fn;
    val_to_vreg_.clear();

    lir_fn_->name = std::string(mir_fn.name());
    lir_fn_->return_type = mir_fn.return_type();
    lir_fn_->calling_conv = cc_;
    if (mir_fn.parent() && mir_fn.parent()->pinned_tls_register()) {
        lir_fn_->reserved_gprs |= reg_mask(kPinnedTlsGpr);
    }

    const_cast<Function&>(mir_fn).rebuild_cfg_predecessors();

    // 0. Pre-analyze function to identify fusible comparisons, loads, and immediate folds
    analyze_function(mir_fn);

    // 1. Create all LIR blocks matching MIR blocks
    for (const auto* bb : mir_fn.blocks()) {
        lir_fn_->create_block_with_id(bb->id(), std::string(bb->name()));
    }

    // 2. Allocate VRegs for all block parameters and instructions
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* param : bb->params()) {
            get_or_alloc_vreg(param);
        }
    }

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (!inst->produces_value()) continue;
            if (skipped_insts_.count(inst)) continue;
            get_or_alloc_vreg(inst->result());
        }
    }

    // 3. Lower entry block parameters from calling convention registers / stack slots
    lower_entry_parameters(mir_fn);

    // 4. Lower all instructions block by block
    for (const auto* bb : mir_fn.blocks()) {
        lower_block(*bb);
    }

    // 5. Connect the CFG. A MIR edge whose block arguments were lowered into
    // an edge trampoline (linked while lowering) runs through that
    // trampoline; a direct edge too would make the target's parameters look
    // live, undefined, on every path into this block, and a gcref parameter
    // would then be reported to the GC with whatever its home holds.
    std::unordered_set<uint32_t> mir_block_ids;
    for (const auto* bb : mir_fn.blocks()) mir_block_ids.insert(bb->id());
    for (size_t i = 0; i < mir_fn.blocks().size(); ++i) {
        const auto* mir_bb = mir_fn.blocks()[i];
        auto* lir_bb = lir_fn_->blocks[i].get();

        auto jumps_directly_to = [&](uint32_t id) {
            for (const auto& li : lir_bb->instructions) {
                if (li->opcode != LirOpcode::Jmp && li->opcode != LirOpcode::Jcc) continue;
                for (const auto& u : li->uses) {
                    if (u.is_label() && u.label_id == id) return true;
                }
            }
            return false;
        };
        auto reached_by_trampoline = [&](const LirBlock* target) {
            for (const LirBlock* s : lir_bb->successors) {
                if (mir_block_ids.count(s->id)) continue;
                if (std::find(s->successors.begin(), s->successors.end(), target) != s->successors.end()) return true;
            }
            return false;
        };

        for (const auto* succ : mir_bb->successors()) {
            LirBlock* succ_lir = lir_fn_->get_block_by_id(succ->id());
            if (!succ_lir) continue;
            if (reached_by_trampoline(succ_lir) && !jumps_directly_to(succ_lir->id)) continue;
            link_blocks(*lir_bb, *succ_lir);
        }
    }

    // 6. Record resume table entries and connect CFG edges for resume targets
    auto* lir_entry = lir_fn_->entry_block();
    for (const auto& rp : mir_fn.resume_points()) {
        if (rp.second) {
            lir_fn_->resume_entries.push_back({rp.first, rp.second->id()});
            auto* target_lir = lir_fn_->get_block_by_id(rp.second->id());
            if (lir_entry && target_lir) {
                lir_entry->successors.push_back(target_lir);
                target_lir->predecessors.push_back(lir_entry);
            }
        }
    }

    // 7. Setup OSR entry if requested or present in MIR
    if (osr_target_ && osr_target_->is_valid()) {
        lir_fn_->osr_entry.enabled = true;
        lir_fn_->osr_entry.loop_header_id = osr_target_->loop_header_id;
        lir_fn_->osr_entry.live_in_vregs.clear();
        lir_fn_->osr_entry.slot_indices.clear();
        for (const Value* v : osr_target_->live_ins) {
            if (v) {
                VReg vr = get_or_alloc_vreg(v);
                lir_fn_->osr_entry.live_in_vregs.push_back(vr);
                lir_fn_->osr_entry.slot_indices.push_back(v->id());
            }
        }
    } else {
        for (const auto* bb : mir_fn.blocks()) {
            for (const auto* inst : *bb) {
                if (inst->opcode() == Opcode::osr_entry) {
                    lir_fn_->osr_entry.enabled = true;
                    lir_fn_->osr_entry.loop_header_id = static_cast<uint32_t>(inst->imm_i64());
                    for (const auto* op : inst->operands()) {
                        if (op) {
                            lir_fn_->osr_entry.live_in_vregs.push_back(get_or_alloc_vreg(op));
                            lir_fn_->osr_entry.slot_indices.push_back(op->id());
                        }
                    }
                    break;
                }
            }
            if (lir_fn_->osr_entry.enabled) break;
        }
    }

    return lir;
}

VReg X64ISel::get_or_alloc_vreg(const Value* val) {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }

    Type t = val->type();
    RegClass rc = (t.is_float() || t.is_vector()) ? RegClass::XMM : RegClass::GPR;
    uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
    if (sz == 0) sz = 8;
    bool is_gc = t.is_gcref();

    VReg v = lir_fn_->allocate_vreg(rc, sz, is_gc);
    val_to_vreg_[val] = v;
    return v;
}

VReg X64ISel::get_vreg(const Value* val) const {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }
    // A value whose every use the analysis folded has no register. Lowering
    // looks some operands up speculatively, so this is not an error here;
    // an invalid register that reaches an emitted instruction is rejected
    // by the register allocator's rewrite.
    return VReg{};
}

void X64ISel::lower_entry_parameters(const Function& mir_fn) {
    const auto* entry = mir_fn.entry_block();
    if (!entry || entry->params().empty()) return;

    auto* lir_entry = lir_fn_->entry_block();
    if (!lir_entry) return;

    auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < entry->param_count(); ++i) {
        const auto* param = entry->param(i);
        VReg param_vreg = get_vreg(param);
        Type t = param->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                if (t.is_float() || t.is_vector()) {
                    XMM xreg = static_cast<XMM>(i);
                    pcopy->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    pcopy->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                }
            } else {
                if (t.is_vector()) throw_unsupported("x64 isel (entry)", "vector parameter passed on the stack");
                int32_t disp = static_cast<int32_t>(48 + (i - 4) * 8);
                lir_fn_->frame.has_stack_args = true;
                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                pcopy->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
            }
        } else {
            if (t.is_float() || t.is_vector()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                    pcopy->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                } else {
                    if (t.is_vector()) throw_unsupported("x64 isel (entry)", "vector parameter passed on the stack");
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(16 + stack_idx * 8);
                    lir_fn_->frame.has_stack_args = true;
                    xmm_idx++;
                    pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                    pcopy->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                }
            } else {
                if (gpr_idx < cc_.num_arg_gprs()) {
                    GPR greg = cc_.arg_gpr(gpr_idx++);
                    pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                    pcopy->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                } else {
                    size_t stack_idx = (gpr_idx - cc_.num_arg_gprs()) + (xmm_idx > cc_.num_arg_xmms() ? (xmm_idx - cc_.num_arg_xmms()) : 0);
                    int32_t disp = static_cast<int32_t>(16 + stack_idx * 8);
                    lir_fn_->frame.has_stack_args = true;
                    gpr_idx++;
                    pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                    pcopy->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                }
            }
        }
    }
    lir_entry->append_inst(std::move(pcopy));
    // No resume dispatch on entry: a resume block is reached only through
    // the resume table (docs/mir_reference.md, "Guard exits"), never by
    // matching an ordinary argument against a resume id.
}

void X64ISel::lower_block(const BasicBlock& bb) {
    auto* lir_bb = lir_fn_->get_block_by_id(bb.id());
    if (!lir_bb) return;

    for (const auto* inst : bb) {
        if (!skipped_insts_.count(inst)) {
            size_t before_count = lir_bb->instructions.size();
            if (!lower_narrow_width_op(*inst, *lir_bb)) {
                const auto widened = widen_narrow_operands(*inst, *lir_bb);
                lower_instruction(*inst, *lir_bb);
                restore_narrow_operands(widened);
            }
            for (size_t i = before_count; i < lir_bb->instructions.size(); ++i) {
                if (lir_bb->instructions[i]) {
                    if (!lir_bb->instructions[i]->loc.is_valid() && inst->loc().is_valid()) {
                        lir_bb->instructions[i]->loc = inst->loc();
                    }
                    if (!lir_bb->instructions[i]->mir_origin) {
                        lir_bb->instructions[i]->mir_origin = inst;
                    }
                }
            }
        }
    }
}

void X64ISel::lower_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& target = inst.branch_target();
    if (!target.block) return;

    if (!target.args.empty()) {
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, sz));
                mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                lir_bb.append_inst(std::move(mov_inst));
            }
        } else {
            auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
            for (size_t i = 0; i < target.args.size(); ++i) {
                VReg arg_v = get_vreg(target.args[i]);
                VReg param_v = get_vreg(target.block->param(i));
                uint8_t sz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, sz));
                pcopy->add_use(LirOperand::vreg(arg_v, sz));
            }
            lir_bb.append_inst(std::move(pcopy));
        }
    }

    auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_inst->add_use(LirOperand::label(target.block->id()));
    jmp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(jmp_inst));
}

void X64ISel::lower_branch_if(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    // Only a comparison the analysis folded into this branch is recomputed
    // here; any other is a value in a register like every other condition.
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode()) &&
                        skipped_insts_.count(cmp_inst);

    Condition branch_cond = Condition::NE;

    if (is_fused_cmp) {
        Opcode cmp_op = cmp_inst->opcode();
        const Condition gpr_c = get_comparison_conditions(cmp_op).first;
        const Value* lhs = cmp_inst->operand(0);
        const Value* rhs = cmp_inst->operand(1);

        if (lhs->type().is_float()) {
            append_fused_float_compare(*cmp_inst, lir_bb, branch_cond);
        } else {
            uint8_t sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
            if (sz == 0) sz = 8;
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            LirOpcode test_lir_op = (sz == 4) ? LirOpcode::Test32 : LirOpcode::Test;

            ImmIntInfo rhs_imm = get_imm_int_info(rhs);
            ImmIntInfo lhs_imm = get_imm_int_info(lhs);

            if ((cmp_op == Opcode::eq || cmp_op == Opcode::ne) && ((rhs_imm.is_imm && rhs_imm.val == 0) || (lhs_imm.is_imm && lhs_imm.val == 0))) {
                const Value* non_zero = (rhs_imm.is_imm && rhs_imm.val == 0) ? lhs : rhs;
                if (non_zero->is_instruction() && skipped_insts_.count(non_zero->defining_instruction()) && non_zero->defining_instruction()->opcode() == Opcode::and_) {
                    const Instruction* and_inst = non_zero->defining_instruction();
                    const Value* a = and_inst->operand(0);
                    const Value* b = and_inst->operand(1);
                    ImmIntInfo imm_b = get_imm_int_info(b);
                    ImmIntInfo imm_a = get_imm_int_info(a);

                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    if (imm_b.is_imm && imm_b.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::imm(imm_b.val, sz));
                    } else if (imm_a.is_imm && imm_a.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                        test_lir->add_use(LirOperand::imm(imm_a.val, sz));
                    } else {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                    }
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (cmp_op == Opcode::eq) ? Condition::E : Condition::NE;
                } else {
                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    VReg reg = get_vreg(non_zero);
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (rhs_imm.is_imm && rhs_imm.val == 0) ? gpr_c : swap_relational_condition(gpr_c);
                }
            } else if (rhs_imm.is_imm && rhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = gpr_c;
            } else if (lhs_imm.is_imm && lhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = swap_relational_condition(gpr_c);
            } else if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::imm(rhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                branch_cond = swap_relational_condition(gpr_c);
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                cmp_lir->add_use(LirOperand::imm(lhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(cmp_lir));
            } else {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(cmp_lir));
            }
        }
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
        branch_cond = Condition::NE;
    }

    auto emit_target_args = [&](LirBlock& bb, const BranchTarget& target) {
        if (target.args.empty()) return;
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, sz));
                mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                bb.append_inst(std::move(mov_inst));
            }
        } else {
            auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
            for (size_t i = 0; i < target.args.size(); ++i) {
                VReg arg_v = get_vreg(target.args[i]);
                VReg param_v = get_vreg(target.block->param(i));
                uint8_t sz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, sz));
                pcopy->add_use(LirOperand::vreg(arg_v, sz));
            }
            bb.append_inst(std::move(pcopy));
        }
    };

    if (t_target.args.empty() && f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(f_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    } else if (t_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        jmp_f->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_f));
    } else if (f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(f_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));
    } else {
        auto* false_trampoline = lir_fn_->create_block("br_if_false");
        link_blocks(lir_bb, *false_trampoline);
        if (auto* f_lir = lir_fn_->get_block_by_id(f_target.block->id())) link_blocks(*false_trampoline, *f_lir);

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(false_trampoline->id));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));

        emit_target_args(*false_trampoline, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        false_trampoline->append_inst(std::move(jmp_f));
    }
}

void X64ISel::lower_return(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        const auto* ret_val = inst.operand(0);
        VReg ret_vreg = get_vreg(ret_val);
        Type t = ret_val->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (t.is_float()) {
            LirOpcode ret_mov_op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else if (t.is_vector()) {
            LirOpcode ret_mov_op = (sz == 32) ? LirOpcode::Vmovaps : LirOpcode::Movaps;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else {
            auto mov_ret = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        }
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(ret_inst));
}

std::unique_ptr<LirFunction> lower_to_x64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
) {
    X64ISel isel(target, cc);
    return isel.lower(mir_fn);
}

} // namespace brass::x64
