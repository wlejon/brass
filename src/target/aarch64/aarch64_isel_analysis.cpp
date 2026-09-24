#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/mir/instruction.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

AArch64ISel::ImmIntInfo AArch64ISel::get_imm_int_info(const Value* val) const {
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

bool AArch64ISel::is_value_dead_after(
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

// Which instructions the selector does not lower on their own.
//
// Only two kinds are skipped, and for both the one consumer that absorbs
// them decides unconditionally, so this analysis cannot disagree with it:
//   - a comparison (and an `and` it tests against zero) whose only use is
//     the br_if / guard / select right after it in the same block;
//   - an address `add` whose every use is a memory operation that folds it.
// Constants and loads an instruction may fold as an operand are always
// lowered; when the consumer did fold them their definitions are left
// without uses, and eliminate_dead_materializations() deletes them after
// selection. Predicting the consumer's choice here instead is how an operand
// ended up with no register.
void AArch64ISel::analyze_function(const Function& mir_fn) {
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
            // Switch case arguments are uses too (see the x64 selector).
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

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::br_if || inst->opcode() == Opcode::guard || inst->opcode() == Opcode::select) {
                const Value* cond = inst->operand(0);
                if (cond && cond->is_instruction()) {
                    const Instruction* def_inst = cond->defining_instruction();
                    if (def_inst && def_inst->parent() == bb && is_comparison(def_inst->opcode()) && !narrow_compare(*def_inst)) {
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
            }
        }
    }

    // An address computation folded into a memory operation is read by that
    // operation through the folded operand only. Only the memory operation's
    // own operands count: an instruction folded one level further down is
    // still read by the (lowered or not) instruction between them.
    std::unordered_map<const Value*, uint32_t> folded_uses;
    auto count_direct = [&](const MemFold& mf, std::initializer_list<const Value*> operands) {
        for (const Value* op : operands) {
            if (!op) continue;
            for (const auto* fi : mf.folded_instructions) {
                if (fi && fi->result() == op && fi->opcode() == Opcode::add) {
                    folded_uses[op]++;
                    break;
                }
            }
        }
    };

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (skipped_insts_.count(inst)) continue;
            switch (inst->opcode()) {
                case Opcode::load:
                case Opcode::store:
                    count_direct(match_address(inst->operand(0), inst->offset()), {inst->operand(0)});
                    break;
                case Opcode::load_indexed:
                case Opcode::store_indexed:
                    count_direct(match_indexed_address(inst->operand(0), inst->operand(1),
                                                       x64::scale_from_int(inst->scale()), inst->offset()),
                                 {inst->operand(0), inst->operand(1)});
                    break;
                default:
                    break;
            }
        }
    }

    for (const auto& [val, fold_count] : folded_uses) {
        auto it = use_count_.find(val);
        if (it != use_count_.end() && it->second == fold_count && val->is_instruction()) {
            skipped_insts_.insert(val->defining_instruction());
        }
    }
}

bool AArch64ISel::is_elidable_materialization(const Instruction& inst) const {
    switch (inst.opcode()) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
            return true;
        case Opcode::add:
        case Opcode::shl:
            // Pure integer arithmetic an address may have folded.
            return !inst.type().is_float() && !inst.type().is_vector();
        case Opcode::load:
        case Opcode::load_indexed: {
            // A load whose only consumer read it through a folded memory
            // operand. A load with no consumer at all stays: it may be there
            // for its fault.
            auto it = use_count_.find(inst.result());
            return it != use_count_.end() && it->second == 1;
        }
        default:
            return false;
    }
}

namespace {

void note_vreg_use(std::unordered_map<uint32_t, uint32_t>& uses, const VReg& v) {
    if (v.is_valid()) uses[v.id]++;
}

void note_operand_uses(std::unordered_map<uint32_t, uint32_t>& uses, const LirOperand& op) {
    if (op.is_vreg()) note_vreg_use(uses, op.vreg_val);
    if (op.is_mem()) {
        note_vreg_use(uses, op.mem_val.base_vreg);
        note_vreg_use(uses, op.mem_val.index_vreg);
    }
}

} // namespace

void AArch64ISel::eliminate_dead_materializations() {
    if (elidable_insts_.empty()) return;
    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_map<uint32_t, uint32_t> uses;
        for (const auto& bb : lir_fn_->blocks) {
            for (const auto& li : bb->instructions) {
                if (!li) continue;
                for (const auto& u : li->uses) note_operand_uses(uses, u);
                // A memory destination reads its address registers.
                for (const auto& d : li->defs) {
                    if (d.is_mem()) note_operand_uses(uses, d);
                }
                for (const VReg& g : li->live_gcrefs) note_vreg_use(uses, g);
            }
        }
        for (const VReg& v : lir_fn_->osr_entry.live_in_vregs) note_vreg_use(uses, v);

        for (auto& bb : lir_fn_->blocks) {
            auto& insts = bb->instructions;
            for (auto& li : insts) {
                if (!li || !elidable_insts_.count(li.get())) continue;
                bool dead = !li->defs.empty();
                for (const auto& d : li->defs) {
                    if (!d.is_vreg()) {
                        dead = false;
                        break;
                    }
                    // Its own operands do not keep it alive (`eor v, v, v`).
                    uint32_t own = 0;
                    for (const auto& u : li->uses) {
                        if (u.is_vreg() && u.vreg_val.id == d.vreg_val.id) own++;
                    }
                    auto it = uses.find(d.vreg_val.id);
                    if (it != uses.end() && it->second > own) {
                        dead = false;
                        break;
                    }
                }
                if (dead) {
                    elidable_insts_.erase(li.get());
                    li.reset();
                    changed = true;
                }
            }
            insts.erase(std::remove(insts.begin(), insts.end(), nullptr), insts.end());
        }
    }
    elidable_insts_.clear();
}

} // namespace brass::aarch64
