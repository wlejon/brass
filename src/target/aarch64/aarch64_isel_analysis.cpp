#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/mir/instruction.hpp>

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
                    if (def_inst && def_inst->parent() == bb && is_comparison(def_inst->opcode())) {
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

    std::unordered_map<const Value*, uint32_t> folded_uses;

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (skipped_insts_.count(inst)) continue;

            if (inst->opcode() == Opcode::load) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::store) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                const Value* src = inst->operand(1);
                ImmIntInfo imm_src = get_imm_int_info(src);
                if (imm_src.is_imm && imm_src.fits_i32) {
                    folded_uses[src]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::load_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), x64::scale_from_int(inst->scale()), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::store_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), x64::scale_from_int(inst->scale()), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
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
                        } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), inst)) {
                            skipped_insts_.insert(rhs->defining_instruction());
                        }
                        continue;
                    }
                }
            }

            bool is_alu = false;
            switch (inst->opcode()) {
                case Opcode::add: case Opcode::sub: case Opcode::mul:
                case Opcode::and_: case Opcode::or_: case Opcode::xor_:
                case Opcode::shl: case Opcode::lshr: case Opcode::ashr:
                case Opcode::udiv: case Opcode::umod:
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

                bool is_comm = (inst->opcode() == Opcode::add || inst->opcode() == Opcode::mul ||
                                inst->opcode() == Opcode::and_ || inst->opcode() == Opcode::or_ ||
                                inst->opcode() == Opcode::xor_);

                if (imm1.is_imm && imm1.fits_i32) {
                    folded_uses[op1]++;
                } else if (is_comm && imm0.is_imm && imm0.fits_i32) {
                    folded_uses[op0]++;
                } else if (op1 && op1->is_instruction() && can_fuse_load(op1->defining_instruction(), inst)) {
                    skipped_insts_.insert(op1->defining_instruction());
                } else if (is_comm && op0 && op0->is_instruction() && can_fuse_load(op0->defining_instruction(), inst)) {
                    skipped_insts_.insert(op0->defining_instruction());
                }
            }
        }
    }

    for (const auto& [val, fold_count] : folded_uses) {
        auto it = use_count_.find(val);
        if (it != use_count_.end() && it->second == fold_count) {
            if (val->is_instruction()) {
                skipped_insts_.insert(val->defining_instruction());
            }
        }
    }
}

} // namespace brass::aarch64
