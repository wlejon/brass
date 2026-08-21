#include <brass/target/x64/x64_isel.hpp>
#include <climits>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::lower_switch(const Instruction& inst, LirBlock& lir_bb) {
    const Value* val = inst.operand(0);
    VReg val_vreg = get_vreg(val);
    uint8_t sz = val_vreg.size;

    auto emit_target_args = [&](LirBlock& bb, const BranchTarget& target) {
        if (target.args.empty()) return;
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t psz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? LirOpcode::Movsd : (psz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, psz));
                mov_inst->add_use(LirOperand::vreg(arg_v, psz));
                bb.append_inst(std::move(mov_inst));
            }
        } else {
            auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
            for (size_t i = 0; i < target.args.size(); ++i) {
                VReg arg_v = get_vreg(target.args[i]);
                VReg param_v = get_vreg(target.block->param(i));
                uint8_t psz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, psz));
                pcopy->add_use(LirOperand::vreg(arg_v, psz));
            }
            bb.append_inst(std::move(pcopy));
        }
    };

    const auto& cases = inst.switch_cases();
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& sc = cases[i];
        int64_t case_val = sc.value;

        if (sz == 4) {
            auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp32);
            cmp_inst->add_use(LirOperand::vreg(val_vreg, 4));
            cmp_inst->add_use(LirOperand::imm(static_cast<int32_t>(case_val), 4));
            lir_bb.append_inst(std::move(cmp_inst));
        } else {
            if (case_val >= INT32_MIN && case_val <= INT32_MAX) {
                auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp);
                cmp_inst->add_use(LirOperand::vreg(val_vreg, 8));
                cmp_inst->add_use(LirOperand::imm(case_val, 8));
                lir_bb.append_inst(std::move(cmp_inst));
            } else {
                VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 8);
                auto movabs = std::make_unique<LirInst>(LirOpcode::Movabs);
                movabs->add_def(LirOperand::vreg(tmp, 8));
                movabs->add_use(LirOperand::imm(case_val, 8));
                lir_bb.append_inst(std::move(movabs));

                auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp);
                cmp_inst->add_use(LirOperand::vreg(val_vreg, 8));
                cmp_inst->add_use(LirOperand::vreg(tmp, 8));
                lir_bb.append_inst(std::move(cmp_inst));
            }
        }

        if (sc.target.args.empty()) {
            auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
            jcc->condition = Condition::E;
            jcc->add_use(LirOperand::label(sc.target.block->id()));
            lir_bb.append_inst(std::move(jcc));
        } else {
            auto* case_trampoline = lir_fn_->create_block("switch_case");
            case_trampoline->predecessors.push_back(&lir_bb);
            lir_bb.successors.push_back(case_trampoline);
            auto* target_lir = lir_fn_->get_block_by_id(sc.target.block->id());
            if (target_lir) {
                case_trampoline->successors.push_back(target_lir);
                target_lir->predecessors.push_back(case_trampoline);
            }

            auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
            jcc->condition = Condition::E;
            jcc->add_use(LirOperand::label(case_trampoline->id));
            lir_bb.append_inst(std::move(jcc));

            emit_target_args(*case_trampoline, sc.target);
            auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
            jmp->add_use(LirOperand::label(sc.target.block->id()));
            case_trampoline->append_inst(std::move(jmp));
        }
    }

    // Default branch (fallthrough)
    emit_target_args(lir_bb, inst.default_target());
    auto jmp_def = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_def->add_use(LirOperand::label(inst.default_target().block->id()));
    jmp_def->mir_origin = &inst;
    lir_bb.append_inst(std::move(jmp_def));
}

} // namespace brass::x64
