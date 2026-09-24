#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/mir/module.hpp>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;
using LirCond = brass::x64::Condition;
using x64::invert;

void AArch64ISel::lower_call(const Instruction& inst, LirBlock& lir_bb) {
    size_t num_args = inst.operand_count();
    size_t start_arg = 0;
    VReg callee_vreg;

    if (inst.opcode() == Opcode::call_indirect) {
        callee_vreg = get_vreg(inst.operand(0));
        start_arg = 1;
        num_args = inst.operand_count() - 1;
    }

    auto call_lir = std::make_unique<LirInst>(
        inst.opcode() == Opcode::call_indirect ? LirOpcode::CallIndirect : LirOpcode::Call
    );

    size_t gpr_idx = 0;
    size_t fpr_idx = 0;
    size_t stack_bytes = 0;
    bool is_apple = (cc_.kind() == CallingConvKind::AppleAAPCS64);

    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(start_arg + i);
        Type t = arg_val->type();

        if (t.is_v256()) {
            VRegPair pair = get_vreg_pair(arg_val);
            if (fpr_idx + 1 < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                FPR r_hi = static_cast<FPR>(fpr_idx++);
                auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_lo->add_def(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                mov_lo->add_use(LirOperand::vreg(pair.lo, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                lir_bb.append_inst(std::move(mov_lo));

                auto mov_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_hi->add_def(LirOperand::preg_aarch64_fpr(r_hi, 16), FixedConstraint::aarch64_fpr(r_hi));
                mov_hi->add_use(LirOperand::vreg(pair.hi, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_hi, 16), FixedConstraint::aarch64_fpr(r_hi));
                lir_bb.append_inst(std::move(mov_hi));
            } else if (fpr_idx < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_lo->add_def(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                mov_lo->add_use(LirOperand::vreg(pair.lo, 16));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                lir_bb.append_inst(std::move(mov_lo));

                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += 16;

                auto mov_stack = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, 16));
                mov_stack->add_use(LirOperand::vreg(pair.hi, 16));
                lir_bb.append_inst(std::move(mov_stack));
            } else {
                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += 32;

                auto mov_stack_lo = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack_lo->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, 16));
                mov_stack_lo->add_use(LirOperand::vreg(pair.lo, 16));
                lir_bb.append_inst(std::move(mov_stack_lo));

                auto mov_stack_hi = std::make_unique<LirInst>(LirOpcode::Movups);
                mov_stack_hi->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp + 16, 16));
                mov_stack_hi->add_use(LirOperand::vreg(pair.hi, 16));
                lir_bb.append_inst(std::move(mov_stack_hi));
            }
            continue;
        }

        VReg arg_vreg = get_vreg(arg_val);
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;
        bool is_fpr = (t.is_float() || t.is_vector());

        LirOpcode mov_op = is_fpr ? ((sz == 16) ? LirOpcode::Movups : (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd))
                                  : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);

        if (is_fpr) {
            if (fpr_idx < 8) {
                FPR freg = static_cast<FPR>(fpr_idx++);
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                mov_arg->add_def(LirOperand::preg_aarch64_fpr(freg, sz), FixedConstraint::aarch64_fpr(freg));
                mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                call_lir->add_use(LirOperand::preg_aarch64_fpr(freg, sz), FixedConstraint::aarch64_fpr(freg));
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        } else {
            if (gpr_idx < 8) {
                GPR greg = static_cast<GPR>(gpr_idx++);
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                mov_arg->add_def(LirOperand::preg_aarch64_gpr(greg, sz), FixedConstraint::aarch64_gpr(greg));
                mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                call_lir->add_use(LirOperand::preg_aarch64_gpr(greg, sz), FixedConstraint::aarch64_gpr(greg));
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t disp = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::aarch64_gpr(GPR::SP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        }
    }

    size_t required_stack_space = (stack_bytes + 15) & ~size_t(15);
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);
    lir_fn_->frame.has_calls = true;

    if (inst.opcode() == Opcode::call_indirect) {
        call_lir->add_use(LirOperand::vreg(callee_vreg, 8));
    } else if (inst.opcode() == Opcode::patchable_call) {
        call_lir->is_patchable = true;
        call_lir->patch_symbol = std::string(inst.symbol());
        std::string callee_name = inst.extra_symbol().empty() ? std::string(inst.symbol()) : std::string(inst.extra_symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->add_use(LirOperand::symbol(callee_name));
    } else {
        call_lir->callee_symbol = std::string(inst.symbol());
        call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    }

    call_lir->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();

    Type ret_t = inst.type();
    if (!ret_t.is_void()) {
        if (ret_t.is_v256()) {
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
        } else if (ret_t.is_float() || ret_t.is_vector()) {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            call_lir->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, ret_sz), FixedConstraint::aarch64_fpr(FPR::V0));
        } else {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            call_lir->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, ret_sz), FixedConstraint::aarch64_gpr(GPR::X0));
        }
    }

    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.produces_value()) {
        if (ret_t.is_v256()) {
            VRegPair dst_pair = get_vreg_pair(inst.result());
            auto mov_lo = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_lo->add_def(LirOperand::vreg(dst_pair.lo, 16));
            mov_lo->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
            lir_bb.append_inst(std::move(mov_lo));

            auto mov_hi = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_hi->add_def(LirOperand::vreg(dst_pair.hi, 16));
            mov_hi->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 16), FixedConstraint::aarch64_fpr(FPR::V1));
            lir_bb.append_inst(std::move(mov_hi));
        } else {
            VReg dst = get_vreg(inst.result());
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            if (ret_sz == 0) ret_sz = 8;
            if (ret_t.is_float()) {
                auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
                mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
                mov_ret->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, ret_sz), FixedConstraint::aarch64_fpr(FPR::V0));
                lir_bb.append_inst(std::move(mov_ret));
            } else if (ret_t.is_vector()) {
                auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movaps);
                mov_ret->add_def(LirOperand::vreg(dst, 16));
                mov_ret->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 16), FixedConstraint::aarch64_fpr(FPR::V0));
                lir_bb.append_inst(std::move(mov_ret));
            } else {
                auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
                mov_ret->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, ret_sz), FixedConstraint::aarch64_gpr(GPR::X0));
                lir_bb.append_inst(std::move(mov_ret));
            }
        }
    }
}

void AArch64ISel::lower_safepoint(const Instruction& inst, LirBlock& lir_bb) {
    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = inst.resume_id();
    sp_inst->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    sp_inst->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();

    for (const auto& entry : val_to_vreg_) {
        if (entry.second.is_gcref) {
            sp_inst->live_gcrefs.push_back(entry.second);
        }
    }
    sp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sp_inst));
}

void AArch64ISel::lower_write_barrier(const Instruction& inst, LirBlock& lir_bb) {
    const Value* obj_val = inst.operand(0);
    const Value* val_val = inst.operand(1);
    if (!obj_val || !val_val) return;

    VReg obj_vreg = get_vreg(obj_val);
    VReg val_vreg = get_vreg(val_val);

    auto mov_obj = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_obj->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
    mov_obj->add_use(LirOperand::vreg(obj_vreg, 8));
    lir_bb.append_inst(std::move(mov_obj));

    auto mov_val = std::make_unique<LirInst>(LirOpcode::Mov);
    mov_val->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8), FixedConstraint::aarch64_gpr(GPR::X1));
    mov_val->add_use(LirOperand::vreg(val_vreg, 8));
    lir_bb.append_inst(std::move(mov_val));

    auto call_wb = std::make_unique<LirInst>(LirOpcode::Call);
    call_wb->callee_symbol = "brass_gc_write_barrier";
    call_wb->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8), FixedConstraint::aarch64_gpr(GPR::X0));
    call_wb->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8), FixedConstraint::aarch64_gpr(GPR::X1));
    call_wb->add_use(LirOperand::symbol("brass_gc_write_barrier"));
    call_wb->clobbered_gprs = cc_.aarch64_caller_saved_gpr_mask();
    call_wb->clobbered_xmms = cc_.aarch64_caller_saved_fpr_mask();
    call_wb->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_wb));
}

void AArch64ISel::lower_guard(const Instruction& inst, LirBlock& lir_bb) {
    if (mir_fn_ && mir_fn_->return_type().is_vector()) {
        // A lower tier's result comes back through one 64-bit word.
        codegen::throw_unsupported("aarch64 isel (guard)", "guard in a function returning a vector");
    }
    const Value* cond_val = inst.operand(0);
    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode());

    LirCond deopt_cond = LirCond::E;

    if (is_fused_cmp) {
        deopt_cond = invert(emit_fused_compare(*cmp_inst, lir_bb));
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
        deopt_cond = LirCond::E;
    }

    auto* deopt_block = lir_fn_->create_block("guard_deopt");
    link_blocks(lir_bb, *deopt_block);

    auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc_inst->condition = deopt_cond;
    jcc_inst->add_use(LirOperand::label(deopt_block->id));
    lir_bb.append_inst(std::move(jcc_inst));

    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = inst.resume_id();
    exit_inst->deopt_reason = inst.offset() != 0 ? static_cast<uint32_t>(inst.offset()) : 1;
    // Only a label that names a function of the module is an exit stub.
    // The function may be an optimized clone in a module of its own: its
    // callees, the stub among them, are in the callee module.
    const Function* stub = callee_module_ && !inst.symbol().empty() ? callee_module_->get_function(inst.symbol())
                           : mir_fn_                                ? mir_fn_->guard_exit_stub(inst)
                                                                    : nullptr;
    if (stub) {
        std::string why;
        if (!mir_fn_->guard_exit_stub_matches(inst, *stub, why)) {
            codegen::throw_unsupported("aarch64 isel (guard)", why);
        }
        exit_inst->exit_symbol = std::string(inst.symbol());
    }

    // Every state value keeps its position (the resume side maps them by
    // index) and its kind, so the lower tier rebuilds exactly typed values.
    exit_inst->deopt_kinds.reserve(inst.state_map().size());
    for (const auto* val : inst.state_map()) {
        if (!val) {
            exit_inst->add_use(LirOperand::imm(0, 8));
            exit_inst->deopt_kinds.push_back(static_cast<uint8_t>(runtime::DeoptValueKind::Int64));
            continue;
        }
        const Type t = val->type();
        runtime::DeoptValueKind kind;
        if (t.is_vector()) {
            codegen::throw_unsupported("aarch64 isel (guard)", "vector value in a guard state map");
        } else if (t.kind() == TypeKind::F64) {
            kind = runtime::DeoptValueKind::Float64;
        } else if (t.kind() == TypeKind::F32) {
            kind = runtime::DeoptValueKind::Float32;
        } else if (t.is_gcref()) {
            kind = runtime::DeoptValueKind::GcRef;
        } else if (t.is_pointer()) {
            kind = runtime::DeoptValueKind::Pointer;
        } else if (t.kind() == TypeKind::I64) {
            kind = runtime::DeoptValueKind::Int64;
        } else {
            kind = runtime::DeoptValueKind::Int32;
        }
        VReg vr = get_vreg(val);
        exit_inst->add_use(LirOperand::vreg(vr, vr.size));
        exit_inst->deopt_kinds.push_back(static_cast<uint8_t>(kind));
    }
    deopt_block->append_inst(std::move(exit_inst));
}

} // namespace brass::aarch64
