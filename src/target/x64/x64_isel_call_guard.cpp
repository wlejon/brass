#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/mir/module.hpp>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

void X64ISel::finish_call(LirInst& call, Type ret_t) const {
    call.clobbered_gprs = cc_.caller_saved_gpr_mask();
    call.clobbered_xmms = cc_.caller_saved_xmm_mask();
    if (ret_t.is_void()) return;
    uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
    if (ret_t.is_float() || ret_t.is_vector()) {
        call.add_def(LirOperand::preg_xmm(XMM::XMM0, ret_sz), FixedConstraint::xmm(XMM::XMM0));
    } else {
        call.add_def(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
    }
}

void X64ISel::lower_call(const Instruction& inst, LirBlock& lir_bb) {
    size_t num_args = inst.operand_count();
    size_t start_arg = 0;
    VReg callee_vreg;

    if (inst.opcode() == Opcode::call_indirect) {
        callee_vreg = get_vreg(inst.operand(0));
        start_arg = 1;
        num_args = inst.operand_count() - 1;
    }

    size_t required_stack_space = 0;
    if (cc_.kind() == CallingConvKind::Win64) {
        required_stack_space = std::max(size_t(32), 32 + (num_args > 4 ? (num_args - 4) * 8 : 0));
    } else {
        size_t stack_args = (num_args > 6 ? (num_args - 6) : 0);
        required_stack_space = stack_args * 8;
    }
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);

    auto call_lir = std::make_unique<LirInst>(
        inst.opcode() == Opcode::call_indirect ? LirOpcode::CallIndirect : LirOpcode::Call
    );

    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(start_arg + i);
        VReg arg_vreg = get_vreg(arg_val);
        Type t = arg_val->type();
        if (t.is_vector()) {
            // Vectors travel in XMM registers, matching the tier-2 entry,
            // the baseline JIT and the invoke thunks: by position on Win64
            // (XMM0-3), the next free XMM register on SysV. No tier has a
            // stack layout for vector arguments, so one that would land on
            // the stack is a compile error.
            const uint8_t vsz = static_cast<uint8_t>(t.size_in_bytes());
            XMM xreg;
            if (cc_.kind() == CallingConvKind::Win64) {
                if (i >= 4) throw_unsupported("x64 isel (call)", "vector argument passed on the stack");
                xreg = static_cast<XMM>(i);
            } else {
                if (xmm_idx >= cc_.num_arg_xmms()) throw_unsupported("x64 isel (call)", "vector argument passed on the stack");
                xreg = cc_.arg_xmm(xmm_idx++);
            }
            auto mov_arg = std::make_unique<LirInst>(vsz == 32 ? LirOpcode::Vmovaps : LirOpcode::Movaps);
            mov_arg->add_def(LirOperand::preg_xmm(xreg, vsz), FixedConstraint::xmm(xreg));
            mov_arg->add_use(LirOperand::vreg(arg_vreg, vsz));
            call_lir->add_use(LirOperand::preg_xmm(xreg, vsz), FixedConstraint::xmm(xreg));
            lir_bb.append_inst(std::move(mov_arg));
            continue;
        }
        uint8_t sz = (t.size_in_bytes() == 4) ? 4 : 8;
        LirOpcode mov_op = t.is_float() ? (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                auto mov_arg = std::make_unique<LirInst>(mov_op);
                if (t.is_float()) {
                    XMM xreg = static_cast<XMM>(i);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                }
                lir_bb.append_inst(std::move(mov_arg));
            } else {
                int32_t disp = static_cast<int32_t>(32 + (i - 4) * 8);
                auto mov_stack = std::make_unique<LirInst>(mov_op);
                mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        } else {
            if (t.is_float()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    auto mov_arg = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                    lir_bb.append_inst(std::move(mov_arg));
                } else {
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    xmm_idx++;
                    auto mov_stack = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            } else {
                if (gpr_idx < cc_.num_arg_gprs()) {
                    GPR greg = cc_.arg_gpr(gpr_idx++);
                    auto mov_arg = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    lir_bb.append_inst(std::move(mov_arg));
                } else {
                    size_t stack_idx = (gpr_idx - cc_.num_arg_gprs()) + (xmm_idx > cc_.num_arg_xmms() ? (xmm_idx - cc_.num_arg_xmms()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    gpr_idx++;
                    auto mov_stack = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            }
        }
    }

    if (cc_.kind() == CallingConvKind::SysV) {
        auto mov_al = std::make_unique<LirInst>(LirOpcode::Mov32);
        mov_al->add_def(LirOperand::preg_gpr(GPR::RAX, 4), FixedConstraint::gpr(GPR::RAX));
        mov_al->add_use(LirOperand::imm32(static_cast<int32_t>(std::min(xmm_idx, size_t(8)))));
        lir_bb.append_inst(std::move(mov_al));
        call_lir->add_use(LirOperand::preg_gpr(GPR::RAX, 4), FixedConstraint::gpr(GPR::RAX));
    }

    if (inst.opcode() == Opcode::call_indirect) {
        call_lir->add_use(LirOperand::vreg(callee_vreg, 8));
    } else if (inst.opcode() == Opcode::patchable_call) {
        call_lir->is_patchable = true;
        call_lir->patch_symbol = std::string(inst.symbol());
        std::string callee_name = inst.extra_symbol().empty() ? std::string(inst.symbol()) : std::string(inst.extra_symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->add_use(LirOperand::symbol(callee_name));
    } else {
        std::string callee_name = std::string(inst.symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->entry_cell = std::string(inst.extra_symbol());
        call_lir->add_use(LirOperand::symbol(callee_name));
    }

    Type ret_t = inst.type();
    finish_call(*call_lir, ret_t);

    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.produces_value()) {
        VReg dst = get_vreg(inst.result());
        uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
        if (ret_t.is_vector()) {
            auto mov_ret = std::make_unique<LirInst>(ret_sz == 32 ? LirOpcode::Vmovaps : LirOpcode::Movaps);
            mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
            mov_ret->add_use(LirOperand::preg_xmm(XMM::XMM0, ret_sz), FixedConstraint::xmm(XMM::XMM0));
            lir_bb.append_inst(std::move(mov_ret));
        } else if (ret_t.is_float()) {
            auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd);
            mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
            mov_ret->add_use(LirOperand::preg_xmm(XMM::XMM0, ret_sz), FixedConstraint::xmm(XMM::XMM0));
            lir_bb.append_inst(std::move(mov_ret));
        } else {
            auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
            mov_ret->add_use(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
            lir_bb.append_inst(std::move(mov_ret));
        }
    }
}

void X64ISel::lower_safepoint(const Instruction& inst, LirBlock& lir_bb) {
    if (cc_.kind() == CallingConvKind::Win64) {
        lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, size_t(32));
    }

    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = inst.resume_id();
    sp_inst->clobbered_gprs = cc_.caller_saved_gpr_mask();
    sp_inst->clobbered_xmms = cc_.caller_saved_xmm_mask();

    for (const auto& entry : val_to_vreg_) {
        if (entry.second.is_gcref) {
            sp_inst->live_gcrefs.push_back(entry.second);
        }
    }
    sp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sp_inst));
}

void X64ISel::lower_guard(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    // A comparison the analysis folded into this guard is emitted here, with
    // what it folds (a constant, a load), as a branch's is; the guard leaves
    // when it does not hold (a float one also when its operands are
    // unordered).
    Condition deopt_cond = Condition::E;
    if (fused_into(cmp_inst, inst)) {
        deopt_cond = invert(append_fused_compare(*cmp_inst, inst, lir_bb));
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
    }

    auto* deopt_block = lir_fn_->create_block("guard_deopt");
    link_blocks(lir_bb, *deopt_block);

    auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc_inst->condition = deopt_cond;
    jcc_inst->add_use(LirOperand::label(deopt_block->id));
    lir_bb.append_inst(std::move(jcc_inst));

    // Lower guard deopt exit payload into deopt_block
    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = inst.resume_id();
    exit_inst->deopt_reason = inst.offset() != 0 ? static_cast<uint32_t>(inst.offset()) : 1;
    // Only a label that names a function of the module is an exit stub.
    // The function may be an optimized clone in a module of its own: its
    // callees, the stub among them, are in the callee module.
    const Function* stub = callee_module_ && !inst.symbol().empty() ? callee_module_->get_function(inst.symbol())
                           : mir_fn_                                ? mir_fn_->guard_exit_stub(inst)
                                                                    : nullptr;
    if (mir_fn_ && mir_fn_->return_type().is_vector() && !(stub && stub->is_resume_twin())) {
        // A lower tier's result comes back through one 64-bit word; a
        // resume twin's comes back in the function's own return register.
        codegen::throw_unsupported("x64 isel (guard)", "guard in a function returning a vector");
    }
    if (stub) {
        std::string why;
        if (!mir_fn_->guard_exit_stub_matches(inst, *stub, why)) {
            codegen::throw_unsupported("x64 isel (guard)", why);
        }
        exit_inst->exit_symbol = std::string(inst.symbol());
        exit_inst->twin_exit = stub->is_resume_twin();
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
            codegen::throw_unsupported("x64 isel (guard)", "vector value in a guard state map");
        } else if (t.kind() == TypeKind::F64) {
            kind = runtime::DeoptValueKind::Float64;
        } else if (t.kind() == TypeKind::F32) {
            kind = runtime::DeoptValueKind::Float32;
        } else if (t.is_gcref()) {
            kind = runtime::DeoptValueKind::GcRef;
        } else if (t.is_tagged()) {
            kind = runtime::DeoptValueKind::Tagged;
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

} // namespace brass::x64
