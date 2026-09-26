// x64 baseline tier: the coroutine operations, as calls into the runtime.
// coro_create allocates a frame of the callee's body descriptor (its frames
// resume in the body's best tier, whichever tier created them) and stores
// the arguments in its slots; coro_resume and coro_destroy are the entries
// every other native tier calls.
#include "baseline_emit_internal.hpp"
#include <brass/mir/coro_transform.hpp>
#include <brass/runtime/coroutine.hpp>
#include <string>

namespace brass::codegen {

using namespace brass::x64;

namespace {

GPR arg_gpr(const X64BaselineEmitter& em, size_t i) {
    static constexpr GPR kWin[4] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
    return em.target.is_windows() ? kWin[i] : em.cc.arg_gprs()[i];
}

void emit_create(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    void* body = em.coro_body ? em.coro_body(inst.symbol()) : nullptr;
    if (!body) {
        throw_unsupported(kX64BaselineStage, "coro_create of " + std::string(inst.symbol()) +
                          ", which has no body descriptor");
    }
    enc.movabs(arg_gpr(em, 0), reinterpret_cast<uint64_t>(body));
    em.call_abs(reinterpret_cast<const void*>(&brass_coro_create_body));
    // The allocation may collect: this frame's roots are described here.
    em.record_safepoint(inst.site_id());
    if (inst.result()) enc.mov(em.slot_addr(inst.result()), GPR::RAX);
    // Argument i fills consecutive slots from coro_slot_count of those
    // before it, where the lowered body loads it. No collection comes
    // between the allocation and these stores, and a frame allocated old is
    // remembered by the runtime, so they need no barrier.
    uint32_t slot = 0;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        const Value* arg = inst.operand(i);
        const int32_t off = static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8);
        slot += coro_slot_count(arg->type());
        if (bl_is_v128(arg->type())) {
            enc.movups(XMM::XMM0, em.slot_addr(arg));
            enc.movups(MemAddress::base_disp(GPR::RAX, off), XMM::XMM0);
        } else {
            enc.mov(GPR::RCX, em.slot_addr(arg));
            enc.mov(MemAddress::base_disp(GPR::RAX, off), GPR::RCX);
        }
    }
}

void emit_resume(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const GPR a0 = arg_gpr(em, 0), a1 = arg_gpr(em, 1), a2 = arg_gpr(em, 2);
    enc.mov(a0, em.slot_addr(inst.operand(0)));
    if (inst.operand_count() > 1 && inst.operand(1)) em.load_gpr(a1, inst.operand(1));
    else enc.xor_(a1, a1);
    if (inst.operand_count() > 2 && inst.operand(2)) em.load_gpr(a2, inst.operand(2));
    else enc.xor_(a2, a2);
    // The entry for generated callers: a throw from the body is raised
    // again natively, so this frame's pads see it.
    em.call_abs(reinterpret_cast<const void*>(&brass_coro_resume_from_generated));
    em.record_safepoint(inst.site_id());
    if (inst.result()) em.store_gpr(inst.result(), GPR::RAX);
}

void emit_destroy(X64BaselineEmitter& em, const Instruction& inst) {
    em.enc.mov(arg_gpr(em, 0), em.slot_addr(inst.operand(0)));
    em.call_abs(reinterpret_cast<const void*>(&brass_coro_destroy));
    em.record_safepoint(inst.site_id());
}

} // namespace

bool emit_baseline_x64_coro_op(X64BaselineEmitter& em, const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::coro_create:
            emit_create(em, inst);
            return true;
        case Opcode::coro_resume:
            emit_resume(em, inst);
            return true;
        case Opcode::coro_destroy:
            emit_destroy(em, inst);
            return true;
        default:
            return false;
    }
}

} // namespace brass::codegen
