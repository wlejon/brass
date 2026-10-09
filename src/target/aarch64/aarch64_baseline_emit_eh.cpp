// AArch64 baseline tier: exceptions, as the x64 tier compiles them
// (codegen/baseline_emit_eh.cpp).
//
// A baseline frame keeps every value in its fp-relative slots and saves no
// callee-saved register but the pinned-TLS one (described in its .eh_frame),
// and sp never moves after the prologue, so a throw can land in it knowing
// only fp: a pad resets sp to fp - frame_bytes and reads the thrown bits in
// X0. A throw reaches the pad by whichever route found the frame: the
// unwinder-driven raise (exception_raise_unwind.cpp; the function's range is
// in the JIT registry with its scopes), or, on Apple, a C++ BrassException
// the C++ unwinder carries up to the frame, whose .eh_frame CIE names
// brass_sysv_personality. Windows ARM64 keeps such a function in Tier 0
// (check_supported): its .xdata names no handler.
#include "aarch64_baseline_emit_internal.hpp"
#include <brass/runtime/exception.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

void AArch64BaselineEmitter::emit_invoke(const Instruction& inst) {
    std::vector<const Value*> args;
    args.reserve(inst.operand_count());
    for (size_t i = 0; i < inst.operand_count(); ++i) args.push_back(inst.operand(i));
    uint32_t call_start = 0, call_end = 0;
    emit_call(inst.symbol(), nullptr, args, inst.produces_value() ? inst.result() : nullptr, inst.site_id(),
              &call_start, &call_end);

    const BranchTarget& unwind = inst.unwind_target();
    Label pad;
    if (unwind.args.empty()) {
        pad = block_labels.at(unwind.block->id());
    } else {
        pad = buffer.create_label();
        unwind_trampolines.emplace_back(pad, &inst);
    }
    eh_scopes.push_back(EhScope{call_start, call_end, pad});

    copy_block_args(inst.normal_target());
    enc.b(block_labels.at(inst.normal_target().block->id()));
}

void AArch64BaselineEmitter::emit_landing_pad(const Instruction& inst) {
    // Entered only by a throw, with fp restored by the unwinder. sp is as it
    // was at the call, which is where the prologue left it; set again so the
    // pad does not depend on how the raise computed it.
    enc.sub(GPR::SP, GPR::FP, static_cast<uint32_t>(frame_bytes));
    if (inst.produces_value() && inst.result()) {
        if (inst.type().is_vector()) throw_unsupported(kA64BaselineStage, "a vector exception value");
        store_gpr(inst.result(), GPR::X0);
    }
}

void AArch64BaselineEmitter::emit_raise(const Instruction& inst) {
    if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
        const Value* v = inst.operand(0);
        if (v->type().is_vector()) throw_unsupported(kA64BaselineStage, "a vector exception value");
        load_gpr(GPR::X0, v);
        call_abs(reinterpret_cast<const void*>(&runtime::brass_default_throw));
    } else {
        call_abs(reinterpret_cast<const void*>(&runtime::brass_default_rethrow));
    }
    // Neither returns. The trap keeps the return address inside the
    // function, where the unwinder looks the frame up.
    enc.brk(0);
}

void AArch64BaselineEmitter::emit_unwind_trampolines() {
    for (const auto& [label, inv] : unwind_trampolines) {
        buffer.bind(label);
        const BranchTarget& unwind = inv->unwind_target();
        // copy_block_args goes through X0, which holds the thrown bits.
        enc.mov(GPR::X10, GPR::X0);
        copy_block_args(unwind);
        enc.mov(GPR::X0, GPR::X10);
        enc.b(block_labels.at(unwind.block->id()));
    }
}

} // namespace brass::aarch64
