// x64 baseline tier: exceptions.
//
// A baseline frame keeps every value in its RBP-relative slots and saves no
// callee-saved register but R13 (pinned TLS, at [RBP-8]), so a throw can land
// in it knowing only RBP: a pad resets RSP to the body's RSP, which the
// unwinder left wherever the call was made from, and reads the thrown bits in
// RAX. A throw reaches the pad by whichever route found the frame: the brass
// frame walker (the function's range is registered with its scopes), or the
// OS unwinder (the Win64 unwind data names brass's personality and carries
// the scope table), which also lands a C++ BrassException thrown by a helper.
#include "baseline_emit_internal.hpp"

namespace brass::codegen {

using namespace brass::x64;

void X64BaselineEmitter::emit_invoke(const Instruction& inst) {
    std::vector<const Value*> args;
    args.reserve(inst.operand_count());
    for (size_t i = 0; i < inst.operand_count(); ++i) args.push_back(inst.operand(i));
    uint32_t call_start = 0, call_end = 0;
    emit_call(inst.symbol(), nullptr, args, inst.produces_value() ? inst.result() : nullptr, inst.site_id(),
              &call_start, &call_end);

    const BranchTarget& unwind = inst.unwind_target();
    Label pad;
    if (unwind.args.empty()) {
        pad = block_label(unwind.block);
    } else {
        pad = buffer.create_label();
        unwind_trampolines.emplace_back(pad, &inst);
    }
    eh_scopes.push_back(EhScope{call_start, call_end, pad});

    copy_block_args(inst.normal_target());
    enc.jmp(block_label(inst.normal_target().block));
}

void X64BaselineEmitter::emit_landing_pad(const Instruction& inst) {
    // Entered only by a throw: the unwinder restored RBP, and RSP as it was
    // at the call (below the call's stack arguments). A pad in a loop would
    // otherwise leak that much stack per catch.
    enc.lea(GPR::RSP, MemAddress::base_disp(GPR::RBP, -frame_size));
    if (inst.produces_value() && inst.result()) {
        if (inst.type().is_vector()) throw_unsupported(kX64BaselineStage, "a vector exception value");
        store_gpr(inst.result(), GPR::RAX);
    }
}

void X64BaselineEmitter::emit_raise(const Instruction& inst) {
    const bool win = target.is_windows();
    // The body's RSP is 16-byte aligned; Win64 wants the callee's shadow space.
    if (win) enc.sub(GPR::RSP, 32);
    if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
        const Value* v = inst.operand(0);
        if (v->type().is_vector()) throw_unsupported(kX64BaselineStage, "a vector exception value");
        load_gpr(win ? GPR::RCX : GPR::RDI, v);
        call_abs(reinterpret_cast<const void*>(&runtime::brass_throw));
    } else {
        call_abs(reinterpret_cast<const void*>(&runtime::brass_rethrow));
    }
    // Neither returns. The trap keeps the return address inside the
    // function, where the unwinder looks the frame up.
    enc.ud2();
}

void X64BaselineEmitter::emit_unwind_trampolines() {
    for (const auto& [label, inv] : unwind_trampolines) {
        buffer.bind(label);
        const BranchTarget& unwind = inv->unwind_target();
        // copy_block_args goes through RAX, which holds the thrown bits.
        enc.mov(GPR::R10, GPR::RAX);
        copy_block_args(unwind);
        enc.mov(GPR::RAX, GPR::R10);
        enc.jmp(block_label(unwind.block));
    }
}

runtime::FunctionExceptionTable baseline_exception_table(const X64BaselineEmitter& em, std::string_view name) {
    runtime::FunctionExceptionTable table{std::string(name), 0, static_cast<uint32_t>(em.buffer.size())};
    table.set_frame_size(static_cast<uint32_t>(em.frame_size));
    table.set_saved_callee_gprs(em.preserves_r13 ? (1u << 13) : 0u);
    for (const auto& s : em.eh_scopes) {
        table.add_scope(s.begin, s.end, static_cast<uint32_t>(em.buffer.label_offset(s.pad)));
    }
    return table;
}

} // namespace brass::codegen
