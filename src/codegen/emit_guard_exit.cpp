// x64 guard exits: the out-of-line code a failed guard runs. Two forms.
//
// A guard whose exit is a resume twin (Function::is_resume_twin) leaves
// directly: its state values go into 8-byte slots under the frame and the
// exit is `twin(resume id, &slots)`, whose return registers are the
// function's. No runtime is involved, so this is the form an AOT image's
// speculated functions use (docs/speculation_and_deopt.md, "Guards in AOT
// images").
//
// Any other guard builds a runtime::DeoptExitRecord and asks the runtime
// (brass_deopt_exit_record) for a lower tier to finish the call in; when
// none does, its exit stub is called as stub(state values...).

#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/deopt.hpp>

#include <cstring>

namespace brass::codegen {

using namespace brass::x64;

namespace {

// Where each state value goes when a guard's exit stub is called as
// stub(state values...). The verifier matched the stub's parameters to the
// state values, so each value's kind picks its argument class.
struct ExitStubArgLoc {
    bool is_float = false;
    int reg = -1;    // register index, or -1 for a stack argument
    int stack = -1;  // stack argument index
};

struct ExitStubArgPlan {
    std::vector<ExitStubArgLoc> locs;
    size_t stack_args = 0;
    // Bytes at the bottom of the guard exit's allocation for the stub call:
    // shadow space, then the stack arguments; a multiple of 16.
    size_t out_area = 0;
};

ExitStubArgPlan plan_exit_stub_args(const LirInst& inst, bool win64) {
    ExitStubArgPlan plan;
    const size_t n = inst.deopt_kinds.size();
    plan.locs.resize(n);
    size_t gpr_used = 0, xmm_used = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto kind = static_cast<runtime::DeoptValueKind>(inst.deopt_kinds[i]);
        ExitStubArgLoc loc;
        loc.is_float = kind == runtime::DeoptValueKind::Float32 || kind == runtime::DeoptValueKind::Float64;
        if (win64) {
            if (i < 4) loc.reg = static_cast<int>(i);
        } else if (loc.is_float) {
            if (xmm_used < 8) loc.reg = static_cast<int>(xmm_used++);
        } else {
            if (gpr_used < 6) loc.reg = static_cast<int>(gpr_used++);
        }
        if (loc.reg < 0) loc.stack = static_cast<int>(plan.stack_args++);
        plan.locs[i] = loc;
    }
    const size_t shadow = win64 ? 32 : 0;
    plan.out_area = (shadow + plan.stack_args * 8 + 15) & ~size_t(15);
    return plan;
}

constexpr size_t kMaxExitAlloc = 0x7FFF0000u;

} // namespace

void EmitContext::emit_exit_stub_call(const LirInst& inst, int32_t slots_disp) {
    // Slots hold a float in its low bits and a narrow integer sign-extended.
    // The stack arguments go into the outgoing area the guard exit allocated
    // under the record together with it, so RSP does not move here and the
    // guard exit's unwind region stays exact across the call.
    const bool win64 = fn_.calling_conv.kind() == CallingConvKind::Win64;
    static constexpr GPR kWinGpr[] = {GPR::RCX, GPR::RDX, GPR::R8, GPR::R9};
    static constexpr GPR kSysvGpr[] = {GPR::RDI, GPR::RSI, GPR::RDX, GPR::RCX, GPR::R8, GPR::R9};
    static constexpr XMM kXmm[] = {XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3,
                                   XMM::XMM4, XMM::XMM5, XMM::XMM6, XMM::XMM7};
    const ExitStubArgPlan plan = plan_exit_stub_args(inst, win64);
    const size_t shadow = win64 ? 32 : 0;
    if (static_cast<size_t>(slots_disp) < plan.out_area) {
        throw_unsupported("x64 emit (guard exit)", "exit stub arguments overlap the deopt record");
    }
    const size_t n = plan.locs.size();
    auto slot = [&](size_t i) { return ptr(GPR::RSP, slots_disp + static_cast<int32_t>(i * 8)); };
    for (size_t i = 0; i < n; ++i) {
        if (plan.locs[i].stack < 0) continue;
        enc_.mov(GPR::R11, slot(i));
        enc_.mov(ptr(GPR::RSP, static_cast<int32_t>(shadow) + plan.locs[i].stack * 8), GPR::R11);
    }
    for (size_t i = 0; i < n; ++i) {
        const auto& loc = plan.locs[i];
        if (loc.reg < 0) continue;
        if (loc.is_float) {
            enc_.movsd(kXmm[loc.reg], slot(i));
        } else {
            enc_.mov(win64 ? kWinGpr[loc.reg] : kSysvGpr[loc.reg], slot(i));
        }
    }
    enc_.call(inst.exit_symbol);
}

size_t EmitContext::open_guard_exit_area(size_t bytes) {
    if (bytes > kMaxExitAlloc) {
        throw_unsupported("x64 emit (guard exit)", "deopt state map too large for one frame");
    }
    // Touch every page of a large area in order before moving RSP (Windows
    // commits stack lazily), like __chkstk: RSP moves once, so the unwind
    // info (the primary's before the sub, this exit's region after it) is
    // exact at every instruction.
    constexpr size_t kPage = 4096;
    for (size_t off = kPage; off < bytes; off += kPage) {
        enc_.mov(ptr(GPR::RSP, -static_cast<int32_t>(off)), GPR::R11);
    }
    if (bytes != 0) enc_.sub(GPR::RSP, static_cast<int32_t>(bytes));
    return buffer_.size();
}

void EmitContext::store_guard_state(const LirInst& inst, int32_t slots_disp) {
    // Every slot is 8 bytes: a float in its low bits, a narrow integer
    // sign-extended. Nothing has moved since the guard's branch, so each
    // value is read from where the allocator left it there.
    for (size_t i = 0; i < inst.uses.size(); ++i) {
        const auto& op = inst.uses[i];
        const int32_t slot_offset = slots_disp + static_cast<int32_t>(i * 8);
        if (op.is_preg()) {
            if (op.preg_val.is_gpr()) {
                const GPR src_gpr = op.preg_val.as_gpr();
                if (op.size == 4) {
                    enc_.movsxd(GPR::R11, src_gpr);
                    enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
                } else {
                    enc_.mov(ptr(GPR::RSP, slot_offset), src_gpr);
                }
            } else if (op.preg_val.is_xmm()) {
                enc_.movsd(ptr(GPR::RSP, slot_offset), op.preg_val.as_xmm());
            }
        } else if (op.is_spill_slot() || op.is_mem()) {
            enc_.mov(GPR::R11, to_mem_address(op));
            enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
        } else if (op.is_imm_int()) {
            enc_.mov(GPR::R11, op.imm_int);
            enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
        } else if (op.is_imm_float()) {
            uint64_t bits = 0;
            if (op.size == 4) {
                const float f = static_cast<float>(op.imm_float);
                uint32_t b32 = 0;
                std::memcpy(&b32, &f, sizeof(b32));
                bits = b32;
            } else {
                std::memcpy(&bits, &op.imm_float, sizeof(bits));
            }
            enc_.mov(GPR::R11, static_cast<int64_t>(bits));
            enc_.mov(ptr(GPR::RSP, slot_offset), GPR::R11);
        } else {
            throw_unsupported("x64 emit (guard exit)",
                              std::string("state value operand of kind ") + std::string(to_string(op.kind)));
        }
    }
}

void EmitContext::emit_guard_exit_epilogue() {
    codegen::FrameInfo mutable_frame = fn_.frame;
    X64FrameLayout::compute_layout(mutable_frame, fn_.calling_conv);
    X64FrameLayout::emit_epilogue(enc_, mutable_frame, fn_.calling_conv);
}

void EmitContext::emit_twin_guard_exit(const LirInst& inst) {
    // twin(resume id, &slots): the slots are this frame's until the twin has
    // read them, which it does before its first GC point, so a reference in
    // one needs no root here. What stays a root of this frame across the
    // call is its tagged locals (a state value may be the address of one),
    // which the call's stack map lists like any other call's.
    const bool win64 = fn_.calling_conv.kind() == CallingConvKind::Win64;
    const size_t shadow = win64 ? 32 : 0;
    const size_t total_alloc = (shadow + inst.uses.size() * 8 + 15) & ~size_t(15);
    const size_t adjust_begin = open_guard_exit_area(total_alloc);
    store_guard_state(inst, static_cast<int32_t>(shadow));
    enc_.mov(win64 ? GPR::RCX : GPR::RDI, static_cast<int32_t>(inst.resume_id));
    enc_.lea(win64 ? GPR::RDX : GPR::RSI, ptr(GPR::RSP, static_cast<int32_t>(shadow)));
    clear_upper_state();
    enc_.call(inst.exit_symbol);
    record_gc_point(inst, buffer_.size(), nullptr);
    if (total_alloc != 0) {
        enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
        stack_adjust_regions_.push_back({static_cast<uint32_t>(adjust_begin), static_cast<uint32_t>(buffer_.size()),
                                         static_cast<uint32_t>(total_alloc)});
    }
    // The twin's result is in the return registers already; a 256-bit one
    // stays in YMM0 for the caller.
    emit_guard_exit_epilogue();
}

void EmitContext::emit_guard_exit(const LirInst& inst) {
    const size_t num_uses = inst.uses.size();
    if (inst.deopt_kinds.size() != num_uses) {
        throw_unsupported("x64 emit (guard exit)", "state map without per-value kinds");
    }
    if (inst.twin_exit) {
        emit_twin_guard_exit(inst);
        return;
    }
    // Builds a runtime::DeoptExitRecord on the stack (header, 8-byte slots,
    // kind bytes; no size limit) and calls brass_deopt_exit_record with its
    // address. Isel names an exit symbol only for a guard with an exit stub.
    const bool has_exit_symbol = !inst.exit_symbol.empty();
    const size_t header_bytes = runtime::DeoptExitRecord::kSlotsOffset;
    const size_t slots_bytes = num_uses * 8;
    const size_t kinds_bytes = (num_uses + 7) & ~size_t(7);
    const bool win64 = fn_.calling_conv.kind() == CallingConvKind::Win64;
    const size_t shadow_space = win64 ? 32 : 0;
    // Below the record: the outgoing area of the calls made here, i.e. the
    // shadow space, plus the exit stub's stack arguments. One allocation
    // holds both, so RSP is total_alloc under the prologue's frame for every
    // call this exit makes.
    const size_t out_area = has_exit_symbol ? plan_exit_stub_args(inst, win64).out_area : shadow_space;
    const size_t record_bytes = header_bytes + slots_bytes + kinds_bytes;
    if (out_area > kMaxExitAlloc || record_bytes > kMaxExitAlloc - out_area) {
        throw_unsupported("x64 emit (guard exit)", "deopt state map too large for one frame");
    }
    const size_t total_alloc = ((out_area + record_bytes + 15) & ~size_t(15));
    // From here to the `add rsp` that releases it, RSP is total_alloc under
    // the prologue's frame (the unwind info says so).
    const size_t adjust_begin = open_guard_exit_area(total_alloc);
    const int32_t rec_disp = static_cast<int32_t>(out_area);
    const int32_t slots_disp = rec_disp + static_cast<int32_t>(header_bytes);
    const int32_t kinds_disp = slots_disp + static_cast<int32_t>(slots_bytes);

    store_guard_state(inst, slots_disp);
    // Kind bytes, eight per store.
    for (size_t i = 0; i < num_uses; i += 8) {
        uint64_t packed = 0;
        for (size_t j = 0; j < 8 && i + j < num_uses; ++j) {
            packed |= static_cast<uint64_t>(inst.deopt_kinds[i + j]) << (8 * j);
        }
        enc_.mov(GPR::R11, static_cast<int64_t>(packed));
        enc_.mov(ptr(GPR::RSP, kinds_disp + static_cast<int32_t>(i)), GPR::R11);
    }

    const uint32_t rid = inst.resume_id;
    const uint32_t rsn = inst.deopt_reason == 0 ? 1 : inst.deopt_reason;
    const uint32_t cnt = static_cast<uint32_t>(num_uses);
    const uint32_t flags = has_exit_symbol ? runtime::DeoptExitRecord::kHasExitSymbol : 0u;

    // Header: code_entry (this function's own entry), resume id, reason,
    // count, flags.
    enc_.lea(GPR::R11, fn_.name);
    enc_.mov(ptr(GPR::RSP, rec_disp + 0), GPR::R11);
    enc_.mov(GPR::R11, static_cast<int64_t>((static_cast<uint64_t>(rsn) << 32) | rid));
    enc_.mov(ptr(GPR::RSP, rec_disp + 8), GPR::R11);
    enc_.mov(GPR::R11, static_cast<int64_t>((static_cast<uint64_t>(flags) << 32) | cnt));
    enc_.mov(ptr(GPR::RSP, rec_disp + 16), GPR::R11);

    const GPR arg0 = win64 ? GPR::RCX : GPR::RDI;
    enc_.lea(arg0, ptr(GPR::RSP, rec_disp));
    // The state values are in the record now; nothing after this point reads
    // a YMM register.
    clear_upper_state();
    enc_.call("brass_deopt_exit_record");

    // RAX points at the lower tier's result bits for this frame.
    auto emit_return_rax_result = [&]() {
        enc_.mov(GPR::RAX, ptr(GPR::RAX, 0));
        if (fn_.return_type.kind() == TypeKind::F64) {
            enc_.movq(XMM::XMM0, GPR::RAX);
        } else if (fn_.return_type.kind() == TypeKind::F32) {
            enc_.movd(XMM::XMM0, GPR::RAX);
        }
        emit_guard_exit_epilogue();
    };

    Label unhandled = buffer_.create_label();
    enc_.test(GPR::RAX, GPR::RAX);
    enc_.j(Condition::E, unhandled);
    // The region covers each `add rsp` itself (it has not run yet at its own
    // address) and nothing after it: the code there runs at the prologue's
    // RSP, which the gap entries describe.
    auto push_region = [&](size_t begin) {
        stack_adjust_regions_.push_back({static_cast<uint32_t>(begin), static_cast<uint32_t>(buffer_.size()),
                                         static_cast<uint32_t>(total_alloc)});
    };
    enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
    push_region(adjust_begin);
    emit_return_rax_result();
    buffer_.bind(unhandled);
    const size_t unhandled_begin = buffer_.size();

    if (has_exit_symbol) {
        // No handler or resumer: the exit stub finishes the call, called as
        // stub(state values...) with the values read back from the record,
        // which is still on the stack. Its return registers are the
        // function's.
        emit_exit_stub_call(inst, slots_disp);
        enc_.add(GPR::RSP, static_cast<int32_t>(total_alloc));
        push_region(unhandled_begin);
        emit_guard_exit_epilogue();
    } else {
        // brass_deopt_exit_record aborts when nothing resumes a guard
        // without an exit stub: never reached.
        enc_.ud2();
        push_region(unhandled_begin);
    }
}

} // namespace brass::codegen
