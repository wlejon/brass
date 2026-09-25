// AArch64 guard exit: the deopt path of a failed speculation guard.
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/runtime/deopt.hpp>
#include <cstring>
#include <string>
#include <vector>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

// Where each state value goes when a guard's exit stub is called as
// stub(state values...): X0-X7 and V0-V7 fill independently, and the rest
// go on the stack in an 8-byte slot each, or on Apple at their natural size
// and alignment (aarch64_isel.cpp's parameter lowering). The verifier
// matched the stub's parameters to the state values, so each value's kind
// picks its class and its operand its size.
struct ExitStubArgLoc {
    bool is_float = false;
    int reg = -1;           // register index, or -1 for a stack argument
    uint32_t stack_off = 0; // from SP at the call
    uint8_t size = 8;       // bytes stored on the stack
};

struct ExitStubArgPlan {
    std::vector<ExitStubArgLoc> locs;
    // Bytes at the bottom of the guard exit's allocation for the stack
    // arguments; a multiple of 16.
    size_t out_area = 0;
};

ExitStubArgPlan plan_exit_stub_args(const LirInst& inst, bool apple) {
    ExitStubArgPlan plan;
    const size_t n = inst.deopt_kinds.size();
    plan.locs.resize(n);
    int gprs = 0, fprs = 0;
    uint32_t stack = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto kind = static_cast<runtime::DeoptValueKind>(inst.deopt_kinds[i]);
        ExitStubArgLoc& loc = plan.locs[i];
        loc.is_float = kind == runtime::DeoptValueKind::Float32 || kind == runtime::DeoptValueKind::Float64;
        int& used = loc.is_float ? fprs : gprs;
        if (used < 8) {
            loc.reg = used++;
            continue;
        }
        uint32_t sz = inst.uses[i].size;
        if (sz != 1 && sz != 2 && sz != 4) sz = 8;
        const uint32_t align = apple ? sz : 8;
        stack = (stack + align - 1) & ~(align - 1);
        loc.stack_off = stack;
        loc.size = static_cast<uint8_t>(apple ? sz : 8);
        stack += loc.size;
    }
    plan.out_area = (static_cast<size_t>(stack) + 15) & ~size_t(15);
    return plan;
}

} // namespace

// SP -= bytes (allocate) or SP += bytes (release) in one instruction that
// writes SP: an immediate add/sub encodes 12 bits, optionally shifted by 12,
// and a larger amount is formed in X16 first (a split immediate would move
// SP twice, the first time to a height no unwind info describes). X16 is
// free here: it carries no value across either end of the record.
void AArch64EmitContext::move_sp_for_guard_exit(bool allocate, size_t bytes) {
    const uint32_t imm = static_cast<uint32_t>(bytes);
    const bool single = imm <= 4095 || ((imm & 0xFFFu) == 0 && (imm >> 12) <= 4095);
    if (single) {
        if (allocate) enc_.sub(GPR::SP, GPR::SP, imm);
        else enc_.add(GPR::SP, GPR::SP, imm);
        return;
    }
    if (allocate) enc_.sub(GPR::X16, GPR::SP, imm);
    else enc_.add(GPR::X16, GPR::SP, imm);
    enc_.add(GPR::SP, GPR::X16, 0u);
}

// Builds a runtime::DeoptExitRecord on the stack (header, 8-byte slots, kind
// bytes; no size limit) and calls brass_deopt_exit_record with its address,
// exactly as the x64 guard exit does. Spilled state values are read straight
// from their frame-pointer-relative slots, so any number of them can be
// spilled. X16 carries each value, X17 an out-of-range record address.
// Below the record, in the same allocation, is the outgoing area of the
// exit stub's stack arguments: SP moves once for both.
void AArch64EmitContext::emit_guard_exit(const LirInst& inst) {
    const size_t num_uses = inst.uses.size();
    if (inst.deopt_kinds.size() != num_uses) {
        throw_unsupported("aarch64 emit (guard exit)", "state map without per-value kinds");
    }
    // Isel names an exit symbol only for a guard with an exit stub.
    const bool has_exit_symbol = !inst.exit_symbol.empty();
    const size_t header_bytes = runtime::DeoptExitRecord::kSlotsOffset;
    const size_t slots_bytes = num_uses * 8;
    const size_t kinds_bytes = (num_uses + 7) & ~size_t(7);
    const bool apple = fn_.calling_conv.kind() == CallingConvKind::AppleAAPCS64;
    const ExitStubArgPlan plan = has_exit_symbol ? plan_exit_stub_args(inst, apple) : ExitStubArgPlan{};
    const size_t rec_off = plan.out_area;
    const size_t total_alloc = (rec_off + header_bytes + slots_bytes + kinds_bytes + 15) & ~size_t(15);
    // add/sub immediates split into two 12-bit halves reach 16 MiB without a
    // scratch register (X16 holds the value being stored).
    if (total_alloc > 0x00FF0000u) {
        throw_unsupported("aarch64 emit (guard exit)", "deopt state map too large for one frame");
    }

    // Touch every page of a large record in order before moving SP (the
    // stack is committed lazily behind a guard page), like __chkstk: each
    // probe addresses its page through X16, and SP moves once, by one
    // instruction, so the unwind info (the prologue's frame until the
    // allocation, this exit's after it) is exact at every instruction.
    constexpr size_t kPage = 4096;
    for (size_t off = kPage; off < total_alloc; off += kPage) {
        enc_.sub(GPR::X16, GPR::SP, static_cast<uint32_t>(off));
        enc_.str(GPR::XZR, ptr(GPR::X16, 0));
    }
    move_sp_for_guard_exit(true, total_alloc);

    // The record sits at SP + rec_off, the stack arguments at SP. Offsets
    // past the scaled 12-bit range go through X17.
    auto sp_mem = [&](size_t off, size_t size) -> MemAddress {
        if (off % size == 0 && off / size <= 4095) return ptr(GPR::SP, static_cast<int64_t>(off));
        enc_.add(GPR::X17, GPR::SP, static_cast<uint32_t>(off));
        return ptr(GPR::X17, 0);
    };
    auto record_mem = [&](size_t off) -> MemAddress { return sp_mem(rec_off + off, 8); };

    for (size_t i = 0; i < num_uses; ++i) {
        const auto& op = inst.uses[i];
        const size_t slot_off = header_bytes + i * 8;

        if (op.is_preg() && op.preg_val.is_gpr()) {
            GPR src_gpr = op.preg_val.as_aarch64_gpr();
            if (op.size == 4) {
                enc_.sxtw(GPR::X16, src_gpr);
                src_gpr = GPR::X16;
            }
            enc_.str(src_gpr, record_mem(slot_off));
        } else if (op.is_preg() && op.preg_val.is_xmm()) {
            enc_.str(op.preg_val.as_aarch64_fpr(), record_mem(slot_off));
        } else if (op.is_spill_slot() || op.is_mem() || op.is_local_slot()) {
            enc_.ldr(GPR::X16, ensure_accessible_mem(to_mem_address(op), GPR::X16));
            enc_.str(GPR::X16, record_mem(slot_off));
        } else if (op.is_imm_int()) {
            enc_.mov(GPR::X16, static_cast<uint64_t>(op.imm_int));
            enc_.str(GPR::X16, record_mem(slot_off));
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
            enc_.mov(GPR::X16, bits);
            enc_.str(GPR::X16, record_mem(slot_off));
        } else {
            throw_unsupported("aarch64 emit (guard exit)",
                              std::string("state value operand of kind ") + std::string(to_string(op.kind)));
        }
    }

    // Kind bytes, eight per store.
    const size_t kinds_off = header_bytes + slots_bytes;
    for (size_t i = 0; i < num_uses; i += 8) {
        uint64_t packed = 0;
        for (size_t j = 0; j < 8 && i + j < num_uses; ++j) {
            packed |= static_cast<uint64_t>(inst.deopt_kinds[i + j]) << (8 * j);
        }
        enc_.mov(GPR::X16, packed);
        enc_.str(GPR::X16, record_mem(kinds_off + i));
    }

    const uint32_t rid = inst.resume_id;
    const uint32_t rsn = inst.deopt_reason == 0 ? 1 : inst.deopt_reason;
    const uint32_t cnt = static_cast<uint32_t>(num_uses);
    const uint32_t flags = has_exit_symbol ? runtime::DeoptExitRecord::kHasExitSymbol : 0u;

    // Header: code_entry (this function's own entry, the key its resumer is
    // registered under), resume id, reason, count, flags.
    enc_.load_symbol_address(GPR::X16, fn_.name);
    enc_.str(GPR::X16, record_mem(0));
    enc_.mov(GPR::X16, (static_cast<uint64_t>(rsn) << 32) | rid);
    enc_.str(GPR::X16, record_mem(8));
    enc_.mov(GPR::X16, (static_cast<uint64_t>(flags) << 32) | cnt);
    enc_.str(GPR::X16, record_mem(16));

    if (rec_off == 0) enc_.mov(GPR::X0, GPR::SP);
    else enc_.add(GPR::X0, GPR::SP, static_cast<uint32_t>(rec_off));
    enc_.bl("brass_deopt_exit_record");

    // Handled: X0 points at the lower tier's result bits for this frame.
    Label unhandled = buffer_.create_label();
    enc_.cbz(GPR::X0, unhandled);
    move_sp_for_guard_exit(false, total_alloc);
    enc_.ldr(GPR::X0, ptr(GPR::X0, 0));
    if (fn_.return_type.kind() == TypeKind::F64) {
        enc_.fmov_from_gpr(FPR::V0, GPR::X0);
    } else if (fn_.return_type.kind() == TypeKind::F32) {
        enc_.fmov_from_gpr32(FPR::V0, GPR::X0);
    }
    AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
    buffer_.bind(unhandled);

    if (has_exit_symbol) {
        // No handler or resumer: the exit stub finishes the call, called as
        // stub(state values...) with the values read back from the record
        // (still on the stack). A slot holds a float in its low bits and a
        // narrow integer sign-extended, so a stack argument is its low
        // bytes. Its return registers are the function's.
        for (size_t i = 0; i < num_uses; ++i) {
            const ExitStubArgLoc& loc = plan.locs[i];
            if (loc.reg >= 0) continue;
            enc_.ldr(GPR::X16, record_mem(header_bytes + i * 8));
            const MemAddress dst = sp_mem(loc.stack_off, loc.size);
            switch (loc.size) {
                case 1: enc_.strb(GPR::X16, dst); break;
                case 2: enc_.strh(GPR::X16, dst); break;
                case 4: enc_.str32(GPR::X16, dst); break;
                default: enc_.str(GPR::X16, dst); break;
            }
        }
        for (size_t i = 0; i < num_uses; ++i) {
            const ExitStubArgLoc& loc = plan.locs[i];
            if (loc.reg < 0) continue;
            const MemAddress src = record_mem(header_bytes + i * 8);
            if (loc.is_float) {
                enc_.ldr(static_cast<FPR>(loc.reg), src);
            } else {
                enc_.ldr(static_cast<GPR>(loc.reg), src);
            }
        }
        enc_.bl(inst.exit_symbol);
        move_sp_for_guard_exit(false, total_alloc);
        AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
    } else {
        // brass_deopt_exit_record aborts when nothing resumes a guard
        // without an exit stub: never reached.
        enc_.brk(0);
    }
}

} // namespace brass::aarch64
