// AArch64 guard exit: the deopt path of a failed speculation guard.
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/runtime/deopt.hpp>
#include <cstring>
#include <string>

namespace brass::aarch64 {

using namespace brass::codegen;

// Builds a runtime::DeoptExitRecord on the stack (header, 8-byte slots, kind
// bytes; no size limit) and calls brass_deopt_exit_record with its address,
// exactly as the x64 guard exit does. Spilled state values are read straight
// from their frame-pointer-relative slots, so any number of them can be
// spilled. X16 carries each value, X17 an out-of-range record address.
void AArch64EmitContext::emit_guard_exit(const LirInst& inst) {
    const size_t num_uses = inst.uses.size();
    if (inst.deopt_kinds.size() != num_uses) {
        throw_unsupported("aarch64 emit (guard exit)", "state map without per-value kinds");
    }
    const bool has_exit_symbol = !inst.exit_symbol.empty() && inst.exit_symbol != "@exit_stub" &&
                                 inst.exit_symbol != "exit_stub";
    const size_t header_bytes = runtime::DeoptExitRecord::kSlotsOffset;
    const size_t slots_bytes = num_uses * 8;
    const size_t kinds_bytes = (num_uses + 7) & ~size_t(7);
    const size_t total_alloc = (header_bytes + slots_bytes + kinds_bytes + 15) & ~size_t(15);
    // add/sub immediates split into two 12-bit halves reach 16 MiB without a
    // scratch register (X16 holds the value being stored).
    if (total_alloc > 0x00FF0000u) {
        throw_unsupported("aarch64 emit (guard exit)", "deopt state map too large for one frame");
    }

    // Allocate page by page, touching each page in order, so a large record
    // never skips a stack guard page.
    constexpr size_t kPage = 4096;
    size_t remaining = total_alloc;
    while (remaining > kPage) {
        enc_.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(kPage));
        enc_.str(GPR::XZR, ptr(GPR::SP, 0));
        remaining -= kPage;
    }
    if (remaining > 0) enc_.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(remaining));

    // The record sits at SP. Offsets past the scaled 12-bit range go through X17.
    auto record_mem = [&](size_t off) -> MemAddress {
        if (off % 8 == 0 && off / 8 <= 4095) return ptr(GPR::SP, static_cast<int64_t>(off));
        enc_.add(GPR::X17, GPR::SP, static_cast<uint32_t>(off));
        return ptr(GPR::X17, 0);
    };

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
    enc_.str(GPR::X16, ptr(GPR::SP, 0));
    enc_.mov(GPR::X16, (static_cast<uint64_t>(rsn) << 32) | rid);
    enc_.str(GPR::X16, ptr(GPR::SP, 8));
    enc_.mov(GPR::X16, (static_cast<uint64_t>(flags) << 32) | cnt);
    enc_.str(GPR::X16, ptr(GPR::SP, 16));

    enc_.mov(GPR::X0, GPR::SP);
    enc_.bl("brass_deopt_exit_record");
    enc_.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_alloc));

    auto emit_return_x0 = [&]() {
        if (fn_.return_type.kind() == TypeKind::F64) {
            enc_.fmov_from_gpr(FPR::V0, GPR::X0);
        } else if (fn_.return_type.kind() == TypeKind::F32) {
            enc_.fmov_from_gpr32(FPR::V0, GPR::X0);
        } else if (fn_.return_type.is_v128()) {
            enc_.fmov_from_gpr(FPR::V0, GPR::X0);
        }
        AArch64FrameLayout::emit_epilogue(enc_, frame_, fn_.calling_conv);
    };

    // Handled: X0 points at the lower tier's result for this frame.
    Label unhandled = buffer_.create_label();
    enc_.cbz(GPR::X0, unhandled);
    enc_.ldr(GPR::X0, ptr(GPR::X0, 0));
    emit_return_x0();
    buffer_.bind(unhandled);

    if (has_exit_symbol) {
        // Generic-twin ABI: exit(resume_id, slots).
        enc_.bl("brass_get_thread_deopt_slots");
        enc_.mov(GPR::X1, GPR::X0);
        enc_.mov32(GPR::X0, rid);
        enc_.bl(inst.exit_symbol);
    } else if (inst.deopt_reason == static_cast<uint32_t>(runtime::DeoptReason::BoundsCheckFailed)) {
        // An out-of-bounds guard nobody resumes must not return garbage.
        enc_.brk(0);
    }
    emit_return_x0();
}

} // namespace brass::aarch64
