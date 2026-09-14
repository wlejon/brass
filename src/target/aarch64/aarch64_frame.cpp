#include <brass/target/aarch64/aarch64_frame.hpp>
#include <algorithm>

namespace brass::aarch64 {

std::vector<GPR> AArch64FrameLayout::get_saved_callee_gprs(const codegen::FrameInfo& frame) {
    std::vector<GPR> result;
    // Callee-saved in AAPCS64: X19..X28
    for (int i = 19; i <= 28; ++i) {
        if (frame.saved_callee_gprs & (1u << i)) {
            result.push_back(static_cast<GPR>(i));
        }
    }
    return result;
}

std::vector<FPR> AArch64FrameLayout::get_saved_callee_fprs(const codegen::FrameInfo& frame) {
    std::vector<FPR> result;
    // Callee-saved in AAPCS64: V8..V15 (bottom 64 bits D8..D15)
    for (int i = 8; i <= 15; ++i) {
        if (frame.saved_callee_xmms & (1u << i)) {
            result.push_back(static_cast<FPR>(i));
        }
    }
    return result;
}

void AArch64FrameLayout::compute_layout(codegen::FrameInfo& frame, const CallingConvention& cc) {
    (void)cc;
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_fprs = get_saved_callee_fprs(frame);

    size_t gpr_bytes = saved_gprs.size() * 8;
    size_t fpr_bytes = saved_fprs.size() * 8;
    size_t spill_bytes = frame.num_spill_slots * 8;
    size_t outgoing_bytes = frame.outgoing_arg_space;
    size_t header_bytes = 16; // FP (X29) + LR (X30)

    size_t raw_total = header_bytes + gpr_bytes + fpr_bytes + spill_bytes + outgoing_bytes;
    // Align total frame size to 16 bytes
    frame.total_frame_size = (raw_total + 15) & ~size_t(15);
    frame.is_leaf = (!frame.has_calls && frame.num_spill_slots == 0 && saved_gprs.empty() && saved_fprs.empty() && outgoing_bytes == 0);
}

MemAddress AArch64FrameLayout::callee_gpr_address(GPR reg, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    for (size_t i = 0; i < saved_gprs.size(); ++i) {
        if (saved_gprs[i] == reg) {
            int64_t disp = 16 + static_cast<int64_t>(i * 8);
            return ptr(GPR::FP, disp);
        }
    }
    return ptr(GPR::FP, 16);
}

MemAddress AArch64FrameLayout::callee_fpr_address(FPR reg, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_fprs = get_saved_callee_fprs(frame);
    size_t gpr_offset = 16 + saved_gprs.size() * 8;

    for (size_t i = 0; i < saved_fprs.size(); ++i) {
        if (saved_fprs[i] == reg) {
            int64_t disp = static_cast<int64_t>(gpr_offset + i * 8);
            return ptr(GPR::FP, disp);
        }
    }
    return ptr(GPR::FP, static_cast<int64_t>(gpr_offset));
}

MemAddress AArch64FrameLayout::spill_slot_address(int32_t slot_idx, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_fprs = get_saved_callee_fprs(frame);
    size_t base_offset = 16 + saved_gprs.size() * 8 + saved_fprs.size() * 8;

    int64_t disp = static_cast<int64_t>(base_offset + static_cast<size_t>(slot_idx) * 8);
    return ptr(GPR::FP, disp);
}

void AArch64FrameLayout::emit_prologue(
    AArch64Encoder& enc,
    const codegen::FrameInfo& frame,
    const CallingConvention& cc
) {
    (void)cc;
    if (frame.is_leaf) {
        return;
    }

    size_t total_size = frame.total_frame_size;
    if (total_size <= 512) {
        // stp fp, lr, [sp, #-total_size]!
        enc.stp(GPR::FP, GPR::LR, pre_idx(GPR::SP, -static_cast<int64_t>(total_size)));
    } else {
        enc.sub(GPR::SP, GPR::SP, static_cast<uint32_t>(total_size));
        enc.stp(GPR::FP, GPR::LR, ptr(GPR::SP, 0));
    }

    // mov fp, sp
    enc.mov(GPR::FP, GPR::SP);

    // Save callee-saved GPRs (paired where possible)
    auto saved_gprs = get_saved_callee_gprs(frame);
    size_t i = 0;
    while (i + 1 < saved_gprs.size()) {
        enc.stp(saved_gprs[i], saved_gprs[i + 1], callee_gpr_address(saved_gprs[i], frame));
        i += 2;
    }
    if (i < saved_gprs.size()) {
        enc.str(saved_gprs[i], callee_gpr_address(saved_gprs[i], frame));
    }

    // Save callee-saved FPRs (paired where possible)
    auto saved_fprs = get_saved_callee_fprs(frame);
    size_t j = 0;
    while (j + 1 < saved_fprs.size()) {
        enc.stp(saved_fprs[j], saved_fprs[j + 1], callee_fpr_address(saved_fprs[j], frame));
        j += 2;
    }
    if (j < saved_fprs.size()) {
        enc.str(saved_fprs[j], callee_fpr_address(saved_fprs[j], frame));
    }
}

void AArch64FrameLayout::emit_epilogue(
    AArch64Encoder& enc,
    const codegen::FrameInfo& frame,
    const CallingConvention& cc
) {
    (void)cc;
    if (frame.is_leaf) {
        enc.ret();
        return;
    }

    // Restore callee-saved FPRs
    auto saved_fprs = get_saved_callee_fprs(frame);
    size_t j = 0;
    while (j + 1 < saved_fprs.size()) {
        enc.ldp(saved_fprs[j], saved_fprs[j + 1], callee_fpr_address(saved_fprs[j], frame));
        j += 2;
    }
    if (j < saved_fprs.size()) {
        enc.ldr(saved_fprs[j], callee_fpr_address(saved_fprs[j], frame));
    }

    // Restore callee-saved GPRs
    auto saved_gprs = get_saved_callee_gprs(frame);
    size_t i = 0;
    while (i + 1 < saved_gprs.size()) {
        enc.ldp(saved_gprs[i], saved_gprs[i + 1], callee_gpr_address(saved_gprs[i], frame));
        i += 2;
    }
    if (i < saved_gprs.size()) {
        enc.ldr(saved_gprs[i], callee_gpr_address(saved_gprs[i], frame));
    }

    size_t total_size = frame.total_frame_size;
    if (total_size <= 512) {
        // ldp fp, lr, [sp], #total_size
        enc.ldp(GPR::FP, GPR::LR, post_idx(GPR::SP, static_cast<int64_t>(total_size)));
    } else {
        enc.ldp(GPR::FP, GPR::LR, ptr(GPR::SP, 0));
        enc.add(GPR::SP, GPR::SP, static_cast<uint32_t>(total_size));
    }

    // ret
    enc.ret();
}

} // namespace brass::aarch64
