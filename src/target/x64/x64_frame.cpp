#include <brass/target/x64/x64_frame.hpp>
#include <algorithm>

namespace brass::x64 {

std::vector<GPR> X64FrameLayout::get_saved_callee_gprs(const codegen::FrameInfo& frame) {
    std::vector<GPR> result;
    for (int i = 0; i < 16; ++i) {
        if (frame.saved_callee_gprs & (1u << i)) {
            GPR reg = static_cast<GPR>(i);
            if (reg != GPR::RSP && reg != GPR::RBP) {
                result.push_back(reg);
            }
        }
    }
    return result;
}

std::vector<XMM> X64FrameLayout::get_saved_callee_xmms(const codegen::FrameInfo& frame) {
    std::vector<XMM> result;
    for (int i = 0; i < 16; ++i) {
        if (frame.saved_callee_xmms & (1u << i)) {
            result.push_back(static_cast<XMM>(i));
        }
    }
    return result;
}

void X64FrameLayout::compute_layout(codegen::FrameInfo& frame, const CallingConvention& cc) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_xmms = get_saved_callee_xmms(frame);

    size_t gpr_bytes = saved_gprs.size() * 8;
    size_t xmm_bytes = saved_xmms.size() * 16;
    size_t spill_bytes = frame.num_spill_slots * 8;
    size_t outgoing_bytes = frame.outgoing_arg_space;

    // Minimum shadow space for Win64 if outgoing calls exist or if outgoing_bytes > 0
    if (cc.kind() == CallingConvKind::Win64 && (outgoing_bytes > 0 || frame.has_calls)) {
        outgoing_bytes = std::max(outgoing_bytes, size_t(32));
    }

    size_t raw_total = gpr_bytes + xmm_bytes + spill_bytes + outgoing_bytes;
    // Align total frame size to 16 bytes
    frame.total_frame_size = (raw_total + 15) & ~size_t(15);
    frame.is_leaf = (!frame.has_calls && frame.total_frame_size == 0 && saved_gprs.empty() && saved_xmms.empty() && outgoing_bytes == 0);
}

MemAddress X64FrameLayout::callee_gpr_address(GPR reg, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    for (size_t i = 0; i < saved_gprs.size(); ++i) {
        if (saved_gprs[i] == reg) {
            int32_t disp = -static_cast<int32_t>((i + 1) * 8);
            return ptr(GPR::RBP, disp);
        }
    }
    return ptr(GPR::RBP, -8);
}

MemAddress X64FrameLayout::callee_xmm_address(XMM reg, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_xmms = get_saved_callee_xmms(frame);
    size_t gpr_offset = saved_gprs.size() * 8;

    for (size_t i = 0; i < saved_xmms.size(); ++i) {
        if (saved_xmms[i] == reg) {
            int32_t disp = -static_cast<int32_t>(gpr_offset + (i + 1) * 16);
            return ptr(GPR::RBP, disp);
        }
    }
    return ptr(GPR::RBP, -16);
}

MemAddress X64FrameLayout::spill_slot_address(int32_t slot_idx, const codegen::FrameInfo& frame) {
    auto saved_gprs = get_saved_callee_gprs(frame);
    auto saved_xmms = get_saved_callee_xmms(frame);
    size_t gpr_offset = saved_gprs.size() * 8;
    size_t xmm_offset = saved_xmms.size() * 16;
    int32_t base_offset = static_cast<int32_t>(gpr_offset + xmm_offset);

    int32_t disp = -(base_offset + (slot_idx + 1) * 8);
    return ptr(GPR::RBP, disp);
}

void X64FrameLayout::emit_prologue(
    X64Encoder& enc,
    const codegen::FrameInfo& frame,
    const CallingConvention& cc
) {
    (void)cc;
    if (frame.is_leaf) {
        return;
    }

    // 1. push rbp
    enc.push(GPR::RBP);

    // 2. mov rbp, rsp
    enc.mov(GPR::RBP, GPR::RSP);

    // 3. sub rsp, total_frame_size (if > 0)
    if (frame.total_frame_size > 0) {
        enc.sub(GPR::RSP, static_cast<int32_t>(frame.total_frame_size));
    }

    // 4. Save callee-saved GPRs to [rbp - ...]
    auto saved_gprs = get_saved_callee_gprs(frame);
    for (GPR g : saved_gprs) {
        enc.mov(callee_gpr_address(g, frame), g);
    }

    // 5. Save callee-saved XMMs to [rbp - ...] (Win64)
    auto saved_xmms = get_saved_callee_xmms(frame);
    for (XMM x : saved_xmms) {
        enc.movsd(callee_xmm_address(x, frame), x);
    }
}

void X64FrameLayout::emit_epilogue(
    X64Encoder& enc,
    const codegen::FrameInfo& frame,
    const CallingConvention& cc
) {
    (void)cc;
    if (frame.is_leaf) {
        enc.ret();
        return;
    }

    // 1. Restore callee-saved XMMs
    auto saved_xmms = get_saved_callee_xmms(frame);
    for (XMM x : saved_xmms) {
        enc.movsd(x, callee_xmm_address(x, frame));
    }

    // 2. Restore callee-saved GPRs
    auto saved_gprs = get_saved_callee_gprs(frame);
    for (GPR g : saved_gprs) {
        enc.mov(g, callee_gpr_address(g, frame));
    }

    // 3. mov rsp, rbp
    enc.mov(GPR::RSP, GPR::RBP);

    // 4. pop rbp
    enc.pop(GPR::RBP);

    // 5. ret
    enc.ret();
}

} // namespace brass::x64
