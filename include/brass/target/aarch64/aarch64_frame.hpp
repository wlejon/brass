#pragma once

#include <brass/target/aarch64/aarch64_registers.hpp>
#include <brass/target/aarch64/aarch64_operands.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/codegen/lir.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace brass::aarch64 {

class AArch64FrameLayout {
public:
    static void compute_layout(codegen::FrameInfo& frame, const CallingConvention& cc);

    static void emit_prologue(AArch64Encoder& enc, const codegen::FrameInfo& frame, const CallingConvention& cc);
    static void emit_epilogue(AArch64Encoder& enc, const codegen::FrameInfo& frame, const CallingConvention& cc);

    static MemAddress spill_slot_address(int32_t slot_idx, const codegen::FrameInfo& frame);
    static MemAddress local_frame_address(int32_t offset, const codegen::FrameInfo& frame);
    static MemAddress incoming_arg_address(int32_t caller_stack_offset, const codegen::FrameInfo& frame);
    static MemAddress callee_gpr_address(GPR reg, const codegen::FrameInfo& frame);
    static MemAddress callee_fpr_address(FPR reg, const codegen::FrameInfo& frame);

    static std::vector<GPR> get_saved_callee_gprs(const codegen::FrameInfo& frame);
    static std::vector<FPR> get_saved_callee_fprs(const codegen::FrameInfo& frame);
};

} // namespace brass::aarch64
