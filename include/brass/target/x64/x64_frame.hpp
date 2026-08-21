#pragma once

#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/codegen/lir.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace brass::x64 {

class X64FrameLayout {
public:
    static void compute_layout(codegen::FrameInfo& frame, const CallingConvention& cc);

    static void emit_prologue(X64Encoder& enc, const codegen::FrameInfo& frame, const CallingConvention& cc);
    static void emit_epilogue(X64Encoder& enc, const codegen::FrameInfo& frame, const CallingConvention& cc);

    static MemAddress spill_slot_address(int32_t slot_idx, const codegen::FrameInfo& frame);
    static MemAddress callee_gpr_address(GPR reg, const codegen::FrameInfo& frame);
    static MemAddress callee_xmm_address(XMM reg, const codegen::FrameInfo& frame);

    static std::vector<GPR> get_saved_callee_gprs(const codegen::FrameInfo& frame);
    static std::vector<XMM> get_saved_callee_xmms(const codegen::FrameInfo& frame);
};

} // namespace brass::x64
