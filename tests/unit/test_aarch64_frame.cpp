#include "test_framework.hpp"
#include <brass/target/aarch64/aarch64_frame.hpp>

using namespace brass;
using namespace brass::aarch64;

TEST_CASE("AArch64 Frame Layout - Leaf and Non-Leaf Computations") {
    CallingConvention cc = CallingConvention::aapcs64();

    // 1. Leaf function
    codegen::FrameInfo leaf_frame;
    leaf_frame.has_calls = false;
    leaf_frame.num_spill_slots = 0;
    leaf_frame.saved_callee_gprs = 0;
    leaf_frame.saved_callee_xmms = 0;
    AArch64FrameLayout::compute_layout(leaf_frame, cc);

    CHECK(leaf_frame.is_leaf);

    CodeBuffer leaf_buf;
    AArch64Encoder leaf_enc(leaf_buf);
    AArch64FrameLayout::emit_prologue(leaf_enc, leaf_frame, cc);
    CHECK_EQ(leaf_buf.size(), 0); // Leaf emits no prologue
    AArch64FrameLayout::emit_epilogue(leaf_enc, leaf_frame, cc);
    CHECK_EQ(leaf_buf.size(), 4); // Only ret instruction

    // 2. Non-leaf function with calls and 2 spill slots
    codegen::FrameInfo nonleaf_frame;
    nonleaf_frame.has_calls = true;
    nonleaf_frame.num_spill_slots = 2;
    // Callee saved X19, X20
    nonleaf_frame.saved_callee_gprs = (1u << 19) | (1u << 20);
    AArch64FrameLayout::compute_layout(nonleaf_frame, cc);

    CHECK(!nonleaf_frame.is_leaf);
    CHECK(nonleaf_frame.total_frame_size >= 48);
    CHECK_EQ(nonleaf_frame.total_frame_size % 16, 0);

    // Slot addresses relative to FP (X29)
    auto slot0 = AArch64FrameLayout::spill_slot_address(0, nonleaf_frame);
    auto slot1 = AArch64FrameLayout::spill_slot_address(1, nonleaf_frame);
    CHECK_EQ(slot0.base, GPR::FP);
    CHECK_EQ(slot1.base, GPR::FP);
    CHECK_EQ(slot1.offset - slot0.offset, 8);

    // Callee saved GPR address
    auto gpr19 = AArch64FrameLayout::callee_gpr_address(GPR::X19, nonleaf_frame);
    auto gpr20 = AArch64FrameLayout::callee_gpr_address(GPR::X20, nonleaf_frame);
    CHECK_EQ(gpr19.base, GPR::FP);
    CHECK_EQ(gpr20.base, GPR::FP);
    CHECK_EQ(gpr20.offset - gpr19.offset, 8);

    CodeBuffer nl_buf;
    AArch64Encoder nl_enc(nl_buf);
    AArch64FrameLayout::emit_prologue(nl_enc, nonleaf_frame, cc);
    CHECK(nl_buf.size() > 0);

    size_t prol_sz = nl_buf.size();
    AArch64FrameLayout::emit_epilogue(nl_enc, nonleaf_frame, cc);
    CHECK(nl_buf.size() > prol_sz);
}
