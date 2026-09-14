#include "test_framework.hpp"
#include <brass/target/aarch64/aarch64_operands.hpp>
#include <sstream>

using namespace brass::aarch64;

TEST_CASE("AArch64 Operands - Addressing Modes and Formatting") {
    // Base only
    auto m1 = ptr(GPR::X0);
    CHECK(m1.has_base());
    CHECK(!m1.has_index());
    CHECK_EQ(m1.offset, 0);
    CHECK_EQ(m1.mode, AddrMode::Offset);
    CHECK_EQ(to_string(m1), "[x0]");

    // Base + disp
    auto m2 = ptr(GPR::SP, 16);
    CHECK_EQ(m2.offset, 16);
    CHECK_EQ(to_string(m2), "[sp, #16]");

    // Pre-indexed
    auto m3 = pre_idx(GPR::SP, -32);
    CHECK(m3.is_pre_indexed());
    CHECK_EQ(to_string(m3), "[sp, #-32]!");

    // Post-indexed
    auto m4 = post_idx(GPR::SP, 32);
    CHECK(m4.is_post_indexed());
    CHECK_EQ(to_string(m4), "[sp], #32");

    // Register offset
    auto m5 = MemAddress::base_index(GPR::X0, GPR::X1, ExtendType::UXTX, 3);
    CHECK(m5.has_base());
    CHECK(m5.has_index());
    CHECK_EQ(m5.mode, AddrMode::RegOffset);
    CHECK_EQ(to_string(m5), "[x0, x1, uxtx #3]");

    // Literal
    auto m6 = MemAddress::literal(100);
    CHECK(m6.is_literal());
    CHECK_EQ(to_string(m6), "[pc, #100]");

    // Shifts & Extends
    CHECK_EQ(to_string(ShiftType::LSL), "lsl");
    CHECK_EQ(to_string(ShiftType::LSR), "lsr");
    CHECK_EQ(to_string(ExtendType::SXTW), "sxtw");
    CHECK_EQ(to_string(ExtendType::UXTW), "uxtw");

    std::ostringstream ss;
    ss << m2;
    CHECK_EQ(ss.str(), "[sp, #16]");
}
