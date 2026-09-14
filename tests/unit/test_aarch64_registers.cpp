#include "test_framework.hpp"
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <sstream>

using namespace brass::aarch64;

TEST_CASE("AArch64 Registers - Codes, IDs, Names, and Formatting") {
    CHECK_EQ(reg_code(GPR::X0), 0);
    CHECK_EQ(reg_code(GPR::X1), 1);
    CHECK_EQ(reg_code(GPR::X29), 29);
    CHECK_EQ(reg_code(GPR::X30), 30);
    CHECK_EQ(reg_code(GPR::SP), 31);
    CHECK_EQ(reg_code(GPR::XZR), 31);
    CHECK_EQ(reg_code(FP), 29);
    CHECK_EQ(reg_code(LR), 30);

    CHECK_EQ(reg_code(FPR::V0), 0);
    CHECK_EQ(reg_code(FPR::V31), 31);

    // Names for 64-bit and 32-bit
    CHECK_EQ(to_string(GPR::X0, OperandSize::Xword), "x0");
    CHECK_EQ(to_string(GPR::X0, OperandSize::Word),  "w0");
    CHECK_EQ(to_string(GPR::X29, OperandSize::Xword), "x29");
    CHECK_EQ(to_string(GPR::X30, OperandSize::Xword), "x30");
    CHECK_EQ(to_string(GPR::SP, OperandSize::Xword),  "sp");
    CHECK_EQ(to_string(GPR::SP, OperandSize::Word),   "wsp");
    CHECK_EQ(to_string(GPR::XZR, OperandSize::Xword), "xzr");
    CHECK_EQ(to_string(GPR::XZR, OperandSize::Word),  "wzr");

    // FPR views
    CHECK_EQ(to_string(FPR::V0, OperandSize::Word),  "s0");
    CHECK_EQ(to_string(FPR::V0, OperandSize::Xword), "d0");
    CHECK_EQ(to_string(FPR::V0, OperandSize::Quad),  "q0");
    CHECK_EQ(to_string(FPR::V31, OperandSize::Xword), "d31");

    // Conditions
    CHECK_EQ(to_string(Condition::EQ), "eq");
    CHECK_EQ(to_string(Condition::NE), "ne");
    CHECK_EQ(to_string(Condition::CS), "cs");
    CHECK_EQ(to_string(Condition::CC), "cc");
    CHECK_EQ(to_string(Condition::MI), "mi");
    CHECK_EQ(to_string(Condition::PL), "pl");
    CHECK_EQ(to_string(Condition::VS), "vs");
    CHECK_EQ(to_string(Condition::VC), "vc");
    CHECK_EQ(to_string(Condition::HI), "hi");
    CHECK_EQ(to_string(Condition::LS), "ls");
    CHECK_EQ(to_string(Condition::GE), "ge");
    CHECK_EQ(to_string(Condition::LT), "lt");
    CHECK_EQ(to_string(Condition::GT), "gt");
    CHECK_EQ(to_string(Condition::LE), "le");
    CHECK_EQ(to_string(Condition::AL), "al");

    // Inversions
    CHECK_EQ(invert(Condition::EQ), Condition::NE);
    CHECK_EQ(invert(Condition::NE), Condition::EQ);
    CHECK_EQ(invert(Condition::CS), Condition::CC);
    CHECK_EQ(invert(Condition::CC), Condition::CS);
    CHECK_EQ(invert(Condition::MI), Condition::PL);
    CHECK_EQ(invert(Condition::PL), Condition::MI);
    CHECK_EQ(invert(Condition::VS), Condition::VC);
    CHECK_EQ(invert(Condition::VC), Condition::VS);
    CHECK_EQ(invert(Condition::HI), Condition::LS);
    CHECK_EQ(invert(Condition::LS), Condition::HI);
    CHECK_EQ(invert(Condition::GE), Condition::LT);
    CHECK_EQ(invert(Condition::LT), Condition::GE);
    CHECK_EQ(invert(Condition::GT), Condition::LE);
    CHECK_EQ(invert(Condition::LE), Condition::GT);
    CHECK_EQ(invert(Condition::AL), Condition::AL);

    // Masks
    CHECK(mask_has(all_gprs_mask(), GPR::X0));
    CHECK(mask_has(all_gprs_mask(), GPR::X30));
    CHECK(!mask_has(0, GPR::X0));
    CHECK(mask_has(reg_mask(GPR::X19), GPR::X19));
    CHECK(!mask_has(reg_mask(GPR::X19), GPR::X20));

    // Stream operator
    std::ostringstream ss;
    ss << GPR::X19 << " " << FPR::V8 << " " << Condition::GE << " " << OperandSize::Xword;
    CHECK_EQ(ss.str(), "x19 d8 ge xword");
}
