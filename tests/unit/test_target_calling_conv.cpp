#include "test_framework.hpp"
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_registers.hpp>

using namespace brass;
using namespace brass::x64;

TEST_CASE("Target Abstraction - Predefined targets and properties") {
    auto win = Target::x64_windows();
    CHECK(win.is_x64());
    CHECK(win.is_windows());
    CHECK(!win.is_linux());
    CHECK(!win.is_macos());
    CHECK(win.is_64bit());
    CHECK_EQ(win.arch(), Arch::x64);
    CHECK_EQ(win.os(), OperatingSystem::Windows);
    CHECK_EQ(win.object_format(), ObjectFormat::COFF);
    CHECK_EQ(win.pointer_size(), 8);
    CHECK_EQ(win.stack_alignment(), 16);

    auto linux_target = Target::x64_linux();
    CHECK(linux_target.is_x64());
    CHECK(linux_target.is_linux());
    CHECK(!linux_target.is_windows());
    CHECK_EQ(linux_target.object_format(), ObjectFormat::ELF64);

    auto macos_target = Target::x64_macos();
    CHECK(macos_target.is_x64());
    CHECK(macos_target.is_macos());
    CHECK_EQ(macos_target.object_format(), ObjectFormat::MachO);

    auto host = Target::host();
    CHECK(host.is_x64());
    CHECK(host.is_64bit());

    CHECK_EQ(to_string(Arch::x64), "x64");
    CHECK_EQ(to_string(Arch::aarch64), "aarch64");
    CHECK_EQ(to_string(OperatingSystem::Windows), "Windows");
    CHECK_EQ(to_string(OperatingSystem::Linux), "Linux");
    CHECK_EQ(to_string(OperatingSystem::macOS), "macOS");
    CHECK_EQ(to_string(ObjectFormat::COFF), "COFF");
    CHECK_EQ(to_string(ObjectFormat::ELF64), "ELF64");
    CHECK_EQ(to_string(ObjectFormat::MachO), "MachO");
}

TEST_CASE("x64 Registers - Enums, Names, Masks, Invert Condition") {
    CHECK_EQ(reg_code(GPR::RAX), 0);
    CHECK_EQ(reg_code(GPR::RCX), 1);
    CHECK_EQ(reg_code(GPR::RDX), 2);
    CHECK_EQ(reg_code(GPR::RBX), 3);
    CHECK_EQ(reg_code(GPR::RSP), 4);
    CHECK_EQ(reg_code(GPR::RBP), 5);
    CHECK_EQ(reg_code(GPR::RSI), 6);
    CHECK_EQ(reg_code(GPR::RDI), 7);
    CHECK_EQ(reg_code(GPR::R8),  0);
    CHECK_EQ(reg_code(GPR::R15), 7);

    CHECK(!is_extended(GPR::RAX));
    CHECK(!is_extended(GPR::RDI));
    CHECK(is_extended(GPR::R8));
    CHECK(is_extended(GPR::R15));

    CHECK(!is_extended(XMM::XMM0));
    CHECK(!is_extended(XMM::XMM7));
    CHECK(is_extended(XMM::XMM8));
    CHECK(is_extended(XMM::XMM15));

    CHECK_EQ(to_string(GPR::RAX, OperandSize::Qword), "rax");
    CHECK_EQ(to_string(GPR::RAX, OperandSize::Dword), "eax");
    CHECK_EQ(to_string(GPR::RAX, OperandSize::Word),  "ax");
    CHECK_EQ(to_string(GPR::RAX, OperandSize::Byte),  "al");

    CHECK_EQ(to_string(GPR::R8, OperandSize::Qword), "r8");
    CHECK_EQ(to_string(GPR::R8, OperandSize::Dword), "r8d");
    CHECK_EQ(to_string(GPR::R8, OperandSize::Word),  "r8w");
    CHECK_EQ(to_string(GPR::R8, OperandSize::Byte),  "r8b");

    CHECK_EQ(to_string(XMM::XMM0), "xmm0");
    CHECK_EQ(to_string(XMM::XMM15), "xmm15");

    // Condition codes & invert
    CHECK_EQ(invert(Condition::E), Condition::NE);
    CHECK_EQ(invert(Condition::NE), Condition::E);
    CHECK_EQ(invert(Condition::L), Condition::GE);
    CHECK_EQ(invert(Condition::GE), Condition::L);
    CHECK_EQ(invert(Condition::LE), Condition::G);
    CHECK_EQ(invert(Condition::G), Condition::LE);
    CHECK_EQ(invert(Condition::B), Condition::AE);
    CHECK_EQ(invert(Condition::AE), Condition::B);
    CHECK_EQ(invert(Condition::BE), Condition::A);
    CHECK_EQ(invert(Condition::A), Condition::BE);
    CHECK_EQ(invert(Condition::O), Condition::NO);
    CHECK_EQ(invert(Condition::NO), Condition::O);
    CHECK_EQ(invert(Condition::S), Condition::NS);
    CHECK_EQ(invert(Condition::NS), Condition::S);
    CHECK_EQ(invert(Condition::P), Condition::NP);
    CHECK_EQ(invert(Condition::NP), Condition::P);

    CHECK_EQ(to_string(Condition::E), "e");
    CHECK_EQ(to_string(Condition::NE), "ne");
    CHECK_EQ(to_string(Condition::L), "l");
    CHECK_EQ(to_string(Condition::LE), "le");
    CHECK_EQ(to_string(Condition::G), "g");
    CHECK_EQ(to_string(Condition::GE), "ge");

    // Bitmasks
    RegMask m = reg_mask(GPR::RAX) | reg_mask(GPR::RCX) | reg_mask(GPR::R12);
    CHECK(mask_has(m, GPR::RAX));
    CHECK(mask_has(m, GPR::RCX));
    CHECK(!mask_has(m, GPR::RDX));
    CHECK(mask_has(m, GPR::R12));
    CHECK(!mask_has(m, GPR::R13));
}

TEST_CASE("Calling Convention - Win64 ABI") {
    auto cc = CallingConvention::win64();
    CHECK_EQ(cc.kind(), CallingConvKind::Win64);
    CHECK_EQ(cc.shadow_space(), size_t(32));

    // GPR arguments
    REQUIRE_EQ(cc.num_arg_gprs(), size_t(4));
    CHECK_EQ(cc.arg_gpr(0), GPR::RCX);
    CHECK_EQ(cc.arg_gpr(1), GPR::RDX);
    CHECK_EQ(cc.arg_gpr(2), GPR::R8);
    CHECK_EQ(cc.arg_gpr(3), GPR::R9);
    CHECK_EQ(cc.arg_gpr(4), GPR::None);

    CHECK(cc.is_arg_gpr(GPR::RCX));
    CHECK(cc.is_arg_gpr(GPR::RDX));
    CHECK(cc.is_arg_gpr(GPR::R8));
    CHECK(cc.is_arg_gpr(GPR::R9));
    CHECK(!cc.is_arg_gpr(GPR::RAX));
    CHECK(!cc.is_arg_gpr(GPR::RDI));
    CHECK(!cc.is_arg_gpr(GPR::RSI));

    // XMM arguments
    REQUIRE_EQ(cc.num_arg_xmms(), size_t(4));
    CHECK_EQ(cc.arg_xmm(0), XMM::XMM0);
    CHECK_EQ(cc.arg_xmm(1), XMM::XMM1);
    CHECK_EQ(cc.arg_xmm(2), XMM::XMM2);
    CHECK_EQ(cc.arg_xmm(3), XMM::XMM3);

    // Returns
    CHECK_EQ(cc.ret_gpr(0), GPR::RAX);
    CHECK_EQ(cc.ret_gpr(1), GPR::RDX);
    CHECK_EQ(cc.ret_xmm(0), XMM::XMM0);

    // Callee-saved GPRs: RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15
    CHECK(cc.is_callee_saved(GPR::RBX));
    CHECK(cc.is_callee_saved(GPR::RBP));
    CHECK(cc.is_callee_saved(GPR::RDI));
    CHECK(cc.is_callee_saved(GPR::RSI));
    CHECK(cc.is_callee_saved(GPR::RSP));
    CHECK(cc.is_callee_saved(GPR::R12));
    CHECK(cc.is_callee_saved(GPR::R13));
    CHECK(cc.is_callee_saved(GPR::R14));
    CHECK(cc.is_callee_saved(GPR::R15));

    // Caller-saved GPRs: RAX, RCX, RDX, R8, R9, R10, R11
    CHECK(cc.is_caller_saved(GPR::RAX));
    CHECK(cc.is_caller_saved(GPR::RCX));
    CHECK(cc.is_caller_saved(GPR::RDX));
    CHECK(cc.is_caller_saved(GPR::R8));
    CHECK(cc.is_caller_saved(GPR::R9));
    CHECK(cc.is_caller_saved(GPR::R10));
    CHECK(cc.is_caller_saved(GPR::R11));
    CHECK(!cc.is_caller_saved(GPR::RBX));

    // Win64 XMM callee-saved (XMM6..XMM15)
    CHECK(!cc.is_callee_saved(XMM::XMM0));
    CHECK(!cc.is_callee_saved(XMM::XMM5));
    CHECK(cc.is_callee_saved(XMM::XMM6));
    CHECK(cc.is_callee_saved(XMM::XMM15));

    // Win64 XMM caller-saved (XMM0..XMM5)
    CHECK(cc.is_caller_saved(XMM::XMM0));
    CHECK(cc.is_caller_saved(XMM::XMM5));
    CHECK(!cc.is_caller_saved(XMM::XMM6));
}

TEST_CASE("Calling Convention - SysV64 ABI") {
    auto cc = CallingConvention::sysv64();
    CHECK_EQ(cc.kind(), CallingConvKind::SysV64);
    CHECK_EQ(cc.shadow_space(), size_t(0));

    // GPR arguments: RDI, RSI, RDX, RCX, R8, R9
    REQUIRE_EQ(cc.num_arg_gprs(), size_t(6));
    CHECK_EQ(cc.arg_gpr(0), GPR::RDI);
    CHECK_EQ(cc.arg_gpr(1), GPR::RSI);
    CHECK_EQ(cc.arg_gpr(2), GPR::RDX);
    CHECK_EQ(cc.arg_gpr(3), GPR::RCX);
    CHECK_EQ(cc.arg_gpr(4), GPR::R8);
    CHECK_EQ(cc.arg_gpr(5), GPR::R9);

    // XMM arguments: XMM0..XMM7
    REQUIRE_EQ(cc.num_arg_xmms(), size_t(8));
    CHECK_EQ(cc.arg_xmm(0), XMM::XMM0);
    CHECK_EQ(cc.arg_xmm(7), XMM::XMM7);

    // Returns
    CHECK_EQ(cc.ret_gpr(0), GPR::RAX);
    CHECK_EQ(cc.ret_gpr(1), GPR::RDX);
    CHECK_EQ(cc.ret_xmm(0), XMM::XMM0);
    CHECK_EQ(cc.ret_xmm(1), XMM::XMM1);

    // Callee-saved: RBX, RSP, RBP, R12..R15
    CHECK(cc.is_callee_saved(GPR::RBX));
    CHECK(cc.is_callee_saved(GPR::RSP));
    CHECK(cc.is_callee_saved(GPR::RBP));
    CHECK(cc.is_callee_saved(GPR::R12));
    CHECK(cc.is_callee_saved(GPR::R13));
    CHECK(cc.is_callee_saved(GPR::R14));
    CHECK(cc.is_callee_saved(GPR::R15));
    CHECK(!cc.is_callee_saved(GPR::RDI));
    CHECK(!cc.is_callee_saved(GPR::RSI));

    // Caller-saved: RAX, RCX, RDX, RSI, RDI, R8..R11
    CHECK(cc.is_caller_saved(GPR::RAX));
    CHECK(cc.is_caller_saved(GPR::RDI));
    CHECK(cc.is_caller_saved(GPR::RSI));
    CHECK(cc.is_caller_saved(GPR::RCX));
    CHECK(cc.is_caller_saved(GPR::RDX));
    CHECK(cc.is_caller_saved(GPR::R8));
    CHECK(cc.is_caller_saved(GPR::R9));
    CHECK(cc.is_caller_saved(GPR::R10));
    CHECK(cc.is_caller_saved(GPR::R11));

    // SysV all XMMs volatile (caller-saved)
    CHECK(cc.is_caller_saved(XMM::XMM0));
    CHECK(cc.is_caller_saved(XMM::XMM6));
    CHECK(cc.is_caller_saved(XMM::XMM15));
    CHECK(!cc.is_callee_saved(XMM::XMM6));
}

TEST_CASE("Calling Convention - Custom Convention Configuration") {
    CustomCallingConvConfig cfg;
    cfg.arg_gprs = { GPR::RAX, GPR::RBX, GPR::R10 };
    cfg.arg_xmms = { XMM::XMM0, XMM::XMM1 };
    cfg.ret_gprs = { GPR::RAX };
    cfg.ret_xmms = { XMM::XMM0 };
    cfg.callee_saved_gprs = reg_mask(GPR::R12) | reg_mask(GPR::R13) | reg_mask(GPR::R14) | reg_mask(GPR::R15);
    cfg.shadow_space = 16;
    // bronze GC reference registers preserved across calls
    cfg.gcref_preserved_gprs = reg_mask(GPR::R13) | reg_mask(GPR::R14);

    auto custom = CallingConvention::custom(cfg);
    CHECK_EQ(custom.kind(), CallingConvKind::Custom);
    CHECK_EQ(custom.shadow_space(), size_t(16));
    CHECK_EQ(custom.num_arg_gprs(), size_t(3));
    CHECK_EQ(custom.arg_gpr(0), GPR::RAX);
    CHECK_EQ(custom.arg_gpr(1), GPR::RBX);
    CHECK_EQ(custom.arg_gpr(2), GPR::R10);

    CHECK(custom.is_callee_saved(GPR::R12));
    CHECK(custom.is_callee_saved(GPR::R13));
    CHECK(custom.is_gcref_preserved(GPR::R13));
    CHECK(custom.is_gcref_preserved(GPR::R14));
    CHECK(!custom.is_gcref_preserved(GPR::R12));
    CHECK(!custom.is_gcref_preserved(GPR::RAX));
}
