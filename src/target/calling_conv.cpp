#include <brass/target/calling_conv.hpp>
#include <algorithm>
#include <ostream>

namespace brass {

bool CallingConvention::is_arg_gpr(x64::GPR reg) const noexcept {
    return std::find(arg_gprs_.begin(), arg_gprs_.end(), reg) != arg_gprs_.end();
}

bool CallingConvention::is_arg_xmm(x64::XMM reg) const noexcept {
    return std::find(arg_xmms_.begin(), arg_xmms_.end(), reg) != arg_xmms_.end();
}

bool CallingConvention::is_ret_gpr(x64::GPR reg) const noexcept {
    return std::find(ret_gprs_.begin(), ret_gprs_.end(), reg) != ret_gprs_.end();
}

bool CallingConvention::is_ret_xmm(x64::XMM reg) const noexcept {
    return std::find(ret_xmms_.begin(), ret_xmms_.end(), reg) != ret_xmms_.end();
}

CallingConvention CallingConvention::win64() {
    using namespace brass::x64;
    CallingConvention cc;
    cc.kind_ = CallingConvKind::Win64;
    cc.shadow_space_ = 32;

    cc.arg_gprs_ = { GPR::RCX, GPR::RDX, GPR::R8, GPR::R9 };
    cc.arg_xmms_ = { XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3 };

    cc.ret_gprs_ = { GPR::RAX, GPR::RDX };
    cc.ret_xmms_ = { XMM::XMM0 };

    cc.callee_saved_gprs_ = reg_mask(GPR::RBX) | reg_mask(GPR::RBP) | reg_mask(GPR::RDI) |
                            reg_mask(GPR::RSI) | reg_mask(GPR::RSP) | reg_mask(GPR::R12) |
                            reg_mask(GPR::R13) | reg_mask(GPR::R14) | reg_mask(GPR::R15);

    // In Win64 ABI, XMM6 through XMM15 are non-volatile (callee-saved)
    cc.callee_saved_xmms_ = 0;
    for (int i = 6; i <= 15; ++i) {
        cc.callee_saved_xmms_ |= static_cast<RegMask>(1u << i);
    }

    cc.caller_saved_gprs_ = reg_mask(GPR::RAX) | reg_mask(GPR::RCX) | reg_mask(GPR::RDX) |
                            reg_mask(GPR::R8)  | reg_mask(GPR::R9)  | reg_mask(GPR::R10) |
                            reg_mask(GPR::R11);

    cc.caller_saved_xmms_ = 0;
    for (int i = 0; i <= 5; ++i) {
        cc.caller_saved_xmms_ |= static_cast<RegMask>(1u << i);
    }

    cc.gcref_preserved_gprs_ = 0;
    return cc;
}

CallingConvention CallingConvention::sysv64() {
    using namespace brass::x64;
    CallingConvention cc;
    cc.kind_ = CallingConvKind::SysV64;
    cc.shadow_space_ = 0;

    cc.arg_gprs_ = { GPR::RDI, GPR::RSI, GPR::RDX, GPR::RCX, GPR::R8, GPR::R9 };
    cc.arg_xmms_ = { XMM::XMM0, XMM::XMM1, XMM::XMM2, XMM::XMM3,
                     XMM::XMM4, XMM::XMM5, XMM::XMM6, XMM::XMM7 };

    cc.ret_gprs_ = { GPR::RAX, GPR::RDX };
    cc.ret_xmms_ = { XMM::XMM0, XMM::XMM1 };

    cc.callee_saved_gprs_ = reg_mask(GPR::RBX) | reg_mask(GPR::RSP) | reg_mask(GPR::RBP) |
                            reg_mask(GPR::R12) | reg_mask(GPR::R13) | reg_mask(GPR::R14) |
                            reg_mask(GPR::R15);

    cc.callee_saved_xmms_ = 0;

    cc.caller_saved_gprs_ = reg_mask(GPR::RAX) | reg_mask(GPR::RCX) | reg_mask(GPR::RDX) |
                            reg_mask(GPR::RSI) | reg_mask(GPR::RDI) | reg_mask(GPR::R8)  |
                            reg_mask(GPR::R9)  | reg_mask(GPR::R10) | reg_mask(GPR::R11);

    cc.caller_saved_xmms_ = 0xFFFF; // XMM0..XMM15 all volatile

    cc.gcref_preserved_gprs_ = 0;
    return cc;
}

CallingConvention CallingConvention::for_target(const Target& target) {
    if (target.is_windows()) {
        return win64();
    }
    return sysv64();
}

CallingConvention CallingConvention::custom(const CustomCallingConvConfig& config) {
    CallingConvention cc;
    cc.kind_ = CallingConvKind::Custom;
    cc.arg_gprs_ = config.arg_gprs;
    cc.arg_xmms_ = config.arg_xmms;
    cc.ret_gprs_ = config.ret_gprs;
    cc.ret_xmms_ = config.ret_xmms;
    cc.callee_saved_gprs_ = config.callee_saved_gprs;
    cc.callee_saved_xmms_ = config.callee_saved_xmms;
    cc.shadow_space_ = config.shadow_space;
    cc.gcref_preserved_gprs_ = config.gcref_preserved_gprs;

    if (config.caller_saved_gprs != 0) {
        cc.caller_saved_gprs_ = config.caller_saved_gprs;
    } else {
        cc.caller_saved_gprs_ = static_cast<x64::RegMask>(~config.callee_saved_gprs & 0xFFFF);
    }

    if (config.caller_saved_xmms != 0) {
        cc.caller_saved_xmms_ = config.caller_saved_xmms;
    } else {
        cc.caller_saved_xmms_ = static_cast<x64::RegMask>(~config.callee_saved_xmms & 0xFFFF);
    }

    return cc;
}

std::string_view to_string(CallingConvKind kind) noexcept {
    switch (kind) {
    case CallingConvKind::Win64: return "Win64";
    case CallingConvKind::SysV64: return "SysV64";
    case CallingConvKind::Custom: return "Custom";
    default: return "unknown";
    }
}

std::ostream& operator<<(std::ostream& os, CallingConvKind kind) {
    return os << to_string(kind);
}

} // namespace brass
