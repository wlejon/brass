#pragma once

#include <brass/target/target.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <iosfwd>

namespace brass {

enum class CallingConvKind : uint8_t {
    Win64,
    SysV64,
    SysV = SysV64,
    AAPCS64,
    AppleAAPCS64,
    Custom,
};

struct CustomCallingConvConfig {
    std::vector<x64::GPR> arg_gprs;
    std::vector<x64::XMM> arg_xmms;
    std::vector<x64::GPR> ret_gprs;
    std::vector<x64::XMM> ret_xmms;
    x64::RegMask callee_saved_gprs = 0;
    x64::RegMask callee_saved_xmms = 0;
    x64::RegMask caller_saved_gprs = 0;
    x64::RegMask caller_saved_xmms = 0;
    size_t shadow_space = 0;
    x64::RegMask gcref_preserved_gprs = 0;

    std::vector<aarch64::GPR> aarch64_arg_gprs;
    std::vector<aarch64::FPR> aarch64_arg_fprs;
    std::vector<aarch64::GPR> aarch64_ret_gprs;
    std::vector<aarch64::FPR> aarch64_ret_fprs;
    aarch64::RegMask aarch64_callee_saved_gprs = 0;
    aarch64::RegMask aarch64_callee_saved_fprs = 0;
    aarch64::RegMask aarch64_caller_saved_gprs = 0;
    aarch64::RegMask aarch64_caller_saved_fprs = 0;
};

class CallingConvention {
public:
    CallingConvention() = default;

    CallingConvKind kind() const noexcept { return kind_; }
    const Target& target() const noexcept { return target_; }
    size_t shadow_space() const noexcept { return shadow_space_; }

    // x64 Register queries
    const std::vector<x64::GPR>& arg_gprs() const noexcept { return arg_gprs_; }
    const std::vector<x64::XMM>& arg_xmms() const noexcept { return arg_xmms_; }
    const std::vector<x64::GPR>& ret_gprs() const noexcept { return ret_gprs_; }
    const std::vector<x64::XMM>& ret_xmms() const noexcept { return ret_xmms_; }

    size_t num_arg_gprs() const noexcept {
        return !aarch64_arg_gprs_.empty() ? aarch64_arg_gprs_.size() : arg_gprs_.size();
    }
    size_t num_arg_xmms() const noexcept {
        return !aarch64_arg_fprs_.empty() ? aarch64_arg_fprs_.size() : arg_xmms_.size();
    }
    size_t num_ret_gprs() const noexcept {
        return !aarch64_ret_gprs_.empty() ? aarch64_ret_gprs_.size() : ret_gprs_.size();
    }
    size_t num_ret_xmms() const noexcept {
        return !aarch64_ret_fprs_.empty() ? aarch64_ret_fprs_.size() : ret_xmms_.size();
    }

    x64::GPR arg_gpr(size_t index) const noexcept {
        if (index < arg_gprs_.size()) return arg_gprs_[index];
        return x64::GPR::None;
    }

    x64::XMM arg_xmm(size_t index) const noexcept {
        if (index < arg_xmms_.size()) return arg_xmms_[index];
        return x64::XMM::None;
    }

    x64::GPR ret_gpr(size_t index = 0) const noexcept {
        if (index < ret_gprs_.size()) return ret_gprs_[index];
        return x64::GPR::None;
    }

    x64::XMM ret_xmm(size_t index = 0) const noexcept {
        if (index < ret_xmms_.size()) return ret_xmms_[index];
        return x64::XMM::None;
    }

    x64::RegMask callee_saved_gpr_mask() const noexcept { return callee_saved_gprs_; }
    x64::RegMask callee_saved_xmm_mask() const noexcept { return callee_saved_xmms_; }
    x64::RegMask caller_saved_gpr_mask() const noexcept { return caller_saved_gprs_; }
    x64::RegMask caller_saved_xmm_mask() const noexcept { return caller_saved_xmms_; }
    x64::RegMask gcref_preserved_gpr_mask() const noexcept { return gcref_preserved_gprs_; }

    bool is_callee_saved(x64::GPR reg) const noexcept {
        return x64::mask_has(callee_saved_gprs_, reg);
    }

    bool is_callee_saved(x64::XMM reg) const noexcept {
        return x64::mask_has(callee_saved_xmms_, reg);
    }

    bool is_caller_saved(x64::GPR reg) const noexcept {
        return x64::mask_has(caller_saved_gprs_, reg);
    }

    bool is_caller_saved(x64::XMM reg) const noexcept {
        return x64::mask_has(caller_saved_xmms_, reg);
    }

    bool is_gcref_preserved(x64::GPR reg) const noexcept {
        return x64::mask_has(gcref_preserved_gprs_, reg);
    }

    bool is_arg_gpr(x64::GPR reg) const noexcept;
    bool is_arg_xmm(x64::XMM reg) const noexcept;
    bool is_ret_gpr(x64::GPR reg) const noexcept;
    bool is_ret_xmm(x64::XMM reg) const noexcept;

    // AArch64 Register queries
    const std::vector<aarch64::GPR>& aarch64_arg_gprs() const noexcept { return aarch64_arg_gprs_; }
    const std::vector<aarch64::FPR>& aarch64_arg_fprs() const noexcept { return aarch64_arg_fprs_; }
    const std::vector<aarch64::GPR>& aarch64_ret_gprs() const noexcept { return aarch64_ret_gprs_; }
    const std::vector<aarch64::FPR>& aarch64_ret_fprs() const noexcept { return aarch64_ret_fprs_; }

    aarch64::GPR aarch64_arg_gpr(size_t index) const noexcept {
        if (index < aarch64_arg_gprs_.size()) return aarch64_arg_gprs_[index];
        return aarch64::GPR::None;
    }

    aarch64::FPR aarch64_arg_fpr(size_t index) const noexcept {
        if (index < aarch64_arg_fprs_.size()) return aarch64_arg_fprs_[index];
        return aarch64::FPR::None;
    }

    aarch64::GPR aarch64_ret_gpr(size_t index = 0) const noexcept {
        if (index < aarch64_ret_gprs_.size()) return aarch64_ret_gprs_[index];
        return aarch64::GPR::None;
    }

    aarch64::FPR aarch64_ret_fpr(size_t index = 0) const noexcept {
        if (index < aarch64_ret_fprs_.size()) return aarch64_ret_fprs_[index];
        return aarch64::FPR::None;
    }

    aarch64::RegMask aarch64_callee_saved_gpr_mask() const noexcept { return aarch64_callee_saved_gprs_; }
    aarch64::RegMask aarch64_callee_saved_fpr_mask() const noexcept { return aarch64_callee_saved_fprs_; }
    aarch64::RegMask aarch64_caller_saved_gpr_mask() const noexcept { return aarch64_caller_saved_gprs_; }
    aarch64::RegMask aarch64_caller_saved_fpr_mask() const noexcept { return aarch64_caller_saved_fprs_; }

    bool is_callee_saved(aarch64::GPR reg) const noexcept {
        return aarch64::mask_has(aarch64_callee_saved_gprs_, reg);
    }

    bool is_callee_saved(aarch64::FPR reg) const noexcept {
        return aarch64::mask_has(aarch64_callee_saved_fprs_, reg);
    }

    bool is_caller_saved(aarch64::GPR reg) const noexcept {
        return aarch64::mask_has(aarch64_caller_saved_gprs_, reg);
    }

    bool is_caller_saved(aarch64::FPR reg) const noexcept {
        return aarch64::mask_has(aarch64_caller_saved_fprs_, reg);
    }

    bool is_arg_gpr(aarch64::GPR reg) const noexcept;
    bool is_arg_fpr(aarch64::FPR reg) const noexcept;
    bool is_ret_gpr(aarch64::GPR reg) const noexcept;
    bool is_ret_fpr(aarch64::FPR reg) const noexcept;

    // Factory methods
    static CallingConvention win64();
    static CallingConvention sysv64();
    static CallingConvention aapcs64();
    static CallingConvention apple_aapcs64();
    static CallingConvention for_target(const Target& target);
    static CallingConvention custom(const CustomCallingConvConfig& config);

private:
    CallingConvKind kind_ = CallingConvKind::Win64;
    Target target_ = Target::x64_windows();

    // x64 members
    std::vector<x64::GPR> arg_gprs_;
    std::vector<x64::XMM> arg_xmms_;
    std::vector<x64::GPR> ret_gprs_;
    std::vector<x64::XMM> ret_xmms_;
    x64::RegMask callee_saved_gprs_ = 0;
    x64::RegMask callee_saved_xmms_ = 0;
    x64::RegMask caller_saved_gprs_ = 0;
    x64::RegMask caller_saved_xmms_ = 0;
    size_t shadow_space_ = 0;
    x64::RegMask gcref_preserved_gprs_ = 0;

    // AArch64 members
    std::vector<aarch64::GPR> aarch64_arg_gprs_;
    std::vector<aarch64::FPR> aarch64_arg_fprs_;
    std::vector<aarch64::GPR> aarch64_ret_gprs_;
    std::vector<aarch64::FPR> aarch64_ret_fprs_;
    aarch64::RegMask aarch64_callee_saved_gprs_ = 0;
    aarch64::RegMask aarch64_callee_saved_fprs_ = 0;
    aarch64::RegMask aarch64_caller_saved_gprs_ = 0;
    aarch64::RegMask aarch64_caller_saved_fprs_ = 0;
};

std::string_view to_string(CallingConvKind kind) noexcept;

std::ostream& operator<<(std::ostream& os, CallingConvKind kind);

} // namespace brass
