#pragma once

#include <brass/target/target.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace brass {

enum class CallingConvKind : uint8_t {
    Win64,
    SysV64,
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
};

class CallingConvention {
public:
    CallingConvention() = default;

    CallingConvKind kind() const noexcept { return kind_; }
    size_t shadow_space() const noexcept { return shadow_space_; }

    const std::vector<x64::GPR>& arg_gprs() const noexcept { return arg_gprs_; }
    const std::vector<x64::XMM>& arg_xmms() const noexcept { return arg_xmms_; }
    const std::vector<x64::GPR>& ret_gprs() const noexcept { return ret_gprs_; }
    const std::vector<x64::XMM>& ret_xmms() const noexcept { return ret_xmms_; }

    size_t num_arg_gprs() const noexcept { return arg_gprs_.size(); }
    size_t num_arg_xmms() const noexcept { return arg_xmms_.size(); }
    size_t num_ret_gprs() const noexcept { return ret_gprs_.size(); }
    size_t num_ret_xmms() const noexcept { return ret_xmms_.size(); }

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

    static CallingConvention win64();
    static CallingConvention sysv64();
    static CallingConvention for_target(const Target& target);
    static CallingConvention custom(const CustomCallingConvConfig& config);

private:
    CallingConvKind kind_ = CallingConvKind::Win64;
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
};

std::string_view to_string(CallingConvKind kind) noexcept;

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, CallingConvKind kind) {
    return os << to_string(kind);
}

} // namespace brass
