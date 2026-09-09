#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>
#include <memory>
#include <string_view>
#include <iosfwd>

namespace brass {

class EscapeAnalysis;

enum class AliasResult : uint8_t {
    NoAlias = 0,
    MayAlias = 1,
    MustAlias = 2
};

std::string_view alias_result_name(AliasResult result) noexcept;
std::ostream& operator<<(std::ostream& os, AliasResult result);

class AliasAnalysis {
public:
    explicit AliasAnalysis(const Function& fn);
    AliasAnalysis(const Function& fn, const EscapeAnalysis* ea);
    ~AliasAnalysis();

    AliasAnalysis(const AliasAnalysis&) = delete;
    AliasAnalysis& operator=(const AliasAnalysis&) = delete;
    AliasAnalysis(AliasAnalysis&&) noexcept;
    AliasAnalysis& operator=(AliasAnalysis&&) noexcept;

    // Disambiguation Query API
    AliasResult alias(const Value* ptr1, int32_t off1, const Value* ptr2, int32_t off2) const;
    AliasResult alias(const Value* ptr1, int32_t off1, Type type1,
                      const Value* ptr2, int32_t off2, Type type2) const;

    // Query whether write_inst could clobber the value loaded by read_inst
    bool can_clobber(const Instruction* write_inst, const Instruction* read_inst) const;

    // Helper queries
    const Value* get_underlying_base(const Value* ptr, int64_t& out_offset) const;
    bool is_distinct_allocation(const Value* base1, const Value* base2) const;
    bool is_non_escaping(const Value* base) const;
    bool is_allocation(const Value* val) const;

    const Function& function() const noexcept { return *fn_; }
    const EscapeAnalysis* escape_analysis() const noexcept;

private:
    const Function* fn_ = nullptr;
    const EscapeAnalysis* external_ea_ = nullptr;
    std::unique_ptr<EscapeAnalysis> owned_ea_;
};

} // namespace brass
