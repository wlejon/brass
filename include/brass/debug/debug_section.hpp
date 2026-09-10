#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/debug/source_map.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace brass {

constexpr uint32_t kDebugSectionMagic = 0x47424442; // 'B' 'D' 'B' 'G'
constexpr uint32_t kDebugSectionVersion = 1;

struct DebugLineEntry {
    uint32_t code_offset = 0;
    DebugLoc loc;

    constexpr bool operator==(const DebugLineEntry& other) const noexcept = default;
};

inline std::ostream& operator<<(std::ostream& os, const DebugLineEntry& e) {
    os << "DebugLineEntry(offset=" << e.code_offset << ", loc=" << e.loc << ")";
    return os;
}

struct DebugVariable {
    std::string name;
    int32_t stack_offset = 0; // Stack offset relative to frame base (%rbp)
    uint32_t type_index = 0;   // CodeView type index or 0
    bool is_parameter = false; // true if formal parameter, false if local variable
    uint32_t decl_file = 1;
    uint32_t decl_line = 0;
    uint32_t decl_column = 0;

    bool operator==(const DebugVariable& other) const noexcept {
        return name == other.name &&
               stack_offset == other.stack_offset &&
               type_index == other.type_index &&
               is_parameter == other.is_parameter &&
               decl_file == other.decl_file &&
               decl_line == other.decl_line &&
               decl_column == other.decl_column;
    }
};

class FunctionDebugTable {
public:
    FunctionDebugTable() = default;
    explicit FunctionDebugTable(std::string function_name, uint32_t code_size = 0)
        : function_name_(std::move(function_name)), code_size_(code_size) {}

    const std::string& function_name() const noexcept { return function_name_; }
    void set_function_name(std::string name) { function_name_ = std::move(name); }

    uint32_t code_size() const noexcept { return code_size_; }
    void set_code_size(uint32_t size) noexcept { code_size_ = size; }

    uintptr_t code_base() const noexcept { return code_base_; }
    void set_code_base(uintptr_t base) noexcept { code_base_ = base; }

    const std::vector<DebugLineEntry>& line_entries() const noexcept { return line_entries_; }
    void add_line_entry(uint32_t code_offset, DebugLoc loc);
    void set_line_entries(std::vector<DebugLineEntry> entries);

    uint32_t decl_file() const noexcept {
        if (decl_file_ != 0) return decl_file_;
        return line_entries_.empty() ? 1 : line_entries_.front().loc.file_id;
    }
    void set_decl_file(uint32_t file_id) noexcept { decl_file_ = file_id; }

    uint32_t decl_line() const noexcept {
        if (decl_line_ != 0) return decl_line_;
        return line_entries_.empty() ? 1 : line_entries_.front().loc.line;
    }
    void set_decl_line(uint32_t line) noexcept { decl_line_ = line; }

    uint32_t prologue_size() const noexcept { return prologue_size_; }
    void set_prologue_size(uint32_t sz) noexcept { prologue_size_ = sz; }

    const std::vector<DebugVariable>& variables() const noexcept { return variables_; }
    std::vector<DebugVariable>& variables() noexcept { return variables_; }
    void add_variable(DebugVariable var) { variables_.push_back(std::move(var)); }
    void set_variables(std::vector<DebugVariable> vars) { variables_ = std::move(vars); }

    DebugLoc resolve_offset(uint32_t offset) const;

    SourceMap to_source_map(const DebugContext& ctx, const std::string& output_name = "") const;

private:
    std::string function_name_;
    uint32_t code_size_ = 0;
    uintptr_t code_base_ = 0;
    uint32_t decl_file_ = 0;
    uint32_t decl_line_ = 0;
    uint32_t prologue_size_ = 0;
    std::vector<DebugLineEntry> line_entries_;
    std::vector<DebugVariable> variables_;
};

// Compact Binary Debug Line Section (.brass_dbg) Serializer & Deserializer
std::vector<uint8_t> serialize_debug_section(const DebugContext& ctx, const std::vector<FunctionDebugTable>& tables);
bool deserialize_debug_section(const uint8_t* data, size_t size, DebugContext& ctx, std::vector<FunctionDebugTable>& tables);

} // namespace brass
