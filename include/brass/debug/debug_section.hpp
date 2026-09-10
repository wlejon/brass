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

    DebugLoc resolve_offset(uint32_t offset) const;

    SourceMap to_source_map(const DebugContext& ctx, const std::string& output_name = "") const;

private:
    std::string function_name_;
    uint32_t code_size_ = 0;
    uintptr_t code_base_ = 0;
    std::vector<DebugLineEntry> line_entries_;
};

// Compact Binary Debug Line Section (.brass_dbg) Serializer & Deserializer
std::vector<uint8_t> serialize_debug_section(const DebugContext& ctx, const std::vector<FunctionDebugTable>& tables);
bool deserialize_debug_section(const uint8_t* data, size_t size, DebugContext& ctx, std::vector<FunctionDebugTable>& tables);

} // namespace brass
