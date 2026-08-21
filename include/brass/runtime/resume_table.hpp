#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <iosfwd>

namespace brass::runtime {

struct ResumeEntry {
    uint32_t resume_id = 0;
    size_t code_offset = 0;
    std::string block_name;

    constexpr ResumeEntry() noexcept = default;
    ResumeEntry(uint32_t id, size_t off, std::string_view name)
        : resume_id(id), code_offset(off), block_name(name) {}

    bool operator==(const ResumeEntry& other) const noexcept = default;
};

class FunctionResumeTable {
public:
    FunctionResumeTable() = default;

    void add_entry(uint32_t resume_id, size_t code_offset, std::string_view block_name = "");
    bool has_entry(uint32_t resume_id) const noexcept;
    size_t get_offset(uint32_t resume_id) const;
    std::string_view get_block_name(uint32_t resume_id) const noexcept;
    void* get_target_address(void* fn_base, uint32_t resume_id) const noexcept;

    const std::vector<ResumeEntry>& entries() const noexcept { return entries_; }
    size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    void clear() noexcept { entries_.clear(); }

private:
    std::vector<ResumeEntry> entries_;
};

class ResumeTableRegistry {
public:
    ResumeTableRegistry() = default;

    void register_table(std::string_view fn_name, FunctionResumeTable table);
    const FunctionResumeTable* get_table(std::string_view fn_name) const noexcept;
    FunctionResumeTable* get_table(std::string_view fn_name) noexcept;
    bool has_table(std::string_view fn_name) const noexcept;
    void clear() noexcept { tables_.clear(); }
    size_t size() const noexcept { return tables_.size(); }
    bool empty() const noexcept { return tables_.empty(); }

    const std::unordered_map<std::string, FunctionResumeTable>& tables() const noexcept {
        return tables_;
    }

private:
    std::unordered_map<std::string, FunctionResumeTable> tables_;
};

std::ostream& operator<<(std::ostream& os, const FunctionResumeTable& table);

} // namespace brass::runtime
