#include <brass/runtime/resume_table.hpp>
#include <stdexcept>
#include <ostream>

namespace brass::runtime {

void FunctionResumeTable::add_entry(uint32_t resume_id, size_t code_offset, std::string_view block_name) {
    for (auto& entry : entries_) {
        if (entry.resume_id == resume_id) {
            entry.code_offset = code_offset;
            entry.block_name = std::string(block_name);
            return;
        }
    }
    entries_.emplace_back(resume_id, code_offset, block_name);
}

bool FunctionResumeTable::has_entry(uint32_t resume_id) const noexcept {
    for (const auto& entry : entries_) {
        if (entry.resume_id == resume_id) {
            return true;
        }
    }
    return false;
}

size_t FunctionResumeTable::get_offset(uint32_t resume_id) const {
    for (const auto& entry : entries_) {
        if (entry.resume_id == resume_id) {
            return entry.code_offset;
        }
    }
    throw std::out_of_range("FunctionResumeTable: resume_id " + std::to_string(resume_id) + " not found");
}

std::string_view FunctionResumeTable::get_block_name(uint32_t resume_id) const noexcept {
    for (const auto& entry : entries_) {
        if (entry.resume_id == resume_id) {
            return entry.block_name;
        }
    }
    return "";
}

void* FunctionResumeTable::get_target_address(void* fn_base, uint32_t resume_id) const noexcept {
    if (!fn_base) return nullptr;
    for (const auto& entry : entries_) {
        if (entry.resume_id == resume_id) {
            return static_cast<uint8_t*>(fn_base) + entry.code_offset;
        }
    }
    return nullptr;
}

void ResumeTableRegistry::register_table(std::string_view fn_name, FunctionResumeTable table) {
    tables_[std::string(fn_name)] = std::move(table);
}

const FunctionResumeTable* ResumeTableRegistry::get_table(std::string_view fn_name) const noexcept {
    auto it = tables_.find(std::string(fn_name));
    if (it != tables_.end()) {
        return &it->second;
    }
    return nullptr;
}

FunctionResumeTable* ResumeTableRegistry::get_table(std::string_view fn_name) noexcept {
    auto it = tables_.find(std::string(fn_name));
    if (it != tables_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool ResumeTableRegistry::has_table(std::string_view fn_name) const noexcept {
    return tables_.find(std::string(fn_name)) != tables_.end();
}

std::ostream& operator<<(std::ostream& os, const FunctionResumeTable& table) {
    os << "FunctionResumeTable {";
    for (size_t i = 0; i < table.entries().size(); ++i) {
        if (i > 0) os << ", ";
        const auto& e = table.entries()[i];
        os << e.resume_id << " -> +" << e.code_offset;
        if (!e.block_name.empty()) {
            os << " (" << e.block_name << ")";
        }
    }
    os << "}";
    return os;
}

} // namespace brass::runtime
