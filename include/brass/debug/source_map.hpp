#pragma once

#include <brass/debug/source_loc.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <string_view>

namespace brass {

namespace vlq {
    std::string encode(int32_t value);
    bool decode(std::string_view::const_iterator& it, std::string_view::const_iterator end, int32_t& result);
}

struct SourceMapping {
    uint32_t code_offset = 0;
    DebugLoc loc;
    std::string symbol;

    constexpr bool operator==(const SourceMapping& other) const noexcept = default;
};

class SourceMap {
public:
    SourceMap() = default;
    explicit SourceMap(std::string file_name) : file_(std::move(file_name)) {}

    const std::string& file() const noexcept { return file_; }
    void set_file(std::string file_name) { file_ = std::move(file_name); }

    const std::vector<std::string>& sources() const noexcept { return sources_; }
    void set_sources(std::vector<std::string> sources);
    void add_source(const std::string& src);
    uint32_t get_or_add_source(const std::string& src);

    const std::vector<std::string>& names() const noexcept { return names_; }
    void set_names(std::vector<std::string> names);
    void add_name(const std::string& name);
    uint32_t get_or_add_name(const std::string& name);

    const std::vector<SourceMapping>& mappings() const noexcept { return mappings_; }

    void add_mapping(uint32_t code_offset, DebugLoc loc, const std::string& symbol = "");

    std::string to_json() const;
    bool write_file(const std::string& path, std::string* err = nullptr) const;
    static std::unique_ptr<SourceMap> parse_json(const std::string& json, std::string* err = nullptr);
    DebugLoc resolve_offset(uint32_t code_offset) const;

private:
    std::string file_;
    std::vector<std::string> sources_;
    std::unordered_map<std::string, uint32_t> source_to_idx_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, uint32_t> name_to_idx_;
    std::vector<SourceMapping> mappings_;
};

} // namespace brass
