#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <ostream>

namespace brass {

struct DebugLoc {
    uint32_t file_id = 0;
    uint32_t line = 0;
    uint32_t column = 0;
    uint32_t inlined_at_id = 0; // 0 = not inlined; >0 index into InlinedScopeTable

    constexpr DebugLoc() noexcept = default;
    constexpr DebugLoc(uint32_t f, uint32_t l, uint32_t c = 0, uint32_t inlined = 0) noexcept
        : file_id(f), line(l), column(c), inlined_at_id(inlined) {}

    constexpr bool is_valid() const noexcept {
        return file_id != 0 || line != 0;
    }

    constexpr bool operator==(const DebugLoc& other) const noexcept = default;
    constexpr bool operator!=(const DebugLoc& other) const noexcept = default;
};

inline std::ostream& operator<<(std::ostream& os, const DebugLoc& loc) {
    os << "DebugLoc(file=" << loc.file_id << ", line=" << loc.line << ", col=" << loc.column
       << ", inlined=" << loc.inlined_at_id << ")";
    return os;
}

struct InlinedScope {
    std::string callee_name;
    DebugLoc callsite_loc;
    uint32_t parent_inlined_at_id = 0;

    InlinedScope() = default;
    InlinedScope(std::string callee, DebugLoc callsite, uint32_t parent_id = 0)
        : callee_name(std::move(callee)), callsite_loc(callsite), parent_inlined_at_id(parent_id) {}

    bool operator==(const InlinedScope& other) const noexcept {
        return callee_name == other.callee_name &&
               callsite_loc == other.callsite_loc &&
               parent_inlined_at_id == other.parent_inlined_at_id;
    }
};

inline std::ostream& operator<<(std::ostream& os, const InlinedScope& s) {
    os << "InlinedScope(callee=" << s.callee_name << ", parent=" << s.parent_inlined_at_id << ")";
    return os;
}

class DebugContext {
public:
    DebugContext() = default;

    // File table interning (1-based IDs: 1, 2, 3...)
    uint32_t get_or_add_file(const std::string& path);
    uint32_t get_file_id(const std::string& path) const;
    const std::string& get_file(uint32_t file_id) const;
    const std::vector<std::string>& files() const noexcept { return files_; }
    size_t file_count() const noexcept { return files_.size(); }

    // Inlined scope tracking (1-based IDs: 1, 2, 3...)
    uint32_t record_inlined_scope(std::string callee, DebugLoc callsite, uint32_t parent_id = 0);
    const InlinedScope* get_inlined_scope(uint32_t inlined_id) const;
    const std::vector<InlinedScope>& inlined_scopes() const noexcept { return inlined_scopes_; }
    size_t inlined_scope_count() const noexcept { return inlined_scopes_.size(); }

    // Wraps an existing inlined scope tree under a new outer parent
    uint32_t wrap_inlined_scope(uint32_t existing_inlined_id, uint32_t outer_inlined_id);

    void clear();

private:
    std::vector<std::string> files_;
    std::unordered_map<std::string, uint32_t> file_to_id_;

    std::vector<InlinedScope> inlined_scopes_;
};

} // namespace brass
