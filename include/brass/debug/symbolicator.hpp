#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/debug/debug_section.hpp>
#include <brass/gc/stack_map.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace brass {

struct StackFrame {
    std::string function_name;
    std::string file;
    uint32_t line = 0;
    uint32_t column = 0;
    bool is_inlined = false;

    std::string to_string() const; // Formats: "    at func (file.js:line:col)"

    bool operator==(const StackFrame& other) const noexcept {
        return function_name == other.function_name &&
               file == other.file &&
               line == other.line &&
               column == other.column &&
               is_inlined == other.is_inlined;
    }
};

inline std::ostream& operator<<(std::ostream& os, const StackFrame& f) {
    os << f.to_string();
    return os;
}

struct StackTrace {
    std::vector<StackFrame> frames;

    std::string format() const;
    bool empty() const noexcept { return frames.empty(); }
    size_t frame_count() const noexcept { return frames.size(); }
};

class Symbolicator {
public:
    Symbolicator() = default;
    Symbolicator(const DebugContext& ctx, const std::vector<FunctionDebugTable>& tables);

    void set_debug_context(const DebugContext& ctx);
    void add_function_table(const FunctionDebugTable& table);
    void set_function_tables(const std::vector<FunctionDebugTable>& tables);
    void register_function_address(const std::string& name, uintptr_t code_base, size_t code_size);

    StackTrace symbolize_ip(uintptr_t code_base, uintptr_t ip) const;
    StackTrace symbolize_offset(const std::string& fn_name, uint32_t offset) const;
    StackTrace symbolize_ip(uintptr_t ip) const;

    StackTrace symbolize_stack(uintptr_t top_rbp, uintptr_t top_return_ip, const ModuleStackMap& stack_maps) const;

    const DebugContext& debug_context() const noexcept { return ctx_; }
    const std::vector<FunctionDebugTable>& tables() const noexcept { return tables_; }

private:
    DebugContext ctx_;
    std::vector<FunctionDebugTable> tables_;
    std::unordered_map<std::string, size_t> fn_to_table_idx_;

    struct RegisteredRange {
        uintptr_t base = 0;
        size_t size = 0;
        std::string name;
    };
    std::vector<RegisteredRange> registered_ranges_;

    void expand_inlined_frames(const std::string& fn_name, DebugLoc loc, std::vector<StackFrame>& frames) const;
};

} // namespace brass
