#pragma once

#include <brass/core/arena.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/core/span.hpp>
#include <brass/mir/function.hpp>
#include <brass/debug/source_loc.hpp>
#include <string_view>
#include <vector>
#include <initializer_list>

namespace brass {

class Module {
public:
    Module() noexcept = default;
    explicit Module(std::string_view name);
    ~Module() = default;

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;

    std::string_view name() const noexcept { return name_; }
    void set_name(std::string_view name);

    Arena& arena() noexcept { return arena_; }
    const Arena& arena() const noexcept { return arena_; }

    StringPool& string_pool() noexcept { return string_pool_; }
    const StringPool& string_pool() const noexcept { return string_pool_; }

    Function* create_function(std::string_view name, Type return_type, Span<const Type> param_types);
    Function* create_function(std::string_view name, Type return_type, std::initializer_list<Type> param_types);
    Function* create_function(std::string_view name, Type return_type);

    const std::vector<Function*>& functions() const noexcept { return functions_; }
    std::vector<Function*>& functions() noexcept { return functions_; }
    size_t function_count() const noexcept { return functions_.size(); }
    Function* get_function(std::string_view name) const noexcept;

    void add_external_symbol(std::string_view sym);
    bool has_external_symbol(std::string_view sym) const noexcept;
    const std::vector<std::string_view>& external_symbols() const noexcept {
        return external_symbols_;
    }

    bool allow_fp_reassociation() const noexcept { return allow_fp_reassociation_; }
    void set_allow_fp_reassociation(bool allow) noexcept { allow_fp_reassociation_ = allow; }

    DebugContext& debug_context() noexcept { return debug_context_; }
    const DebugContext& debug_context() const noexcept { return debug_context_; }

private:
    Arena arena_;
    StringPool string_pool_;
    std::string_view name_;
    std::vector<Function*> functions_;
    std::vector<std::string_view> external_symbols_;
    bool allow_fp_reassociation_ = false;
    DebugContext debug_context_;
};

} // namespace brass
