#pragma once

#include <brass/core/arena.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/core/span.hpp>
#include <brass/mir/function.hpp>
#include <brass/debug/source_loc.hpp>
#include <string_view>
#include <vector>
#include <unordered_map>
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

    // When set, every function in the module keeps one callee-saved register
    // (x64: R13, aarch64: X28) out of allocation and reads its thread-local
    // block through it (Opcode::pinned_tls_read). The module entry writes it
    // (Opcode::pinned_tls_write); every other way into the module's code —
    // a callback the runtime invokes — must arrive with the register already
    // set, which is the runtime's trampoline's job.
    bool pinned_tls_register() const noexcept { return pinned_tls_register_; }
    void set_pinned_tls_register(bool pinned) noexcept { pinned_tls_register_ = pinned; }

    bool has_loop_optimizations() const noexcept { return has_loop_optimizations_; }
    void set_has_loop_optimizations(bool opt) noexcept { has_loop_optimizations_ = opt; }

    DebugContext& debug_context() noexcept { return debug_context_; }
    const DebugContext& debug_context() const noexcept { return debug_context_; }

private:
    Arena arena_;
    StringPool string_pool_;
    std::string_view name_;
    std::vector<Function*> functions_;
    std::unordered_map<std::string_view, Function*> function_map_;
    std::vector<std::string_view> external_symbols_;
    bool allow_fp_reassociation_ = false;
    bool pinned_tls_register_ = false;
    bool has_loop_optimizations_ = false;
    DebugContext debug_context_;
};

} // namespace brass
