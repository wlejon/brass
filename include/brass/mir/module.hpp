#pragma once

#include <brass/core/arena.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/core/span.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/debug/source_loc.hpp>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <utility>
#include <initializer_list>

namespace brass {

class Module {
public:
    Module() noexcept = default;
    explicit Module(std::string_view name);
    ~Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) noexcept = default;
    // Not defaulted: the functions being replaced must leave the runtime
    // registries first, as in the destructor.
    Module& operator=(Module&& other) noexcept;

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
    void rename_function(Function* fn, std::string_view new_name);

    void add_external_symbol(std::string_view sym);
    bool has_external_symbol(std::string_view sym) const noexcept;
    const std::vector<std::string_view>& external_symbols() const noexcept {
        return external_symbols_;
    }

    // The runtime-symbol registry (runtime_symbols.hpp). Declares external
    // `sym` (if it is not yet) with `role`. Passes rely on the role's
    // promise, so only a frontend that owns the function's contract may make
    // it. Printed as `extern @sym role...`.
    void add_symbol_role(std::string_view sym, SymbolRole role);
    bool has_symbol_role(std::string_view sym, SymbolRole role) const noexcept;
    // Every role declared on `sym`, in kAllSymbolRoles order.
    std::vector<SymbolRole> symbol_roles(std::string_view sym) const;

    // add_symbol_role(sym, SymbolRole::Allocator): calls to `sym` return a
    // fresh object no other pointer refers to, and touch no memory the
    // caller can see.
    void add_allocation_function(std::string_view sym);
    bool is_allocation_function(std::string_view sym) const noexcept;

    // Read-only data the module itself defines: symbol `sym` names a
    // NUL-terminated copy of `text` the module owns. Every engine that loads
    // the module defines it (object emission puts it in read-only data; the
    // baseline JIT resolves it to the module's copy), so `func_addr @sym`
    // needs no host registration. Redefining `sym` with other text throws.
    void define_string_symbol(std::string_view sym, std::string_view text);
    // The module's NUL-terminated copy for `sym`, or null if not defined here.
    const char* string_symbol(std::string_view sym) const noexcept;
    const std::vector<std::pair<std::string_view, std::string_view>>& string_symbols() const noexcept {
        return string_symbols_;
    }

    // Copies `src`'s external declarations and their roles, and its string
    // symbols, for a module built to optimize or compile code cloned out of
    // `src`.
    void copy_declarations_from(const Module& src);

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
    // Bit i set: the symbol has role kAllSymbolRoles[i].
    std::unordered_map<std::string_view, uint32_t> symbol_roles_;
    std::vector<std::pair<std::string_view, std::string_view>> string_symbols_; // (sym, text), both interned
    bool allow_fp_reassociation_ = false;
    bool pinned_tls_register_ = false;
    bool has_loop_optimizations_ = false;
    DebugContext debug_context_;
};

} // namespace brass
