#include <brass/mir/module.hpp>
#include <brass/runtime/code_installer.hpp>
#include <algorithm>

namespace brass {

Module::Module(std::string_view name)
    : arena_(), string_pool_(), name_(string_pool_.intern(name)) {}

Module::~Module() {
    // Runtime registries hold raw pointers into this module; drop them before
    // the arena frees the functions, so a new module reusing the addresses is
    // never mistaken for this one.
    runtime::forget_module(*this);
}

Module& Module::operator=(Module&& other) noexcept {
    if (this == &other) return *this;
    runtime::forget_module(*this);
    arena_ = std::move(other.arena_);
    string_pool_ = std::move(other.string_pool_);
    name_ = other.name_;
    functions_ = std::move(other.functions_);
    function_map_ = std::move(other.function_map_);
    external_symbols_ = std::move(other.external_symbols_);
    symbol_roles_ = std::move(other.symbol_roles_);
    allow_fp_reassociation_ = other.allow_fp_reassociation_;
    pinned_tls_register_ = other.pinned_tls_register_;
    has_loop_optimizations_ = other.has_loop_optimizations_;
    debug_context_ = std::move(other.debug_context_);
    return *this;
}

void Module::set_name(std::string_view name) {
    name_ = string_pool_.intern(name);
}

Function* Module::create_function(std::string_view name, Type return_type, Span<const Type> param_types) {
    std::string_view interned_name = string_pool_.intern(name);
    std::vector<Type> params;
    params.reserve(param_types.size());
    for (size_t i = 0; i < param_types.size(); ++i) {
        params.push_back(param_types[i]);
    }

    Function* fn = arena_.make<Function>(interned_name, return_type, std::move(params));
    fn->set_parent(this);
    functions_.push_back(fn);
    function_map_[interned_name] = fn;
    return fn;
}

Function* Module::create_function(std::string_view name, Type return_type, std::initializer_list<Type> param_types) {
    return create_function(name, return_type, Span<const Type>(param_types.begin(), param_types.size()));
}

Function* Module::create_function(std::string_view name, Type return_type) {
    return create_function(name, return_type, Span<const Type>());
}

Function* Module::get_function(std::string_view name) const noexcept {
    auto it = function_map_.find(name);
    if (it != function_map_.end()) {
        return it->second;
    }
    return nullptr;
}

void Module::rename_function(Function* fn, std::string_view new_name) {
    if (!fn) return;
    function_map_.erase(fn->name());
    std::string_view interned = string_pool_.intern(new_name);
    fn->set_name(interned);
    function_map_[interned] = fn;
}

void Module::add_external_symbol(std::string_view sym) {
    std::string_view interned_sym = string_pool_.intern(sym);
    if (!has_external_symbol(interned_sym)) {
        external_symbols_.push_back(interned_sym);
    }
}

bool Module::has_external_symbol(std::string_view sym) const noexcept {
    return std::find(external_symbols_.begin(), external_symbols_.end(), sym) != external_symbols_.end();
}

namespace {

uint32_t role_bit(SymbolRole role) noexcept {
    return uint32_t{1} << static_cast<uint32_t>(role);
}

} // namespace

void Module::add_symbol_role(std::string_view sym, SymbolRole role) {
    add_external_symbol(sym);
    symbol_roles_[string_pool_.intern(sym)] |= role_bit(role);
}

bool Module::has_symbol_role(std::string_view sym, SymbolRole role) const noexcept {
    auto it = symbol_roles_.find(sym);
    return it != symbol_roles_.end() && (it->second & role_bit(role)) != 0;
}

std::vector<SymbolRole> Module::symbol_roles(std::string_view sym) const {
    std::vector<SymbolRole> roles;
    for (SymbolRole role : kAllSymbolRoles) {
        if (has_symbol_role(sym, role)) roles.push_back(role);
    }
    return roles;
}

void Module::add_allocation_function(std::string_view sym) {
    add_symbol_role(sym, SymbolRole::Allocator);
}

bool Module::is_allocation_function(std::string_view sym) const noexcept {
    return has_symbol_role(sym, SymbolRole::Allocator);
}

void Module::copy_declarations_from(const Module& src) {
    for (std::string_view sym : src.external_symbols()) {
        add_external_symbol(sym);
        for (SymbolRole role : src.symbol_roles(sym)) add_symbol_role(sym, role);
    }
}

} // namespace brass
