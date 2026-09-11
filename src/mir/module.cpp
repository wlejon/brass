#include <brass/mir/module.hpp>
#include <algorithm>

namespace brass {

Module::Module(std::string_view name)
    : arena_(), string_pool_(), name_(string_pool_.intern(name)) {}

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

void Module::add_external_symbol(std::string_view sym) {
    std::string_view interned_sym = string_pool_.intern(sym);
    if (!has_external_symbol(interned_sym)) {
        external_symbols_.push_back(interned_sym);
    }
}

bool Module::has_external_symbol(std::string_view sym) const noexcept {
    return std::find(external_symbols_.begin(), external_symbols_.end(), sym) != external_symbols_.end();
}

} // namespace brass
