#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/module.hpp>

namespace brass {

std::string_view symbol_role_name(SymbolRole role) noexcept {
    switch (role) {
        case SymbolRole::Allocator: return "allocator";
        case SymbolRole::Pure: return "pure";
        case SymbolRole::ArrayNew: return "array_new";
        case SymbolRole::ArrayGet: return "array_get";
        case SymbolRole::ArraySet: return "array_set";
        case SymbolRole::FloatRem: return "frem";
    }
    return "unknown";
}

std::optional<SymbolRole> parse_symbol_role(std::string_view text) noexcept {
    for (SymbolRole role : kAllSymbolRoles) {
        if (symbol_role_name(role) == text) return role;
    }
    return std::nullopt;
}

const Module* module_of(const Instruction& inst) noexcept {
    const BasicBlock* bb = inst.parent();
    const Function* fn = bb ? bb->parent() : nullptr;
    return fn ? fn->parent() : nullptr;
}

bool callee_has_role(const Instruction& inst, SymbolRole role) noexcept {
    // Only a plain direct call: a patchable call's target can be replaced
    // after compilation, and an invoke is a terminator no pass may rewrite
    // as a plain operation.
    if (inst.opcode() != Opcode::call) return false;
    if (inst.symbol().empty()) return false;
    const Module* mod = module_of(inst);
    return mod && mod->has_symbol_role(inst.symbol(), role);
}

} // namespace brass
