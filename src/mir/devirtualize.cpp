#include <brass/mir/devirtualize.hpp>

namespace brass {

bool devirtualize_call(Instruction* inst, Module& mod) {
    if (!inst || inst->opcode() != Opcode::patchable_call) {
        return false;
    }

    std::string_view callee_name = inst->extra_symbol().empty() ? inst->symbol() : inst->extra_symbol();
    if (callee_name.empty()) {
        return false;
    }

    Function* target_fn = mod.get_function(callee_name);
    if (!target_fn) {
        return false;
    }

    // Devirtualize patchable_call into direct call
    inst->set_opcode(Opcode::call);
    inst->set_symbol(mod.string_pool().intern(callee_name));
    inst->set_extra_symbol("");
    return true;
}

bool devirtualize_function(Function& fn, Module& mod) {
    bool changed = false;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::patchable_call) {
                changed |= devirtualize_call(inst, mod);
            }
        }
    }
    return changed;
}

bool devirtualize_module(Module& mod) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        changed |= devirtualize_function(*fn, mod);
    }
    return changed;
}

} // namespace brass
