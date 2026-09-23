#include "verifier_gc.hpp"
#include <brass/mir/block.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <unordered_map>

namespace brass {

namespace {

bool derived(const Value* val, int depth) noexcept {
    if (!val || !val->type().is_gcref() || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    switch (def->opcode()) {
        case Opcode::add:
        case Opcode::sub:
            return true;
        case Opcode::select:
            // Chains of selects are short in practice; past the bound the
            // answer errs towards "derived", the restrictive side.
            if (depth > 16) return true;
            return derived(def->operand(1), depth + 1) || derived(def->operand(2), depth + 1);
        default:
            return false;
    }
}

// Operand positions through which a derived gcref would escape into a
// place the GC or another frame reads as an object reference.
bool escapes_through_operand(const Instruction& inst, size_t idx) noexcept {
    switch (inst.opcode()) {
        case Opcode::ret: return true;
        case Opcode::store: return idx == 1;
        case Opcode::store_indexed: return idx == 2;
        case Opcode::write_barrier: return idx == 1;
        default: return may_trigger_gc(inst);
    }
}

} // namespace

bool is_derived_gcref(const Value* val) noexcept {
    return derived(val, 0);
}

bool may_trigger_gc(const Instruction& inst) noexcept {
    const Opcode op = inst.opcode();
    if (op == Opcode::safepoint || is_coro_op(op)) return true;
    if (!is_call(op)) return false;
    // A declared-pure runtime function never allocates.
    return !callee_has_role(inst, SymbolRole::Pure);
}

void verify_derived_gcrefs(
    const Function& fn,
    const std::string& fn_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        // Derived gcrefs defined so far in this block that are still usable:
        // a GC point ends the usability of every one defined before it.
        std::unordered_map<const Value*, bool> usable;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            const std::string where = fn_prefix + "Block '" + std::string(bb->name()) +
                                      "': Instruction '" + std::string(opcode_name(inst->opcode())) + "': ";
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                const Value* op = inst->operand(i);
                if (!is_derived_gcref(op)) continue;
                auto it = usable.find(op);
                if (it == usable.end()) {
                    report_error(where + "derived gcref operand " + std::to_string(i) +
                                 " is defined in another block; derived gcrefs are block-local.");
                } else if (!it->second) {
                    report_error(where + "derived gcref operand " + std::to_string(i) +
                                 " is live across a GC point; recompute it after the GC point.");
                } else if (escapes_through_operand(*inst, i)) {
                    report_error(where + "derived gcref operand " + std::to_string(i) +
                                 " escapes as an object reference.");
                }
            }
            for (const Value* sv : inst->state_map()) {
                if (is_derived_gcref(sv)) {
                    report_error(where + "derived gcref in deopt state; the GC cannot relocate it.");
                }
            }
            bool in_edge = false;
            auto check_edge = [&](const BranchTarget& t) {
                for (const Value* a : t.args) if (is_derived_gcref(a)) in_edge = true;
            };
            check_edge(inst->branch_target());
            check_edge(inst->true_target());
            check_edge(inst->false_target());
            for (const SwitchCase& sc : inst->switch_cases()) check_edge(sc.target);
            if (in_edge) {
                report_error(where + "derived gcref passed as a branch argument; derived gcrefs are block-local.");
            }

            if (may_trigger_gc(*inst)) {
                for (auto& entry : usable) entry.second = false;
            }
            if (inst->result() && is_derived_gcref(inst->result())) {
                usable[inst->result()] = true;
            }
        }
    }
}

} // namespace brass
