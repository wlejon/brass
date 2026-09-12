#include <brass/mir/write_barrier_elim.hpp>
#include <unordered_set>
#include <iostream>

namespace brass {

bool WriteBarrierElimination::is_non_pointer_value(const Value* val) const noexcept {
    if (!val) return true;

    Type t = val->type();
    if (t.is_float() || t.is_vector() || t.is_void()) {
        return true;
    }

    if (t.kind() == TypeKind::I32) {
        return true; // 32-bit integers are never pointers
    }

    if (val->is_instruction()) {
        const Instruction* def = val->defining_instruction();
        if (!def) return false;

        Opcode op = def->opcode();
        if (op == Opcode::iconst_i32 || op == Opcode::fconst_f64) {
            return true;
        }
        if (op == Opcode::iconst_i64) {
            // Immediate integer constants, including 0 (null) or numbers
            return true;
        }
        if (op == Opcode::bitcast_i64_f64) {
            return true; // Floats cast to i64 (like NaN-boxed floats)
        }
        if (is_arithmetic(op) || is_bitwise(op) || is_comparison(op)) {
            return true;
        }
        if (op == Opcode::trunc_i32 || op == Opcode::fptosi_i32 || op == Opcode::fptosi_i64) {
            return true;
        }
    }

    return false;
}

bool WriteBarrierElimination::is_allocation_inst(const Instruction* inst) const noexcept {
    if (!inst) return false;

    if (inst->is_call()) {
        std::string_view callee = inst->symbol();
        if (callee == "brass_gc_alloc" ||
            callee == "bronze_create_object" ||
            callee == "bronze_create_array" ||
            callee == "bronze_env_create" ||
            callee == "bronze_create_func" ||
            callee == "bronze_create_async_machine") {
            return true;
        }
    }

    return false;
}

bool WriteBarrierElimination::run_on_function(Function& fn) {
    bool changed = false;
    std::vector<Instruction*> to_remove;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        std::unordered_set<const Value*> dirtied_in_block;
        std::unordered_set<const Value*> young_in_block;

        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if ((inst->is_call() && inst->symbol() != "bronze_tls_block_addr") || inst->opcode() == Opcode::safepoint) {
                // Calls or safepoints could trigger GC and clean cards or promote objects
                dirtied_in_block.clear();
                young_in_block.clear();
            }

            if (is_allocation_inst(inst)) {
                if (inst->result()) {
                    young_in_block.insert(inst->result());
                }
            }

            // Check if young object escapes via store
            if (inst->opcode() == Opcode::store) {
                const Value* val_stored = inst->operand(1);
                if (val_stored) young_in_block.erase(val_stored);
            } else if (inst->opcode() == Opcode::store_indexed) {
                const Value* val_stored = inst->operand(2);
                if (val_stored) young_in_block.erase(val_stored);
            }

            if (inst->opcode() == Opcode::write_barrier) {
                stats_.total_barriers++;
                const Value* obj = inst->operand(0);
                const Value* val = inst->operand(1);

                bool eliminate = false;

                // Rule 1: val is known non-pointer, immediate, int, float, null
                if (is_non_pointer_value(val)) {
                    stats_.eliminated_non_pointer++;
                    eliminate = true;
                }
                // Rule 2: obj is newly allocated within current function and has not escaped
                else if (obj && young_in_block.count(obj) > 0) {
                    stats_.eliminated_young_provenance++;
                    eliminate = true;
                }
                // Rule 3: Redundant barrier on obj already dirtied in current block without intervening call/safepoint
                else if (obj && dirtied_in_block.count(obj) > 0) {
                    stats_.eliminated_redundant++;
                    eliminate = true;
                }

                if (eliminate) {
                    to_remove.push_back(inst);
                    changed = true;
                } else {
                    stats_.remaining_barriers++;
                    if (obj) {
                        dirtied_in_block.insert(obj);
                    }
                }
            }
        }
    }

    for (Instruction* inst : to_remove) {
        if (inst && inst->parent()) {
            inst->parent()->remove_instruction(inst);
        }
    }

    return changed;
}

bool WriteBarrierElimination::run_on_module(Module& mod) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= run_on_function(*fn);
        }
    }
    if (dump_stats_) {
        dump_stats(std::cerr);
    }
    return changed;
}

void WriteBarrierElimination::dump_stats(std::ostream& os) const {
    os << "Write Barrier Elimination (WBE) Statistics:\n"
       << "  Total Write Barriers:              " << stats_.total_barriers << "\n"
       << "  Eliminated (Non-Pointer/Immediate):" << stats_.eliminated_non_pointer << "\n"
       << "  Eliminated (Young Provenance):     " << stats_.eliminated_young_provenance << "\n"
       << "  Eliminated (Redundant In Block):   " << stats_.eliminated_redundant << "\n"
       << "  Total Eliminated:                  " << stats_.total_eliminated() << "\n"
       << "  Remaining Active Barriers:         " << stats_.remaining_barriers << "\n";
}

} // namespace brass
