#include <brass/mir/write_barrier_elim.hpp>
#include <brass/gc/gc_limits.hpp>
#include <brass/mir/gc_refs.hpp>
#include <iostream>
#include <set>
#include <unordered_set>
#include <utility>

namespace brass {

namespace {

// Depth bound for the operand walk that proves a value carries no pointer.
// Past it the value is assumed to be a pointer, which keeps the barrier.
constexpr int kMaxNonPointerDepth = 8;

bool constant_integer(const Value* val, int64_t& out) noexcept {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i64) {
        out = def->imm_i64();
        return true;
    }
    if (def->opcode() == Opcode::iconst_i32) {
        out = def->imm_i32();
        return true;
    }
    return false;
}

bool non_pointer_value(const Value* val, int depth) noexcept {
    if (!val) return true;
    const Type t = val->type();
    if (t.is_gc_root() || t.is_pointer()) return false;
    // Floats, vectors, 32-bit and narrower integers never hold a 64-bit
    // heap address, and a comparison result is 0 or 1.
    if (t.is_float() || t.is_vector() || t.is_void()) return true;
    if (t.kind() != TypeKind::I64) return true;

    // An i64 block parameter or function argument may carry a (possibly
    // NaN-boxed) reference: nothing is known about it.
    if (!val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;

    const Opcode op = def->opcode();
    if (op == Opcode::iconst_i64 || op == Opcode::iconst_i32) return true;
    if (is_comparison(op)) return true;
    switch (op) {
        case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
        case Opcode::sext_i64: case Opcode::zext_i64:
        case Opcode::fptosi_i64: case Opcode::fptosi_i64_f32:
        case Opcode::bitcast_f64_i64:
            // Counts, widened 32-bit values and converted floats are numbers
            // by construction; a boxed double is not a heap address.
            return true;
        default:
            break;
    }
    if (depth >= kMaxNonPointerDepth) return false;
    // Arithmetic and bit operations can build an address out of a pointer
    // operand (tagging, untagging, offsetting), so the result is a number
    // only when every input is.
    if (is_arithmetic(op) || is_bitwise(op) || op == Opcode::select) {
        const size_t first = op == Opcode::select ? 1 : 0;
        for (size_t i = first; i < def->operand_count(); ++i) {
            if (!non_pointer_value(def->operand(i), depth + 1)) return false;
        }
        return true;
    }
    return false;
}

} // namespace

bool WriteBarrierElimination::is_non_pointer_value(const Value* val) const noexcept {
    return non_pointer_value(val, 0);
}

bool WriteBarrierElimination::is_allocation_inst(const Instruction* inst) const noexcept {
    if (!inst || inst->opcode() != Opcode::call) return false;
    // Only brass's own allocator has a known placement policy. A request
    // larger than half the nursery goes straight to tenured space, so the
    // result is young only for a constant size below the guaranteed bound.
    if (inst->symbol() != "brass_gc_alloc" || inst->operand_count() < 1) return false;
    int64_t size = 0;
    if (!constant_integer(inst->operand(0), size)) return false;
    return size >= 0 && static_cast<uint64_t>(size) <= kMaxAlwaysYoungPayloadBytes;
}

bool WriteBarrierElimination::run_on_function(Function& fn) {
    bool changed = false;
    std::vector<Instruction*> to_remove;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        // The runtime barrier marks the card of `obj` only when `obj` is old
        // and the stored value is young, so a barrier is redundant only
        // after one with the same object *and* the same value: an earlier
        // barrier storing an old value leaves the card clean.
        std::set<std::pair<const Value*, const Value*>> barriered;
        std::unordered_set<const Value*> young_in_block;

        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            // A collection may promote young objects and clean cards, so
            // facts about ages and marked cards die at a GC point.
            if (may_trigger_gc(*inst)) {
                barriered.clear();
                young_in_block.clear();
            }

            if (is_allocation_inst(inst) && inst->result()) {
                young_in_block.insert(inst->result());
            }

            if (inst->opcode() != Opcode::write_barrier) continue;

            stats_.total_barriers++;
            const Value* obj = inst->operand(0);
            const Value* val = inst->operand(1);

            bool eliminate = false;
            if (is_non_pointer_value(val)) {
                // Rule 1: the stored value cannot be a young reference.
                stats_.eliminated_non_pointer++;
                eliminate = true;
            } else if (obj && young_in_block.count(obj) > 0) {
                // Rule 2: a young object needs no card mark, and nothing
                // since its allocation could have promoted it.
                stats_.eliminated_young_provenance++;
                eliminate = true;
            } else if (obj && barriered.count({obj, val}) > 0) {
                // Rule 3: an identical barrier already ran with no collection
                // in between, so it left the card exactly as this one would.
                stats_.eliminated_redundant++;
                eliminate = true;
            }

            if (eliminate) {
                to_remove.push_back(inst);
                changed = true;
            } else {
                stats_.remaining_barriers++;
                if (obj) barriered.insert({obj, val});
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
