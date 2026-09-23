#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/uses.hpp>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace brass {

std::string ArrayContractionStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Array Contraction Statistics ===\n"
       << "  Arrays contracted: " << arrays_contracted << "\n"
       << "  Loads eliminated: " << loads_eliminated << "\n"
       << "  Stores eliminated: " << stores_eliminated << "\n"
       << "  Allocations eliminated: " << allocations_eliminated << "\n";
    return ss.str();
}

namespace {

// Two kinds of buffer are contracted. A raw buffer (malloc, the GC
// allocator) is read and written with load_indexed / store_indexed, and a
// load returns the bytes the last store to the same address wrote. A runtime
// array (a declared `array_new`) is read and written through the runtime's
// `array_get` / `array_set` calls, and a read returns the last value written
// at the same index only while that index is a plain in-range element index:
// the runtime drops writes at negative indices and truncates huge ones, so
// any other index could read something else.
enum class BufferKind { Raw, RuntimeArray };

bool is_elem_set(const Instruction* inst) {
    return callee_has_role(*inst, SymbolRole::ArraySet) && inst->operand_count() >= 3;
}

bool is_elem_get(const Instruction* inst) {
    return callee_has_role(*inst, SymbolRole::ArrayGet) && inst->operand_count() == 2;
}

struct Access {
    Instruction* inst = nullptr;
    bool is_store = false;
};

// Every use of the buffer, if every use is one contraction can account for:
// element reads and writes inside the loop that address the buffer itself
// (never store it or pass it on), and write barriers on it. Any other use —
// another call, a branch argument, a select, a return, deopt state — lets the
// buffer be observed outside the accesses being contracted.
bool collect_accesses(const Function& fn, const LoopInfo& loop, const Value* buf, BufferKind kind,
                      std::vector<Access>& accesses, std::vector<Instruction*>& barriers) {
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            bool uses_as_address = inst->operand_count() > 0 && inst->operand(0) == buf;
            for (size_t i = 1; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == buf) return false;
            }
            auto passes = [&](const BranchTarget& bt) {
                for (const Value* a : bt.args) if (a == buf) return true;
                return false;
            };
            if (passes(inst->branch_target()) || passes(inst->true_target()) || passes(inst->false_target())) return false;
            for (const auto& sc : inst->switch_cases()) if (passes(sc.target)) return false;
            for (const Value* sv : inst->state_map()) if (sv == buf) return false;
            if (!uses_as_address) continue;

            const Opcode op = inst->opcode();
            if (op == Opcode::write_barrier) {
                barriers.push_back(inst);
                continue;
            }
            if (!loop.contains(bb)) return false;
            if (kind == BufferKind::Raw && (op == Opcode::store_indexed || op == Opcode::load_indexed)) {
                accesses.push_back({inst, op == Opcode::store_indexed});
            } else if (kind == BufferKind::RuntimeArray && (is_elem_set(inst) || is_elem_get(inst))) {
                accesses.push_back({inst, is_elem_set(inst)});
            } else {
                return false;
            }
        }
    }
    return true;
}

bool index_is_plain_element(const Value* idx, const BasicBlock* bb, const RangeAnalysis& ra) {
    if (!idx || (idx->type() != Type::i64() && idx->type() != Type::i32())) return false;
    ValueRange r = ra.get_range_at(idx, bb);
    return !r.is_empty() && r.min_val >= 0 && r.max_val < INT32_MAX;
}

// Whether `store` writes exactly what `load` then reads.
bool same_location(const Instruction* store, const Instruction* load, BufferKind kind) {
    if (store->operand(1) != load->operand(1)) return false;
    if (kind == BufferKind::Raw) {
        return store->scale() == load->scale() && store->offset() == load->offset() &&
               store->memory_type() == load->memory_type() && store->operand(2)->type() == load->type();
    }
    return store->operand(2)->type() == load->type();
}

// A load whose result is immediately unboxed can take the unboxed value
// directly when the forwarded value is the matching box.
void forward_through_unbox(Function& fn, Value* loaded, Value* stored) {
    if (!stored || !stored->is_instruction()) return;
    const Instruction* sdef = stored->defining_instruction();
    if (!sdef || sdef->operand_count() < 1) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || inst->operand_count() < 1 || inst->operand(0) != loaded || !inst->result()) continue;
            const bool unbox_call = inst->opcode() == Opcode::call && inst->symbol() == "unbox_f64" &&
                                    sdef->opcode() == Opcode::call && sdef->symbol() == "box_f64";
            const bool bit_roundtrip = inst->opcode() == Opcode::bitcast_f64_i64 &&
                                       sdef->opcode() == Opcode::bitcast_i64_f64;
            if ((unbox_call || bit_roundtrip) && sdef->operand(0)->type() == inst->type()) {
                replace_all_uses(fn, inst->result(), sdef->operand(0));
            }
        }
    }
}

bool contract_buffer(Function& fn, const LoopInfo& loop, Instruction* alloc_inst, BufferKind kind,
                     const RangeAnalysis& ra, const ArrayContractionOptions& options) {
    Value* buf = alloc_inst->result();
    if (!buf) return false;

    std::vector<Access> accesses;
    std::vector<Instruction*> barriers;
    if (!collect_accesses(fn, loop, buf, kind, accesses, barriers)) return false;

    bool has_store = false;
    bool has_load = false;
    for (const Access& a : accesses) {
        (a.is_store ? has_store : has_load) = true;
        if (kind == BufferKind::RuntimeArray && !index_is_plain_element(a.inst->operand(1), a.inst->parent(), ra)) {
            return false;
        }
    }
    if (!has_store || !has_load) return false;

    // Every read must be answered by the last write to the buffer before it
    // in its own block; a write in between at an index not proven equal
    // could have replaced the value.
    std::unordered_set<const Instruction*> buffer_accesses;
    for (const Access& a : accesses) buffer_accesses.insert(a.inst);
    std::vector<std::pair<Instruction*, Value*>> forwards;
    for (const Access& a : accesses) {
        if (a.is_store) continue;
        Instruction* last_store = nullptr;
        for (Instruction* prev = a.inst->prev(); prev; prev = prev->prev()) {
            if (buffer_accesses.count(prev) && (prev->opcode() == Opcode::store_indexed || is_elem_set(prev))) {
                last_store = prev;
                break;
            }
        }
        if (!last_store || !same_location(last_store, a.inst, kind)) return false;
        forwards.emplace_back(a.inst, last_store->operand(2));
    }

    for (auto& [load, value] : forwards) {
        Value* loaded = load->result();
        if (loaded) {
            forward_through_unbox(fn, loaded, value);
            replace_all_uses(fn, loaded, value);
        }
        load->parent()->remove_instruction(load);
        if (options.stats) options.stats->loads_eliminated++;
    }
    for (const Access& a : accesses) {
        if (!a.is_store) continue;
        a.inst->parent()->remove_instruction(a.inst);
        if (options.stats) options.stats->stores_eliminated++;
    }
    for (Instruction* wb : barriers) wb->parent()->remove_instruction(wb);
    alloc_inst->parent()->remove_instruction(alloc_inst);
    if (options.stats) {
        options.stats->allocations_eliminated++;
        options.stats->arrays_contracted++;
    }
    return true;
}

bool contract_in_loop(Function& fn, const LoopInfo& loop, const RangeAnalysis& ra,
                      const ArrayContractionOptions& options) {
    std::vector<std::pair<Instruction*, BufferKind>> candidates;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || inst->opcode() != Opcode::call || !inst->result()) continue;
            if (callee_has_role(*inst, SymbolRole::ArrayNew)) {
                candidates.emplace_back(inst, BufferKind::RuntimeArray);
            } else if (is_allocation_call(inst)) {
                candidates.emplace_back(inst, BufferKind::Raw);
            }
        }
    }

    bool any = false;
    for (auto& [alloc_inst, kind] : candidates) {
        any |= contract_buffer(fn, loop, alloc_inst, kind, ra, options);
    }
    return any;
}

} // namespace

bool contract_arrays_in_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const ArrayContractionOptions& options
) {
    (void)dom;
    RangeAnalysis ra(fn);
    return contract_in_loop(fn, loop, ra, options);
}

bool array_contraction_pass(
    Function& fn,
    const DominatorTree& dom,
    const ArrayContractionOptions& options
) {
    bool changed = false;

    if (options.fuse_loops_first) {
        LoopFusionOptions fusion_opts;
        changed |= loop_fusion_pass(fn, dom, fusion_opts);
    }

    fn.rebuild_cfg_predecessors();
    DominatorTree current_dom(fn);
    LoopAnalysis la(fn, current_dom);
    RangeAnalysis ra(fn, current_dom, la);

    for (LoopInfo* loop : la.post_order_loops()) {
        if (!loop) continue;
        if (contract_in_loop(fn, *loop, ra, options)) {
            changed = true;
        }
    }

    return changed;
}

} // namespace brass
