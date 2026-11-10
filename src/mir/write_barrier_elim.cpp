#include <brass/mir/write_barrier_elim.hpp>
#include <brass/gc/gc_limits.hpp>
#include <brass/gc/object.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/tagged_alloc.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace brass {

namespace {

// Depth bound for the operand walk that proves a value carries no pointer.
// Past it the value is assumed to be a pointer, which keeps the barrier.
constexpr int kMaxNonPointerDepth = 8;
// Bound on the block parameters one proof may cross.
constexpr size_t kMaxParamWalk = 64;

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

// Whether `val` provably never holds a heap address. `params` (non-null)
// lets the walk cross block parameters: a parameter is a number when every
// value an edge passes it is, the parameters on the walk's path assumed so
// (a cycle of parameters that only pass each other numbers carries none).
// Null: a block parameter or argument is assumed to be a pointer, the rule
// the proof certificates check (verifier_certs.cpp).
bool non_pointer_value(const Value* val, int depth, std::unordered_set<const Value*>* params,
                       const std::unordered_map<const Value*, std::vector<const Value*>>* incoming) noexcept {
    if (!val) return true;
    const Type t = val->type();
    if (t.is_gc_root() || t.is_pointer()) return false;
    // Floats, vectors, 32-bit and narrower integers never hold a 64-bit
    // heap address, and a comparison result is 0 or 1.
    if (t.is_float() || t.is_vector() || t.is_void()) return true;
    if (t.kind() != TypeKind::I64) return true;

    if (!val->is_instruction()) {
        // An i64 function argument may carry a (possibly NaN-boxed)
        // reference; a block parameter is what its edges pass it.
        if (!params || !incoming) return false;
        auto it = incoming->find(val);
        if (it == incoming->end()) return false;
        if (params->count(val)) return true;
        if (params->size() >= kMaxParamWalk) return false;
        params->insert(val);
        for (const Value* in : it->second) {
            if (!non_pointer_value(in, depth + 1, params, incoming)) return false;
        }
        return true;
    }
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
            if (!non_pointer_value(def->operand(i), depth + 1, params, incoming)) return false;
        }
        return true;
    }
    return false;
}

// The value `v` is an address inside, or a tagged reference to, when it is
// computed from it by adding a constant, masking with a mask that keeps the
// low 48 address bits (gc::kAddressMask), or OR-ing in a constant above
// them (a tag): stripped of those steps, the value it was made from, and
// the constant byte offset added (in `offset`).
const Value* reference_base(const Value* v, int64_t& offset) noexcept {
    offset = 0;
    for (int steps = 0; steps < 8 && v && v->is_instruction(); ++steps) {
        const Instruction* def = v->defining_instruction();
        // A tagged value's bits, or bits made a tagged value: the same word.
        if (def && (def->opcode() == Opcode::bitcast_i64_tagged || def->opcode() == Opcode::bitcast_tagged_i64)) {
            v = def->operand(0);
            continue;
        }
        if (!def || def->operand_count() != 2) break;
        int64_t c = 0;
        const Opcode op = def->opcode();
        if (op == Opcode::add && constant_integer(def->operand(1), c)) {
            offset += c;
            v = def->operand(0);
        } else if (op == Opcode::add && constant_integer(def->operand(0), c)) {
            offset += c;
            v = def->operand(1);
        } else if (op == Opcode::and_ && constant_integer(def->operand(1), c) &&
                   (static_cast<uint64_t>(c) & gc::kAddressMask) == gc::kAddressMask) {
            v = def->operand(0);
        } else if ((op == Opcode::or_ || op == Opcode::xor_) && constant_integer(def->operand(1), c) &&
                   (static_cast<uint64_t>(c) & gc::kAddressMask) == 0) {
            // A tag put on (or) or taken off (xor) above the address bits.
            v = def->operand(0);
        } else {
            break;
        }
    }
    return v;
}

using Pair = std::pair<const Value*, const Value*>;

// What holds at a program point since the last GC point on every path to it:
// the young objects allocated (by value) and the barriers run (object,
// value). `top` is the unvisited state, which every fact holds in.
struct Facts {
    bool top = true;
    std::vector<const Value*> young;  // sorted
    std::vector<Pair> barriered;      // sorted

    void clear() {
        top = false;
        young.clear();
        barriered.clear();
    }
    void meet(const Facts& other) {
        if (other.top) return;
        if (top) {
            *this = other;
            return;
        }
        std::vector<const Value*> y;
        std::set_intersection(young.begin(), young.end(), other.young.begin(), other.young.end(),
                              std::back_inserter(y));
        std::vector<Pair> p;
        std::set_intersection(barriered.begin(), barriered.end(), other.barriered.begin(),
                              other.barriered.end(), std::back_inserter(p));
        young = std::move(y);
        barriered = std::move(p);
    }
    bool operator==(const Facts& o) const {
        return top == o.top && young == o.young && barriered == o.barriered;
    }
    void add_young(const Value* v) {
        auto it = std::lower_bound(young.begin(), young.end(), v);
        if (it == young.end() || *it != v) young.insert(it, v);
    }
    bool has_young(const Value* v) const { return std::binary_search(young.begin(), young.end(), v); }
    void add_barrier(const Pair& p) {
        auto it = std::lower_bound(barriered.begin(), barriered.end(), p);
        if (it == barriered.end() || *it != p) barriered.insert(it, p);
    }
    bool has_barrier(const Pair& p) const { return std::binary_search(barriered.begin(), barriered.end(), p); }
};

} // namespace

bool WriteBarrierElimination::is_non_pointer_value(const Value* val) const noexcept {
    return non_pointer_value(val, 0, nullptr, nullptr);
}

bool WriteBarrierElimination::is_allocation_inst(const Instruction* inst) const noexcept {
    if (!inst || inst->opcode() != Opcode::call) return false;
    // The module's tagged allocator promises the same placement
    // (Module::tagged_allocator, gc_contract.md 2.13).
    if (is_tagged_allocation(inst)) {
        const int64_t size = tagged_allocation_bytes(inst);
        return size >= 0 && static_cast<uint64_t>(size) <= kMaxAlwaysYoungPayloadBytes;
    }
    // Only brass's own allocator has a known placement policy. A request
    // larger than half the nursery goes straight to tenured space, so the
    // result is young only for a constant size below the guaranteed bound.
    if (inst->symbol() != "brass_gc_alloc" || inst->operand_count() < 1) return false;
    int64_t size = 0;
    if (!constant_integer(inst->operand(0), size)) return false;
    return size >= 0 && static_cast<uint64_t>(size) <= kMaxAlwaysYoungPayloadBytes;
}

int64_t WriteBarrierElimination::young_allocation_bytes(const Instruction* inst) const noexcept {
    if (!is_allocation_inst(inst)) return -1;
    int64_t size = 0;
    constant_integer(inst->operand(0), size);
    return size;
}


bool WriteBarrierElimination::run_on_function(Function& fn) {
    const std::vector<BasicBlock*>& blocks = fn.blocks();
    if (blocks.empty()) return false;

    // The values each block parameter is passed, and each block's
    // predecessors, from the terminators.
    std::unordered_map<const Value*, std::vector<const Value*>> incoming;
    std::unordered_map<const BasicBlock*, std::vector<const BasicBlock*>> preds;
    bool has_barrier = false;
    for (BasicBlock* bb : blocks) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::write_barrier) has_barrier = true;
        }
        const Instruction* term = bb->terminator();
        if (!term) continue;
        for_each_edge(*term, [&](const BranchTarget& t) {
            if (!t.block) return;
            preds[t.block].push_back(bb);
            for (size_t i = 0; i < t.args.size() && i < t.block->param_count(); ++i) {
                incoming[t.block->param(i)].push_back(t.args[i]);
            }
        });
    }
    if (!has_barrier) return false;
    // A block an edge may enter with values no terminator names (the entry,
    // a resume target, a landing pad) knows nothing about its parameters.
    for (const BasicBlock* bb : blocks) {
        if (!bb) continue;
        const bool landing = bb->head() && bb->head()->opcode() == Opcode::landing_pad;
        if (bb == fn.entry_block() || landing) {
            for (const Value* p : bb->params()) incoming.erase(p);
        }
    }
    for (const auto& rp : fn.resume_points()) {
        if (rp.second) {
            for (const Value* p : rp.second->params()) incoming.erase(p);
        }
    }

    // The facts entering each block: the meet of its predecessors' exits,
    // nothing at the entry and at a block entered from outside the CFG.
    std::unordered_map<const BasicBlock*, Facts> out;
    auto transfer = [&](const BasicBlock* bb, Facts f) {
        for (const Instruction* inst : *bb) {
            // A collection may promote young objects and clean cards, so
            // facts about ages and marked cards die at a GC point.
            if (may_trigger_gc(*inst)) f.clear();
            if (is_allocation_inst(inst) && inst->result()) f.add_young(inst->result());
            if (inst->opcode() == Opcode::write_barrier) f.add_barrier({inst->operand(0), inst->operand(1)});
        }
        return f;
    };
    auto entering = [&](const BasicBlock* bb) {
        Facts in;
        const bool external = bb == fn.entry_block() || (bb->head() && bb->head()->opcode() == Opcode::landing_pad);
        bool resumed = false;
        for (const auto& rp : fn.resume_points()) resumed = resumed || rp.second == bb;
        auto pit = preds.find(bb);
        if (external || resumed || pit == preds.end()) {
            in.clear();
            return in;
        }
        for (const BasicBlock* p : pit->second) {
            auto oit = out.find(p);
            if (oit != out.end()) in.meet(oit->second);
        }
        return in;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (const BasicBlock* bb : blocks) {
            if (!bb) continue;
            Facts in = entering(bb);
            if (in.top) continue;
            Facts next = transfer(bb, std::move(in));
            Facts& cur = out[bb];
            if (!(cur == next)) {
                cur = std::move(next);
                changed = true;
            }
        }
    }

    bool changed = false;
    std::vector<Instruction*> to_remove;
    for (BasicBlock* bb : blocks) {
        if (!bb) continue;
        Facts f = entering(bb);
        if (f.top) continue;  // unreachable
        // The facts the proof certificates can check: this block's own,
        // with no block parameter crossed (verifier_certs.cpp).
        std::set<Pair> local_barriered;
        std::unordered_set<const Value*> local_young;

        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (may_trigger_gc(*inst)) {
                f.clear();
                local_barriered.clear();
                local_young.clear();
            }
            if (is_allocation_inst(inst) && inst->result()) {
                f.add_young(inst->result());
                local_young.insert(inst->result());
            }
            if (inst->opcode() != Opcode::write_barrier) continue;

            stats_.total_barriers++;
            const Value* obj = inst->operand(0);
            const Value* val = inst->operand(1);

            // The object the target is an address in or a reference to.
            int64_t offset = 0;
            const Value* base = obj ? reference_base(obj, offset) : nullptr;
            const Instruction* base_def = base && base->is_instruction() ? base->defining_instruction() : nullptr;
            const int64_t bytes = young_allocation_bytes(base_def);
            const bool in_young = base && f.has_young(base) && offset >= 0 && offset < std::max<int64_t>(bytes, 8);

            std::unordered_set<const Value*> params;
            bool eliminate = false;
            bool certified = false;
            BarrierCertKind cert_kind = BarrierCertKind::None;
            if (non_pointer_value(val, 0, &params, &incoming)) {
                // Rule 1: the stored value cannot be a young reference.
                stats_.eliminated_non_pointer++;
                eliminate = true;
                certified = is_non_pointer_value(val);
                int64_t c_imm = 0;
                cert_kind = (constant_integer(val, c_imm) && c_imm == 0)
                    ? BarrierCertKind::ConstantOrNullRef
                    : BarrierCertKind::NonRefStore;
            } else if (in_young) {
                // Rule 2: a young object needs no card mark, and nothing
                // since its allocation could have promoted it.
                stats_.eliminated_young_provenance++;
                eliminate = true;
                certified = obj && local_young.count(obj) > 0;
                cert_kind = BarrierCertKind::YoungGenBumpAlloc;
            } else if (obj && f.has_barrier({obj, val})) {
                // Rule 3: an identical barrier already ran with no collection
                // in between on every path here, so it left the card (and
                // the marker's shading) exactly as this one would.
                stats_.eliminated_redundant++;
                eliminate = true;
                certified = local_barriered.count({obj, val}) > 0;
                cert_kind = BarrierCertKind::RedundantSameEpoch;
            }

            if (eliminate && certified) {
                // Attach the proof certificate to associated store(s) in this GC epoch
                for (Instruction* cur = inst->prev(); cur != nullptr; cur = cur->prev()) {
                    if (may_trigger_gc(*cur)) break;
                    if (cur->opcode() == Opcode::write_barrier &&
                        cur->operand(0) == obj && cur->operand(1) == val) {
                        break;
                    }
                    if ((cur->opcode() == Opcode::store && cur->operand(0) == obj && cur->operand(1) == val) ||
                        (cur->opcode() == Opcode::store_indexed && cur->operand(0) == obj && cur->operand(2) == val)) {
                        if (cur->barrier_cert() == BarrierCertKind::None) {
                            cur->set_barrier_cert(cert_kind);
                        }
                    }
                }
                for (Instruction* cur = inst->next(); cur != nullptr; cur = cur->next()) {
                    if (may_trigger_gc(*cur)) break;
                    if ((cur->opcode() == Opcode::store && cur->operand(0) == obj && cur->operand(1) == val) ||
                        (cur->opcode() == Opcode::store_indexed && cur->operand(0) == obj && cur->operand(2) == val)) {
                        if (cur->barrier_cert() == BarrierCertKind::None) {
                            cur->set_barrier_cert(cert_kind);
                        }
                    }
                }
            }
            if (eliminate) {
                to_remove.push_back(inst);
                changed = true;
            } else {
                stats_.remaining_barriers++;
                if (obj) {
                    f.add_barrier({obj, val});
                    local_barriered.insert({obj, val});
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
       << "  Eliminated (Redundant):            " << stats_.eliminated_redundant << "\n"
       << "  Total Eliminated:                  " << stats_.total_eliminated() << "\n"
       << "  Remaining Active Barriers:         " << stats_.remaining_barriers << "\n";
}

} // namespace brass
