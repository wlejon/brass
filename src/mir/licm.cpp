// Loop-invariant code motion (scalar_opt.hpp hoist_loop_invariants).
//
// A pure instruction whose operands the loop does not compute moves to the
// preheader, and so does a load that runs on every iteration (its block
// dominates the latches) when nothing in the loop writes what it reads.
//
// What reads where an object is now (gc_remat.hpp GcSensitive: a tagged
// value's bits, an address made of them, a word loaded through one) moves
// the same way when the loop has no GC point: the object can move only at
// the back edge's interrupt poll, and the poll computes such a value again
// after its handler (interrupt_poll.cpp).
//
// A load through an address made of a tagged value's bits, whatever it
// reads, moves only when it is safe to run at the preheader. A front end
// reads an object's words only where a test it made says what the object
// is: a test inside the loop, or a fact that holds everywhere the loop runs
// (a type, or a test before the loop, which dominates the preheader). So
// the load moves only when no test that can run before it in an iteration
// (a guard, br_if or switch on a path from the header, or before it in its
// block) depends on a tagged value its address is made of. The loop's exit
// test counts like any other.
//
// A value made of a tagged value's bits that is not one itself (a tag test,
// a kind compare) moves with copies of the instructions it is made of,
// whatever GC points the loop has: a collection changes no tag or kind.
//
// Nothing moves that a write or GC point of the loop clobbers, even on a
// cold path: computing it again after each such point (a version per path,
// merged at the header) was tried, and the block parameters it adds cost the
// hot path more than the loads they save (register pressure across the
// loop's calls).
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/uses.hpp>
#include "gc_remat.hpp"
#include "ir_clone.hpp"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

// A dependence walk longer than this and the load stays where it is.
constexpr size_t kWalkBudget = 4096;

class Licm {
public:
    Licm(Function& fn, LoopInfo& loop, const DominatorTree& dom, BasicBlock* preheader)
        : fn_(fn), loop_(loop), dom_(dom), preheader_(preheader) {}

    bool run();

private:
    bool invariant(const Value* op) const;
    void scan();
    // Whether a write or GC point of the loop clobbers `inst` hoisted.
    bool clobbered(const Instruction* inst, bool is_load);
    // The tagged values `v` is computed from (through instructions and
    // block parameters, wherever defined); false when the walk is too long.
    bool roots_of(const Value* v, std::vector<const Value*>& roots) const;
    bool tested_before(const Instruction* load);
    // An in-loop address computed from invariant tagged values by pure
    // instructions (no load, nothing hoisted, on the way), and its copy in
    // the preheader.
    bool address_chain(const Value* v);
    Value* copy_chain(Value* v);
    bool try_hoist(BasicBlock* bb, Instruction* cur);

    Function& fn_;
    LoopInfo& loop_;
    const DominatorTree& dom_;
    BasicBlock* preheader_;
    std::optional<AliasAnalysis> aa_;
    GcSensitive sensitive_;

    bool scanned_ = false;
    std::vector<Instruction*> writers_;
    std::vector<Instruction*> gc_points_;
    std::vector<const Instruction*> tests_;
    std::unordered_map<const Instruction*, std::vector<const Value*>> test_roots_;
    std::unordered_set<const Instruction*> test_overflow_;
    std::unordered_map<const BasicBlock*, std::unordered_set<const BasicBlock*>> before_;

    std::unordered_map<const Value*, bool> chain_;
    std::unordered_map<const Value*, Value*> copies_;
};

bool Licm::invariant(const Value* op) const {
    if (!op) return true;
    if (op->is_block_param()) return !loop_.contains(op->defining_block());
    if (op->is_instruction()) {
        const Instruction* def = op->defining_instruction();
        if (def && loop_.contains(def->parent()) && def->parent() != preheader_) return false;
    }
    return true;
}

void Licm::scan() {
    if (scanned_) return;
    scanned_ = true;
    for (BasicBlock* bb : loop_.blocks()) {
        if (!bb || bb == preheader_) continue;
        for (Instruction* inst : *bb) {
            const Opcode wop = inst->opcode();
            if (wop == Opcode::store || wop == Opcode::store_indexed || wop == Opcode::vstore || inst->is_call() ||
                may_trigger_gc(*inst)) {
                writers_.push_back(inst);
            }
            if (may_trigger_gc(*inst)) gc_points_.push_back(inst);
            const Opcode op = inst->opcode();
            if (op == Opcode::guard || op == Opcode::br_if || op == Opcode::switch_) tests_.push_back(inst);
        }
    }
}

bool Licm::clobbered(const Instruction* inst, bool is_load) {
    if (is_load) {
        if (!aa_) aa_.emplace(fn_);
        for (const Instruction* w : writers_) {
            if (aa_->can_clobber(w, inst)) return true;
        }
    }
    return sensitive_(inst->result()) && !gc_points_.empty();
}

bool Licm::roots_of(const Value* v, std::vector<const Value*>& roots) const {
    std::vector<const Value*> work{v};
    std::unordered_set<const Value*> seen;
    while (!work.empty()) {
        const Value* cur = work.back();
        work.pop_back();
        if (!cur || !seen.insert(cur).second) continue;
        if (seen.size() > kWalkBudget) return false;
        if (cur->is_block_param()) {
            const BasicBlock* bb = cur->defining_block();
            if (!bb) continue;
            const uint32_t idx = cur->param_index();
            for (BasicBlock* p : bb->predecessors()) {
                Instruction* term = p ? p->terminator() : nullptr;
                if (!term) continue;
                for_each_edge(*term, [&](BranchTarget& t) {
                    if (t.block == bb && idx < t.args.size()) work.push_back(t.args[idx]);
                });
            }
            continue;
        }
        if (!cur->is_instruction()) continue;
        const Instruction* def = cur->defining_instruction();
        if (!def) continue;
        if (def->opcode() == Opcode::bitcast_i64_tagged) roots.push_back(def->operand(0));
        for (const Value* op : def->operands()) work.push_back(op);
    }
    return true;
}

bool Licm::tested_before(const Instruction* load) {
    std::vector<const Value*> mine;
    for (const Value* op : load->operands()) {
        if (!roots_of(op, mine)) return true;
    }
    if (mine.empty()) return false;
    std::sort(mine.begin(), mine.end());
    auto depends = [&](const Instruction* test) {
        auto it = test_roots_.find(test);
        if (it == test_roots_.end()) {
            std::vector<const Value*> roots;
            bool ok = true;
            for (const Value* op : test->operands()) ok = ok && roots_of(op, roots);
            if (!ok) test_overflow_.insert(test);
            it = test_roots_.emplace(test, std::move(roots)).first;
        }
        if (test_overflow_.count(test)) return true;
        for (const Value* r : it->second) {
            if (std::binary_search(mine.begin(), mine.end(), r)) return true;
        }
        return false;
    };
    // The blocks an iteration can pass before it reaches the load's.
    const BasicBlock* home = load->parent();
    auto found = before_.find(home);
    if (found == before_.end()) {
        std::unordered_set<const BasicBlock*> reach;
        std::vector<const BasicBlock*> work;
        if (home != loop_.header()) {
            for (const BasicBlock* p : home->predecessors()) work.push_back(p);
        }
        while (!work.empty()) {
            const BasicBlock* b = work.back();
            work.pop_back();
            if (!b || !loop_.contains(b) || b == preheader_ || !reach.insert(b).second) continue;
            if (b == loop_.header()) continue;
            for (const BasicBlock* p : b->predecessors()) work.push_back(p);
        }
        found = before_.emplace(home, std::move(reach)).first;
    }
    for (const Instruction* test : tests_) {
        if (found->second.count(test->parent()) && depends(test)) return true;
    }
    for (const Instruction* i = home->head(); i && i != load; i = i->next()) {
        const Opcode op = i->opcode();
        if ((op == Opcode::guard || op == Opcode::br_if || op == Opcode::switch_) && depends(i)) return true;
    }
    return false;
}

bool Licm::address_chain(const Value* v) {
    if (!v || invariant(v) || !v->is_instruction()) return false;
    if (const auto it = chain_.find(v); it != chain_.end()) return it->second;
    chain_[v] = false;  // a cycle is impossible outside block parameters
    const Instruction* d = v->defining_instruction();
    bool ok = (d->opcode() == Opcode::bitcast_i64_tagged || is_pure_instruction(d)) && d->produces_value() &&
              sensitive_.clonable(v);
    for (const Value* op : d->operands()) ok = ok && ((invariant(op) && !sensitive_(op)) || address_chain(op));
    return chain_[v] = ok;
}

Value* Licm::copy_chain(Value* v) {
    if (invariant(v)) return v;
    if (const auto it = copies_.find(v); it != copies_.end()) return it->second;
    const Instruction* d = v->defining_instruction();
    std::vector<Value*> ops;
    for (Value* op : d->operands()) ops.push_back(copy_chain(op));
    ir::ValueMap map;
    Instruction* copy = ir::clone_shell(fn_, *d, map);
    for (Value* op : ops) copy->add_operand(op);
    preheader_->insert_before(copy, preheader_->terminator());
    copies_[v] = copy->result();
    return copy->result();
}

bool Licm::try_hoist(BasicBlock* bb, Instruction* cur) {
    Instruction* ph_term = preheader_->terminator();
    const bool is_load = cur->opcode() == Opcode::load;
    const bool is_untag = cur->opcode() == Opcode::bitcast_i64_tagged;
    if (!cur->produces_value() || !(is_load || is_untag || is_pure_instruction(cur))) return false;
    const bool gc_sensitive = sensitive_(cur->result());
    // Only what copies of instructions compute again moves (gc_remat.hpp).
    if (gc_sensitive && !sensitive_.clonable(cur->result())) return false;
    // An address made of invariant tagged values stays: it costs a register
    // across the loop and little to compute, and a load through it, or a
    // value made of it that is no address, moves with a copy of it.
    if (!is_load && address_chain(cur->result())) return false;
    for (Value* op : cur->operands()) {
        if (!invariant(op) && !((is_load || !gc_sensitive) && address_chain(op))) return false;
    }
    if (is_load) {
        if (is_derived_gcref(cur->result())) return false;
        const Value* base = cur->operand(0);
        bool dominates_latches = (bb == loop_.header()) || !loop_.latches().empty();
        if (bb != loop_.header()) {
            for (BasicBlock* latch : loop_.latches()) {
                if (!latch || !dom_.dominates(bb, latch)) {
                    dominates_latches = false;
                    break;
                }
            }
        }
        if (!dominates_latches) return false;
        // A compare of the base before the load: a null test it may stand
        // behind.
        auto compared = [&](const BasicBlock* check_bb, const Instruction* stop_at) {
            for (const Instruction* inst : *check_bb) {
                if (inst == stop_at) break;
                if ((inst->opcode() == Opcode::eq || inst->opcode() == Opcode::ne) &&
                    (inst->operand(0) == base || inst->operand(1) == base)) {
                    return true;
                }
            }
            return false;
        };
        if ((bb != loop_.header() && compared(loop_.header(), nullptr)) || compared(bb, cur)) return false;
    }
    // A load through an address made of a tagged value's bits, whatever it
    // reads (a kind, a length, an address).
    const bool tagged_address = is_load && sensitive_(cur->operand(0));
    if (is_load || gc_sensitive) {
        scan();
        if (tagged_address && tested_before(cur)) return false;
        if (clobbered(cur, is_load)) return false;
    }

    if (is_load) {
        // The same load in the preheader, or above it when nothing between
        // writes what it reads, answers for it.
        if (!aa_) aa_.emplace(fn_);
        auto find_existing = [&](Instruction* from_inst) -> Value* {
            for (Instruction* p = from_inst; p != nullptr; p = p->prev()) {
                if (p->opcode() == Opcode::load && p->operand(0) == cur->operand(0) && p->offset() == cur->offset() &&
                    p->memory_type() == cur->memory_type() && p->type() == cur->type()) {
                    return p->result();
                }
                if (aa_->can_clobber(p, cur)) return nullptr;
            }
            return nullptr;
        };
        Value* existing = find_existing(ph_term->prev());
        if (!existing && preheader_->predecessors().size() == 1) {
            BasicBlock* ppred = preheader_->predecessors()[0];
            bool ph_clobbers = false;
            for (Instruction* p = ph_term->prev(); p != nullptr; p = p->prev()) {
                if (aa_->can_clobber(p, cur)) {
                    ph_clobbers = true;
                    break;
                }
            }
            if (!ph_clobbers && ppred && ppred->terminator()) existing = find_existing(ppred->terminator()->prev());
        }
        if (existing) {
            replace_all_uses(fn_, cur->result(), existing);
            bb->remove_instruction(cur);
            return true;
        }
    }
    for (size_t i = 0; i < cur->operand_count(); ++i) {
        Value* op = cur->operand(i);
        if (!invariant(op)) cur->set_operand(i, copy_chain(op));
    }
    bb->remove_instruction(cur);
    preheader_->insert_before(cur, ph_term);
    // What it made invariant may complete a chain found broken before.
    chain_.clear();
    return true;
}

bool Licm::run() {
    if (!preheader_->terminator()) return false;
    bool any = false;
    for (bool round = true; round;) {
        round = false;
        for (BasicBlock* bb : loop_.blocks()) {
            if (!bb || bb == preheader_) continue;
            for (Instruction* cur = bb->head(); cur;) {
                Instruction* next = cur->next();
                if (try_hoist(bb, cur)) round = any = true;
                cur = next;
            }
        }
    }
    return any;
}

bool licm_loop(Function& fn, LoopInfo& loop, const DominatorTree& dom) {
    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;
    return Licm(fn, loop, dom, preheader).run();
}

} // namespace

bool hoist_loop_invariants(Function& fn) {
    if (!fn.entry_block()) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    bool changed = false;
    for (LoopInfo* loop : loops.post_order_loops()) {
        if (loop) changed |= licm_loop(fn, *loop, dom);
    }
    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass
