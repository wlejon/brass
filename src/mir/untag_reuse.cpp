#include <brass/mir/untag_reuse.hpp>

#include <brass/mir/builder.hpp>
#include <brass/mir/critical_edge.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/uses.hpp>

#include "ir_clone.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

// An address a block holds: `xor(bitcast_i64_tagged tagged, mask)`.
struct Held {
    // The tagged value; or, for a bare `xor(bits, mask)` met before any
    // tagged value was made from `bits`, that i64 (which no read through a
    // tagged value names, so it matches only another bare one).
    const Value* tagged = nullptr;
    int64_t mask = 0;
    // The value holding it; null while it is the parameter `join` would take.
    Value* value = nullptr;
    BasicBlock* join = nullptr;
    // The mask as some read of it had it, for an arm that must compute it.
    Value* mask_value = nullptr;

    // Whether an arm can compute it: only from a tagged value, which names
    // the object wherever a collection put it.
    bool computable() const { return tagged->type() == Type::tagged(); }

    bool same_address(const Held& o) const { return tagged == o.tagged && mask == o.mask; }
};

struct State {
    std::vector<Held> held;
    // Tagged values made by `bitcast_tagged_i64` with no GC point since:
    // each is still the word it was made from.
    std::vector<const Instruction*> fresh;

    void clear() {
        held.clear();
        fresh.clear();
    }
    bool is_fresh(const Instruction* made) const {
        return std::find(fresh.begin(), fresh.end(), made) != fresh.end();
    }
};

Held* find(State& s, const Held& k) {
    for (Held& h : s.held) {
        if (h.same_address(k)) return &h;
    }
    return nullptr;
}

const Instruction* def_of(const Value* v) { return v && v->is_instruction() ? v->defining_instruction() : nullptr; }

bool is_iconst(const Instruction* i) { return i && i->opcode() == Opcode::iconst_i64; }

// `inst` as a read of an address through a tagged value: its bitcast and its
// mask constant.
bool as_read(const Instruction& inst, const Instruction*& bits, const Instruction*& mask) {
    if (inst.opcode() != Opcode::xor_ || inst.operand_count() != 2 || inst.type() != Type::i64()) return false;
    for (size_t i = 0; i < 2; ++i) {
        bits = def_of(inst.operand(i));
        mask = def_of(inst.operand(1 - i));
        if (bits && bits->opcode() == Opcode::bitcast_i64_tagged && bits->operand_count() == 1 && bits->operand(0) &&
            is_iconst(mask)) {
            return true;
        }
    }
    return false;
}

const BasicBlock* block_of(const Value* v) {
    if (v->is_block_param()) return v->defining_block();
    const Instruction* d = def_of(v);
    return d ? d->parent() : nullptr;
}

std::vector<BasicBlock*> reverse_postorder(Function& fn) {
    std::vector<BasicBlock*> post;
    std::unordered_set<const BasicBlock*> seen{fn.entry_block()};
    std::vector<std::pair<BasicBlock*, size_t>> stack{{fn.entry_block(), 0}};
    while (!stack.empty()) {
        BasicBlock* bb = stack.back().first;
        const auto succs = bb->successors();
        size_t& next = stack.back().second;
        if (next == succs.size()) {
            post.push_back(bb);
            stack.pop_back();
            continue;
        }
        BasicBlock* s = succs[next++];
        if (s && seen.insert(s).second) stack.push_back({s, 0});
    }
    std::reverse(post.begin(), post.end());
    return post;
}

class Reuse {
public:
    explicit Reuse(Function& fn) : fn_(fn), dom_(fn) {
        for (const auto& rp : fn.resume_points()) {
            if (rp.second) outside_.insert(rp.second);
        }
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (makes_tagged(*inst)) rooted_.insert(inst->operand(0));
            }
        }
    }

    size_t run() {
        for (BasicBlock* bb : reverse_postorder(fn_)) {
            State s = entry_state(bb);
            for (Instruction* inst : *bb) {
                if (may_trigger_gc(*inst)) {
                    s.clear();
                    continue;
                }
                if (makes_tagged(*inst)) {
                    // Bare reads of the word met so far are reads of this
                    // tagged value from here on.
                    for (Held& h : s.held) {
                        if (h.tagged == inst->operand(0)) h.tagged = inst->result();
                    }
                    s.fresh.push_back(inst);
                    continue;
                }
                Held k;
                if (!read_of(*inst, s, k)) continue;
                if (Held* h = find(s, k)) {
                    resolve(*h);
                    if (!h->value) h->value = parameter(*h);
                    replaced_[inst->result()] = h->value;
                    dead_.push_back(inst);
                } else {
                    s.held.push_back(k);
                }
            }
            out_[bb] = std::move(s);
        }
        if (dead_.empty()) return 0;
        rewrite();
        return dead_.size();
    }

private:
    static bool makes_tagged(const Instruction& inst) {
        return inst.opcode() == Opcode::bitcast_tagged_i64 && inst.operand_count() == 1 && inst.operand(0) &&
               inst.result();
    }

    // `inst` as an address read in state `s`: a read through a tagged value,
    // or a bare `xor(bits, mask)` of an i64 a tagged value is made from. The
    // bare one reads what the tagged value holds while that value is fresh
    // (the front end tests a loaded word and roots it in one stretch), and
    // stands under the i64's own name until then.
    bool read_of(const Instruction& inst, const State& s, Held& k) const {
        const Instruction* bits = nullptr;
        const Instruction* mask = nullptr;
        if (as_read(inst, bits, mask)) {
            k = Held{bits->operand(0), mask->imm_i64(), inst.result(), nullptr, mask->result()};
            return true;
        }
        if (inst.opcode() != Opcode::xor_ || inst.operand_count() != 2 || inst.type() != Type::i64()) return false;
        for (size_t i = 0; i < 2; ++i) {
            const Value* raw = inst.operand(i);
            mask = def_of(inst.operand(1 - i));
            if (!raw || !is_iconst(mask) || rooted_.count(raw) == 0) continue;
            const Value* name = raw;
            for (const Instruction* made : s.fresh) {
                if (made->operand(0) == raw) name = made->result();
            }
            k = Held{name, mask->imm_i64(), inst.result(), nullptr, mask->result()};
            return true;
        }
        return false;
    }

    bool plain_branch(const BasicBlock* bb) const {
        const Instruction* t = bb->terminator();
        return t && t->opcode() == Opcode::br && t->branch_target().block;
    }

    State entry_state(BasicBlock* bb) {
        const auto& preds = bb->predecessors();
        if (preds.empty() || bb == fn_.entry_block() || outside_.count(bb) != 0) return {};
        if (bb->head() && bb->head()->opcode() == Opcode::landing_pad) return {};
        std::vector<State*> ends;
        for (const BasicBlock* p : preds) {
            const auto it = out_.find(p);
            if (it == out_.end()) return {};  // a back edge
            ends.push_back(&it->second);
        }
        if (preds.size() == 1) return *ends[0];
        for (const BasicBlock* p : preds) {
            if (!plain_branch(p)) return {};
        }
        State s;
        for (const Instruction* made : ends[0]->fresh) {
            bool all = true;
            for (size_t j = 1; j < preds.size(); ++j) all = all && ends[j]->is_fresh(made);
            if (all) s.fresh.push_back(made);
        }
        for (size_t i = 0; i < preds.size(); ++i) {
            for (Held& first : ends[i]->held) {
                if (find(s, first)) continue;
                resolve(first);
                const Held h = first;
                bool same = true;
                bool computable = true;
                for (size_t j = 0; j < preds.size(); ++j) {
                    Held* o = find(*ends[j], h);
                    if (o) {
                        resolve(*o);
                        same = same && o->value == first.value && o->join == first.join;
                    } else {
                        same = false;
                        const BasicBlock* def = h.computable() ? block_of(h.tagged) : nullptr;
                        computable = computable && def && dom_.dominates(def, preds[j]);
                    }
                }
                if (same) {
                    s.held.push_back(h);
                } else if (computable) {
                    s.held.push_back(Held{h.tagged, h.mask, nullptr, bb, h.mask_value});
                }
            }
        }
        return s;
    }

    // A parameter made since `h` was copied here is what it names.
    void resolve(Held& h) const {
        if (h.value) return;
        const auto it = params_.find(h.join);
        if (it == params_.end()) return;
        for (const Held& made : it->second) {
            if (made.same_address(h)) h.value = made.value;
        }
    }

    // The parameter of `h.join` that takes the address from every arm.
    Value* parameter(const Held h) {
        BasicBlock* join = h.join;
        Value* param = ir::new_block_param(fn_, join, Type::i64());
        params_[join].push_back(Held{h.tagged, h.mask, param, join, h.mask_value});
        for (BasicBlock* p : join->predecessors()) {
            State& end = out_[p];
            Value* v = nullptr;
            if (Held* o = find(end, h)) {
                resolve(*o);
                if (!o->value) o->value = parameter(*o);
                v = o->value;
            } else {
                v = compute_at_end(p, h);
                end.held.push_back(Held{h.tagged, h.mask, v, nullptr, h.mask_value});
            }
            p->terminator()->branch_target().args.push_back(v);
        }
        return param;
    }

    Value* compute_at_end(BasicBlock* p, const Held& h) {
        Builder b(fn_);
        b.position_before(p->terminator());
        const BasicBlock* mask_block = block_of(h.mask_value);
        Value* mask = mask_block && dom_.dominates(mask_block, p) ? h.mask_value : b.build_iconst_i64(h.mask);
        return b.build_xor(b.build_bitcast_i64_tagged(const_cast<Value*>(h.tagged)), mask);
    }

    void rewrite() {
        std::unordered_map<const Value*, uint32_t> uses;
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                for_each_use_slot(*inst, [&](Value*& v) {
                    if (const auto it = replaced_.find(v); it != replaced_.end()) v = it->second;
                    if (v) ++uses[v];
                });
            }
        }
        for (Instruction* inst : dead_) {
            const Instruction* bits = nullptr;
            const Instruction* mask = nullptr;
            const bool through_tagged = as_read(*inst, bits, mask);
            inst->parent()->remove_instruction(inst);
            if (through_tagged && --uses[bits->result()] == 0) {
                bits->parent()->remove_instruction(const_cast<Instruction*>(bits));
            }
        }
    }

    Function& fn_;
    DominatorTree dom_;
    std::unordered_set<const BasicBlock*> outside_;
    // The i64 values some tagged value is made from.
    std::unordered_set<const Value*> rooted_;
    std::unordered_map<const BasicBlock*, State> out_;
    // The parameters made, by join.
    std::unordered_map<const BasicBlock*, std::vector<Held>> params_;
    std::unordered_map<const Value*, Value*> replaced_;
    std::vector<Instruction*> dead_;
};

} // namespace

size_t reuse_untagged_addresses(Function& fn) {
    if (fn.blocks().empty() || !fn.entry_block()) return 0;
    split_critical_edges(fn);
    fn.rebuild_cfg_predecessors();
    return Reuse(fn).run();
}

} // namespace brass
