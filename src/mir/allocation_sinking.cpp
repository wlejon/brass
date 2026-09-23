#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/uses.hpp>
#include "ir_clone.hpp"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Allocation sinking in four steps, for one allocation at a time:
//
//  1. Analysis. The object's "web" is the allocation's result plus every
//     block parameter all of whose incoming values are in the web. A use of a
//     web value is benign when it is a field load or store through it, or an
//     edge argument into a web parameter; any other use (a call argument, a
//     stored value, deopt state, an edge into a non-web parameter, a load or
//     store of an unsupported shape) is an escape. Within the region the
//     allocation dominates, a block is "escaped at entry" when any
//     predecessor is escaped at exit, and escaped at exit when it is escaped
//     at entry or contains an escape.
//  2. Every edge from a block still virtual at exit into a block escaped at
//     entry gets its own materialization block.
//  3. Phi placement on the new CFG: field values need block parameters at the
//     iterated dominance frontier of the blocks that define them, among the
//     blocks virtual at entry; the materialized pointer needs one at the
//     iterated dominance frontier of its definitions, among the blocks
//     escaped at entry (a web parameter already is one).
//  4. Renaming over the dominator tree: while virtual, loads read the current
//     field value and stores set it; at the first escape of a block virtual
//     at entry the allocation is re-emitted with every field stored into it;
//     once escaped, web values are replaced by the current pointer.

namespace brass {

namespace {

constexpr size_t kMaxRounds = 8;

bool is_scalar_field_type(Type t) {
    return t == Type::i32() || t == Type::i64() || t == Type::f32() || t == Type::f64();
}

Value* zero_of(Builder& b, Type t) {
    if (t == Type::i32()) return b.build_iconst_i32(0);
    if (t == Type::i64()) return b.build_iconst_i64(0);
    if (t == Type::f32()) return b.build_fconst_f32(0.0f);
    if (t == Type::f64()) return b.build_fconst_f64(0.0);
    throw std::logic_error("allocation sinking: no zero constant for a field of this type");
}

class Sinker {
public:
    Sinker(Function& fn, Instruction* alloc, const AllocationSinkingOptions& options, PartialEscapeStats& stats)
        : fn_(fn), alloc_(alloc), alloc_bb_(alloc->parent()), options_(options), stats_(stats) {}

    bool run() {
        if (!analyze()) return false;
        transform();
        return true;
    }

private:
    // ---- analysis ------------------------------------------------------

    bool in_region(const BasicBlock* bb) const { return region_.count(bb) != 0; }
    bool in_web(const Value* v) const { return v && web_.count(v) != 0; }

    // Every edge slot of every terminator, with the block it leaves.
    template <typename F>
    void for_each_edge_slot(F&& f) {
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            Instruction* term = bb->terminator();
            if (term) for_each_edge(*term, [&](BranchTarget& bt) { f(bb, *term, bt); });
        }
    }

    void compute_web() {
        const Value* obj = alloc_->result();
        web_ = {obj};
        // Grow: every parameter that receives a web value.
        std::unordered_set<const Value*> params;
        for (bool grew = true; grew;) {
            grew = false;
            for_each_edge_slot([&](BasicBlock*, Instruction&, BranchTarget& bt) {
                for (size_t i = 0; i < bt.args.size() && i < bt.block->param_count(); ++i) {
                    const Value* p = bt.block->param(i);
                    if ((bt.args[i] == obj || params.count(bt.args[i])) && p->type() == obj->type() &&
                        in_region(bt.block) && bt.block != alloc_bb_ && params.insert(p).second) {
                        grew = true;
                    }
                }
            });
        }
        // Shrink: a parameter stays only while every incoming value is in
        // the web (the object on all paths, never "the object or another").
        std::unordered_map<const Value*, bool> has_incoming;
        for (bool shrank = true; shrank;) {
            shrank = false;
            has_incoming.clear();
            std::vector<const Value*> drop;
            for_each_edge_slot([&](BasicBlock*, Instruction&, BranchTarget& bt) {
                for (size_t i = 0; i < bt.block->param_count(); ++i) {
                    const Value* p = bt.block->param(i);
                    if (!params.count(p)) continue;
                    has_incoming[p] = true;
                    const Value* a = i < bt.args.size() ? bt.args[i] : nullptr;
                    if (a != obj && !params.count(a)) drop.push_back(p);
                }
            });
            for (const Value* p : params) {
                if (!has_incoming[p]) drop.push_back(p);
            }
            for (const Value* p : drop) shrank |= params.erase(p) != 0;
        }
        web_.insert(params.begin(), params.end());
    }

    bool field_shape(const Instruction& inst, int32_t& off, Type& type) const {
        if (inst.opcode() == Opcode::load) {
            type = inst.type();
            if (inst.memory_type() != type) return false;
        } else if (inst.opcode() == Opcode::store) {
            if (!inst.operand(1) || in_web(inst.operand(1))) return false;
            type = inst.operand(1)->type();
            if (inst.memory_type() != type) return false;
        } else {
            return false;
        }
        off = inst.offset();
        return inst.operand_count() >= 1 && in_web(inst.operand(0)) && off >= 0 && is_scalar_field_type(type);
    }

    void compute_fields() {
        std::map<int32_t, Type> seen;
        std::unordered_set<int32_t> bad;
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                int32_t off = 0;
                Type type = Type::void_type();
                if (!field_shape(*inst, off, type)) continue;
                auto [it, fresh] = seen.emplace(off, type);
                if (!fresh && it->second != type) bad.insert(off);
            }
        }
        // Overlapping fields cannot be separate scalars.
        for (auto it = seen.begin(); it != seen.end(); ++it) {
            auto next = std::next(it);
            if (next == seen.end()) break;
            if (static_cast<int64_t>(it->first) + static_cast<int64_t>(it->second.size_in_bytes()) > next->first) {
                bad.insert(it->first);
                bad.insert(next->first);
            }
        }
        for (const auto& [off, type] : seen) {
            if (!bad.count(off)) fields_.emplace(off, type);
        }
    }

    bool is_field_load(const Instruction& inst) const {
        int32_t off = 0;
        Type type = Type::void_type();
        if (inst.opcode() != Opcode::load || !field_shape(inst, off, type)) return false;
        auto it = fields_.find(off);
        return it != fields_.end() && it->second == type;
    }

    bool is_field_store(const Instruction& inst) const {
        int32_t off = 0;
        Type type = Type::void_type();
        if (inst.opcode() != Opcode::store || !field_shape(inst, off, type)) return false;
        auto it = fields_.find(off);
        return it != fields_.end() && it->second == type;
    }

    // True when `inst` uses a web value in a way that needs the real object.
    bool escapes_at(const Instruction& inst) const {
        if (&inst == alloc_) return false;
        const bool field_access = is_field_load(inst) || is_field_store(inst);
        for (size_t i = 0; i < inst.operand_count(); ++i) {
            if (in_web(inst.operand(i)) && !(i == 0 && field_access)) return true;
        }
        for (const Value* sv : inst.state_map()) {
            if (in_web(sv)) return true;
        }
        bool escape = false;
        for_each_edge(inst, [&](const BranchTarget& bt) {
            for (size_t i = 0; i < bt.args.size(); ++i) {
                if (!in_web(bt.args[i])) continue;
                if (i >= bt.block->param_count() || !in_web(bt.block->param(i))) escape = true;
            }
        });
        return escape;
    }

    bool analyze() {
        if (!alloc_bb_ || !alloc_->result()) return false;
        fn_.rebuild_cfg_predecessors();
        DominatorTree dom(fn_);
        if (!dom.is_reachable(alloc_bb_)) return false;
        for (BasicBlock* bb : fn_.blocks()) {
            if (bb && dom.is_reachable(bb) && dom.dominates(alloc_bb_, bb)) region_.insert(bb);
        }
        // New block parameters need an argument on every incoming edge,
        // including edges from unreachable code the renaming never visits.
        for (const BasicBlock* bb : region_) {
            if (bb == alloc_bb_) continue;
            for (const BasicBlock* pred : bb->predecessors()) {
                if (!dom.is_reachable(pred)) return false;
            }
        }
        compute_web();
        compute_fields();
        if (fields_.size() > options_.max_fields) return false;

        // A web value used where the dominator tree cannot reach it would be
        // left dangling.
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || in_region(bb)) continue;
            for (Instruction* inst : *bb) {
                bool uses_web = false;
                for_each_use(*inst, [&](const Value* v) { uses_web |= in_web(v); });
                if (uses_web) return false;
            }
        }

        // First escape per block (after the allocation in its own block).
        for (const BasicBlock* bb : region_) {
            bool started = bb != alloc_bb_;
            for (Instruction* inst : *bb) {
                if (inst == alloc_) { started = true; continue; }
                if (started && escapes_at(*inst)) {
                    first_escape_[bb] = inst;
                    break;
                }
            }
        }
        // Escaping in its own block is no better than where it is now.
        if (first_escape_.count(alloc_bb_)) return false;

        // Escape state: monotone forward dataflow over the region.
        for (bool changed = true; changed;) {
            changed = false;
            for (const BasicBlock* bb : region_) {
                if (bb == alloc_bb_) continue;
                bool in = false;
                for (const BasicBlock* pred : bb->predecessors()) {
                    if (in_region(pred) && escaped_out(pred)) in = true;
                }
                if (in && !escaped_in_.count(bb)) {
                    escaped_in_.insert(bb);
                    changed = true;
                }
            }
        }

        // Worth doing only when the object dies unescaped on some path, or
        // some field access happens while it is still virtual. (Never a
        // loss: the object escapes at most once per allocation, so it is
        // materialized at most as often as it was allocated.)
        if (!dies_virtual() && !accessed_while_virtual()) return false;

        // Edges that need their own materialization block.
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || !in_region(bb) || escaped_out(bb)) continue;
            Instruction* term = bb->terminator();
            if (!term) continue;
            bool needs = false;
            for_each_edge(*term, [&](BranchTarget& bt) {
                if (bt.block != alloc_bb_ && in_region(bt.block) && escaped_in_.count(bt.block)) needs = true;
            });
            if (!needs) continue;
            // Only plain branches can be split; an invoke's edges cannot.
            const Opcode op = term->opcode();
            if (op != Opcode::br && op != Opcode::br_if && op != Opcode::switch_) return false;
            mat_edge_blocks_.push_back(bb);
        }
        return true;
    }

    bool dies_virtual() const {
        for (const BasicBlock* bb : region_) {
            if (escaped_out(bb) || !bb->terminator()) continue;
            if (bb->successors().empty()) return true;
            for (const BasicBlock* succ : bb->successors()) {
                if (!in_region(succ) || succ == alloc_bb_) return true;
            }
        }
        return false;
    }

    // True when `bb` lies on a cycle of the region that does not pass
    // through the allocation's block (it can run repeatedly per allocation).
    bool on_region_cycle(const BasicBlock* bb) const {
        const std::vector<BasicBlock*> succs = bb->successors();
        std::vector<const BasicBlock*> work(succs.begin(), succs.end());
        std::unordered_set<const BasicBlock*> seen;
        while (!work.empty()) {
            const BasicBlock* x = work.back();
            work.pop_back();
            if (x == bb) return true;
            if (!in_region(x) || x == alloc_bb_ || !seen.insert(x).second) continue;
            for (const BasicBlock* s : x->successors()) work.push_back(s);
        }
        return false;
    }

    // True when scalarization removes work: a field load while the object is
    // virtual, or a field store that can run repeatedly while it is. (Stores
    // that run once would only move to the materialization point.)
    bool accessed_while_virtual() const {
        for (const BasicBlock* bb : region_) {
            if (escaped_in_.count(bb)) continue;
            bool started = bb != alloc_bb_;
            auto escape_it = first_escape_.find(bb);
            const Instruction* first_escape = escape_it != first_escape_.end() ? escape_it->second : nullptr;
            for (const Instruction* inst : *bb) {
                if (inst == alloc_) { started = true; continue; }
                if (!started) continue;
                if (inst == first_escape) break;
                if (is_field_load(*inst)) return true;
                if (is_field_store(*inst) && bb != alloc_bb_ && on_region_cycle(bb)) return true;
            }
        }
        return false;
    }

    bool escaped_out(const BasicBlock* bb) const {
        return escaped_in_.count(bb) || first_escape_.count(bb) || mat_blocks_.count(bb);
    }

    // ---- transformation --------------------------------------------------

    void split_escape_edges() {
        for (BasicBlock* bb : mat_edge_blocks_) {
            Instruction* term = bb->terminator();
            for_each_edge(*term, [&](BranchTarget& bt) {
                if (bt.block == alloc_bb_ || !in_region(bt.block) || !escaped_in_.count(bt.block)) return;
                BasicBlock* mat = ir::new_block(fn_, "mat_pea_b" + std::to_string(bt.block->id()));
                Builder b(*fn_.parent());
                b.set_function(&fn_);
                b.position_at_end(mat);
                b.build_br(bt.block, bt.args);
                bt.block = mat;
                bt.args.clear();
                mat_blocks_.insert(mat);
                region_.insert(mat);
                first_escape_[mat] = mat->terminator();
                stats_.materialization_edges++;
            });
        }
    }

    // Iterated dominance frontier of `defs`, filtered by `keep`.
    template <typename Keep>
    std::unordered_set<BasicBlock*> idf(const std::vector<BasicBlock*>& defs, Keep&& keep) const {
        std::unordered_set<BasicBlock*> out;
        std::vector<BasicBlock*> work(defs.begin(), defs.end());
        std::unordered_set<BasicBlock*> queued(defs.begin(), defs.end());
        while (!work.empty()) {
            BasicBlock* x = work.back();
            work.pop_back();
            auto it = df_.find(x);
            if (it == df_.end()) continue;
            for (BasicBlock* y : it->second) {
                if (keep(y)) out.insert(y);
                if (queued.insert(y).second) work.push_back(y);
            }
        }
        return out;
    }

    void compute_frontiers(const DominatorTree& dom) {
        for (BasicBlock* b : fn_.blocks()) {
            if (!b || b->predecessors().size() < 2 || !dom.is_reachable(b)) continue;
            const BasicBlock* idom_b = dom.immediate_dominator(b);
            for (BasicBlock* pred : b->predecessors()) {
                const BasicBlock* runner = pred;
                while (runner && runner != idom_b && dom.is_reachable(runner)) {
                    df_[runner].push_back(b);
                    runner = dom.immediate_dominator(runner);
                }
            }
        }
    }

    Value* web_param_of(const BasicBlock* bb) const {
        for (Value* p : bb->params()) {
            if (in_web(p)) return p;
        }
        return nullptr;
    }

    void place_params() {
        auto virtual_merge = [&](BasicBlock* y) {
            return in_region(y) && y != alloc_bb_ && !escaped_in_.count(y);
        };
        for (const auto& [off, type] : fields_) {
            std::vector<BasicBlock*> defs = {alloc_bb_};
            for (BasicBlock* bb : fn_.blocks()) {
                if (!bb || !in_region(bb) || escaped_in_.count(bb)) continue;
                for (Instruction* inst : *bb) {
                    if (is_field_store(*inst) && inst->offset() == off) {
                        defs.push_back(bb);
                        break;
                    }
                }
            }
            for (BasicBlock* y : idf(defs, virtual_merge)) field_params_[y].emplace_back(off, nullptr);
        }
        for (auto& [y, list] : field_params_) {
            std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (auto& [off, param] : list) param = ir::new_block_param(fn_, y, fields_.at(off));
        }

        std::vector<BasicBlock*> ptr_defs;
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || !in_region(bb)) continue;
            const bool mat_here = !escaped_in_.count(bb) && first_escape_.count(bb);
            if (mat_here || (escaped_in_.count(bb) && web_param_of(bb))) ptr_defs.push_back(bb);
        }
        auto escaped_merge = [&](BasicBlock* y) {
            return in_region(y) && y != alloc_bb_ && escaped_in_.count(y) && !web_param_of(y);
        };
        for (BasicBlock* y : idf(ptr_defs, escaped_merge)) {
            ptr_params_[y] = ir::new_block_param(fn_, y, alloc_->result()->type());
        }
    }

    struct State {
        std::map<int32_t, Value*> fields;
        Value* ptr = nullptr;
        bool is_virtual = false;
    };

    Value* materialize(Instruction* before, const State& st) {
        ir::ValueMap values;
        Instruction* copy = ir::clone_instruction(fn_, *alloc_, values, {});
        before->parent()->insert_before(copy, before);
        Builder b(*fn_.parent());
        b.set_function(&fn_);
        b.position_before(before);
        for (const auto& [off, type] : fields_) {
            b.build_store(type, copy->result(), off, st.fields.at(off));
        }
        stats_.materialized_allocations++;
        return copy->result();
    }

    void rename_block(BasicBlock* bb, State& st) {
        if (bb != alloc_bb_) {
            st.is_virtual = !escaped_in_.count(bb);
            if (st.is_virtual) {
                auto it = field_params_.find(bb);
                if (it != field_params_.end()) {
                    for (const auto& [off, param] : it->second) st.fields[off] = param;
                }
            } else if (Value* wp = web_param_of(bb)) {
                st.ptr = wp;
            } else if (auto it = ptr_params_.find(bb); it != ptr_params_.end()) {
                st.ptr = it->second;
            }
        }
        std::vector<Instruction*> insts;
        for (Instruction* inst : *bb) insts.push_back(inst);
        auto escape_it = first_escape_.find(bb);
        Instruction* first_escape = escape_it != first_escape_.end() ? escape_it->second : nullptr;
        bool before_alloc = bb == alloc_bb_;
        for (Instruction* inst : insts) {
            if (before_alloc) {
                if (inst != alloc_) continue;
                before_alloc = false;
                Builder b(*fn_.parent());
                b.set_function(&fn_);
                b.position_before(alloc_);
                for (const auto& [off, type] : fields_) st.fields[off] = zero_of(b, type);
                st.is_virtual = true;
                dead_.push_back(alloc_);
                continue;
            }
            if (st.is_virtual) {
                if (inst == first_escape) {
                    st.ptr = materialize(inst, st);
                    st.is_virtual = false;
                } else if (is_field_load(*inst)) {
                    replace_all_uses(fn_, inst->result(), st.fields.at(inst->offset()));
                    dead_.push_back(inst);
                    stats_.scalarized_loads++;
                    continue;
                } else if (is_field_store(*inst)) {
                    st.fields[inst->offset()] = inst->operand(1);
                    dead_.push_back(inst);
                    stats_.scalarized_stores++;
                    continue;
                }
            }
            if (!st.is_virtual) {
                for_each_use_slot(*inst, [&](Value*& slot) {
                    if (in_web(slot)) slot = st.ptr;
                });
            }
        }
        Instruction* term = bb->terminator();
        if (!term) return;
        for_each_edge(*term, [&](BranchTarget& bt) {
            if (auto it = field_params_.find(bt.block); it != field_params_.end()) {
                if (!st.is_virtual) throw std::logic_error("allocation sinking: escaped edge into a virtual block");
                for (const auto& [off, param] : it->second) bt.args.push_back(st.fields.at(off));
            }
            if (ptr_params_.count(bt.block)) {
                if (st.is_virtual || !st.ptr) throw std::logic_error("allocation sinking: no pointer on an escaped edge");
                bt.args.push_back(st.ptr);
            }
        });
    }

    void transform() {
        split_escape_edges();
        fn_.rebuild_cfg_predecessors();
        DominatorTree dom(fn_);
        compute_frontiers(dom);
        place_params();

        // Renaming over the dominator tree, children seeing their parent's
        // state at its end.
        std::vector<std::pair<BasicBlock*, State>> work;
        work.emplace_back(alloc_bb_, State{});
        while (!work.empty()) {
            auto [bb, st] = std::move(work.back());
            work.pop_back();
            rename_block(bb, st);
            for (const BasicBlock* child : dom.children(bb)) {
                if (in_region(child)) work.emplace_back(const_cast<BasicBlock*>(child), st);
            }
        }

        for (Instruction* inst : dead_) {
            if (inst->parent()) inst->parent()->remove_instruction(inst);
        }
        // Web parameters of blocks the object is virtual in carried nothing
        // real; drop them with their incoming arguments.
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || !in_region(bb) || escaped_in_.count(bb)) continue;
            for (size_t i = bb->param_count(); i-- > 0;) {
                if (in_web(bb->param(i))) remove_block_param(*bb, i);
            }
        }
        fn_.rebuild_cfg_predecessors();
        stats_.sunk_allocations++;
        stats_.virtual_allocations++;
    }

    Function& fn_;
    Instruction* alloc_;
    BasicBlock* alloc_bb_;
    const AllocationSinkingOptions& options_;
    PartialEscapeStats& stats_;

    std::unordered_set<const BasicBlock*> region_;
    std::unordered_set<const Value*> web_;
    std::map<int32_t, Type> fields_;
    std::unordered_map<const BasicBlock*, Instruction*> first_escape_;
    std::unordered_set<const BasicBlock*> escaped_in_;
    std::vector<BasicBlock*> mat_edge_blocks_;
    std::unordered_set<const BasicBlock*> mat_blocks_;
    std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>> df_;
    std::unordered_map<BasicBlock*, std::vector<std::pair<int32_t, Value*>>> field_params_;
    std::unordered_map<BasicBlock*, Value*> ptr_params_;
    std::vector<Instruction*> dead_;
};

} // namespace

AllocationSinkingPass::AllocationSinkingPass(Function& fn, const AllocationSinkingOptions& options)
    : fn_(fn), options_(options) {
    if (options_.stats) stats_ = *options_.stats;
}

bool AllocationSinkingPass::sink_one(Instruction* alloc) {
    Sinker sinker(fn_, alloc, options_, stats_);
    return sinker.run();
}

bool AllocationSinkingPass::run() {
    // Coroutine state machines resume into blocks no edge names.
    if (!fn_.resume_points().empty() || !fn_.entry_block()) return false;
    bool changed = false;
    for (size_t round = 0; round < kMaxRounds; ++round) {
        std::vector<Instruction*> allocs;
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (inst->opcode() == Opcode::call && inst->result() && is_allocation_call(inst)) allocs.push_back(inst);
            }
        }
        bool round_changed = false;
        for (Instruction* alloc : allocs) {
            // Each transformation rewrites the CFG; the next allocation is
            // analyzed afresh.
            if (alloc->parent() && sink_one(alloc)) round_changed = true;
        }
        changed |= round_changed;
        if (!round_changed) break;
    }
    if (options_.stats) *options_.stats = stats_;
    return changed;
}

bool sink_allocations(Function& fn, const AllocationSinkingOptions& options) {
    AllocationSinkingPass pass(fn, options);
    return pass.run();
}

bool sink_allocations(Module& mod, const AllocationSinkingOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) changed |= sink_allocations(*fn, options);
    }
    return changed;
}

} // namespace brass
