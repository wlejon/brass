#include <brass/mir/gvn.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/memory_ssa.hpp>
#include "gvn_table.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace brass {

namespace {

void apply_substitutions(Function& fn, const std::unordered_map<Value*, Value*>& subst_map) {
    if (subst_map.empty()) return;

    auto resolve = [&](Value* v) -> Value* {
        if (!v) return nullptr;
        auto it = subst_map.find(v);
        if (it == subst_map.end()) return v;
        Value* root = it->second;
        size_t depth = 0;
        while (root && depth++ < 100) {
            auto it2 = subst_map.find(root);
            if (it2 == subst_map.end()) break;
            root = it2->second;
        }
        return root;
    };

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op = inst->operand(i);
                Value* r = resolve(op);
                if (r != op) {
                    inst->set_operand(i, r);
                }
            }
            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                Value* op = inst->branch_target().args[i];
                Value* r = resolve(op);
                if (r != op) {
                    inst->branch_target().args[i] = r;
                }
            }
            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                Value* op = inst->true_target().args[i];
                Value* r = resolve(op);
                if (r != op) {
                    inst->true_target().args[i] = r;
                }
            }
            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                Value* op = inst->false_target().args[i];
                Value* r = resolve(op);
                if (r != op) {
                    inst->false_target().args[i] = r;
                }
            }
            for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                Value* op = inst->default_target().args[i];
                Value* r = resolve(op);
                if (r != op) {
                    inst->default_target().args[i] = r;
                }
            }
            for (auto& sc : inst->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    Value* op = sc.target.args[i];
                    Value* r = resolve(op);
                    if (r != op) {
                        sc.target.args[i] = r;
                    }
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                Value* op = inst->state_map()[i];
                Value* r = resolve(op);
                if (r != op) {
                    inst->state_map()[i] = r;
                }
            }
        }
    }
}

struct PendingStore {
    Instruction* inst = nullptr;
    const Value* base = nullptr;
    int32_t offset = 0;
    Type memory_type = Type::void_type();
};

bool run_dse_pass(Function& fn, const AliasAnalysis& aa, GvnStats* stats) {
    bool changed = false;
    std::unordered_set<Instruction*> dead_instructions;

    // Track surviving pending stores at block entry for single-predecessor blocks
    std::unordered_map<const BasicBlock*, std::vector<PendingStore>> block_entry_stores;

    // Process blocks in reverse order
    const auto& blocks = fn.blocks();
    for (auto r_it = blocks.rbegin(); r_it != blocks.rend(); ++r_it) {
        BasicBlock* bb = *r_it;
        if (!bb) continue;

        std::vector<PendingStore> pending;

        // If this block has a single successor and the successor is a single-predecessor block,
        // inherit the surviving stores from the top of the successor
        const auto succs = bb->successors();
        if (succs.size() == 1 && succs[0] && succs[0]->predecessors().size() == 1) {
            auto it = block_entry_stores.find(succs[0]);
            if (it != block_entry_stores.end()) {
                pending = it->second;
            }
        }

        Instruction* cur = bb->tail();
        while (cur) {
            Instruction* prev = cur->prev();
            Opcode op = cur->opcode();

            if (op == Opcode::store || op == Opcode::vstore) {
                const Value* base = cur->operand(0);
                int32_t off = cur->offset();
                Type mtype = cur->memory_type();

                bool is_dead = false;
                for (const auto& ps : pending) {
                    if (ps.memory_type == mtype &&
                        aa.alias(base, off, mtype, ps.base, ps.offset, ps.memory_type) == AliasResult::MustAlias) {
                        is_dead = true;
                        break;
                    }
                }

                if (is_dead) {
                    dead_instructions.insert(cur);
                    if (stats) stats->dead_stores_eliminated++;
                    changed = true;
                } else {
                    // Remove any pending store that could be clobbered / partially overwritten
                    for (auto it = pending.begin(); it != pending.end(); ) {
                        if (aa.alias(base, off, mtype, it->base, it->offset, it->memory_type) != AliasResult::NoAlias) {
                            it = pending.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    pending.push_back({cur, base, off, mtype});
                }
            } else if (op == Opcode::store_indexed) {
                for (auto it = pending.begin(); it != pending.end(); ) {
                    if (aa.can_clobber(cur, it->inst)) {
                        it = pending.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else if (op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload || is_call(op)) {
                for (auto it = pending.begin(); it != pending.end(); ) {
                    if (aa.can_clobber(it->inst, cur)) {
                        it = pending.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else if (op == Opcode::ret || op == Opcode::safepoint) {
                for (auto it = pending.begin(); it != pending.end(); ) {
                    int64_t dummy = 0;
                    const Value* b = aa.get_underlying_base(it->base, dummy);
                    if (!aa.is_non_escaping(b)) {
                        it = pending.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            cur = prev;
        }

        block_entry_stores[bb] = std::move(pending);
    }

    // Remove dead store instructions
    for (Instruction* dead_inst : dead_instructions) {
        if (dead_inst && dead_inst->parent()) {
            dead_inst->parent()->remove_instruction(dead_inst);
        }
    }

    return changed;
}

} // namespace

bool gvn_function(Function& fn) {
    GvnOptions opts;
    return gvn_function(fn, opts);
}

bool gvn_function(Function& fn, const GvnOptions& options) {
    if (fn.name().starts_with("__wrapper_")) return false;
    fn.rebuild_cfg_predecessors();
    bool any_changed = false;

    // Run GVN up to 3 fixed-point iterations
    for (size_t iter = 0; iter < 3; ++iter) {
        bool iter_changed = false;

        DominatorTree dom(fn);
        AliasAnalysis aa(fn);
        MemorySSA mssa(fn, dom, aa);
        GvnTable table;

        const BasicBlock* entry = fn.entry_block();
        if (!entry || !dom.is_reachable(entry)) break;

        std::vector<Instruction*> to_remove;
        std::unordered_map<Value*, Value*> subst_map;

        auto resolve = [&](Value* v) -> Value* {
            if (!v) return nullptr;
            auto it = subst_map.find(v);
            if (it == subst_map.end()) return v;
            Value* root = it->second;
            size_t depth = 0;
            while (root && depth++ < 100) {
                auto it2 = subst_map.find(root);
                if (it2 == subst_map.end()) break;
                root = it2->second;
            }
            it->second = root;
            return root;
        };

        auto visit_block = [&](auto& self, const BasicBlock* bb) -> void {
            table.enter_scope();

            Instruction* cur = const_cast<BasicBlock*>(bb)->head();
            while (cur) {
                Instruction* next = cur->next();
                Opcode op = cur->opcode();

                // Canonicalize operands of cur
                for (size_t i = 0; i < cur->operand_count(); ++i) {
                    Value* op_val = cur->operand(i);
                    Value* r = resolve(op_val);
                    if (r != op_val) {
                        cur->set_operand(i, r);
                    }
                }

                // 1. Pure Expression CSE with Commutative Canonicalization
                if (options.enable_cse && is_pure_gvn_op(cur)) {
                    GvnExpression expr = GvnExpression::from_instruction(cur);
                    Value* existing = table.lookup_expression(expr);
                    if (existing && existing != cur->result()) {
                        Value* leader = resolve(existing);
                        subst_map[cur->result()] = leader;
                        to_remove.push_back(cur);
                        if (options.stats) options.stats->expressions_eliminated++;
                        iter_changed = true;
                        cur = next;
                        continue;
                    } else {
                        table.insert_expression(expr, cur->result());
                    }
                }

                // 2. Redundant Load Elimination (RLE)
                if (options.enable_rle && (op == Opcode::load || op == Opcode::vload)) {
                    Value* res_val = cur->result();
                    const Value* base = cur->operand(0);
                    int32_t off = cur->offset();
                    Type mtype = cur->memory_type();

                    MemoryUse* use = mssa.get_memory_use(cur);
                    MemoryAccess* def_acc = use ? use->defining_access() : nullptr;

                    // 2A. Store-to-Load Forwarding
                    bool forwarded = false;
                    if (def_acc && def_acc->is_def()) {
                        Instruction* def_inst = def_acc->origin_instruction();
                        if (def_inst && (def_inst->opcode() == Opcode::store || def_inst->opcode() == Opcode::vstore)) {
                            if (def_inst->parent() == cur->parent() || dom.dominates(def_inst->parent(), cur->parent())) {
                                if (def_inst->memory_type() == mtype &&
                                    aa.alias(def_inst->operand(0), def_inst->offset(), def_inst->memory_type(),
                                             base, off, mtype) == AliasResult::MustAlias) {
                                    Value* stored_val = resolve(def_inst->operand(1));
                                    subst_map[res_val] = stored_val;
                                    to_remove.push_back(cur);
                                    if (options.stats) options.stats->loads_forwarded++;
                                    iter_changed = true;
                                    forwarded = true;
                                }
                            }
                        }
                    }

                    if (forwarded) {
                        cur = next;
                        continue;
                    }

                    // 2B. Load-to-Load Elimination
                    int64_t dummy = 0;
                    const Value* underlying_base = aa.get_underlying_base(base, dummy);
                    uint32_t mem_id = def_acc ? def_acc->id() : 0;
                    int32_t total_off = off + static_cast<int32_t>(dummy);
                    AvailableLoadKey load_key{ underlying_base ? underlying_base : base, total_off, mtype, mem_id };

                    Value* dominating_load_val = table.lookup_load(load_key);
                    if (dominating_load_val && dominating_load_val != res_val) {
                        dominating_load_val = resolve(dominating_load_val);
                        subst_map[res_val] = dominating_load_val;
                        to_remove.push_back(cur);
                        if (options.stats) options.stats->loads_eliminated++;
                        iter_changed = true;
                        cur = next;
                        continue;
                    } else {
                        table.insert_load(load_key, res_val);
                    }
                }

                cur = next;
            }

            for (const BasicBlock* child : dom.children(bb)) {
                if (child) {
                    self(self, child);
                }
            }

            table.exit_scope();
        };

        visit_block(visit_block, entry);

        for (Instruction* inst : to_remove) {
            if (inst && inst->parent()) {
                inst->parent()->remove_instruction(inst);
            }
        }

        if (!subst_map.empty()) {
            apply_substitutions(fn, subst_map);
        }

        // 3. Dead Store Elimination (DSE)
        if (options.enable_dse) {
            iter_changed |= run_dse_pass(fn, aa, options.stats);
        }

        if (iter_changed) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
        } else {
            break;
        }
    }

    return any_changed;
}

bool gvn_module(Module& mod) {
    GvnOptions opts;
    return gvn_module(mod, opts);
}

bool gvn_module(Module& mod, const GvnOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn && !fn->name().starts_with("__wrapper_")) {
            changed |= gvn_function(*fn, options);
        }
    }
    return changed;
}


} // namespace brass
