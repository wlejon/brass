#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

void for_each_branch_target(Instruction* term, auto&& fn) {
    if (!term) return;
    if (term->opcode() == Opcode::br) {
        fn(term->branch_target());
    } else if (term->opcode() == Opcode::br_if) {
        fn(term->true_target());
        fn(term->false_target());
    } else if (term->opcode() == Opcode::switch_) {
        fn(term->default_target());
        for (auto& sc : term->switch_cases()) {
            fn(sc.target);
        }
    }
}

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) inst->set_operand(i, new_val);
            }
            for_each_branch_target(inst, [&](BranchTarget& bt) {
                for (size_t i = 0; i < bt.args.size(); ++i) {
                    if (bt.args[i] == old_val) bt.args[i] = new_val;
                }
            });
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) inst->state_map()[i] = new_val;
            }
        }
    }
}

Value* get_or_create_zero(Module& mod, Function& fn, Type type) {
    Builder b(mod);
    b.set_function(&fn);
    BasicBlock* entry = fn.entry_block();
    Instruction* insert_pos = entry ? entry->head() : nullptr;
    while (insert_pos && brass::is_constant(insert_pos->opcode())) {
        insert_pos = insert_pos->next();
    }
    if (insert_pos) {
        b.position_before(insert_pos);
    } else if (entry) {
        b.position_at_end(entry);
    }

    if (type.kind() == TypeKind::I32) {
        return b.build_iconst_i32(0);
    } else if (type.kind() == TypeKind::I64) {
        return b.build_iconst_i64(0);
    } else if (type.kind() == TypeKind::F32 || type.kind() == TypeKind::F64) {
        return b.build_fconst_f64(0.0);
    }
    return b.build_iconst_i64(0);
}

void replace_uses_in_block(BasicBlock* bb, const std::unordered_set<const Value*>& aliases, Value* new_val) {
    if (!bb || !new_val) return;
    for (Instruction* inst : *bb) {
        if (!inst) continue;
        for (size_t i = 0; i < inst->operand_count(); ++i) {
            if (aliases.count(inst->operand(i)) > 0) {
                inst->set_operand(i, new_val);
            }
        }
        for_each_branch_target(inst, [&](BranchTarget& bt) {
            for (size_t i = 0; i < bt.args.size(); ++i) {
                if (aliases.count(bt.args[i]) > 0) {
                    bt.args[i] = new_val;
                }
            }
        });
        for (size_t i = 0; i < inst->state_map().size(); ++i) {
            if (aliases.count(inst->state_map()[i]) > 0) {
                inst->state_map()[i] = new_val;
            }
        }
    }
}

void replace_uses_in_dom_subtree(const DominatorTree& dom, BasicBlock* root,
                                const std::unordered_set<const Value*>& aliases, Value* new_val) {
    if (!root || !new_val) return;
    replace_uses_in_block(root, aliases, new_val);
    for (const BasicBlock* child : dom.children(root)) {
        replace_uses_in_dom_subtree(dom, const_cast<BasicBlock*>(child), aliases, new_val);
    }
}

} // namespace

AllocationSinkingPass::AllocationSinkingPass(Function& fn, const AllocationSinkingOptions& options)
    : fn_(fn), options_(options) {
    if (options_.stats) {
        stats_ = *options_.stats;
    }
}

bool AllocationSinkingPass::run() {
    bool changed = false;
    constexpr size_t max_rounds = 8;

    for (size_t round = 0; round < max_rounds; ++round) {
        fn_.rebuild_cfg_predecessors();
        DominatorTree dom(fn_);
        LoopAnalysis loops(fn_, dom);

        PartialEscapeAnalysis pea(fn_);
        const auto& candidates = pea.candidate_allocations();
        if (candidates.empty()) {
            break;
        }

        bool round_changed = false;
        for (const Value* alloc_val : candidates) {
            if (process_candidate(alloc_val, pea, dom)) {
                round_changed = true;
                changed = true;
                break; // Re-analyze after transforming each candidate
            }
        }

        if (!round_changed) {
            break;
        }
    }

    if (options_.stats) {
        *options_.stats = stats_;
    }

    return changed;
}

bool AllocationSinkingPass::process_candidate(const Value* alloc_val,
                                              const PartialEscapeAnalysis& pea,
                                              DominatorTree& dom) {
    if (!alloc_val || !alloc_val->is_instruction()) return false;
    Instruction* alloc_inst = alloc_val->defining_instruction();
    if (!alloc_inst || !alloc_inst->parent()) return false;
    BasicBlock* alloc_bb = alloc_inst->parent();

    const auto& aliases = pea.get_aliases(alloc_val);
    auto frontier = pea.get_materialization_frontier(alloc_val);

    // 1. Gather all fields accessed and candidate memory instructions
    std::map<int32_t, Type> fields;
    std::vector<Instruction*> candidate_loads;
    std::vector<Instruction*> candidate_stores;

    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || inst == alloc_inst) continue;

            if (inst->opcode() == Opcode::load && aliases.count(inst->operand(0)) > 0) {
                candidate_loads.push_back(inst);
                fields[inst->offset()] = inst->type();
            } else if (inst->opcode() == Opcode::store && aliases.count(inst->operand(0)) > 0) {
                candidate_stores.push_back(inst);
                Type st_t = inst->operand(1) ? inst->operand(1)->type() : inst->memory_type();
                fields[inst->offset()] = st_t;
            }
        }
    }

    Builder builder(*fn_.parent());
    builder.set_function(&fn_);

    // 2. Identify virtual blocks
    std::unordered_set<BasicBlock*> virtual_blocks;
    for (BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        if (pea.get_block_state(bb, alloc_val) == ObjectState::Virtual) {
            virtual_blocks.insert(bb);
        }
    }
    virtual_blocks.insert(alloc_bb);

    // 3. Materialization injection on escape edges
    struct MatEdgeInfo {
        BasicBlock* mat_bb;
        Value* mat_ptr;
    };
    std::unordered_map<const BasicBlock*, std::vector<MatEdgeInfo>> edge_mat_map;

    for (const CFGEdge& edge : frontier) {
        BasicBlock* U = const_cast<BasicBlock*>(edge.from);
        BasicBlock* V = const_cast<BasicBlock*>(edge.to);
        if (!U || !V) continue;

        std::string mat_name = "mat_pea_b" + std::to_string(V->id());
        BasicBlock* mat_bb = builder.append_block(mat_name);

        builder.position_at_end(mat_bb);

        // a. Re-emit allocation instruction
        Value* mat_ptr = builder.build_call(alloc_inst->symbol(), alloc_inst->type(), alloc_inst->operands());
        stats_.materialized_allocations++;
        stats_.materialization_edges++;
        edge_mat_map[U].push_back({mat_bb, mat_ptr});

        // b. Rewire edge U -> V to U -> mat_bb -> V
        Instruction* u_term = U->terminator();
        std::vector<Value*> forwarded_args;

        auto rewire = [&](BranchTarget& bt) {
            if (bt.block == V) {
                forwarded_args = bt.args;
                for (size_t i = 0; i < forwarded_args.size(); ++i) {
                    if (aliases.count(forwarded_args[i]) > 0) {
                        forwarded_args[i] = mat_ptr;
                    }
                }
                bt.block = mat_bb;
                bt.args.clear();
            }
        };

        if (u_term) {
            if (u_term->opcode() == Opcode::br) {
                rewire(u_term->branch_target());
            } else if (u_term->opcode() == Opcode::br_if) {
                rewire(u_term->true_target());
                rewire(u_term->false_target());
            } else if (u_term->opcode() == Opcode::switch_) {
                rewire(u_term->default_target());
                for (auto& sc : u_term->switch_cases()) {
                    rewire(sc.target);
                }
            }
        }

        // In mat_bb: branch to V
        builder.position_at_end(mat_bb);
        builder.build_br(V, forwarded_args);

        // Replace direct uses in V and its subtree
        replace_uses_in_dom_subtree(dom, V, aliases, mat_ptr);
    }

    // 4. Scalarize loads and stores within the virtual region
    // Dominance frontier computation for phi insertion
    std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>> df;
    for (BasicBlock* b : fn_.blocks()) {
        if (!b || b->predecessors().size() < 2) continue;
        const BasicBlock* idom_b = dom.immediate_dominator(b);
        for (BasicBlock* pred : b->predecessors()) {
            const BasicBlock* runner = pred;
            while (runner && runner != idom_b && dom.is_reachable(runner)) {
                df[runner].push_back(b);
                runner = dom.immediate_dominator(runner);
            }
        }
    }

    // Compute IDF for each field
    struct FieldEntry {
        int32_t offset;
        Type type;
    };
    std::vector<FieldEntry> sorted_fields;
    for (const auto& [off, type] : fields) {
        sorted_fields.push_back({off, type});
    }

    std::unordered_map<BasicBlock*, std::vector<FieldEntry>> fields_for_block;
    std::unordered_map<BasicBlock*, std::unordered_map<int32_t, Value*>> phi_params;

    for (const FieldEntry& f : sorted_fields) {
        std::vector<BasicBlock*> def_blocks;
        def_blocks.push_back(alloc_bb);
        for (Instruction* st : candidate_stores) {
            if (st->offset() == f.offset && st->parent()) {
                if (virtual_blocks.count(st->parent()) > 0) {
                    def_blocks.push_back(st->parent());
                }
            }
        }

        std::unordered_set<BasicBlock*> idf_set;
        std::vector<BasicBlock*> worklist(def_blocks.begin(), def_blocks.end());
        std::unordered_set<BasicBlock*> in_worklist(def_blocks.begin(), def_blocks.end());

        size_t w_i = 0;
        while (w_i < worklist.size()) {
            BasicBlock* b = worklist[w_i++];
            auto it = df.find(b);
            if (it != df.end()) {
                for (BasicBlock* y : it->second) {
                    if (y != alloc_bb && dom.dominates(alloc_bb, y) && virtual_blocks.count(y) > 0 && idf_set.insert(y).second) {
                        if (in_worklist.insert(y).second) {
                            worklist.push_back(y);
                        }
                    }
                }
            }
        }

        for (BasicBlock* b : idf_set) {
            fields_for_block[b].push_back(f);
        }
    }

    // Insert block parameters at merge points
    for (auto& [b, flist] : fields_for_block) {
        std::sort(flist.begin(), flist.end(), [](const FieldEntry& a, const FieldEntry& b_f) {
            return a.offset < b_f.offset;
        });
        for (const FieldEntry& f : flist) {
            Value* param = builder.add_block_param(b, f.type);
            phi_params[b][f.offset] = param;
        }
    }

    // 5. SSA renaming traversal on dominator tree
    std::unordered_set<Instruction*> insts_to_remove;

    auto scalarize_traversal = [&](auto& self, BasicBlock* bb,
                                  std::unordered_map<int32_t, Value*> current_def) -> void {
        if (!bb) return;

        // Phi parameters at block entry
        auto it_phi = phi_params.find(bb);
        if (it_phi != phi_params.end()) {
            for (const auto& [off, param_val] : it_phi->second) {
                current_def[off] = param_val;
            }
        }

        // Initial zero defs in alloc_bb
        if (bb == alloc_bb) {
            for (const FieldEntry& f : sorted_fields) {
                if (current_def.find(f.offset) == current_def.end()) {
                    current_def[f.offset] = get_or_create_zero(*fn_.parent(), fn_, f.type);
                }
            }
        }

        // Process instructions in bb
        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();

            if (cur == alloc_inst) {
                insts_to_remove.insert(cur);
            } else if (cur->opcode() == Opcode::store && aliases.count(cur->operand(0)) > 0) {
                current_def[cur->offset()] = cur->operand(1);
                insts_to_remove.insert(cur);
                stats_.scalarized_stores++;
            } else if (cur->opcode() == Opcode::load && aliases.count(cur->operand(0)) > 0) {
                int32_t off = cur->offset();
                Value* loaded_val = current_def[off];
                if (!loaded_val) {
                    loaded_val = get_or_create_zero(*fn_.parent(), fn_, cur->type());
                    current_def[off] = loaded_val;
                }
                replace_all_uses(fn_, cur->result(), loaded_val);
                insts_to_remove.insert(cur);
                stats_.scalarized_loads++;
            }
            cur = next;
        }

        // If bb has frontier edges to materialization blocks, emit stores into mat_ptr
        auto it_m = edge_mat_map.find(bb);
        if (it_m != edge_mat_map.end()) {
            for (const auto& minfo : it_m->second) {
                Instruction* term = minfo.mat_bb->terminator();
                if (term) {
                    builder.position_before(term);
                } else {
                    builder.position_at_end(minfo.mat_bb);
                }
                for (const FieldEntry& f : sorted_fields) {
                    Value* outgoing = current_def[f.offset];
                    if (!outgoing) {
                        outgoing = get_or_create_zero(*fn_.parent(), fn_, f.type);
                    }
                    builder.build_store(f.type, minfo.mat_ptr, f.offset, outgoing);
                }
            }
        }

        // Update branch arguments to successors that have phi parameters
        for (BasicBlock* succ : bb->successors()) {
            if (!succ) continue;
            auto it_f = fields_for_block.find(succ);
            if (it_f == fields_for_block.end() || it_f->second.empty()) continue;

            Instruction* term = bb->terminator();
            if (!term) continue;

            for (const FieldEntry& f : it_f->second) {
                Value* outgoing = current_def[f.offset];
                if (!outgoing) {
                    outgoing = get_or_create_zero(*fn_.parent(), fn_, f.type);
                }
                for_each_branch_target(term, [&](BranchTarget& bt) {
                    if (bt.block == succ) {
                        bt.args.push_back(outgoing);
                    }
                });
            }
        }

        // Recurse to dominator children in virtual_blocks
        for (const BasicBlock* child : dom.children(bb)) {
            BasicBlock* child_bb = const_cast<BasicBlock*>(child);
            if (virtual_blocks.count(child_bb) > 0) {
                self(self, child_bb, current_def);
            }
        }
    };

    std::unordered_map<int32_t, Value*> initial_defs;
    scalarize_traversal(scalarize_traversal, alloc_bb, initial_defs);

    // 6. Remove dead instructions
    for (Instruction* inst : insts_to_remove) {
        if (inst && inst->parent()) {
            inst->parent()->remove_instruction(inst);
        }
    }
    stats_.sunk_allocations++;
    stats_.virtual_allocations++;

    // 7. Remove dead alias block parameters
    std::unordered_map<BasicBlock*, std::vector<uint32_t>> params_to_remove;
    for (const Value* v : aliases) {
        if (v && v->is_block_param()) {
            BasicBlock* def_bb = v->defining_block();
            if (def_bb && virtual_blocks.count(def_bb) > 0) {
                params_to_remove[def_bb].push_back(v->param_index());
            }
        }
    }

    for (auto& [b, idxs] : params_to_remove) {
        std::sort(idxs.begin(), idxs.end(), std::greater<uint32_t>());
        for (uint32_t idx : idxs) {
            if (idx < b->param_count()) {
                b->params().erase(b->params().begin() + idx);
                for (size_t k = idx; k < b->param_count(); ++k) {
                    b->params()[k]->set_block_param(b, static_cast<uint32_t>(k));
                }
            }
            for (BasicBlock* pred : fn_.blocks()) {
                if (!pred) continue;
                Instruction* term = pred->terminator();
                if (!term) continue;
                for_each_branch_target(term, [&](BranchTarget& bt) {
                    if (bt.block == b && idx < bt.args.size()) {
                        bt.args.erase(bt.args.begin() + idx);
                    }
                });
            }
        }
    }

    fn_.rebuild_cfg_predecessors();
    return true;
}

bool sink_allocations(Function& fn, const AllocationSinkingOptions& options) {
    AllocationSinkingPass pass(fn, options);
    return pass.run();
}

bool sink_allocations(Module& mod, const AllocationSinkingOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= sink_allocations(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
