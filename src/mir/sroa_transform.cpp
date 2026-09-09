#include "sroa_transform.hpp"
#include <algorithm>

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

} // namespace

SroaTransformer::SroaTransformer(Function& fn, const SroaOptions& options, SroaStats* stats)
    : fn_(fn), options_(options), stats_(stats) {}

Value* SroaTransformer::get_or_create_zero_constant(Builder& b, BasicBlock* entry, Type type) {
    uint8_t key = static_cast<uint8_t>(type.kind());
    auto it = zero_constants_.find(key);
    if (it != zero_constants_.end()) {
        return it->second;
    }

    Instruction* insert_pos = entry ? entry->head() : nullptr;
    while (insert_pos && brass::is_constant(insert_pos->opcode())) {
        insert_pos = insert_pos->next();
    }
    if (insert_pos) {
        b.position_before(insert_pos);
    } else if (entry) {
        b.position_at_end(entry);
    }

    Value* zero_val = nullptr;
    if (type.kind() == TypeKind::I32) {
        zero_val = b.build_iconst_i32(0);
    } else if (type.kind() == TypeKind::I64) {
        zero_val = b.build_iconst_i64(0);
    } else if (type.kind() == TypeKind::F32 || type.kind() == TypeKind::F64) {
        zero_val = b.build_fconst_f64(0.0);
    } else {
        zero_val = b.build_iconst_i64(0);
    }

    zero_constants_[key] = zero_val;
    return zero_val;
}

bool SroaTransformer::run() {
    bool changed = false;
    constexpr size_t max_rounds = 8;

    for (size_t round = 0; round < max_rounds; ++round) {
        fn_.rebuild_cfg_predecessors();
        EscapeAnalysisOptions ea_opts;
        EscapeAnalysis ea(fn_, ea_opts);

        const auto& non_escaping = ea.non_escaping_allocations();
        if (non_escaping.empty()) {
            break;
        }

        bool round_changed = false;
        for (const Value* alloc_val : non_escaping) {
            if (process_candidate(alloc_val, ea)) {
                round_changed = true;
                changed = true;
                break; // Re-analyze after dissolving an allocation
            }
        }

        if (!round_changed) {
            break;
        }
    }

    return changed;
}

bool SroaTransformer::process_candidate(const Value* alloc_val, const EscapeAnalysis& /*ea*/) {
    if (!alloc_val || !alloc_val->is_instruction()) return false;
    Instruction* alloc_inst = alloc_val->defining_instruction();
    if (!alloc_inst) return false;

    // 1. Gather all aliases of alloc_val across block parameters
    std::unordered_set<const Value*> aliases;
    aliases.insert(alloc_val);

    bool alias_changed = true;
    while (alias_changed) {
        alias_changed = false;
        for (const BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            const Instruction* term = bb->terminator();
            if (!term) continue;

            auto check_bt = [&](const BranchTarget& bt) {
                if (!bt.block) return;
                for (size_t i = 0; i < bt.args.size(); ++i) {
                    if (aliases.count(bt.args[i]) > 0 && i < bt.block->param_count()) {
                        const Value* param = bt.block->param(i);
                        if (param && aliases.insert(param).second) {
                            alias_changed = true;
                        }
                    }
                }
            };

            if (term->opcode() == Opcode::br) {
                check_bt(term->branch_target());
            } else if (term->opcode() == Opcode::br_if) {
                check_bt(term->true_target());
                check_bt(term->false_target());
            } else if (term->opcode() == Opcode::switch_) {
                check_bt(term->default_target());
                for (const auto& sc : term->switch_cases()) {
                    check_bt(sc.target);
                }
            }
        }
    }

    // 2. Validate all uses of all aliases and collect fields accessed
    std::map<int32_t, Type> fields;
    std::unordered_set<int32_t> live_offsets;
    std::vector<Instruction*> candidate_loads;
    std::vector<Instruction*> candidate_stores;

    for (const BasicBlock* bb : fn_.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst == alloc_inst) continue;

            // Check if any alias is used as an operand
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                const Value* op = inst->operand(i);
                if (aliases.count(op) > 0) {
                    if (inst->opcode() == Opcode::load && i == 0) {
                        candidate_loads.push_back(inst);
                        live_offsets.insert(inst->offset());
                        auto [it, inserted] = fields.insert({inst->offset(), inst->type()});
                        if (!inserted && it->second != inst->type()) {
                            return false; // Conflicting types at same offset
                        }
                    } else if (inst->opcode() == Opcode::store && i == 0) {
                        candidate_stores.push_back(inst);
                        Value* stored_val = inst->operand(1);
                        if (aliases.count(stored_val) > 0) {
                            return false; // Storing aggregate pointer into itself
                        }
                        Type stored_type = stored_val ? stored_val->type() : inst->memory_type();
                        auto [it, inserted] = fields.insert({inst->offset(), stored_type});
                        if (!inserted && it->second != stored_type) {
                            return false; // Conflicting types at same offset
                        }
                    } else {
                        // Any other use of the pointer prevents full SROA
                        return false;
                    }
                }
            }

            // Check branch arguments: alias may only be passed to an alias block parameter
            for_each_branch_target(inst, [&](const BranchTarget& bt) {
                if (!bt.block) return;
                for (size_t i = 0; i < bt.args.size(); ++i) {
                    if (aliases.count(bt.args[i]) > 0) {
                        if (i >= bt.block->param_count() || aliases.count(bt.block->param(i)) == 0) {
                            // Passed to a non-alias parameter
                        }
                    }
                }
            });
        }
    }

    if (fields.size() > options_.max_fields_per_object) {
        return false;
    }

    // 3. If there are no live loads, all stores and the allocation are dead!
    if (live_offsets.empty()) {
        for (Instruction* st : candidate_stores) {
            if (st && st->parent()) {
                st->parent()->remove_instruction(st);
                if (stats_) stats_->stores_eliminated++;
            }
        }
        if (alloc_inst->parent()) {
            alloc_inst->parent()->remove_instruction(alloc_inst);
            if (stats_) stats_->allocations_eliminated++;
        }
        return true;
    }

    // 4. Determine DefBlocks for each live field
    BasicBlock* alloc_bb = alloc_inst->parent();
    std::vector<FieldInfo> sorted_live_fields;
    for (const auto& [off, type] : fields) {
        if (live_offsets.count(off) > 0) {
            sorted_live_fields.push_back({off, type});
        }
    }

    DominatorTree dom(fn_);

    // Compute Dominance Frontiers
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

    // Compute IDF for each live field
    std::unordered_map<int32_t, std::unordered_set<BasicBlock*>> field_idf;
    for (const FieldInfo& f : sorted_live_fields) {
        std::vector<BasicBlock*> def_blocks;
        def_blocks.push_back(alloc_bb);
        for (Instruction* st : candidate_stores) {
            if (st->offset() == f.offset && st->parent()) {
                def_blocks.push_back(st->parent());
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
                    if (idf_set.insert(y).second) {
                        if (in_worklist.insert(y).second) {
                            worklist.push_back(y);
                        }
                    }
                }
            }
        }
        field_idf[f.offset] = std::move(idf_set);
    }

    // 5. Insert block parameters at merge blocks (IDF) for live fields
    Builder builder(*fn_.parent());
    builder.set_function(&fn_);

    std::unordered_map<BasicBlock*, std::vector<FieldInfo>> fields_for_block;
    std::unordered_map<BasicBlock*, std::unordered_map<int32_t, Value*>> phi_params;

    for (BasicBlock* b : fn_.blocks()) {
        if (!b) continue;
        for (const FieldInfo& f : sorted_live_fields) {
            if (field_idf[f.offset].count(b) > 0) {
                fields_for_block[b].push_back(f);
            }
        }
    }

    for (auto& [b, flist] : fields_for_block) {
        std::sort(flist.begin(), flist.end(), [](const FieldInfo& a, const FieldInfo& b_f) {
            return a.offset < b_f.offset;
        });
        for (const FieldInfo& f : flist) {
            Value* param = builder.create_value(f.type);
            b->add_param(param);
            phi_params[b][f.offset] = param;
            if (stats_) stats_->block_params_created++;
        }
    }

    // 6. Dominator Tree Traversal & SSA Renaming
    std::unordered_set<Instruction*> insts_to_remove;
    BasicBlock* entry_bb = fn_.entry_block();

    auto rename_traversal = [&](auto& self, BasicBlock* bb, std::unordered_map<int32_t, Value*> current_def) -> void {
        if (!bb) return;

        // Phi parameters at block entry
        auto it_phi = phi_params.find(bb);
        if (it_phi != phi_params.end()) {
            for (const auto& [off, param_val] : it_phi->second) {
                current_def[off] = param_val;
            }
        }

        // Initial zero definition in alloc block
        if (bb == alloc_bb) {
            for (const FieldInfo& f : sorted_live_fields) {
                if (current_def.find(f.offset) == current_def.end()) {
                    current_def[f.offset] = get_or_create_zero_constant(builder, entry_bb, f.type);
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
                int32_t off = cur->offset();
                if (live_offsets.count(off) > 0) {
                    current_def[off] = cur->operand(1);
                }
                insts_to_remove.insert(cur);
                if (stats_) stats_->stores_eliminated++;
            } else if (cur->opcode() == Opcode::load && aliases.count(cur->operand(0)) > 0) {
                int32_t off = cur->offset();
                Value* loaded_val = current_def[off];
                if (!loaded_val) {
                    loaded_val = get_or_create_zero_constant(builder, entry_bb, cur->type());
                    current_def[off] = loaded_val;
                }
                replace_all_uses(fn_, cur->result(), loaded_val);
                insts_to_remove.insert(cur);
                if (stats_) stats_->loads_eliminated++;
            }
            cur = next;
        }

        // Update branch arguments to successors
        for (BasicBlock* succ : bb->successors()) {
            if (!succ) continue;
            auto it_f = fields_for_block.find(succ);
            if (it_f == fields_for_block.end() || it_f->second.empty()) continue;

            Instruction* term = bb->terminator();
            if (!term) continue;

            for (const FieldInfo& f : it_f->second) {
                Value* outgoing = current_def[f.offset];
                if (!outgoing) {
                    outgoing = get_or_create_zero_constant(builder, entry_bb, f.type);
                }
                for_each_branch_target(term, [&](BranchTarget& bt) {
                    if (bt.block == succ) {
                        bt.args.push_back(outgoing);
                    }
                });
            }
        }

        // Recurse to dominator children
        for (const BasicBlock* child : dom.children(bb)) {
            self(self, const_cast<BasicBlock*>(child), current_def);
        }
    };

    std::unordered_map<int32_t, Value*> initial_defs;
    rename_traversal(rename_traversal, entry_bb, initial_defs);

    // 7. Remove dead instructions
    for (Instruction* inst : insts_to_remove) {
        if (inst && inst->parent()) {
            inst->parent()->remove_instruction(inst);
        }
    }
    if (stats_) stats_->allocations_eliminated++;

    // 8. Remove dead alias block parameters
    std::unordered_map<BasicBlock*, std::vector<uint32_t>> params_to_remove;
    for (const Value* v : aliases) {
        if (v && v->is_block_param()) {
            BasicBlock* def_bb = v->defining_block();
            if (def_bb) {
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

} // namespace brass
