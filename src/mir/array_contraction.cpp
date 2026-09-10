#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

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

static void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) inst->set_operand(i, new_val);
            }
            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                if (inst->branch_target().args[i] == old_val) inst->branch_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                if (inst->true_target().args[i] == old_val) inst->true_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                if (inst->false_target().args[i] == old_val) inst->false_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                if (inst->default_target().args[i] == old_val) inst->default_target().args[i] = new_val;
            }
            for (auto& sc : inst->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    if (sc.target.args[i] == old_val) sc.target.args[i] = new_val;
                }
            }
        }
    }
}

static Value* unwrap_boxed_val(Value* val) {
    while (val && val->is_instruction()) {
        Instruction* def = val->defining_instruction();
        if (!def) break;
        if (def->opcode() == Opcode::call && def->symbol() == "box_f64" && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        if ((def->opcode() == Opcode::bitcast_i64_f64 || def->opcode() == Opcode::bitcast_f64_i64) && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        break;
    }
    return val;
}

static Value* unwrap_boxed_index(Value* val) {
    while (val && val->is_instruction()) {
        Instruction* def = val->defining_instruction();
        if (!def) break;
        if (def->opcode() == Opcode::call && def->symbol() == "box_f64" && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        if ((def->opcode() == Opcode::bitcast_i64_f64 || def->opcode() == Opcode::bitcast_f64_i64 ||
             def->opcode() == Opcode::sitofp_f64_i64 || def->opcode() == Opcode::sitofp_f64_i32 ||
             def->opcode() == Opcode::fptosi_i64 || def->opcode() == Opcode::fptosi_i32 ||
             def->opcode() == Opcode::sext_i64 || def->opcode() == Opcode::zext_i64) && def->operand_count() >= 1) {
            val = def->operand(0);
            continue;
        }
        break;
    }
    return val;
}

static bool are_indices_congruent(Value* idx1, Value* idx2, Value* iv_param = nullptr) {
    if (!idx1 || !idx2) return false;
    if (idx1 == idx2) return true;
    Value* u1 = unwrap_boxed_index(idx1);
    Value* u2 = unwrap_boxed_index(idx2);
    if (u1 == u2) return true;
    if (iv_param && (u1 == iv_param || idx1 == iv_param) && (u2 == iv_param || idx2 == iv_param)) {
        return true;
    }
    return false;
}

static bool does_array_escape(const Function& fn, const LoopInfo& loop, const Value* alloc_val) {
    if (!alloc_val) return true;

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();

            // Return with alloc_val escapes
            if (op == Opcode::ret) {
                if (inst->operand_count() > 0 && inst->operand(0) == alloc_val) return true;
            }

            // Storing alloc_val into memory escapes
            if (op == Opcode::store && inst->operand(1) == alloc_val) return true;
            if (op == Opcode::store_indexed && inst->operand(2) == alloc_val) return true;

            // Passing alloc_val to a function call
            if (is_call(op)) {
                std::string_view sym = inst->symbol();
                bool is_allowed = (sym == "bronze_elem_get" || sym == "bronze_elem_set" ||
                                   sym == "bronze_prop_set" || sym == "bronze_prop_get" ||
                                   sym == "write_barrier");
                if (!is_allowed) {
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        if (inst->operand(i) == alloc_val) return true;
                    }
                } else if (sym == "bronze_elem_get" || sym == "bronze_elem_set") {
                    if (!loop.contains(bb)) {
                        for (size_t i = 0; i < inst->operand_count(); ++i) {
                            if (inst->operand(i) == alloc_val) return true;
                        }
                    }
                }
            } else if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
                if (!loop.contains(bb) && inst->operand_count() > 0 && inst->operand(0) == alloc_val) {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

bool contract_arrays_in_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const ArrayContractionOptions& options
) {
    (void)dom;
    bool any_contracted = false;

    // 1. Identify allocation instructions whose result is used in this loop
    std::vector<Instruction*> alloc_insts;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::call && is_allocation_callee(inst->symbol())) {
                if (inst->result() && !does_array_escape(fn, loop, inst->result())) {
                    alloc_insts.push_back(inst);
                }
            }
        }
    }

    for (Instruction* alloc_inst : alloc_insts) {
        Value* alloc_val = alloc_inst->result();
        if (!alloc_val) continue;

        // Check stores and loads in the loop
        struct StoreInfo {
            Instruction* inst = nullptr;
            Value* index = nullptr;
            Value* value = nullptr;
        };
        struct LoadInfo {
            Instruction* inst = nullptr;
            Value* index = nullptr;
        };

        std::vector<StoreInfo> stores;
        std::vector<LoadInfo> loads;

        for (BasicBlock* bb : loop.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst) continue;
                Opcode op = inst->opcode();

                if (op == Opcode::store_indexed && inst->operand(0) == alloc_val) {
                    stores.push_back({inst, inst->operand(1), inst->operand(2)});
                } else if (op == Opcode::load_indexed && inst->operand(0) == alloc_val) {
                    loads.push_back({inst, inst->operand(1)});
                } else if (op == Opcode::call && inst->symbol() == "bronze_elem_set" && inst->operand(0) == alloc_val) {
                    stores.push_back({inst, inst->operand(1), inst->operand(2)});
                } else if (op == Opcode::call && inst->symbol() == "bronze_elem_get" && inst->operand(0) == alloc_val) {
                    loads.push_back({inst, inst->operand(1)});
                }
            }
        }

        if (stores.empty() || loads.empty()) continue;

        // Check if all loads have a dominating congruent store
        bool all_loads_matched = true;
        std::vector<std::pair<Instruction*, Value*>> load_replacements;

        for (const auto& ld : loads) {
            Instruction* matching_store = nullptr;
            Value* forwarded_val = nullptr;

            // Search backward in the same block for the most recent matching store
            for (auto it = stores.rbegin(); it != stores.rend(); ++it) {
                const auto& st = *it;
                if (st.inst->parent() == ld.inst->parent()) {
                    // Check order in same block
                    bool store_before_load = false;
                    for (Instruction* cur = st.inst->next(); cur != nullptr; cur = cur->next()) {
                        if (cur == ld.inst) {
                            store_before_load = true;
                            break;
                        }
                    }
                    if (store_before_load && are_indices_congruent(st.index, ld.index)) {
                        matching_store = st.inst;
                        forwarded_val = st.value;
                        break;
                    }
                }
            }

            if (matching_store && forwarded_val) {
                load_replacements.push_back({ld.inst, forwarded_val});
            } else {
                all_loads_matched = false;
                break;
            }
        }

        if (!all_loads_matched || load_replacements.empty()) continue;

        // 2. Perform contraction:
        // Replace all loaded values with the forwarded stored values
        for (auto& [ld_inst, fwd_val] : load_replacements) {
            Value* ld_res = ld_inst->result();
            if (ld_res) {
                // If ld_res is immediately unboxed and fwd_val was boxed, simplify directly
                for (BasicBlock* bb : loop.blocks()) {
                    if (!bb) continue;
                    for (Instruction* inst : *bb) {
                        if (!inst) continue;
                        if ((inst->opcode() == Opcode::bitcast_f64_i64 || (inst->opcode() == Opcode::call && inst->symbol() == "unbox_f64")) &&
                            inst->operand_count() > 0 && inst->operand(0) == ld_res) {
                            Value* raw_fwd = unwrap_boxed_val(fwd_val);
                            replace_all_uses(fn, inst->result(), raw_fwd);
                        }
                    }
                }
                replace_all_uses(fn, ld_res, fwd_val);
            }
            ld_inst->parent()->remove_instruction(ld_inst);
            if (options.stats) options.stats->loads_eliminated++;
        }

        // 3. Remove now-dead stores to alloc_val
        for (const auto& st : stores) {
            st.inst->parent()->remove_instruction(st.inst);
            if (options.stats) options.stats->stores_eliminated++;
        }

        // 4. Remove any remaining initialization/property sets or write barriers for alloc_val
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            std::vector<Instruction*> to_remove;
            for (Instruction* inst : *bb) {
                if (!inst) continue;
                Opcode op = inst->opcode();
                if (op == Opcode::write_barrier && inst->operand_count() > 0 && inst->operand(0) == alloc_val) {
                    to_remove.push_back(inst);
                } else if (op == Opcode::call && inst->symbol() == "bronze_prop_set" && inst->operand_count() > 0 && inst->operand(0) == alloc_val) {
                    to_remove.push_back(inst);
                }
            }
            for (Instruction* inst : to_remove) {
                bb->remove_instruction(inst);
            }
        }

        // 5. Remove the allocation instruction itself
        if (alloc_inst->parent()) {
            alloc_inst->parent()->remove_instruction(alloc_inst);
            if (options.stats) options.stats->allocations_eliminated++;
            if (options.stats) options.stats->arrays_contracted++;
            any_contracted = true;
        }
    }

    return any_contracted;
}

bool array_contraction_pass(
    Function& fn,
    const DominatorTree& dom,
    const ArrayContractionOptions& options
) {
    bool changed = false;

    // First, check if there are adjacent loops where loop 1 produces a buffer and loop 2 consumes it
    // Attempting loop fusion on them enables array contraction
    LoopFusionOptions fusion_opts;
    loop_fusion_pass(fn, dom, fusion_opts);

    fn.rebuild_cfg_predecessors();
    DominatorTree current_dom(fn);
    LoopAnalysis la(fn, current_dom);

    for (LoopInfo* loop : la.post_order_loops()) {
        if (!loop) continue;
        if (contract_arrays_in_loop(fn, *loop, current_dom, options)) {
            changed = true;
        }
    }

    return changed;
}

} // namespace brass
