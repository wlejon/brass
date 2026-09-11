#include "il_lowering.hpp"
#include "il_lowering_coro.hpp"
#include "il_runtime.hpp"
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/loop_unswitch.hpp>
#include <brass/mir/jump_threading.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <cstring>
#include <iostream>

namespace brass::il {

Type lower_type(BronzeType t) {
    switch (t) {
        case BronzeType::Void: return Type::void_type();
        case BronzeType::Bool: return Type::i32();
        case BronzeType::I32: return Type::i32();
        case BronzeType::F64: return Type::f64();
        case BronzeType::Str: return Type::ptr();
        case BronzeType::Dynamic: return Type::i64();
        case BronzeType::Unknown: return Type::i64();
    }
    return Type::i64();
}

IlLowering::IlLowering(const TranslatorOptions& options, DiagnosticReporter* diag)
    : options_(options), diag_(diag), prop_lowering_(options.enable_pic) {}

Value* IlLowering::ensure_type(Value* val, Type target_type, Builder& b) {
    if (!val || val->type() == target_type) return val;

    Type src_type = val->type();
    if (src_type == Type::f64() && target_type == Type::i32()) {
        return b.build_fptosi_i32(val);
    }
    if (src_type == Type::i32() && target_type == Type::f64()) {
        return b.build_sitofp_f64_i32(val);
    }
    if (src_type == Type::i32() && target_type == Type::i64()) {
        return b.build_sext_i64(val);
    }
    if (src_type == Type::i64() && target_type == Type::i32()) {
        return b.build_trunc_i32(val);
    }
    if (src_type == Type::f64() && target_type == Type::i64()) {
        if (val->defining_instruction() &&
            val->defining_instruction()->opcode() == Opcode::bitcast_f64_i64 &&
            val->defining_instruction()->operand(0)->type() == Type::i64()) {
            return val->defining_instruction()->operand(0);
        }
        Value* bits = b.build_bitcast_i64_f64(val);
        Value* abs_bits = b.build_and(bits, b.build_iconst_i64(static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL)));
        Value* is_nan = b.build_ugt(abs_bits, b.build_iconst_i64(static_cast<int64_t>(0x7FF0000000000000ULL)));
        return b.build_select(is_nan, b.build_iconst_i64(static_cast<int64_t>(0x7FF8000000000000ULL)), bits);
    }
    if (src_type == Type::i64() && target_type == Type::f64()) {
        return b.build_bitcast_f64_i64(val);
    }
    return val;
}

std::string IlLowering::resolve_callee(const std::string& callee_name) const {
    auto it_fn = caller_to_callee_map_.find(current_fn_idx_);
    if (it_fn != caller_to_callee_map_.end()) {
        auto it_name = it_fn->second.find(callee_name);
        if (it_name != it_fn->second.end()) {
            return it_name->second;
        }
    }
    return callee_name;
}

std::string IlLowering::resolve_create_func_callee(const std::string& callee_name) {
    auto it_caller = caller_create_func_targets_.find(current_fn_idx_);
    if (it_caller != caller_create_func_targets_.end()) {
        auto it_targets = it_caller->second.find(callee_name);
        if (it_targets != it_caller->second.end() && !it_targets->second.empty()) {
            size_t k = create_func_counter_[callee_name]++;
            if (k < it_targets->second.size()) {
                return it_targets->second[k];
            }
        }
    }
    return resolve_callee(callee_name);
}

Value* IlLowering::get_key_id(Builder& b, uint32_t key_idx) {
    Value* map_addr = b.build_func_addr("__bronze_key_map");
    return b.build_load(Type::i32(), map_addr, static_cast<int32_t>(key_idx * sizeof(uint32_t)));
}

uint32_t IlLowering::find_key_constant(const std::string& name) const {
    for (size_t i = 0; i < options_.key_constants.size(); ++i) {
        if (options_.key_constants[i] == name) return static_cast<uint32_t>(i);
    }
    return 0;
}

Value* IlLowering::get_val_by_id(uint32_t id, Builder& b, const std::unordered_map<uint32_t, Value*>& val_map) {
    Value* result = nullptr;
    if (module_env_regs_.count(id)) {
        Value* env_addr = b.build_func_addr("__bronze_module_env");
        result = b.build_load(Type::i64(), env_addr, 0);
    } else if (current_fn_frame_ptr_ != nullptr) {
        auto it = current_fn_slot_of_.find(id);
        if (it != current_fn_slot_of_.end()) {
            result = b.build_load(Type::i64(), current_fn_frame_ptr_, static_cast<int32_t>(16 + it->second * 8));
        }
    }
    if (!result) {
        auto it = val_map.find(id);
        if (it != val_map.end()) result = it->second;
    }
    return result;
}

void IlLowering::set_inst_result(uint32_t result_id, Value* res_val, Builder& b, std::unordered_map<uint32_t, Value*>& val_map) {
    if (result_id == UINT32_MAX || !res_val) return;
    val_map[result_id] = res_val;
    if (current_fn_frame_ptr_ != nullptr) {
        auto it = current_fn_slot_of_.find(result_id);
        if (it != current_fn_slot_of_.end()) {
            Value* stored_val = ensure_type(res_val, Type::i64(), b);
            b.build_store(Type::i64(), current_fn_frame_ptr_, static_cast<int32_t>(16 + it->second * 8), stored_val);
        }
    }
}

static bool instructions_are_identical(const BronzeInstruction& a, const BronzeInstruction& b) {
    if (a.op != b.op || a.result_type != b.result_type || a.result_id != b.result_id) return false;
    if (a.operands != b.operands) return false;
    if (std::memcmp(&a.imm_f64, &b.imm_f64, sizeof(double)) != 0) return false;
    if (a.imm_i64 != b.imm_i64 || a.imm_bool != b.imm_bool) return false;
    if (a.box_type != b.box_type || a.raw_unbox != b.raw_unbox) return false;
    if (a.callee_name != b.callee_name || a.string_literal != b.string_literal) return false;
    if (a.depth != b.depth || a.index != b.index || a.param_count != b.param_count) return false;
    if (a.target.block_id != b.target.block_id || a.target.args != b.target.args) return false;
    if (a.else_target.block_id != b.else_target.block_id || a.else_target.args != b.else_target.args) return false;
    return true;
}

static bool functions_are_identical(const BronzeFunction& a, const BronzeFunction& b) {
    if (a.params != b.params || a.return_type != b.return_type || a.is_exported != b.is_exported) return false;
    if (a.blocks.size() != b.blocks.size()) return false;
    for (size_t i = 0; i < a.blocks.size(); ++i) {
        const auto& ba = a.blocks[i];
        const auto& bb = b.blocks[i];
        if (ba.id != bb.id || ba.params != bb.params || ba.handler_id != bb.handler_id) return false;
        if (ba.instructions.size() != bb.instructions.size()) return false;
        for (size_t j = 0; j < ba.instructions.size(); ++j) {
            if (!instructions_are_identical(ba.instructions[j], bb.instructions[j])) return false;
        }
    }
    return true;
}

std::unique_ptr<Module> IlLowering::lower_module(const BronzeModuleAST& ast) {
    auto mod = std::make_unique<Module>(ast.name);
    mod->set_allow_fp_reassociation(options_.allow_fp_reassociation);
    current_file_id_ = ast.name.empty() ? 0 : mod->debug_context().get_or_add_file(ast.name);

    // Register external runtime helper functions
    register_all_module_external_symbols(mod.get(), options_.entry_symbol);

    current_ast_ = &ast;




    // 1. Forward-declare all functions (uniquifying any duplicate function names from Bronze)
    std::unordered_map<std::string, std::vector<size_t>> name_to_indices;
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        name_to_indices[ast.functions[i].name].push_back(i);
    }

    caller_to_callee_map_.clear();
    std::vector<std::string> resolved_names(ast.functions.size());
    std::unordered_set<std::string> curry_names;
    for (const auto& [name, indices] : name_to_indices) {
        if (indices.size() == 1) {
            resolved_names[indices[0]] = name;
        } else {
            // Check if all duplicates are identical
            bool all_identical = true;
            for (size_t k = 1; k < indices.size(); ++k) {
                if (!functions_are_identical(ast.functions[indices[0]], ast.functions[indices[k]])) {
                    all_identical = false;
                    break;
                }
            }
            if (all_identical) {
                resolved_names[indices[0]] = name;
                for (size_t k = 1; k < indices.size(); ++k) {
                    resolved_names[indices[k]] = ""; // skip duplicate definition
                }
                continue;
            }

            if (name == "main") {
                size_t entry_idx = indices.back();
                for (size_t idx : indices) {
                    if (ast.functions[idx].return_type == BronzeType::Void && ast.functions[idx].params.empty()) {
                        entry_idx = idx;
                        break;
                    }
                }
                size_t user_k = 0;
                for (size_t idx : indices) {
                    if (idx == entry_idx) {
                        resolved_names[idx] = "main";
                    } else {
                        resolved_names[idx] = "main$" + std::to_string(user_k++);
                    }
                }
                std::string target_callee = (indices[0] == entry_idx && indices.size() > 1) ? resolved_names[indices[1]] : resolved_names[indices[0]];
                for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
                    caller_to_callee_map_[fn_idx]["main"] = target_callee;
                }
                continue;
            }

            // Check if any function in indices has a self-referential create.func (curry pattern)
            size_t non_leaf_idx = SIZE_MAX;
            for (size_t idx : indices) {
                bool has_self_create = false;
                for (const auto& blk : ast.functions[idx].blocks) {
                    for (const auto& inst : blk.instructions) {
                        if (inst.op == BronzeOp::CreateFunc && inst.callee_name == name) {
                            has_self_create = true;
                            break;
                        }
                    }
                    if (has_self_create) break;
                }
                if (has_self_create) {
                    non_leaf_idx = idx;
                    break;
                }
            }

            if (non_leaf_idx != SIZE_MAX) {
                // Curry pattern: non-leaf closure retains name; leaf closure gets $leaf
                curry_names.insert(name);
                for (size_t idx : indices) {
                    if (idx == non_leaf_idx) {
                        resolved_names[idx] = name;
                    } else {
                        resolved_names[idx] = name + "$leaf";
                    }
                }
                for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
                    if (fn_idx == non_leaf_idx) {
                        caller_to_callee_map_[fn_idx][name] = name + "$leaf";
                    } else {
                        caller_to_callee_map_[fn_idx][name] = name;
                    }
                }
            } else {
                for (size_t k = 0; k < indices.size(); ++k) {
                    resolved_names[indices[k]] = name + "$" + std::to_string(k);
                }
            }
        }
    }

    // Map callers to callees for duplicate function names based on CreateFunc instantiation order and lexical scope inheritance
    caller_create_func_targets_.clear();
    std::unordered_map<std::string, size_t> create_func_occurrence;
    std::unordered_map<size_t, size_t> closure_parent_fn;
    for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
        for (const auto& blk : ast.functions[fn_idx].blocks) {
            for (const auto& inst : blk.instructions) {
                if (inst.op == BronzeOp::CreateFunc) {
                    const std::string& cname = inst.callee_name;
                    auto it = name_to_indices.find(cname);
                    if (it != name_to_indices.end() && it->second.size() > 1 && cname != "main" && !curry_names.count(cname)) {
                        size_t k = create_func_occurrence[cname]++;
                        if (k < it->second.size()) {
                            size_t child_fn_idx = it->second[k];
                            const std::string& child_res_name = resolved_names[child_fn_idx];
                            if (!child_res_name.empty()) {
                                caller_create_func_targets_[fn_idx][cname].push_back(child_res_name);
                                caller_to_callee_map_[fn_idx][cname] = child_res_name;
                                closure_parent_fn[child_fn_idx] = fn_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    // Propagate mappings from parent scopes down to child closures
    for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
        auto pit = closure_parent_fn.find(fn_idx);
        if (pit != closure_parent_fn.end()) {
            size_t p = pit->second;
            auto it_p = caller_to_callee_map_.find(p);
            if (it_p != caller_to_callee_map_.end()) {
                for (const auto& [cname, target] : it_p->second) {
                    if (!curry_names.count(cname) && caller_to_callee_map_[fn_idx].find(cname) == caller_to_callee_map_[fn_idx].end()) {
                        caller_to_callee_map_[fn_idx][cname] = target;
                    }
                }
            }
        }
    }

    // Default fallback for any remaining duplicate function references
    for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
        for (const auto& [name, indices] : name_to_indices) {
            if (indices.size() > 1 && name != "main" && !curry_names.count(name) && !resolved_names[indices[0]].empty()) {
                if (caller_to_callee_map_[fn_idx].find(name) == caller_to_callee_map_[fn_idx].end()) {
                    caller_to_callee_map_[fn_idx][name] = resolved_names[indices[0]];
                }
            }
        }
    }

    std::unordered_map<std::string, uint32_t> callee_param_counts;
    std::unordered_set<std::string> closure_functions;
    for (const auto& fn : ast.functions) {
        for (const auto& blk : fn.blocks) {
            for (const auto& inst : blk.instructions) {
                if (inst.op == BronzeOp::CreateFunc) {
                    callee_param_counts[inst.callee_name] = inst.param_count;
                    closure_functions.insert(inst.callee_name);
                }
            }
        }
    }

    for (size_t i = 0; i < ast.functions.size(); ++i) {
        if (resolved_names[i].empty()) continue;
        const auto& fn_ast = ast.functions[i];
        const std::string& fn_name = resolved_names[i];

        std::vector<Type> param_types;
        for (const auto& p : fn_ast.params) {
            param_types.push_back(lower_type(p.second));
        }
        Type ret_type = lower_type(fn_ast.return_type);
        Function* fn = mod->create_function(fn_name, ret_type, Span<const Type>(param_types.data(), param_types.size()));
        fn->set_allow_fp_reassociation(options_.allow_fp_reassociation);

        if (fn_name != "main") {
            std::vector<Type> wrapper_param_types = {Type::i64(), Type::i64(), Type::i32(), Type::ptr()};
            Function* wfn = mod->create_function(
                "__wrapper_" + fn_name,
                Type::i64(),
                Span<const Type>(wrapper_param_types.data(), wrapper_param_types.size()));
            wfn->set_allow_fp_reassociation(options_.allow_fp_reassociation);
        }
    }

    // 2. Lower each function body
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        if (resolved_names[i].empty()) continue;
        current_fn_idx_ = i;
        const auto& fn_ast = ast.functions[i];
        const std::string& fn_name = resolved_names[i];
        if (!lower_function(fn_ast, *mod, fn_name)) {
            return nullptr;
        }
        if (fn_name != "main") {
            uint32_t arity = static_cast<uint32_t>(fn_ast.params.size());
            auto it_ar = callee_param_counts.find(fn_ast.name);
            if (it_ar != callee_param_counts.end()) {
                arity = it_ar->second;
            } else {
                auto it_res = callee_param_counts.find(fn_name);
                if (it_res != callee_param_counts.end()) {
                    arity = it_res->second;
                }
            }
            bool is_closure = closure_functions.count(fn_ast.name) || closure_functions.count(fn_name);
            if (!emit_wrapper(fn_ast, *mod, fn_name, arity, is_closure)) {
                return nullptr;
            }
        }
    }

    // 3. Verify module
    if (!verify_module(*mod, diag_)) {
        return nullptr;
    }

    // 4. Transform coroutines into state machines
    CoroTransformOptions coro_opts;
    coro_opts.first_slot_index = 1;
    CoroTransformPass(coro_opts).run_on_module(*mod);

    // 5. Optionally optimize module
    if (options_.enable_optimizations) {
        LoopOptOptions opt_opts;
        opt_opts.enable_fp_reassociation = options_.allow_fp_reassociation;
        opt_opts.enable_f64_demote = options_.enable_f64_demote;
        opt_opts.enable_vectorize = options_.enable_vectorize;
        opt_opts.enable_slp = options_.enable_slp;
        opt_opts.enable_loop_tile = options_.enable_loop_tile;
        opt_opts.tile_size_i = options_.tile_size;
        opt_opts.tile_size_j = options_.tile_size;
        opt_opts.tile_size_k = options_.tile_size;
        opt_opts.enable_sroa = options_.enable_sroa;
        opt_opts.enable_gvn = options_.enable_gvn;
        opt_opts.enable_sccp = options_.enable_sccp;
        opt_opts.enable_guard_elim = options_.enable_guard_elim;
        opt_opts.enable_cfg_simplify = options_.enable_cfg_simplify;
        opt_opts.enable_loop_unswitch = options_.enable_loop_unswitch;
        opt_opts.enable_jump_threading = options_.enable_jump_threading;
        opt_opts.enable_trace_layout = options_.enable_trace_layout;
        opt_opts.enable_partial_escape = options_.enable_partial_escape;
        opt_opts.enable_allocation_sinking = options_.enable_allocation_sinking;
        opt_opts.enable_loop_fusion = options_.enable_loop_fusion;
        opt_opts.enable_loop_distribution = options_.enable_loop_distribution;
        opt_opts.enable_array_contraction = options_.enable_array_contraction;
        opt_opts.dump_loop_transform_stats = options_.dump_loop_transform_stats;
        opt_opts.stats = options_.loop_transform_stats_collector;
        opt_opts.pea_stats = options_.pea_stats_collector;
        opt_opts.demote_stats = options_.demote_stats_collector;
        opt_opts.enable_avx2 = options_.enable_avx2;
        opt_opts.enable_fma = options_.enable_fma;
        opt_opts.vector_width = options_.vector_width;
        opt_opts.dump_fma_stats = options_.dump_fma_stats;
        opt_opts.fma_stats = options_.fma_stats_collector;
        opt_opts.enable_parallel_loops = options_.enable_parallel_loops;
        opt_opts.parallel_threshold = options_.parallel_threshold;
        opt_opts.parallel_workers = options_.parallel_workers;
        opt_opts.dump_parallel_stats = options_.dump_parallel_stats;
        opt_opts.parallel_stats = options_.parallel_stats_collector;
        if (options_.enable_sroa) {
            sroa_module(*mod);
        }
        if (options_.enable_gvn) {
            gvn_module(*mod);
        }
        if (options_.enable_gvn_pre) {
            GvnPreOptions pre_opts;
            pre_opts.stats = options_.pre_stats_collector;
            gvn_pre_module(*mod, pre_opts);
        }
        if (options_.enable_sccp) {
            SccpOptions sccp_opts;
            sccp_opts.enable_guard_elim = options_.enable_guard_elim;
            sccp_module(*mod, sccp_opts);
        }
        if (options_.enable_cfg_simplify) {
            cfg_simplify_module(*mod);
        }
        if (options_.enable_loop_unswitch) {
            unswitch_loops_in_module(*mod);
        }
        if (options_.enable_jump_threading) {
            jump_thread_module(*mod);
        }
        if ((options_.enable_loop_unswitch || options_.enable_jump_threading) && options_.enable_cfg_simplify) {
            cfg_simplify_module(*mod);
        }
        if (options_.enable_inlining) {
            InlinerOptions inliner_opts;
            inliner_opts.enable_sroa = options_.enable_sroa;
            inliner_opts.enable_gvn = options_.enable_gvn;
            optimize_module_ipo(*mod, inliner_opts, opt_opts);
        } else {
            optimize_module_loops(*mod, opt_opts);
        }
        if (options_.enable_wbe) {
            WriteBarrierElimination wbe(options_.dump_wbe_stats);
            wbe.run_on_module(*mod);
        }
        if (!verify_module(*mod, diag_)) {
            return nullptr;
        }
    }

    return mod;
}

bool IlLowering::lower_function(const BronzeFunction& fn_ast, Module& mod, const std::string& fn_name) {
    Function* fn = mod.get_function(fn_name);
    if (!fn) return false;

    module_env_regs_.clear();
    current_fn_slot_of_.clear();
    create_func_counter_.clear();
    current_fn_frame_ptr_ = nullptr;
    uint32_t total_slots = 0;

    for (const auto& p : fn_ast.params) {
        if (p.second == BronzeType::Dynamic || p.second == BronzeType::Unknown) {
            if (!current_fn_slot_of_.count(p.first)) {
                current_fn_slot_of_[p.first] = total_slots++;
            }
        }
    }
    for (const auto& blk : fn_ast.blocks) {
        for (const auto& p : blk.params) {
            if (p.second == BronzeType::Dynamic || p.second == BronzeType::Unknown) {
                if (!current_fn_slot_of_.count(p.first)) {
                    current_fn_slot_of_[p.first] = total_slots++;
                }
            }
        }
        for (const auto& inst : blk.instructions) {
            if (inst.result_id != UINT32_MAX &&
                (inst.result_type == BronzeType::Dynamic || inst.result_type == BronzeType::Unknown)) {
                if (!current_fn_slot_of_.count(inst.result_id)) {
                    current_fn_slot_of_[inst.result_id] = total_slots++;
                }
            }
        }
    }

    Builder b(mod);
    b.set_function(fn);

    std::unordered_map<uint32_t, BasicBlock*> block_map;
    std::unordered_map<uint32_t, Value*> val_map;

    // 1. Create all basic blocks in advance
    for (const auto& blk_ast : fn_ast.blocks) {
        std::string bb_name = "b" + std::to_string(blk_ast.id);
        BasicBlock* bb = b.append_block(bb_name);
        block_map[blk_ast.id] = bb;
    }

    // 2. Set up entry block parameters from function arguments
    if (!fn_ast.blocks.empty() && block_map.count(fn_ast.blocks[0].id)) {
        BasicBlock* entry_bb = block_map[fn_ast.blocks[0].id];
        for (size_t i = 0; i < fn_ast.params.size(); ++i) {
            uint32_t param_id = fn_ast.params[i].first;
            Type param_type = lower_type(fn_ast.params[i].second);
            Value* param_val = b.add_block_param(entry_bb, param_type);
            val_map[param_id] = param_val;
        }

        b.position_at_end(entry_bb);
        if (total_slots > 0) {
            current_fn_frame_ptr_ = b.build_call("bronze_gc_frame_push", Type::ptr(),
                                                {b.build_iconst_i32(static_cast<int32_t>(total_slots))});
            for (size_t i = 0; i < fn_ast.params.size(); ++i) {
                uint32_t param_id = fn_ast.params[i].first;
                if (current_fn_slot_of_.count(param_id)) {
                    Value* pval = ensure_type(val_map[param_id], Type::i64(), b);
                    b.build_store(Type::i64(), current_fn_frame_ptr_,
                                  static_cast<int32_t>(16 + current_fn_slot_of_[param_id] * 8), pval);
                }
            }
        }

        if (fn_name == "main") {
            Value* env_addr = b.build_func_addr("__bronze_module_env");
            Value* count_val = b.build_iconst_i64(1);
            b.build_call("bronze_register_value_cells", Type::void_type(), {env_addr, count_val});
            Value* tpl_addr = b.build_func_addr("__bronze_template_cells");
            Value* tpl_cells_count = b.build_iconst_i64(1024);
            b.build_call("bronze_register_value_cells", Type::void_type(), {tpl_addr, tpl_cells_count});
            const std::string key_sym = (options_.entry_symbol.empty() || options_.entry_symbol == "main" || options_.entry_symbol == "bronze_main")
                                            ? "bronze_main_key_constants"
                                            : (options_.entry_symbol + "_key_constants");
            Value* manifest_addr = b.build_func_addr(key_sym);
            Value* map_addr = b.build_func_addr("__bronze_key_map");
            b.build_call("bronze_register_key_manifest", Type::void_type(), {manifest_addr, map_addr});
            if (options_.enable_census && options_.census_site_count > 0) {
                Value* out_path = b.build_func_addr("__bronze_census_out_path");
                Value* sites_addr = b.build_func_addr("__bronze_census_sites");
                Value* site_count = b.build_iconst_i32(static_cast<int32_t>(options_.census_site_count));
                b.build_call("bronze_census_register", Type::void_type(), {out_path, sites_addr, site_count, map_addr});
            }
            for (size_t f = 0; f < options_.source_files.size(); ++f) {
                if (options_.source_files[f].entry_count == 0) continue;
                Value* text_addr = b.build_func_addr("__bronze_source_text_" + std::to_string(f));
                Value* text_len = b.build_iconst_i32(static_cast<int32_t>(options_.source_files[f].text_len));
                Value* entries_addr = b.build_func_addr("__bronze_source_entries_" + std::to_string(f));
                Value* entries_count = b.build_iconst_i32(static_cast<int32_t>(options_.source_files[f].entry_count));
                b.build_call("bronze_register_fn_sources", Type::void_type(), {text_addr, text_len, entries_addr, entries_count});
            }
        }
    }

    // 3. Set up non-entry block parameters
    for (size_t i = 1; i < fn_ast.blocks.size(); ++i) {
        const auto& blk_ast = fn_ast.blocks[i];
        BasicBlock* bb = block_map[blk_ast.id];
        for (const auto& p : blk_ast.params) {
            uint32_t param_id = p.first;
            Type param_type = lower_type(p.second);
            Value* param_val = b.add_block_param(bb, param_type);
            val_map[param_id] = param_val;
        }
    }

    // 4. Lower instructions block by block
    for (const auto& blk_ast : fn_ast.blocks) {
        BasicBlock* bb = block_map[blk_ast.id];
        b.position_at_end(bb);
        uint32_t cont_counter = 0;

        if (blk_ast.id != fn_ast.blocks[0].id && current_fn_frame_ptr_ != nullptr) {
            for (const auto& p : blk_ast.params) {
                if (current_fn_slot_of_.count(p.first)) {
                    Value* pval = ensure_type(val_map[p.first], Type::i64(), b);
                    b.build_store(Type::i64(), current_fn_frame_ptr_,
                                  static_cast<int32_t>(16 + current_fn_slot_of_[p.first] * 8), pval);
                }
            }
        }

        for (const auto& inst_ast : blk_ast.instructions) {
            if (current_file_id_ != 0 && inst_ast.line > 0) {
                b.set_current_loc(DebugLoc(current_file_id_, inst_ast.line, inst_ast.column));
            }
            if (!lower_instruction(inst_ast, b, fn, val_map, block_map, blk_ast.handler_id, blk_ast.id, &cont_counter)) {
                return false;
            }
        }
    }

    fn->rebuild_cfg_predecessors();
    return true;
}

bool IlLowering::emit_wrapper(const BronzeFunction& fn_ast, Module& mod, const std::string& fn_name, uint32_t declared_param_count, bool is_closure) {
    Function* wfn = mod.get_function("__wrapper_" + fn_name);
    if (!wfn) return false;

    Builder b(mod);
    b.set_function(wfn);

    BasicBlock* bb = b.append_block("entry");
    Value* val_env = b.add_block_param(bb, Type::i64());
    Value* val_this = b.add_block_param(bb, Type::i64());
    Value* val_argc = b.add_block_param(bb, Type::i32());
    Value* val_argv = b.add_block_param(bb, Type::ptr());
    b.position_at_end(bb);

    bool needs_env = false;
    bool needs_this = false;
    bool needs_arguments = false;
    bool has_rest = false;
    bool is_strict = false;
    size_t first_source_param = 0;

    auto it_meta = options_.function_meta.find(fn_ast.name);
    if (it_meta == options_.function_meta.end()) {
        it_meta = options_.function_meta.find(fn_name);
    }
    if (it_meta != options_.function_meta.end()) {
        needs_env = it_meta->second.needs_env;
        needs_this = it_meta->second.needs_this;
        needs_arguments = it_meta->second.needs_arguments;
        has_rest = it_meta->second.has_rest_param;
        is_strict = it_meta->second.is_strict;
        first_source_param = it_meta->second.first_source_param;
    } else {
        bool has_module_env_get = false;
        bool uses_p0_as_super_this = false;
        bool uses_p0_as_env = false;
        uint32_t param0_id = fn_ast.params.empty() ? UINT32_MAX : fn_ast.params[0].first;
        for (const auto& blk : fn_ast.blocks) {
            for (const auto& inst : blk.instructions) {
                if (inst.op == BronzeOp::ModuleEnvGet) has_module_env_get = true;
                if (inst.op == BronzeOp::SuperCall && inst.operands.size() >= 2 && inst.operands[1] == param0_id) {
                    uses_p0_as_super_this = true;
                }
                if (inst.op == BronzeOp::EnvGet || inst.op == BronzeOp::EnvSet ||
                    inst.op == BronzeOp::EnvCreate || inst.op == BronzeOp::CreateFunc) {
                    if (!inst.operands.empty() && inst.operands[0] == param0_id) {
                        uses_p0_as_env = true;
                    }
                }
            }
        }
        if (has_module_env_get || uses_p0_as_super_this) {
            needs_env = false;
            needs_this = true;
        } else if (uses_p0_as_env) {
            needs_env = true;
            needs_this = (fn_ast.params.size() > declared_param_count + 1);
        } else {
            bool param0_is_this = false;
            for (const auto& blk : fn_ast.blocks) {
                for (const auto& inst : blk.instructions) {
                    if (inst.op == BronzeOp::PropGet || inst.op == BronzeOp::PropSet ||
                        inst.op == BronzeOp::MethodCall || inst.op == BronzeOp::ElemGet ||
                        inst.op == BronzeOp::ElemSet) {
                        if (!inst.operands.empty() && inst.operands[0] == param0_id) {
                            param0_is_this = true;
                            break;
                        }
                    } else if (inst.op == BronzeOp::Ret) {
                        if (!inst.operands.empty() && inst.operands[0] == param0_id) {
                            param0_is_this = true;
                            break;
                        }
                    }
                }
                if (param0_is_this) break;
            }
            if (param0_is_this) {
                needs_env = false;
                needs_this = true;
            } else {
                needs_env = is_closure;
                needs_this = false;
            }
        }
        first_source_param = (needs_env ? 1 : 0) + (needs_this ? 1 : 0);
    }

    const size_t named_count = (fn_ast.params.size() > first_source_param + (has_rest ? 1 : 0))
        ? (fn_ast.params.size() - first_source_param - (has_rest ? 1 : 0))
        : 0;

    std::vector<Value*> loaded;
    for (size_t n = 0; n < named_count; ++n) {
        Value* raw = nullptr;
        if (needs_arguments) {
            Value* idx = b.build_iconst_i32(static_cast<int32_t>(n));
            raw = b.build_call("bronze_arg_at", Type::i64(), {val_argc, val_argv, idx});
        } else {
            raw = b.build_load(Type::i64(), val_argv, static_cast<int32_t>(n * 8));
        }
        loaded.push_back(raw);
    }

    Value* wrap_frame = nullptr;
    if (needs_arguments || has_rest) {
        uint32_t total_slots = 4 + static_cast<uint32_t>(named_count);
        wrap_frame = b.build_call("bronze_gc_frame_push", Type::ptr(), {b.build_iconst_i32(static_cast<int32_t>(total_slots))});
        b.build_store(Type::i64(), wrap_frame, 16 + 0 * 8, val_env);
        b.build_store(Type::i64(), wrap_frame, 16 + 1 * 8, val_this);
        b.build_store(Type::i64(), wrap_frame, 16 + 2 * 8, b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));
        b.build_store(Type::i64(), wrap_frame, 16 + 3 * 8, b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));
        for (size_t n = 0; n < named_count; ++n) {
            b.build_store(Type::i64(), wrap_frame, 16 + static_cast<int32_t>((4 + n) * 8), loaded[n]);
        }
    }

    Value* arguments_arg = nullptr;
    if (needs_arguments) {
        Value* callee_val = is_strict ? b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)) : val_env;
        Value* is_strict_val = b.build_iconst_i32(is_strict ? 1 : 0);
        arguments_arg = b.build_call("bronze_arguments_object", Type::i64(), {val_argc, val_argv, callee_val, is_strict_val});
        b.build_store(Type::i64(), wrap_frame, 16 + 2 * 8, arguments_arg);
        val_env = b.build_load(Type::i64(), wrap_frame, 16 + 0 * 8);
        val_this = b.build_load(Type::i64(), wrap_frame, 16 + 1 * 8);
        for (size_t n = 0; n < named_count; ++n) {
            loaded[n] = b.build_load(Type::i64(), wrap_frame, 16 + static_cast<int32_t>((4 + n) * 8));
        }
    }

    Value* rest_arg = nullptr;
    if (has_rest) {
        uint32_t first_rest = static_cast<uint32_t>(fn_ast.params.size() - 1 - first_source_param);
        rest_arg = b.build_call("bronze_rest_args", Type::i64(), {val_argc, val_argv, b.build_iconst_i32(static_cast<int32_t>(first_rest))});
        b.build_store(Type::i64(), wrap_frame, 16 + 3 * 8, rest_arg);
        val_env = b.build_load(Type::i64(), wrap_frame, 16 + 0 * 8);
        val_this = b.build_load(Type::i64(), wrap_frame, 16 + 1 * 8);
        if (needs_arguments) {
            arguments_arg = b.build_load(Type::i64(), wrap_frame, 16 + 2 * 8);
        }
        for (size_t n = 0; n < named_count; ++n) {
            loaded[n] = b.build_load(Type::i64(), wrap_frame, 16 + static_cast<int32_t>((4 + n) * 8));
        }
    }

    std::vector<Value*> call_args;
    if (needs_env) call_args.push_back(val_env);
    if (needs_this) call_args.push_back(val_this);
    if (needs_arguments) call_args.push_back(arguments_arg);

    for (size_t p = first_source_param; p < fn_ast.params.size(); ++p) {
        size_t source_idx = p - first_source_param;
        if (has_rest && p + 1 == fn_ast.params.size()) {
            call_args.push_back(rest_arg);
            break;
        }
        Value* raw_i64 = loaded[source_idx];
        BronzeType param_type = fn_ast.params[p].second;
        if (param_type == BronzeType::F64) {
            bool is_pinned = false;
            uint32_t pin_key = 0;
            if (it_meta != options_.function_meta.end() && p < it_meta->second.params_pinned.size() && it_meta->second.params_pinned[p]) {
                is_pinned = true;
                pin_key = (p < it_meta->second.param_pin_keys.size()) ? it_meta->second.param_pin_keys[p] : 0;
            }
            if (is_pinned) {
                BasicBlock* cur_bb = b.current_block();
                BasicBlock* bad_bb = b.append_block("w_pin_bad_" + std::to_string(p));
                BasicBlock* ok_bb = b.append_block("w_pin_ok_" + std::to_string(p));
                b.position_at_end(cur_bb);
                Value* is_num = b.build_ule(raw_i64, b.build_iconst_i64(static_cast<int64_t>(0xFFF0000000000000ULL)));
                b.build_br_if(is_num, ok_bb, bad_bb);

                b.position_at_end(bad_bb);
                b.build_call("bronze_pin_violation", Type::i64(), {get_key_id(b, pin_key), raw_i64});
                if (wrap_frame) {
                    b.build_call("bronze_gc_frame_pop", Type::void_type(), {});
                }
                b.build_ret(b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));

                b.position_at_end(ok_bb);
                call_args.push_back(b.build_bitcast_f64_i64(raw_i64));
            } else {
                call_args.push_back(b.build_call("bronze_unbox_f64", Type::f64(), {raw_i64}));
            }
        } else if (param_type == BronzeType::I32) {
            call_args.push_back(b.build_call("bronze_unbox_i32", Type::i32(), {raw_i64}));
        } else if (param_type == BronzeType::Bool) {
            call_args.push_back(b.build_and(b.build_call("bronze_unbox_bool", Type::i32(), {raw_i64}), b.build_iconst_i32(1)));
        } else {
            call_args.push_back(raw_i64);
        }
    }

    Type ret_type = lower_type(fn_ast.return_type);
    Value* call_res = b.build_call(fn_name, ret_type, call_args);

    Value* ret_val = nullptr;
    if (fn_ast.return_type == BronzeType::Void) {
        ret_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
    } else if (fn_ast.return_type == BronzeType::F64) {
        ret_val = b.build_call("bronze_box_f64", Type::i64(), {call_res});
    } else if (fn_ast.return_type == BronzeType::I32) {
        ret_val = b.build_call("bronze_box_i32", Type::i64(), {call_res});
    } else if (fn_ast.return_type == BronzeType::Bool) {
        ret_val = b.build_call("bronze_box_bool", Type::i64(), {call_res});
    } else {
        ret_val = call_res;
    }
    if (wrap_frame) {
        b.build_call("bronze_gc_frame_pop", Type::void_type(), {});
    }
    b.build_ret(ret_val);
    wfn->rebuild_cfg_predecessors();
    return true;
}

} // namespace brass::il
