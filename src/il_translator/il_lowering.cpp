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
        return b.build_bitcast_i64_f64(val);
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

std::unique_ptr<Module> IlLowering::lower_module(const BronzeModuleAST& ast) {
    auto mod = std::make_unique<Module>(ast.name);
    mod->set_allow_fp_reassociation(options_.allow_fp_reassociation);
    current_file_id_ = ast.name.empty() ? 0 : mod->debug_context().get_or_add_file(ast.name);

    // Register external runtime helper functions
    mod->add_external_symbol("bronze_print_f64");
    mod->add_external_symbol("bronze_print_i32");
    mod->add_external_symbol("bronze_print_dynamic");
    mod->add_external_symbol("bronze_print_newline");
    mod->add_external_symbol("bronze_f64_mod");
    mod->add_external_symbol("bronze_name_resolve");
    mod->add_external_symbol("bronze_env_create");
    mod->add_external_symbol("bronze_env_get");
    mod->add_external_symbol("bronze_env_set");
    mod->add_external_symbol("bronze_create_func");
    mod->add_external_symbol("bronze_create_array");
    mod->add_external_symbol("bronze_create_object");
    mod->add_external_symbol("bronze_prop_get");
    mod->add_external_symbol("bronze_prop_set");
    mod->add_external_symbol("bronze_elem_get");
    mod->add_external_symbol("bronze_elem_set");
    mod->add_external_symbol("bronze_method_def");
    mod->add_external_symbol("bronze_ic_get");
    mod->add_external_symbol("bronze_ic_set");
    mod->add_external_symbol("brass_ic_get_prop");
    mod->add_external_symbol("brass_ic_set_prop");
    mod->add_external_symbol("brass_dynamic_object_get_prop_str");
    mod->add_external_symbol("brass_dynamic_object_set_prop_str");
    mod->add_external_symbol("bronze_call_dynamic_0");
    mod->add_external_symbol("bronze_call_dynamic_1");
    mod->add_external_symbol("bronze_call_dynamic_2");
    mod->add_external_symbol("bronze_call_dynamic_3");
    mod->add_external_symbol("bronze_call_dynamic_4");
    mod->add_external_symbol("bronze_call_dynamic_5");
    mod->add_external_symbol("bronze_call_dynamic_6");
    mod->add_external_symbol("bronze_call_dynamic_7");
    mod->add_external_symbol("bronze_call_dynamic_8");
    mod->add_external_symbol("bronze_call_dynamic_n");
    mod->add_external_symbol("bronze_create_async_machine");
    mod->add_external_symbol("bronze_async_start");
    mod->add_external_symbol("bronze_async_await");
    mod->add_external_symbol("bronze_iter_open");
    mod->add_external_symbol("bronze_iter_step");
    mod->add_external_symbol("brass_coro_create");
    mod->add_external_symbol("brass_coro_resume");
    mod->add_external_symbol("brass_coro_is_done");
    mod->add_external_symbol("brass_coro_destroy");
    mod->add_external_symbol("bronze_register_value_cells");
    mod->add_external_symbol("bronze_concat_begin");
    mod->add_external_symbol("bronze_concat_append");
    mod->add_external_symbol("bronze_concat_end");
    mod->add_external_symbol("bronze_global_get_name");
    mod->add_external_symbol("bronze_construct_0");
    mod->add_external_symbol("bronze_construct_1");
    mod->add_external_symbol("bronze_construct_2");
    mod->add_external_symbol("bronze_construct_3");
    mod->add_external_symbol("bronze_construct");
    mod->add_external_symbol("bronze_class_extends");
    mod->add_external_symbol("bronze_super_call");
    mod->add_external_symbol("bronze_super_get");
    mod->add_external_symbol("bronze_instanceof");
    mod->add_external_symbol("bronze_has_property");
    mod->add_external_symbol("bronze_is_nullish");

    current_ast_ = &ast;

    caller_to_callee_map_.clear();

    // 1. Forward-declare all functions (uniquifying any duplicate function names from Bronze)
    std::unordered_map<std::string, std::vector<size_t>> name_to_indices;
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        name_to_indices[ast.functions[i].name].push_back(i);
    }

    std::vector<std::string> resolved_names(ast.functions.size());
    for (const auto& [name, indices] : name_to_indices) {
        if (indices.size() == 1) {
            resolved_names[indices[0]] = name;
        } else {
            for (size_t k = 0; k < indices.size(); ++k) {
                resolved_names[indices[k]] = (k == 0) ? name : (name + "$" + std::to_string(k));
            }

            size_t next_k = 0;
            for (size_t fn_idx = 0; fn_idx < ast.functions.size(); ++fn_idx) {
                bool is_self = false;
                for (size_t k = 0; k < indices.size(); ++k) {
                    if (indices[k] == fn_idx) {
                        is_self = true;
                        caller_to_callee_map_[fn_idx][name] = resolved_names[fn_idx];
                        break;
                    }
                }
                if (is_self) continue;

                bool references_name = false;
                for (const auto& blk : ast.functions[fn_idx].blocks) {
                    for (const auto& inst : blk.instructions) {
                        if (inst.callee_name == name) {
                            references_name = true;
                            break;
                        }
                    }
                    if (references_name) break;
                }
                if (references_name) {
                    if (next_k < indices.size()) {
                        caller_to_callee_map_[fn_idx][name] = resolved_names[indices[next_k++]];
                    } else {
                        caller_to_callee_map_[fn_idx][name] = resolved_names[indices.back()];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < ast.functions.size(); ++i) {
        const auto& fn_ast = ast.functions[i];
        const std::string& fn_name = resolved_names[i];

        std::vector<Type> param_types;
        for (const auto& p : fn_ast.params) {
            param_types.push_back(lower_type(p.second));
        }
        Type ret_type = lower_type(fn_ast.return_type);
        Function* fn = mod->create_function(fn_name, ret_type, Span<const Type>(param_types.data(), param_types.size()));
        fn->set_allow_fp_reassociation(options_.allow_fp_reassociation);
    }

    // 2. Lower each function body
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        current_fn_idx_ = i;
        const auto& fn_ast = ast.functions[i];
        const std::string& fn_name = resolved_names[i];
        if (!lower_function(fn_ast, *mod, fn_name)) {
            return nullptr;
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
        if (fn_name == "main") {
            b.position_at_end(entry_bb);
            Value* env_addr = b.build_func_addr("__bronze_module_env");
            Value* count_val = b.build_iconst_i64(1);
            b.build_call("bronze_register_value_cells", Type::void_type(), {env_addr, count_val});
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

} // namespace brass::il
