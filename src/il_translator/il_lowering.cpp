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

std::unique_ptr<Module> IlLowering::lower_module(const BronzeModuleAST& ast) {
    auto mod = std::make_unique<Module>(ast.name);
    mod->set_allow_fp_reassociation(options_.allow_fp_reassociation);

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

    // 1. Forward-declare all functions (uniquifying any duplicate anonymous function names from Bronze)
    std::unordered_map<std::string, std::vector<size_t>> name_to_indices;
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        name_to_indices[ast.functions[i].name].push_back(i);
    }

    std::vector<std::string> resolved_names(ast.functions.size());
    for (const auto& [name, indices] : name_to_indices) {
        if (indices.size() == 1) {
            resolved_names[indices[0]] = name;
        } else {
            // Multiple functions with the same name: distinguish leaf vs non-leaf closures
            for (size_t idx : indices) {
                bool has_create_func = false;
                for (const auto& blk : ast.functions[idx].blocks) {
                    for (const auto& inst : blk.instructions) {
                        if (inst.op == BronzeOp::CreateFunc) {
                            has_create_func = true;
                            break;
                        }
                    }
                    if (has_create_func) break;
                }
                if (has_create_func) {
                    resolved_names[idx] = name;
                } else {
                    resolved_names[idx] = name + "$leaf";
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
            if (!lower_instruction(inst_ast, b, fn, val_map, block_map, blk_ast.handler_id, blk_ast.id, &cont_counter)) {
                return false;
            }
        }
    }

    fn->rebuild_cfg_predecessors();
    return true;
}

bool IlLowering::lower_instruction(
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    const std::unordered_map<uint32_t, BasicBlock*>& block_map,
    uint32_t handler_id,
    uint32_t block_id,
    uint32_t* cont_counter
) {
    Type res_type = lower_type(inst_ast.result_type);
    Value* res_val = nullptr;

    auto get_opd = [&](size_t idx) -> Value* {
        if (idx < inst_ast.operands.size()) {
            uint32_t id = inst_ast.operands[idx];
            if (val_map.count(id)) return val_map[id];
        }
        return nullptr;
    };

    if (is_coro_il_op(inst_ast.op)) {
        if (!lower_coro_instruction(inst_ast, b, fn, val_map, res_val)) {
            return false;
        }
        if (inst_ast.result_id != UINT32_MAX && res_val) {
            val_map[inst_ast.result_id] = res_val;
        }
        return true;
    }

    switch (inst_ast.op) {
        case BronzeOp::ConstF64:
            res_val = b.build_fconst_f64(inst_ast.imm_f64);
            break;
        case BronzeOp::ConstI32:
            res_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.imm_i64));
            break;
        case BronzeOp::ConstBool:
            res_val = b.build_iconst_i32(inst_ast.imm_bool ? 1 : 0);
            break;
        case BronzeOp::ConstUndefined:
            res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            break;
        case BronzeOp::ConstNull:
            res_val = b.build_iconst_i64(static_cast<int64_t>(kNullTag));
            break;
        case BronzeOp::ConstBigInt:
            res_val = b.build_iconst_i64(inst_ast.imm_i64);
            break;

        case BronzeOp::NameResolve: {
            if (inst_ast.string_literal == "print" || inst_ast.string_literal == "console.log") {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kPrintTag));
            } else if (inst_ast.string_literal == "print.err") {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kPrintErrTag));
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            break;
        }

        case BronzeOp::CallDynamic: {
            Value* callee_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;

            if (argc == 0) {
                res_val = b.build_call("bronze_call_dynamic_0", Type::i64(), {callee_val, this_val});
            } else if (argc == 1) {
                Value* arg0 = ensure_type(get_opd(2), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_1", Type::i64(), {callee_val, this_val, arg0});
            } else if (argc == 2) {
                Value* arg0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* arg1 = ensure_type(get_opd(3), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_2", Type::i64(), {callee_val, this_val, arg0, arg1});
            } else if (argc == 3) {
                Value* arg0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* arg1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* arg2 = ensure_type(get_opd(4), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_3", Type::i64(), {callee_val, this_val, arg0, arg1, arg2});
            } else if (argc == 4) {
                Value* arg0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* arg1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* arg2 = ensure_type(get_opd(4), Type::i64(), b);
                Value* arg3 = ensure_type(get_opd(5), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_4", Type::i64(), {callee_val, this_val, arg0, arg1, arg2, arg3});
            } else if (argc == 5) {
                Value* a0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* a2 = ensure_type(get_opd(4), Type::i64(), b);
                Value* a3 = ensure_type(get_opd(5), Type::i64(), b);
                Value* a4 = ensure_type(get_opd(6), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_5", Type::i64(), {callee_val, this_val, a0, a1, a2, a3, a4});
            } else if (argc == 6) {
                Value* a0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* a2 = ensure_type(get_opd(4), Type::i64(), b);
                Value* a3 = ensure_type(get_opd(5), Type::i64(), b);
                Value* a4 = ensure_type(get_opd(6), Type::i64(), b);
                Value* a5 = ensure_type(get_opd(7), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_6", Type::i64(), {callee_val, this_val, a0, a1, a2, a3, a4, a5});
            } else if (argc == 7) {
                Value* a0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* a2 = ensure_type(get_opd(4), Type::i64(), b);
                Value* a3 = ensure_type(get_opd(5), Type::i64(), b);
                Value* a4 = ensure_type(get_opd(6), Type::i64(), b);
                Value* a5 = ensure_type(get_opd(7), Type::i64(), b);
                Value* a6 = ensure_type(get_opd(8), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_7", Type::i64(), {callee_val, this_val, a0, a1, a2, a3, a4, a5, a6});
            } else if (argc == 8) {
                Value* a0 = ensure_type(get_opd(2), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(3), Type::i64(), b);
                Value* a2 = ensure_type(get_opd(4), Type::i64(), b);
                Value* a3 = ensure_type(get_opd(5), Type::i64(), b);
                Value* a4 = ensure_type(get_opd(6), Type::i64(), b);
                Value* a5 = ensure_type(get_opd(7), Type::i64(), b);
                Value* a6 = ensure_type(get_opd(8), Type::i64(), b);
                Value* a7 = ensure_type(get_opd(9), Type::i64(), b);
                res_val = b.build_call("bronze_call_dynamic_8", Type::i64(), {callee_val, this_val, a0, a1, a2, a3, a4, a5, a6, a7});
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            break;
        }

        case BronzeOp::EnvCreate: {
            Value* parent_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* size_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.param_count));
            res_val = b.build_call("bronze_env_create", Type::i64(), {parent_val, size_val});
            break;
        }

        case BronzeOp::EnvGet:
        case BronzeOp::EnvGetTdz: {
            Value* env_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* depth_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.depth));
            Value* idx_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            res_val = b.build_call("bronze_env_get", Type::i64(), {env_val, depth_val, idx_val});
            break;
        }

        case BronzeOp::EnvSet: {
            Value* env_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* depth_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.depth));
            Value* idx_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            Value* val = ensure_type(get_opd(1), Type::i64(), b);
            b.build_call("bronze_env_set", Type::void_type(), {env_val, depth_val, idx_val, val});
            b.build_write_barrier(env_val, val);
            break;
        }

        case BronzeOp::EnvInitTdz: {
            break;
        }

        case BronzeOp::CreateFunc: {
            Value* argc_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.param_count));
            Value* env_val = ensure_type(get_opd(0), Type::i64(), b);
            std::string callee = inst_ast.callee_name;
            if (fn->name() == callee) {
                callee = callee + "$leaf";
            }
            // Strings are arena-allocated in Module's StringPool, living for the module lifetime.
            const char* name_ptr = fn->parent()->string_pool().intern(callee).data();
            res_val = b.build_call("bronze_create_func", Type::i64(), {
                b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(name_ptr))),
                argc_val,
                env_val
            });
            break;
        }

        case BronzeOp::CreateArray: {
            Value* size_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.param_count));
            res_val = b.build_call("bronze_create_array", Type::i64(), {size_val});
            break;
        }

        case BronzeOp::CreateObject: {
            res_val = b.build_call("bronze_create_object", Type::i64(), {});
            break;
        }

        case BronzeOp::PropGet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            res_val = prop_lowering_.lower_prop_get(
                b, obj_val, inst_ast.string_literal, inst_ast.index, inst_ast.depth
            );
            break;
        }

        case BronzeOp::PropSet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* val = ensure_type(get_opd(1), Type::i64(), b);
            prop_lowering_.lower_prop_set(
                b, obj_val, inst_ast.string_literal, inst_ast.index, val,
                inst_ast.depth, static_cast<uint32_t>(inst_ast.imm_i64), 0
            );
            b.build_write_barrier(obj_val, val);
            break;
        }

        case BronzeOp::MethodDef: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* closure_val = ensure_type(get_opd(1), Type::i64(), b);
            prop_lowering_.lower_method_def(
                b, obj_val, inst_ast.string_literal, inst_ast.index, closure_val
            );
            b.build_write_barrier(obj_val, closure_val);
            break;
        }

        case BronzeOp::ElemGet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* idx_val = ensure_type(get_opd(1), Type::i64(), b);
            res_val = prop_lowering_.lower_elem_get(b, obj_val, idx_val);
            break;
        }

        case BronzeOp::ElemSet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* idx_val = ensure_type(get_opd(1), Type::i64(), b);
            Value* val = ensure_type(get_opd(2), Type::i64(), b);
            prop_lowering_.lower_elem_set(b, obj_val, idx_val, val, inst_ast.index);
            b.build_write_barrier(obj_val, val);
            break;
        }

        case BronzeOp::Add: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
                Value* r = b.build_add(op0, op1);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                op1 = ensure_type(op1, Type::i32(), b);
                res_val = b.build_add(op0, op1);
            }
            break;
        }

        case BronzeOp::Sub: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
                Value* r = b.build_sub(op0, op1);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                op1 = ensure_type(op1, Type::i32(), b);
                res_val = b.build_sub(op0, op1);
            }
            break;
        }

        case BronzeOp::Mul: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
                Value* r = b.build_mul(op0, op1);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                op1 = ensure_type(op1, Type::i32(), b);
                res_val = b.build_mul(op0, op1);
            }
            break;
        }

        case BronzeOp::Div: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
                Value* r = b.build_sdiv(op0, op1);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                op1 = ensure_type(op1, Type::i32(), b);
                res_val = b.build_sdiv(op0, op1);
            }
            break;
        }

        case BronzeOp::Mod: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
                Value* r = b.build_call("bronze_f64_mod", Type::f64(), {op0, op1});
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                op1 = ensure_type(op1, Type::i32(), b);
                res_val = b.build_smod(op0, op1);
            }
            break;
        }

        case BronzeOp::Neg: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                Value* r = b.build_neg(op0);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = ensure_type(op0, Type::i32(), b);
                res_val = b.build_neg(op0);
            }
            break;
        }

        case BronzeOp::BitAnd: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_and(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::BitOr: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_or(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::BitXor: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_xor(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::Shl: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_shl(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::Shr: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_ashr(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::UShr: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_lshr(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }
        case BronzeOp::BitNot: {
            Value* op0 = ensure_type(get_opd(0), Type::i32(), b);
            Value* r = b.build_xor(op0, b.build_iconst_i32(-1));
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            break;
        }

        case BronzeOp::ToInt32: {
            Value* op0 = get_opd(0);
            res_val = ensure_type(op0, Type::i32(), b);
            break;
        }
        case BronzeOp::ToNumeric: {
            Value* op0 = get_opd(0);
            res_val = ensure_type(op0, Type::f64(), b);
            break;
        }

        case BronzeOp::CmpLt:
        case BronzeOp::RelLt: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_slt(op0, op1);
            break;
        }
        case BronzeOp::CmpLe:
        case BronzeOp::RelLe: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_sle(op0, op1);
            break;
        }
        case BronzeOp::CmpGt:
        case BronzeOp::RelGt: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_sgt(op0, op1);
            break;
        }
        case BronzeOp::CmpGe:
        case BronzeOp::RelGe: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_sge(op0, op1);
            break;
        }
        case BronzeOp::CmpEq:
        case BronzeOp::StrictEq:
        case BronzeOp::LooseEq: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_eq(op0, op1);
            break;
        }
        case BronzeOp::CmpNe: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64() || op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = ensure_type(op0, Type::f64(), b);
                op1 = ensure_type(op1, Type::f64(), b);
            }
            res_val = b.build_ne(op0, op1);
            break;
        }

        case BronzeOp::Box: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (inst_ast.box_type == BronzeType::F64) {
                Value* f_val = ensure_type(op0, Type::f64(), b);
                res_val = b.build_bitcast_i64_f64(f_val);
            } else if (inst_ast.box_type == BronzeType::I32) {
                Value* i_val = ensure_type(op0, Type::i32(), b);
                Value* sext = b.build_sext_i64(i_val);
                Value* tag = b.build_iconst_i64(static_cast<int64_t>(kInt32Tag));
                res_val = b.build_or(b.build_and(sext, b.build_iconst_i64(0xFFFFFFFFLL)), tag);
            } else if (inst_ast.box_type == BronzeType::Bool) {
                Value* b_val = ensure_type(op0, Type::i32(), b);
                Value* zext = b.build_zext_i64(b_val);
                Value* tag = b.build_iconst_i64(static_cast<int64_t>(kBoolTag));
                res_val = b.build_or(zext, tag);
            } else {
                res_val = ensure_type(op0, Type::i64(), b);
            }
            break;
        }

        case BronzeOp::Unbox: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (inst_ast.result_type == BronzeType::F64) {
                Value* i_val = ensure_type(op0, Type::i64(), b);
                res_val = b.build_bitcast_f64_i64(i_val);
            } else if (inst_ast.result_type == BronzeType::I32) {
                res_val = b.build_trunc_i32(ensure_type(op0, Type::i64(), b));
            } else if (inst_ast.result_type == BronzeType::Bool) {
                Value* i_val = ensure_type(op0, Type::i64(), b);
                res_val = b.build_ne(b.build_and(i_val, b.build_iconst_i64(1)), b.build_iconst_i64(0));
            } else {
                res_val = op0;
            }
            break;
        }

        case BronzeOp::Call: {
            Function* callee = fn->parent()->get_function(inst_ast.callee_name);
            Type callee_ret = callee ? callee->return_type() : lower_type(inst_ast.result_type);
            std::vector<Value*> args;
            for (size_t i = 0; i < inst_ast.operands.size(); ++i) {
                Value* arg = get_opd(i);
                if (callee && i < callee->param_types().size()) {
                    arg = ensure_type(arg, callee->param_types()[i], b);
                }
                args.push_back(arg);
            }
            if (handler_id != UINT32_MAX && block_map.count(handler_id)) {
                BasicBlock* cur_bb = b.current_block();
                BasicBlock* unwind_bb = block_map.at(handler_id);
                uint32_t cid = cont_counter ? ++(*cont_counter) : 1;
                std::string cont_name = "b" + std::to_string(block_id) + "_cont" + std::to_string(cid);
                BasicBlock* normal_bb = b.append_block(cont_name);
                b.position_at_end(cur_bb);
                Instruction* inv = b.build_invoke(inst_ast.callee_name, callee_ret, Span<Value* const>(args.data(), args.size()), normal_bb, unwind_bb);
                res_val = inv->result();
                b.position_at_end(normal_bb);
            } else {
                res_val = b.build_call(inst_ast.callee_name, callee_ret, Span<Value* const>(args.data(), args.size()));
            }
            break;
        }

        case BronzeOp::Throw: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (handler_id != UINT32_MAX && block_map.count(handler_id)) {
                BasicBlock* cur_bb = b.current_block();
                BasicBlock* unwind_bb = block_map.at(handler_id);
                uint32_t cid = cont_counter ? ++(*cont_counter) : 1;
                std::string cont_name = "b" + std::to_string(block_id) + "_cont" + std::to_string(cid);
                BasicBlock* normal_bb = b.append_block(cont_name);
                b.position_at_end(cur_bb);
                b.build_invoke("brass_throw", Type::void_type(), {op0}, normal_bb, unwind_bb);
                b.position_at_end(normal_bb);
                b.build_unreachable();
            } else {
                b.build_throw(op0);
            }
            break;
        }

        case BronzeOp::ExcTake: {
            res_val = b.build_landing_pad(res_type);
            break;
        }

        case BronzeOp::Print:
        case BronzeOp::PrintErr: {
            for (size_t i = 0; i < inst_ast.operands.size(); ++i) {
                Value* arg = get_opd(i);
                if (!arg) continue;
                if (arg->type() == Type::f64()) {
                    b.build_call("bronze_print_f64", Type::void_type(), {arg});
                } else if (arg->type() == Type::i32()) {
                    b.build_call("bronze_print_i32", Type::void_type(), {arg});
                } else {
                    Value* d_arg = ensure_type(arg, Type::i64(), b);
                    b.build_call("bronze_print_dynamic", Type::void_type(), {d_arg});
                }
            }
            b.build_call("bronze_print_newline", Type::void_type());
            break;
        }

        case BronzeOp::Jump: {
            if (!block_map.count(inst_ast.target.block_id)) return false;
            BasicBlock* target_bb = block_map.at(inst_ast.target.block_id);
            std::vector<Value*> target_args;
            for (size_t i = 0; i < inst_ast.target.args.size(); ++i) {
                uint32_t aid = inst_ast.target.args[i];
                Value* aval = val_map.count(aid) ? val_map[aid] : nullptr;
                if (i < target_bb->params().size() && aval) {
                    aval = ensure_type(aval, target_bb->params()[i]->type(), b);
                }
                target_args.push_back(aval);
            }
            b.build_br(target_bb, target_args);
            break;
        }

        case BronzeOp::Branch: {
            Value* cond = get_opd(0);
            if (!cond) return false;
            cond = ensure_type(cond, Type::i32(), b);

            if (!block_map.count(inst_ast.target.block_id) || !block_map.count(inst_ast.else_target.block_id)) {
                return false;
            }
            BasicBlock* true_bb = block_map.at(inst_ast.target.block_id);
            BasicBlock* false_bb = block_map.at(inst_ast.else_target.block_id);

            std::vector<Value*> true_args;
            for (size_t i = 0; i < inst_ast.target.args.size(); ++i) {
                uint32_t aid = inst_ast.target.args[i];
                Value* aval = val_map.count(aid) ? val_map[aid] : nullptr;
                if (i < true_bb->params().size() && aval) {
                    aval = ensure_type(aval, true_bb->params()[i]->type(), b);
                }
                true_args.push_back(aval);
            }

            std::vector<Value*> false_args;
            for (size_t i = 0; i < inst_ast.else_target.args.size(); ++i) {
                uint32_t aid = inst_ast.else_target.args[i];
                Value* aval = val_map.count(aid) ? val_map[aid] : nullptr;
                if (i < false_bb->params().size() && aval) {
                    aval = ensure_type(aval, false_bb->params()[i]->type(), b);
                }
                false_args.push_back(aval);
            }

            b.build_br_if(cond, true_bb, true_args, false_bb, false_args);
            break;
        }

        case BronzeOp::Ret: {
            Value* ret_val = get_opd(0);
            if (ret_val && fn->return_type() != Type::void_type()) {
                ret_val = ensure_type(ret_val, fn->return_type(), b);
            }
            b.build_ret(ret_val);
            break;
        }

        default:
            return false;
    }

    if (inst_ast.result_id != UINT32_MAX && res_val) {
        val_map[inst_ast.result_id] = res_val;
    }
    return true;
}

} // namespace brass::il
