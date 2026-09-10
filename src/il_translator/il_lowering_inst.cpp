#include "il_lowering.hpp"
#include "il_lowering_coro.hpp"
#include "il_runtime.hpp"
#include <brass/mir/verifier.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cctype>

namespace brass::il {

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

        case BronzeOp::GlobalGet: {
            const char* name_ptr = nullptr;
            if (!inst_ast.string_literal.empty()) {
                name_ptr = fn->parent()->string_pool().intern(inst_ast.string_literal).data();
            }
            if (name_ptr) {
                res_val = b.build_call("bronze_global_get_name", Type::i64(), {
                    b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(name_ptr)))
                });
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            break;
        }

        case BronzeOp::ConcatBegin: {
            Value* lhs = ensure_type(get_opd(0), Type::i64(), b);
            Value* rhs = ensure_type(get_opd(1), Type::i64(), b);
            Value* rem = b.build_iconst_i32(static_cast<int32_t>(inst_ast.imm_i64));
            res_val = b.build_call("bronze_concat_begin", Type::i64(), {lhs, rhs, rem});
            break;
        }

        case BronzeOp::ConcatAppend: {
            Value* lhs = ensure_type(get_opd(0), Type::i64(), b);
            Value* rhs = ensure_type(get_opd(1), Type::i64(), b);
            res_val = b.build_call("bronze_concat_append", Type::i64(), {lhs, rhs});
            break;
        }

        case BronzeOp::ConcatEnd: {
            Value* val = ensure_type(get_opd(0), Type::i64(), b);
            res_val = b.build_call("bronze_concat_end", Type::i64(), {val});
            break;
        }

        case BronzeOp::IsNumber: {
            Value* src = ensure_type(get_opd(0), Type::i64(), b);
            Value* limit = b.build_iconst_i64(static_cast<int64_t>(0xFFF0000000000000ULL));
            res_val = b.build_ule(src, limit);
            break;
        }

        case BronzeOp::IsDenseArray: {
            res_val = b.build_iconst_i32(0);
            break;
        }

        case BronzeOp::IsNullish: {
            Value* op0 = ensure_type(get_opd(0), Type::i64(), b);
            res_val = b.build_call("bronze_is_nullish", Type::i32(), {op0});
            break;
        }

        case BronzeOp::InstanceOf: {
            Value* op0 = ensure_type(get_opd(0), Type::i64(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i64(), b);
            res_val = b.build_call("bronze_instanceof", Type::i32(), {op0, op1});
            break;
        }

        case BronzeOp::In: {
            Value* op0 = ensure_type(get_opd(0), Type::i64(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i64(), b);
            res_val = b.build_call("bronze_has_property", Type::i32(), {op0, op1});
            break;
        }

        case BronzeOp::ClassExtend: {
            Value* sub = ensure_type(get_opd(0), Type::i64(), b);
            Value* sup = ensure_type(get_opd(1), Type::i64(), b);
            b.build_call("bronze_class_extends", Type::void_type(), {sub, sup});
            break;
        }

        case BronzeOp::Construct: {
            Value* ctor = ensure_type(get_opd(0), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;
            if (argc == 0) {
                res_val = b.build_call("bronze_construct_0", Type::i64(), {ctor});
            } else if (argc == 1) {
                Value* a0 = ensure_type(get_opd(1), Type::i64(), b);
                res_val = b.build_call("bronze_construct_1", Type::i64(), {ctor, a0});
            } else if (argc == 2) {
                Value* a0 = ensure_type(get_opd(1), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(2), Type::i64(), b);
                res_val = b.build_call("bronze_construct_2", Type::i64(), {ctor, a0, a1});
            } else if (argc == 3) {
                Value* a0 = ensure_type(get_opd(1), Type::i64(), b);
                Value* a1 = ensure_type(get_opd(2), Type::i64(), b);
                Value* a2 = ensure_type(get_opd(3), Type::i64(), b);
                res_val = b.build_call("bronze_construct_3", Type::i64(), {ctor, a0, a1, a2});
            } else {
                res_val = b.build_call("bronze_construct_0", Type::i64(), {ctor});
            }
            break;
        }

        case BronzeOp::MethodCall: {
            Value* recv = ensure_type(get_opd(0), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;
            std::string callee = resolve_callee(inst_ast.callee_name);
            if (!callee.empty() && std::isdigit(static_cast<unsigned char>(callee[0]))) {
                uint32_t f_idx = static_cast<uint32_t>(std::stoul(callee));
                if (current_ast_ && f_idx < current_ast_->functions.size()) {
                    callee = resolve_callee(current_ast_->functions[f_idx].name);
                }
            }
            if (!callee.empty() && fn->parent()->get_function(callee)) {
                Function* direct_fn = fn->parent()->get_function(callee);
                std::vector<Value*> call_args;
                size_t p_idx = 0;
                if (p_idx < direct_fn->param_types().size() && direct_fn->param_types().size() == argc + 2) {
                    call_args.push_back(b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));
                    p_idx++;
                }
                if (p_idx < direct_fn->param_types().size()) {
                    call_args.push_back(ensure_type(recv, direct_fn->param_types()[p_idx++], b));
                }
                for (size_t a = 0; a < argc; ++a) {
                    Value* arg_val = get_opd(1 + a);
                    if (p_idx < direct_fn->param_types().size()) {
                        arg_val = ensure_type(arg_val, direct_fn->param_types()[p_idx++], b);
                    }
                    call_args.push_back(arg_val);
                }
                res_val = b.build_call(callee, direct_fn->return_type(), Span<Value* const>(call_args.data(), call_args.size()));
            } else {
                Value* method = b.build_call("bronze_prop_get", Type::i64(), {recv, b.build_iconst_i32(static_cast<int32_t>(inst_ast.index))});
                std::vector<Value*> dyn_args = {method, recv};
                for (size_t a = 0; a < argc; ++a) {
                    dyn_args.push_back(ensure_type(get_opd(1 + a), Type::i64(), b));
                }
                if (argc == 0) {
                    res_val = b.build_call("bronze_call_dynamic_0", Type::i64(), dyn_args);
                } else if (argc == 1) {
                    res_val = b.build_call("bronze_call_dynamic_1", Type::i64(), dyn_args);
                } else if (argc == 2) {
                    res_val = b.build_call("bronze_call_dynamic_2", Type::i64(), dyn_args);
                } else if (argc == 3) {
                    res_val = b.build_call("bronze_call_dynamic_3", Type::i64(), dyn_args);
                } else if (argc == 4) {
                    res_val = b.build_call("bronze_call_dynamic_4", Type::i64(), dyn_args);
                } else if (argc == 5) {
                    res_val = b.build_call("bronze_call_dynamic_5", Type::i64(), dyn_args);
                } else if (argc == 6) {
                    res_val = b.build_call("bronze_call_dynamic_6", Type::i64(), dyn_args);
                } else if (argc == 7) {
                    res_val = b.build_call("bronze_call_dynamic_7", Type::i64(), dyn_args);
                } else if (argc == 8) {
                    res_val = b.build_call("bronze_call_dynamic_8", Type::i64(), dyn_args);
                } else {
                    res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
                }
            }
            break;
        }

        case BronzeOp::SuperCall: {
            Value* base_ctor = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> dyn_args = {base_ctor, this_val};
            for (size_t a = 0; a < argc; ++a) {
                dyn_args.push_back(ensure_type(get_opd(2 + a), Type::i64(), b));
            }
            if (argc == 0) {
                res_val = b.build_call("bronze_call_dynamic_0", Type::i64(), dyn_args);
            } else if (argc == 1) {
                res_val = b.build_call("bronze_call_dynamic_1", Type::i64(), dyn_args);
            } else if (argc == 2) {
                res_val = b.build_call("bronze_call_dynamic_2", Type::i64(), dyn_args);
            } else if (argc == 3) {
                res_val = b.build_call("bronze_call_dynamic_3", Type::i64(), dyn_args);
            } else if (argc == 4) {
                res_val = b.build_call("bronze_call_dynamic_4", Type::i64(), dyn_args);
            } else if (argc == 5) {
                res_val = b.build_call("bronze_call_dynamic_5", Type::i64(), dyn_args);
            } else if (argc == 6) {
                res_val = b.build_call("bronze_call_dynamic_6", Type::i64(), dyn_args);
            } else if (argc == 7) {
                res_val = b.build_call("bronze_call_dynamic_7", Type::i64(), dyn_args);
            } else if (argc == 8) {
                res_val = b.build_call("bronze_call_dynamic_8", Type::i64(), dyn_args);
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            break;
        }

        case BronzeOp::SuperGet: {
            Value* proto = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            Value* kidx = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            res_val = b.build_call("bronze_super_get", Type::i64(), {proto, kidx, this_val});
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

        case BronzeOp::ModuleEnvSet: {
            Value* env_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* env_addr = b.build_func_addr("__bronze_module_env");
            b.build_store(Type::i64(), env_addr, 0, env_val);
            break;
        }

        case BronzeOp::ModuleEnvGet: {
            Value* env_addr = b.build_func_addr("__bronze_module_env");
            res_val = b.build_load(Type::i64(), env_addr, 0);
            break;
        }

        case BronzeOp::FuncRef: {
            std::string callee = inst_ast.callee_name;
            if (!callee.empty() && std::isdigit(static_cast<unsigned char>(callee[0]))) {
                uint32_t f_idx = static_cast<uint32_t>(std::stoul(callee));
                if (current_ast_ && f_idx < current_ast_->functions.size()) {
                    callee = current_ast_->functions[f_idx].name;
                }
            }
            callee = resolve_callee(callee);
            Value* code_addr = b.build_func_addr(callee);
            Value* env_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            Value* argc_val = b.build_iconst_i32(0);
            res_val = b.build_call("bronze_create_func", Type::i64(), {
                code_addr,
                argc_val,
                env_val
            });
            break;
        }

        case BronzeOp::CreateFunc: {
            Value* argc_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.param_count));
            Value* env_val = ensure_type(get_opd(0), Type::i64(), b);
            std::string callee = resolve_callee(inst_ast.callee_name);
            Value* code_addr = b.build_func_addr(callee);
            res_val = b.build_call("bronze_create_func", Type::i64(), {
                code_addr,
                argc_val,
                env_val
            });
            break;
        }

        case BronzeOp::PinGuard:
        case BronzeOp::CensusRecord:
            break;

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

        case BronzeOp::ElemGet:
        case BronzeOp::ElemGetTyped: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* idx_val = ensure_type(get_opd(1), Type::i64(), b);
            res_val = prop_lowering_.lower_elem_get(b, obj_val, idx_val);
            break;
        }

        case BronzeOp::ElemSet:
        case BronzeOp::ElemSetTyped: {
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
            if (inst_ast.box_type == BronzeType::Str && inst_ast.operands.empty()) {
                const char* interned = fn->parent()->string_pool().intern(inst_ast.string_literal).data();
                res_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
                break;
            }
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
            std::string callee_name = resolve_callee(inst_ast.callee_name);
            Function* callee = fn->parent()->get_function(callee_name);
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
                Instruction* inv = b.build_invoke(callee_name, callee_ret, Span<Value* const>(args.data(), args.size()), normal_bb, unwind_bb);
                res_val = inv->result();
                b.position_at_end(normal_bb);
            } else {
                res_val = b.build_call(callee_name, callee_ret, Span<Value* const>(args.data(), args.size()));
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
