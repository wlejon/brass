#include "il_lowering.hpp"
#include "il_lowering_coro.hpp"
#include "il_lowering_ops.hpp"
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

    auto build_call_dynamic = [&](const std::vector<Value*>& dyn_args) -> Value* {
        size_t argc = dyn_args.size() > 2 ? dyn_args.size() - 2 : 0;
        if (argc <= 8) {
            std::string helper = "bronze_call_dynamic_" + std::to_string(argc);
            return b.build_call(helper, Type::i64(), Span<Value* const>(dyn_args.data(), dyn_args.size()));
        }
        return b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
    };

    auto emit_default_ret = [&]() {
        if (fn->return_type() == Type::void_type()) {
            b.build_ret_void();
        } else if (fn->return_type() == Type::f64()) {
            b.build_ret(b.build_fconst_f64(0.0));
        } else if (fn->return_type() == Type::i32()) {
            b.build_ret(b.build_iconst_i32(0));
        } else {
            b.build_ret(b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));
        }
    };

    auto emit_exception_check = [&]() {
        BasicBlock* cur_bb = b.current_block();
        uint32_t cid = cont_counter ? ++(*cont_counter) : 1;
        BasicBlock* cont_bb = b.append_block("b" + std::to_string(block_id) + "_cont" + std::to_string(cid));
        BasicBlock* unw_bb = nullptr;
        bool created_unw = false;
        if (handler_id != UINT32_MAX && block_map.count(handler_id)) {
            unw_bb = block_map.at(handler_id);
        } else {
            unw_bb = b.append_block("b" + std::to_string(block_id) + "_unw" + std::to_string(cid));
            created_unw = true;
        }

        b.position_at_end(cur_bb);
        Value* pending = b.build_call("bronze_exception_pending", Type::i32(), {});
        Value* is_pending = b.build_ne(pending, b.build_iconst_i32(0));
        b.build_br_if(is_pending, unw_bb, cont_bb);

        if (created_unw) {
            b.position_at_end(unw_bb);
            if (fn->name() == "main") {
                b.build_call("bronze_uncaught_exception", Type::void_type(), {});
                b.build_unreachable();
            } else {
                emit_default_ret();
            }
        }
        b.position_at_end(cont_bb);
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

    if (is_ops_il_op(inst_ast.op)) {
        if (!lower_ops_instruction(this, inst_ast, b, fn, val_map, res_val)) {
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
            uint32_t key_idx = 0;
            bool found = false;
            if (!inst_ast.string_literal.empty()) {
                for (size_t k = 0; k < options_.key_constants.size(); ++k) {
                    if (options_.key_constants[k] == inst_ast.string_literal) {
                        key_idx = static_cast<uint32_t>(k);
                        found = true;
                        break;
                    }
                }
            } else {
                key_idx = inst_ast.index;
                found = true;
            }
            if (found) {
                res_val = b.build_call("bronze_global_get", Type::i64(), {
                    b.build_iconst_i32(static_cast<int32_t>(key_idx)),
                    b.build_iconst_i64(0)
                });
                emit_exception_check();
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
            res_val = b.build_and(res_val, b.build_iconst_i32(1));
            break;
        }

        case BronzeOp::TypeOf: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            Value* src = ensure_type(op0, Type::i64(), b);
            res_val = b.build_call("bronze_typeof", Type::i64(), {src});
            break;
        }

        case BronzeOp::InstanceOf: {
            Value* op0 = ensure_type(get_opd(0), Type::i64(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i64(), b);
            res_val = b.build_call("bronze_instanceof", Type::i32(), {op0, op1});
            res_val = b.build_and(res_val, b.build_iconst_i32(1));
            break;
        }

        case BronzeOp::In: {
            Value* op0 = ensure_type(get_opd(0), Type::i64(), b);
            Value* op1 = ensure_type(get_opd(1), Type::i64(), b);
            res_val = b.build_call("bronze_has_property", Type::i32(), {op0, op1});
            res_val = b.build_and(res_val, b.build_iconst_i32(1));
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
            std::vector<Value*> call_args = {ctor};
            for (uint32_t i = 0; i < argc && i < 8; ++i) {
                call_args.push_back(ensure_type(get_opd(1 + i), Type::i64(), b));
            }
            std::string helper = "bronze_construct_" + std::to_string(std::min(argc, 8u));
            res_val = b.build_call(helper, Type::i64(), call_args);
            emit_exception_check();
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
            Function* direct_fn = (!callee.empty() && fn->parent()) ? fn->parent()->get_function(callee) : nullptr;
            bool can_direct = false;
            if (direct_fn) {
                auto it_m = options_.function_meta.find(callee);
                if (it_m != options_.function_meta.end()) {
                    can_direct = it_m->second.needs_this && !it_m->second.needs_arguments &&
                                 !it_m->second.has_rest_param && !it_m->second.needs_env &&
                                 (direct_fn->param_types().size() == argc + 1);
                } else {
                    can_direct = (direct_fn->param_types().size() == argc + 1);
                }
            }
            if (can_direct) {
                std::vector<Value*> call_args = {ensure_type(recv, direct_fn->param_types()[0], b)};
                for (size_t p = 1; p < direct_fn->param_types().size(); ++p) {
                    size_t arg_idx = p - 1;
                    Value* a = arg_idx < argc ? get_opd(1 + arg_idx) : nullptr;
                    call_args.push_back(a ? ensure_type(a, direct_fn->param_types()[p], b) :
                        (direct_fn->param_types()[p] == Type::f64() ? b.build_fconst_f64(0.0) :
                         direct_fn->param_types()[p] == Type::i32() ? b.build_iconst_i32(0) :
                         b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag))));
                }
                res_val = b.build_call(callee, direct_fn->return_type(), Span<Value* const>(call_args.data(), call_args.size()));
                if (direct_fn->return_type() == Type::void_type()) {
                    res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
                }
            } else {
                Value* null_entry = b.build_iconst_i64(0);
                Value* method = b.build_call("bronze_prop_get", Type::i64(), {recv, b.build_iconst_i32(static_cast<int32_t>(inst_ast.index)), null_entry});
                emit_exception_check();
                std::vector<Value*> dyn_args = {method, recv};
                for (size_t a = 0; a < argc; ++a) {
                    dyn_args.push_back(ensure_type(get_opd(1 + a), Type::i64(), b));
                }
                res_val = build_call_dynamic(dyn_args);
            }
            emit_exception_check();
            break;
        }

        case BronzeOp::SuperCall: {
            Value* base_ctor = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> super_args = {base_ctor, this_val};
            for (size_t a = 0; a < argc; ++a) {
                super_args.push_back(ensure_type(get_opd(2 + a), Type::i64(), b));
            }
            if (argc <= 8) {
                std::string helper = "bronze_super_call_" + std::to_string(argc);
                res_val = b.build_call(helper, Type::i64(), Span<Value* const>(super_args.data(), super_args.size()));
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            emit_exception_check();
            break;
        }

        case BronzeOp::SuperGet: {
            Value* proto = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            Value* kidx = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            res_val = b.build_call("bronze_super_get", Type::i64(), {proto, kidx, this_val});
            break;
        }

        case BronzeOp::ObjectKeys: {
            Value* obj = ensure_type(get_opd(0), Type::i64(), b);
            res_val = b.build_call("bronze_object_keys", Type::i64(), {obj});
            emit_exception_check();
            break;
        }

        case BronzeOp::ForInKeys: {
            Value* obj = ensure_type(get_opd(0), Type::i64(), b);
            res_val = b.build_call("bronze_for_in_keys", Type::i64(), {obj});
            emit_exception_check();
            break;
        }

        case BronzeOp::CallDynamic: {
            Value* callee_val = ensure_type(get_opd(0), Type::i64(), b);
            Value* this_val = ensure_type(get_opd(1), Type::i64(), b);
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> dyn_args = {callee_val, this_val};
            for (size_t a = 0; a < argc; ++a) {
                dyn_args.push_back(ensure_type(get_opd(2 + a), Type::i64(), b));
            }
            res_val = build_call_dynamic(dyn_args);
            emit_exception_check();
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
            Value* code_addr = b.build_func_addr("__wrapper_" + callee);
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
            Value* code_addr = b.build_func_addr("__wrapper_" + callee);
            res_val = b.build_call("bronze_create_func", Type::i64(), {
                code_addr,
                argc_val,
                env_val
            });
            break;
        }

        case BronzeOp::PinGuard: {
            Value* bits = ensure_type(get_opd(0), Type::i64(), b);
            uint32_t key_idx = 0;
            for (size_t k = 0; k < options_.key_constants.size(); ++k) {
                if (options_.key_constants[k] == inst_ast.string_literal) {
                    key_idx = static_cast<uint32_t>(k);
                    break;
                }
            }
            int64_t kind = inst_ast.imm_i64; // 0: Number, 1: NumberOrNullish, 2: DenseArray

            if (kind == 2) {
                b.build_call("bronze_pin_check_array", Type::void_type(), {
                    b.build_iconst_i32(static_cast<int32_t>(key_idx)),
                    bits
                });
                emit_exception_check();
            } else {
                BasicBlock* cur_bb = b.current_block();
                uint32_t cid = cont_counter ? ++(*cont_counter) : 1;
                BasicBlock* bad_bb = b.append_block("b" + std::to_string(block_id) + "_pin_bad" + std::to_string(cid));
                BasicBlock* ok_bb = b.append_block("b" + std::to_string(block_id) + "_pin_ok" + std::to_string(cid));

                b.position_at_end(cur_bb);
                Value* is_num = b.build_ule(bits, b.build_iconst_i64(static_cast<int64_t>(0xFFF0000000000000ULL)));
                Value* is_ok = is_num;
                if (kind == 1) { // NumberOrNullish
                    Value* is_null = b.build_eq(bits, b.build_iconst_i64(static_cast<int64_t>(0xFFF5000000000000ULL)));
                    Value* is_undef = b.build_eq(bits, b.build_iconst_i64(static_cast<int64_t>(0xFFF6000000000000ULL)));
                    Value* is_nullish = b.build_or(is_null, is_undef);
                    is_ok = b.build_or(is_num, is_nullish);
                }
                b.build_br_if(is_ok, ok_bb, bad_bb);


                b.position_at_end(bad_bb);
                b.build_call("bronze_pin_violation", Type::i64(), {
                    b.build_iconst_i32(static_cast<int32_t>(key_idx)),
                    bits
                });

                if (handler_id != UINT32_MAX && block_map.count(handler_id)) {
                    b.build_br(block_map.at(handler_id));
                } else if (fn->name() == "main") {
                    b.build_call("bronze_uncaught_exception", Type::void_type(), {});
                    b.build_unreachable();
                } else {
                    emit_default_ret();
                }

                b.position_at_end(ok_bb);
            }
            break;
        }

        case BronzeOp::CensusRecord: {
            Value* val = get_opd(0);
            if (!val) break;
            val = ensure_type(val, Type::i64(), b);
            uint32_t key_idx = 0;
            if (!inst_ast.string_literal.empty()) {
                for (size_t k = 0; k < options_.key_constants.size(); ++k) {
                    if (options_.key_constants[k] == inst_ast.string_literal) {
                        key_idx = static_cast<uint32_t>(k);
                        break;
                    }
                }
            } else {
                key_idx = inst_ast.index;
            }
            Value* key_val = b.build_iconst_i32(static_cast<int32_t>(key_idx));
            Value* site_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.imm_i64));
            b.build_call("bronze_census_record", Type::void_type(), {key_val, site_val, val});
            break;
        }


        case BronzeOp::MathImul: {
            Value* lhs = ensure_type(get_opd(0), Type::i32(), b);
            Value* rhs = ensure_type(get_opd(1), Type::i32(), b);
            res_val = b.build_mul(lhs, rhs);
            break;
        }

        case BronzeOp::Pow: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (inst_ast.result_type == BronzeType::Dynamic) {
                res_val = b.build_call("bronze_dynamic_pow", Type::i64(), {ensure_type(op0, Type::i64(), b), ensure_type(op1, Type::i64(), b)});
            } else {
                res_val = b.build_call("bronze_pow", Type::f64(), {ensure_type(op0, Type::f64(), b), ensure_type(op1, Type::f64(), b)});
            }
            break;
        }

        case BronzeOp::MathUnary: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            op0 = ensure_type(op0, Type::f64(), b);
            std::string_view fn = "sqrt";
            switch (inst_ast.imm_i64) {
                case 0: fn = "sqrt"; break;
                case 1: fn = "fabs"; break;
                case 2: fn = "floor"; break;
                case 3: fn = "ceil"; break;
                case 4: fn = "trunc"; break;
                case 5: fn = "sin"; break;
                case 6: fn = "cos"; break;
                default: break;
            }
            res_val = b.build_call(fn, Type::f64(), {op0});
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

        case BronzeOp::MethodDefComputed: {
            Value* target = ensure_type(get_opd(0), Type::i64(), b);
            Value* key = ensure_type(get_opd(1), Type::i64(), b);
            Value* closure_val = ensure_type(get_opd(2), Type::i64(), b);
            b.build_call("bronze_method_def_computed", Type::void_type(), {target, key, closure_val});
            b.build_write_barrier(target, closure_val);
            break;
        }

        case BronzeOp::DefineOwnAttr: {
            Value* target = ensure_type(get_opd(0), Type::i64(), b);
            Value* value = ensure_type(get_opd(1), Type::i64(), b);
            Value* key_id = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            Value* mask = b.build_iconst_i32(static_cast<int32_t>(inst_ast.imm_i64));
            b.build_call("bronze_define_own_attr", Type::void_type(), {target, key_id, value, mask});
            break;
        }

        case BronzeOp::AccessorDef: {
            Value* target = ensure_type(get_opd(0), Type::i64(), b);
            Value* key_id = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
            Value* getter = ensure_type(get_opd(1), Type::i64(), b);
            Value* setter = ensure_type(get_opd(2), Type::i64(), b);
            Value* enum_val = b.build_iconst_i32(inst_ast.imm_bool ? 1 : 0);
            b.build_call("bronze_accessor_def", Type::void_type(), {target, key_id, getter, setter, enum_val});
            break;
        }

        case BronzeOp::AccessorDefComputed: {
            Value* target = ensure_type(get_opd(0), Type::i64(), b);
            Value* key = ensure_type(get_opd(1), Type::i64(), b);
            Value* getter = ensure_type(get_opd(2), Type::i64(), b);
            Value* setter = ensure_type(get_opd(3), Type::i64(), b);
            Value* enum_val = b.build_iconst_i32(inst_ast.imm_bool ? 1 : 0);
            b.build_call("bronze_accessor_def_computed", Type::void_type(), {target, key, getter, setter, enum_val});
            break;
        }

        case BronzeOp::ModuleNamespace: {
            Value* src = ensure_type(get_opd(0), Type::i64(), b);
            res_val = b.build_call("bronze_module_namespace", Type::i64(), {src});
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


        case BronzeOp::Box: {
            if (inst_ast.box_type == BronzeType::Str && inst_ast.operands.empty()) {
                Value* key_idx = b.build_iconst_i32(static_cast<int32_t>(inst_ast.index));
                res_val = b.build_call("bronze_box_str_key", Type::i64(), {key_idx});
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
                b_val = b.build_and(b_val, b.build_iconst_i32(1));
                Value* zext = b.build_zext_i64(b_val);
                Value* tag = b.build_iconst_i64(static_cast<int64_t>(kBoolTag));
                res_val = b.build_or(zext, tag);
            } else if (inst_ast.box_type == BronzeType::Str) {
                Value* ptr_val = ensure_type(op0, Type::i64(), b);
                res_val = b.build_call("bronze_box_str", Type::i64(), {ptr_val});
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
                res_val = b.build_call("bronze_unbox_bool", Type::i32(), {i_val});
                res_val = b.build_and(res_val, b.build_iconst_i32(1));
            } else if (inst_ast.result_type == BronzeType::Str) {
                Value* i_val = ensure_type(op0, Type::i64(), b);
                res_val = b.build_call("bronze_unbox_str", Type::i64(), {i_val});
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
                if (arg) args.push_back(arg);
            }
            res_val = b.build_call(callee_name, callee_ret, Span<Value* const>(args.data(), args.size()));
            emit_exception_check();
            break;
        }

        case BronzeOp::Throw: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            Value* op0_i64 = ensure_type(op0, Type::i64(), b);
            b.build_call("bronze_exception_set", Type::void_type(), {op0_i64});
            if (handler_id != UINT32_MAX && block_map.count(handler_id)) {
                b.build_br(block_map.at(handler_id));
            } else if (fn->name() == "main") {
                b.build_call("bronze_uncaught_exception", Type::void_type(), {});
                b.build_unreachable();
            } else {
                emit_default_ret();
            }
            break;
        }

        case BronzeOp::ExcTake: {
            res_val = b.build_call("bronze_exception_take", Type::i64(), {});
            break;
        }


        case BronzeOp::Print:
        case BronzeOp::PrintErr: {
            for (size_t i = 0; i < inst_ast.operands.size(); ++i) {
                if (i > 0) b.build_call("bronze_print_space", Type::void_type(), {});
                Value* arg = get_opd(i);
                if (!arg) continue;
                if (arg->type() == Type::f64()) b.build_call("bronze_print_f64", Type::void_type(), {arg});
                else if (arg->type() == Type::i32()) b.build_call("bronze_print_i32", Type::void_type(), {arg});
                else b.build_call("bronze_print_dynamic", Type::void_type(), {ensure_type(arg, Type::i64(), b)});
            }
            b.build_call("bronze_print_newline", Type::void_type());
            break;
        }

        case BronzeOp::Jump: {
            if (!block_map.count(inst_ast.target.block_id)) return false;
            BasicBlock* target_bb = block_map.at(inst_ast.target.block_id);
            std::vector<Value*> target_args;
            for (size_t i = 0; i < inst_ast.target.args.size(); ++i) {
                Value* aval = val_map.count(inst_ast.target.args[i]) ? val_map[inst_ast.target.args[i]] : nullptr;
                if (i < target_bb->params().size() && aval) aval = ensure_type(aval, target_bb->params()[i]->type(), b);
                target_args.push_back(aval);
            }
            b.build_br(target_bb, target_args);
            break;
        }

        case BronzeOp::Branch: {
            Value* cond = get_opd(0);
            if (!cond || !block_map.count(inst_ast.target.block_id) || !block_map.count(inst_ast.else_target.block_id)) return false;
            cond = ensure_type(cond, Type::i32(), b);
            cond = b.build_and(cond, b.build_iconst_i32(1));
            BasicBlock* true_bb = block_map.at(inst_ast.target.block_id);
            BasicBlock* false_bb = block_map.at(inst_ast.else_target.block_id);
            auto get_args = [&](const auto& tgt, BasicBlock* bb) {
                std::vector<Value*> args;
                for (size_t i = 0; i < tgt.args.size(); ++i) {
                    Value* a = val_map.count(tgt.args[i]) ? val_map[tgt.args[i]] : nullptr;
                    if (i < bb->params().size() && a) a = ensure_type(a, bb->params()[i]->type(), b);
                    args.push_back(a);
                }
                return args;
            };
            b.build_br_if(cond, true_bb, get_args(inst_ast.target, true_bb), false_bb, get_args(inst_ast.else_target, false_bb));
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

        default: {
            if (res_type == Type::void_type()) {
                break;
            } else if (res_type == Type::i64()) {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
                break;
            } else if (res_type == Type::i32()) {
                res_val = b.build_iconst_i32(0);
                break;
            } else if (res_type == Type::f64()) {
                res_val = b.build_fconst_f64(0.0);
                break;
            }
            return false;
        }
    }

    if (inst_ast.result_id != UINT32_MAX && res_val) {
        val_map[inst_ast.result_id] = res_val;
    }
    return true;
}

} // namespace brass::il
