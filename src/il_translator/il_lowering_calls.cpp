#include "il_lowering_calls.hpp"
#include "il_lowering.hpp"
#include "il_runtime.hpp"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace brass::il {

bool is_call_il_op(BronzeOp op) {
    switch (op) {
        case BronzeOp::Construct:
        case BronzeOp::ConstructSpread:
        case BronzeOp::MethodCall:
        case BronzeOp::MethodCallSpread:
        case BronzeOp::SuperCall:
        case BronzeOp::SuperCallSpread:
        case BronzeOp::SuperGet:
        case BronzeOp::SuperSet:
        case BronzeOp::CallDynamic:
        case BronzeOp::DynamicCallSpread:
        case BronzeOp::FuncRef:
        case BronzeOp::CreateFunc:
            return true;
        default:
            return false;
    }
}

static Value* build_call_dynamic(Builder& b, const std::vector<Value*>& dyn_args) {
    size_t argc = dyn_args.size() > 2 ? dyn_args.size() - 2 : 0;
    if (argc <= 16) {
        std::string helper = "bronze_call_dynamic_" + std::to_string(argc);
        return b.build_call(helper, Type::i64(), Span<Value* const>(dyn_args.data(), dyn_args.size()));
    }
    return b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
}

bool lower_call_instruction(
    IlLowering* lowering,
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    Value*& res_val,
    const std::function<void()>& emit_exception_check
) {
    auto get_opd = [&](size_t idx) -> Value* {
        if (idx < inst_ast.operands.size()) {
            uint32_t id = inst_ast.operands[idx];
            if (lowering) {
                return lowering->get_val_by_id(id, b, val_map);
            }
            if (val_map.count(id)) return val_map[id];
        }
        return nullptr;
    };

    auto ensure_type = [&](Value* val, Type target_type) -> Value* {
        return lowering ? lowering->ensure_type(val, target_type, b) : val;
    };

    switch (inst_ast.op) {
        case BronzeOp::Construct: {
            Value* ctor = ensure_type(get_opd(0), Type::i64());
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> call_args = {ctor};
            for (uint32_t i = 0; i < argc && i < 16; ++i) {
                call_args.push_back(ensure_type(get_opd(1 + i), Type::i64()));
            }
            std::string helper = "bronze_construct_" + std::to_string(std::min(argc, 16u));
            res_val = b.build_call(helper, Type::i64(), call_args);
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::ConstructSpread: {
            Value* callee = ensure_type(get_opd(0), Type::i64());
            Value* args = ensure_type(get_opd(1), Type::i64());
            res_val = b.build_call("bronze_construct_spread", Type::i64(), {callee, args});
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::MethodCall: {
            Value* recv = ensure_type(get_opd(0), Type::i64());
            uint32_t argc = inst_ast.param_count;
            std::string callee = lowering ? lowering->resolve_callee(inst_ast.callee_name) : inst_ast.callee_name;
            if (!callee.empty() && std::isdigit(static_cast<unsigned char>(callee[0]))) {
                uint32_t f_idx = static_cast<uint32_t>(std::stoul(callee));
                if (lowering && lowering->current_ast() && f_idx < lowering->current_ast()->functions.size()) {
                    callee = lowering->resolve_callee(lowering->current_ast()->functions[f_idx].name);
                }
            }
            Function* direct_fn = (!callee.empty() && fn && fn->parent()) ? fn->parent()->get_function(callee) : nullptr;
            bool can_direct = false;
            if (direct_fn && lowering) {
                auto it_m = lowering->options().function_meta.find(callee);
                if (it_m != lowering->options().function_meta.end()) {
                    can_direct = it_m->second.needs_this && !it_m->second.needs_arguments &&
                                 !it_m->second.has_rest_param && !it_m->second.needs_env &&
                                 (direct_fn->param_types().size() == argc + 1);
                } else {
                    can_direct = (direct_fn->param_types().size() == argc + 1);
                }
            } else if (direct_fn) {
                can_direct = (direct_fn->param_types().size() == argc + 1);
            }

            if (can_direct) {
                std::vector<Value*> call_args = {ensure_type(recv, direct_fn->param_types()[0])};
                for (size_t p = 1; p < direct_fn->param_types().size(); ++p) {
                    size_t arg_idx = p - 1;
                    Value* a = arg_idx < argc ? get_opd(1 + arg_idx) : nullptr;
                    call_args.push_back(a ? ensure_type(a, direct_fn->param_types()[p]) :
                        (direct_fn->param_types()[p] == Type::f64() ? b.build_fconst_f64(0.0) :
                         direct_fn->param_types()[p] == Type::i32() ? b.build_iconst_i32(0) :
                         b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag))));
                }
                res_val = b.build_call(callee, direct_fn->return_type(), Span<Value* const>(call_args.data(), call_args.size()));
                if (direct_fn->return_type() == Type::void_type()) {
                    res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
                }
            } else if (Value* site = (lowering ? lowering->prop_lowering().ic_site(b, inst_ast.ic_index) : nullptr)) {
                Value* argv = nullptr;
                bool pushed_frame = false;
                if (argc == 0) {
                    argv = b.build_iconst_i64(0);
                } else if (lowering && lowering->current_fn_frame_ptr() != nullptr) {
                    const int32_t base = static_cast<int32_t>(16 + lowering->method_argv_slot() * 8);
                    for (size_t a = 0; a < argc; ++a) {
                        Value* arg = ensure_type(get_opd(1 + a), Type::i64());
                        b.build_store(Type::i64(), lowering->current_fn_frame_ptr(),
                                      static_cast<int32_t>(base + a * 8), arg);
                    }
                    argv = b.build_add(lowering->current_fn_frame_ptr(), b.build_iconst_i64(base));
                } else {
                    Value* frame = b.build_call("bronze_gc_frame_push", Type::ptr(),
                                                {b.build_iconst_i32(static_cast<int32_t>(argc))});
                    for (size_t a = 0; a < argc; ++a) {
                        Value* arg = ensure_type(get_opd(1 + a), Type::i64());
                        b.build_store(Type::i64(), frame, static_cast<int32_t>(16 + a * 8), arg);
                    }
                    argv = b.build_add(frame, b.build_iconst_i64(16));
                    pushed_frame = true;
                }
                Value* argc_val = b.build_iconst_i32(static_cast<int32_t>(argc));
                Value* key_id = lowering ? lowering->get_key_id(b, inst_ast.index) : b.build_iconst_i32(inst_ast.index);
                res_val = b.build_call("bronze_call_method", Type::i64(),
                                       {recv, key_id, argc_val, argv, site});
                if (pushed_frame) {
                    b.build_call("bronze_gc_frame_pop", Type::void_type(), {});
                }
            } else {
                Value* null_entry = b.build_iconst_i64(0);
                Value* key_id = lowering ? lowering->get_key_id(b, inst_ast.index) : b.build_iconst_i32(inst_ast.index);
                Value* method = b.build_call("bronze_prop_get", Type::i64(), {recv, key_id, null_entry});
                if (emit_exception_check) emit_exception_check();
                recv = ensure_type(get_opd(0), Type::i64());
                std::vector<Value*> dyn_args = {method, recv};
                for (size_t a = 0; a < argc; ++a) {
                    dyn_args.push_back(ensure_type(get_opd(1 + a), Type::i64()));
                }
                res_val = build_call_dynamic(b, dyn_args);
            }
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::MethodCallSpread: {
            Value* this_val = ensure_type(get_opd(0), Type::i64());
            Value* args = ensure_type(get_opd(1), Type::i64());
            Value* key_id = lowering ? lowering->get_key_id(b, inst_ast.index) : b.build_iconst_i32(inst_ast.index);
            Value* site = lowering ? lowering->prop_lowering().ic_site(b, inst_ast.ic_index) : nullptr;
            Value* entry = site ? site : b.build_iconst_i64(0);
            res_val = b.build_call("bronze_call_method_spread", Type::i64(), {this_val, key_id, args, entry});
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::SuperCall: {
            Value* base_ctor = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> super_args = {base_ctor, this_val};
            for (size_t a = 0; a < argc; ++a) {
                super_args.push_back(ensure_type(get_opd(2 + a), Type::i64()));
            }
            if (argc <= 16) {
                std::string helper = "bronze_super_call_" + std::to_string(argc);
                res_val = b.build_call(helper, Type::i64(), Span<Value* const>(super_args.data(), super_args.size()));
            } else {
                res_val = b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag));
            }
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::SuperCallSpread: {
            Value* base_ctor = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            Value* args = ensure_type(get_opd(2), Type::i64());
            res_val = b.build_call("bronze_super_call_spread", Type::i64(), {base_ctor, this_val, args});
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::SuperGet: {
            Value* proto = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            Value* kidx = lowering ? lowering->get_key_id(b, inst_ast.index) : b.build_iconst_i32(inst_ast.index);
            res_val = b.build_call("bronze_super_get", Type::i64(), {proto, kidx, this_val});
            break;
        }

        case BronzeOp::SuperSet: {
            Value* proto = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            Value* val = ensure_type(get_opd(2), Type::i64());
            Value* kidx = lowering ? lowering->get_key_id(b, inst_ast.index) : b.build_iconst_i32(inst_ast.index);
            Value* strict = b.build_iconst_i32(inst_ast.imm_i64 != 0 ? 1 : 0);
            b.build_call("bronze_super_set", Type::void_type(), {proto, kidx, this_val, val, strict});
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::CallDynamic: {
            Value* callee_val = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            uint32_t argc = inst_ast.param_count;
            std::vector<Value*> dyn_args = {callee_val, this_val};
            for (size_t a = 0; a < argc; ++a) {
                dyn_args.push_back(ensure_type(get_opd(2 + a), Type::i64()));
            }
            res_val = build_call_dynamic(b, dyn_args);
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::DynamicCallSpread: {
            Value* callee = ensure_type(get_opd(0), Type::i64());
            Value* this_val = ensure_type(get_opd(1), Type::i64());
            Value* args = ensure_type(get_opd(2), Type::i64());
            res_val = b.build_call("bronze_dynamic_call_spread", Type::i64(), {callee, this_val, args});
            if (emit_exception_check) emit_exception_check();
            break;
        }

        case BronzeOp::FuncRef: {
            std::string callee = inst_ast.callee_name;
            if (!callee.empty() && std::isdigit(static_cast<unsigned char>(callee[0]))) {
                uint32_t f_idx = static_cast<uint32_t>(std::stoul(callee));
                if (lowering && lowering->current_ast() && f_idx < lowering->current_ast()->functions.size()) {
                    callee = lowering->current_ast()->functions[f_idx].name;
                }
            }
            callee = lowering ? lowering->resolve_callee(callee) : callee;
            Value* code_addr = b.build_func_addr("__wrapper_" + callee);
            uint32_t arity = 0;
            uint32_t length = 0;
            uint32_t name_key_val = 0xFFFFFFFFu;
            uint32_t fn_flags_val = 0x03;
            if (lowering) {
                auto it_meta = lowering->options().function_meta.find(callee);
                if (it_meta != lowering->options().function_meta.end()) {
                    fn_flags_val = it_meta->second.fn_flags;
                    name_key_val = it_meta->second.name_key;
                    length = it_meta->second.required_args;
                    arity = it_meta->second.adapt_arity;
                } else if (lowering->current_ast()) {
                    for (const auto& f : lowering->current_ast()->functions) {
                        if (f.name == callee) {
                            length = static_cast<uint32_t>(f.params.size());
                            arity = length;
                            break;
                        }
                    }
                }
            }
            Value* arity_val = b.build_iconst_i32(static_cast<int32_t>(arity));
            Value* length_val = b.build_iconst_i32(static_cast<int32_t>(length));
            Value* name_key = (name_key_val != 0xFFFFFFFFu && lowering)
                ? lowering->get_key_id(b, name_key_val)
                : b.build_iconst_i32(static_cast<int32_t>(0xFFFFFFFFu));
            Value* fn_flags = b.build_iconst_i32(static_cast<int32_t>(fn_flags_val));
            Value* slot_cell = b.build_iconst_i64(0);
            res_val = b.build_call("bronze_function_singleton", Type::i64(), {
                code_addr,
                arity_val,
                length_val,
                name_key,
                fn_flags,
                slot_cell
            });
            break;
        }

        case BronzeOp::CreateFunc: {
            std::string callee = lowering ? lowering->resolve_create_func_callee(inst_ast.callee_name) : inst_ast.callee_name;
            Value* code_addr = b.build_func_addr("__wrapper_" + callee);
            Value* arity_val = b.build_iconst_i32(static_cast<int32_t>(inst_ast.param_count));
            Value* env_val = ensure_type(get_opd(0), Type::i64());
            uint32_t length = 0;
            uint32_t name_key_val = 0xFFFFFFFFu;
            uint32_t fn_flags_val = 0x03;
            if (lowering) {
                auto it_meta = lowering->options().function_meta.find(callee);
                if (it_meta != lowering->options().function_meta.end()) {
                    fn_flags_val = it_meta->second.fn_flags;
                    name_key_val = it_meta->second.name_key;
                    length = it_meta->second.required_args;
                }
            }
            Value* length_val = b.build_iconst_i32(static_cast<int32_t>(length));
            Value* name_key = (name_key_val != 0xFFFFFFFFu && lowering)
                ? lowering->get_key_id(b, name_key_val)
                : b.build_iconst_i32(static_cast<int32_t>(0xFFFFFFFFu));
            Value* fn_flags = b.build_iconst_i32(static_cast<int32_t>(fn_flags_val));
            res_val = b.build_call("bronze_create_function", Type::i64(), {
                code_addr,
                arity_val,
                length_val,
                name_key,
                fn_flags,
                env_val
            });
            break;
        }

        default:
            return false;
    }

    return true;
}

} // namespace brass::il
