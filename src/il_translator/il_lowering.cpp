#include "il_lowering.hpp"
#include "il_lowering_coro.hpp"
#include "il_runtime.hpp"
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/il_translator/il_pipeline.hpp>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace brass::il {

Type lower_type(BronzeType t) {
    switch (t) {
        case BronzeType::Void: return Type::void_type();
        case BronzeType::Bool: return Type::i8();
        case BronzeType::I32: return Type::i32();
        case BronzeType::F64: return Type::f64();
        case BronzeType::Str: return Type::ptr();
        case BronzeType::Dynamic: return Type::i64();
        case BronzeType::Unknown: return Type::i64();
    }
    return Type::i64();
}

IlLowering::IlLowering(const TranslatorOptions& options, DiagnosticReporter* diag)
    : options_(options), diag_(diag), prop_lowering_(options.enable_pic, options.enable_inlined_fastpaths),
      alloc_lowering_(options.enable_tlab, options.use_bronze_tlab ? AllocLoweringHelper::Model::BronzeTLS : AllocLoweringHelper::Model::BrassHostGC) {}

Value* IlLowering::ensure_type(Value* val, Type target_type, Builder& b) {
    if (!val || val->type() == target_type) return val;

    Type src_type = val->type();
    if (target_type == Type::i8()) {
        if (src_type == Type::i64()) {
            val = b.build_trunc_i32(val);
        }
        return b.build_trunc_i8(val);
    }
    if (src_type == Type::i8()) {
        val = b.build_zext_i64(val);
        if (target_type == Type::i32()) {
            return b.build_trunc_i32(val);
        }
        src_type = Type::i64();
    }
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
    Value* map_addr = b.build_func_addr(module_sym("__bronze_key_map"));
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
        Value* env_addr = b.build_func_addr(module_sym("__bronze_module_env"));
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
    if (a.static_slot != b.static_slot || a.ic_index != b.ic_index || a.is_mono != b.is_mono || a.is_fn_recv != b.is_fn_recv) return false;
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
    mod->set_pinned_tls_register(options_.pin_tls_register);
    current_file_id_ = mod->debug_context().get_or_add_file(ast.name.empty() ? "<anonymous>" : ast.name);

    // Register external runtime helper functions
    register_all_module_external_symbols(mod.get(), options_.entry_symbol);
    prop_lowering_.set_key_map_sym(module_sym("__bronze_key_map"));
    prop_lowering_.set_ic_table(module_sym("__bronze_ic_table"), options_.ic_site_count);

    current_ast_ = &ast;




    // 1. Forward-declare all functions (uniquifying any duplicate function names from Bronze)
    std::unordered_map<std::string, std::vector<size_t>> name_to_indices;
    for (size_t i = 0; i < ast.functions.size(); ++i) {
        name_to_indices[ast.functions[i].name].push_back(i);
    }

    caller_to_callee_map_.clear();
    external_signatures_.clear();
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
        if (fn_ast.blocks.empty()) {
            external_signatures_[fn_name] = {ret_type, std::move(param_types)};
            continue;
        }
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
        const auto& fn_ast = ast.functions[i];
        if (fn_ast.blocks.empty()) continue;
        current_fn_idx_ = i;
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
        PassPipelineHooks hooks;
        hooks.after_pass = [&](std::string_view name) {
            if (!options_.verify_after_each_pass) return true;
            if (!verify_module(*mod, diag_)) {
                std::fprintf(stderr, "[FATAL] Broken after %.*s\n",
                             static_cast<int>(name.size()), name.data());
                return false;
            }
            return true;
        };
        if (!run_pass_pipeline(*mod, pass_pipeline_options(options_), hooks)) {
            return nullptr;
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
    method_argv_slot_ = 0;
    uint32_t total_slots = 0;

    bool is_coro_fn = false;
    for (const auto& blk : fn_ast.blocks) {
        for (const auto& inst : blk.instructions) {
            if (inst.op == BronzeOp::Yield) {
                is_coro_fn = true;
                break;
            }
        }
        if (is_coro_fn) break;
    }

    if (!is_coro_fn) {
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
        // The argv block a dynamic method call stages its arguments in
        // (`bronze_call_method` takes `const uint64_t* argv`): the widest
        // method call's worth of slots at the END of the same GC frame, so
        // the arguments are rooted — and forwarded — exactly as the frame's
        // other Values are while the helper's property read can collect.
        // Nested calls cannot overlap: an argument is a value already
        // computed by the time its call stores it, and the stores happen
        // right before the call. Sized over every method call, including
        // the ones that resolve to a direct edge, which merely leaves a few
        // slots holding the `undefined` the push gave them.
        uint32_t widest_method_argc = 0;
        for (const auto& blk : fn_ast.blocks) {
            for (const auto& inst : blk.instructions) {
                if (inst.op == BronzeOp::MethodCall && inst.param_count > widest_method_argc) {
                    widest_method_argc = inst.param_count;
                }
                if ((inst.op == BronzeOp::Construct || inst.op == BronzeOp::SuperCall || inst.op == BronzeOp::CallDynamic) &&
                    inst.param_count > 16 && inst.param_count > widest_method_argc) {
                    widest_method_argc = inst.param_count;
                }
            }
        }
        method_argv_slot_ = total_slots;
        total_slots += widest_method_argc;
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
        if (options_.pin_tls_register) {
            // The module entry is the one function the runtime calls without
            // its trampoline (bro and the CLI call the exported symbol
            // directly), so it fetches the block itself. Every other
            // function arrives with the register set by its caller: a
            // compiled caller never touches it, a runtime caller went
            // through `bronze_enter_js`.
            if (fn_name == "main") {
                Value* tls = b.build_call("bronze_tls_enter", Type::i64(), {});
                b.build_pinned_tls_write(tls);
            }
            // Stack-limit check, before anything is pushed that an early
            // return would have to pop: below the limit, the runtime raises
            // the RangeError and the function returns as if it had thrown.
            Value* tls = b.build_pinned_tls_read();
            Value* limit = b.build_load(Type::i64(), tls, kBronzeTlsStackLimitOff);
            Value* sp = b.build_read_sp();
            Value* below = b.build_ult(sp, limit);
            // append_block moves the insertion point, so come back to the
            // entry for the branch.
            BasicBlock* overflow_bb = b.append_block("stack_overflow");
            BasicBlock* body_bb = b.append_block("stack_ok");
            b.position_at_end(entry_bb);
            b.build_br_if(below, overflow_bb, body_bb);
            b.position_at_end(overflow_bb);
            b.build_call("bronze_stack_overflow", Type::void_type(), {});
            if (fn->return_type() == Type::void_type()) {
                b.build_ret_void();
            } else if (fn->return_type() == Type::f64()) {
                b.build_ret(b.build_fconst_f64(0.0));
            } else if (fn->return_type() == Type::i32()) {
                b.build_ret(b.build_iconst_i32(0));
            } else {
                b.build_ret(b.build_iconst_i64(static_cast<int64_t>(kUndefinedTag)));
            }
            // The IL's first block now lowers into the checked continuation;
            // its parameters stay on the real entry block, which dominates it.
            block_map[fn_ast.blocks[0].id] = body_bb;
            b.position_at_end(body_bb);
            stack_check_entry_bb_ = entry_bb;
            stack_check_overflow_bb_ = overflow_bb;
            stack_check_body_bb_ = body_bb;
        }
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
            Value* env_addr = b.build_func_addr(module_sym("__bronze_module_env"));
            Value* count_val = b.build_iconst_i64(1);
            b.build_call("bronze_register_value_cells", Type::void_type(), {env_addr, count_val});
            Value* tpl_addr = b.build_func_addr(module_sym("__bronze_template_cells"));
            const size_t tpl_count = std::max<size_t>(1024, static_cast<size_t>(options_.template_site_count) + 128);
            Value* tpl_cells_count = b.build_iconst_i64(static_cast<int64_t>(tpl_count));
            b.build_call("bronze_register_value_cells", Type::void_type(), {tpl_addr, tpl_cells_count});
            const std::string key_sym = (options_.entry_symbol.empty() || options_.entry_symbol == "main" || options_.entry_symbol == "bronze_main")
                                            ? "bronze_main_key_constants"
                                            : (options_.entry_symbol + "_key_constants");
            Value* manifest_addr = b.build_func_addr(key_sym);
            Value* map_addr = b.build_func_addr(module_sym("__bronze_key_map"));
            b.build_call("bronze_register_key_manifest", Type::void_type(), {manifest_addr, map_addr});
            // The method-call sites' env words, registered as value cells
            // (bronze_abi.h, bronze_register_method_ic_cells): a latched
            // direct-form entry may hold a closure's environment record —
            // a heap Value in module data — and only registration keeps it
            // current across a collection.
            if (options_.ic_site_count > 0 && !options_.method_ic_sites.empty()) {
                Value* table_addr = b.build_func_addr(module_sym("__bronze_ic_table"));
                Value* sites_addr = b.build_func_addr(module_sym("__bronze_method_ic_sites"));
                Value* site_count = b.build_iconst_i64(static_cast<int64_t>(options_.method_ic_sites.size()));
                b.build_call("bronze_register_method_ic_cells", Type::void_type(), {table_addr, sites_addr, site_count});
            }
            if (options_.enable_census && options_.census_site_count > 0) {
                Value* out_path = b.build_func_addr(module_sym("__bronze_census_out_path"));
                Value* sites_addr = b.build_func_addr(module_sym("__bronze_census_sites"));
                Value* site_count = b.build_iconst_i32(static_cast<int32_t>(options_.census_site_count));
                b.build_call("bronze_census_register", Type::void_type(), {out_path, sites_addr, site_count, map_addr});
            }
            for (size_t f = 0; f < options_.source_files.size(); ++f) {
                if (options_.source_files[f].entry_count == 0) continue;
                Value* text_addr = b.build_func_addr(module_sym("__bronze_source_text_" + std::to_string(f)));
                Value* text_len = b.build_iconst_i32(static_cast<int32_t>(options_.source_files[f].text_len));
                Value* entries_addr = b.build_func_addr(module_sym("__bronze_source_entries_" + std::to_string(f)));
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
            } else {
                b.clear_current_loc();
            }
            if (!lower_instruction(inst_ast, b, fn, val_map, block_map, blk_ast.handler_id, blk_ast.id, &cont_counter)) {
                return false;
            }
        }
    }

    if (stack_check_entry_bb_ != nullptr) {
        // A function that calls nothing cannot recurse, and its frame is the
        // one the prologue elides when nothing forces it — so the check goes
        // unless the body made a call.
        bool has_call = false;
        for (BasicBlock* bb : fn->blocks()) {
            if (bb == stack_check_overflow_bb_) continue;
            for (Instruction* inst = bb->head(); inst && !has_call; inst = inst->next()) {
                has_call = inst->is_call();
            }
            if (has_call) break;
        }
        if (!has_call) {
            BasicBlock* entry_bb = stack_check_entry_bb_;
            while (entry_bb->head()) entry_bb->remove_instruction(entry_bb->head());
            b.position_at_end(entry_bb);
            b.build_br(stack_check_body_bb_);
            fn->remove_block(stack_check_overflow_bb_);
        }
        stack_check_entry_bb_ = nullptr;
        stack_check_overflow_bb_ = nullptr;
        stack_check_body_bb_ = nullptr;
    }

    fn->rebuild_cfg_predecessors();
    fn->sort_blocks_rpo();
    fn->rebuild_cfg_predecessors();
    return true;
}

} // namespace brass::il
