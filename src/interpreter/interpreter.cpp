#include <brass/interpreter/interpreter.hpp>
#include <iostream>
#include <cmath>

namespace brass {

Interpreter::Interpreter(size_t gc_semispace_size)
    : gc_(gc_semispace_size) {
    gc_.set_root_provider([this](std::vector<uintptr_t*>& roots) {
        this->collect_all_roots(roots);
    });
    register_builtin_host_functions();
}

void Interpreter::register_builtin_host_functions() {
    register_external_function("brass_gc_alloc", [](Interpreter& interp, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) {
            throw InterpreterException("brass_gc_alloc requires at least 1 argument (size)");
        }
        size_t size = static_cast<size_t>(args[0].is_i32() ? args[0].as_u32() : args[0].as_u64());
        uint64_t pointer_mask = (args.size() > 1) ? args[1].as_u64() : 0ULL;
        uint32_t type_tag = (args.size() > 2) ? args[2].as_u32() : 0U;
        uintptr_t addr = interp.allocate_gc(size, pointer_mask, type_tag);
        return RuntimeValue::from_gcref(addr);
    });

    register_external_function("brass_gc_collect", [](Interpreter& interp, const std::vector<RuntimeValue>&) -> RuntimeValue {
        interp.gc().collect();
        return RuntimeValue::from_void();
    });

    register_external_function("sqrt", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::sqrt(args[0].as_f64()));
    });

    register_external_function("fabs", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::fabs(args[0].as_f64()));
    });

    register_external_function("floor", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::floor(args[0].as_f64()));
    });

    register_external_function("ceil", [](Interpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        if (args.empty()) return RuntimeValue::from_f64(0.0);
        return RuntimeValue::from_f64(std::ceil(args[0].as_f64()));
    });
}

void Interpreter::collect_all_roots(std::vector<uintptr_t*>& roots) {
    for (InterpreterFrame* f = current_frame_; f != nullptr; f = f->caller()) {
        f->collect_roots(roots);
    }
}

uintptr_t Interpreter::allocate_gc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> roots;
    collect_all_roots(roots);
    return gc_.allocate(size, pointer_mask, type_tag, roots);
}

void Interpreter::register_external_function(std::string_view name, HostFn fn) {
    external_functions_[std::string(name)] = std::move(fn);
}

bool Interpreter::has_external_function(std::string_view name) const noexcept {
    return external_functions_.find(std::string(name)) != external_functions_.end();
}

void Interpreter::register_function_pointer(uintptr_t ptr, const Function* fn) {
    function_pointers_[ptr] = fn;
}

void Interpreter::register_function_pointer(uintptr_t ptr, HostFn fn) {
    host_function_pointers_[ptr] = std::move(fn);
}

void Interpreter::patch_const(std::string_view symbol, int64_t val) {
    patched_consts_[std::string(symbol)] = val;
}

int64_t Interpreter::get_patched_const(std::string_view symbol, int64_t default_val) const {
    auto it = patched_consts_.find(std::string(symbol));
    if (it != patched_consts_.end()) {
        return it->second;
    }
    return default_val;
}

void Interpreter::patch_call(std::string_view site, std::string_view target) {
    patched_calls_[std::string(site)] = std::string(target);
}

std::string_view Interpreter::get_patched_call(std::string_view site, std::string_view default_callee) const {
    auto it = patched_calls_.find(std::string(site));
    if (it != patched_calls_.end()) {
        return it->second;
    }
    return default_callee;
}

RuntimeValue Interpreter::run(const Function& fn) {
    std::vector<RuntimeValue> empty_args;
    return run(fn, empty_args);
}

RuntimeValue Interpreter::run(const Function& fn, const std::vector<RuntimeValue>& args) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    return execute_function(fn, args);
}

RuntimeValue Interpreter::run(std::string_view fn_name) {
    std::vector<RuntimeValue> empty_args;
    return run(fn_name, empty_args);
}

RuntimeValue Interpreter::run(std::string_view fn_name, const std::vector<RuntimeValue>& args) {
    if (module_) {
        const Function* fn = module_->get_function(fn_name);
        if (fn) {
            return execute_function(*fn, args);
        }
    }
    auto it = external_functions_.find(std::string(fn_name));
    if (it != external_functions_.end()) {
        return it->second(*this, args);
    }
    throw InterpreterException("Function not found: " + std::string(fn_name));
}

RuntimeValue Interpreter::run(const Module& mod, std::string_view entry_name) {
    std::vector<RuntimeValue> empty_args;
    return run(mod, entry_name, empty_args);
}

RuntimeValue Interpreter::run(const Module& mod, std::string_view entry_name, const std::vector<RuntimeValue>& args) {
    module_ = &mod;
    const Function* fn = mod.get_function(entry_name);
    if (!fn) {
        throw InterpreterException("Entry function not found in module: " + std::string(entry_name));
    }
    return execute_function(*fn, args);
}

RuntimeValue Interpreter::resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    BasicBlock* target_bb = fn.get_resume_target(resume_id);
    if (!target_bb) {
        throw InterpreterException("Resume target ID " + std::to_string(resume_id) + " not found in function " + std::string(fn.name()));
    }
    return execute_function_from_block(fn, target_bb, state_values);
}

RuntimeValue Interpreter::execute_function(const Function& fn, const std::vector<RuntimeValue>& args) {
    BasicBlock* entry = fn.entry_block();
    if (!entry) {
        return RuntimeValue::from_void();
    }
    return execute_function_from_block(fn, entry, args);
}

RuntimeValue Interpreter::execute_function_from_block(const Function& fn, BasicBlock* start_block, const std::vector<RuntimeValue>& block_args) {
    if (call_depth_ + 1 > max_call_depth_) {
        throw InterpreterException("Maximum interpreter call depth exceeded (" + std::to_string(max_call_depth_) + ")");
    }

    InterpreterFrame frame(&fn, current_frame_);
    InterpreterFrame* prev_frame = current_frame_;
    current_frame_ = &frame;
    call_depth_++;

    struct StackGuard {
        InterpreterFrame*& cur;
        InterpreterFrame* prev;
        size_t& depth;
        ~StackGuard() {
            cur = prev;
            depth--;
        }
    } guard{current_frame_, prev_frame, call_depth_};

    // Bind block arguments to entry block parameters
    for (size_t i = 0; i < start_block->param_count() && i < block_args.size(); ++i) {
        frame.set_value(start_block->param(i), block_args[i]);
    }

    BasicBlock* cur_bb = start_block;

    while (cur_bb != nullptr) {
        bool transitioned = false;

        for (Instruction* inst = cur_bb->head(); inst != nullptr; inst = inst->next()) {
            if (max_instructions_ > 0 && ++total_instructions_executed_ > max_instructions_) {
                throw InterpreterException("Maximum instruction execution count exceeded (" + std::to_string(max_instructions_) + ")");
            } else {
                total_instructions_executed_++;
            }

            switch (inst->opcode()) {
                case Opcode::iconst_i32: {
                    frame.set_value(inst->result(), RuntimeValue::from_i32(inst->imm_i32()));
                    break;
                }
                case Opcode::iconst_i64: {
                    frame.set_value(inst->result(), RuntimeValue::from_i64(inst->imm_i64()));
                    break;
                }
                case Opcode::fconst_f64: {
                    frame.set_value(inst->result(), RuntimeValue::from_f64(inst->imm_f64()));
                    break;
                }
                case Opcode::patchable_const_i32: {
                    int64_t v = get_patched_const(inst->symbol(), inst->imm_i32());
                    frame.set_value(inst->result(), RuntimeValue::from_i32(static_cast<int32_t>(v)));
                    break;
                }
                case Opcode::patchable_const_i64: {
                    int64_t v = get_patched_const(inst->symbol(), inst->imm_i64());
                    frame.set_value(inst->result(), RuntimeValue::from_i64(v));
                    break;
                }

                case Opcode::sext_i64: {
                    frame.set_value(inst->result(), val_sext_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::zext_i64: {
                    frame.set_value(inst->result(), val_zext_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::trunc_i32: {
                    frame.set_value(inst->result(), val_trunc_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::fptosi_i32: {
                    frame.set_value(inst->result(), val_fptosi_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::fptosi_i64: {
                    frame.set_value(inst->result(), val_fptosi_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::sitofp_f64_i32: {
                    frame.set_value(inst->result(), val_sitofp_f64_i32(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::sitofp_f64_i64: {
                    frame.set_value(inst->result(), val_sitofp_f64_i64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::bitcast_i64_f64: {
                    frame.set_value(inst->result(), val_bitcast_i64_f64(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::bitcast_f64_i64: {
                    frame.set_value(inst->result(), val_bitcast_f64_i64(frame.get_value(inst->operand(0))));
                    break;
                }

                case Opcode::add: {
                    frame.set_value(inst->result(), val_add(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sub: {
                    frame.set_value(inst->result(), val_sub(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::mul: {
                    frame.set_value(inst->result(), val_mul(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sdiv: {
                    frame.set_value(inst->result(), val_sdiv(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::udiv: {
                    frame.set_value(inst->result(), val_udiv(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::smod: {
                    frame.set_value(inst->result(), val_smod(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::umod: {
                    frame.set_value(inst->result(), val_umod(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::neg: {
                    frame.set_value(inst->result(), val_neg(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::and_: {
                    frame.set_value(inst->result(), val_and(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::or_: {
                    frame.set_value(inst->result(), val_or(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::xor_: {
                    frame.set_value(inst->result(), val_xor(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::shl: {
                    frame.set_value(inst->result(), val_shl(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::lshr: {
                    frame.set_value(inst->result(), val_lshr(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ashr: {
                    frame.set_value(inst->result(), val_ashr(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::not_: {
                    frame.set_value(inst->result(), val_not(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::clz: {
                    frame.set_value(inst->result(), val_clz(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::ctz: {
                    frame.set_value(inst->result(), val_ctz(frame.get_value(inst->operand(0))));
                    break;
                }
                case Opcode::popcnt: {
                    frame.set_value(inst->result(), val_popcnt(frame.get_value(inst->operand(0))));
                    break;
                }

                case Opcode::eq: {
                    frame.set_value(inst->result(), val_eq(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ne: {
                    frame.set_value(inst->result(), val_ne(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::slt: {
                    frame.set_value(inst->result(), val_slt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ult: {
                    frame.set_value(inst->result(), val_ult(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sle: {
                    frame.set_value(inst->result(), val_sle(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ule: {
                    frame.set_value(inst->result(), val_ule(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sgt: {
                    frame.set_value(inst->result(), val_sgt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::ugt: {
                    frame.set_value(inst->result(), val_ugt(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::sge: {
                    frame.set_value(inst->result(), val_sge(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }
                case Opcode::uge: {
                    frame.set_value(inst->result(), val_uge(frame.get_value(inst->operand(0)), frame.get_value(inst->operand(1))));
                    break;
                }

                case Opcode::load: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue res = gc_.read_memory(base.raw_bits(), inst->offset(), inst->type());
                    frame.set_value(inst->result(), res);
                    break;
                }
                case Opcode::store: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue val = frame.get_value(inst->operand(1));
                    gc_.write_memory(base.raw_bits(), inst->offset(), inst->memory_type(), val);
                    break;
                }
                case Opcode::load_indexed: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue idx = frame.get_value(inst->operand(1));
                    int64_t idx_val = idx.is_i32() ? idx.as_i32() : idx.as_i64();
                    int32_t effective_offset = static_cast<int32_t>(idx_val * inst->scale()) + inst->offset();
                    RuntimeValue res = gc_.read_memory(base.raw_bits(), effective_offset, inst->type());
                    frame.set_value(inst->result(), res);
                    break;
                }
                case Opcode::store_indexed: {
                    RuntimeValue base = frame.get_value(inst->operand(0));
                    RuntimeValue idx = frame.get_value(inst->operand(1));
                    RuntimeValue val = frame.get_value(inst->operand(2));
                    int64_t idx_val = idx.is_i32() ? idx.as_i32() : idx.as_i64();
                    int32_t effective_offset = static_cast<int32_t>(idx_val * inst->scale()) + inst->offset();
                    gc_.write_memory(base.raw_bits(), effective_offset, inst->memory_type(), val);
                    break;
                }

                case Opcode::call: {
                    std::string_view callee = inst->symbol();
                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count());
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    const Function* target_fn = module_ ? module_->get_function(callee) : nullptr;
                    if (target_fn) {
                        call_res = execute_function(*target_fn, call_args);
                    } else {
                        auto it = external_functions_.find(std::string(callee));
                        if (it != external_functions_.end()) {
                            call_res = it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Call to undefined function: " + std::string(callee));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::call_indirect: {
                    RuntimeValue callee_val = frame.get_value(inst->operand(0));
                    uintptr_t callee_ptr = callee_val.as_ptr();

                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count() > 0 ? inst->operand_count() - 1 : 0);
                    for (size_t i = 1; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    auto fn_it = function_pointers_.find(callee_ptr);
                    if (fn_it != function_pointers_.end()) {
                        call_res = execute_function(*fn_it->second, call_args);
                    } else {
                        auto host_it = host_function_pointers_.find(callee_ptr);
                        if (host_it != host_function_pointers_.end()) {
                            call_res = host_it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Call indirect to unregistered target pointer: " + std::to_string(callee_ptr));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::patchable_call: {
                    std::string_view default_callee = inst->extra_symbol().empty() ? inst->symbol() : inst->extra_symbol();
                    std::string_view callee = get_patched_call(inst->symbol(), default_callee);

                    std::vector<RuntimeValue> call_args;
                    call_args.reserve(inst->operand_count());
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        call_args.push_back(frame.get_value(inst->operand(i)));
                    }

                    RuntimeValue call_res;
                    const Function* target_fn = module_ ? module_->get_function(callee) : nullptr;
                    if (target_fn) {
                        call_res = execute_function(*target_fn, call_args);
                    } else {
                        auto it = external_functions_.find(std::string(callee));
                        if (it != external_functions_.end()) {
                            call_res = it->second(*this, call_args);
                        } else {
                            throw InterpreterException("Patchable call to undefined function: " + std::string(callee));
                        }
                    }

                    if (inst->result()) {
                        frame.set_value(inst->result(), call_res);
                    }
                    break;
                }

                case Opcode::safepoint: {
                    if (gc_.stress_mode()) {
                        gc_.collect();
                    }
                    break;
                }

                case Opcode::guard: {
                    RuntimeValue cond = frame.get_value(inst->operand(0));
                    if (cond.as_i32() == 0) {
                        std::vector<RuntimeValue> captured;
                        captured.reserve(inst->state_map().size());
                        for (Value* v : inst->state_map()) {
                            captured.push_back(frame.get_value(v));
                        }

                        last_deopt_.deoptimized = true;
                        last_deopt_.exit_stub = std::string(inst->symbol());
                        last_deopt_.state_map = std::move(captured);
                        last_deopt_.resume_id = inst->resume_id();

                        if (deopt_handler_) {
                            return deopt_handler_(*this, last_deopt_);
                        }

                        if (module_) {
                            const Function* stub_fn = module_->get_function(inst->symbol());
                            if (stub_fn) {
                                return execute_function(*stub_fn, last_deopt_.state_map);
                            }
                        }

                        throw DeoptException(last_deopt_);
                    }
                    break;
                }

                case Opcode::resume_point: {
                    // Resume point metadata marker: no-op during forward execution
                    break;
                }

                case Opcode::br: {
                    std::vector<RuntimeValue> target_args;
                    target_args.reserve(inst->branch_target().args.size());
                    for (Value* a : inst->branch_target().args) {
                        target_args.push_back(frame.get_value(a));
                    }

                    BasicBlock* next_bb = inst->branch_target().block;
                    if (!next_bb) {
                        throw InterpreterException("Branch to null basic block");
                    }

                    for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                        frame.set_value(next_bb->param(i), target_args[i]);
                    }

                    cur_bb = next_bb;
                    transitioned = true;
                    break;
                }

                case Opcode::br_if: {
                    RuntimeValue cond = frame.get_value(inst->operand(0));
                    const BranchTarget& target = (cond.as_i32() != 0) ? inst->true_target() : inst->false_target();

                    std::vector<RuntimeValue> target_args;
                    target_args.reserve(target.args.size());
                    for (Value* a : target.args) {
                        target_args.push_back(frame.get_value(a));
                    }

                    BasicBlock* next_bb = target.block;
                    if (!next_bb) {
                        throw InterpreterException("Conditional branch to null basic block");
                    }

                    for (size_t i = 0; i < next_bb->param_count() && i < target_args.size(); ++i) {
                        frame.set_value(next_bb->param(i), target_args[i]);
                    }

                    cur_bb = next_bb;
                    transitioned = true;
                    break;
                }

                case Opcode::ret: {
                    if (inst->operand_count() > 0 && inst->operand(0) != nullptr) {
                        return frame.get_value(inst->operand(0));
                    }
                    return RuntimeValue::from_void();
                }

                case Opcode::unreachable: {
                    throw InterpreterException("Execution reached unreachable instruction");
                }
            }

            if (transitioned) {
                break;
            }
        }

        if (!transitioned) {
            // Block ended without a terminator (malformed MIR or fallen off end)
            break;
        }
    }

    return RuntimeValue::from_void();
}

} // namespace brass
