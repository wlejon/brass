#include "fast_interpreter_impl.hpp"
#include <brass/runtime/code_installer.hpp>
#include <iostream>

namespace brass {

void FastInterpreter::register_external_function(std::string_view name, FastHostFn fn) {
    external_functions_[std::string(name)] = std::move(fn);
}

void FastInterpreter::register_external_function(std::string_view name, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn) {
    external_functions_[std::string(name)] = [f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        return f(args);
    };
}

void FastInterpreter::register_external_function(std::string_view name, brass::HostFn fn) {
    external_functions_[std::string(name)] = [f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        Interpreter dummy(1024 * 1024);
        return f(dummy, args);
    };
}

bool FastInterpreter::has_external_function(std::string_view name) const noexcept {
    return external_functions_.find(std::string(name)) != external_functions_.end();
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, const Function* fn) {
    function_pointers_[ptr] = fn;
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, const BytecodeFunction* bfn) {
    bytecode_function_pointers_[ptr] = bfn;
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, FastHostFn fn) {
    host_function_pointers_[ptr] = std::move(fn);
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn) {
    host_function_pointers_[ptr] = [f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        return f(args);
    };
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, brass::HostFn fn) {
    host_function_pointers_[ptr] = [f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        Interpreter dummy(1024 * 1024);
        return f(dummy, args);
    };
}

const Function* FastInterpreter::find_function_by_pointer(uintptr_t ptr) const noexcept {
    auto it = function_pointers_.find(ptr);
    return (it != function_pointers_.end()) ? it->second : nullptr;
}

const BytecodeFunction* FastInterpreter::find_bytecode_function_by_pointer(uintptr_t ptr) const noexcept {
    auto it = bytecode_function_pointers_.find(ptr);
    return (it != bytecode_function_pointers_.end()) ? it->second : nullptr;
}

void FastInterpreter::patch_const(std::string_view symbol, int64_t val) {
    patched_consts_[std::string(symbol)] = val;
}

int64_t FastInterpreter::get_patched_const(std::string_view symbol, int64_t default_val) const {
    auto it = patched_consts_.find(std::string(symbol));
    return (it != patched_consts_.end()) ? it->second : default_val;
}

void FastInterpreter::patch_call(std::string_view site, std::string_view target) {
    patched_calls_[std::string(site)] = std::string(target);
}

std::string_view FastInterpreter::get_patched_call(std::string_view site, std::string_view default_callee) const {
    auto it = patched_calls_.find(std::string(site));
    return (it != patched_calls_.end()) ? std::string_view(it->second) : default_callee;
}

RuntimeValue FastInterpreter::execute_call(FastFrame& frame, const CallSiteInfo& cs, BytecodeOp call_op) {
    std::string_view callee_name = cs.callee;
    if (call_op == BytecodeOp::patchable_call) {
        std::string_view default_callee = cs.extra_symbol.empty() ? cs.callee : cs.extra_symbol;
        callee_name = get_patched_call(cs.callee, default_callee);
    }

    // 0. Check Tiering & Native entry in FunctionDispatchTable
    auto* handle = runtime::FunctionDispatchTable::instance().find(callee_name);
    if (handle && handle->has_native_entry()) {
        const Function* mir_fn = handle->mir_function();
        std::vector<RuntimeValue> call_args;
        call_args.reserve(cs.arg_regs.size());
        for (size_t i = 0; i < cs.arg_regs.size(); ++i) {
            uint8_t src_r = cs.arg_regs[i];
            uint64_t bits = frame.registers[src_r];
            Type arg_ty = (mir_fn && i < mir_fn->param_count()) ? mir_fn->param_type(i) : Type::i64();
            call_args.push_back(RuntimeValue::from_bits(arg_ty, bits));
        }
        RuntimeValue ret_val = handle->call_native(call_args);
        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    // 1. Check if it's a known bytecode function or can be compiled from MIR
    const BytecodeFunction* target_bfn = nullptr;
    if (bytecode_module_) {
        target_bfn = bytecode_module_->get_function(callee_name);
    }
    if (!target_bfn && module_) {
        const Function* mir_fn = module_->get_function(callee_name);
        if (mir_fn) {
            target_bfn = get_or_compile(*mir_fn);
        }
    }

    if (target_bfn) {
        auto& feedback = runtime::TieringRegistry::instance().get_or_create(callee_name);
        feedback.record_invocation();

        if (handle && handle->has_native_entry()) {
            const Function* mir_fn = handle->mir_function();
            std::vector<RuntimeValue> call_args;
            call_args.reserve(cs.arg_regs.size());
            for (size_t i = 0; i < cs.arg_regs.size(); ++i) {
                uint8_t src_r = cs.arg_regs[i];
                uint64_t bits = frame.registers[src_r];
                Type arg_ty = (mir_fn && i < mir_fn->param_count()) ? mir_fn->param_type(i) : Type::i64();
                call_args.push_back(RuntimeValue::from_bits(arg_ty, bits));
            }
            RuntimeValue ret_val = handle->call_native(call_args);
            if (cs.dst_reg != 255) {
                frame.registers[cs.dst_reg] = ret_val.raw_bits();
                if (ret_val.is_vector()) {
                    uint8_t* vregs = frame.ensure_vector_regs();
                    std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
                }
            }
            return ret_val;
        }

        if (call_depth_ >= max_call_depth_) {
            throw InterpreterException("Call stack depth limit exceeded (" + std::to_string(max_call_depth_) + ")");
        }

        uint32_t num_regs = std::max<uint32_t>(target_bfn->num_registers, 1);
        uint64_t* callee_regs = static_cast<uint64_t*>(BRASS_ALLOCA(num_regs * sizeof(uint64_t)));
        std::memset(callee_regs, 0, num_regs * sizeof(uint64_t));

        FastFrame callee_frame;
        callee_frame.bfn = target_bfn;
        callee_frame.registers = callee_regs;
        callee_frame.num_registers = num_regs;

        for (size_t i = 0; i < cs.arg_regs.size() && i < num_regs; ++i) {
            uint8_t src_r = cs.arg_regs[i];
            callee_regs[i] = frame.registers[src_r];
            if (frame.vector_regs) {
                uint8_t* callee_vregs = callee_frame.ensure_vector_regs();
                std::memcpy(callee_vregs + i * 32, frame.vector_regs + src_r * 32, 32);
            }
        }

        FrameGuard guard(*this, callee_frame);
        RuntimeValue ret_val = execute_frame(callee_frame);

        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    // 2. Check external host functions
    auto it = external_functions_.find(std::string(callee_name));
    if (it != external_functions_.end()) {
        const Function* mir_fn = module_ ? module_->get_function(callee_name) : nullptr;
        std::vector<RuntimeValue> call_args;
        call_args.reserve(cs.arg_regs.size());
        for (size_t i = 0; i < cs.arg_regs.size(); ++i) {
            uint8_t src_r = cs.arg_regs[i];
            uint64_t bits = frame.registers[src_r];
            Type arg_ty = Type::i64();
            if (mir_fn && i < mir_fn->param_count()) {
                arg_ty = mir_fn->param_type(i);
            } else if (callee_name == "sqrt" || callee_name == "fabs" || callee_name == "floor" || callee_name == "ceil" || callee_name == "bronze_print_f64") {
                arg_ty = Type::f64();
            }
            call_args.push_back(RuntimeValue::from_bits(arg_ty, bits));
        }

        RuntimeValue ret_val = it->second(*this, call_args);
        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    throw InterpreterException("Call to undefined function: " + std::string(callee_name));
}

RuntimeValue FastInterpreter::execute_call_indirect(FastFrame& frame, const CallSiteInfo& cs) {
    uintptr_t ptr = static_cast<uintptr_t>(frame.registers[cs.callee_reg]);

    // 1. Check bytecode function pointers
    auto bfn_it = bytecode_function_pointers_.find(ptr);
    if (bfn_it != bytecode_function_pointers_.end()) {
        const BytecodeFunction* target_bfn = bfn_it->second;
        if (call_depth_ >= max_call_depth_) {
            throw InterpreterException("Call stack depth limit exceeded (" + std::to_string(max_call_depth_) + ")");
        }
        uint32_t num_regs = std::max<uint32_t>(target_bfn->num_registers, 1);
        uint64_t* callee_regs = static_cast<uint64_t*>(BRASS_ALLOCA(num_regs * sizeof(uint64_t)));
        std::memset(callee_regs, 0, num_regs * sizeof(uint64_t));

        FastFrame callee_frame;
        callee_frame.bfn = target_bfn;
        callee_frame.registers = callee_regs;
        callee_frame.num_registers = num_regs;

        for (size_t i = 0; i < cs.arg_regs.size() && i < num_regs; ++i) {
            uint8_t src_r = cs.arg_regs[i];
            callee_regs[i] = frame.registers[src_r];
            if (frame.vector_regs) {
                uint8_t* callee_vregs = callee_frame.ensure_vector_regs();
                std::memcpy(callee_vregs + i * 32, frame.vector_regs + src_r * 32, 32);
            }
        }

        FrameGuard guard(*this, callee_frame);
        RuntimeValue ret_val = execute_frame(callee_frame);
        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    // 2. Check MIR function pointers
    auto fn_it = function_pointers_.find(ptr);
    if (fn_it != function_pointers_.end()) {
        const BytecodeFunction* target_bfn = get_or_compile(*fn_it->second);
        if (call_depth_ >= max_call_depth_) {
            throw InterpreterException("Call stack depth limit exceeded (" + std::to_string(max_call_depth_) + ")");
        }
        uint32_t num_regs = std::max<uint32_t>(target_bfn->num_registers, 1);
        uint64_t* callee_regs = static_cast<uint64_t*>(BRASS_ALLOCA(num_regs * sizeof(uint64_t)));
        std::memset(callee_regs, 0, num_regs * sizeof(uint64_t));

        FastFrame callee_frame;
        callee_frame.bfn = target_bfn;
        callee_frame.registers = callee_regs;
        callee_frame.num_registers = num_regs;

        for (size_t i = 0; i < cs.arg_regs.size() && i < num_regs; ++i) {
            uint8_t src_r = cs.arg_regs[i];
            callee_regs[i] = frame.registers[src_r];
            if (frame.vector_regs) {
                uint8_t* callee_vregs = callee_frame.ensure_vector_regs();
                std::memcpy(callee_vregs + i * 32, frame.vector_regs + src_r * 32, 32);
            }
        }

        FrameGuard guard(*this, callee_frame);
        RuntimeValue ret_val = execute_frame(callee_frame);
        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    // 3. Check host function pointers
    auto host_it = host_function_pointers_.find(ptr);
    if (host_it != host_function_pointers_.end()) {
        std::vector<RuntimeValue> call_args;
        call_args.reserve(cs.arg_regs.size());
        for (uint8_t src_r : cs.arg_regs) {
            call_args.push_back(RuntimeValue::from_u64(frame.registers[src_r]));
        }
        RuntimeValue ret_val = host_it->second(*this, call_args);
        if (cs.dst_reg != 255) {
            frame.registers[cs.dst_reg] = ret_val.raw_bits();
            if (ret_val.is_vector()) {
                uint8_t* vregs = frame.ensure_vector_regs();
                std::memcpy(vregs + cs.dst_reg * 32, ret_val.vec_bytes(), 32);
            }
        }
        return ret_val;
    }

    throw InterpreterException("Call indirect to unregistered target pointer: " + std::to_string(ptr));
}

RuntimeValue FastInterpreter::run(const Function& fn) {
    return run(fn, {});
}

RuntimeValue FastInterpreter::run(const Function& fn, const std::vector<RuntimeValue>& args) {
    if (fn.parent()) {
        module_ = fn.parent();
        if (!runtime::TieringRegistry::instance().active_module()) {
            runtime::TieringRegistry::instance().set_active_module(fn.parent());
        }
    }

    auto* handle = runtime::FunctionDispatchTable::instance().find(fn.name());
    if (handle && handle->mir_function() == &fn) {
        void* native_code = handle->native_entry();
        if (native_code != nullptr) {
            return handle->call_native(args);
        }
    }

    auto& feedback = runtime::TieringRegistry::instance().get_or_create(fn.name());
    feedback.record_invocation();

    if (handle && handle->mir_function() == &fn) {
        void* native_code = handle->native_entry();
        if (native_code != nullptr) {
            return handle->call_native(args);
        }
    }

    const BytecodeFunction* bfn = get_or_compile(fn);
    return run(*bfn, args);
}

RuntimeValue FastInterpreter::run(const BytecodeFunction& fn) {
    return run(fn, {});
}

RuntimeValue FastInterpreter::run(const BytecodeFunction& fn, const std::vector<RuntimeValue>& args) {
    uint32_t num_regs = std::max<uint32_t>(fn.num_registers, 1);
    uint64_t* registers = static_cast<uint64_t*>(BRASS_ALLOCA(num_regs * sizeof(uint64_t)));
    std::memset(registers, 0, num_regs * sizeof(uint64_t));

    FastFrame frame;
    frame.bfn = &fn;
    frame.mir_fn = module_ ? module_->get_function(fn.name) : nullptr;
    frame.registers = registers;
    frame.num_registers = num_regs;

    for (size_t i = 0; i < args.size() && i < num_regs; ++i) {
        registers[i] = args[i].raw_bits();
        if (args[i].is_vector()) {
            uint8_t* vregs = frame.ensure_vector_regs();
            std::memcpy(vregs + i * 32, args[i].vec_bytes(), 32);
        }
    }

    FrameGuard guard(*this, frame);
    return execute_frame(frame);
}

RuntimeValue FastInterpreter::run(std::string_view fn_name) {
    return run(fn_name, {});
}

RuntimeValue FastInterpreter::run(std::string_view fn_name, const std::vector<RuntimeValue>& args) {
    if (bytecode_module_) {
        const BytecodeFunction* bfn = bytecode_module_->get_function(fn_name);
        if (bfn) return run(*bfn, args);
    }
    if (module_) {
        const Function* fn = module_->get_function(fn_name);
        if (fn) return run(*fn, args);
    }
    throw InterpreterException("Function @" + std::string(fn_name) + " not found");
}

RuntimeValue FastInterpreter::run(const Module& mod, std::string_view entry_name) {
    return run(mod, entry_name, {});
}

RuntimeValue FastInterpreter::run(const Module& mod, std::string_view entry_name, const std::vector<RuntimeValue>& args) {
    module_ = &mod;
    const Function* fn = mod.get_function(entry_name);
    if (!fn) {
        throw InterpreterException("Entry function @" + std::string(entry_name) + " not found in module");
    }
    return run(*fn, args);
}

} // namespace brass
