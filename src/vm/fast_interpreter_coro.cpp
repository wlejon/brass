#include "fast_interpreter_impl.hpp"
#include <brass/runtime/coroutine.hpp>
#include <algorithm>

namespace brass {

uintptr_t FastInterpreter::coro_create(const BytecodeFunction* bfn, const std::vector<uint64_t>& args) {
    if (!bfn) return 0;

    auto coro = std::make_unique<FastCoroState>();
    coro->bfn = bfn;
    coro->state_id = 0;
    coro->is_done = false;
    coro->pc = 0;
    uint32_t num_regs = std::max<uint32_t>(bfn->num_registers, 1);
    coro->registers.assign(num_regs, 0);

    for (size_t i = 0; i < args.size() && i < num_regs; ++i) {
        coro->registers[i] = args[i];
    }

    uintptr_t c_frame_addr = brass_coro_create(nullptr, std::max<uint32_t>(num_regs, 16), 0);
    auto* cf = reinterpret_cast<runtime::BrassCoroFrame*>(c_frame_addr);
    if (cf) {
        cf->state_id = 0;
        cf->is_done = 0;
        cf->yielded_val = 0;
        cf->resume_arg = 0;
        for (size_t i = 0; i < args.size() && i < 16; ++i) {
            cf->slots[i] = args[i];
        }
    }
    coro->c_frame = cf;

    uintptr_t handle = c_frame_addr ? c_frame_addr : reinterpret_cast<uintptr_t>(coro.get());
    active_coros_[handle] = std::move(coro);
    return handle;
}

uintptr_t FastInterpreter::coro_create(const Function& fn, const std::vector<RuntimeValue>& args) {
    if (fn.parent()) {
        module_ = fn.parent();
    }
    const BytecodeFunction* bfn = get_or_compile(fn);
    std::vector<uint64_t> raw_args;
    raw_args.reserve(args.size());
    for (const auto& a : args) {
        raw_args.push_back(a.raw_bits());
    }
    return coro_create(bfn, raw_args);
}

uintptr_t FastInterpreter::coro_create(const Module& mod, std::string_view callee, const std::vector<RuntimeValue>& args) {
    module_ = &mod;
    const Function* fn = mod.get_function(callee);
    if (!fn) {
        throw InterpreterException("coro_create: callee function not found in module: " + std::string(callee));
    }
    return coro_create(*fn, args);
}

uintptr_t FastInterpreter::coro_create(std::string_view callee, const std::vector<RuntimeValue>& args) {
    const BytecodeFunction* bfn = nullptr;
    if (bytecode_module_) {
        bfn = bytecode_module_->get_function(callee);
    }
    if (!bfn && module_) {
        const Function* fn = module_->get_function(callee);
        if (fn) {
            bfn = get_or_compile(*fn);
        }
    }
    if (!bfn) {
        throw InterpreterException("coro_create: callee function not found: " + std::string(callee));
    }

    std::vector<uint64_t> raw_args;
    raw_args.reserve(args.size());
    for (const auto& a : args) {
        raw_args.push_back(a.raw_bits());
    }
    return coro_create(bfn, raw_args);
}

void FastInterpreter::coro_suspend(FastFrame& frame, uint8_t dst_reg, uint8_t yield_reg, uint32_t resume_id) {
    FastCoroState* coro = active_coro_frame_;
    if (!coro) {
        throw InterpreterException("coro_suspend called without active coroutine state");
    }

    uint64_t yield_val = (yield_reg < frame.num_registers) ? frame.registers[yield_reg] : 0;
    coro->yielded_val = yield_val;
    coro->state_id = (resume_id != 0) ? resume_id : (coro->state_id + 1);
    coro->resume_dst_reg = dst_reg;
    coro->pc = frame.pc + 1;

    coro->registers.assign(frame.registers, frame.registers + frame.num_registers);
    if (frame.vector_regs) {
        coro->vector_storage = frame.vector_storage;
    }

    if (coro->c_frame) {
        coro->c_frame->state_id = coro->state_id;
        coro->c_frame->yielded_val = yield_val;
        coro->c_frame->is_done = 0;
    }

    throw FastCoroSuspendException(yield_val, coro->state_id);
}

uint64_t FastInterpreter::coro_resume(uintptr_t handle, uint64_t input_val) {
    auto it = active_coros_.find(handle);
    if (it == active_coros_.end()) {
        return 0;
    }

    FastCoroState* coro = it->second.get();
    if (!coro || coro->is_done) {
        return coro ? coro->yielded_val : 0;
    }

    if (coro->state_id != 0 && coro->resume_dst_reg < coro->registers.size()) {
        coro->registers[coro->resume_dst_reg] = input_val;
    }
    coro->resume_arg = input_val;
    if (coro->c_frame) {
        coro->c_frame->resume_arg = input_val;
    }

    FastFrame coro_frame;
    coro_frame.bfn = coro->bfn;
    coro_frame.registers = coro->registers.data();
    coro_frame.num_registers = static_cast<uint32_t>(coro->registers.size());
    coro_frame.pc = coro->pc;
    if (!coro->vector_storage.empty()) {
        coro_frame.vector_storage = coro->vector_storage;
        coro_frame.vector_regs = coro_frame.vector_storage.data();
    }

    FastCoroState* prev_active = active_coro_frame_;
    active_coro_frame_ = coro;

    FrameGuard guard(*this, coro_frame);
    try {
        RuntimeValue ret_val = execute_frame(coro_frame);
        active_coro_frame_ = prev_active;
        coro->is_done = true;
        coro->yielded_val = ret_val.raw_bits();
        if (coro->c_frame) {
            coro->c_frame->is_done = 1;
            coro->c_frame->yielded_val = coro->yielded_val;
        }
        return coro->yielded_val;
    } catch (const FastCoroSuspendException& se) {
        active_coro_frame_ = prev_active;
        coro->state_id = se.state_id != 0 ? se.state_id : 1;
        coro->yielded_val = se.yielded_val;
        if (coro->c_frame) {
            coro->c_frame->state_id = coro->state_id;
            coro->c_frame->yielded_val = se.yielded_val;
            coro->c_frame->is_done = 0;
        }
        return coro->yielded_val;
    } catch (...) {
        active_coro_frame_ = prev_active;
        throw;
    }
}

RuntimeValue FastInterpreter::coro_resume_val(uintptr_t handle, RuntimeValue input_val) {
    uint64_t ret = coro_resume(handle, input_val.raw_bits());
    auto it = active_coros_.find(handle);
    if (it != active_coros_.end() && it->second && it->second->bfn) {
        return RuntimeValue::from_bits(it->second->bfn->return_type, ret);
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(ret));
}

void FastInterpreter::coro_destroy(uintptr_t handle) {
    auto it = active_coros_.find(handle);
    if (it != active_coros_.end()) {
        it->second->is_done = true;
        if (it->second->c_frame) {
            brass_coro_destroy(reinterpret_cast<uintptr_t>(it->second->c_frame));
        }
        active_coros_.erase(it);
    }
}

bool FastInterpreter::coro_is_done(uintptr_t handle) const {
    auto it = active_coros_.find(handle);
    if (it != active_coros_.end() && it->second) {
        return it->second->is_done;
    }
    return brass_coro_is_done(handle) != 0;
}

FastCoroState* FastInterpreter::get_coro_state(uintptr_t handle) {
    auto it = active_coros_.find(handle);
    return (it != active_coros_.end()) ? it->second.get() : nullptr;
}

} // namespace brass
