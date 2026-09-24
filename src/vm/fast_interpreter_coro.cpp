#include "fast_interpreter_impl.hpp"
#include <brass/runtime/coroutine.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/gc/native_frames.hpp>
#include <algorithm>
#include <cstring>

namespace brass {

namespace {

// The unlowered path hands arguments over as 8-byte register words; a
// vector argument does not fit one, so it is an error rather than a
// truncation. (Lowered bodies take vectors: their frame slots are sized.)
std::vector<uint64_t> coro_raw_args(const std::vector<RuntimeValue>& args) {
    std::vector<uint64_t> raw_args;
    raw_args.reserve(args.size());
    for (const auto& a : args) {
        if (a.is_vector()) {
            throw InterpreterException("FastInterpreter coro_create: vector arguments to a coroutine body "
                                       "not lowered by CoroTransformPass are not supported");
        }
        raw_args.push_back(a.raw_bits());
    }
    return raw_args;
}

} // namespace

uintptr_t FastInterpreter::coro_create_lowered(const Function& fn, const std::vector<RuntimeValue>& args) {
    const CoroFrameLayout layout = compute_coro_frame_layout(fn);
    uint32_t arg_slots = 0;
    for (const RuntimeValue& a : args) arg_slots += coro_slot_count(a.type());
    const uint32_t slot_count = std::max(layout.slot_count, arg_slots);

    // No generated frame: this interpreter's roots reach the heap through
    // its scope and provider.
    // Its body is MIR (CORO_FLAG_MIR_BODY), which every tier can resume.
    const uintptr_t frame_addr = runtime::create_mir_coro_frame(fn, slot_count, layout.pointer_mask);
    auto* cf = reinterpret_cast<runtime::BrassCoroFrame*>(frame_addr);
    if (!cf) {
        throw InterpreterException("coro_create: frame allocation failed");
    }
    // Arguments fill consecutive slots, a vector spanning coro_slot_count of
    // them, where the lowered body loads each.
    uint32_t slot = 0;
    for (const RuntimeValue& a : args) {
        if (a.is_vector()) {
            std::memcpy(&cf->slots[slot], a.vec_bytes(), a.is_v256() ? 32 : 16);
        } else {
            cf->slots[slot] = static_cast<uint64_t>(a.raw_bits());
        }
        slot += coro_slot_count(a.type());
    }
    return frame_addr;
}

const Function* FastInterpreter::lowered_coro_body(uintptr_t handle) const {
    const Function* body = runtime::mir_coro_body(reinterpret_cast<const runtime::BrassCoroFrame*>(handle));    if (body && !is_lowered_coro_body(*body)) {
        throw InterpreterException("coro_resume: coroutine body " + std::string(body->name()) +
                                   " has not been lowered by CoroTransformPass");
    }
    return body;
}

uint64_t FastInterpreter::coro_resume_lowered(uintptr_t handle, uint64_t input_val) {
    auto* cf = reinterpret_cast<runtime::BrassCoroFrame*>(handle);
    if (cf->is_done) {
        return cf->yielded_val;
    }
    // A MIR body (an interpreter created the frame, in any module) runs
    // here, in its own module (call_from_native); generated code's natively.
    const Function* body = lowered_coro_body(handle);
    if (!body) {
        if (!cf->fn_ptr) {
            throw InterpreterException("coro_resume: coroutine frame has no body");
        }
        // Generated code's frame: run it natively.
        return brass_coro_resume(handle, input_val);
    }
    cf->resume_arg = input_val;
    // The body dispatches on state_id and maintains is_done itself. It may
    // collect and move the frame; `handle` is a root so the writes below
    // reach the live copy.
    ThreadRootsScope frame_root([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &handle);
    const RuntimeValue r = call_from_native(*body, {RuntimeValue::from_ptr(handle)});
    cf = reinterpret_cast<runtime::BrassCoroFrame*>(handle);
    cf->yielded_val = static_cast<uint64_t>(r.raw_bits());
    if (cf->is_done) {
        runtime::unregister_active_coro_frame(cf);
    }
    return cf->yielded_val;
}

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

    // No generated frame: this interpreter's roots reach the heap through
    // its scope and provider.
    uintptr_t c_frame_addr = brass_coro_create_at(nullptr, std::max<uint32_t>(num_regs, 16), 0, 0, 0);
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
        use_module(fn.parent());
    }
    if (is_lowered_coro_body(fn)) {
        return coro_create_lowered(fn, args);
    }
    const BytecodeFunction* bfn = get_or_compile(fn);
    return coro_create(bfn, coro_raw_args(args));
}

uintptr_t FastInterpreter::coro_create(const Module& mod, std::string_view callee, const std::vector<RuntimeValue>& args) {
    use_module(&mod);
    const Function* fn = mod.get_function(callee);
    if (!fn) {
        throw InterpreterException("coro_create: callee function not found in module: " + std::string(callee));
    }
    return coro_create(*fn, args);
}

uintptr_t FastInterpreter::coro_create(std::string_view callee, const std::vector<RuntimeValue>& args) {
    if (module_) {
        const Function* fn = module_->get_function(callee);
        if (fn && is_lowered_coro_body(*fn)) {
            return coro_create_lowered(*fn, args);
        }
    }
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
    return coro_create(bfn, coro_raw_args(args));
}

void FastInterpreter::coro_suspend(FastFrame& frame, uint32_t dst_reg, uint32_t yield_reg, uint32_t resume_id) {
    FastCoroState* coro = active_coro_frame_;
    if (!coro) {
        throw InterpreterException("coro_suspend called without active coroutine state");
    }

    uint64_t yield_val = (yield_reg < frame.num_registers) ? frame.registers[yield_reg] : 0;
    coro->yielded_val = yield_val;
    coro->state_id = (resume_id != 0) ? resume_id : (coro->state_id + 1);
    coro->resume_dst_reg = dst_reg < frame.num_registers ? static_cast<BcReg>(dst_reg) : kNoReg;
    // coro_suspend is two code words (the second is the resume id).
    coro->pc = frame.pc + static_cast<uint32_t>(bytecode_inst_words(BytecodeOp::coro_suspend));

    coro->registers.assign(frame.registers, frame.registers + frame.num_registers);
    if (frame.vector_regs) {
        coro->vector_storage.assign(frame.vector_regs,
                                    frame.vector_regs + static_cast<size_t>(frame.num_registers) * kFastVecBytes);
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
        if (handle == 0) {
            throw InterpreterException("coro_resume of a null coroutine frame");
        }
        return coro_resume_lowered(handle, input_val);
    }

    FastCoroState* coro = it->second.get();
    if (!coro || coro->is_done) {
        return coro ? coro->yielded_val : 0;
    }

    if (coro->state_id != 0 && coro->resume_dst_reg != kNoReg && coro->resume_dst_reg < coro->registers.size()) {
        coro->registers[coro->resume_dst_reg] = input_val;
    }
    coro->resume_arg = input_val;
    if (coro->c_frame) {
        coro->c_frame->resume_arg = input_val;
    }

    FastFrame coro_frame;
    coro_frame.bfn = coro->bfn;
    coro_frame.info = &fn_info(*coro->bfn, coro->mir_fn);
    coro_frame.mir_fn = coro_frame.info->mir_fn;
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
    // A lowered body's result type, read before the resume (which may move
    // the frame).
    const Function* body = nullptr;
    if (handle != 0 && active_coros_.find(handle) == active_coros_.end()) {
        body = lowered_coro_body(handle);
    }
    uint64_t ret = coro_resume(handle, input_val.raw_bits());
    if (body) {
        return RuntimeValue::from_bits(body->return_type(), ret);
    }
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
        return;
    }
    // A lowered coroutine's frame (or generated code's): done, and no
    // longer a root.
    brass_coro_destroy(handle);
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
