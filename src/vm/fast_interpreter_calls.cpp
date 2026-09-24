// FastInterpreter calls: host / symbol / function-pointer registration,
// call-site resolution (cached per site, re-resolved when the module,
// symbols, patches or runtime registries change), the call paths to
// bytecode, native and host callees, and the run / resume entry points.

#include "fast_interpreter_impl.hpp"
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>

namespace brass {

// One reusable argument vector per nesting level of host / native calls.
struct ArgBufferScope {
    FastInterpreter& interp;
    std::vector<RuntimeValue>* buf = nullptr;

    explicit ArgBufferScope(FastInterpreter& in) : interp(in) {
        const size_t depth = interp.arg_buffer_depth_;
        if (depth >= interp.arg_buffers_.size()) {
            interp.arg_buffers_.push_back(std::make_unique<std::vector<RuntimeValue>>());
        }
        buf = interp.arg_buffers_[depth].get();
        buf->clear();
        ++interp.arg_buffer_depth_;
    }
    ~ArgBufferScope() { --interp.arg_buffer_depth_; }
    ArgBufferScope(const ArgBufferScope&) = delete;
    ArgBufferScope& operator=(const ArgBufferScope&) = delete;
};

void* FastAllocaArena::allocate_slow(size_t size, size_t align) {
    const size_t need = size + align;
    size_t next = chunks_.empty() ? 0 : cur_ + 1;
    if (next < chunks_.size() && chunks_[next].size < need) {
        // Chunks past the current one are unused (marks are LIFO), so a
        // too-small one can be replaced.
        chunks_[next] = Chunk{};
    }
    if (next >= chunks_.size() || !chunks_[next].data) {
        Chunk c;
        c.size = std::max(kChunkSize, need);
        c.data = std::make_unique<uint8_t[]>(c.size);
        if (next < chunks_.size()) {
            chunks_[next] = std::move(c);
        } else {
            chunks_.push_back(std::move(c));
        }
    }
    cur_ = next;
    off_ = 0;
    return allocate(size, align);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void FastInterpreter::register_external_function(std::string_view name, FastHostFn fn) {
    external_functions_[std::string(name)] = std::move(fn);
    invalidate_call_caches();
}

void FastInterpreter::register_external_function(std::string_view name, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn) {
    register_external_function(name, FastHostFn([f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        return f(args);
    }));
}

void FastInterpreter::register_external_function(std::string_view name, brass::HostFn fn) {
    register_external_function(name, FastHostFn([f = std::move(fn)](FastInterpreter& self, const std::vector<RuntimeValue>& args) {
        return f(self.host_adapter_interpreter(), args);
    }));
}

Interpreter& FastInterpreter::host_adapter_interpreter() {
    // brass::HostFn callbacks take an Interpreter; one is built on first
    // use and shared by every such callback.
    if (!host_adapter_) host_adapter_ = std::make_unique<Interpreter>(1024 * 1024);
    return *host_adapter_;
}

bool FastInterpreter::has_external_function(std::string_view name) const noexcept {
    return external_functions_.find(std::string(name)) != external_functions_.end();
}

void FastInterpreter::register_external_symbol(std::string_view name, void* addr) {
    external_symbols_[std::string(name)] = addr;
}

void* FastInterpreter::find_external_symbol(std::string_view name) const noexcept {
    auto it = external_symbols_.find(std::string(name));
    return it != external_symbols_.end() ? it->second : nullptr;
}

bool FastInterpreter::has_external_symbol(std::string_view name) const noexcept {
    return external_symbols_.find(std::string(name)) != external_symbols_.end();
}

void FastInterpreter::retire_caches() {
    for (auto& [fn, bfn] : compiled_functions_) retired_bytecode_.push_back(std::move(bfn));
    compiled_functions_.clear();
    for (auto& [bfn, info] : fn_infos_) retired_infos_.push_back(std::move(info));
    fn_infos_.clear();
    invalidate_call_caches();
}

void FastInterpreter::release_retired() noexcept {
    // Running frames and suspended coroutines may point into retired code.
    if (call_depth_ != 0 || !active_coros_.empty()) return;
    retired_bytecode_.clear();
    retired_infos_.clear();
}

void FastInterpreter::use_module(const Module* mod) {
    if (mod == module_) return;
    if (module_ != nullptr) retire_caches();
    module_ = mod;
    invalidate_call_caches();
}

void FastInterpreter::set_module(const Module* mod) {
    use_module(mod);
    if (module_) {
        for (const auto* fn : module_->functions()) {
            if (fn) register_function_pointer(reinterpret_cast<uintptr_t>(fn), fn);
        }
    }
}

void FastInterpreter::set_bytecode_module(const BytecodeModule* bmod) {
    if (bmod != bytecode_module_) {
        retire_caches();
        bytecode_module_ = bmod;
    }
    if (bytecode_module_) {
        for (const auto& fn : bytecode_module_->functions()) {
            if (fn) register_function_pointer(reinterpret_cast<uintptr_t>(fn.get()), fn.get());
        }
    }
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, const Function* fn) {
    auto [it, inserted] = function_pointers_.try_emplace(ptr, fn);
    if (!inserted) {
        if (it->second == fn) return;
        it->second = fn;
    }
    invalidate_call_caches();
}

uintptr_t FastInterpreter::function_address(const Function& fn) {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(&fn);
    if (fn.block_count() > 0) {
        ptr = reinterpret_cast<uintptr_t>(dispatch_table().pipeline().function_address(fn.name(), &fn));
    }
    register_function_pointer(ptr, &fn);
    return ptr;
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, const BytecodeFunction* bfn) {
    auto [it, inserted] = bytecode_function_pointers_.try_emplace(ptr, bfn);
    if (!inserted) {
        if (it->second == bfn) return;
        it->second = bfn;
    }
    invalidate_call_caches();
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, FastHostFn fn) {
    host_function_pointers_[ptr] = std::move(fn);
    invalidate_call_caches();
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn) {
    register_function_pointer(ptr, FastHostFn([f = std::move(fn)](FastInterpreter&, const std::vector<RuntimeValue>& args) {
        return f(args);
    }));
}

void FastInterpreter::register_function_pointer(uintptr_t ptr, brass::HostFn fn) {
    register_function_pointer(ptr, FastHostFn([f = std::move(fn)](FastInterpreter& self, const std::vector<RuntimeValue>& args) {
        return f(self.host_adapter_interpreter(), args);
    }));
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
    if (patched_consts_.empty()) return default_val;
    auto it = patched_consts_.find(std::string(symbol));
    return (it != patched_consts_.end()) ? it->second : default_val;
}

void FastInterpreter::patch_call(std::string_view site, std::string_view target) {
    patched_calls_[std::string(site)] = std::string(target);
    invalidate_call_caches();
}

std::string_view FastInterpreter::get_patched_call(std::string_view site, std::string_view default_callee) const {
    auto it = patched_calls_.find(std::string(site));
    return (it != patched_calls_.end()) ? std::string_view(it->second) : default_callee;
}

// ---------------------------------------------------------------------------
// Call-site resolution
// ---------------------------------------------------------------------------

FastFnInfo& FastInterpreter::fn_info(const BytecodeFunction& bfn, const Function* mir_fn) {
    auto it = fn_infos_.find(&bfn);
    if (it != fn_infos_.end()) {
        if (mir_fn && !it->second->mir_fn) it->second->mir_fn = mir_fn;
        return *it->second;
    }
    auto info = std::make_unique<FastFnInfo>();
    info->bfn = &bfn;
    info->mir_fn = mir_fn ? mir_fn : (module_ ? module_->get_function(bfn.name) : nullptr);
    info->num_regs = std::max<uint32_t>(bfn.num_registers, 1);
    for (uint32_t i = 0; i < bfn.num_params && i < bfn.register_types.size(); ++i) {
        if (bfn.register_types[i].is_vector()) info->has_vector_params = true;
    }
    info->calls.resize(bfn.call_sites.size());
    FastFnInfo& ref = *info;
    fn_infos_.emplace(&bfn, std::move(info));
    return ref;
}

void FastInterpreter::set_dispatch_table(runtime::FunctionDispatchTable* table) {
    dispatch_table_ = table;
    invalidate_call_caches();
}

runtime::FunctionDispatchTable& FastInterpreter::dispatch_table() const noexcept {
    return dispatch_table_ ? *dispatch_table_ : runtime::FunctionDispatchTable::instance();
}

void FastInterpreter::resolve_call_target(FastFnInfo& info, uint32_t cs_idx) {
    const uint64_t gen = runtime::registry_generation();
    const CallSiteInfo& cs = info.bfn->call_sites[cs_idx];
    std::string_view name = cs.callee;
    if (cs.patchable) {
        name = get_patched_call(cs.callee, cs.extra_symbol.empty() ? std::string_view(cs.callee)
                                                                   : std::string_view(cs.extra_symbol));
    }

    FastCallTarget t;
    t.name = std::string(name);
    const BytecodeFunction* bfn = bytecode_module_ ? bytecode_module_->get_function(name) : nullptr;
    const Function* mir = module_ ? module_->get_function(name) : nullptr;
    if (!bfn && mir) bfn = get_or_compile(*mir);
    if (bfn) {
        t.callee = &fn_info(*bfn, mir);
        t.callee_mir = t.callee->mir_fn;
    } else {
        auto it = external_functions_.find(t.name);
        if (it != external_functions_.end()) t.host = &it->second;
    }
    t.handle = dispatch_table().find(name);
    // A miss is never cached: the next call resolves again.
    const bool found = t.callee || t.host || (t.handle && t.handle->native_entry());
    t.epoch = found ? resolve_epoch_ : 0;
    t.registry_gen = gen;
    info.calls[cs_idx] = std::move(t);
}

void FastInterpreter::resolve_indirect_target(FastCallTarget& t, uintptr_t ptr) {
    const uint64_t gen = runtime::registry_generation();
    FastCallTarget r;
    r.indirect_ptr = ptr;
    if (!function_pointers_.count(ptr) && !bytecode_function_pointers_.count(ptr) &&
        !host_function_pointers_.count(ptr)) {
        // A code address native code made of a program function.
        const std::string name = dispatch_table().function_name_at(reinterpret_cast<const void*>(ptr));
        const Function* f = nullptr;
        if (!name.empty()) {
            if (runtime::FunctionHandle* h = dispatch_table().find(name)) f = h->mir_function();
            if (!f && module_) f = module_->get_function(name);
        }
        if (f) function_pointers_.emplace(ptr, f);
    }
    if (auto it = bytecode_function_pointers_.find(ptr); it != bytecode_function_pointers_.end()) {
        r.callee = &fn_info(*it->second);
        r.name = it->second->name;
    } else if (auto fit = function_pointers_.find(ptr); fit != function_pointers_.end()) {
        const Function* f = fit->second;
        r.callee = &fn_info(*get_or_compile(*f), f);
        r.callee_mir = f;
        r.name = std::string(f->name());
        r.handle = dispatch_table().find(f->name());
    } else if (auto hit = host_function_pointers_.find(ptr); hit != host_function_pointers_.end()) {
        r.host = &hit->second;
    } else {
        r.name = std::to_string(ptr);
    }
    r.epoch = (r.callee || r.host) ? resolve_epoch_ : 0;
    r.registry_gen = gen;
    t = std::move(r);
}

// ---------------------------------------------------------------------------
// Call paths
// ---------------------------------------------------------------------------

void FastInterpreter::execute_call(FastFrame& frame, uint32_t cs_idx) {
    FastFnInfo& info = *frame.info;
    FastCallTarget& t = info.calls[cs_idx];
    if (BRASS_UNLIKELY(t.epoch != resolve_epoch_ || t.registry_gen != runtime::registry_generation())) {
        resolve_call_target(info, cs_idx);
    }
    const CallSiteInfo& cs = frame.bfn->call_sites[cs_idx];
    dispatch_call(t, frame, cs, cs.patchable ? "Patchable call to undefined function: " : "Call to undefined function: ");
}

void FastInterpreter::execute_call_indirect(FastFrame& frame, uint32_t cs_idx) {
    FastFnInfo& info = *frame.info;
    const CallSiteInfo& cs = frame.bfn->call_sites[cs_idx];
    FastCallTarget& t = info.calls[cs_idx];
    const uintptr_t ptr = static_cast<uintptr_t>(frame.registers[cs.callee_reg]);
    if (BRASS_UNLIKELY(t.indirect_ptr != ptr || t.epoch != resolve_epoch_ ||
                       t.registry_gen != runtime::registry_generation())) {
        resolve_indirect_target(t, ptr);
    }
    dispatch_call(t, frame, cs, "Call indirect to unregistered target pointer: ");
}

void FastInterpreter::dispatch_call(FastCallTarget& t, FastFrame& frame, const CallSiteInfo& cs, const char* what) {
    RuntimeValue result;
    if (t.callee) {
        // Native code installed for exactly this function runs instead.
        runtime::FunctionHandle* h = t.handle;
        if (h && t.callee_mir && h->mir_function() == t.callee_mir && h->native_entry()) {
            result = call_native(*h, frame, cs);
        } else {
            result = call_bytecode(t, frame, cs);
        }
    } else if (t.handle && t.handle->native_entry()) {
        result = call_native(*t.handle, frame, cs);
    } else if (t.host) {
        result = call_host(*t.host, frame, cs);
    } else {
        throw InterpreterException(what + t.name);
    }
    if (cs.dst_reg != kNoReg) fast_set_reg(frame, cs.dst_reg, result);
}

RuntimeValue FastInterpreter::call_host(const FastHostFn& fn, const FastFrame& frame, const CallSiteInfo& cs) {
    ArgBufferScope scope(*this);
    std::vector<RuntimeValue>& args = *scope.buf;
    for (BcReg r : cs.arg_regs) args.push_back(fast_reg_value(frame, r));
    return fn(*this, args);
}

RuntimeValue FastInterpreter::call_native(runtime::FunctionHandle& handle, const FastFrame& frame, const CallSiteInfo& cs) {
    ArgBufferScope scope(*this);
    std::vector<RuntimeValue>& args = *scope.buf;
    for (BcReg r : cs.arg_regs) args.push_back(fast_reg_value(frame, r));
    return handle.call_native(args);
}

RuntimeValue FastInterpreter::call_bytecode(FastCallTarget& t, FastFrame& caller, const CallSiteInfo& cs) {
    FastFnInfo& callee = *t.callee;
    callee.tiering(dispatch_table_).record_invocation();
    // Reaching the tier-up threshold may have installed native code.
    runtime::FunctionHandle* h = t.handle;
    if (BRASS_UNLIKELY(h && t.callee_mir && h->mir_function() == t.callee_mir && h->native_entry())) {
        return call_native(*h, caller, cs);
    }
    if (BRASS_UNLIKELY(call_depth_ >= max_call_depth_)) {
        throw InterpreterException("Call stack depth limit exceeded (" + std::to_string(max_call_depth_) + ")");
    }
    if (BRASS_UNLIKELY(++total_instructions_executed_ > max_instructions_ && max_instructions_ > 0)) {
        throw_instruction_limit();
    }

    const uint32_t n = callee.num_regs;
    const FastAllocaArena::Mark mark = alloca_arena_->mark();
    uint64_t* regs = static_cast<uint64_t*>(alloca_arena_->allocate(static_cast<size_t>(n) * sizeof(uint64_t), alignof(uint64_t)));
    const size_t nargs = std::min<size_t>(cs.arg_regs.size(), n);
    for (size_t i = 0; i < nargs; ++i) regs[i] = caller.registers[cs.arg_regs[i]];

    FastFrame frame;
    frame.bfn = callee.bfn;
    frame.mir_fn = callee.mir_fn;
    frame.info = &callee;
    frame.registers = regs;
    frame.num_registers = n;
    if (BRASS_UNLIKELY(callee.has_vector_params && caller.vector_regs)) {
        for (size_t i = 0; i < nargs; ++i) {
            if (!callee.bfn->register_types[i].is_vector()) continue;
            std::memcpy(frame.ensure_vector_regs() + i * kFastVecBytes,
                        caller.vector_regs + static_cast<size_t>(cs.arg_regs[i]) * kFastVecBytes, kFastVecBytes);
        }
    }
    FrameGuard guard(*this, frame, mark);
    return execute_frame(frame);
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

RuntimeValue FastInterpreter::enter_frame(FastFnInfo& info, const std::vector<RuntimeValue>& args, uint32_t start_pc,
                                          const std::vector<BcReg>* arg_regs) {
    if (call_depth_ >= max_call_depth_) {
        throw InterpreterException("Call stack depth limit exceeded (" + std::to_string(max_call_depth_) + ")");
    }
    const uint32_t n = info.num_regs;
    const FastAllocaArena::Mark mark = alloca_arena_->mark();
    uint64_t* regs = static_cast<uint64_t*>(alloca_arena_->allocate(static_cast<size_t>(n) * sizeof(uint64_t), alignof(uint64_t)));

    FastFrame frame;
    frame.bfn = info.bfn;
    frame.mir_fn = info.mir_fn;
    frame.info = &info;
    frame.registers = regs;
    frame.num_registers = n;
    frame.pc = start_pc;
    FrameGuard guard(*this, frame, mark);
    for (size_t i = 0; i < args.size(); ++i) {
        const uint32_t reg = arg_regs ? (i < arg_regs->size() ? (*arg_regs)[i] : kNoReg) : static_cast<uint32_t>(i);
        if (reg >= n) continue;
        fast_set_reg(frame, reg, args[i]);
    }
    return execute_frame(frame);
}

RuntimeValue FastInterpreter::run(const Function& fn) {
    return run(fn, {});
}

RuntimeValue FastInterpreter::run(const Function& fn, const std::vector<RuntimeValue>& args) {
    runtime::ProgramScope program_scope(dispatch_table());
    if (fn.parent()) {
        use_module(fn.parent());
        if (!dispatch_table().tiering().active_module()) {
            dispatch_table().tiering().set_active_module(fn.parent());
        }
    }
    release_retired();

    auto* handle = dispatch_table().find(fn.name());
    if (handle && handle->mir_function() == &fn && handle->native_entry() != nullptr) {
        return handle->call_native(args);
    }

    const BytecodeFunction* bfn = get_or_compile(fn);
    FastFnInfo& info = fn_info(*bfn, &fn);
    info.tiering(dispatch_table_).record_invocation();

    if (handle && handle->mir_function() == &fn && handle->native_entry() != nullptr) {
        return handle->call_native(args);
    }
    return enter_frame(info, args, 0, nullptr);
}

RuntimeValue FastInterpreter::run(const BytecodeFunction& fn) {
    return run(fn, {});
}

RuntimeValue FastInterpreter::run(const BytecodeFunction& fn, const std::vector<RuntimeValue>& args) {
    runtime::ProgramScope program_scope(dispatch_table());
    release_retired();
    return enter_frame(fn_info(fn), args, 0, nullptr);
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
    use_module(&mod);
    const Function* fn = mod.get_function(entry_name);
    if (!fn && entry_name != "main") {
        fn = mod.get_function("main");
    }
    if (!fn && !mod.functions().empty()) {
        fn = mod.functions().front();
    }
    if (!fn) {
        throw InterpreterException("Entry function @" + std::string(entry_name) + " not found in module");
    }
    return run(*fn, args);
}

RuntimeValue FastInterpreter::resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    if (fn.parent()) use_module(fn.parent());
    const BytecodeFunction* bfn = get_or_compile(fn);
    fn_info(*bfn, &fn);
    return resume(*bfn, resume_id, state_values);
}

RuntimeValue FastInterpreter::resume(const BytecodeFunction& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values) {
    const ResumePointEntry* target = nullptr;
    for (const auto& rp : fn.resume_points) {
        if (rp.resume_id == resume_id) {
            target = &rp;
            break;
        }
    }
    if (!target) {
        throw InterpreterException("Resume target ID " + std::to_string(resume_id) + " not found in function " + fn.name);
    }
    release_retired();
    if (!target->state_regs.empty()) {
        // A guard's resume: its state-map values and the block's parameters
        // both take state value i (Interpreter::resume_with_frame).
        std::vector<RuntimeValue> vals;
        std::vector<BcReg> regs;
        for (size_t i = 0; i < state_values.size(); ++i) {
            vals.push_back(state_values[i]);
            regs.push_back(i < target->state_regs.size() ? target->state_regs[i] : kNoReg);
        }
        for (size_t i = 0; i < state_values.size() && i < target->param_regs.size(); ++i) {
            vals.push_back(state_values[i]);
            regs.push_back(target->param_regs[i]);
        }
        return enter_frame(fn_info(fn), vals, target->target_pc, &regs);
    }
    return enter_frame(fn_info(fn), state_values, target->target_pc,
                       target->param_regs.empty() ? nullptr : &target->param_regs);
}

namespace {
// Switching to the callee's module retires the compile caches (kept alive
// for the running frames); the frames below resolve calls in theirs again
// once it returns.
struct FastModuleRestore {
    FastInterpreter& interp;
    const Module* saved;
    ~FastModuleRestore() { interp.restore_module_after_native(saved); }
};
} // namespace

void FastInterpreter::restore_module_after_native(const Module* mod) { use_module(mod); }

RuntimeValue FastInterpreter::call_from_native(const Function& fn, const std::vector<RuntimeValue>& args) {
    FastModuleRestore restore{*this, module_};
    if (fn.parent()) use_module(fn.parent());
    const BytecodeFunction* bfn = get_or_compile(fn);
    return enter_frame(fn_info(*bfn, &fn), args, 0, nullptr);
}

RuntimeValue FastInterpreter::resume_from_native(const Function& fn, uint32_t resume_id,
                                                 const std::vector<RuntimeValue>& state_values) {
    FastModuleRestore restore{*this, module_};
    // The exits the guard takes, in its order (Interpreter::
    // resume_after_guard): the exit stub, whose result is the function's,
    // else the resume target.
    const Instruction* guard = fn.find_guard(resume_id);
    if (!guard) {
        throw InterpreterException("No guard with resume id " + std::to_string(resume_id) + " in function " +
                                   std::string(fn.name()));
    }
    if (const Function* stub = fn.guard_exit_stub(*guard)) return run(*stub, state_values);
    if (!fn.get_resume_target(resume_id)) {
        throw InterpreterException("Guard " + std::to_string(resume_id) + " in function " + std::string(fn.name()) +
                                   " has neither an exit stub nor a resume target");
    }
    return resume(fn, resume_id, state_values);
}

} // namespace brass
