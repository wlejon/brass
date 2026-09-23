#pragma once

#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <stdexcept>

namespace brass {

class GenerationalGC;
class FastInterpreter;
namespace runtime {
class FunctionHandle;
}

struct FastFrame;
struct FastCoroState;
struct FastFnInfo;
struct FastCallTarget;
class FastAllocaArena;

using FastHostFn = std::function<RuntimeValue(FastInterpreter& interp, const std::vector<RuntimeValue>& args)>;
using FastDeoptHandler = std::function<RuntimeValue(FastInterpreter& interp, const DeoptResult& deopt)>;

class FastInterpreter {
public:
    using HostFn = FastHostFn;
    using DeoptHandler = FastDeoptHandler;

    explicit FastInterpreter(size_t gc_semispace_size = MiniCheneyGC::DEFAULT_SEMISPACE_SIZE);
    ~FastInterpreter();

    FastInterpreter(const FastInterpreter&) = delete;
    FastInterpreter& operator=(const FastInterpreter&) = delete;
    FastInterpreter(FastInterpreter&&) noexcept;
    FastInterpreter& operator=(FastInterpreter&&) noexcept;

    // Active interpreter on current thread
    static FastInterpreter* current() noexcept;
    static void set_current(FastInterpreter* interp) noexcept;

    // Module management
    void set_module(const Module* mod);
    const Module* module() const noexcept { return module_; }

    void set_bytecode_module(const BytecodeModule* bmod);
    const BytecodeModule* bytecode_module() const noexcept { return bytecode_module_; }

    // GC access
    MiniCheneyGC& gc() noexcept { return gc_; }
    const MiniCheneyGC& gc() const noexcept { return gc_; }

    void set_generational_gc(GenerationalGC* gc) noexcept;
    GenerationalGC* generational_gc() noexcept { return gen_gc_; }
    const GenerationalGC* generational_gc() const noexcept { return gen_gc_; }

    // Allocation in managed GC heap
    uintptr_t allocate_gc(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);

    // Host / External function registration
    void register_external_function(std::string_view name, FastHostFn fn);
    void register_external_function(std::string_view name, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn);
    void register_external_function(std::string_view name, brass::HostFn fn);
    bool has_external_function(std::string_view name) const noexcept;

    // External symbol registration (data and function pointers)
    void register_external_symbol(std::string_view name, void* addr);
    void* find_external_symbol(std::string_view name) const noexcept;
    bool has_external_symbol(std::string_view name) const noexcept;

    // Function pointer registration
    void register_function_pointer(uintptr_t ptr, const Function* fn);
    void register_function_pointer(uintptr_t ptr, const BytecodeFunction* bfn);
    void register_function_pointer(uintptr_t ptr, FastHostFn fn);
    void register_function_pointer(uintptr_t ptr, std::function<RuntimeValue(const std::vector<RuntimeValue>&)> fn);
    void register_function_pointer(uintptr_t ptr, brass::HostFn fn);
    const Function* find_function_by_pointer(uintptr_t ptr) const noexcept;
    const BytecodeFunction* find_bytecode_function_by_pointer(uintptr_t ptr) const noexcept;

    // Dynamic patching
    void patch_const(std::string_view symbol, int64_t val);
    int64_t get_patched_const(std::string_view symbol, int64_t default_val) const;
    void patch_call(std::string_view site, std::string_view target);
    std::string_view get_patched_call(std::string_view site, std::string_view default_callee) const;

    // Speculation and Deoptimization
    const DeoptResult& last_deopt() const noexcept { return last_deopt_; }
    void clear_last_deopt() noexcept { last_deopt_ = DeoptResult{}; }
    void set_deopt_handler(FastDeoptHandler handler) { deopt_handler_ = std::move(handler); }

    // Execution methods
    RuntimeValue run(const Function& fn);
    RuntimeValue run(const Function& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue run(const BytecodeFunction& fn);
    RuntimeValue run(const BytecodeFunction& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue run(std::string_view fn_name);
    RuntimeValue run(std::string_view fn_name, const std::vector<RuntimeValue>& args);
    RuntimeValue run(const Module& mod, std::string_view entry_name);
    RuntimeValue run(const Module& mod, std::string_view entry_name, const std::vector<RuntimeValue>& args);

    // Resume execution
    RuntimeValue resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values);
    RuntimeValue resume(const BytecodeFunction& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values);

    // Execution limits & diagnostics. The instruction budget is charged per
    // loop iteration (by the loop's length in bytecode) and per call, so
    // instruction_count() approximates the instructions executed and a
    // runaway loop or recursion hits the limit.
    void set_max_call_depth(size_t max_depth) noexcept { max_call_depth_ = max_depth; }
    size_t max_call_depth() const noexcept { return max_call_depth_; }
    void set_max_instructions(uint64_t max_insts) noexcept { max_instructions_ = max_insts; }
    uint64_t max_instructions() const noexcept { return max_instructions_; }
    uint64_t instruction_count() const noexcept { return total_instructions_executed_; }
    void reset_instruction_count() noexcept { total_instructions_executed_ = 0; }

    // Frame inspection & roots
    FastFrame* current_frame() noexcept { return current_frame_; }
    const FastFrame* current_frame() const noexcept { return current_frame_; }
    void set_current_frame(FastFrame* frame) noexcept { current_frame_ = frame; }
    void collect_all_roots(std::vector<uintptr_t*>& roots);

    void inc_call_depth() noexcept { ++call_depth_; }
    void dec_call_depth() noexcept { if (call_depth_ > 0) --call_depth_; }
    size_t current_call_depth() const noexcept { return call_depth_; }

    RuntimeValue current_exception() const noexcept { return current_exception_; }
    void set_current_exception(RuntimeValue val) noexcept { current_exception_ = val; }

    void set_tls_block(uint64_t tls) noexcept { tls_block_ = tls; }
    uint64_t tls_block() const noexcept { return tls_block_; }

    // Compilation cache management
    const BytecodeFunction* get_or_compile(const Function& fn);
    void clear_compile_cache() noexcept;

    // Coroutine operations
    uintptr_t coro_create(const BytecodeFunction* bfn, const std::vector<uint64_t>& args = {});
    uintptr_t coro_create(const Function& fn, const std::vector<RuntimeValue>& args = {});
    uintptr_t coro_create(const Module& mod, std::string_view callee, const std::vector<RuntimeValue>& args = {});
    uintptr_t coro_create(std::string_view callee, const std::vector<RuntimeValue>& args = {});
    uint64_t coro_resume(uintptr_t handle, uint64_t input_val = 0);
    RuntimeValue coro_resume_val(uintptr_t handle, RuntimeValue input_val = RuntimeValue::from_i64(0));
    void coro_suspend(FastFrame& frame, uint32_t dst_reg, uint32_t yield_reg, uint32_t resume_id);
    void coro_destroy(uintptr_t handle);
    bool coro_is_done(uintptr_t handle) const;
    FastCoroState* get_coro_state(uintptr_t handle);

    // Exception handling helpers
    void handle_throw(FastFrame& frame, uint32_t reg, const BytecodeWord*& pc, const BytecodeWord* code_base);
    void handle_invoke(FastFrame& frame, uint32_t cs_idx, const BytecodeWord*& pc, const BytecodeWord* code_base);
    void handle_resume(FastFrame& frame, uint32_t reg);

    // OSR backedge helper
    bool handle_osr_backedge(FastFrame& frame, uint32_t target_pc, RuntimeValue& out_res);

    // Internal execution helpers
    RuntimeValue execute_frame(FastFrame& frame);
    // Calls call site `cs_idx` of the frame's function (call, patchable_call,
    // invoke) and writes the result register.
    void execute_call(FastFrame& frame, uint32_t cs_idx);
    void execute_call_indirect(FastFrame& frame, uint32_t cs_idx);
    void execute_vector_op(FastFrame& frame, BytecodeWord inst, const BytecodeWord* pc);
    void handle_write_barrier(FastFrame& frame, uint32_t obj_reg, uint32_t val_reg);
    void handle_safepoint(FastFrame& frame);

    // Per-function runtime state (call-site caches, tiering counters) of a
    // bytecode function this interpreter runs.
    FastFnInfo& fn_info(const BytecodeFunction& bfn, const Function* mir_fn = nullptr);

private:
    void register_builtin_host_functions();
    // Drops every cached call-site resolution (module, symbol or patch
    // changes).
    void invalidate_call_caches() noexcept { ++resolve_epoch_; }
    // Makes `mod` the current module. Switching modules retires the compile
    // cache: a later Function at a dead one's address must not hit it.
    void use_module(const Module* mod);
    // Moves compiled code and per-function infos aside (kept alive for
    // running frames) and invalidates every call-site cache.
    void retire_caches();
    void release_retired() noexcept;
    void resolve_call_target(FastFnInfo& info, uint32_t cs_idx);
    void resolve_indirect_target(FastCallTarget& t, uintptr_t ptr);
    void dispatch_call(FastCallTarget& t, FastFrame& frame, const CallSiteInfo& cs, const char* what);
    RuntimeValue call_host(const FastHostFn& fn, const FastFrame& frame, const CallSiteInfo& cs);
    RuntimeValue call_native(runtime::FunctionHandle& handle, const FastFrame& frame, const CallSiteInfo& cs);
    RuntimeValue call_bytecode(FastCallTarget& t, FastFrame& caller, const CallSiteInfo& cs);
    // Runs `info`'s function from `start_pc` with args[i] in register
    // (*arg_regs)[i], or in register i when arg_regs is null.
    RuntimeValue enter_frame(FastFnInfo& info, const std::vector<RuntimeValue>& args, uint32_t start_pc,
                             const std::vector<BcReg>* arg_regs);
    Interpreter& host_adapter_interpreter();
    [[noreturn]] void throw_instruction_limit() const;

    const Module* module_ = nullptr;
    const BytecodeModule* bytecode_module_ = nullptr;
    MiniCheneyGC gc_;
    GenerationalGC* gen_gc_ = nullptr;

    FastFrame* current_frame_ = nullptr;
    size_t call_depth_ = 0;
    size_t max_call_depth_ = 10000;
    uint64_t max_instructions_ = 0;
    uint64_t total_instructions_executed_ = 0;
    uint64_t tls_block_ = 0;

    std::unordered_map<std::string, FastHostFn> external_functions_;
    std::unordered_map<std::string, void*> external_symbols_;
    std::unordered_map<uintptr_t, const Function*> function_pointers_;
    std::unordered_map<uintptr_t, const BytecodeFunction*> bytecode_function_pointers_;
    std::unordered_map<uintptr_t, FastHostFn> host_function_pointers_;
    std::unordered_map<std::string, int64_t> patched_consts_;
    std::unordered_map<std::string, std::string> patched_calls_;

    std::unordered_map<const Function*, std::unique_ptr<BytecodeFunction>> compiled_functions_;
    BytecodeCompiler compiler_;
    std::unordered_map<const BytecodeFunction*, std::unique_ptr<FastFnInfo>> fn_infos_;
    uint64_t resolve_epoch_ = 1;
    // Compiled code and infos dropped while frames may still use them;
    // freed once no frame is active.
    std::vector<std::unique_ptr<BytecodeFunction>> retired_bytecode_;
    std::vector<std::unique_ptr<FastFnInfo>> retired_infos_;

    // Argument vectors for host and native calls, one per nesting level so
    // re-entrant calls never share one; reused across calls.
    std::vector<std::unique_ptr<std::vector<RuntimeValue>>> arg_buffers_;
    size_t arg_buffer_depth_ = 0;
    friend struct ArgBufferScope;

    // The Interpreter handed to brass::HostFn callbacks, built once.
    std::unique_ptr<Interpreter> host_adapter_;
    std::unique_ptr<FastAllocaArena> alloca_arena_;
    friend struct FrameGuard;

    DeoptResult last_deopt_;
    FastDeoptHandler deopt_handler_;
    RuntimeValue current_exception_;

    std::unordered_map<uintptr_t, std::unique_ptr<FastCoroState>> active_coros_;
    FastCoroState* active_coro_frame_ = nullptr;
};

} // namespace brass
