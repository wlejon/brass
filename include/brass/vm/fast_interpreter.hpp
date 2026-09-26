#pragma once

#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/gc/heap.hpp>
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

class FastInterpreter;
namespace runtime {
class FunctionHandle;
class FunctionDispatchTable;
}

struct FastFrame;
struct FastCoroState;
struct FastFnInfo;
struct FastCallTarget;
class FastAllocaArena;

// One frame a FastInterpreter is running on the calling thread
// (FastInterpreter::for_each_frame_on_thread).
struct InterpretedFrameInfo {
    // Where the frame's record lives: on this thread's native stack, inside
    // the interpreter's native frame that runs it, so its address orders it
    // among the native frames of the same stack.
    const void* frame_address = nullptr;
    // The MIR function it runs (null for bytecode compiled without one).
    const Function* function = nullptr;
    // The source position of the instruction it is at: the call it is in
    // for every frame but one that has not reached a call.
    DebugLoc loc;
};

using FastHostFn = std::function<RuntimeValue(FastInterpreter& interp, const std::vector<RuntimeValue>& args)>;
using FastDeoptHandler = std::function<RuntimeValue(FastInterpreter& interp, const DeoptResult& deopt)>;

class FastInterpreter {
public:
    using HostFn = FastHostFn;
    using DeoptHandler = FastDeoptHandler;

    // Allocates from `heap`; null: the thread's current heap
    // (gc::Heap::current()) when one is bound, else a heap of its own.
    explicit FastInterpreter(gc::Heap* heap = nullptr);
    // Allocates from a heap of its own, configured by `config`.
    explicit FastInterpreter(const gc::HeapConfig& config);
    ~FastInterpreter();

    FastInterpreter(const FastInterpreter&) = delete;
    FastInterpreter& operator=(const FastInterpreter&) = delete;
    // Not movable: its heap's root source captures `this`, as do the
    // thread's current interpreter and running frames. Hold one through a
    // unique_ptr to move it.
    FastInterpreter(FastInterpreter&&) = delete;
    FastInterpreter& operator=(FastInterpreter&&) = delete;

    // Active interpreter on current thread
    static FastInterpreter* current() noexcept;
    static void set_current(FastInterpreter* interp) noexcept;

    // Visits every frame the FastInterpreters on the calling thread are
    // running, of every interpreter (nested ones, and one entered again from
    // native code), innermost first, until `visit` returns false. For a host
    // walking its own stack, e.g. to report a stack trace.
    static void for_each_frame_on_thread(const std::function<bool(const InterpretedFrameInfo&)>& visit);
    // The innermost frame on this thread (the head of the chain the visit
    // walks); maintained by the interpreter as frames are entered and left.
    static FastFrame*& thread_frame_top() noexcept;

    // Module management
    void set_module(const Module* mod);
    const Module* module() const noexcept { return module_; }

    void set_bytecode_module(const BytecodeModule* bmod);
    const BytecodeModule* bytecode_module() const noexcept { return bytecode_module_; }

    // The program whose function handles calls are routed through (native
    // code installed for a callee runs instead of its bytecode). Null, the
    // default, is the default program (FunctionDispatchTable::instance()).
    // The table must outlive every call made through it.
    void set_dispatch_table(runtime::FunctionDispatchTable* table);
    runtime::FunctionDispatchTable& dispatch_table() const noexcept;

    // The heap this interpreter allocates from, as Interpreter::heap(): its
    // frames are the heap's roots, and it is the thread's current heap
    // while the interpreter runs.
    gc::Heap& heap() noexcept { return *heap_; }
    const gc::Heap& heap() const noexcept { return *heap_; }
    bool owns_heap() const noexcept { return own_heap_ != nullptr; }
    // Allocates from `heap` from now on (null: a heap of its own). Not while
    // it runs.
    void use_heap(gc::Heap* heap);

    // A zeroed object from heap() (Interpreter::allocate_gc).
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
    // As Interpreter::function_address: the program's one pointer to `fn`.
    uintptr_t function_address(const Function& fn);
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
    // Run on this thread's native stack by native code this interpreter's
    // frames called (as Interpreter::call_from_native and
    // resume_from_native): the new frames sit above theirs, so they
    // allocate in this heap and their gcrefs are this GC's roots. The
    // frames below keep running in their module.
    RuntimeValue call_from_native(const Function& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue resume_from_native(const Function& fn, uint32_t resume_id,
                                    const std::vector<RuntimeValue>& state_values);
    // Makes `mod` current again after a call_from_native/resume_from_native.
    void restore_module_after_native(const Module* mod);

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
    // The slots of this interpreter's gcref- and tagged-typed registers
    // (frames and suspended coroutines), its `alloca.tagged` words, its
    // exception in flight and its last deopt state.
    void collect_all_roots(std::vector<uintptr_t*>& roots);
    // As collect_all_roots, the gcrefs and the tagged values apart.
    void collect_typed_roots(std::vector<uintptr_t*>& gcrefs, std::vector<uintptr_t*>& tagged);
    // The nonzero registers of no reference type, which may still hold a
    // reference a host or native callee returned untyped.
    void collect_untyped_registers(std::vector<uintptr_t*>& slots);

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
    // `mode` is the resume mode a lowered body reads (CoroResumeMode).
    uint64_t coro_resume(uintptr_t handle, uint64_t input_val = 0, uint32_t mode = 0);
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
    // call_indirect to native code that is no program function (a host
    // function pointer the program loaded): called with the call site's
    // argument and result types, as native code would call it.
    RuntimeValue call_raw_native(uintptr_t ptr, const FastFrame& frame, const CallSiteInfo& cs);
    RuntimeValue call_bytecode(FastCallTarget& t, FastFrame& caller, const CallSiteInfo& cs);
    // Runs `info`'s function from `start_pc` with args[i] in register
    // (*arg_regs)[i], or in register i when arg_regs is null.
    RuntimeValue enter_frame(FastFnInfo& info, const std::vector<RuntimeValue>& args, uint32_t start_pc,
                             const std::vector<BcReg>* arg_regs);
    Interpreter& host_adapter_interpreter();
    [[noreturn]] void throw_instruction_limit() const;

    const Module* module_ = nullptr;
    const BytecodeModule* bytecode_module_ = nullptr;
    runtime::FunctionDispatchTable* dispatch_table_ = nullptr;
    void attach_heap(gc::Heap* heap);
    void detach_heap() noexcept;
    // fn(slot, kind) for every nonzero register of the running frames and
    // suspended coroutines, with the root kind its type gives it.
    enum class RegisterRootKind : uint8_t { None, GcRef, Tagged };
    template <typename Fn>
    void for_each_register(Fn&& fn);
    std::unique_ptr<gc::Heap> own_heap_;
    gc::Heap* heap_ = nullptr;
    gc::Heap::RootSourceId root_source_ = 0;

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
    // The `alloca.tagged` buffers of the active frames (start, words), in
    // allocation order: a frame drops its own on exit (FrameGuard).
    std::vector<std::pair<uint64_t*, uint32_t>> tagged_allocas_;
    friend struct FrameGuard;

    DeoptResult last_deopt_;
    FastDeoptHandler deopt_handler_;
    RuntimeValue current_exception_;

    // Lowered coroutine bodies (CoroTransformPass) run as on every other
    // tier: the handle is the BrassCoroFrame, arguments fill its slots
    // (coro_slot_count each) and each resume calls the body with the frame.
    // The frame's body is the descriptor of the MIR body in this
    // interpreter's program (CORO_FLAG_BODY), as the Interpreter's is.
    // active_coros_ holds only the unlowered, register-snapshot coroutines
    // this interpreter also runs.
    uintptr_t coro_create_lowered(const Function& fn, const std::vector<RuntimeValue>& args);
    uint64_t coro_resume_lowered(uintptr_t frame, uint64_t input_val, uint32_t mode);
    // The MIR body of the frame `handle`, or null (generated code's frame).
    const Function* lowered_coro_body(uintptr_t handle) const;

    std::unordered_map<uintptr_t, std::unique_ptr<FastCoroState>> active_coros_;
    FastCoroState* active_coro_frame_ = nullptr;
};

} // namespace brass
