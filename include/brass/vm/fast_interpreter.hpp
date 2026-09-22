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

struct FastFrame;
struct FastCoroState;

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

    // Module management
    void set_module(const Module* mod) noexcept { module_ = mod; }
    const Module* module() const noexcept { return module_; }

    void set_bytecode_module(const BytecodeModule* bmod) noexcept { bytecode_module_ = bmod; }
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

    // Execution limits & diagnostics
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
    void coro_suspend(FastFrame& frame, uint8_t dst_reg, uint8_t yield_reg, uint32_t resume_id);
    void coro_destroy(uintptr_t handle);
    bool coro_is_done(uintptr_t handle) const;
    FastCoroState* get_coro_state(uintptr_t handle);

    // Exception handling helpers
    void handle_throw(FastFrame& frame, uint8_t reg, const uint32_t*& pc, const uint32_t* code_base);
    void handle_invoke(FastFrame& frame, const CallSiteInfo& cs, const uint32_t*& pc, const uint32_t* code_base);
    void handle_resume(FastFrame& frame, uint8_t reg);

    // OSR backedge helper
    bool handle_osr_backedge(FastFrame& frame, uint32_t target_pc, RuntimeValue& out_res);

    // Internal execution helpers
    RuntimeValue execute_frame(FastFrame& frame);
    RuntimeValue execute_call(FastFrame& frame, const CallSiteInfo& cs, BytecodeOp call_op);
    RuntimeValue execute_call_indirect(FastFrame& frame, const CallSiteInfo& cs);
    void execute_vector_op(FastFrame& frame, uint32_t inst);
    void handle_write_barrier(FastFrame& frame, uint8_t obj_reg, uint8_t val_reg);
    void handle_safepoint(FastFrame& frame);

private:
    void register_builtin_host_functions();

    const Module* module_ = nullptr;
    const BytecodeModule* bytecode_module_ = nullptr;
    MiniCheneyGC gc_;
    GenerationalGC* gen_gc_ = nullptr;

    FastFrame* current_frame_ = nullptr;
    size_t call_depth_ = 0;
    size_t max_call_depth_ = 10000;
    uint64_t max_instructions_ = 0;
    uint64_t total_instructions_executed_ = 0;

    std::unordered_map<std::string, FastHostFn> external_functions_;
    std::unordered_map<uintptr_t, const Function*> function_pointers_;
    std::unordered_map<uintptr_t, const BytecodeFunction*> bytecode_function_pointers_;
    std::unordered_map<uintptr_t, FastHostFn> host_function_pointers_;
    std::unordered_map<std::string, int64_t> patched_consts_;
    std::unordered_map<std::string, std::string> patched_calls_;

    std::unordered_map<const Function*, std::unique_ptr<BytecodeFunction>> compiled_functions_;
    BytecodeCompiler compiler_;

    DeoptResult last_deopt_;
    FastDeoptHandler deopt_handler_;
    RuntimeValue current_exception_;

    std::unordered_map<uintptr_t, std::unique_ptr<FastCoroState>> active_coros_;
    FastCoroState* active_coro_frame_ = nullptr;
};

} // namespace brass
