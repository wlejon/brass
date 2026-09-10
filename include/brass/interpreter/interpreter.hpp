#pragma once

#include <brass/interpreter/value.hpp>
#include <brass/interpreter/frame.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>

namespace brass {

class Interpreter;
class GenerationalGC;

class InterpreterException : public std::runtime_error {
public:
    explicit InterpreterException(const std::string& msg) : std::runtime_error(msg) {}
};

struct DeoptResult {
    bool deoptimized = false;
    std::string exit_stub;
    std::vector<RuntimeValue> state_map;
    uint32_t resume_id = 0;
};

class DeoptException : public std::runtime_error {
public:
    explicit DeoptException(DeoptResult result)
        : std::runtime_error("Deoptimization triggered to exit stub: " + result.exit_stub),
          result_(std::move(result)) {}

    const DeoptResult& result() const noexcept { return result_; }

private:
    DeoptResult result_;
};

class InterpreterThrownException : public std::exception {
public:
    explicit InterpreterThrownException(RuntimeValue val) : value_(val) {}
    RuntimeValue value() const noexcept { return value_; }
    const char* what() const noexcept override { return "InterpreterThrownException"; }

private:
    RuntimeValue value_;
};

using HostFn = std::function<RuntimeValue(Interpreter& interp, const std::vector<RuntimeValue>& args)>;
using DeoptHandler = std::function<RuntimeValue(Interpreter& interp, const DeoptResult& deopt)>;

class Interpreter {
public:
    explicit Interpreter(size_t gc_semispace_size = MiniCheneyGC::DEFAULT_SEMISPACE_SIZE);
    ~Interpreter() = default;

    Interpreter(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter(Interpreter&&) noexcept = default;
    Interpreter& operator=(Interpreter&&) noexcept = default;

    // Module management
    void set_module(const Module* mod) noexcept { module_ = mod; }
    const Module* module() const noexcept { return module_; }

    // Garbage Collector access
    MiniCheneyGC& gc() noexcept { return gc_; }
    const MiniCheneyGC& gc() const noexcept { return gc_; }

    void set_generational_gc(GenerationalGC* gc) noexcept { gen_gc_ = gc; }
    GenerationalGC* generational_gc() noexcept { return gen_gc_; }
    const GenerationalGC* generational_gc() const noexcept { return gen_gc_; }

    // Allocation in managed GC heap
    uintptr_t allocate_gc(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);

    // Host / External function registration
    void register_external_function(std::string_view name, HostFn fn);
    bool has_external_function(std::string_view name) const noexcept;
    void register_function_pointer(uintptr_t ptr, const Function* fn);
    void register_function_pointer(uintptr_t ptr, HostFn fn);

    // Dynamic patching
    void patch_const(std::string_view symbol, int64_t val);
    int64_t get_patched_const(std::string_view symbol, int64_t default_val) const;
    void patch_call(std::string_view site, std::string_view target);
    std::string_view get_patched_call(std::string_view site, std::string_view default_callee) const;

    // Speculation and Deoptimization
    const DeoptResult& last_deopt() const noexcept { return last_deopt_; }
    void clear_last_deopt() noexcept { last_deopt_ = DeoptResult{}; }
    void set_deopt_handler(DeoptHandler handler) { deopt_handler_ = std::move(handler); }

    // Execution methods (Explicit overloads for GCC 12 compatibility)
    RuntimeValue run(const Function& fn);
    RuntimeValue run(const Function& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue run(std::string_view fn_name);
    RuntimeValue run(std::string_view fn_name, const std::vector<RuntimeValue>& args);
    RuntimeValue run(const Module& mod, std::string_view entry_name);
    RuntimeValue run(const Module& mod, std::string_view entry_name, const std::vector<RuntimeValue>& args);

    // Resume execution at a generic twin resume point
    RuntimeValue resume(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values);

    // Execution limits & diagnostics
    void set_max_call_depth(size_t max_depth) noexcept { max_call_depth_ = max_depth; }
    void set_max_instructions(uint64_t max_insts) noexcept { max_instructions_ = max_insts; }
    uint64_t instruction_count() const noexcept { return total_instructions_executed_; }
    void reset_instruction_count() noexcept { total_instructions_executed_ = 0; }

    // Frame inspection & roots
    InterpreterFrame* current_frame() noexcept { return current_frame_; }
    const InterpreterFrame* current_frame() const noexcept { return current_frame_; }
    void collect_all_roots(std::vector<uintptr_t*>& roots);

    RuntimeValue current_exception() const noexcept { return current_exception_; }
    void set_current_exception(RuntimeValue val) noexcept { current_exception_ = val; }

private:
    RuntimeValue execute_function(const Function& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue execute_function_from_block(const Function& fn, BasicBlock* start_block, const std::vector<RuntimeValue>& block_args);
    void register_builtin_host_functions();

    const Module* module_ = nullptr;
    MiniCheneyGC gc_;
    GenerationalGC* gen_gc_ = nullptr;

    InterpreterFrame* current_frame_ = nullptr;
    size_t call_depth_ = 0;
    size_t max_call_depth_ = 10000;
    uint64_t max_instructions_ = 0; // 0 = unlimited
    uint64_t total_instructions_executed_ = 0;

    std::unordered_map<std::string, HostFn> external_functions_;
    std::unordered_map<uintptr_t, const Function*> function_pointers_;
    std::unordered_map<uintptr_t, HostFn> host_function_pointers_;
    std::unordered_map<std::string, int64_t> patched_consts_;
    std::unordered_map<std::string, std::string> patched_calls_;

    DeoptResult last_deopt_;
    DeoptHandler deopt_handler_;
    RuntimeValue current_exception_;
};

} // namespace brass
