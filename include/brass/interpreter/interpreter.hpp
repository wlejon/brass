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

namespace runtime {
class FunctionDispatchTable;
}

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

// The exception value a landing pad of type `t` receives: a value a native
// callee threw is bare bits (i64-kind), which the pad's type reinterprets (an
// integer narrower than 64 bits keeps its low bits). A typed value is
// reinterpreted only at its own width (i64/f64/ptr/gcref, i32/f32); a pad
// whose type cannot hold the thrown value (another width, a vector for a
// scalar or the reverse) is a hard error: InterpreterException.
RuntimeValue retype_exception_value(RuntimeValue v, Type t);

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
    // Not movable: its heap's root provider captures `this`, as do the
    // thread's active interpreter and running frames. Hold one through a
    // unique_ptr to move it.
    Interpreter(Interpreter&&) = delete;
    Interpreter& operator=(Interpreter&&) = delete;

    // Module management
    void set_module(const Module* mod) noexcept { module_ = mod; }
    const Module* module() const noexcept { return module_; }

    // The program whose function handles calls are registered in and routed
    // through; null (the default) is FunctionDispatchTable::instance(). The
    // table must outlive every call made through it.
    void set_dispatch_table(runtime::FunctionDispatchTable* table) noexcept { dispatch_table_ = table; }
    runtime::FunctionDispatchTable& dispatch_table() const noexcept;

    // Garbage Collector access: the heap this interpreter allocates from,
    // its own unless it borrows one.
    MiniCheneyGC& gc() noexcept { return borrowed_gc_ ? *borrowed_gc_ : gc_; }
    const MiniCheneyGC& gc() const noexcept { return borrowed_gc_ ? *borrowed_gc_ : gc_; }
    // Allocates from `heap` (null: its own) instead. The caller keeps this
    // interpreter's frames among `heap`'s roots while it runs (its root
    // provider only knows the heap owner's), e.g. with a ThreadRootsScope.
    void borrow_gc(MiniCheneyGC* heap) noexcept { borrowed_gc_ = heap; }

    void set_generational_gc(GenerationalGC* gc) noexcept { gen_gc_ = gc; }
    // Allocates from, and collects (brass_gc_collect), the generational
    // `heap` (null: stop) instead of gc(); also its write barrier. As with
    // borrow_gc, the caller keeps this interpreter's frames among `heap`'s
    // roots while it runs (a ThreadRootsScope); `heap`'s root provider is
    // left alone.
    void borrow_generational_gc(GenerationalGC* heap) noexcept {
        borrowed_gen_gc_ = heap;
        gen_gc_ = heap;
    }
    GenerationalGC* borrowed_generational_gc() noexcept { return borrowed_gen_gc_; }
    GenerationalGC* generational_gc() noexcept { return gen_gc_; }
    const GenerationalGC* generational_gc() const noexcept { return gen_gc_; }

    // Allocation in managed GC heap
    uintptr_t allocate_gc(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);

    // Host / External function registration
    void register_external_function(std::string_view name, HostFn fn);
    bool has_external_function(std::string_view name) const noexcept;
    void register_function_pointer(uintptr_t ptr, const Function* fn);
    void register_function_pointer(uintptr_t ptr, HostFn fn);
    const Function* find_function_by_pointer(uintptr_t ptr) const noexcept;
    // The program's one pointer to `fn`, what func_addr yields: the code
    // address every tier uses (MultiTierPipeline::function_address) for a
    // function with a body, else the Function's own address.
    uintptr_t function_address(const Function& fn);
    // The function a call_indirect through `ptr` runs: one registered with
    // register_function_pointer, or the program function at a code address
    // (FunctionDispatchTable::function_name_at). Null if neither.
    const Function* function_at(uintptr_t ptr);

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
    RuntimeValue resume_with_frame(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values, InterpreterFrame& frame);
    // Finishes a call whose guard `resume_id` failed in native code, the way
    // the interpreter's own guard does: the exit stub if the guard has one,
    // else the resume target (in `frame` when given). Neither is an error.
    RuntimeValue resume_after_guard(const Function& fn, uint32_t resume_id, const std::vector<RuntimeValue>& state_values, InterpreterFrame* frame);

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

    // The innermost interpreter running a frame on this thread, or null.
    // Native code it called (a native-to-Tier-0 bridge) re-enters it, so the
    // callee shares its heap and a MIR exception unwinds to its handlers.
    static Interpreter* active_on_thread() noexcept;
    // Runs `fn` as a call from native code on this thread (Tier 0, or its
    // native entry if it has one). A MIR exception propagates as
    // InterpreterThrownException.
    RuntimeValue call_from_native(const Function& fn, const std::vector<RuntimeValue>& args) {
        return execute_function(fn, args);
    }
    // As call_from_native, with `fn`'s own module current for the call (a
    // coroutine body of another module resolves its callees there); the
    // frames below keep running in theirs.
    RuntimeValue call_in_own_module(const Function& fn, const std::vector<RuntimeValue>& args);
    // Finishes, on this thread's native stack, a call whose guard
    // `resume_id` failed in native code that this interpreter's frames
    // called (resume_after_guard, without a frame): the continuation's
    // frames sit above them, so it allocates in this heap and its gcrefs
    // are this GC's roots.
    RuntimeValue resume_from_native(const Function& fn, uint32_t resume_id,
                                    const std::vector<RuntimeValue>& state_values);

private:
    struct ActiveScope {
        explicit ActiveScope(Interpreter* interp) noexcept;
        ~ActiveScope();
        ActiveScope(const ActiveScope&) = delete;
        ActiveScope& operator=(const ActiveScope&) = delete;
        Interpreter* prev;
    };
    RuntimeValue execute_function(const Function& fn, const std::vector<RuntimeValue>& args);
    RuntimeValue execute_function_from_block(const Function& fn, BasicBlock* start_block, const std::vector<RuntimeValue>& block_args, InterpreterFrame* existing_frame = nullptr);
    void register_builtin_host_functions();

    const Module* module_ = nullptr;
    runtime::FunctionDispatchTable* dispatch_table_ = nullptr;
    MiniCheneyGC gc_;
    MiniCheneyGC* borrowed_gc_ = nullptr;
    GenerationalGC* gen_gc_ = nullptr;
    GenerationalGC* borrowed_gen_gc_ = nullptr;

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
