#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/lazy_symbols.hpp>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace brass::codegen {

class BaselineCompiledFunction {
public:
    BaselineCompiledFunction() = default;
    BaselineCompiledFunction(
        std::string_view name,
        Type return_type,
        std::vector<Type> param_types,
        std::shared_ptr<JitMemoryBlock> memory,
        void* entry_point,
        size_t code_size,
        FunctionStackMap stack_map
    );

    std::string_view name() const noexcept { return name_; }
    Type return_type() const noexcept { return return_type_; }
    const std::vector<Type>& param_types() const noexcept { return param_types_; }
    void* entry_point() const noexcept { return entry_point_; }
    size_t code_size() const noexcept { return code_size_; }
    const FunctionStackMap& stack_map() const noexcept { return stack_map_; }
    FunctionStackMap& stack_map() noexcept { return stack_map_; }
    std::shared_ptr<JitMemoryBlock> memory() const noexcept { return memory_; }
    // Keeps the lazy-link stubs the code calls through alive with it.
    void set_link_keepalive(std::shared_ptr<const void> keepalive) { link_keepalive_ = std::move(keepalive); }
    bool is_valid() const noexcept { return entry_point_ != nullptr; }

    template <typename FuncPtr>
    FuncPtr get_function_ptr() const noexcept {
        return reinterpret_cast<FuncPtr>(entry_point_);
    }

    RuntimeValue invoke(const std::vector<RuntimeValue>& args = {}) const;

private:
    std::string name_;
    Type return_type_ = Type::void_type();
    std::vector<Type> param_types_;
    std::shared_ptr<JitMemoryBlock> memory_;
    void* entry_point_ = nullptr;
    size_t code_size_ = 0;
    FunctionStackMap stack_map_;
    std::shared_ptr<const void> link_keepalive_;
};

using BaselineSymbolResolver = std::function<void*(std::string_view)>;

class BaselineJitCompiler {
public:
    explicit BaselineJitCompiler(Target target = Target::host());
    ~BaselineJitCompiler();

    BaselineJitCompiler(const BaselineJitCompiler&) = delete;
    BaselineJitCompiler& operator=(const BaselineJitCompiler&) = delete;

    const Target& target() const noexcept { return target_; }
    void set_target(const Target& target) noexcept { target_ = target; }

    // Also fills the lazy-link cell of `name` if compiled code already
    // references it (see below).
    void register_external_symbol(std::string_view name, void* addr);
    void set_symbol_resolver(BaselineSymbolResolver resolver);

    // A direct call or func_addr whose symbol does not resolve at compile
    // time (x64) goes through a per-symbol stub that resolves on first call:
    // through register_external_symbol, the custom resolver or the dispatch
    // table, whichever has it by then. func_addr yields the stub, a stable,
    // callable address. A call that finds the symbol still unresolved is a
    // hard error (reported, then ud2), never a call through null. The stubs
    // stay alive as long as any function compiled against them.
    void* lazy_stub(std::string_view name);
    const std::shared_ptr<LazySymbolTable>& lazy_symbols() const noexcept { return lazy_; }

    BaselineCompiledFunction compile(const Function& fn);
    BaselineCompiledFunction compile(const Function& fn, Target target);
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod);
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod, Target target);

    void* resolve_symbol(std::string_view name) const;

    // Whether the x64 baseline tier compiles `op`. A function using an opcode
    // it does not is rejected at compile time: compile() throws
    // UnsupportedOperation (stage "x64 baseline") and the tiering layer keeps
    // the function in the interpreter. The same exception rejects a function
    // with vector-typed values and a guard with no exit stub or resume target.
    static bool x64_supports_opcode(Opcode op) noexcept;

private:
    Target target_;
    mutable std::mutex symbols_mutex_;
    std::unordered_map<std::string, void*> symbols_;
    BaselineSymbolResolver custom_resolver_;
    std::shared_ptr<LazySymbolTable> lazy_;
};

} // namespace brass::codegen
