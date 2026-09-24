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
#include <unordered_set>
#include <cstdint>
#include <cstddef>

namespace brass::runtime {
class FunctionDispatchTable;
}

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
    // The symbols the code calls directly through a lazy-link stub because
    // they did not resolve when it was compiled.
    const std::vector<std::string>& lazy_call_symbols() const noexcept { return lazy_call_symbols_; }
    void set_lazy_call_symbols(std::vector<std::string> names) { lazy_call_symbols_ = std::move(names); }
    // The symbols whose address (func_addr) the code takes as a lazy-link
    // stub because they did not resolve when it was compiled.
    const std::vector<std::string>& lazy_addr_symbols() const noexcept { return lazy_addr_symbols_; }
    void set_lazy_addr_symbols(std::vector<std::string> names) { lazy_addr_symbols_ = std::move(names); }
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
    std::vector<std::string> lazy_call_symbols_;
    std::vector<std::string> lazy_addr_symbols_;
    // The code's entry in the code stack-map registry, shared by the copies
    // of this function. Last, so it goes before the code memory does.
    std::shared_ptr<const void> stack_map_registration_;
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
    // Called by a lazy stub of a module function (one whose address code
    // took) that has no native entry when first called through: compiles
    // and installs it, returning its entry, or null when it cannot be.
    // Runs on the calling thread, outside the stub table's lock.
    void set_on_demand_compiler(BaselineSymbolResolver compile);

    // The program whose handles resolve_symbol() falls back to and
    // compile_module() publishes into. Null, the default, is the default
    // program (FunctionDispatchTable::instance()). The table must outlive
    // the compiler and every function it compiled.
    void set_dispatch_table(runtime::FunctionDispatchTable* table);
    runtime::FunctionDispatchTable& dispatch_table() const;

    // A direct call or func_addr whose symbol does not resolve at compile
    // time (x64) goes through a per-symbol stub that resolves on first call:
    // through register_external_symbol, the custom resolver or the dispatch
    // table, whichever has it by then. func_addr yields the stub, a stable,
    // callable address. A call that finds the symbol still unresolved is a
    // hard error (reported, then ud2), never a call through null. The stubs
    // stay alive as long as any function compiled against them.
    void* lazy_stub(std::string_view name);
    // The one address of program function `name` that every tier uses as
    // its function pointer: its lazy stub, claimed for the program (it
    // resolves through the dispatch table and compiles the function on
    // first call, never to a registered symbol of the name) and registered
    // in the dispatch table (register_code_address).
    void* module_function_stub(std::string_view name);
    // What func_addr of `name` in `fn` yields when it names a function of
    // fn's module with a body and module functions are shadowed
    // (set_module_functions_shadow): module_function_stub(name). Null
    // otherwise (the symbol resolves as any other).
    void* function_address_in(const Function& fn, std::string_view name);
    const std::shared_ptr<LazySymbolTable>& lazy_symbols() const noexcept { return lazy_; }

    BaselineCompiledFunction compile(const Function& fn);
    BaselineCompiledFunction compile(const Function& fn, Target target);
    // Compiles and publishes every function of `mod`, which is taken to be
    // the whole program: a direct call whose callee is neither a function of
    // `mod` nor resolvable once the module's symbols are in is a hard error
    // (std::runtime_error naming it) instead of a lazy stub that could only
    // trap. compile() of a single function keeps linking such a callee
    // lazily, and a func_addr stays lazy in both.
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod);
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod, Target target);

    void* resolve_symbol(std::string_view name) const;
    // resolve_symbol, after the string data `fn`'s module defines itself
    // (Module::define_string_symbol) and the functions it defines: a module
    // function shadows a registered symbol of its name. One not compiled yet
    // resolves to null (a stub); compile_module points the stub at the
    // module's copy when it installs it. With set_module_functions_shadow,
    // the name is also claimed for the module: its stub resolves only through
    // the dispatch table, never to a registered symbol, whenever the function
    // is compiled (a tier-up that compiles one function at a time).
    void* resolve_symbol_in(const Function& fn, std::string_view name) const;

    // For a host whose registered symbols are all outside the program (the
    // tiering pipeline: runtime and libm functions) and whose program
    // functions reach native code only through the dispatch table. Off by
    // default: a host may register the code of a module function under its
    // own name (compiled by itself or another tier).
    void set_module_functions_shadow(bool shadow);

    // Whether the baseline tier compiles `op`, on x64 and AArch64 alike. A
    // function using an opcode it does not is rejected at compile time:
    // compile() throws UnsupportedOperation (stage "x64 baseline" or
    // "aarch64 baseline") and the tiering layer keeps the function in the
    // interpreter. The same exception rejects a function with vector-typed
    // values the tier does not compile and a guard with no exit stub or
    // resume target.
    static bool supports_opcode(Opcode op) noexcept;
    static bool x64_supports_opcode(Opcode op) noexcept { return supports_opcode(op); }
    // compile()'s up-front check alone, without compiling: false when it
    // would reject `fn` before emitting code. (Emission may still reject.)
    bool passes_prescan(const Function& fn, Target target) const;

private:
    Target target_;
    mutable std::mutex symbols_mutex_;
    std::unordered_map<std::string, void*> symbols_;
    BaselineSymbolResolver custom_resolver_;
    BaselineSymbolResolver on_demand_compiler_; // under symbols_mutex_
    runtime::FunctionDispatchTable* dispatch_table_ = nullptr; // under symbols_mutex_
    // Names a stub was made for on behalf of a module that defines the
    // function (resolve_symbol_in); under symbols_mutex_.
    mutable std::unordered_set<std::string> module_owned_;
    bool module_functions_shadow_ = false; // under symbols_mutex_
    std::shared_ptr<LazySymbolTable> lazy_;

    // The lazy stubs' resolver: a module-owned name through the dispatch
    // table only, any other through resolve_symbol.
    void* resolve_lazy(std::string_view name) const;
};

} // namespace brass::codegen
