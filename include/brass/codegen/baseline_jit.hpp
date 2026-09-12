#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/codegen/jit_exec.hpp>
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
};

using BaselineSymbolResolver = std::function<void*(std::string_view)>;

class BaselineJitCompiler {
public:
    explicit BaselineJitCompiler(Target target = Target::host());
    ~BaselineJitCompiler() = default;

    const Target& target() const noexcept { return target_; }
    void set_target(const Target& target) noexcept { target_ = target; }

    void register_external_symbol(std::string_view name, void* addr);
    void set_symbol_resolver(BaselineSymbolResolver resolver) { custom_resolver_ = std::move(resolver); }

    BaselineCompiledFunction compile(const Function& fn);
    BaselineCompiledFunction compile(const Function& fn, Target target);
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod);
    std::vector<BaselineCompiledFunction> compile_module(const Module& mod, Target target);

    void* resolve_symbol(std::string_view name) const;

private:
    Target target_;
    std::unordered_map<std::string, void*> symbols_;
    BaselineSymbolResolver custom_resolver_;
};

} // namespace brass::codegen
