#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/object/object_writer.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace brass::codegen {

class JitMemoryBlock {
public:
    JitMemoryBlock() = default;
    explicit JitMemoryBlock(size_t size);
    ~JitMemoryBlock();

    JitMemoryBlock(const JitMemoryBlock&) = delete;
    JitMemoryBlock& operator=(const JitMemoryBlock&) = delete;
    JitMemoryBlock(JitMemoryBlock&& other) noexcept;
    JitMemoryBlock& operator=(JitMemoryBlock&& other) noexcept;

    uint8_t* data() noexcept { return ptr_; }
    const uint8_t* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool is_valid() const noexcept { return ptr_ != nullptr; }

    void make_executable();
    void make_read_write();
    void reset();

private:
    uint8_t* ptr_ = nullptr;
    size_t size_ = 0;
};

class JitExecutionEngine {
public:
    explicit JitExecutionEngine(const Target& target);
    JitExecutionEngine();
    ~JitExecutionEngine();

    JitExecutionEngine(const JitExecutionEngine&) = delete;
    JitExecutionEngine& operator=(const JitExecutionEngine&) = delete;
    JitExecutionEngine(JitExecutionEngine&&) noexcept;
    JitExecutionEngine& operator=(JitExecutionEngine&&) noexcept;

    // Register host external function/symbol
    void register_external_symbol(std::string_view name, void* address);

    // Compilation & loading
    bool compile_and_load(const Module& mod);
    bool load_object(const object::ObjectFile& obj);

    // Function/symbol lookup
    void* get_symbol_address(std::string_view name) const;

    template <typename FuncPtr>
    FuncPtr get_function_ptr(std::string_view name) const {
        return reinterpret_cast<FuncPtr>(get_symbol_address(name));
    }

    // Dynamic invocation helper using RuntimeValue
    RuntimeValue invoke(std::string_view name, const std::vector<RuntimeValue>& args);
    RuntimeValue invoke(std::string_view name);

private:
    Target target_;
    JitMemoryBlock code_mem_;

    std::unordered_map<std::string, void*> symbol_table_;
    std::unordered_map<std::string, void*> external_symbols_;
    std::unordered_map<std::string, std::pair<Type, std::vector<Type>>> function_signatures_;

    // Windows SEH registration tracking
    void* pdata_table_ = nullptr;
    size_t pdata_count_ = 0;
    uintptr_t code_base_ = 0;

    void register_seh_tables(const object::ObjectFile& obj, uint8_t* base_ptr);
    void unregister_seh_tables();
};

} // namespace brass::codegen
