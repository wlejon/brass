#pragma once

#include <brass/vm/bytecode.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <memory>

namespace brass {

class BytecodeCompiler {
public:
    BytecodeCompiler() = default;
    ~BytecodeCompiler() = default;

    std::unique_ptr<BytecodeFunction> compile(const Function& fn);
    std::unique_ptr<BytecodeModule> compile(const Module& mod);
};

} // namespace brass
