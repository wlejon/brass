#pragma once

#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/function.hpp>
#include <brass/target/target.hpp>
#include <functional>
#include <memory>
#include <string_view>

namespace brass::runtime {
class TieringRegistry;
}

namespace brass::aarch64 {

using BaselineSymbolResolver = std::function<void*(std::string_view)>;

// `tiering` is the registry of the program the code belongs to (null: the
// default program's): the function's TieringFeedback is resolved from it at
// compile time and baked into the invocation hook, as the x64 baseline does.
// `lazy` supplies the stub a func_addr yields for a symbol that does not
// resolve at compile time (a sibling compiled later); the compiled function
// keeps the table alive.
codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver = nullptr,
    runtime::TieringRegistry* tiering = nullptr,
    std::shared_ptr<codegen::LazySymbolTable> lazy = nullptr
);

} // namespace brass::aarch64
