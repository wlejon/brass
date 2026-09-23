#pragma once

#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/function.hpp>
#include <brass/target/target.hpp>
#include <functional>
#include <string_view>

namespace brass::runtime {
class TieringRegistry;
}

namespace brass::aarch64 {

using BaselineSymbolResolver = std::function<void*(std::string_view)>;

// `tiering` is the registry of the program the code belongs to (null: the
// default program's): the function's TieringFeedback is resolved from it at
// compile time and baked into the invocation hook, as the x64 baseline does.
codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver = nullptr,
    runtime::TieringRegistry* tiering = nullptr
);

} // namespace brass::aarch64
