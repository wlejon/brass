#pragma once

#include <brass/codegen/baseline_jit.hpp>
#include <brass/mir/function.hpp>
#include <brass/target/target.hpp>
#include <functional>
#include <string_view>

namespace brass::aarch64 {

using BaselineSymbolResolver = std::function<void*(std::string_view)>;

codegen::BaselineCompiledFunction compile_baseline_aarch64(
    const Function& fn,
    Target target,
    BaselineSymbolResolver resolver = nullptr
);

} // namespace brass::aarch64
