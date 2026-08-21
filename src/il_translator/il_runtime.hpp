#pragma once

#include <cstdint>

namespace brass::codegen {
class JitExecutionEngine;
}

namespace brass::il {

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit);

} // namespace brass::il
