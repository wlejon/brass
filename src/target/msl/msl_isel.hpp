#pragma once

#include <brass/target/msl_target.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/types.hpp>
#include "../spirv/spirv_structurize.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::msl {

class MslEmitter {
public:
    struct KernelResult {
        std::string function_code;
        std::string params_struct_code;
        uint32_t num_buffers = 0;
        std::vector<uint32_t> scalar_param_offsets;
        uint32_t scalar_params_bytes = 0;
        bool uses_shared_memory = false;
        uint32_t shared_memory_bytes = 0;
        bool needs_bf16_helper = false;
        bool needs_sdot4_helper = false;
    };

    static KernelResult lower(const Function& fn, const target::MslOptions& opts);
};

} // namespace brass::msl
