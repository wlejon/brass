#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace brass {
class Module;
class Function;
} // namespace brass

namespace brass::target {

struct MslOptions {
    uint32_t msl_major = 2;
    uint32_t msl_minor = 4;
    uint32_t workgroup_x = 256;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    bool fast_math = true;
};

struct MslKernel {
    std::string source;
    std::string entry_name;
    uint32_t num_buffers = 0;
    std::vector<uint32_t> scalar_param_offsets;
    uint32_t scalar_params_bytes = 0;
    uint32_t workgroup_x = 256;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    bool uses_shared_memory = false;
    uint32_t shared_memory_bytes = 0;
};

class MslTarget {
public:
    static MslKernel compile(const Function& fn, const MslOptions& opts = {});
    static std::string compile_module(const Module& mod, const MslOptions& opts = {});

    MslKernel compile_function(const Function& fn, const MslOptions& opts = {}) const {
        return compile(fn, opts);
    }
};

} // namespace brass::target
