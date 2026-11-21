#pragma once

#include "gpu_equiv_internal.hpp"
#include <vector>

namespace brass::verify {

struct DivConstraint {
    BVNodeId dividend = 0;
    BVNodeId divisor = 0;
};

std::vector<uint32_t> collect_buffer_dim_params(const GpuBufferSpec& bspec, const GpuSymEnv& env);

BVNodeId build_buffer_capacity_64(BVContext& ctx, const GpuBufferSpec& bspec, const GpuSymEnv& env);

bool try_prove_global_access_algebraic(
    BVContext& ctx,
    BVNodeId light_domain,
    BVNodeId guard_domain,
    BVNodeId active,
    BVNodeId byte_off,
    uint32_t access_bytes,
    const GpuBufferSpec* bspec,
    const GpuSymEnv& env,
    const std::vector<DivConstraint>& div_pairs,
    uint32_t max_conflicts);

} // namespace brass::verify
