#pragma once

#include <brass/verify/bv_solver.hpp>
#include <cstdint>
#include <vector>

#ifndef BRASS_VERIFY_UNSAT_SAMPLES
#define BRASS_VERIFY_UNSAT_SAMPLES 1
#endif

namespace brass::verify {

inline constexpr uint32_t shift_mask(uint32_t w) noexcept {
    if (w <= 1) return 0;
    uint32_t p = 1;
    while (p < w) p <<= 1;
    return p - 1;
}

bool try_concrete_search(
    const BVContext& ctx,
    BVNodeId bad_cond,
    std::vector<uint64_t>& out_model
);

void debug_verify_unsat_samples(
    const BVContext& ctx,
    BVNodeId bad_cond
);

} // namespace brass::verify
