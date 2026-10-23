#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>

namespace brass {

struct SuperoptStats {
    uint32_t masks_elided = 0;
    uint32_t extensions_elided = 0;
    uint32_t dags_synthesized = 0;
    uint32_t constants_synthesized = 0;
    uint32_t lir_windows_optimized = 0;
    uint32_t magic_divs_reduced = 0;
    uint32_t indices_narrowed = 0;
    uint32_t solver_queries = 0;
};

struct SuperoptOptions {
    uint32_t max_window_depth = 5;
    uint32_t max_sat_conflicts = 2000;
    uint64_t lir_live_out_gpr_mask = 0xFFFFULL;
    bool lir_check_flags = false;
    bool enable_gpu_superopt = false;
    SuperoptStats* stats = nullptr;
};

// Runs solver-verified MIR superoptimization on a single function:
//   1. Redundant bitmask elision (`and(x, mask) -> x`, `or(x, mask) -> x`) proven under path/range constraints.
//   2. Redundant sign/zero-extension & truncation elision (`sext_i64(trunc_i32(x)) -> x`, `zext_i64(trunc_i32(x)) -> x`).
//   3. Solver-proven multi-instruction DAG synthesis & strength reduction (`sdiv(x, 2^k) -> lshr(x, k)` and
//      `smod(x, 2^k) -> and(x, 2^k - 1)` when `x >= 0` is proven, boolean/bitwise factoring, identity collapse).
bool run_mir_superopt(
    Function& fn,
    Module& mod,
    const SuperoptOptions& opts = {});

bool run_mir_superopt(
    Module& mod,
    const SuperoptOptions& opts = {});

// Runs solver-verified peephole superoptimization on a straight-line x64 LIR basic block.
// Every candidate rewrite is verified bit-exact by `verify::verify_x64_lir_block_equivalence` before acceptance.
bool run_x64_lir_superopt(
    codegen::LirBlock& block,
    const SuperoptOptions& opts = {});

} // namespace brass

#include <brass/mir/gpu_superopt.hpp>
