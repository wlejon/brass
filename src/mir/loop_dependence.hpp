#pragma once

#include <brass/mir/loop_nest.hpp>
#include <brass/mir/function.hpp>

namespace brass {

void analyze_nest_memory_accesses(Function& fn, LoopNest& nest);
void compute_nest_dependences(Function& fn, LoopNest& nest);
bool check_nest_tiling_legality(const LoopNest& nest);
bool check_nest_interchange_legality(const LoopNest& nest, size_t level_a, size_t level_b);
bool detect_matrix_multiply_pattern(const LoopNest& nest);

} // namespace brass
