#pragma once

#include <brass/brass.hpp>
#include <string_view>
#include <cstdint>
#include <vector>

namespace brass::test {

void generate_fuzz_expr(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_loops(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_gc(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_speculation(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_memory(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_patching(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_matrix_i64(Module& mod, std::string_view fn_name, uint64_t seed);
void generate_fuzz_matrix_f64(Module& mod, std::string_view fn_name, uint64_t seed);

} // namespace brass::test
