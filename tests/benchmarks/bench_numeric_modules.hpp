#pragma once

#include <brass/brass.hpp>
#include <memory>

namespace brass::bench {

std::unique_ptr<Module> build_fib_module();
std::unique_ptr<Module> build_sieve_module();
std::unique_ptr<Module> build_collatz_module();
std::unique_ptr<Module> build_matmul_i64_module();
std::unique_ptr<Module> build_matmul_f64_module();
std::unique_ptr<Module> build_list_module();

} // namespace brass::bench
