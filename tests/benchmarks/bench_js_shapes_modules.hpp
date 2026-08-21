#pragma once

#include <brass/brass.hpp>
#include <memory>

namespace brass::bench {

std::unique_ptr<Module> build_nanbox_tag_test_module();
std::unique_ptr<Module> build_shape_guard_module();
std::unique_ptr<Module> build_patchable_ic_module();
std::unique_ptr<Module> build_gc_alloc_loop_module();

} // namespace brass::bench
