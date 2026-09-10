#pragma once

#include <brass/mir/module.hpp>
#include <brass/interpreter/value.hpp>
#include <string>
#include <vector>

namespace brass {

struct CompileObjectOptions {
    std::string obj_format;
    bool enable_schedule_insns = true;
    bool enable_software_pipeline = false;
    std::string output_file;
    std::string input_file;
};

bool execute_compile_object(const Module& mod, const CompileObjectOptions& opts);

struct EmitSharedOptions {
    std::string obj_format;
    std::string shared_output_file;
    std::string output_file;
    std::string input_file;
};

bool execute_emit_shared(const Module& mod, const EmitSharedOptions& opts);

struct RunFunctionOptions {
    std::string run_fn;
    std::vector<std::string> run_arg_strings;
    bool use_jit = false;
    bool gc_stress = false;
    bool enable_osr = false;
    uint64_t osr_threshold = 100;
    bool enable_schedule_insns = true;
    bool enable_software_pipeline = false;
    bool dump_ic_stats = false;
    bool dump_tiering_stats = false;
    bool enable_background_compile = false;
    size_t jit_threads = 2;
    bool dump_jit_thread_stats = false;
};

bool execute_run_function(Module& mod, const RunFunctionOptions& opts);

} // namespace brass
