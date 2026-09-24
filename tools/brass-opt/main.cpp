#include <brass/brass.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/runtime/type_feedback.hpp>
#include "opt_actions.hpp"
#include "opt_cli.hpp"
#include "opt_pipeline.hpp"
#include "opt_reports.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

using brass::opt::OptCli;

bool read_file(const std::string& path, std::string& content) {
    std::ostringstream ss;
    if (path == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream file(path, std::ios::in | std::ios::binary);
        if (!file.is_open()) return false;
        ss << file.rdbuf();
    }
    content = ss.str();
    return true;
}

bool write_file(const std::string& path, const std::string& content) {
    if (path.empty() || path == "-") {
        std::cout << content;
        return true;
    }
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) return false;
    file << content;
    return true;
}

bool verified(const brass::Module& mod, const char* after) {
    brass::DiagnosticReporter diag;
    if (brass::verify_module(mod, &diag) && !diag.has_errors()) return true;
    std::cerr << "Verification failed after " << after << ":\n" << diag.format_all();
    return false;
}

std::unique_ptr<brass::Module> load_module(const std::string& input_file) {
    std::string source_text;
    if (!read_file(input_file, source_text)) {
        std::cerr << "Error: Could not read input file '" << input_file << "'\n";
        return nullptr;
    }
    brass::DiagnosticReporter diag;
    auto mod = brass::parse_module(source_text, &diag, input_file);
    if (!mod || diag.has_errors() || !brass::verify_module(*mod, &diag) || diag.has_errors()) {
        std::cerr << diag.format_all();
        return nullptr;
    }
    // Coroutine bodies execute only in lowered form in every tier, and the
    // optimizer expects them lowered, so lower them first (as the IL
    // translator does).
    brass::CoroTransformPass().run_on_module(*mod);
    if (!verified(*mod, "coroutine lowering")) return nullptr;
    return mod;
}

// Profile loading, profile-guided optimization, the branch probability
// report and instrumentation. Returns false on an error already reported.
bool run_pgo(const OptCli& cli, brass::Module& mod) {
    std::unique_ptr<brass::pgo::ProfileData> profile;
    if (!cli.pgo_use_file.empty()) {
        std::string err;
        profile = brass::pgo::ProfileData::read_from_file(cli.pgo_use_file, &err);
        if (!profile) {
            std::cerr << "Error: Could not load profile from '" << cli.pgo_use_file << "': " << err << "\n";
            return false;
        }
        if (!cli.dump_branch_probabilities) {
            brass::pgo::optimize_module_pgo(mod, *profile);
            if (!verified(mod, "PGO optimization")) return false;
        }
    }
    if (cli.dump_branch_probabilities) {
        if (!profile) {
            std::cerr << "Error: --dump-branch-probabilities requires --pgo-use=<file.bprof>\n";
            return false;
        }
        brass::opt::report_branch_probabilities(mod, *profile, std::cout);
    }
    if (cli.enable_pgo_instrument) {
        brass::pgo::instrument_module(mod);
        if (!verified(mod, "PGO instrumentation")) return false;
    }
    return true;
}

// The outputs after optimization: debug-info tools, object or shared
// library, running a function, the roundtrip check, or printing the module.
int emit(const OptCli& cli, brass::Module& mod) {
    const bool other_output = cli.compile_object || cli.emit_shared || !cli.run_fn.empty();
    if (cli.dump_debug_lines) {
        brass::execute_dump_debug_lines(mod);
        if (!other_output) return 0;
    }
    if (!cli.symbolize_offset_arg.empty()) {
        return brass::execute_symbolize_offset(mod, cli.symbolize_offset_arg) ? 0 : 1;
    }
    if (!cli.emit_source_map_file.empty()) {
        if (!brass::execute_emit_source_map(mod, cli.emit_source_map_file)) return 1;
        if (!other_output) return 0;
    }
    if (cli.compile_object) {
        brass::CompileObjectOptions o;
        o.obj_format = cli.obj_format;
        o.enable_schedule_insns = cli.enable_schedule_insns;
        o.enable_software_pipeline = cli.enable_software_pipeline;
        o.output_file = cli.output_file;
        o.input_file = cli.input_file;
        return brass::execute_compile_object(mod, o) ? 0 : 1;
    }
    if (cli.emit_shared) {
        brass::EmitSharedOptions o;
        o.obj_format = cli.obj_format;
        o.shared_output_file = cli.shared_output_file;
        o.output_file = cli.output_file;
        o.input_file = cli.input_file;
        return brass::execute_emit_shared(mod, o) ? 0 : 1;
    }
    if (!cli.run_fn.empty()) {
        brass::RunFunctionOptions o;
        o.run_fn = cli.run_fn;
        o.run_arg_strings = cli.run_arg_strings;
        o.use_jit = cli.use_jit;
        o.use_baseline_jit = cli.use_baseline_jit;
        o.gc_stress = cli.gc_stress;
        o.enable_osr = cli.enable_osr;
        o.osr_threshold = cli.osr_threshold;
        o.enable_schedule_insns = cli.enable_schedule_insns;
        o.enable_software_pipeline = cli.enable_software_pipeline;
        o.dump_tiering_stats = cli.dump_tiering_stats;
        o.enable_background_compile = cli.enable_background_compile;
        o.jit_threads = cli.jit_threads;
        o.dump_jit_thread_stats = cli.dump_jit_thread_stats;
        return brass::execute_run_function(mod, o) ? 0 : 1;
    }
    if (cli.check_roundtrip) return brass::opt::check_roundtrip(mod, std::cout, std::cerr);
    if (cli.verify_only && !cli.print_canonical) {
        std::cout << "Module '" << mod.name() << "' verified successfully (" << mod.function_count()
                  << " functions).\n";
        return 0;
    }
    if (!write_file(cli.output_file, brass::to_string(mod))) {
        std::cerr << "Error: Could not write output file '" << cli.output_file << "'\n";
        return 1;
    }
    if (cli.dump_tfv_stats) brass::runtime::FeedbackRegistry::instance().dump_stats(std::cout);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    OptCli cli;
    switch (brass::opt::parse_command_line(argc, argv, cli, std::cout, std::cerr)) {
        case brass::opt::ParseOutcome::ExitOk: return 0;
        case brass::opt::ParseOutcome::ExitError: return 1;
        case brass::opt::ParseOutcome::Run: break;
    }

    brass::opt::OptStats stats;
    const brass::Pipeline pipeline = brass::opt::build_pipeline(cli, stats, std::cout);
    if (cli.print_pipeline) {
        for (const std::string& name : pipeline.names()) std::cout << name << "\n";
        return 0;
    }
    if (cli.input_file.empty()) {
        std::cerr << "Error: No input file specified.\n";
        return 1;
    }
    std::unique_ptr<brass::Module> mod = load_module(cli.input_file);
    if (!mod) return 1;
    if (!run_pgo(cli, *mod)) return 1;

    if (cli.run_escape_analysis) brass::opt::report_escape_analysis(*mod, std::cout);
    if (cli.enable_partial_escape && !cli.enable_allocation_sinking) {
        brass::opt::report_partial_escape(*mod, std::cout);
    }
    if (cli.run_alias_analysis) brass::opt::report_alias_analysis(*mod, std::cout);

    brass::PassPipelineHooks hooks;
    hooks.after_pass = [&mod](std::string_view step) { return verified(*mod, std::string(step).c_str()); };
    if (!brass::run_pipeline(*mod, pipeline, hooks).completed) return 1;

    brass::opt::dump_stats(cli, stats, std::cout);
    return emit(cli, *mod);
}
