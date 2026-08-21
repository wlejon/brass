#include <brass/brass.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/mir/f64_demote.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::il;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: brass-il <input.il> [--run] [--emit-mir] [--demote-stats] [-o <output.obj>] [--no-opt] [--no-demote] [--reassoc]\n";
        return 1;
    }

    std::string input_file;
    std::string output_obj;
    bool run_jit = false;
    bool emit_mir = false;
    bool raw_output = false;
    bool show_demote_stats = false;
    TranslatorOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--run") {
            run_jit = true;
        } else if (arg == "--emit-mir") {
            emit_mir = true;
        } else if (arg == "--demote-stats") {
            show_demote_stats = true;
        } else if (arg == "--raw-output") {
            raw_output = true;
        } else if (arg == "--no-opt") {
            options.enable_optimizations = false;
        } else if (arg == "--no-demote") {
            options.enable_f64_demote = false;
        } else if (arg == "--reassoc") {
            options.allow_fp_reassociation = true;
        } else if (arg == "-o" && i + 1 < argc) {
            output_obj = argv[++i];
        } else if (arg.rfind("-o", 0) == 0 && arg.size() > 2) {
            output_obj = arg.substr(2);
        } else if (arg[0] != '-') {
            input_file = arg;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input .il file specified.\n";
        return 1;
    }

    std::ifstream ifs(input_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Cannot open input file: " << input_file << "\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string il_source = buffer.str();

    DemoteStats demote_stats;
    if (show_demote_stats) {
        options.demote_stats = true;
        options.demote_stats_collector = &demote_stats;
    }

    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_source, options, &diag);
    if (!res.success || !res.module) {
        std::cerr << "Bronze IL Translation Failed:\n";
        std::cerr << diag.format_all() << "\n";
        if (!res.error_message.empty()) {
            std::cerr << res.error_message << "\n";
        }
        return 1;
    }

    if (show_demote_stats) {
        demote_stats.module_name = res.module->name();
        std::cout << demote_stats.format_report() << "\n";
    }

    if (emit_mir) {
        std::cout << to_string(*res.module) << "\n";
    }

    if (!output_obj.empty()) {
        HostEngine engine;
        if (!engine.compile_to_object(*res.module, output_obj)) {
            std::cerr << "Error: Failed to compile module to object format: " << output_obj << "\n";
            return 1;
        }
        std::cout << "[brass-il] Successfully wrote object to: " << output_obj << "\n";
    }

    if (run_jit || output_obj.empty()) {
        codegen::JitExecutionEngine jit;
        register_bronze_runtime_symbols(&jit);

        if (!jit.compile_and_load(*res.module)) {
            std::cerr << "Error: JIT compilation failed.\n";
            return 1;
        }

        Function* main_fn = res.module->get_function("main");
        if (main_fn) {
            if (main_fn->return_type() == Type::f64()) {
                auto fn_ptr = jit.get_function_ptr<double(*)()>("main");
                if (fn_ptr) {
                    double r = fn_ptr();
                    if (!raw_output) {
                        std::cout << "[brass-il] main() returned f64: " << r << "\n";
                    }
                }
            } else if (main_fn->return_type() == Type::i32()) {
                auto fn_ptr = jit.get_function_ptr<int32_t(*)()>("main");
                if (fn_ptr) {
                    int32_t r = fn_ptr();
                    if (!raw_output) {
                        std::cout << "[brass-il] main() returned i32: " << r << "\n";
                    }
                }
            } else if (main_fn->return_type() == Type::i64()) {
                auto fn_ptr = jit.get_function_ptr<int64_t(*)()>("main");
                if (fn_ptr) {
                    int64_t r = fn_ptr();
                    if (!raw_output) {
                        std::cout << "[brass-il] main() returned i64: " << r << "\n";
                    }
                }
            } else {
                auto fn_ptr = jit.get_function_ptr<void(*)()>("main");
                if (fn_ptr) {
                    fn_ptr();
                    if (!raw_output) {
                        std::cout << "[brass-il] main() executed (void).\n";
                    }
                }
            }
        }
    }

    return 0;
}
