#include <brass/brass.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "brass-opt - Brass MIR Optimizer and Compiler Tool (v" << brass::version_string() << ")\n"
              << "Usage: " << prog << " [options] <input-file>\n\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  -v, --verify          Parse and verify MIR module\n"
              << "  -p, --print           Parse, verify, and print canonical MIR\n"
              << "  --check-roundtrip     Assert byte-identical parse(print(x)) roundtrip\n"
              << "  -o <file>             Write output to <file> instead of stdout\n";
}

bool read_file(const std::string& path, std::string& content) {
    if (path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        content = ss.str();
        return true;
    }
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
    return true;
}

bool write_file(const std::string& path, const std::string& content) {
    if (path.empty() || path == "-") {
        std::cout << content;
        return true;
    }
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    std::string input_file;
    std::string output_file;
    bool verify_only = false;
    bool print_canonical = false;
    bool check_roundtrip = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verify") {
            verify_only = true;
        } else if (arg == "-p" || arg == "--print") {
            print_canonical = true;
        } else if (arg == "--check-roundtrip") {
            check_roundtrip = true;
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: -o requires an output file argument\n";
                return 1;
            }
        } else if (arg.starts_with("-o")) {
            output_file = arg.substr(2);
        } else if (!arg.empty() && arg[0] == '-') {
            if (arg == "-") {
                input_file = "-";
            } else {
                std::cerr << "Error: Unknown option '" << arg << "'\n";
                return 1;
            }
        } else {
            input_file = arg;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified.\n";
        return 1;
    }

    std::string source_text;
    if (!read_file(input_file, source_text)) {
        std::cerr << "Error: Could not read input file '" << input_file << "'\n";
        return 1;
    }

    brass::DiagnosticReporter diag;
    auto mod = brass::parse_module(source_text, &diag, input_file);
    if (!mod || diag.has_errors()) {
        std::cerr << diag.format_all();
        return 1;
    }

    bool ok = brass::verify_module(*mod, &diag);
    if (!ok || diag.has_errors()) {
        std::cerr << diag.format_all();
        return 1;
    }

    if (check_roundtrip) {
        std::string canonical1 = brass::to_string(*mod);
        brass::DiagnosticReporter rt_diag;
        auto mod2 = brass::parse_module(canonical1, &rt_diag, "<canonical-roundtrip>");
        if (!mod2 || rt_diag.has_errors()) {
            std::cerr << "Roundtrip parse failed:\n" << rt_diag.format_all() << "\n";
            return 1;
        }

        bool rt_ok = brass::verify_module(*mod2, &rt_diag);
        if (!rt_ok || rt_diag.has_errors()) {
            std::cerr << "Roundtrip verify failed:\n" << rt_diag.format_all() << "\n";
            return 1;
        }

        std::string canonical2 = brass::to_string(*mod2);
        if (canonical1 != canonical2) {
            std::cerr << "Roundtrip mismatch: canonical representation is not byte-identical!\n";
            std::cerr << "--- First Canon ---\n" << canonical1
                      << "--- Second Canon ---\n" << canonical2 << "\n";
            return 1;
        }

        std::cout << "Roundtrip verified: byte-identical canonical representation ("
                  << canonical1.size() << " bytes)\n";
        return 0;
    }

    if (verify_only && !print_canonical) {
        std::cout << "Module '" << mod->name() << "' verified successfully ("
                  << mod->function_count() << " functions).\n";
        return 0;
    }

    std::string canonical = brass::to_string(*mod);
    if (!write_file(output_file, canonical)) {
        std::cerr << "Error: Could not write output file '" << output_file << "'\n";
        return 1;
    }

    return 0;
}
