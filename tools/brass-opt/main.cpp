#include <brass/brass.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "brass-opt - Brass MIR Optimizer and Compiler (v" << brass::version_string() << ")\n";
    if (argc < 2) {
        std::cout << "Usage: brass-opt [options] <input-file>\n";
        return 0;
    }
    return 0;
}
