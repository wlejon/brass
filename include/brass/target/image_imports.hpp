#pragma once

#include <string>
#include <vector>

namespace brass::target {

// One library a linked image imports from, and the symbols it may take from
// it. Every undefined symbol a relocation names must be found in exactly one
// of these or the writer refuses the image naming the symbol: an import is
// never guessed and an unresolved reference is never left pointing at zero.
//
// `library` is spelled the way the OS loader wants it: a DLL file name on
// Windows (`bronze_runtime_shared.dll`), a soname on ELF
// (`libbronze_runtime_shared.so`), an install name on Mach-O
// (`@rpath/libbronze_runtime_shared.dylib`).
struct ImportLibrary {
    std::string library;
    std::vector<std::string> symbols;
};

} // namespace brass::target
