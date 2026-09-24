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
// Windows (`host_runtime.dll`), a soname on ELF (`libhost_runtime.so`),
// an install name on Mach-O (`@rpath/libhost_runtime.dylib`).
struct ImportLibrary {
    std::string library;
    std::vector<std::string> symbols;
};

} // namespace brass::target
