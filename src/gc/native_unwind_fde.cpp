// The unwinder's FDE lookup for native_unwind.cpp, in a file of its own:
// _Unwind_Find_FDE is exported by libgcc and LLVM libunwind alike but
// declared by neither's <unwind.h> the same way (libgcc's is internal), so
// it is declared here, where no <unwind.h> can disagree with it.

#include <cstdint>

#if !defined(_WIN32) && !defined(__APPLE__) && (defined(__x86_64__) || defined(__aarch64__))

struct dwarf_eh_bases {
    void* tbase;
    void* dbase;
    void* func;
};

extern "C" const void* _Unwind_Find_FDE(void* pc, struct dwarf_eh_bases* bases);

namespace brass::detail {

// The FDE describing the code at pc, and the start of its function.
const void* find_fde(uintptr_t pc, uintptr_t& function_start) noexcept {
    dwarf_eh_bases bases{};
    const void* fde = _Unwind_Find_FDE(reinterpret_cast<void*>(pc), &bases);
    function_start = reinterpret_cast<uintptr_t>(bases.func);
    return fde;
}

} // namespace brass::detail

#endif
