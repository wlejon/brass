#pragma once

// Directives for defining a global function in file-scope asm (GCC/Clang),
// per object format: Mach-O prefixes C symbols with '_', ELF wants
// .type/.size, COFF wants .def. Used by the hand-written AArch64 stubs
// (runtime/exception_throw_aarch64.cpp, codegen/jit_invoke.cpp).

#if defined(__APPLE__)
#define BRASS_ASM_FN_SYM(name) "_" #name
#define BRASS_ASM_FN_BEGIN(name) ".globl _" #name "\n.p2align 2\n_" #name ":\n"
#define BRASS_ASM_FN_END(name) ""
#elif defined(_WIN32)
#define BRASS_ASM_FN_SYM(name) #name
#define BRASS_ASM_FN_BEGIN(name) \
    ".globl " #name "\n.def " #name "; .scl 2; .type 32; .endef\n.p2align 2\n" #name ":\n"
#define BRASS_ASM_FN_END(name) ""
#else
#define BRASS_ASM_FN_SYM(name) #name
#define BRASS_ASM_FN_BEGIN(name) ".globl " #name "\n.type " #name ", %function\n.p2align 2\n" #name ":\n"
#define BRASS_ASM_FN_END(name) ".size " #name ", .-" #name "\n"
#endif
