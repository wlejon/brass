// brass_throw / brass_rethrow entry stubs for GCC/Clang on x86-64.
//
// The frame walker in brass_throw_impl needs the callee-saved registers exactly
// as the throwing JIT frame left them. A C++ function body cannot promise that:
// the compiler may reuse a callee-saved register (e.g. rdi for a `rep stos`
// zero-init on Win64) before any inline asm gets to read it. So the capture is
// done here in hand-written asm, before anything else touches the registers.
//
// Both stubs hand brass_throw_impl the throwing frame itself (its rbp and the
// return address into it), so the walker also restores whatever callee-saved
// registers that frame spilled. The MSVC build has the same stubs in
// src/gc/gc_msvc_x64.asm.

#include <brass/runtime/exception.hpp>

#if defined(__x86_64__) || defined(_M_X64)

// brass_rethrow re-raises the pending exception; the stubs fetch it through
// this plain C entry point so no C++ ABI details leak into the asm.
extern "C" uint64_t brass_current_exception_bits() noexcept {
    return brass::runtime::brass_get_current_exception().raw();
}

#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))

#if defined(_WIN32)

// Win64: val in rcx. Frame: push rbp; 256 bytes = 32 shadow + SavedRegisters
// (216) rounded up to keep rsp 16-byte aligned at the call.
// SEH unwind info is required because brass_throw_impl may fall back to a C++
// throw that has to unwind through this frame.
__asm__(
    ".text\n"
    ".globl brass_throw\n"
    ".def brass_throw; .scl 2; .type 32; .endef\n"
    ".seh_proc brass_throw\n"
    "brass_throw:\n"
    "    pushq %rbp\n"
    "    .seh_pushreg %rbp\n"
    "    movq %rsp, %rbp\n"
    "    .seh_setframe %rbp, 0\n"
    "    subq $256, %rsp\n"
    "    .seh_stackalloc 256\n"
    "    .seh_endprologue\n"
    "    movq %r15, 32(%rsp)\n"
    "    movq %r14, 40(%rsp)\n"
    "    movq %r13, 48(%rsp)\n"
    "    movq %r12, 56(%rsp)\n"
    "    movq %rdi, 64(%rsp)\n"
    "    movq %rsi, 72(%rsp)\n"
    "    movq %rbx, 80(%rsp)\n"
    "    leaq 32(%rsp), %rdx\n"
    "    movq 0(%rbp), %r8\n"
    "    movq 8(%rbp), %r9\n"
    "    call brass_throw_impl\n"
    "    ud2\n"
    ".seh_endproc\n"
    "\n"
    ".globl brass_rethrow\n"
    ".def brass_rethrow; .scl 2; .type 32; .endef\n"
    ".seh_proc brass_rethrow\n"
    "brass_rethrow:\n"
    "    subq $40, %rsp\n"
    "    .seh_stackalloc 40\n"
    "    .seh_endprologue\n"
    "    call brass_current_exception_bits\n"
    "    addq $40, %rsp\n"
    "    movq %rax, %rcx\n"
    // Tail jump: brass_throw must see the JIT frame's return address, not ours.
    "    jmp brass_throw\n"
    ".seh_endproc\n"
);

#else

#if defined(__APPLE__)
#define BRASS_ASM_SYM(name) "_" #name
#define BRASS_ASM_CALL(name) "_" #name
#define BRASS_ASM_TYPE(name) ""
#define BRASS_ASM_SIZE(name) ""
#else
#define BRASS_ASM_SYM(name) #name
#define BRASS_ASM_CALL(name) #name "@PLT"
#define BRASS_ASM_TYPE(name) ".type " #name ", @function\n"
#define BRASS_ASM_SIZE(name) ".size " #name ", .-" #name "\n"
#endif

// SysV: val in rdi. brass_throw_impl(val=rdi, regs=rsi, caller_rbp=rdx,
// caller_ip=rcx). 224 bytes holds SavedRegisters (216) and keeps rsp aligned.
// CFI lets a fallback C++ throw unwind through the stub.
__asm__(
    ".text\n"
    ".globl " BRASS_ASM_SYM(brass_throw) "\n"
    BRASS_ASM_TYPE(brass_throw)
    BRASS_ASM_SYM(brass_throw) ":\n"
    "    .cfi_startproc\n"
    "    pushq %rbp\n"
    "    .cfi_def_cfa_offset 16\n"
    "    .cfi_offset %rbp, -16\n"
    "    movq %rsp, %rbp\n"
    "    .cfi_def_cfa_register %rbp\n"
    "    subq $224, %rsp\n"
    "    movq %r15, 0(%rsp)\n"
    "    movq %r14, 8(%rsp)\n"
    "    movq %r13, 16(%rsp)\n"
    "    movq %r12, 24(%rsp)\n"
    "    movq %rdi, 32(%rsp)\n"
    "    movq %rsi, 40(%rsp)\n"
    "    movq %rbx, 48(%rsp)\n"
    "    movq %rsp, %rsi\n"
    "    movq 0(%rbp), %rdx\n"
    "    movq 8(%rbp), %rcx\n"
    "    call " BRASS_ASM_CALL(brass_throw_impl) "\n"
    "    ud2\n"
    "    .cfi_endproc\n"
    BRASS_ASM_SIZE(brass_throw)
    "\n"
    ".globl " BRASS_ASM_SYM(brass_rethrow) "\n"
    BRASS_ASM_TYPE(brass_rethrow)
    BRASS_ASM_SYM(brass_rethrow) ":\n"
    "    .cfi_startproc\n"
    "    subq $8, %rsp\n"
    "    .cfi_def_cfa_offset 16\n"
    "    call " BRASS_ASM_CALL(brass_current_exception_bits) "\n"
    "    addq $8, %rsp\n"
    "    .cfi_def_cfa_offset 8\n"
    "    movq %rax, %rdi\n"
    // Tail jump: brass_throw must see the JIT frame's return address, not ours.
    "    jmp " BRASS_ASM_CALL(brass_throw) "\n"
    "    .cfi_endproc\n"
    BRASS_ASM_SIZE(brass_rethrow)
);

#undef BRASS_ASM_SYM
#undef BRASS_ASM_CALL
#undef BRASS_ASM_TYPE
#undef BRASS_ASM_SIZE

#endif

#endif
