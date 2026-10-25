// The stack switch of gc::Fiber off Windows (fiber.cpp), as file-scope asm.
// Windows x64's is gc_msvc_x64.asm's (which also keeps the TIB's stack bounds
// right for the running stack); Windows on ARM64 switches with the system's
// fibers.
//
// brass_fiber_swap(save_sp, load_sp) is an ordinary call to its caller: it
// pushes the callee-saved registers, stores the stack pointer through
// save_sp, takes load_sp as the stack pointer and pops what the other
// stack's own call pushed, returning into that stack's caller. fiber.cpp
// reads the saved frame to start a collection's walk of a switched-out stack
// (position_of), so the layouts here and there are one contract.
//
// brass_fiber_start is where a new stack's first switch returns to (the
// frame Fiber::create lays out). Its CFI says the return address is
// undefined, which ends every unwind there: a C++ exception, the GC's walk,
// a debugger's backtrace.

#if !defined(_WIN32)

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

#if defined(__x86_64__)

// Saved frame, from the stored stack pointer: mxcsr (4 bytes) and the x87
// control word, r15, r14, r13, r12, rbx, rbp, the return address.
__asm__(
    ".text\n"
    ".globl " BRASS_ASM_SYM(brass_fiber_swap) "\n"
    BRASS_ASM_TYPE(brass_fiber_swap)
    ".p2align 4\n"
    BRASS_ASM_SYM(brass_fiber_swap) ":\n"
    "    pushq %rbp\n"
    "    pushq %rbx\n"
    "    pushq %r12\n"
    "    pushq %r13\n"
    "    pushq %r14\n"
    "    pushq %r15\n"
    "    subq $8, %rsp\n"
    "    stmxcsr (%rsp)\n"
    "    fnstcw 4(%rsp)\n"
    "    movq %rsp, (%rdi)\n"
    "    movq %rsi, %rsp\n"
    "    ldmxcsr (%rsp)\n"
    "    fldcw 4(%rsp)\n"
    "    addq $8, %rsp\n"
    "    popq %r15\n"
    "    popq %r14\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rbx\n"
    "    popq %rbp\n"
    "    ret\n"
    BRASS_ASM_SIZE(brass_fiber_swap)
    "\n"
    ".globl " BRASS_ASM_SYM(brass_fiber_start) "\n"
    BRASS_ASM_TYPE(brass_fiber_start)
    ".p2align 4\n"
    BRASS_ASM_SYM(brass_fiber_start) ":\n"
    "    .cfi_startproc\n"
    "    .cfi_undefined %rip\n"
    "    xorl %ebp, %ebp\n"
    "    movq %r12, %rdi\n"
    "    call " BRASS_ASM_CALL(brass_fiber_main) "\n"
    "    ud2\n"
    "    .cfi_endproc\n"
    BRASS_ASM_SIZE(brass_fiber_start)
);

#elif defined(__aarch64__)

// Saved frame, 176 bytes from the stored stack pointer: x19..x28, x29, x30,
// d8..d15, fpcr.
__asm__(
    ".text\n"
    ".globl " BRASS_ASM_SYM(brass_fiber_swap) "\n"
    BRASS_ASM_TYPE(brass_fiber_swap)
    ".p2align 4\n"
    BRASS_ASM_SYM(brass_fiber_swap) ":\n"
    "    sub sp, sp, #176\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    stp d8, d9, [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mrs x9, fpcr\n"
    "    str x9, [sp, #160]\n"
    "    mov x9, sp\n"
    "    str x9, [x0]\n"
    "    mov sp, x1\n"
    "    ldr x9, [sp, #160]\n"
    "    msr fpcr, x9\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    ldp d8, d9, [sp, #96]\n"
    "    ldp d10, d11, [sp, #112]\n"
    "    ldp d12, d13, [sp, #128]\n"
    "    ldp d14, d15, [sp, #144]\n"
    "    add sp, sp, #176\n"
    "    ret\n"
    BRASS_ASM_SIZE(brass_fiber_swap)
    "\n"
    ".globl " BRASS_ASM_SYM(brass_fiber_start) "\n"
    BRASS_ASM_TYPE(brass_fiber_start)
    ".p2align 4\n"
    BRASS_ASM_SYM(brass_fiber_start) ":\n"
    "    .cfi_startproc\n"
    "    .cfi_undefined x30\n"
    "    mov x29, xzr\n"
    "    mov x0, x19\n"
    "    bl " BRASS_ASM_SYM(brass_fiber_main) "\n"
    "    brk #1\n"
    "    .cfi_endproc\n"
    BRASS_ASM_SIZE(brass_fiber_start)
);

#endif

#undef BRASS_ASM_SYM
#undef BRASS_ASM_CALL
#undef BRASS_ASM_TYPE
#undef BRASS_ASM_SIZE

#endif
