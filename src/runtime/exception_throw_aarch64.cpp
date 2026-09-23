// brass_throw / brass_rethrow / landing-pad jump for GCC/Clang on AArch64.
//
// As on x86-64 (exception_throw_x64.cpp), the frame walker needs the
// callee-saved registers exactly as the throwing JIT frame left them, which a
// C++ body cannot promise: its prologue may already have reused x19..x28 or
// d8..d15. GCC also ignores __attribute__((naked)) on AArch64, so these are
// file-scope asm.
//
// SavedRegisters layout (runtime/exception.hpp): 7 x86-64 words, then
// x19..x28, fp, lr at 56, then d8..d15 at 152; 216 bytes in all.

#include <brass/runtime/exception.hpp>
#include "core/asm_symbol.hpp"

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__aarch64__) || defined(_M_ARM64))

extern "C" uint64_t brass_current_exception_bits() noexcept {
    return brass::runtime::brass_get_current_exception().raw();
}

#if defined(__APPLE__)
#define BRASS_A64_CALL(name) "_" #name
#else
#define BRASS_A64_CALL(name) #name
#endif

// brass_throw(val = x0): frame record + SavedRegisters (216, rounded to 224).
// brass_throw_impl(val = x0, regs = x1, caller_fp = x2, caller_ip = x3), with
// the caller being the throwing frame: the record this stub just pushed holds
// its fp and the return address into it.
__asm__(
    ".text\n"
    BRASS_ASM_FN_BEGIN(brass_throw)
    "    .cfi_startproc\n"
    "    stp x29, x30, [sp, #-240]!\n"
    "    .cfi_def_cfa_offset 240\n"
    "    .cfi_offset x29, -240\n"
    "    .cfi_offset x30, -232\n"
    "    mov x29, sp\n"
    "    .cfi_def_cfa_register x29\n"
    "    add x1, sp, #16\n"
    "    stp x19, x20, [x1, #56]\n"
    "    stp x21, x22, [x1, #72]\n"
    "    stp x23, x24, [x1, #88]\n"
    "    stp x25, x26, [x1, #104]\n"
    "    stp x27, x28, [x1, #120]\n"
    "    stp d8, d9, [x1, #152]\n"
    "    stp d10, d11, [x1, #168]\n"
    "    stp d12, d13, [x1, #184]\n"
    "    stp d14, d15, [x1, #200]\n"
    "    ldp x2, x3, [x29]\n"
    "    bl " BRASS_A64_CALL(brass_throw_impl) "\n"
    "    brk #0\n"
    "    .cfi_endproc\n"
    BRASS_ASM_FN_END(brass_throw)
    "\n"
    BRASS_ASM_FN_BEGIN(brass_rethrow)
    "    .cfi_startproc\n"
    "    stp x29, x30, [sp, #-16]!\n"
    "    .cfi_def_cfa_offset 16\n"
    "    .cfi_offset x29, -16\n"
    "    .cfi_offset x30, -8\n"
    "    mov x29, sp\n"
    "    bl " BRASS_A64_CALL(brass_current_exception_bits) "\n"
    "    ldp x29, x30, [sp], #16\n"
    "    .cfi_def_cfa_offset 0\n"
    "    .cfi_restore x29\n"
    "    .cfi_restore x30\n"
    // Tail branch: brass_throw must see the JIT frame's return address.
    "    b " BRASS_A64_CALL(brass_throw) "\n"
    "    .cfi_endproc\n"
    BRASS_ASM_FN_END(brass_rethrow)
    "\n"
    // brass_a64_jump_to_landing_pad(ip = x0, fp = x1, sp = x2, val = x3,
    // regs = x4): restores the callee-saved registers the walker collected,
    // installs the landing frame and enters the pad with the value in x0.
    BRASS_ASM_FN_BEGIN(brass_a64_jump_to_landing_pad)
    "    mov x16, x0\n"
    "    ldp x19, x20, [x4, #56]\n"
    "    ldp x21, x22, [x4, #72]\n"
    "    ldp x23, x24, [x4, #88]\n"
    "    ldp x25, x26, [x4, #104]\n"
    "    ldp x27, x28, [x4, #120]\n"
    "    ldp d8, d9, [x4, #152]\n"
    "    ldp d10, d11, [x4, #168]\n"
    "    ldp d12, d13, [x4, #184]\n"
    "    ldp d14, d15, [x4, #200]\n"
    "    mov x29, x1\n"
    "    mov sp, x2\n"
    "    mov x0, x3\n"
    "    br x16\n"
    BRASS_ASM_FN_END(brass_a64_jump_to_landing_pad)
);

extern "C" [[noreturn]] void brass_a64_jump_to_landing_pad(
    void* landing_pad_ip, void* target_fp, void* target_sp, uint64_t val_bits,
    const brass::runtime::SavedRegisters* regs);

namespace brass::runtime {

[[noreturn]] void brass_jump_to_landing_pad(
    void* landing_pad_ip,
    void* target_rbp,
    void* target_rsp,
    HostValue val,
    const SavedRegisters& regs
) {
    brass_a64_jump_to_landing_pad(landing_pad_ip, target_rbp, target_rsp, val.raw(), &regs);
}

} // namespace brass::runtime

#endif
