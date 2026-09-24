#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <brass/gc/native_frames.hpp>
#include "core/asm_symbol.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace brass::codegen {

void partition_aarch64_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    AArch64InvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
) {
    out_args = AArch64InvokeArgs{};
    out_args.target_fn = target_fn;
    out_args.pinned_tls = runtime::host_pinned_tls_block();
    stack_words.clear();

    // The stack half of the convention every AArch64 tier reads (the tier-2
    // entry in aarch64_isel.cpp, the baseline in aarch64_baseline_emit_call.cpp):
    // an 8-byte slot per value (16, 16-aligned, for a vector), or on Apple
    // the value's natural size and alignment.
#if defined(__APPLE__)
    constexpr bool apple = true;
#else
    constexpr bool apple = false;
#endif
    std::vector<uint8_t> stack;

    size_t gpr_idx = 0;
    size_t fpr_idx = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        const Type* pt = (param_types && i < param_types->size()) ? &(*param_types)[i] : nullptr;
        const Type t = pt ? *pt : arg.type();
        const bool is_fpr = t.is_float() || t.is_vector();

        auto push_stack = [&](const uint8_t* bytes, size_t sz, size_t align, size_t slot) {
            const size_t at = (stack.size() + align - 1) & ~(align - 1);
            stack.resize(at + slot, 0);
            std::memcpy(stack.data() + at, bytes, std::min(sz, slot));
        };

        // A 256-bit vector is two 128-bit halves: two V registers, or the
        // last V register and the stack, or 32 bytes of stack
        // (aarch64_isel.cpp).
        if (t.is_v256() || arg.is_v256()) {
            const uint8_t* b = static_cast<const uint8_t*>(arg.v128_bytes());
            if (fpr_idx + 1 < 8) {
                std::memcpy(out_args.v[fpr_idx++], b, 16);
                std::memcpy(out_args.v[fpr_idx++], b + 16, 16);
            } else if (fpr_idx < 8) {
                std::memcpy(out_args.v[fpr_idx++], b, 16);
                push_stack(b + 16, 16, 16, 16);
            } else {
                push_stack(b, 32, 16, 32);
            }
            continue;
        }

        // The value's bytes, little-endian, at its size.
        uint8_t bytes[16] = {0};
        size_t sz = t.size_in_bytes();
        if (sz == 0 || sz > 16) sz = 8;
        if (arg.is_vector()) {
            std::memcpy(bytes, arg.v128_bytes(), 16);
            sz = 16;
        } else if (t.is_float()) {
            // Converted to the parameter's width (an f64 value for an f32
            // parameter, and the reverse).
            const uint64_t w = native_float_arg_bits(arg, &t);
            std::memcpy(bytes, &w, sizeof(w));
        } else {
            // Masked to the parameter's width; a float value for an integer
            // parameter is an error.
            const uint64_t v = native_int_arg_bits(arg, pt);
            std::memcpy(bytes, &v, sizeof(v));
        }

        if (is_fpr && fpr_idx < 8) {
            std::memcpy(out_args.v[fpr_idx++], bytes, arg.is_vector() ? 16 : 8);
            continue;
        }
        if (!is_fpr && gpr_idx < 8) {
            std::memcpy(&out_args.x[gpr_idx++], bytes, 8);
            continue;
        }
        push_stack(bytes, sz, apple ? std::min<size_t>(sz, 16) : (sz >= 16 ? 16 : 8),
                   apple ? sz : (sz >= 16 ? 16 : 8));
    }

    // Whole words, an even number of them: 16-byte stack alignment.
    stack.resize((stack.size() + 15) & ~size_t(15), 0);
    stack_words.resize(stack.size() / 8);
    if (!stack.empty()) std::memcpy(stack_words.data(), stack.data(), stack.size());

    out_args.stack_words = stack_words.empty() ? nullptr : stack_words.data();
    out_args.stack_word_count = stack_words.size();
}

RuntimeValue aarch64_invoke_result_value(Type ret_type, const AArch64InvokeResult& result) {
    if (ret_type.is_v256()) {
        // V0 then V1 (aarch64_isel_branch.cpp).
        alignas(16) uint8_t b[32];
        std::memcpy(b, result.q0, 16);
        std::memcpy(b + 16, result.q1, 16);
        return RuntimeValue::from_v256(ret_type, b);
    }
    // Every other type converts as on x64: X0 or the low bits of V0, narrow
    // integers sign-extended from their width.
    return native_return_value(ret_type, result.x0, result.q0);
}

} // namespace brass::codegen

#if defined(__x86_64__) || defined(_M_X64)

namespace brass::codegen {

#if !(defined(__GNUC__) || defined(__clang__))
extern "C" {
void brass_call_jit_v256_0(void* addr, uint8_t* out);
void brass_call_jit_v256_1(void* addr, uint8_t* out, const uint8_t* a0);
void brass_call_jit_v256_2(void* addr, uint8_t* out, const uint8_t* a0, const uint8_t* a1);
void brass_call_jit_v256_3(void* addr, uint8_t* out, const uint8_t* a0, const uint8_t* a1, const uint8_t* a2);
}
#endif

extern "C" void x64_sysv_invoke_thunk(
    const X64SysVInvokeArgs* args,
    X64SysVInvokeResult* result
);

namespace {

// A 256-bit vector result comes back in YMM0, which the invoke thunks do not
// capture. Such a function is called here with its (at most three) 256-bit
// vector arguments in YMM0..YMM2, and the result stored to `out`.
void call_jit_v256(void* addr, const std::vector<RuntimeValue>& args, uint8_t* out) {
#if defined(__GNUC__) || defined(__clang__)
    switch (args.size()) {
        case 0:
            asm volatile(
                "subq $40, %%rsp\n\t"
                "call *%1\n\t"
                "vmovups %%ymm0, (%0)\n\t"
                "vzeroupper\n\t"
                "addq $40, %%rsp\n\t"
                :
                : "r"(out), "r"(addr)
                : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
            );
            return;
        case 1:
            asm volatile(
                "subq $40, %%rsp\n\t"
                "vmovups (%2), %%ymm0\n\t"
                "call *%1\n\t"
                "vmovups %%ymm0, (%0)\n\t"
                "vzeroupper\n\t"
                "addq $40, %%rsp\n\t"
                :
                : "r"(out), "r"(addr), "r"(args[0].vec_bytes())
                : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
            );
            return;
        case 2:
            asm volatile(
                "subq $40, %%rsp\n\t"
                "vmovups (%2), %%ymm0\n\t"
                "vmovups (%3), %%ymm1\n\t"
                "call *%1\n\t"
                "vmovups %%ymm0, (%0)\n\t"
                "vzeroupper\n\t"
                "addq $40, %%rsp\n\t"
                :
                : "r"(out), "r"(addr), "r"(args[0].vec_bytes()), "r"(args[1].vec_bytes())
                : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
            );
            return;
        case 3:
            asm volatile(
                "subq $40, %%rsp\n\t"
                "vmovups (%2), %%ymm0\n\t"
                "vmovups (%3), %%ymm1\n\t"
                "vmovups (%4), %%ymm2\n\t"
                "call *%1\n\t"
                "vmovups %%ymm0, (%0)\n\t"
                "vzeroupper\n\t"
                "addq $40, %%rsp\n\t"
                :
                : "r"(out), "r"(addr), "r"(args[0].vec_bytes()), "r"(args[1].vec_bytes()), "r"(args[2].vec_bytes())
                : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
            );
            return;
    }
#else
    switch (args.size()) {
        case 0: brass_call_jit_v256_0(addr, out); return;
        case 1: brass_call_jit_v256_1(addr, out, args[0].vec_bytes()); return;
        case 2: brass_call_jit_v256_2(addr, out, args[0].vec_bytes(), args[1].vec_bytes()); return;
        case 3:
            brass_call_jit_v256_3(addr, out, args[0].vec_bytes(), args[1].vec_bytes(), args[2].vec_bytes());
            return;
    }
#endif
    throw std::runtime_error("JIT Error: a 256-bit vector result takes at most three arguments");
}

} // namespace

RuntimeValue JitExecutionEngine::invoke(std::string_view name) {
    return invoke(name, {});
}

RuntimeValue JitExecutionEngine::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    void* addr = get_symbol_address(name);
    if (!addr) {
        throw std::runtime_error("JIT Error: Function '" + std::string(name) + "' not found");
    }

    Type ret_type = Type::i64();
    const std::vector<Type>* param_types = nullptr;
    auto sig_it = function_signatures_.find(std::string(name));
    if (sig_it != function_signatures_.end()) {
        ret_type = sig_it->second.first;
        param_types = &sig_it->second.second;
    }

    bool any_v256 = false;
    bool all_v256 = true;
    for (size_t i = 0; i < args.size(); ++i) {
        bool v256 = args[i].is_v256() || (param_types && i < param_types->size() && (*param_types)[i].is_v256());
        any_v256 = any_v256 || v256;
        all_v256 = all_v256 && v256;
    }
    if (ret_type.is_v256()) {
        if (!all_v256 || args.size() > 3) {
            throw std::runtime_error("JIT Error: '" + std::string(name) + "' returns " + to_string(ret_type) +
                                     "; invoke takes such a function only with at most three 256-bit vector arguments");
        }
        alignas(32) uint8_t b[32];
        // Every call into generated code from here holds an entry scope: a
        // native throw's pad search stops at it (exception_win64.cpp) and
        // leaves as a C++ exception into this frame's callers.
        GeneratedCodeEntryScope entry;
        call_jit_v256(addr, args, b);
        return RuntimeValue::from_v256(ret_type, b);
    }
    if (any_v256) {
        throw std::runtime_error("JIT Error: '" + std::string(name) +
                                 "' takes a 256-bit vector argument; invoke takes one only for a 256-bit vector result");
    }

    // Every other signature, of any arity: the thunk passes the arguments by
    // parameter type and captures RAX and XMM0, converted once below.
    GeneratedCodeEntryScope entry;  // walks need not unwind the host's stack
#if defined(_WIN32)
    X64Win64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_x64_win64_invoke_args(args, param_types, addr, invoke_args, stack_words);

    X64Win64InvokeResult result;
    x64_win64_invoke_thunk(&invoke_args, &result);
#else
    X64SysVInvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_x64_sysv_invoke_args(args, param_types, addr, invoke_args, stack_words);

    X64SysVInvokeResult result;
#if defined(__GNUC__) || defined(__clang__)
    x64_sysv_invoke_thunk(&invoke_args, &result);
#endif
#endif

    return native_return_value(ret_type, result.rax, result.xmm0);
}

} // namespace brass::codegen

#elif defined(__aarch64__) || defined(_M_ARM64)

namespace brass::codegen {

// An MSVC build (cl or clang-cl) takes the thunk from gc/gc_msvc_arm64.asm.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
// File-scope asm: GCC ignores __attribute__((naked)) on AArch64. The CFI lets
// an unwinder walk from JIT code (whose frames are registered with it, see
// jit_unwind_registry) through this thunk into its C++ caller.
// partition_aarch64_invoke_args pads the stack words to an even count, so sp
// stays 16-byte aligned.
__asm__(
    ".text\n"
    BRASS_ASM_FN_BEGIN(aarch64_invoke_thunk)
    "    .cfi_startproc\n"
    "    stp x29, x30, [sp, #-48]!\n"
    "    .cfi_def_cfa_offset 48\n"
    "    .cfi_offset x29, -48\n"
    "    .cfi_offset x30, -40\n"
    "    mov x29, sp\n"
    "    .cfi_def_cfa_register x29\n"
    "    stp x19, x20, [sp, #16]\n"
    "    .cfi_offset x19, -32\n"
    "    .cfi_offset x20, -24\n"
    "    str x28, [sp, #32]\n"
    "    .cfi_offset x28, -16\n"
    "    mov x19, x1\n"                // x19 = result
    "    mov x20, x0\n"                // x20 = args
    "    ldr x2, [x20, #200]\n"        // stack_word_count
    "    cbz x2, 1f\n"
    "    lsl x3, x2, #3\n"
    "    sub sp, sp, x3\n"
    "    ldr x1, [x20, #192]\n"        // stack_words
    "    mov x4, sp\n"
    "2:\n"
    "    ldr x5, [x1], #8\n"
    "    str x5, [x4], #8\n"
    "    subs x2, x2, #1\n"
    "    b.ne 2b\n"
    "1:\n"
    "    add x1, x20, #64\n"           // v0..v7
    "    ldp q0, q1, [x1, #0]\n"
    "    ldp q2, q3, [x1, #32]\n"
    "    ldp q4, q5, [x1, #64]\n"
    "    ldp q6, q7, [x1, #96]\n"
    "    ldr x16, [x20, #208]\n"       // target
    "    ldr x28, [x20, #216]\n"       // pinned TLS block
    "    ldp x0, x1, [x20, #0]\n"      // x0..x7
    "    ldp x2, x3, [x20, #16]\n"
    "    ldp x4, x5, [x20, #32]\n"
    "    ldp x6, x7, [x20, #48]\n"
    "    blr x16\n"
    "    str x0, [x19, #0]\n"
    "    str x1, [x19, #8]\n"
    "    str q0, [x19, #16]\n"
    "    str q1, [x19, #32]\n"
    "    mov sp, x29\n"
    "    ldr x28, [sp, #32]\n"
    "    ldp x19, x20, [sp, #16]\n"
    "    ldp x29, x30, [sp], #48\n"
    "    .cfi_def_cfa sp, 0\n"
    "    .cfi_restore x28\n"
    "    .cfi_restore x19\n"
    "    .cfi_restore x20\n"
    "    .cfi_restore x29\n"
    "    .cfi_restore x30\n"
    "    ret\n"
    "    .cfi_endproc\n"
    BRASS_ASM_FN_END(aarch64_invoke_thunk)
);
#endif

RuntimeValue JitExecutionEngine::invoke(std::string_view name) {
    return invoke(name, {});
}

RuntimeValue JitExecutionEngine::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    void* addr = get_symbol_address(name);
    if (!addr) {
        throw std::runtime_error("JIT Error: Function '" + std::string(name) + "' not found");
    }

    Type ret_type = Type::i64();
    const std::vector<Type>* param_types = nullptr;
    auto sig_it = function_signatures_.find(std::string(name));
    if (sig_it != function_signatures_.end()) {
        ret_type = sig_it->second.first;
        param_types = &sig_it->second.second;
    }

    AArch64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_aarch64_invoke_args(args, param_types, addr, invoke_args, stack_words);

    AArch64InvokeResult result;
    // As on x64 (Windows ARM64 included): a native throw's pad search stops
    // here (entry_boundary, exception_win64.cpp) and leaves as a C++
    // exception into this frame's callers; walks need not unwind the host's
    // stack.
    GeneratedCodeEntryScope entry;
    aarch64_invoke_thunk(&invoke_args, &result);
    return aarch64_invoke_result_value(ret_type, result);
}

} // namespace brass::codegen

#else // !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__) && !defined(_M_ARM64)

namespace brass::codegen {

RuntimeValue JitExecutionEngine::invoke(std::string_view name) {
    return invoke(name, {});
}

RuntimeValue JitExecutionEngine::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    (void)name;
    (void)args;
    throw std::runtime_error("JitExecutionEngine::invoke is only supported on x86_64 and aarch64");
}

} // namespace brass::codegen

#endif
