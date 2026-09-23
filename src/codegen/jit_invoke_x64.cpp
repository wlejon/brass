#include <brass/codegen/jit_exec.hpp>
#include <cstring>
#include <cstdint>
#include <vector>

namespace brass::codegen {

void partition_x64_sysv_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    X64SysVInvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
) {
    out_args = X64SysVInvokeArgs{};
    out_args.target_fn = target_fn;
    stack_words.clear();

    size_t gpr_idx = 0;
    size_t xmm_idx = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        bool is_float = false;
        bool is_vec = false;
        if (param_types && i < param_types->size()) {
            const Type& pt = (*param_types)[i];
            is_float = pt.is_float();
            is_vec = pt.is_vector();
        } else {
            is_float = arg.is_f32() || arg.is_f64();
            is_vec = arg.is_vector();
        }

        if (is_vec) {
            if (xmm_idx < 8) {
                std::memcpy(out_args.xmm[xmm_idx], arg.v128_bytes(), 16);
                xmm_idx++;
            } else {
                uint64_t words[2] = {0, 0};
                std::memcpy(words, arg.v128_bytes(), 16);
                stack_words.push_back(words[0]);
                stack_words.push_back(words[1]);
            }
        } else if (is_float) {
            if (xmm_idx < 8) {
                if (arg.is_f32()) {
                    float f = arg.as_f32();
                    std::memcpy(out_args.xmm[xmm_idx], &f, sizeof(float));
                } else {
                    double d = arg.as_f64();
                    std::memcpy(out_args.xmm[xmm_idx], &d, sizeof(double));
                }
                xmm_idx++;
            } else {
                if (arg.is_f32()) {
                    float f = arg.as_f32();
                    uint64_t w = 0;
                    std::memcpy(&w, &f, sizeof(float));
                    stack_words.push_back(w);
                } else {
                    double d = arg.as_f64();
                    uint64_t w = 0;
                    std::memcpy(&w, &d, sizeof(double));
                    stack_words.push_back(w);
                }
            }
        } else {
            uint64_t val = 0;
            if (arg.is_i32()) val = static_cast<uint64_t>(static_cast<uint32_t>(arg.as_i32()));
            else if (arg.is_i64()) val = static_cast<uint64_t>(arg.as_i64());
            else if (arg.is_ptr()) val = static_cast<uint64_t>(arg.as_ptr());
            else if (arg.is_gcref()) val = static_cast<uint64_t>(arg.as_gcref());
            else val = arg.raw_bits();

            if (gpr_idx < 6) {
                out_args.gpr[gpr_idx] = val;
                gpr_idx++;
            } else {
                stack_words.push_back(val);
            }
        }
    }

    if (!stack_words.empty()) {
        out_args.stack_words = stack_words.data();
        out_args.stack_word_count = stack_words.size();
    }
}

void partition_x64_win64_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    X64Win64InvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
) {
    out_args = X64Win64InvokeArgs{};
    out_args.target_fn = target_fn;
    stack_words.clear();

    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        bool is_float = false;
        bool is_vec = false;
        if (param_types && i < param_types->size()) {
            const Type& pt = (*param_types)[i];
            is_float = pt.is_float();
            is_vec = pt.is_vector();
        } else {
            is_float = arg.is_f32() || arg.is_f64();
            is_vec = arg.is_vector();
        }

        if (i < 4) {
            if (is_vec) {
                std::memcpy(out_args.xmm[i], arg.v128_bytes(), 16);
            } else if (is_float) {
                if (arg.is_f32()) {
                    float f = arg.as_f32();
                    std::memcpy(out_args.xmm[i], &f, sizeof(float));
                    uint64_t bits = 0;
                    std::memcpy(&bits, &f, sizeof(float));
                    out_args.gpr[i] = bits;
                } else {
                    double d = arg.as_f64();
                    std::memcpy(out_args.xmm[i], &d, sizeof(double));
                    uint64_t bits = 0;
                    std::memcpy(&bits, &d, sizeof(double));
                    out_args.gpr[i] = bits;
                }
            } else {
                uint64_t val = 0;
                if (arg.is_i32()) val = static_cast<uint64_t>(static_cast<uint32_t>(arg.as_i32()));
                else if (arg.is_i64()) val = static_cast<uint64_t>(arg.as_i64());
                else if (arg.is_ptr()) val = static_cast<uint64_t>(arg.as_ptr());
                else if (arg.is_gcref()) val = static_cast<uint64_t>(arg.as_gcref());
                else val = arg.raw_bits();
                out_args.gpr[i] = val;
            }
        } else {
            if (is_vec) {
                uint64_t words[2] = {0, 0};
                std::memcpy(words, arg.v128_bytes(), 16);
                stack_words.push_back(words[0]);
                stack_words.push_back(words[1]);
            } else if (is_float) {
                if (arg.is_f32()) {
                    float f = arg.as_f32();
                    uint64_t w = 0;
                    std::memcpy(&w, &f, sizeof(float));
                    stack_words.push_back(w);
                } else {
                    double d = arg.as_f64();
                    uint64_t w = 0;
                    std::memcpy(&w, &d, sizeof(double));
                    stack_words.push_back(w);
                }
            } else {
                uint64_t val = 0;
                if (arg.is_i32()) val = static_cast<uint64_t>(static_cast<uint32_t>(arg.as_i32()));
                else if (arg.is_i64()) val = static_cast<uint64_t>(arg.as_i64());
                else if (arg.is_ptr()) val = static_cast<uint64_t>(arg.as_ptr());
                else if (arg.is_gcref()) val = static_cast<uint64_t>(arg.as_gcref());
                else val = arg.raw_bits();
                stack_words.push_back(val);
            }
        }
    }

    if (!stack_words.empty()) {
        out_args.stack_words = stack_words.data();
        out_args.stack_word_count = stack_words.size();
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
// DWARF CFI for the thunk's frame on ELF / Mach-O (the compiler opens the
// FDE of a naked function as for any other), so that a C++ exception thrown
// by a host function the JIT code called unwinds through it. MinGW describes
// frames with SEH directives instead, and this thunk has no SEH info.
#if defined(_WIN32)
#define BRASS_THUNK_CFI(s) ""
#else
#define BRASS_THUNK_CFI(s) s
#endif
extern "C" __attribute__((naked)) void x64_sysv_invoke_thunk(
    const X64SysVInvokeArgs* args,
    X64SysVInvokeResult* result
) {
    __asm__ volatile(
        "pushq %rbp\n\t"
        BRASS_THUNK_CFI(".cfi_def_cfa_offset 16\n\t.cfi_offset %rbp, -16\n\t")
        "movq %rsp, %rbp\n\t"
        BRASS_THUNK_CFI(".cfi_def_cfa_register %rbp\n\t")
        "pushq %r12\n\t"
        "pushq %r13\n\t"
        "pushq %r14\n\t"
        "pushq %r15\n\t"
        "pushq %rbx\n\t"
        BRASS_THUNK_CFI(".cfi_offset %r12, -24\n\t.cfi_offset %r13, -32\n\t.cfi_offset %r14, -40\n\t"
                        ".cfi_offset %r15, -48\n\t.cfi_offset %rbx, -56\n\t")

        "movq %rdi, %r12\n\t"          // r12 = args
        "movq %rsi, %r13\n\t"          // r13 = result

        // 16-byte align rsp before allocating arguments
        "andq $-16, %rsp\n\t"

        // Check if stack_word_count > 0
        "movq 184(%r12), %rcx\n\t"     // rcx = stack_word_count
        "testq %rcx, %rcx\n\t"
        "jz 1f\n\t"

        // If count is odd, pad 8 bytes first so rsp remains 16-byte aligned after allocating count words
        "testq $1, %rcx\n\t"
        "jz 2f\n\t"
        "subq $8, %rsp\n\t"

        "2:\n\t"
        "movq %rcx, %rax\n\t"
        "shlq $3, %rax\n\t"             // rax = count * 8 bytes
        "subq %rax, %rsp\n\t"          // allocate on stack

        // Copy stack_words to [rsp]
        "movq 176(%r12), %rsi\n\t"     // rsi = stack_words source
        "movq %rsp, %rdi\n\t"          // rdi = rsp dest
        "rep movsq\n\t"

        "1:\n\t"
        // Load XMM0..XMM7
        "movdqa 48(%r12), %xmm0\n\t"
        "movdqa 64(%r12), %xmm1\n\t"
        "movdqa 80(%r12), %xmm2\n\t"
        "movdqa 96(%r12), %xmm3\n\t"
        "movdqa 112(%r12), %xmm4\n\t"
        "movdqa 128(%r12), %xmm5\n\t"
        "movdqa 144(%r12), %xmm6\n\t"
        "movdqa 160(%r12), %xmm7\n\t"

        // Load target function address into r11
        "movq 192(%r12), %r11\n\t"

        // AL = 8 (number of vector registers used for varargs / SysV ABI)
        "movb $8, %al\n\t"

        // Load GPRs (RDI, RSI, RDX, RCX, R8, R9)
        "movq 0(%r12), %rdi\n\t"
        "movq 8(%r12), %rsi\n\t"
        "movq 16(%r12), %rdx\n\t"
        "movq 24(%r12), %rcx\n\t"
        "movq 32(%r12), %r8\n\t"
        "movq 40(%r12), %r9\n\t"

        // Call target function
        "call *%r11\n\t"

        // Store return values
        "movq %rax, 0(%r13)\n\t"
        "movq %rdx, 8(%r13)\n\t"
        "movdqa %xmm0, 16(%r13)\n\t"

        // Epilogue
        "leaq -40(%rbp), %rsp\n\t"
        "popq %rbx\n\t"
        "popq %r15\n\t"
        "popq %r14\n\t"
        "popq %r13\n\t"
        "popq %r12\n\t"
        "popq %rbp\n\t"
        BRASS_THUNK_CFI(".cfi_def_cfa %rsp, 8\n\t")
        "ret\n\t"
    );
}
#undef BRASS_THUNK_CFI
#endif

#if defined(_WIN32) && !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
extern "C" __attribute__((naked)) void x64_win64_invoke_thunk(
    const X64Win64InvokeArgs* args,
    X64Win64InvokeResult* result
) {
    __asm__ volatile(
        "pushq %rbp\n\t"
        "movq %rsp, %rbp\n\t"
        "pushq %rbx\n\t"
        "pushq %rsi\n\t"
        "pushq %rdi\n\t"
        "pushq %r12\n\t"
        "pushq %r13\n\t"

        "movq %rcx, %r12\n\t"
        "movq %rdx, %r13\n\t"

        "movq 104(%r12), %rcx\n\t"
        "leaq 4(%rcx), %rax\n\t"
        "testq $1, %rax\n\t"
        "jnz 1f\n\t"
        "incq %rax\n\t"
        "1:\n\t"
        "shlq $3, %rax\n\t"
        "subq %rax, %rsp\n\t"

        "testq %rcx, %rcx\n\t"
        "jz 2f\n\t"
        "movq 96(%r12), %rsi\n\t"
        "leaq 32(%rsp), %rdi\n\t"
        "rep movsq\n\t"
        "2:\n\t"

        "movdqu 32(%r12), %xmm0\n\t"
        "movdqu 48(%r12), %xmm1\n\t"
        "movdqu 64(%r12), %xmm2\n\t"
        "movdqu 80(%r12), %xmm3\n\t"

        "movq 0(%r12), %rcx\n\t"
        "movq 8(%r12), %rdx\n\t"
        "movq 16(%r12), %r8\n\t"
        "movq 24(%r12), %r9\n\t"

        "movq 112(%r12), %r11\n\t"
        "call *%r11\n\t"

        "movq %rax, 0(%r13)\n\t"
        "movdqu %xmm0, 16(%r13)\n\t"

        "leaq -40(%rbp), %rsp\n\t"
        "popq %r13\n\t"
        "popq %r12\n\t"
        "popq %rdi\n\t"
        "popq %rsi\n\t"
        "popq %rbx\n\t"
        "popq %rbp\n\t"
        "ret\n\t"
    );
}
#elif !defined(_WIN32)
extern "C" void x64_win64_invoke_thunk(
    const X64Win64InvokeArgs* /*args*/,
    X64Win64InvokeResult* /*result*/
) {
}
#endif
#endif

} // namespace brass::codegen
