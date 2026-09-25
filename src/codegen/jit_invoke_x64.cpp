#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brass::codegen {

uint64_t native_int_arg_bits(const RuntimeValue& arg, const Type* param) {
    if (param) {
        if (arg.is_f32() || arg.is_f64() || arg.is_vector()) {
            throw std::runtime_error("invoke: a " + to_string(arg.type()) +
                                     " argument for a " + to_string(*param) + " parameter");
        }
        switch (param->kind()) {
            case TypeKind::I8: return static_cast<uint8_t>(arg.as_u64());
            case TypeKind::I16: return static_cast<uint16_t>(arg.as_u64());
            case TypeKind::I32: return static_cast<uint32_t>(arg.as_u64());
            case TypeKind::I64: return static_cast<uint64_t>(arg.as_i64());
            default: break;
        }
    }
    if (arg.is_i32()) return static_cast<uint64_t>(static_cast<uint32_t>(arg.as_i32()));
    if (arg.is_i64()) return static_cast<uint64_t>(arg.as_i64());
    if (arg.is_ptr()) return static_cast<uint64_t>(arg.as_ptr());
    if (arg.is_gcref()) return static_cast<uint64_t>(arg.as_gcref());
    return arg.raw_bits();
}

uint64_t native_float_arg_bits(const RuntimeValue& arg, const Type* param) {
    bool f32 = param ? param->kind() == TypeKind::F32 : arg.is_f32();
    uint64_t bits = 0;
    if (f32) {
        float f = arg.as_f32();
        std::memcpy(&bits, &f, sizeof(float));
    } else {
        double d = arg.as_f64();
        std::memcpy(&bits, &d, sizeof(double));
    }
    return bits;
}

RuntimeValue native_return_value(Type ret, uint64_t gpr, const uint8_t* vec) {
    switch (ret.kind()) {
        case TypeKind::Void: return RuntimeValue::from_void();
        case TypeKind::I8: return RuntimeValue::from_i32(static_cast<int32_t>(static_cast<int8_t>(gpr)));
        case TypeKind::I16: return RuntimeValue::from_i32(static_cast<int32_t>(static_cast<int16_t>(gpr)));
        case TypeKind::I32: return RuntimeValue::from_i32(static_cast<int32_t>(gpr));
        case TypeKind::I64: return RuntimeValue::from_i64(static_cast<int64_t>(gpr));
        case TypeKind::F32: {
            float f = 0.0f;
            std::memcpy(&f, vec, sizeof(float));
            return RuntimeValue::from_f32(f);
        }
        case TypeKind::F64: {
            double d = 0.0;
            std::memcpy(&d, vec, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
        case TypeKind::Ptr: return RuntimeValue::from_ptr(static_cast<uintptr_t>(gpr));
        case TypeKind::GCRef: return RuntimeValue::from_gcref(static_cast<uintptr_t>(gpr));
        case TypeKind::Tagged: return RuntimeValue::from_tagged(gpr);
        case TypeKind::F32x4:
        case TypeKind::F64x2:
        case TypeKind::I32x4:
        case TypeKind::I64x2: return RuntimeValue::from_v128(ret, vec);
        case TypeKind::F32x8:
        case TypeKind::F64x4:
        case TypeKind::I32x8:
        case TypeKind::I64x4: break;
    }
    throw std::runtime_error("invoke: a " + to_string(ret) +
                             " result does not fit the invoke thunk's return registers");
}

void partition_x64_sysv_invoke_args(
    const std::vector<RuntimeValue>& args,
    const std::vector<Type>* param_types,
    void* target_fn,
    X64SysVInvokeArgs& out_args,
    std::vector<uint64_t>& stack_words
) {
    out_args = X64SysVInvokeArgs{};
    out_args.target_fn = target_fn;
    out_args.pinned_tls = runtime::host_pinned_tls_block();
    stack_words.clear();

    size_t gpr_idx = 0;
    size_t xmm_idx = 0;

    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        const Type* pt = (param_types && i < param_types->size()) ? &(*param_types)[i] : nullptr;
        bool is_float = pt ? pt->is_float() : (arg.is_f32() || arg.is_f64());
        bool is_vec = pt ? pt->is_vector() : arg.is_vector();

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
            uint64_t w = native_float_arg_bits(arg, pt);
            if (xmm_idx < 8) {
                std::memcpy(out_args.xmm[xmm_idx], &w, sizeof(w));
                xmm_idx++;
            } else {
                stack_words.push_back(w);
            }
        } else {
            uint64_t val = native_int_arg_bits(arg, pt);

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
    out_args.pinned_tls = runtime::host_pinned_tls_block();
    stack_words.clear();

    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        const Type* pt = (param_types && i < param_types->size()) ? &(*param_types)[i] : nullptr;
        bool is_float = pt ? pt->is_float() : (arg.is_f32() || arg.is_f64());
        bool is_vec = pt ? pt->is_vector() : arg.is_vector();

        if (i < 4) {
            if (is_vec) {
                std::memcpy(out_args.xmm[i], arg.v128_bytes(), 16);
            } else if (is_float) {
                uint64_t bits = native_float_arg_bits(arg, pt);
                std::memcpy(out_args.xmm[i], &bits, sizeof(bits));
                out_args.gpr[i] = bits;
            } else {
                out_args.gpr[i] = native_int_arg_bits(arg, pt);
            }
        } else {
            if (is_vec) {
                uint64_t words[2] = {0, 0};
                std::memcpy(words, arg.v128_bytes(), 16);
                stack_words.push_back(words[0]);
                stack_words.push_back(words[1]);
            } else if (is_float) {
                stack_words.push_back(native_float_arg_bits(arg, pt));
            } else {
                stack_words.push_back(native_int_arg_bits(arg, pt));
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
        "movq %rsi, %rbx\n\t"          // rbx = result (r13 is the pinned TLS register)

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

        // Load target function address into r11, the pinned TLS block into r13
        "movq 192(%r12), %r11\n\t"
        "movq 200(%r12), %r13\n\t"

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
        "movq %rax, 0(%rbx)\n\t"
        "movq %rdx, 8(%rbx)\n\t"
        "movdqa %xmm0, 16(%rbx)\n\t"

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
        "movq %rdx, %rbx\n\t"          // rbx = result (r13 is the pinned TLS register)

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
        "movq 120(%r12), %r13\n\t"
        "call *%r11\n\t"

        "movq %rax, 0(%rbx)\n\t"
        "movdqu %xmm0, 16(%rbx)\n\t"

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
