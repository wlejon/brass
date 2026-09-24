#include <brass/codegen/jit_exec.hpp>
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
        const Type t = (param_types && i < param_types->size()) ? (*param_types)[i] : arg.type();
        const bool is_fpr = t.is_float() || t.is_vector();

        // The value's bytes, little-endian, at its size.
        uint8_t bytes[16] = {0};
        size_t sz = t.size_in_bytes();
        if (sz == 0 || sz > 16) sz = 8;
        if (arg.is_vector()) {
            std::memcpy(bytes, arg.v128_bytes(), 16);
            sz = 16;
        } else if (t.kind() == TypeKind::F32 || arg.is_f32()) {
            const float f = arg.is_f32() ? arg.as_f32() : static_cast<float>(arg.as_f64());
            std::memcpy(bytes, &f, sizeof(float));
        } else if (t.is_float()) {
            const double d = arg.as_f64();
            std::memcpy(bytes, &d, sizeof(double));
        } else {
            const uint64_t v = arg.as_u64();
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
        const size_t align = apple ? std::min<size_t>(sz, 16) : (sz >= 16 ? 16 : 8);
        const size_t slot = apple ? sz : (sz >= 16 ? 16 : 8);
        const size_t at = (stack.size() + align - 1) & ~(align - 1);
        stack.resize(at + slot, 0);
        std::memcpy(stack.data() + at, bytes, std::min(sz, slot));
    }

    // Whole words, an even number of them: 16-byte stack alignment.
    stack.resize((stack.size() + 15) & ~size_t(15), 0);
    stack_words.resize(stack.size() / 8);
    if (!stack.empty()) std::memcpy(stack_words.data(), stack.data(), stack.size());

    out_args.stack_words = stack_words.empty() ? nullptr : stack_words.data();
    out_args.stack_word_count = stack_words.size();
}

} // namespace brass::codegen

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace brass::codegen {

#if defined(__GNUC__) || defined(__clang__)
inline __m128 call_jit_vec1_ret_vec(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline float call_jit_vec1_ret_f32(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register float res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline double call_jit_vec1_ret_f64(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register double res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec1_ret_i64(void* addr, __m128 a0) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0)
        : "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline __m128 call_jit_vec2_ret_vec(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline float call_jit_vec2_ret_f32(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register float res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec2_ret_i64(void* addr, __m128 a0, __m128 a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0), "x"(r_xmm1)
        : "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

#if defined(_WIN32)
inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rax", "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rax", "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdx asm("rdx") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdx)
        : "rcx", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rax", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rax", "rdx", "r8", "r9", "r10", "r11", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rcx asm("rcx") = a0;
    register __m128 r_xmm1 asm("xmm1") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "r"(r_rcx), "x"(r_xmm1)
        : "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}
#else // SysV AMD64 (macOS, Linux)
inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdi asm("rdi") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "x"(r_xmm0), "r"(r_rdi)
        : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdi asm("rdi") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdi)
        : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    register __m128 r_xmm0 asm("xmm0") = a0;
    register int64_t r_rdi asm("rdi") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "x"(r_xmm0), "r"(r_rdi)
        : "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rdi asm("rdi") = a0;
    register __m128 r_xmm0 asm("xmm0") = a1;
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%0\n\t"
        "addq $32, %%rsp"
        :
        : "r"(addr), "r"(r_rdi), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
}

inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rdi asm("rdi") = a0;
    register __m128 r_xmm0 asm("xmm0") = a1;
    register __m128 res asm("xmm0");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=x"(res)
        : "r"(addr), "r"(r_rdi), "x"(r_xmm0)
        : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}

inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    register int64_t r_rdi asm("rdi") = a0;
    register __m128 r_xmm0 asm("xmm0") = a1;
    register int64_t res asm("rax");
    asm volatile(
        "subq $32, %%rsp\n\t"
        "call *%1\n\t"
        "addq $32, %%rsp"
        : "=r"(res)
        : "r"(addr), "r"(r_rdi), "x"(r_xmm0)
        : "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
    );
    return res;
}
#endif
#else
extern "C" {
__m128 brass_call_jit_int_vec_ret_vec(void* addr, int64_t a0, const __m128* a1);
void brass_call_jit_int_vec_ret_void(void* addr, int64_t a0, const __m128* a1);
int64_t brass_call_jit_int_vec_ret_i64(void* addr, int64_t a0, const __m128* a1);
__m128 brass_call_jit_vec_int_ret_vec(void* addr, const __m128* a0, int64_t a1);
void brass_call_jit_vec_int_ret_void(void* addr, const __m128* a0, int64_t a1);
int64_t brass_call_jit_vec_int_ret_i64(void* addr, const __m128* a0, int64_t a1);
__m128 brass_call_jit_vec1_ret_vec(void* addr, const __m128* a0);
float brass_call_jit_vec1_ret_f32(void* addr, const __m128* a0);
double brass_call_jit_vec1_ret_f64(void* addr, const __m128* a0);
int64_t brass_call_jit_vec1_ret_i64(void* addr, const __m128* a0);
__m128 brass_call_jit_vec2_ret_vec(void* addr, const __m128* a0, const __m128* a1);
float brass_call_jit_vec2_ret_f32(void* addr, const __m128* a0, const __m128* a1);
int64_t brass_call_jit_vec2_ret_i64(void* addr, const __m128* a0, const __m128* a1);
void brass_call_jit_v256_0(void* addr, uint8_t* out);
void brass_call_jit_v256_1(void* addr, uint8_t* out, const uint8_t* a0);
void brass_call_jit_v256_2(void* addr, uint8_t* out, const uint8_t* a0, const uint8_t* a1);
void brass_call_jit_v256_3(void* addr, uint8_t* out, const uint8_t* a0, const uint8_t* a1, const uint8_t* a2);
void brass_call_jit_v128_3(void* addr, uint8_t* out, const uint8_t* a0, const uint8_t* a1, const uint8_t* a2);
}

inline __m128 call_jit_vec1_ret_vec(void* addr, __m128 a0) {
    return brass_call_jit_vec1_ret_vec(addr, &a0);
}
inline float call_jit_vec1_ret_f32(void* addr, __m128 a0) {
    return brass_call_jit_vec1_ret_f32(addr, &a0);
}
inline double call_jit_vec1_ret_f64(void* addr, __m128 a0) {
    return brass_call_jit_vec1_ret_f64(addr, &a0);
}
inline int64_t call_jit_vec1_ret_i64(void* addr, __m128 a0) {
    return brass_call_jit_vec1_ret_i64(addr, &a0);
}
inline __m128 call_jit_vec2_ret_vec(void* addr, __m128 a0, __m128 a1) {
    return brass_call_jit_vec2_ret_vec(addr, &a0, &a1);
}
inline float call_jit_vec2_ret_f32(void* addr, __m128 a0, __m128 a1) {
    return brass_call_jit_vec2_ret_f32(addr, &a0, &a1);
}
inline int64_t call_jit_vec2_ret_i64(void* addr, __m128 a0, __m128 a1) {
    return brass_call_jit_vec2_ret_i64(addr, &a0, &a1);
}
inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    brass_call_jit_vec_int_ret_void(addr, &a0, a1);
}
inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    return brass_call_jit_vec_int_ret_vec(addr, &a0, a1);
}
inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    return brass_call_jit_vec_int_ret_i64(addr, &a0, a1);
}
inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    brass_call_jit_int_vec_ret_void(addr, a0, &a1);
}
inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    return brass_call_jit_int_vec_ret_vec(addr, a0, &a1);
}
inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    return brass_call_jit_int_vec_ret_i64(addr, a0, &a1);
}
#endif

extern "C" void x64_sysv_invoke_thunk(
    const X64SysVInvokeArgs* args,
    X64SysVInvokeResult* result
);

RuntimeValue JitExecutionEngine::invoke(std::string_view name) {
    return invoke(name, {});
}

RuntimeValue JitExecutionEngine::invoke(std::string_view name, const std::vector<RuntimeValue>& args) {
    void* addr = get_symbol_address(name);
    if (!addr) {
        throw std::runtime_error("JIT Error: Function '" + std::string(name) + "' not found");
    }

    Type ret_type = Type::i64();
    auto sig_it = function_signatures_.find(std::string(name));
    if (sig_it != function_signatures_.end()) {
        ret_type = sig_it->second.first;
    }

    // Helper lambdas to extract typed values
    auto get_int = [&](size_t idx) -> int64_t {
        if (idx >= args.size()) return 0;
        return args[idx].as_i64();
    };

    auto get_float = [&](size_t idx) -> double {
        if (idx >= args.size()) return 0.0;
        return args[idx].as_f64();
    };

    // 0 arguments
    if (args.empty()) {
        if (ret_type.is_void()) {
            reinterpret_cast<void(*)()>(addr)();
            return RuntimeValue::from_void();
        } else if (ret_type.is_vector()) {
            if (ret_type.is_v256()) {
                alignas(32) uint8_t b[32];
#if defined(__GNUC__) || defined(__clang__)
                asm volatile(
                    "subq $40, %%rsp\n\t"
                    "call *%1\n\t"
                    "vmovups %%ymm0, (%0)\n\t"
                    "vzeroupper\n\t"
                    "addq $40, %%rsp\n\t"
                    :
                    : "r"(b), "r"(addr)
                    : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
                );
#else
                brass_call_jit_v256_0(addr, b);
#endif
                return RuntimeValue::from_v256(ret_type, b);
            }
            __m128 r = reinterpret_cast<__m128(*)()>(addr)();
            alignas(16) uint8_t b[16];
            std::memcpy(b, &r, 16);
            return RuntimeValue::from_v128(ret_type, b);
        } else if (ret_type.is_float()) {
            if (ret_type.kind() == TypeKind::F32) {
                float r = reinterpret_cast<float(*)()>(addr)();
                return RuntimeValue::from_f32(r);
            }
            double r = reinterpret_cast<double(*)()>(addr)();
            return RuntimeValue::from_f64(r);
        } else if (ret_type.kind() == TypeKind::I32) {
            int32_t r = reinterpret_cast<int32_t(*)()>(addr)();
            return RuntimeValue::from_i32(r);
        } else {
            int64_t r = reinterpret_cast<int64_t(*)()>(addr)();
            return RuntimeValue::from_i64(r);
        }
    }

    // 1 argument
    if (args.size() == 1) {
        if (args[0].is_v256()) {
            if (ret_type.is_v256()) {
                alignas(32) uint8_t b[32];
#if defined(__GNUC__) || defined(__clang__)
                asm volatile(
                    "subq $40, %%rsp\n\t"
                    "vmovups (%2), %%ymm0\n\t"
                    "call *%1\n\t"
                    "vmovups %%ymm0, (%0)\n\t"
                    "vzeroupper\n\t"
                    "addq $40, %%rsp\n\t"
                    :
                    : "r"(b), "r"(addr), "r"(args[0].vec_bytes())
                    : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
                );
#else
                brass_call_jit_v256_1(addr, b, args[0].vec_bytes());
#endif
                return RuntimeValue::from_v256(ret_type, b);
            }
        }
        if (args[0].is_vector()) {
            __m128 a0;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            if (ret_type.is_vector()) {
                __m128 r = call_jit_vec1_ret_vec(addr, a0);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = call_jit_vec1_ret_f32(addr, a0);
                    return RuntimeValue::from_f32(r);
                }
                double r = call_jit_vec1_ret_f64(addr, a0);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = call_jit_vec1_ret_i64(addr, a0);
                if (ret_type.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(r));
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_f64() || args[0].is_f32()) {
            if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = reinterpret_cast<float(*)(double)>(addr)(get_float(0));
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double)>(addr)(get_float(0));
                return RuntimeValue::from_i64(r);
            }
        } else {
            if (ret_type.is_vector()) {
                __m128 r = reinterpret_cast<__m128(*)(int64_t)>(addr)(get_int(0));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = reinterpret_cast<float(*)(int64_t)>(addr)(get_int(0));
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(int64_t)>(addr)(get_int(0));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t)>(addr)(get_int(0));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 2 arguments
    if (args.size() == 2) {
        if (args[0].is_v256() && args[1].is_v256()) {
            if (ret_type.is_v256()) {
                alignas(32) uint8_t b[32];
#if defined(__GNUC__) || defined(__clang__)
                asm volatile(
                    "subq $40, %%rsp\n\t"
                    "vmovups (%2), %%ymm0\n\t"
                    "vmovups (%3), %%ymm1\n\t"
                    "call *%1\n\t"
                    "vmovups %%ymm0, (%0)\n\t"
                    "vzeroupper\n\t"
                    "addq $40, %%rsp\n\t"
                    :
                    : "r"(b), "r"(addr), "r"(args[0].vec_bytes()), "r"(args[1].vec_bytes())
                    : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
                );
#else
                brass_call_jit_v256_2(addr, b, args[0].vec_bytes(), args[1].vec_bytes());
#endif
                return RuntimeValue::from_v256(ret_type, b);
            }
        }
        if (args[0].is_vector() && args[1].is_vector()) {
            __m128 a0, a1;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            std::memcpy(&a1, args[1].v128_bytes(), 16);
            if (ret_type.is_vector()) {
                __m128 r = call_jit_vec2_ret_vec(addr, a0, a1);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = call_jit_vec2_ret_f32(addr, a0, a1);
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(__m128, __m128)>(addr)(a0, a1);
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = call_jit_vec2_ret_i64(addr, a0, a1);
                return RuntimeValue::from_i64(r);
            }
        } else if (args[0].is_vector() && !args[1].is_vector()) {
            __m128 a0;
            std::memcpy(&a0, args[0].v128_bytes(), 16);
            if (ret_type.is_void()) {
                call_jit_vec_int_ret_void(addr, a0, get_int(1));
                return RuntimeValue::from_void();
            } else if (ret_type.is_vector()) {
                __m128 r = call_jit_vec_int_ret_vec(addr, a0, get_int(1));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else {
                int64_t r = call_jit_vec_int_ret_i64(addr, a0, get_int(1));
                return RuntimeValue::from_i64(r);
            }
        } else if (!args[0].is_vector() && args[1].is_vector()) {
            __m128 a1;
            std::memcpy(&a1, args[1].v128_bytes(), 16);
            if (ret_type.is_void()) {
                call_jit_int_vec_ret_void(addr, get_int(0), a1);
                return RuntimeValue::from_void();
            } else if (ret_type.is_vector()) {
                __m128 r = call_jit_int_vec_ret_vec(addr, get_int(0), a1);
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else {
                int64_t r = call_jit_int_vec_ret_i64(addr, get_int(0), a1);
                return RuntimeValue::from_i64(r);
            }
        } else if ((args[0].is_f64() || args[0].is_f32()) && (args[1].is_f64() || args[1].is_f32())) {
            if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = reinterpret_cast<float(*)(double, double)>(addr)(get_float(0), get_float(1));
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double)>(addr)(get_float(0), get_float(1));
                return RuntimeValue::from_i64(r);
            }
        } else if (!args[0].is_f64() && !args[0].is_f32() && !args[1].is_f64() && !args[1].is_f32()) {
            if (ret_type.is_vector()) {
                __m128 r = reinterpret_cast<__m128(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                alignas(16) uint8_t b[16];
                std::memcpy(b, &r, 16);
                return RuntimeValue::from_v128(ret_type, b);
            } else if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = reinterpret_cast<float(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(addr)(get_int(0), get_int(1));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 3 arguments
    if (args.size() == 3) {
        if (args[0].is_v256() && args[1].is_v256() && args[2].is_v256()) {
            if (ret_type.is_v256()) {
                alignas(32) uint8_t b[32];
#if defined(__GNUC__) || defined(__clang__)
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
                    : "r"(b), "r"(addr), "r"(args[0].vec_bytes()), "r"(args[1].vec_bytes()), "r"(args[2].vec_bytes())
                    : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
                );
#else
                brass_call_jit_v256_3(addr, b, args[0].vec_bytes(), args[1].vec_bytes(), args[2].vec_bytes());
#endif
                return RuntimeValue::from_v256(ret_type, b);
            }
        }
        if (args[0].is_v128() && args[1].is_v128() && args[2].is_v128()) {
            if (ret_type.is_v128()) {
                alignas(16) uint8_t b[16];
#if defined(__GNUC__) || defined(__clang__)
                asm volatile(
                    "subq $40, %%rsp\n\t"
                    "movups (%2), %%xmm0\n\t"
                    "movups (%3), %%xmm1\n\t"
                    "movups (%4), %%xmm2\n\t"
                    "call *%1\n\t"
                    "movups %%xmm0, (%0)\n\t"
                    "addq $40, %%rsp\n\t"
                    :
                    : "r"(b), "r"(addr), "r"(args[0].v128_bytes()), "r"(args[1].v128_bytes()), "r"(args[2].v128_bytes())
                    : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory"
                );
#else
                brass_call_jit_v128_3(addr, b, args[0].v128_bytes(), args[1].v128_bytes(), args[2].v128_bytes());
#endif
                return RuntimeValue::from_v128(ret_type, b);
            }
        }
        if ((args[0].is_f32() || args[0].is_f64()) && (args[1].is_f32() || args[1].is_f64()) && (args[2].is_f32() || args[2].is_f64())) {
            if (ret_type.kind() == TypeKind::F32) {
                float r = reinterpret_cast<float(*)(float, float, float)>(addr)(args[0].as_f32(), args[1].as_f32(), args[2].as_f32());
                return RuntimeValue::from_f32(r);
            } else {
                double r = reinterpret_cast<double(*)(double, double, double)>(addr)(args[0].as_f64(), args[1].as_f64(), args[2].as_f64());
                return RuntimeValue::from_f64(r);
            }
        }
        if (args[0].is_f64() && args[1].is_f64() && !args[2].is_f64()) {
            if (ret_type.is_float()) {
                if (ret_type.kind() == TypeKind::F32) {
                    float r = reinterpret_cast<float(*)(double, double, int64_t)>(addr)(get_float(0), get_float(1), get_int(2));
                    return RuntimeValue::from_f32(r);
                }
                double r = reinterpret_cast<double(*)(double, double, int64_t)>(addr)(get_float(0), get_float(1), get_int(2));
                return RuntimeValue::from_f64(r);
            } else {
                int64_t r = reinterpret_cast<int64_t(*)(double, double, int64_t)>(addr)(get_float(0), get_float(1), get_int(2));
                return RuntimeValue::from_i64(r);
            }
        }
    }

    // 3 or more arguments: support arbitrary counts, stack passing, and mixed integer/float signatures
    const std::vector<Type>* param_types = nullptr;
    if (sig_it != function_signatures_.end()) {
        param_types = &sig_it->second.second;
    }

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

    if (ret_type.is_void()) {
        return RuntimeValue::from_void();
    }
    if (ret_type.is_vector()) {
        return RuntimeValue::from_v128(ret_type, result.xmm0);
    }
    if (ret_type.is_float()) {
        if (ret_type.kind() == TypeKind::F32) {
            float f = 0.0f;
            std::memcpy(&f, result.xmm0, sizeof(float));
            return RuntimeValue::from_f32(f);
        } else {
            double d = 0.0;
            std::memcpy(&d, result.xmm0, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
    }
    if (ret_type.kind() == TypeKind::I32) {
        return RuntimeValue::from_i32(static_cast<int32_t>(result.rax));
    }
    if (ret_type.is_pointer()) {
        return RuntimeValue::from_ptr(static_cast<uintptr_t>(result.rax));
    }
    if (ret_type.is_gcref()) {
        return RuntimeValue::from_gcref(static_cast<uintptr_t>(result.rax));
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(result.rax));
}

} // namespace brass::codegen

#elif defined(__aarch64__) || defined(_M_ARM64)

namespace brass::codegen {

#if defined(__GNUC__) || defined(__clang__)
// File-scope asm: GCC ignores __attribute__((naked)) on AArch64. The CFI lets
// an unwinder walk from JIT code (whose frames are registered with it, see
// jit_unwind_registry) through this thunk into its C++ caller.
// partition_aarch64_invoke_args pads the stack words to an even count, so sp
// stays 16-byte aligned.
__asm__(
    ".text\n"
    BRASS_ASM_FN_BEGIN(aarch64_invoke_thunk)
    "    .cfi_startproc\n"
    "    stp x29, x30, [sp, #-32]!\n"
    "    .cfi_def_cfa_offset 32\n"
    "    .cfi_offset x29, -32\n"
    "    .cfi_offset x30, -24\n"
    "    mov x29, sp\n"
    "    .cfi_def_cfa_register x29\n"
    "    stp x19, x20, [sp, #16]\n"
    "    .cfi_offset x19, -16\n"
    "    .cfi_offset x20, -8\n"
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
    "    ldp x0, x1, [x20, #0]\n"      // x0..x7
    "    ldp x2, x3, [x20, #16]\n"
    "    ldp x4, x5, [x20, #32]\n"
    "    ldp x6, x7, [x20, #48]\n"
    "    blr x16\n"
    "    str x0, [x19, #0]\n"
    "    str x1, [x19, #8]\n"
    "    str q0, [x19, #16]\n"
    "    mov sp, x29\n"
    "    ldp x19, x20, [sp, #16]\n"
    "    ldp x29, x30, [sp], #32\n"
    "    .cfi_def_cfa sp, 0\n"
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
#if defined(__GNUC__) || defined(__clang__)
    aarch64_invoke_thunk(&invoke_args, &result);
#endif

    if (ret_type.is_void()) {
        return RuntimeValue::from_void();
    }
    if (ret_type.is_vector()) {
        return RuntimeValue::from_v128(ret_type, result.q0);
    }
    if (ret_type.is_float()) {
        if (ret_type.kind() == TypeKind::F32) {
            float f = 0.0f;
            std::memcpy(&f, result.q0, sizeof(float));
            return RuntimeValue::from_f32(f);
        } else {
            double d = 0.0;
            std::memcpy(&d, result.q0, sizeof(double));
            return RuntimeValue::from_f64(d);
        }
    }
    if (ret_type.kind() == TypeKind::I32) {
        return RuntimeValue::from_i32(static_cast<int32_t>(result.x0));
    }
    if (ret_type.is_pointer()) {
        return RuntimeValue::from_ptr(static_cast<uintptr_t>(result.x0));
    }
    if (ret_type.is_gcref()) {
        return RuntimeValue::from_gcref(static_cast<uintptr_t>(result.x0));
    }
    return RuntimeValue::from_i64(static_cast<int64_t>(result.x0));
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

