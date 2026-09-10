#include <brass/codegen/jit_exec.hpp>
#include <immintrin.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

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
#else
inline __m128 call_jit_vec1_ret_vec(void* addr, __m128 a0) {
    return reinterpret_cast<__m128(*)(__m128)>(addr)(a0);
}
inline float call_jit_vec1_ret_f32(void* addr, __m128 a0) {
    return reinterpret_cast<float(*)(__m128)>(addr)(a0);
}
inline double call_jit_vec1_ret_f64(void* addr, __m128 a0) {
    return reinterpret_cast<double(*)(__m128)>(addr)(a0);
}
inline int64_t call_jit_vec1_ret_i64(void* addr, __m128 a0) {
    return reinterpret_cast<int64_t(*)(__m128)>(addr)(a0);
}
inline __m128 call_jit_vec2_ret_vec(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<__m128(*)(__m128, __m128)>(addr)(a0, a1);
}
inline float call_jit_vec2_ret_f32(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<float(*)(__m128, __m128)>(addr)(a0, a1);
}
inline int64_t call_jit_vec2_ret_i64(void* addr, __m128 a0, __m128 a1) {
    return reinterpret_cast<int64_t(*)(__m128, __m128)>(addr)(a0, a1);
}
inline void call_jit_vec_int_ret_void(void* addr, __m128 a0, int64_t a1) {
    reinterpret_cast<void(*)(__m128, int64_t)>(addr)(a0, a1);
}
inline __m128 call_jit_vec_int_ret_vec(void* addr, __m128 a0, int64_t a1) {
    return reinterpret_cast<__m128(*)(__m128, int64_t)>(addr)(a0, a1);
}
inline int64_t call_jit_vec_int_ret_i64(void* addr, __m128 a0, int64_t a1) {
    return reinterpret_cast<int64_t(*)(__m128, int64_t)>(addr)(a0, a1);
}
inline void call_jit_int_vec_ret_void(void* addr, int64_t a0, __m128 a1) {
    reinterpret_cast<void(*)(int64_t, __m128)>(addr)(a0, a1);
}
inline __m128 call_jit_int_vec_ret_vec(void* addr, int64_t a0, __m128 a1) {
    return reinterpret_cast<__m128(*)(int64_t, __m128)>(addr)(a0, a1);
}
inline int64_t call_jit_int_vec_ret_i64(void* addr, int64_t a0, __m128 a1) {
    return reinterpret_cast<int64_t(*)(int64_t, __m128)>(addr)(a0, a1);
}
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

    // Check if arguments or return value contain floats
    bool has_float_arg = false;
    for (const auto& a : args) {
        if (a.is_f64() || a.is_f32()) has_float_arg = true;
    }

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
        } else if (args[0].is_f64() && args[1].is_f64()) {
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
        } else {
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

    // 3 to 8 arguments (Integer / Pointer paths)
    if (!has_float_arg) {
        int64_t a0 = get_int(0), a1 = get_int(1), a2 = get_int(2), a3 = get_int(3);
        int64_t a4 = get_int(4), a5 = get_int(5), a6 = get_int(6), a7 = get_int(7);

        if (ret_type.is_float()) {
            if (ret_type.kind() == TypeKind::F32) {
                auto fn8_f = reinterpret_cast<float(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>(addr);
                float r = fn8_f(a0, a1, a2, a3, a4, a5, a6, a7);
                return RuntimeValue::from_f32(r);
            }
            auto fn8_f = reinterpret_cast<double(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>(addr);
            double r = fn8_f(a0, a1, a2, a3, a4, a5, a6, a7);
            return RuntimeValue::from_f64(r);
        }

        auto fn8 = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)>(addr);
        int64_t r = fn8(a0, a1, a2, a3, a4, a5, a6, a7);

        if (ret_type.is_void()) return RuntimeValue::from_void();
        if (ret_type.kind() == TypeKind::I32) return RuntimeValue::from_i32(static_cast<int32_t>(r));
        if (ret_type.is_pointer()) return RuntimeValue::from_ptr(static_cast<uintptr_t>(r));
        if (ret_type.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(r));
        return RuntimeValue::from_i64(r);
    } else {
        // Multi-arg Float path
        double f0 = get_float(0), f1 = get_float(1), f2 = get_float(2), f3 = get_float(3);
        double f4 = get_float(4), f5 = get_float(5), f6 = get_float(6), f7 = get_float(7);

        if (ret_type.is_float()) {
            if (ret_type.kind() == TypeKind::F32) {
                auto fn_f8 = reinterpret_cast<float(*)(double, double, double, double, double, double, double, double)>(addr);
                float r = fn_f8(f0, f1, f2, f3, f4, f5, f6, f7);
                return RuntimeValue::from_f32(r);
            }
        }
        auto fn_f8 = reinterpret_cast<double(*)(double, double, double, double, double, double, double, double)>(addr);
        double r = fn_f8(f0, f1, f2, f3, f4, f5, f6, f7);
        return RuntimeValue::from_f64(r);
    }
}

} // namespace brass::codegen
