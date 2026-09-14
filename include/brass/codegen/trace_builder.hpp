#pragma once

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>

namespace brass::codegen {

// ─── TraceBuilder ───────────────────────────────────────────────────────────
//
// Specialized MIR builder for automatic tracing JIT kernels. Provides vectorized
// activation functions (SiLU, GELU, ReLU) using branchless minimax polynomials
// and vector reduction helpers.
class TraceBuilder : public KernelBuilder {
public:
    TraceBuilder(Module& mod, Function* fn)
        : KernelBuilder(mod, fn) {}

    // Inlined vector SiLU for AVX2 8-wide: silu(x) = x / (1 + exp(-x))
    // Uses 5th-order polynomial with 1/16 scaling and 4 squarings (p^16)
    Value* vsilu_f32x8(Value* x) {
        Value* c_neg1 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(-1.0f));
        Value* c_inv16 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.0625f));
        Value* neg_x = vmul(x, c_neg1);
        Value* u = vmul(neg_x, c_inv16);

        Value* c_1_120 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 120.0f));
        Value* c_1_24  = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 24.0f));
        Value* c_1_6   = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 6.0f));
        Value* c_1_2   = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.5f));
        Value* c_1_0   = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f));

        Value* p5 = vfma(u, c_1_120, c_1_24);
        Value* p4 = vfma(u, p5, c_1_6);
        Value* p3 = vfma(u, p4, c_1_2);
        Value* p2 = vfma(u, p3, c_1_0);
        Value* p1 = vfma(u, p2, c_1_0);

        Value* sq1 = vmul(p1, p1);
        Value* sq2 = vmul(sq1, sq1);
        Value* sq3 = vmul(sq2, sq2);
        Value* exp_neg_x = vmul(sq3, sq3);

        Value* denom = vadd(c_1_0, exp_neg_x);
        return vdiv(x, denom);
    }

    // Scalar SiLU: silu(x) = x / (1 + exp(-x))
    Value* silu_f32(Value* x) {
        Value* neg_x = builder().build_neg(x);
        Value* c_inv16 = builder().build_fconst_f32(0.0625f);
        Value* u = mul(neg_x, c_inv16);

        Value* c_1_120 = builder().build_fconst_f32(1.0f / 120.0f);
        Value* c_1_24  = builder().build_fconst_f32(1.0f / 24.0f);
        Value* c_1_6   = builder().build_fconst_f32(1.0f / 6.0f);
        Value* c_1_2   = builder().build_fconst_f32(0.5f);
        Value* c_1_0   = builder().build_fconst_f32(1.0f);

        Value* p5 = builder().build_fma_f32(u, c_1_120, c_1_24);
        Value* p4 = builder().build_fma_f32(u, p5, c_1_6);
        Value* p3 = builder().build_fma_f32(u, p4, c_1_2);
        Value* p2 = builder().build_fma_f32(u, p3, c_1_0);
        Value* p1 = builder().build_fma_f32(u, p2, c_1_0);

        Value* sq1 = mul(p1, p1);
        Value* sq2 = mul(sq1, sq1);
        Value* sq3 = mul(sq2, sq2);
        Value* exp_neg_x = mul(sq3, sq3);

        Value* denom = add(c_1_0, exp_neg_x);
        return div(x, denom);
    }

    // Vector ReLU: max(x, 0)
    Value* vrelu_f32x8(Value* x) {
        return relu(x);
    }

    // Scalar ReLU: max(x, 0)
    Value* relu_f32(Value* x) {
        return relu(x);
    }

    // Vector GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Value* vgelu_f32x8(Value* x) {
        Value* c_05 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.5f));
        Value* c_10 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f));
        Value* c_sqrt_2_pi = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.7978845608f));
        Value* c_coeff = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.044715f));

        Value* x2 = vmul(x, x);
        Value* x3 = vmul(x2, x);
        Value* inner = vfma(c_coeff, x3, x);
        Value* arg = vmul(c_sqrt_2_pi, inner);

        // tanh(y) = 2 / (1 + exp(-2y)) - 1
        Value* c_neg2 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(-2.0f));
        Value* neg_2y = vmul(arg, c_neg2);
        Value* c_inv16 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.0625f));
        Value* u = vmul(neg_2y, c_inv16);

        Value* c_1_120 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 120.0f));
        Value* c_1_24  = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 24.0f));
        Value* c_1_6   = vbroadcast(Type::f32x8(), builder().build_fconst_f32(1.0f / 6.0f));
        Value* c_1_2   = vbroadcast(Type::f32x8(), builder().build_fconst_f32(0.5f));

        Value* p5 = vfma(u, c_1_120, c_1_24);
        Value* p4 = vfma(u, p5, c_1_6);
        Value* p3 = vfma(u, p4, c_1_2);
        Value* p2 = vfma(u, p3, c_10);
        Value* p1 = vfma(u, p2, c_10);

        Value* sq1 = vmul(p1, p1);
        Value* sq2 = vmul(sq1, sq1);
        Value* sq3 = vmul(sq2, sq2);
        Value* exp_val = vmul(sq3, sq3);

        Value* denom = vadd(c_10, exp_val);
        Value* c_20 = vbroadcast(Type::f32x8(), builder().build_fconst_f32(2.0f));
        Value* two_over_denom = vdiv(c_20, denom);
        Value* tanh_val = vsub(two_over_denom, c_10);

        Value* one_plus_tanh = vadd(c_10, tanh_val);
        Value* half_x = vmul(c_05, x);
        return vmul(half_x, one_plus_tanh);
    }

    // Scalar GELU
    Value* gelu_f32(Value* x) {
        Value* c_05 = builder().build_fconst_f32(0.5f);
        Value* c_10 = builder().build_fconst_f32(1.0f);
        Value* c_sqrt_2_pi = builder().build_fconst_f32(0.7978845608f);
        Value* c_coeff = builder().build_fconst_f32(0.044715f);

        Value* x2 = mul(x, x);
        Value* x3 = mul(x2, x);
        Value* inner = builder().build_fma_f32(c_coeff, x3, x);
        Value* arg = mul(c_sqrt_2_pi, inner);

        Value* c_neg2 = builder().build_fconst_f32(-2.0f);
        Value* neg_2y = mul(arg, c_neg2);
        Value* c_inv16 = builder().build_fconst_f32(0.0625f);
        Value* u = mul(neg_2y, c_inv16);

        Value* c_1_120 = builder().build_fconst_f32(1.0f / 120.0f);
        Value* c_1_24  = builder().build_fconst_f32(1.0f / 24.0f);
        Value* c_1_6   = builder().build_fconst_f32(1.0f / 6.0f);
        Value* c_1_2   = builder().build_fconst_f32(0.5f);

        Value* p5 = builder().build_fma_f32(u, c_1_120, c_1_24);
        Value* p4 = builder().build_fma_f32(u, p5, c_1_6);
        Value* p3 = builder().build_fma_f32(u, p4, c_1_2);
        Value* p2 = builder().build_fma_f32(u, p3, c_10);
        Value* p1 = builder().build_fma_f32(u, p2, c_10);

        Value* sq1 = mul(p1, p1);
        Value* sq2 = mul(sq1, sq1);
        Value* sq3 = mul(sq2, sq2);
        Value* exp_val = mul(sq3, sq3);

        Value* denom = add(c_10, exp_val);
        Value* c_20 = builder().build_fconst_f32(2.0f);
        Value* two_over_denom = div(c_20, denom);
        Value* tanh_val = sub(two_over_denom, c_10);

        Value* one_plus_tanh = add(c_10, tanh_val);
        Value* half_x = mul(c_05, x);
        return mul(half_x, one_plus_tanh);
    }

    // Horizontal sum of an 8-wide float vector using destination scratch buffer
    Value* reduce_vsum8(Value* scratch, Value* vsum) {
        vstore_f32x8(scratch, vsum);
        Value* s0 = load_f32(scratch, 0);  Value* s1 = load_f32(scratch, 4);
        Value* s2 = load_f32(scratch, 8);  Value* s3 = load_f32(scratch, 12);
        Value* s4 = load_f32(scratch, 16); Value* s5 = load_f32(scratch, 20);
        Value* s6 = load_f32(scratch, 24); Value* s7 = load_f32(scratch, 28);
        Value* sum0123 = add(add(s0, s1), add(s2, s3));
        Value* sum4567 = add(add(s4, s5), add(s6, s7));
        return add(sum0123, sum4567);
    }
};

} // namespace brass::codegen
