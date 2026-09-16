// Stage 5b: the two LayerNorm GPU kernels as MIR (docs/ptx_backend_design.md,
// "Stage 5b notes"). Both are one block per row (block size a multiple of 32,
// up to 1024) with a two-pass statistic -- mean first, then the mean of the
// squared deviations -- exactly like the hand-written PTX they replaced
// (ml_fusion_ptx_legacy_norm.cpp, kept for the differential tests until
// Stage 6):
//
//   fused_layernorm_modulate_kernel   y = ((x - mean) * rstd * gamma + beta)
//                                       * (1 + scale) + shift
//   fused_residual_layernorm_kernel   x += res in place (during the mean
//                                     pass); y = (x - mean) * rstd * gamma + beta
//
// Rows whose length is not a multiple of 4 take the scalar path for the whole
// row (float4 loads need 16-byte alignment). The recipe is unchanged: the
// mean pass is plain adds, the variance pass is sub + fma, mean and variance
// are div.approx by cvt.rn.f32.u32(d), rstd is rsqrt.approx(var + eps).

#include "ml_fusion_ptx_kernels_common.hpp"

namespace brass::codegen {

using namespace ptx_kernels;

namespace {

// Pass 1 of the LayerNorm-modulate kernel: per-thread row sum (plain adds,
// lane order 0..3 as the string kernel does).
Value* row_sum(KernelBuilder& kb, const RowBlock& rb, Value* x_row, Value* d) {
    Builder& b = kb.builder();
    Value* acc = kb.for_range_reduce(rb.vstart, rb.vec_d, rb.vstep, kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        Value* vx = kb.vload_f32x4(at(kb, x_row, f32_offset(kb, i)));
        for (uint32_t lane = 0; lane < 4; ++lane) acc = kb.add(acc, b.build_vextract_lane(vx, lane));
        return acc;
    });
    return kb.for_range_reduce(rb.sstart, d, rb.ntid, acc, [&](Value* i, Value* acc) {
        return kb.add(acc, kb.load_f32(at(kb, x_row, f32_offset(kb, i))));
    });
}

// Pass 2 (shared by both kernels): per-thread sum of (x - mean)^2 via fma.
Value* row_sum_sq_dev(KernelBuilder& kb, const RowBlock& rb, Value* x_row, Value* d, Value* mean) {
    Builder& b = kb.builder();
    Value* acc = kb.for_range_reduce(rb.vstart, rb.vec_d, rb.vstep, kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        Value* vx = kb.vload_f32x4(at(kb, x_row, f32_offset(kb, i)));
        for (uint32_t lane = 0; lane < 4; ++lane) {
            Value* dx = kb.sub(b.build_vextract_lane(vx, lane), mean);
            acc = b.build_fma_f32(dx, dx, acc);
        }
        return acc;
    });
    return kb.for_range_reduce(rb.sstart, d, rb.ntid, acc, [&](Value* i, Value* acc) {
        Value* dx = kb.sub(kb.load_f32(at(kb, x_row, f32_offset(kb, i))), mean);
        return b.build_fma_f32(dx, dx, acc);
    });
}

// mean = div.approx(block_sum, (f32)d)
Value* block_mean(KernelBuilder& kb, Value* partial, Value* scratch, Value* d_f32) {
    return kb.div_approx(kb.block_reduce_sum_f32(partial, scratch), d_f32);
}

// rstd = rsqrt.approx(div.approx(block_sum_sq_dev, (f32)d) + eps)
Value* block_rstd(KernelBuilder& kb, Value* partial, Value* scratch, Value* d_f32, Value* eps) {
    Value* var = kb.div_approx(kb.block_reduce_sum_f32(partial, scratch), d_f32);
    return kb.rsqrt_approx(kb.add(var, eps));
}

} // namespace

// =========================================================================
// 1. fused_layernorm_modulate_kernel(x, gamma, beta, scale, shift, y, u32 r, u32 d, f32 eps)
//    y = ((x - mean) * rstd * gamma + beta) * (1 + scale) + shift, one block per row
// =========================================================================

Function* MlFusionCompiler::build_ptx_layernorm_modulate(Module& mod) {
    Function* fn = mod.create_function("fused_layernorm_modulate_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                                        Type::i32(), Type::i32(), Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* x = p[0]; Value* gamma = p[1]; Value* beta = p[2]; Value* scale = p[3]; Value* shift = p[4];
    Value* y = p[5]; Value* rows = p[6]; Value* d = p[7]; Value* eps = p[8];

    Value* scratch = kb.shared_alloc_f32(32);
    RowBlock rb = row_block_prologue(kb, rows, d);
    Value* x_row = at(kb, x, rb.row_off);
    Value* y_row = at(kb, y, rb.row_off);
    Value* d_f32 = kb.u32_to_f32(d);

    // Pass 1: mean.  Pass 2: variance about the mean.
    Value* mean = block_mean(kb, row_sum(kb, rb, x_row, d), scratch, d_f32);
    Value* rstd = block_rstd(kb, row_sum_sq_dev(kb, rb, x_row, d, mean), scratch, d_f32, eps);

    // Pass 3: normalize, affine, modulate.
    kb.for_range(rb.vstart, rb.vec_d, rb.vstep, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* vx = kb.vload_f32x4(at(kb, x_row, off));
        Value* vg = kb.vload_f32x4(at(kb, gamma, off));
        Value* vb = kb.vload_f32x4(at(kb, beta, off));
        Value* vs = kb.vload_f32x4(at(kb, scale, off));
        Value* vh = kb.vload_f32x4(at(kb, shift, off));
        Value* t = kb.vmul(kb.vsub(vx, kb.vbroadcast(Type::f32x4(), mean)), kb.vbroadcast(Type::f32x4(), rstd));
        t = kb.vfma(t, vg, vb);
        Value* s1 = kb.vadd(vs, kb.vbroadcast(Type::f32x4(), kb.const_f32(1.0f)));
        kb.vstore_f32x4(at(kb, y_row, off), kb.vfma(t, s1, vh));
    });
    kb.for_range(rb.sstart, d, rb.ntid, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* xv = kb.load_f32(at(kb, x_row, off));
        Value* gv = kb.load_f32(at(kb, gamma, off));
        Value* bv = kb.load_f32(at(kb, beta, off));
        Value* sv = kb.load_f32(at(kb, scale, off));
        Value* hv = kb.load_f32(at(kb, shift, off));
        Value* t = kb.mul(kb.sub(xv, mean), rstd);
        t = b.build_fma_f32(t, gv, bv);
        Value* s1 = kb.add(sv, kb.const_f32(1.0f));
        kb.store_f32(at(kb, y_row, off), b.build_fma_f32(t, s1, hv));
    });

    b.build_ret_void();
    return fn;
}

// =========================================================================
// 2. fused_residual_layernorm_kernel(x, res, gamma, beta, y, u32 b, u32 d, f32 eps)
//    x[row] += res[row] (in place); y = (x - mean) * rstd * gamma + beta, one block per row
// =========================================================================

Function* MlFusionCompiler::build_ptx_residual_layernorm(Module& mod) {
    Function* fn = mod.create_function("fused_residual_layernorm_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                                        Type::i32(), Type::i32(), Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* x = p[0]; Value* res = p[1]; Value* gamma = p[2]; Value* beta = p[3]; Value* y = p[4];
    Value* rows = p[5]; Value* d = p[6]; Value* eps = p[7];

    Value* scratch = kb.shared_alloc_f32(32);
    RowBlock rb = row_block_prologue(kb, rows, d);
    Value* x_row = at(kb, x, rb.row_off);
    Value* res_row = at(kb, res, rb.row_off);
    Value* y_row = at(kb, y, rb.row_off);
    Value* d_f32 = kb.u32_to_f32(d);

    // Pass 1: x += res in place, accumulating the row sum (plain adds).
    Value* acc = kb.for_range_reduce(rb.vstart, rb.vec_d, rb.vstep, kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        Value* off = f32_offset(kb, i);
        Value* px = at(kb, x_row, off);
        Value* v = kb.vadd(kb.vload_f32x4(px), kb.vload_f32x4(at(kb, res_row, off)));
        kb.vstore_f32x4(px, v);
        for (uint32_t lane = 0; lane < 4; ++lane) acc = kb.add(acc, b.build_vextract_lane(v, lane));
        return acc;
    });
    acc = kb.for_range_reduce(rb.sstart, d, rb.ntid, acc, [&](Value* i, Value* acc) {
        Value* off = f32_offset(kb, i);
        Value* px = at(kb, x_row, off);
        Value* v = kb.add(kb.load_f32(px), kb.load_f32(at(kb, res_row, off)));
        kb.store_f32(px, v);
        return kb.add(acc, v);
    });
    Value* mean = block_mean(kb, acc, scratch, d_f32);

    // Pass 2: variance about the mean (re-reads the updated x).
    Value* rstd = block_rstd(kb, row_sum_sq_dev(kb, rb, x_row, d, mean), scratch, d_f32, eps);

    // Pass 3: y = (x - mean) * rstd * gamma + beta
    kb.for_range(rb.vstart, rb.vec_d, rb.vstep, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* vx = kb.vload_f32x4(at(kb, x_row, off));
        Value* vg = kb.vload_f32x4(at(kb, gamma, off));
        Value* vb = kb.vload_f32x4(at(kb, beta, off));
        Value* t = kb.vmul(kb.vsub(vx, kb.vbroadcast(Type::f32x4(), mean)), kb.vbroadcast(Type::f32x4(), rstd));
        kb.vstore_f32x4(at(kb, y_row, off), kb.vfma(t, vg, vb));
    });
    kb.for_range(rb.sstart, d, rb.ntid, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* xv = kb.load_f32(at(kb, x_row, off));
        Value* gv = kb.load_f32(at(kb, gamma, off));
        Value* bv = kb.load_f32(at(kb, beta, off));
        Value* t = kb.mul(kb.sub(xv, mean), rstd);
        kb.store_f32(at(kb, y_row, off), b.build_fma_f32(t, gv, bv));
    });

    b.build_ret_void();
    return fn;
}

} // namespace brass::codegen
