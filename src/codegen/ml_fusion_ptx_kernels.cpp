// Element-wise and RMSNorm GPU kernels written as MIR through the
// KernelBuilder GPU helpers (docs/ptx_kernel_authoring.md):
//
//   fused_swiglu_kernel                grid-stride over n/4 float4s, then a
//                                      scalar tail [n & ~3, n) walked by every
//                                      block with stride ntid
//   fused_adaln_modulate[_gated]_kernel one block per row; d & ~3 float4s per
//                                      block then the scalar remainder
//   fused_residual_rms_norm_kernel     one block per row; x += res in place,
//                                      block sum of squares, y = x*gamma*rrms
//
// Rows whose length is not a multiple of 4 take the scalar path for the whole
// row (the float4 loads need 16-byte alignment). SiLU is ex2.approx +
// rcp.approx and the RMS is div.approx + rsqrt.approx; the tolerances in
// test_gpu_execution.cpp and test_gpu_kernels.cpp assume these recipes.
//
// The addressing / prologue helpers (f32_offset, at, silu_fast,
// row_block_prologue, entry_params) live in ml_fusion_ptx_kernels_common.hpp,
// shared with the LayerNorm, GEMV and quantized GEMV kernel files.

#include "ml_fusion_ptx_kernels_common.hpp"

namespace brass::codegen {

using namespace ptx_kernels;

// =========================================================================
// 1. fused_swiglu_kernel(gate, up, out, u32 n): grid-stride SiLU(gate) * up
// =========================================================================

Function* MlFusionCompiler::build_ptx_swiglu(Module& mod) {
    Function* fn = mod.create_function("fused_swiglu_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* gate = p[0]; Value* up = p[1]; Value* out = p[2]; Value* n = p[3];

    Value* tid = kb.tid_x();
    Value* ntid = kb.ntid_x();
    Value* vidx = kb.global_tid_x();
    Value* vstride = kb.mul(kb.nctaid_x(), ntid);
    Value* nvec = b.build_lshr(n, kb.const_i32(2));

    // float4 body: out[4i..4i+3] = silu(gate) * up
    kb.for_range(vidx, nvec, vstride, [&](Value* i) {
        Value* off = b.build_shl(b.build_zext_i64(i), kb.const_i32(4));
        Value* vg = kb.vload_f32x4(at(kb, gate, off));
        Value* vu = kb.vload_f32x4(at(kb, up, off));
        Value* res = vg;
        for (uint32_t lane = 0; lane < 4; ++lane) {
            Value* g = b.build_vextract_lane(vg, lane);
            Value* u = b.build_vextract_lane(vu, lane);
            res = b.build_vinsert_lane(res, kb.mul(silu_fast(kb, g), u), lane);
        }
        kb.vstore_f32x4(at(kb, out, off), res);
    });

    // scalar tail [nvec * 4, n), every block walks it with stride ntid
    Value* tail_start = kb.add(b.build_shl(nvec, kb.const_i32(2)), tid);
    kb.for_range(tail_start, n, ntid, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* g = kb.load_f32(at(kb, gate, off));
        Value* u = kb.load_f32(at(kb, up, off));
        kb.store_f32(at(kb, out, off), kb.mul(silu_fast(kb, g), u));
    });

    b.build_ret_void();
    return fn;
}

// =========================================================================
// 2. fused_adaln_modulate[_gated]_kernel(x, scale, shift, [gate,] y, u32 l, u32 d)
//    y[row] = x[row] * (1 + scale) + shift [* gate], one block per row
// =========================================================================

Function* MlFusionCompiler::build_ptx_adaln_modulate(Module& mod, bool gated) {
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::ptr()};
    if (gated) params.push_back(Type::ptr());
    params.insert(params.end(), {Type::ptr(), Type::i32(), Type::i32()});
    Function* fn = mod.create_function(gated ? "fused_adaln_modulate_gated_kernel" : "fused_adaln_modulate_kernel",
                                       Type::void_type(), params);
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    size_t k = 0;
    Value* x = p[k++]; Value* scale = p[k++]; Value* shift = p[k++];
    Value* gate = gated ? p[k++] : nullptr;
    Value* y = p[k++]; Value* l = p[k++]; Value* d = p[k++];

    RowBlock rb = row_block_prologue(kb, l, d);
    Value* x_row = at(kb, x, rb.row_off);
    Value* y_row = at(kb, y, rb.row_off);

    // float4 path
    kb.for_range(rb.vstart, rb.vec_d, rb.vstep, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* vx = kb.vload_f32x4(at(kb, x_row, off));
        Value* vs = kb.vload_f32x4(at(kb, scale, off));
        Value* vh = kb.vload_f32x4(at(kb, shift, off));
        Value* ones = kb.vbroadcast(Type::f32x4(), kb.const_f32(1.0f));
        Value* v = kb.vfma(vx, kb.vadd(vs, ones), vh);
        if (gated) v = kb.vmul(v, kb.vload_f32x4(at(kb, gate, off)));
        kb.vstore_f32x4(at(kb, y_row, off), v);
    });

    // scalar remainder
    kb.for_range(rb.sstart, d, rb.ntid, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* xv = kb.load_f32(at(kb, x_row, off));
        Value* sv = kb.load_f32(at(kb, scale, off));
        Value* hv = kb.load_f32(at(kb, shift, off));
        Value* v = b.build_fma_f32(xv, kb.add(sv, kb.const_f32(1.0f)), hv);
        if (gated) v = kb.mul(v, kb.load_f32(at(kb, gate, off)));
        kb.store_f32(at(kb, y_row, off), v);
    });

    b.build_ret_void();
    return fn;
}

// =========================================================================
// 3. fused_residual_rms_norm_kernel(x, res, gamma, y, u32 b, u32 d, f32 eps)
//    x[row] += res[row] (in place); y[row] = x[row] * gamma * rsqrt(mean(x^2) + eps)
//    One block per row (block size a multiple of 32), 32-float shared scratch.
// =========================================================================

Function* MlFusionCompiler::build_ptx_residual_rms_norm(Module& mod) {
    Function* fn = mod.create_function("fused_residual_rms_norm_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                                        Type::i32(), Type::i32(), Type::f32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* x = p[0]; Value* res = p[1]; Value* gamma = p[2]; Value* y = p[3];
    Value* rows = p[4]; Value* d = p[5]; Value* eps = p[6];

    Value* scratch = kb.shared_alloc_f32(32);
    RowBlock rb = row_block_prologue(kb, rows, d);
    Value* x_row = at(kb, x, rb.row_off);
    Value* res_row = at(kb, res, rb.row_off);
    Value* y_row = at(kb, y, rb.row_off);

    // Pass 1: x += res in place, accumulate sum of squares per thread.
    Value* acc = kb.for_range_reduce(rb.vstart, rb.vec_d, rb.vstep, kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        Value* off = f32_offset(kb, i);
        Value* px = at(kb, x_row, off);
        Value* v = kb.vadd(kb.vload_f32x4(px), kb.vload_f32x4(at(kb, res_row, off)));
        kb.vstore_f32x4(px, v);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            Value* e = b.build_vextract_lane(v, lane);
            acc = b.build_fma_f32(e, e, acc);
        }
        return acc;
    });
    acc = kb.for_range_reduce(rb.sstart, d, rb.ntid, acc, [&](Value* i, Value* acc) {
        Value* off = f32_offset(kb, i);
        Value* px = at(kb, x_row, off);
        Value* v = kb.add(kb.load_f32(px), kb.load_f32(at(kb, res_row, off)));
        kb.store_f32(px, v);
        return b.build_fma_f32(v, v, acc);
    });

    // Block total -> every thread computes the same rrms (div.approx + rsqrt.approx).
    Value* total = kb.block_reduce_sum_f32(acc, scratch);
    Value* mean_sq = kb.div_approx(total, kb.u32_to_f32(d));
    Value* rrms = kb.rsqrt_approx(kb.add(mean_sq, eps));

    // Pass 2: y = x * gamma * rrms
    kb.for_range(rb.vstart, rb.vec_d, rb.vstep, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* vx = kb.vload_f32x4(at(kb, x_row, off));
        Value* vg = kb.vload_f32x4(at(kb, gamma, off));
        Value* vr = kb.vbroadcast(Type::f32x4(), rrms);
        kb.vstore_f32x4(at(kb, y_row, off), kb.vmul(kb.vmul(vx, vg), vr));
    });
    kb.for_range(rb.sstart, d, rb.ntid, [&](Value* i) {
        Value* off = f32_offset(kb, i);
        Value* xv = kb.load_f32(at(kb, x_row, off));
        Value* gv = kb.load_f32(at(kb, gamma, off));
        kb.store_f32(at(kb, y_row, off), kb.mul(kb.mul(xv, gv), rrms));
    });

    b.build_ret_void();
    return fn;
}

} // namespace brass::codegen
