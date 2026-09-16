// Stage 5a: GPU kernels written as MIR through the KernelBuilder GPU helpers
// (docs/ptx_backend_design.md, "Stage 5a notes"). Each builder reproduces the
// launch contract and the numerical recipe of the hand-written PTX kernel it
// replaced (src/codegen/ml_fusion_ptx_legacy.cpp, kept only for the
// differential tests until Stage 6):
//
//   fused_swiglu_kernel                grid-stride over n/4 float4s, then a
//                                      scalar tail [n & ~3, n) walked by every
//                                      block with stride ntid (as before)
//   fused_adaln_modulate[_gated]_kernel one block per row; d & ~3 float4s per
//                                      block then the scalar remainder
//   fused_residual_rms_norm_kernel     one block per row; x += res in place,
//                                      block sum of squares, y = x*gamma*rrms
//
// Rows whose length is not a multiple of 4 take the scalar path for the whole
// row (the float4 loads need 16-byte alignment). SiLU is ex2.approx +
// rcp.approx and the RMS is div.approx + rsqrt.approx, exactly as before, so
// the tolerances in test_gpu_execution.cpp are unchanged.

#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>

namespace brass::codegen {

namespace {

// ---------------------------------------------------------------------------
// Small addressing / math helpers shared by the kernels
// ---------------------------------------------------------------------------

// Byte offset of f32 element `i` (i32): (u64)i << 2.
Value* f32_offset(KernelBuilder& kb, Value* i) {
    Builder& b = kb.builder();
    return b.build_shl(b.build_zext_i64(i), kb.const_i32(2));
}

Value* at(KernelBuilder& kb, Value* base, Value* byte_off) {
    return kb.builder().build_add(base, byte_off);
}

// silu(g) = g * rcp(1 + ex2(-g * log2 e))  (the fast recipe of the string kernel)
Value* silu_fast(KernelBuilder& kb, Value* g) {
    Value* e = kb.exp_fast(kb.builder().build_neg(g));
    Value* r = kb.rcp_approx(kb.add(e, kb.const_f32(1.0f)));
    return kb.mul(g, r);
}

// Row-per-block prologue: returns after emitting `if (row >= rows) return;`.
// vec_d is d & ~3, or 0 when d % 4 != 0 (rows are then not 16-byte aligned
// and the whole row takes the scalar path). vstart/vstep are the float4 loop
// bounds for this thread; the scalar remainder runs from vec_d + tid by ntid.
struct RowBlock {
    Value* row_off;   // i64 byte offset of this block's row
    Value* tid;
    Value* ntid;
    Value* vec_d;
    Value* vstart;    // tid * 4
    Value* vstep;     // ntid * 4
    Value* sstart;    // vec_d + tid
};

RowBlock row_block_prologue(KernelBuilder& kb, Value* rows, Value* d) {
    Builder& b = kb.builder();
    Value* row = kb.ctaid_x();
    kb.if_then(b.build_uge(row, rows), [&] { b.build_ret_void(); });

    RowBlock rb;
    rb.row_off = b.build_shl(b.build_mul(b.build_zext_i64(row), b.build_zext_i64(d)), kb.const_i32(2));
    rb.tid = kb.tid_x();
    rb.ntid = kb.ntid_x();
    Value* aligned = b.build_and(d, kb.const_i32(~3));
    Value* misaligned = b.build_ne(b.build_and(d, kb.const_i32(3)), kb.const_i32(0));
    rb.vec_d = b.build_select(misaligned, kb.const_i32(0), aligned);
    rb.vstart = b.build_shl(rb.tid, kb.const_i32(2));
    rb.vstep = b.build_shl(rb.ntid, kb.const_i32(2));
    rb.sstart = kb.add(rb.vec_d, rb.tid);
    return rb;
}

// Creates the entry block with one block parameter per kernel parameter.
std::vector<Value*> entry_params(KernelBuilder& kb, Function* fn) {
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    std::vector<Value*> out;
    for (Type t : fn->param_types()) out.push_back(b.add_block_param(entry, t));
    return out;
}

} // namespace

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
