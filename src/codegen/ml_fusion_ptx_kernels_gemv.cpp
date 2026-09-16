// Stage 5b: the two single-token (M = 1) GEMV GPU kernels as MIR
// (docs/ptx_backend_design.md, "Stage 5b notes"). One block per output row
// (block size a multiple of 32, up to 1024); the block dots its weight row(s)
// with x, reduces, and thread 0 writes y[row]. They reproduce the hand-written
// PTX they replaced (ml_fusion_ptx_legacy_gemv.cpp, kept for the differential
// tests until Stage 6):
//
//   fused_gemv_swiglu_kernel(w_gate, w_up, x, y, u32 n, u32 k)
//       y[row] = silu(w_gate[row] . x) * (w_up[row] . x)
//   fused_gemv_residual_kernel(w_down, x, res, y, u32 n, u32 k)
//       y[row] = w_down[row] . x + res[row]
//
// The K loop is float4 loads with a scalar fma chain per lane (lane order
// 0..3, both dot products carried through one loop) over [tid*4, k & ~3) by
// ntid*4, then a scalar tail over [k & ~3, k) by ntid. When k % 4 != 0 the
// rows are not 16-byte aligned and the whole row goes through the scalar
// tail, as before. The SiLU is the string kernel's clamped fast recipe:
// t = clamp(g * -log2e, -88, 88); g * rcp.approx(1 + ex2.approx(t)).

#include "ml_fusion_ptx_kernels_common.hpp"

namespace brass::codegen {

using namespace ptx_kernels;

namespace {

// Row-per-block prologue for the GEMV kernels: `if (row >= n) return;`, the
// weight-row byte offset, and the K-loop bounds (k_vec = k & ~3, or 0 when
// k % 4 != 0). Identical to row_block_prologue with d = k.
struct GemvRow {
    RowBlock rb;
    Value* row_byte_off;  // (u64)row << 2: offset of y[row] / res[row]
};

GemvRow gemv_prologue(KernelBuilder& kb, Value* n, Value* k) {
    GemvRow g;
    g.rb = row_block_prologue(kb, n, k);
    g.row_byte_off = f32_offset(kb, g.rb.row);
    return g;
}

// Dot products of `x` with each of `w_rows` over the row, per thread: the
// float4 loop then the scalar tail, one accumulator per weight row, each lane
// folded with fma in order (the string kernels' exact chain).
std::vector<Value*> gemv_partial_dots(KernelBuilder& kb, const RowBlock& rb, const std::vector<Value*>& w_rows,
                                      Value* x, Value* k) {
    Builder& b = kb.builder();
    std::vector<Value*> zeros(w_rows.size(), kb.const_f32(0.0f));
    std::vector<Value*> accs = kb.for_range_reduce_n(rb.vstart, rb.vec_d, rb.vstep, zeros,
        [&](Value* i, const std::vector<Value*>& accs) {
            Value* off = f32_offset(kb, i);
            std::vector<Value*> vw;
            for (Value* w : w_rows) vw.push_back(kb.vload_f32x4(at(kb, w, off)));
            Value* vx = kb.vload_f32x4(at(kb, x, off));
            std::vector<Value*> next = accs;
            for (size_t r = 0; r < w_rows.size(); ++r) {
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    next[r] = b.build_fma_f32(b.build_vextract_lane(vw[r], lane), b.build_vextract_lane(vx, lane), next[r]);
                }
            }
            return next;
        });
    return kb.for_range_reduce_n(rb.sstart, k, rb.ntid, accs, [&](Value* i, const std::vector<Value*>& accs) {
        Value* off = f32_offset(kb, i);
        std::vector<Value*> wv;
        for (Value* w : w_rows) wv.push_back(kb.load_f32(at(kb, w, off)));
        Value* xv = kb.load_f32(at(kb, x, off));
        std::vector<Value*> next = accs;
        for (size_t r = 0; r < w_rows.size(); ++r) next[r] = b.build_fma_f32(wv[r], xv, next[r]);
        return next;
    });
}

// silu(g) with the GEMV string kernel's clamp: t = max(min(g * -log2e, 88), -88)
// -> g * rcp(1 + ex2(t)). (Clamping keeps ex2 finite for |g| > 61.)
Value* silu_fast_clamped(KernelBuilder& kb, Value* g) {
    Value* t = kb.mul(g, kb.const_f32(-1.44269504f));
    t = kb.fmax(t, kb.const_f32(-88.0f));
    t = kb.fmin(t, kb.const_f32(88.0f));
    Value* e = kb.ex2_approx(t);
    Value* r = kb.rcp_approx(kb.add(kb.const_f32(1.0f), e));
    return kb.mul(g, r);
}

} // namespace

// =========================================================================
// 1. fused_gemv_swiglu_kernel(w_gate, w_up, x, y, u32 n, u32 k)
// =========================================================================

Function* MlFusionCompiler::build_ptx_gemv_swiglu(Module& mod) {
    Function* fn = mod.create_function("fused_gemv_swiglu_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* w_gate = p[0]; Value* w_up = p[1]; Value* x = p[2]; Value* y = p[3]; Value* n = p[4]; Value* k = p[5];

    Value* scratch = kb.shared_alloc_f32(32);
    GemvRow g = gemv_prologue(kb, n, k);
    Value* gate_row = at(kb, w_gate, g.rb.row_off);
    Value* up_row = at(kb, w_up, g.rb.row_off);

    std::vector<Value*> partial = gemv_partial_dots(kb, g.rb, {gate_row, up_row}, x, k);
    Value* gate = kb.block_reduce_sum_f32(partial[0], scratch);
    Value* up = kb.block_reduce_sum_f32(partial[1], scratch);

    kb.if_then(b.build_eq(g.rb.tid, kb.const_i32(0)), [&] {
        kb.store_f32(at(kb, y, g.row_byte_off), kb.mul(silu_fast_clamped(kb, gate), up));
    });
    b.build_ret_void();
    return fn;
}

// =========================================================================
// 2. fused_gemv_residual_kernel(w_down, x, res, y, u32 n, u32 k)
// =========================================================================

Function* MlFusionCompiler::build_ptx_gemv_residual(Module& mod) {
    Function* fn = mod.create_function("fused_gemv_residual_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* w_down = p[0]; Value* x = p[1]; Value* res = p[2]; Value* y = p[3]; Value* n = p[4]; Value* k = p[5];

    Value* scratch = kb.shared_alloc_f32(32);
    GemvRow g = gemv_prologue(kb, n, k);
    Value* down_row = at(kb, w_down, g.rb.row_off);

    std::vector<Value*> partial = gemv_partial_dots(kb, g.rb, {down_row}, x, k);
    Value* dot = kb.block_reduce_sum_f32(partial[0], scratch);

    kb.if_then(b.build_eq(g.rb.tid, kb.const_i32(0)), [&] {
        Value* r = kb.load_f32(at(kb, res, g.row_byte_off));
        kb.store_f32(at(kb, y, g.row_byte_off), kb.add(dot, r));
    });
    b.build_ret_void();
    return fn;
}

} // namespace brass::codegen
