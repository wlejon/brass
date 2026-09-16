// Stage 5c: the two single-token (M = 1) quantized GEMV GPU kernels as MIR
// (docs/ptx_backend_design.md, "Stage 5c notes"). One block per output row;
// the block dequantizes its weight row on the fly, dots it with x, reduces,
// and thread 0 writes y[row]. They reproduce the hand-written PTX they
// replaced (ml_fusion_ptx_legacy_quant.cpp, kept for the differential tests
// until Stage 6) and the block formats of the CPU dequantizers in
// ml_fusion_quant_cpu.cpp (brass_dequant_q8_0_block / brass_dequant_q4k_block):
//
//   fused_gemv_q8_0_kernel(w, x, y, u32 n, u32 k)     k % 32 == 0
//       Q8_0 block (34 bytes): f16 d, int8 qs[32]; w = d * qs[i]
//       8 threads per block (4 int8 / 4 floats each); block size % 32 == 0
//   fused_gemv_q4_k_kernel(w, x, y, u32 n, u32 k)     k % 256 == 0
//       Q4_K super-block (144 bytes): f16 d, f16 dmin, u8 scales[12], u8 qs[128];
//       8 sub-blocks of 32 with 6-bit sc/m (get_scale_min_k4 layout);
//       w = d * sc[is] * nibble - dmin * m[is]; sub-block 2p is the low nibble
//       of qs[32p..32p+31], sub-block 2p + 1 the high nibble.
//       64 threads per super-block (is = sub-block, lg = quad); block size % 64 == 0
//
// Thread mapping and arithmetic are the string kernels': blocks_per_row =
// k >> 5 (k >> 8), each thread walks the row's blocks from sb_local = tid /
// threads_per_block by ntid / threads_per_block, loads the header, its 4
// weights and the matching float4 of x, and folds the four products with
// fma in lane order into one f32 accumulator; then block_reduce_sum_f32.
// The string kernels stepped by a hard-coded 32 (Q8_0) / 4 (Q4_K) -- i.e.
// they assumed a 256-thread block; the MIR kernels derive the stride from
// ntid, which is identical at 256 and correct for the other block sizes.

#include "ml_fusion_ptx_kernels_common.hpp"

namespace brass::codegen {

using namespace ptx_kernels;

namespace {

// Row-per-block prologue shared by both kernels: `if (row >= n) return;`,
// blocks_per_row = k >> block_k_log2, row_base = w + row * blocks_per_row *
// block_bytes (u64), and this thread's first block / stride over the row's
// blocks with threads_per_block = 1 << tpb_log2 threads cooperating on each.
struct QuantRow {
    Value* row;
    Value* tid;
    Value* ntid;
    Value* bpr;        // i32 blocks per row
    Value* row_base;   // ptr to this row's first block
    Value* sb_local;   // tid >> tpb_log2
    Value* stride;     // ntid >> tpb_log2
};

QuantRow quant_prologue(KernelBuilder& kb, Value* w, Value* n, Value* k, int block_k_log2, int64_t block_bytes,
                        int tpb_log2) {
    Builder& b = kb.builder();
    QuantRow q;
    q.row = kb.ctaid_x();
    kb.if_then(b.build_uge(q.row, n), [&] { b.build_ret_void(); });
    q.bpr = b.build_lshr(k, kb.const_i32(block_k_log2));
    Value* row_blocks = b.build_mul(b.build_zext_i64(q.row), b.build_zext_i64(q.bpr));
    q.row_base = at(kb, w, b.build_mul(row_blocks, kb.const_i64(block_bytes)));
    q.tid = kb.tid_x();
    q.ntid = kb.ntid_x();
    q.sb_local = b.build_lshr(q.tid, kb.const_i32(tpb_log2));
    q.stride = b.build_lshr(q.ntid, kb.const_i32(tpb_log2));
    return q;
}

// Pointer to block `sb` of the row.
Value* block_ptr(KernelBuilder& kb, const QuantRow& q, Value* sb, int64_t block_bytes) {
    Builder& b = kb.builder();
    return at(kb, q.row_base, b.build_mul(b.build_zext_i64(sb), kb.const_i64(block_bytes)));
}

// The float4 of x at element index `elem` (i32).
Value* x_float4(KernelBuilder& kb, Value* x, Value* elem) {
    return kb.vload_f32x4(at(kb, x, f32_offset(kb, elem)));
}

// acc' = fma(w, x[lane], acc) for the four lanes in order.
Value* fma_lanes(KernelBuilder& kb, Value* const w4[4], Value* xv, Value* acc) {
    Builder& b = kb.builder();
    for (uint32_t lane = 0; lane < 4; ++lane) {
        acc = b.build_fma_f32(w4[lane], b.build_vextract_lane(xv, lane), acc);
    }
    return acc;
}

// Epilogue shared by both kernels: block sum, thread 0 stores y[row].
void store_row_sum(KernelBuilder& kb, const QuantRow& q, Value* y, Value* acc, Value* scratch) {
    Builder& b = kb.builder();
    Value* total = kb.block_reduce_sum_f32(acc, scratch);
    kb.if_then(b.build_eq(q.tid, kb.const_i32(0)), [&] {
        kb.store_f32(at(kb, y, f32_offset(kb, q.row)), total);
    });
    b.build_ret_void();
}

} // namespace

// =========================================================================
// 1. fused_gemv_q8_0_kernel(w, x, y, u32 n, u32 k)
// =========================================================================

Function* MlFusionCompiler::build_ptx_gemv_q8_0(Module& mod) {
    constexpr int64_t kBlockBytes = 34;   // f16 d + 32 x int8
    Function* fn = mod.create_function("fused_gemv_q8_0_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* w = p[0]; Value* x = p[1]; Value* y = p[2]; Value* n = p[3]; Value* k = p[4];

    Value* scratch = kb.shared_alloc_f32(32);
    QuantRow q = quant_prologue(kb, w, n, k, /*block_k_log2=*/5, kBlockBytes, /*tpb_log2=*/3);
    Value* lane4 = b.build_shl(b.build_and(q.tid, kb.const_i32(7)), kb.const_i32(2)); // (tid & 7) * 4
    Value* qs_off = b.build_zext_i64(kb.add(lane4, kb.const_i32(2)));                 // 2 + lane * 4

    Value* acc = kb.for_range_reduce(q.sb_local, q.bpr, q.stride, kb.const_f32(0.0f), [&](Value* sb, Value* acc) {
        Value* blk = block_ptr(kb, q, sb, kBlockBytes);
        Value* d = kb.f16_to_f32(kb.load_u16(blk));
        // Four int8 at blk + 2 + lane * 4 as two 16-bit loads (2-byte aligned only)
        Value* qs = at(kb, blk, qs_off);
        Value* q4 = b.build_or(kb.load_u16(qs, 0), b.build_shl(kb.load_u16(qs, 2), kb.const_i32(16)));
        Value* xv = x_float4(kb, x, kb.add(b.build_shl(sb, kb.const_i32(5)), lane4));
        Value* w4[4];
        for (int j = 0; j < 4; ++j) {
            // sign-extend byte j: (q4 << (24 - 8j)) >> 24 (arithmetic)
            Value* shifted = j == 3 ? q4 : b.build_shl(q4, kb.const_i32(24 - 8 * j));
            Value* qi = b.build_ashr(shifted, kb.const_i32(24));
            w4[j] = kb.mul(kb.i32_to_f32(qi), d);
        }
        return fma_lanes(kb, w4, xv, acc);
    });
    store_row_sum(kb, q, y, acc, scratch);
    return fn;
}

// =========================================================================
// 2. fused_gemv_q4_k_kernel(w, x, y, u32 n, u32 k)
// =========================================================================

Function* MlFusionCompiler::build_ptx_gemv_q4_k(Module& mod) {
    constexpr int64_t kBlockBytes = 144;  // 16-byte header + 128 bytes of nibbles
    Function* fn = mod.create_function("fused_gemv_q4_k_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    std::vector<Value*> p = entry_params(kb, fn);
    Value* w = p[0]; Value* x = p[1]; Value* y = p[2]; Value* n = p[3]; Value* k = p[4];

    Value* scratch = kb.shared_alloc_f32(32);
    QuantRow q = quant_prologue(kb, w, n, k, /*block_k_log2=*/8, kBlockBytes, /*tpb_log2=*/6);

    // t = tid & 63; is = t >> 3 (sub-block 0..7); lg = t & 7 (quad within the sub-block)
    Value* t = b.build_and(q.tid, kb.const_i32(63));
    Value* is = b.build_lshr(t, kb.const_i32(3));
    Value* lg4 = b.build_shl(b.build_and(t, kb.const_i32(7)), kb.const_i32(2));      // lg * 4
    Value* hi4 = b.build_shl(b.build_and(is, kb.const_i32(1)), kb.const_i32(2));     // 4 when the high nibble
    Value* pair32 = b.build_shl(b.build_lshr(is, kb.const_i32(1)), kb.const_i32(5)); // (is >> 1) * 32
    Value* qs_off = b.build_zext_i64(kb.add(kb.add(pair32, lg4), kb.const_i32(16))); // 16 + pair * 32 + lg * 4
    Value* xoff = kb.add(b.build_shl(is, kb.const_i32(5)), lg4);                     // is * 32 + lg * 4
    Value* sc_shift = b.build_shl(b.build_and(is, kb.const_i32(3)), kb.const_i32(3)); // (is & 3) * 8
    Value* is_lo = b.build_ult(is, kb.const_i32(4));

    Value* acc = kb.for_range_reduce(q.sb_local, q.bpr, q.stride, kb.const_f32(0.0f), [&](Value* sb, Value* acc) {
        Value* blk = block_ptr(kb, q, sb, kBlockBytes);
        // 16-byte header: {d | dmin << 16, scales[0..3], scales[4..7], scales[8..11]}
        Value* hdr = kb.vload(Type::i32x4(), blk);
        Value* h0 = b.build_vextract_lane(hdr, 0);
        Value* d = kb.f16_to_f32(b.build_and(h0, kb.const_i32(0xFFFF)));
        Value* dmin = kb.f16_to_f32(b.build_lshr(h0, kb.const_i32(16)));
        // get_scale_min_k4: j = is & 3; s0 = scales[j], s4 = scales[j + 4], s8 = scales[j + 8]
        auto byte_at = [&](uint32_t lane) {
            return b.build_and(b.build_lshr(b.build_vextract_lane(hdr, lane), sc_shift), kb.const_i32(0xFF));
        };
        Value* s0 = byte_at(1);
        Value* s4 = byte_at(2);
        Value* s8 = byte_at(3);
        Value* sc_lo = b.build_and(s0, kb.const_i32(0x3F));
        Value* m_lo = b.build_and(s4, kb.const_i32(0x3F));
        Value* sc_hi = b.build_or(b.build_and(s8, kb.const_i32(0x0F)),
                                  b.build_shl(b.build_and(b.build_lshr(s0, kb.const_i32(6)), kb.const_i32(3)), kb.const_i32(4)));
        Value* m_hi = b.build_or(b.build_and(b.build_lshr(s8, kb.const_i32(4)), kb.const_i32(0x0F)),
                                 b.build_shl(b.build_and(b.build_lshr(s4, kb.const_i32(6)), kb.const_i32(3)), kb.const_i32(4)));
        Value* sc = b.build_select(is_lo, sc_lo, sc_hi);
        Value* m = b.build_select(is_lo, m_lo, m_hi);
        Value* wscale = kb.mul(d, kb.u32_to_f32(sc));
        Value* neg_wmin = b.build_neg(kb.mul(dmin, kb.u32_to_f32(m)));

        Value* q4 = kb.load_i32(at(kb, blk, qs_off));   // 4 bytes of nibbles at blk + qoff
        Value* xv = x_float4(kb, x, kb.add(b.build_shl(sb, kb.const_i32(8)), xoff));
        Value* w4[4];
        for (int j = 0; j < 4; ++j) {
            // nibble j: byte j of q4, high half when hi4 == 4
            Value* nib = b.build_and(b.build_lshr(q4, kb.add(hi4, kb.const_i32(8 * j))), kb.const_i32(0x0F));
            w4[j] = b.build_fma_f32(wscale, kb.u32_to_f32(nib), neg_wmin);   // wscale * nib - wmin
        }
        return fma_lanes(kb, w4, xv, acc);
    });
    store_row_sum(kb, q, y, acc, scratch);
    return fn;
}

} // namespace brass::codegen
