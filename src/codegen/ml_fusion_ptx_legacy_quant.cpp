// LEGACY (Stage 5c): the two hand-written quantized GEMV PTX kernels that are
// now built as MIR in ml_fusion_ptx_kernels_quant.cpp. The text below is
// byte-for-byte what MlFusionCompiler::emit_ptx_fused_gemv_q8_0 and
// emit_ptx_fused_gemv_q4_k used to return (ml_fusion_quant_ptx.cpp); it exists
// only as the reference side of the differential tests in
// tests/unit/test_gpu_kernel_migration_quant.cpp and is deleted in Stage 6.
// Do not extend, do not call from library code.
//
// Note: both templates hard-code a 256-thread block (Q8_0 steps the block
// index by 32 = 256 / 8, Q4_K by 4 = 256 / 64), so they only compute the full
// dot product when launched with block = 256. The MIR kernels derive the
// stride from %ntid.x instead (identical at 256).

#include "ml_fusion_ptx_legacy.hpp"

#include <sstream>

namespace brass::codegen::legacy {

// =========================================================================
// 1. Fused Q8_0 GEMV PTX (Single-token decode M=1, N x K)
// =========================================================================

std::string legacy_ptx_gemv_q8_0(const target::PtxOptions& opts) {
    std::ostringstream ss;
    ss << legacy_ptx_header(opts);
    ss << R"PTX(
.visible .entry fused_gemv_q8_0_kernel(
    .param .u64 param_w,
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u32 param_n,
    .param .u32 param_k
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<64>;
    .shared .align 4 .f32 smem[32];

    ld.param.u64 %rd0, [param_w];
    ld.param.u64 %rd1, [param_x];
    ld.param.u64 %rd2, [param_y];
    ld.param.u32 %r0, [param_n];
    ld.param.u32 %r1, [param_k];

    // row = blockIdx.x
    mov.u32 %r2, %ctaid.x;
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 ret;

    // blocks_per_row = K >> 5 (K / 32)
    shr.u32 %r3, %r1, 5;

    // row_bytes = row * blocks_per_row * 34
    cvt.u64.u32 %rd4, %r2;
    cvt.u64.u32 %rd5, %r3;
    mul.lo.u64 %rd6, %rd4, %rd5;
    mul.lo.u64 %rd7, %rd6, 34;
    add.u64 %rd8, %rd0, %rd7; // row_base

    mov.f32 %f10, 0f00000000; // acc

    // tid = threadIdx.x (0..255)
    // 256 threads process 32 blocks (1024 elements) per step
    // sb_local = tid >> 3 (0..31)
    // lane = tid & 7 (0..7)
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 3; // sb_local
    and.b32 %r6, %r4, 7; // lane

    mov.u32 %r10, 0; // base_sb

$L_q8_loop:
    setp.ge.u32 %p1, %r10, %r3;
    @%p1 bra $L_q8_loop_end;

    add.u32 %r11, %r10, %r5; // sb = base_sb + sb_local
    setp.ge.u32 %p2, %r11, %r3;
    @%p2 bra $L_q8_skip;

    // blk_offset = sb * 34
    cvt.u64.u32 %rd10, %r11;
    mul.lo.u64 %rd11, %rd10, 34;
    add.u64 %rd12, %rd8, %rd11; // blk_ptr

    // Load fp16 scale d at blk_ptr + 0
    ld.global.u16 %r12, [%rd12];
    cvt.f32.f16 %f11, %r12; // f_d

    // Load 4 int8 weights at blk_ptr + 2 + lane * 4
    // Use two 16-bit loads for 2-byte alignment safety
    shl.b32 %r13, %r6, 2; // lane * 4
    add.u32 %r14, %r13, 2; // 2 + lane * 4
    cvt.u64.u32 %rd13, %r14;
    add.u64 %rd14, %rd12, %rd13; // qs_ptr

    ld.global.u16 %r15, [%rd14];
    ld.global.u16 %r16, [%rd14 + 2];
    shl.b32 %r17, %r16, 16;
    or.b32 %r18, %r15, %r17; // q4

    // Load 4 floats from x at (sb * 32 + lane * 4) * 4 bytes
    shl.b32 %r19, %r11, 5; // sb * 32
    add.u32 %r20, %r19, %r13; // k = sb * 32 + lane * 4
    cvt.u64.u32 %rd15, %r20;
    shl.b64 %rd16, %rd15, 2; // k * 4
    add.u64 %rd17, %rd1, %rd16; // x_ptr
    ld.global.v4.f32 {%f20, %f21, %f22, %f23}, [%rd17];

    // Unpack weight 0
    shl.b32 %r21, %r18, 24;
    shr.s32 %r22, %r21, 24;
    cvt.rn.f32.s32 %f24, %r22;
    mul.f32 %f24, %f24, %f11;
    fma.rn.f32 %f10, %f24, %f20, %f10;

    // Unpack weight 1
    shl.b32 %r23, %r18, 16;
    shr.s32 %r24, %r23, 24;
    cvt.rn.f32.s32 %f25, %r24;
    mul.f32 %f25, %f25, %f11;
    fma.rn.f32 %f10, %f25, %f21, %f10;

    // Unpack weight 2
    shl.b32 %r25, %r18, 8;
    shr.s32 %r26, %r25, 24;
    cvt.rn.f32.s32 %f26, %r26;
    mul.f32 %f26, %f26, %f11;
    fma.rn.f32 %f10, %f26, %f22, %f10;

    // Unpack weight 3
    shr.s32 %r28, %r18, 24;
    cvt.rn.f32.s32 %f27, %r28;
    mul.f32 %f27, %f27, %f11;
    fma.rn.f32 %f10, %f27, %f23, %f10;

$L_q8_skip:
    add.u32 %r10, %r10, 32; // base_sb += 32
    bra $L_q8_loop;

$L_q8_loop_end:
    // Warp-shuffle reduction
    shfl.sync.down.b32 %f30, %f10, 16, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f30;
    shfl.sync.down.b32 %f30, %f10, 8, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f30;
    shfl.sync.down.b32 %f30, %f10, 4, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f30;
    shfl.sync.down.b32 %f30, %f10, 2, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f30;
    shfl.sync.down.b32 %f30, %f10, 1, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f30;

    and.b32 %r30, %r4, 31; // lane_id
    shr.u32 %r31, %r4, 5;  // warp_id
    setp.eq.u32 %p3, %r30, 0;

    mov.u32 %r32, smem;
    shl.b32 %r33, %r31, 2;
    add.u32 %r34, %r32, %r33;
    @%p3 st.shared.f32 [%r34], %f10;

    bar.sync 0;

    setp.eq.u32 %p4, %r31, 0;
    @!%p4 ret;

    // Warp 0 reduces warp sums
    mov.u32 %r35, %ntid.x;
    shr.u32 %r36, %r35, 5; // num_warps
    setp.lt.u32 %p5, %r30, %r36;
    mov.f32 %f40, 0f00000000;

    shl.b32 %r37, %r30, 2;
    add.u32 %r38, %r32, %r37;
    @%p5 ld.shared.f32 %f40, [%r38];

    shfl.sync.down.b32 %f41, %f40, 16, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 8, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 4, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 2, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 1, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;

    setp.eq.u32 %p6, %r30, 0;
    @!%p6 ret;

    cvt.u64.u32 %rd30, %r2;
    shl.b64 %rd31, %rd30, 2;
    add.u64 %rd32, %rd2, %rd31;
    st.global.f32 [%rd32], %f40;

    ret;
}
)PTX";
    return ss.str();
}

// =========================================================================
// 2. Fused Q4_K GEMV PTX (Single-token decode M=1, N x K)
// =========================================================================

std::string legacy_ptx_gemv_q4_k(const target::PtxOptions& opts) {
    std::ostringstream ss;
    ss << legacy_ptx_header(opts);
    ss << R"PTX(
.visible .entry fused_gemv_q4_k_kernel(
    .param .u64 param_w,
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u32 param_n,
    .param .u32 param_k
)
{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<64>;
    .shared .align 4 .f32 smem[32];

    ld.param.u64 %rd0, [param_w];
    ld.param.u64 %rd1, [param_x];
    ld.param.u64 %rd2, [param_y];
    ld.param.u32 %r0, [param_n];
    ld.param.u32 %r1, [param_k];

    // row = blockIdx.x
    mov.u32 %r2, %ctaid.x;
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 ret;

    // blocks_per_row = K >> 8 (K / 256)
    shr.u32 %r3, %r1, 8;

    // row_bytes = row * blocks_per_row * 144
    cvt.u64.u32 %rd4, %r2;
    cvt.u64.u32 %rd5, %r3;
    mul.lo.u64 %rd6, %rd4, %rd5;
    mul.lo.u64 %rd7, %rd6, 144;
    add.u64 %rd8, %rd0, %rd7; // row_base

    mov.f32 %f10, 0f00000000; // acc

    // tid = threadIdx.x (0..255)
    // 256 threads process 4 super-blocks (1024 elements) per step
    // sb_local = tid >> 6 (0..3)
    // t = tid & 63 (0..63)
    // is = t >> 3 (0..7, sub-block)
    // lg = t & 7 (0..7, quad)
    mov.u32 %r4, %tid.x;
    shr.u32 %r5, %r4, 6; // sb_local
    and.b32 %r6, %r4, 63; // t
    shr.u32 %r7, %r6, 3; // is
    and.b32 %r8, %r6, 7; // lg

    // hi = is & 1
    and.b32 %r9, %r7, 1;
    setp.ne.u32 %p8, %r9, 0;

    // pair = is >> 1 (0..3)
    shr.u32 %r50, %r7, 1;
    // qoff = 16 + pair * 32 + lg * 4
    shl.b32 %r51, %r50, 5; // pair * 32
    shl.b32 %r52, %r8, 2;  // lg * 4
    add.u32 %r53, %r51, %r52;
    add.u32 %r54, %r53, 16; // qoff

    // xoff = is * 32 + lg * 4
    shl.b32 %r55, %r7, 5; // is * 32
    add.u32 %r56, %r55, %r52; // xoff

    mov.u32 %r10, 0; // base_sb

$L_q4_loop:
    setp.ge.u32 %p1, %r10, %r3;
    @%p1 bra $L_q4_loop_end;

    add.u32 %r11, %r10, %r5; // sb = base_sb + sb_local
    setp.ge.u32 %p2, %r11, %r3;
    @%p2 bra $L_q4_skip;

    // blk_offset = sb * 144
    cvt.u64.u32 %rd10, %r11;
    mul.lo.u64 %rd11, %rd10, 144;
    add.u64 %rd12, %rd8, %rd11; // blk_ptr

    // Load 16-byte header
    ld.global.v4.u32 {%r12, %r13, %r14, %r15}, [%rd12];

    // d and dmin from %r12
    and.b32 %r16, %r12, 0x0000FFFF;
    cvt.f32.f16 %f11, %r16; // d
    shr.u32 %r17, %r12, 16;
    cvt.f32.f16 %f12, %r17; // dmin

    // Unpack sc and m for this thread's sub-block (is)
    setp.lt.u32 %p3, %r7, 4;
    @!%p3 bra $L_is_ge_4;

    // is < 4:
    shl.b32 %r18, %r7, 3; // is * 8
    shr.u32 %r19, %r13, %r18;
    and.b32 %r20, %r19, 0x3F; // sc
    shr.u32 %r21, %r14, %r18;
    and.b32 %r22, %r21, 0x3F; // m
    bra $L_sc_m_done;

$L_is_ge_4:
    // is >= 4:
    sub.u32 %r23, %r7, 4; // j = is - 4
    shl.b32 %r18, %r23, 3; // j * 8
    shr.u32 %r24, %r15, %r18; // scales[j+8]
    and.b32 %r24, %r24, 0xFF;
    shr.u32 %r25, %r13, %r18; // scales[j]
    and.b32 %r25, %r25, 0xFF;
    shr.u32 %r26, %r14, %r18; // scales[j+4]
    and.b32 %r26, %r26, 0xFF;

    and.b32 %r27, %r24, 0x0F;
    shr.u32 %r28, %r25, 6;
    and.b32 %r28, %r28, 0x03;
    shl.b32 %r29, %r28, 4;
    or.b32 %r20, %r27, %r29; // sc

    shr.u32 %r30, %r24, 4;
    and.b32 %r31, %r30, 0x0F;
    shr.u32 %r32, %r26, 6;
    and.b32 %r32, %r32, 0x03;
    shl.b32 %r33, %r32, 4;
    or.b32 %r22, %r31, %r33; // m

$L_sc_m_done:
    cvt.rn.f32.u32 %f13, %r20;
    cvt.rn.f32.u32 %f14, %r22;
    mul.f32 %f15, %f11, %f13; // wscale = d * sc
    mul.f32 %f16, %f12, %f14; // wmin = dmin * m
    neg.f32 %f17, %f16;        // -wmin

    // Load 4 bytes from qs at blk_ptr + qoff
    cvt.u64.u32 %rd13, %r54;
    add.u64 %rd14, %rd12, %rd13;
    ld.global.u32 %r34, [%rd14]; // q4

    // Load 4 floats from x at (sb * 256 + xoff) * 4 bytes
    shl.b32 %r35, %r11, 8; // sb * 256
    add.u32 %r36, %r35, %r56; // k = sb * 256 + xoff
    cvt.u64.u32 %rd15, %r36;
    shl.b64 %rd16, %rd15, 2;
    add.u64 %rd17, %rd1, %rd16;
    ld.global.v4.f32 {%f20, %f21, %f22, %f23}, [%rd17];

    // Element 0:
    and.b32 %r37, %r34, 0xFF;
    @%p8 shr.u32 %r37, %r37, 4;
    and.b32 %r37, %r37, 0x0F;
    cvt.rn.f32.u32 %f24, %r37;
    fma.rn.f32 %f25, %f15, %f24, %f17; // w = wscale * nib - wmin
    fma.rn.f32 %f10, %f25, %f20, %f10;

    // Element 1:
    shr.u32 %r38, %r34, 8;
    and.b32 %r38, %r38, 0xFF;
    @%p8 shr.u32 %r38, %r38, 4;
    and.b32 %r38, %r38, 0x0F;
    cvt.rn.f32.u32 %f26, %r38;
    fma.rn.f32 %f27, %f15, %f26, %f17;
    fma.rn.f32 %f10, %f27, %f21, %f10;

    // Element 2:
    shr.u32 %r39, %r34, 16;
    and.b32 %r39, %r39, 0xFF;
    @%p8 shr.u32 %r39, %r39, 4;
    and.b32 %r39, %r39, 0x0F;
    cvt.rn.f32.u32 %f28, %r39;
    fma.rn.f32 %f29, %f15, %f28, %f17;
    fma.rn.f32 %f10, %f29, %f22, %f10;

    // Element 3:
    shr.u32 %r40, %r34, 24;
    @%p8 shr.u32 %r40, %r40, 4;
    and.b32 %r40, %r40, 0x0F;
    cvt.rn.f32.u32 %f30, %r40;
    fma.rn.f32 %f31, %f15, %f30, %f17;
    fma.rn.f32 %f10, %f31, %f23, %f10;

$L_q4_skip:
    add.u32 %r10, %r10, 4; // base_sb += 4
    bra $L_q4_loop;

$L_q4_loop_end:
    // Warp-shuffle reduction
    shfl.sync.down.b32 %f32, %f10, 16, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f32;
    shfl.sync.down.b32 %f32, %f10, 8, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f32;
    shfl.sync.down.b32 %f32, %f10, 4, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f32;
    shfl.sync.down.b32 %f32, %f10, 2, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f32;
    shfl.sync.down.b32 %f32, %f10, 1, 0x1f, 0xffffffff;
    add.f32 %f10, %f10, %f32;

    and.b32 %r41, %r4, 31; // lane_id
    shr.u32 %r42, %r4, 5;  // warp_id
    setp.eq.u32 %p4, %r41, 0;

    mov.u32 %r43, smem;
    shl.b32 %r44, %r42, 2;
    add.u32 %r45, %r43, %r44;
    @%p4 st.shared.f32 [%r45], %f10;

    bar.sync 0;

    setp.eq.u32 %p5, %r42, 0;
    @!%p5 ret;

    // Warp 0 reduces warp sums
    mov.u32 %r46, %ntid.x;
    shr.u32 %r47, %r46, 5; // num_warps
    setp.lt.u32 %p6, %r41, %r47;
    mov.f32 %f40, 0f00000000;

    shl.b32 %r48, %r41, 2;
    add.u32 %r49, %r43, %r48;
    @%p6 ld.shared.f32 %f40, [%r49];

    shfl.sync.down.b32 %f41, %f40, 16, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 8, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 4, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 2, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;
    shfl.sync.down.b32 %f41, %f40, 1, 0x1f, 0xffffffff;
    add.f32 %f40, %f40, %f41;

    setp.eq.u32 %p7, %r41, 0;
    @!%p7 ret;

    cvt.u64.u32 %rd30, %r2;
    shl.b64 %rd31, %rd30, 2;
    add.u64 %rd32, %rd2, %rd31;
    st.global.f32 [%rd32], %f40;

    ret;
}
)PTX";
    return ss.str();
}

} // namespace brass::codegen::legacy
