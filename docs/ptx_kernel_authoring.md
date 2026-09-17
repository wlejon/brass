# Writing PTX Kernels with KernelBuilder

How GPU kernels are written in brass: as MIR, through the `KernelBuilder`
GPU helpers, lowered by the PTX backend described in
[ptx_backend_design.md](ptx_backend_design.md). This guide is the reference
for the intrinsic table, the helpers, the shared memory model, the ten fused
ML kernels the library ships, and the checklist for adding and testing a
new kernel. Nothing here involves PTX text; a kernel author never writes a
mnemonic.

## A kernel is a MIR function

```cpp
Module mod("m");
Function* fn = mod.create_function("my_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
KernelBuilder kb(mod, fn);
Builder& b = kb.builder();
BasicBlock* entry = b.append_block("entry");
b.position_at_end(entry);
Value* in  = b.add_block_param(entry, Type::ptr());   // one block param per kernel param
Value* out = b.add_block_param(entry, Type::ptr());
Value* n   = b.add_block_param(entry, Type::i32());

Value* i = kb.global_tid_x();
kb.if_then(b.build_slt(i, n), [&] {
    kb.store_f32_indexed(out, i, kb.exp_fast(kb.load_f32_indexed(in, i)));
});
b.build_ret_void();

std::string ptx = target::PtxTarget::emit_function(*fn);   // ISel -> cleanup -> verify -> print
```

- The function is `void` (an `.entry` cannot return values); results go
  through pointer parameters. Parameter types are `ptr`, `i32`, `i64`,
  `f32`, `f64`; `ptr`/`i64` print as `.param .u64`, `i32` as `.param .u32`.
- The entry block declares one block parameter per kernel parameter, in
  order (`entry_params` in `src/codegen/ml_fusion_ptx_kernels_common.hpp`
  does this for the fused kernels).
- Plain MIR (`add`, `mul`, `fma`, `shl`, `lshr`, `ashr`, `and`, `or`,
  `select`, comparisons, `zext_i64`, `load`/`store`, `vload`/`vstore`,
  vector ops, `br`/`br_if` with block arguments) lowers directly; everything
  PTX-specific is an intrinsic call with a `KernelBuilder` wrapper.
- `ptx_*` names are GPU-only: `KernelJit` registers no CPU symbols for them,
  so a kernel using them is a PTX-only builder. The `fabsf/fminf/fmaxf/
  expf/logf/sqrtf/...` aliases work on both sides: real libm calls on the
  CPU, inline PTX on the GPU.
- Launch: `gpu::CudaModule::load(ptx)` then `launch_1d(entry, grid, block,
  args, shared_bytes = 0)` (`include/brass/gpu/cuda_driver.hpp`). Static
  shared usage is the sum of the declared arrays; nothing is passed at
  launch.

## Intrinsic table

MIR signature on the left (`build_call(name, result_type, {args})`), PTX on
the right. Arguments named `const` must be `iconst_i32`/`iconst_i64`
results; `lane`/`id` arguments may be constants (printed as immediates) or
i32/i64 registers (narrowed with `cvt.u32.u64` when 64-bit). Optional
`[, off]` byte offsets fold into `[reg + disp]` when constant, otherwise an
`add.s64` is emitted.

| Name(s) | Signature | PTX |
| --- | --- | --- |
| `ptx_tid_{x,y,z}` `ptx_ctaid_{x,y,z}` `ptx_ntid_{x,y,z}` `ptx_nctaid_{x,y,z}` | `() -> i32` | `mov.u32 %r, %tid.x` ... (read once per kernel, in the prologue) |
| `ptx_laneid`/`ptx_lane_id`, `ptx_warpid`/`ptx_warp_id`, `ptx_nwarpid`, `ptx_smid`, `ptx_nsmid`, `ptx_clock` | `() -> i32` | `mov.u32 %r, %laneid` ... (`%warpid` is the hardware warp slot, not `tid/32`) |
| `ptx_clock64`, `ptx_globaltimer` | `() -> i64` | `mov.u64 %rd, %clock64` / `%globaltimer` |
| `ptx_global_tid_x`/`ptx_global_id_x` | `() -> i32` | `mad.lo.s32 ctaid.x, ntid.x, tid.x` |
| `rsqrtf rsqrt ptx_rsqrt` / `sqrtf sqrt ptx_sqrt` / `sinf sin ptx_sin` / `cosf cos ptx_cos` / `ex2f ex2 ptx_ex2` / `lg2f ptx_lg2` / `ptx_rcp ptx_rcp_approx` | `(f32) -> f32` | `rsqrt/sqrt/sin/cos/ex2/lg2/rcp.approx.f32` |
| `expf exp ptx_exp` | `(f32) -> f32` | `mul.f32 x, log2e; ex2.approx.f32` |
| `logf log ptx_log` | `(f32) -> f32` | `lg2.approx.f32; mul.f32 ln2` |
| `ptx_sqrt_rn` | `(f32|f64) -> same` | `sqrt.rn.f32|f64` |
| `fabsf fabs ptx_fabs` | `(f32|f64) -> same` | `abs.f32|f64` |
| `fminf fmin ptx_fmin` / `fmaxf fmax ptx_fmax` | `(T, T) -> T`, T = f32|f64 | `min/max.f32|f64` |
| `ptx_div_approx` | `(f32, f32) -> f32` | `div.approx.f32` (`div.full` is not modelled; MIR `sdiv` on f32 is `div.rn.f32`) |
| `i32_to_f32 ptx_i32_to_f32` / `ptx_u32_to_f32` | `(i32) -> f32` | `cvt.rn.f32.s32` / `cvt.rn.f32.u32` |
| `ptx_i64_to_f32` / `ptx_u64_to_f32` | `(i64) -> f32` | `cvt.rn.f32.s64` / `.u64` |
| `ptx_f32_to_i32` / `ptx_f32_to_u32` | `(f32) -> i32` | `cvt.rzi.s32.f32` / `cvt.rzi.u32.f32` (saturating, truncating) |
| `ptx_f16_to_f32` | `(i32 bits) -> f32` | `cvt.f32.f16 %f, %r` (low 16 bits of the B32) |
| `ptx_f32_to_f16` | `(f32) -> i32 bits` | `cvt.rn.f16.f32 %r, %f` |
| `ptx_f32_to_f64` / `ptx_f64_to_f32` | `(f32) -> f64` / `(f64) -> f32` | `cvt.f64.f32` / `cvt.rn.f32.f64` |
| `bar.sync` `ptx_sync` | `() -> void` | `bar.sync 0` |
| `ptx_bar_sync` | `(i32 id) -> void` | `bar.sync id` |
| `ptx_bar_sync_count` | `(i32 id, i32 nthreads) -> void` | `bar.sync id, nthreads` |
| `ptx_shfl_{down,up,bfly,xor,idx}_f32` | `(f32, i32 delta) -> f32` | `shfl.sync.<mode>.b32 d, v, delta, clamp, 0xffffffff` (clamp 0 for `up`, 0x1f otherwise, as nvcc emits) |
| `ptx_shfl_{down,up,bfly,xor,idx}_i32` | `(i32, i32 delta) -> i32` | same |
| `ptx_shfl_down_sync_f32` `shfl_down_sync_f32` | `(i32 mask, f32, i32 delta) -> f32` | `shfl.sync.down.b32 ... 0x1f, mask` |
| `ptx_atom_add_f32` | `(ptr, f32) -> f32 old` | `atom.global.add.f32` |
| `ptx_atom_add_i32` `ptx_atom_add_u32` / `ptx_atom_add_i64` | `(ptr, i32) -> i32` / `(ptr, i64) -> i64` | `atom.global.add.u32` / `.u64` |
| `ptx_atom_min_i32` `ptx_atom_max_i32` / `ptx_atom_exch_i32` | `(ptr, i32) -> i32` | `atom.global.min/max.s32` / `atom.global.exch.b32` |
| `ptx_atom_shared_add_f32` / `ptx_atom_shared_add_i32` | `(shared ptr, T) -> T` | `atom.shared.add.f32` / `.u32` |
| `ptx_mul_wide_u32` / `ptx_mul_wide_s32` | `(i32, i32) -> i64` | `mul.wide.u32` / `.s32` |
| `ptx_mul_hi_u32` | `(i32, i32) -> i32` | `mul.hi.u32` |
| `ptx_mad_lo_u32` | `(i32, i32, i32) -> i32` | `mad.lo.u32` |
| `ptx_shared_alloc_{f32,i32,f64,i64}` | `(const i32 count) -> ptr` | `.shared .align 16 .<f32|u32|f64|u64> smem_<n>[count]` + `mov.u64 %rd, smem_<n>` |
| `ptx_shared_load_{f32,i32,f64,i64}` | `(ptr[, i32|i64 off]) -> T` | `ld.shared.<T> d, [ptr + off]` |
| `ptx_shared_load_{f32,i32,f64,i64}_indexed` | `(ptr, i32|i64 index) -> T` | `ld.shared.<T> d, [ptr + index*sizeof(T)]` |
| `ptx_shared_store_{f32,i32,f64,i64}` | `(ptr, T value[, off]) -> void` | `st.shared.<T> [ptr + off], value` |
| `ptx_shared_store_{f32,i32,f64,i64}_indexed` | `(ptr, index, T value) -> void` | `st.shared.<T> [ptr + index*sizeof(T)], value` |
| `ptx_load_{u8,s8,u16,s16}` | `(ptr[, off]) -> i32` | `ld.global.u8/s8/u16/s16 %r, [ptr + off]` (zero-/sign-extended into the B32) |
| `ptx_store_{u8,u16}` | `(ptr, i32 value[, off]) -> void` | `st.global.u8/u16 [ptr + off], %r` (low bits) |

Not in the table, by decision: `div.full.f32` (`div.approx` or `div.rn`
cover the kernels), an `i32x4`-specific load (MIR `vload` of `i32x4`
already prints `ld.global.v4.u32`), `mad.lo` pattern matching
(`ptx_mad_lo_u32` exists; ptxas fuses `mul`+`add` anyway).

## KernelBuilder helpers (`include/brass/codegen/kernel_jit.hpp`, `src/codegen/kernel_builder_ptx.cpp`)

Thin wrappers, each a few lines of MIR around the intrinsics above.

- constants: `const_i32(v) const_i64(v) const_f32(v)`
- indices (i32): `tid_x/y/z() ctaid_x/y/z() ntid_x/y/z() nctaid_x/y/z()
  lane_id() warp_id()` (= `tid_x >> 5`, the warp index within the block)
  `global_tid_x()`
- barriers: `sync()` (`bar.sync 0`), `bar_sync(id)`
- shuffles: `shfl_down_f32(v, delta|Value*) shfl_up_f32(v, delta)
  shfl_bfly_f32(v, mask) shfl_idx_f32(v, lane|Value*) shfl_down_i32
  shfl_bfly_i32 shfl_idx_i32`
- reductions: `warp_reduce_sum_f32(v)` / `warp_reduce_max_f32(v)` (the
  16/8/4/2/1 `shfl.down` butterfly; result valid in lane 0, the other
  lanes hold partial sums). `block_reduce_sum_f32(v, scratch)`: warp
  reduce -> lane 0 of each warp stores `scratch[warp]` (an `if_then`) ->
  `bar.sync` -> every warp reads `scratch[lane]` for `lane < (ntid+31)/32`
  (else 0), warp-reduces and broadcasts lane 0 with `shfl.idx`, so *every
  thread* returns the block total -> `bar.sync` so `scratch` can be reused
  at once. Requirements: `scratch` is a `shared_alloc_f32` of at least 32
  elements, block size a multiple of 32 up to 1024, all threads reach the
  call. The builder is left in a new block (the `if_then` join), so values
  computed afterwards must be emitted after the call, which is the natural
  order anyway.
- shared memory: `shared_alloc_f32(n) shared_alloc_i32(n)
  shared_load_f32(smem, byte_off = 0) shared_load_f32_indexed(smem, index)
  shared_store_f32(smem, val, byte_off = 0) shared_store_f32_indexed(smem,
  index, val)` and the `_i32` forms
- fast math: `rcp_approx div_approx ex2_approx lg2_approx exp_fast log_fast
  rsqrt_approx sqrt_approx fabs fmin fmax`
- conversions: `f16_to_f32 f32_to_f16 u32_to_f32 i32_to_f32 f32_to_u32
  f32_to_i32`
- narrow memory: `load_u8/s8/u16/s16(ptr, byte_off = 0) store_u8/u16(ptr,
  val, byte_off = 0)`
- atomics: `atom_add_f32(ptr, v) atom_add_i32(ptr, v)`
- control flow (all leave the builder positioned in the join/exit block
  they create; loops run while `i < end`, signed, with `i` of `start`'s
  type):
  - `if_then(cond, body)` -- creates `if_then`/`if_join` blocks; the body
    may end in its own terminator (e.g. an early `ret`).
  - `for_range(start, end, step, body(i), unroll = 1)` -- `head(i)`: `i <
    end` -> body -> `br head(i + step)`. With `unroll == 1` the body may end
    in its own terminator.
  - `for_range_reduce(start, end, step, init, body(i, acc) -> acc', unroll
    = 1)` -- one loop-carried value; the result is the head block's `acc`
    parameter, valid in the exit block because the head dominates it. The
    body returns the new accumulator and must not end in a terminator.
  - `for_range_reduce_n(start, end, step, inits, body(i, accs) -> accs',
    unroll = 1)` -- any number of carried values (one block argument each);
    the body returns the new values in the order of `inits`.
  - **Unrolling.** `unroll > 1` emits a main loop that runs `unroll` copies
    of the body per iteration (`body(i)`, `body(i + step)`, ...) while `i +
    step * (unroll - 1) < end`, stepping by `step * unroll`, followed by a
    remainder loop with the plain `step` that continues from where the main
    loop stopped, carrying its accumulators. The per-thread element order is
    unchanged, so a floating-point reduction gives the same result as the
    rolled loop. `step * (unroll - 1)` and `step * unroll` are computed once
    before the loop. An unrolled body must not end in its own terminator
    (the copies are concatenated in one block; the builder throws
    `invalid_argument`). Use it where the stride is a runtime value (a
    grid- or block-stride loop): ptxas unrolls loops with literal strides
    itself but keeps a runtime-stride loop rolled, with one iteration's
    loads in flight per thread.

## Shared memory model

A `.shared` array is a per-kernel declaration created by
`ptx_shared_alloc_<T>(count)` (`shared_alloc_f32(n)` etc.): `PtxISel`
appends a `SharedDecl` (`.align 16`, name `smem_<index>`, unique within the
function) and materializes the array's shared-window address with `mov.u64
%rd, smem_<n>` into the `ptr` result. That 64-bit value is only meaningful
to the `ptx_shared_*` intrinsics, which emit `ld.shared`/`st.shared` with a
B64 base register. Plain MIR `load`/`store`/`vload`/`vstore` are always
`.global`; there is no address-space provenance tracking, so passing a
shared pointer to `load` is a silent bug -- always use the shared
intrinsics. Integer arithmetic on the pointer (`add ptr, i64`) is fine as
long as the result is again consumed only by shared intrinsics. Element
counts must be compile-time constants (a non-constant count throws
`PtxISel: ptx_shared_alloc_* requires a positive compile-time constant
element count`). Two kernels in one module may both declare `smem_0`; PTX
scopes the names per `.entry`.

## Narrow loads and f16

MIR has no i8/i16 types, so 8/16-bit accesses are intrinsics that extend
into an i32; `ptx_f16_to_f32` takes such an i32. The Q8_0 header is
`load_u16(blk)` -> `f16_to_f32`; the Q4_K header is a `vload(i32x4)` whose
lane 0 is split with `and 0xFFFF` / `lshr 16` before `f16_to_f32`. Q8_0
blocks are only 2-byte aligned, so their four int8 weights are read as two
`load_u16` and packed into one word before sign extension (`shl 24-8j; ashr
24`).

## The fused ML kernels

`MlFusionCompiler::build_ptx_*` (declared in
`include/brass/codegen/ml_fusion.hpp`, thin `emit_ptx_*` wrappers in
`src/codegen/ml_fusion_ptx.cpp`):

| File | Kernels | Shared helpers |
| --- | --- | --- |
| `src/codegen/ml_fusion_ptx_kernels_common.hpp` | -- | `f32_offset`, `at`, `silu_fast`, `RowBlock`/`row_block_prologue` (early `ret` for `row >= rows`, row byte offset, `d & ~3` with the `d % 4 != 0` -> scalar-path guard, float4/scalar loop bounds), `entry_params` |
| `ml_fusion_ptx_kernels.cpp` | `fused_swiglu_kernel`, `fused_swiglu_packed_kernel`, `fused_adaln_modulate[_gated]_kernel`, `fused_residual_rms_norm_kernel` | |
| `ml_fusion_ptx_kernels_norm.cpp` | `fused_layernorm_modulate_kernel`, `fused_residual_layernorm_kernel` | `row_sum`, `row_sum_sq_dev`, `block_mean`, `block_rstd` |
| `ml_fusion_ptx_kernels_gemv.cpp` | `fused_gemv_swiglu_kernel`, `fused_gemv_residual_kernel` | `gemv_prologue`, `gemv_partial_dots` (float4 loop and scalar tail for N weight rows at once), `silu_fast_clamped` |
| `ml_fusion_ptx_kernels_quant.cpp` | `fused_gemv_q8_0_kernel`, `fused_gemv_q4_k_kernel` | `quant_prologue`, `block_ptr`, `x_float4`, `fma_lanes`, `store_row_sum` |

**Launch contracts** (`u32` parameters; block size a multiple of 32 up to
1024 unless stated):

| Kernel | Parameters | Launch | Computes |
| --- | --- | --- | --- |
| `fused_swiglu_kernel` | `gate, up, out, n` | grid-stride | `out = silu(gate) * up` over n/4 float4s, then a scalar tail `[n & ~3, n)` walked by every block with stride `ntid` |
| `fused_swiglu_packed_kernel` | `x, y, b, d` | grid-stride over the `b * d` outputs | the same over one packed gate\|up projection: `x` is `[b, 2d]` with gate in columns `[0, d)` and up in `[d, 2d)` of each row, `y` is `[b, d]`; float4s only when `d % 4 == 0`, else all scalar. The layout `brotensor::swiglu_forward` is defined on |
| `fused_adaln_modulate_kernel` | `x, scale, shift, y, l, d` | one block per row (`l` rows of `d`) | `y = x * (1 + scale) + shift` |
| `fused_adaln_modulate_gated_kernel` | `x, scale, shift, gate, y, l, d` | one block per row | `... * gate` |
| `fused_residual_rms_norm_kernel` | `x, res, gamma, y, b, d, f32 eps` | one block per row | `x += res` in place; `y = x * gamma * rsqrt(mean(x^2) + eps)` |
| `fused_layernorm_modulate_kernel` | `x, gamma, beta, scale, shift, y, r, d, eps` | one block per row | `y = ((x - mean) * rstd * gamma + beta) * (1 + scale) + shift` |
| `fused_residual_layernorm_kernel` | `x, res, gamma, beta, y, b, d, eps` | one block per row | `x += res` in place; `y = (x - mean) * rstd * gamma + beta` |
| `fused_gemv_swiglu_kernel` | `w_gate, w_up, x, y, n, k` | one block per output row | `y[row] = silu(w_gate[row] . x) * (w_up[row] . x)` |
| `fused_gemv_residual_kernel` | `w_down, x, res, y, n, k` | one block per output row | `y[row] = w_down[row] . x + res[row]` |
| `fused_gemv_q8_0_kernel` | `w, x, y, n, k` | one block per row; `k % 32 == 0`; block `% 32 == 0` | `y[row] = dequant(w[row]) . x` over Q8_0 blocks |
| `fused_gemv_q4_k_kernel` | `w, x, y, n, k` | one block per row; `k % 256 == 0`; block `% 64 == 0` | same over Q4_K super-blocks |

Rows whose length is not a multiple of 4 take the scalar path for the whole
row (float4 loads need 16-byte alignment). One block per row means rows
beyond the grid are untouched; thread 0 of each GEMV block writes `y[row]`.

**Numerical recipes** (the tolerances below assume them):

- SiLU: `g * rcp.approx(1 + ex2.approx(-g * log2e))` (`silu_fast`); the
  GEMV SwiGLU clamps the exponent to `[-88, 88]` first
  (`silu_fast_clamped`), which keeps `ex2` finite for `|g| > 61`.
- RMSNorm: block sum of squares -> `div.approx(sum, cvt.rn.f32.u32 d)` ->
  `add eps` -> `rsqrt.approx`.
- LayerNorm: two-pass -- mean from plain `add`s in lane order, then `sub` +
  `fma` of the squared deviations about that mean; both statistics are
  `div.approx` by `cvt.rn.f32.u32 d`, rstd is `rsqrt.approx(var + eps)`.
  Every thread computes the same statistics from the broadcast block sum,
  so no shared scalar is needed; the two `block_reduce_sum_f32` calls share
  one 32-float scratch.
- GEMV: float4 loads with a scalar `fma.rn` chain per lane (lane order
  0..3, both SwiGLU dot products carried through one loop) over `[tid*4, k
  & ~3)` by `ntid*4`, then a scalar tail over `[k & ~3, k)` by `ntid`.
- Quantized GEMV: `blocks_per_row = k >> 5` (Q8_0) / `k >> 8` (Q4_K); 8
  (64) threads cooperate on each block; each thread walks the row's blocks
  from `sb_local = tid / tpb` by `stride = ntid / tpb`, loads the header,
  its 4 weights and the matching float4 of `x`, and folds the four products
  with `fma` in lane order into one f32 accumulator; then
  `block_reduce_sum_f32`. The block loop is unrolled
  `MlFusionCompiler::kQ8Unroll` / `kQ4KUnroll` (= 4) times.
  - Q8_0 block, 34 bytes: `f16 d`, then 32 `int8`; `w = cvt.rn.f32.s32(q) *
    d`.
  - Q4_K super-block, 144 bytes: a 16-byte header loaded as one
    `vload(i32x4)` -- lane 0 is `d | dmin << 16`, lanes 1..3 are
    `scales[0..3]`, `scales[4..7]`, `scales[8..11]` -- then 128 bytes of
    nibbles. Sub-block `is = (tid & 63) >> 3`, quad `lg = tid & 7`; the
    6-bit `sc`/`m` follow `get_scale_min_k4`: with `j = is & 3` and
    `s0/s4/s8 = scales[j]/[j+4]/[j+8]`, `is < 4` gives `sc = s0 & 0x3F, m =
    s4 & 0x3F`, else `sc = (s8 & 0xF) | ((s0 >> 6) << 4), m = (s8 >> 4) |
    ((s4 >> 6) << 4)`; both are computed and chosen with `select`. The 4
    nibbles come from one `ld.global.u32` at `blk + 16 + (is >> 1) * 32 + lg
    * 4`; sub-block `2p` is the low nibble of `qs[32p..32p+31]`, `2p+1` the
    high nibble, so lane `j` is `(q4 >> (hi4 + 8j)) & 0xF` with `hi4 = (is &
    1) * 4`. `w = fma(d * sc, nib, -(dmin * m))`.
  - The CPU dequantizers (`brass_dequant_q8_0_block` /
    `brass_dequant_q4k_block` in `ml_fusion_quant_cpu.cpp`) are the ground
    truth for both layouts.

## Writing and testing a new kernel

1. Write the builder next to the existing ones (a new `ml_fusion_ptx_kernels_*.cpp`
   when the file would pass 1,000 lines), reusing the prologue helpers.
   Declare it in `ml_fusion.hpp` with its launch contract in the comment
   block, and add a thin `emit_ptx_*` in `ml_fusion_ptx.cpp`.
2. If a PTX feature is missing, add an intrinsic (rule + table row in
   `ptx_isel_intrinsics*.cpp`, signature row in `test_ptx_intrinsics.cpp`,
   line in the table above) and a `KernelBuilder` wrapper; never a string.
3. Test in `tests/unit/test_gpu_kernels*.cpp` with the `ptx_test_support.hpp`
   helpers:
   - `lower_ok(*fn)` (ISel + verifier) and `emit_checked(*fn)` (the facade +
     ptxas when on PATH); check the recipe on the text (`has(ptx,
     "rsqrt.approx.f32")`, no `call `, no `st.global.v4` for a per-block
     scalar output, ...), and that the public `emit_ptx_*` returns the same
     text.
   - On device (`gpu_ready()`; `upload`, `launch`, `download`), compare with
     a double-precision host reference over a shape grid that hits every
     path: row length a multiple of 4 and not, element counts that are not
     a multiple of the block, rows above the grid (they must stay 0), block
     sizes 32 and 1024, and for a strided loop, trip counts that leave a
     remainder. Print the observed maximum relative difference per shape.
   - Tolerances (relative, `|a - b| <= tol * (1 + |b|)`): exact float
     arithmetic 1e-5 (in-place `x += res` 1e-6); `ex2.approx`/`rcp.approx`
     SiLU and `div.approx`/`rsqrt.approx` statistics 2e-3; a float GEMV dot
     product 1e-4, SiLU of a GEMV dot product 5e-3; the quantized GEMVs 1e-4
     against the double-accumulation host reference and the CPU JIT
     (observed: everything within 2.6e-6, most shapes within 2e-7).
4. Run `--filter=PTX`, `--filter=cleanup`, `--filter="GPU kernels"`,
   `--filter=GPU`, `--filter="ML Fusion"`. Every GPU/ptxas test prints a
   visible `[SKIP]` and passes when the tool or device is absent.
5. To look at what ptxas makes of it: write the PTX to a file (any
   `emit_ptx_*` call), `ptxas -arch=sm_89 -v k.ptx -o k.cubin` (register
   and shared-memory usage), `cuobjdump --dump-sass k.cubin` (count `FFMA`
   and `LDG` per loop body to see whether ptxas unrolled). The `unroll`
   parameter of the loop helpers is the lever when a runtime-stride loop
   stays rolled; the numbers for Q8_0/Q4_K are in the design document's
   History section.
