# PTX Backend Design

Status: in progress. This document is the contract for the PTX backend
restructuring. Each stage below is implemented as a separate commit.

## Problem

The PTX target historically had two parallel implementations:

1. `PtxTarget` (`src/target/ptx_target.cpp`): a single-pass walk over MIR that
   wrote PTX text directly into an `ostringstream`. Type suffixes, predicate
   materialization, block-argument copies, and temporary registers were all
   decided inline while producing strings. There was no data model in which
   "`ret` with an operand inside `.entry`" or "`shl.b64` with a 64-bit shift
   count" is a type error.
2. Hand-written `R"PTX(...)"` templates in `src/codegen/ml_fusion_*_ptx.cpp`
   (~2,100 lines of hand register-allocated assembly) for the fused ML kernels,
   duplicating MIR builders for several of the same kernels in
   `src/codegen/ml_fusion.cpp`.

The x64 backend is shaped as MIR -> `X64ISel` -> LIR (typed instruction data
structure) -> passes -> `X64Encoder` -> bytes. PTX skipped the middle layers.

## Target shape

```
MIR --PtxISel--> ptx::Function (typed PTX IR) --passes--> PtxPrinter --> text
                                     |
                                     +--> PtxVerifier (structural checks, no ptxas)
```

The byte encoder of the x64 pipeline is replaced by a text printer; everything
else is the same shape. PTX has virtual registers, no frames and no encoding,
so there is no register allocator and no `CodeBuffer`.

### ptx IR (`include/brass/target/ptx/ptx_ir.hpp`)

- `ptx::RegClass { Pred, B32, B64, F32, F64 }` with one counter per class per
  function. Register counts are known by construction; `.reg` declarations are
  derived from the function, never guessed.
- `ptx::Type` enum: the suffix set (`.pred .b8 .b16 .b32 .b64 .u8 .u16 .u32
  .u64 .s8 .s16 .s32 .s64 .f16 .f32 .f64`). One table maps MIR `Type` ->
  `ptx::Type`; nothing else decides suffixes.
- `ptx::Opcode` enum for every mnemonic the backend emits (`ld st mov cvt add
  sub mul mad fma div rem neg abs min max and or xor not shl shr setp selp
  bra ret call shfl bar rsqrt sqrt sin cos ex2 lg2 rcp ...`).
- Modifiers as typed fields, not strings: rounding (`.rn .rz .rm .rp`),
  `.approx`, `.ftz`, state space (`.global .shared .param .local`), vector
  width (`.v2 .v4`), `.lo/.hi/.wide`, comparison operator, `.sync` mask, etc.
- `ptx::Operand` tagged union: register (class + index), immediate (int or
  float; the printer owns exact-hex float formatting), address
  (`[reg + disp]`, `[symbol + disp]`), vector tuple `{r0, r1, r2, r3}`, label,
  kernel param, special register (`%tid.x`, `%ctaid.x`, `%ntid.x`,
  `%nctaid.x`, `%laneid`, `%warpid`, ...).
- `ptx::Inst { opcode, type(s), modifiers, guard predicate (optional, negatable),
  dst operands, src operands, MIR origin }`.
- `ptx::Block { label, insts }` and `ptx::Function { name, params, shared
  declarations, reg counts, blocks, is_entry }`.
- Kernel params carry a `ptx::Type` and a name; the entry-point signature is
  derived from them.

#### Stage 2 implementation notes (deviations from the sketch above)

- Register/type compatibility is stricter than ptxas except for bit types:
  `.f32`/`.f64` operands must be F32/F64 registers and `.u*/.s*` operands
  must be B32/B64, but `.b32` accepts B32 *or* F32 and `.b64` accepts B64
  *or* F64. The relaxation is required by `shfl.sync.down.b32` on f32
  accumulators and by `mov.b64` bitcasts between %rd and %fd, both of which
  the existing kernels use.
- Sub-32-bit types (`.b8 .b16 .u8 .u16 .s8 .s16`) and `.f16` map to B32
  registers (`reg_class_for`), matching `ld.global.u16 %r` and
  `cvt.f32.f16 %f, %r` in the hand-written kernels.
- One extra operand kind, `Symbol`: the address-of-shared-array source in
  `mov.u32 %r, smem` and the callee of `call`. `Param` operands print as
  `[name]` and are only legal as `ld.param` sources.
- Extra opcodes beyond the list: `atom` (with an `AtomOp` modifier, printed
  as `atom.global.add.f32`), `trap`, `exit`.
- The suffix table is three functions, all in `ptx_ir`: `type_for` (data
  suffix: u32/u64/f32/f64, used for ld/st/param/cvt), `signed_type_for`
  (s32/s64 for signed arithmetic and comparisons) and `bit_type_for`
  (b32/b64 for mov/selp/bitwise/shifts). `wide_type_for` gives the
  `mul.wide`/`mad.wide` result type.
- Integer immediates print in decimal (`4294967295` for a full lane mask);
  ptxas accepts this. Float immediates print as `0f`/`0d` hex and carry
  their own width (`imm_f32`/`imm_f64`), which the verifier checks against
  the instruction type.
- The verifier additionally enforces the modifier requirements ptxas has for
  PTX 7.x: integer `mul`/`mad` need `.lo/.hi/.wide`, `fma` needs a rounding
  mode, float `div` needs `.approx` or a rounding mode, `shfl`/`bar` need
  `.sync`, `rsqrt/sin/cos/ex2/lg2` need `.approx`, int<->float `cvt` needs a
  rounding mode, `and/or/xor/shl` need bit types. Diagnostics carry the
  function, block label, instruction index and printed instruction text.

### PtxPrinter (`ptx_printer.hpp`)

A single dumb walk. It makes no decisions beyond formatting. It prints the
header (`.version`, `.target`, `.address_size`), the entry signature, `.reg`
declarations (from counts), `.shared` declarations, then blocks in order.

### PtxVerifier (`ptx_verifier.hpp`)

Runs on any platform without ptxas. Checks at minimum:
- register index < declared count for its class,
- operand register class matches the instruction's type width
  (e.g. `.f32` operands are F32 registers, `.b64/.u64/.s64` are B64),
- guard is a Pred register,
- `setp` destination is Pred; `selp` condition is Pred,
- vector operand arity matches `.vN`,
- shift amounts are B32,
- `ret` inside `.entry` has no operand,
- every `bra` target label exists,
- `ld.param` sources name a declared param.

### PtxISel (`ptx_isel.hpp`)

Lowers MIR -> `ptx::Function`. Rules:
- Type suffix selection is table-driven (MIR `Type` -> `ptx::Type`).
- Comparisons produce Pred registers; `select`/`br_if` on a non-comparison
  value materialize a predicate with an explicit `setp.ne`.
- 64-bit shift counts are converted to B32 with `cvt.u32.u64`.
- Unsigned MIR ops (`udiv`, `urem`, `ult`, ...) emit unsigned PTX types.
- Block arguments are lowered with a parallel-copy resolver (handles swaps and
  cycles via a scratch register), predicated on the branch condition for the
  taken edge of `br_if`.
- Intrinsics are a table: MIR builtin call name -> lowering rule. This
  replaces string matching on callee names inside the instruction switch.
- Non-void kernels are rejected with a diagnostic (`.entry` cannot return).

#### Stage 3 implementation notes

File layout (mirrors `x64_isel_*.cpp`; every file stays under 1,000 lines):

| File | Concern |
| --- | --- |
| `include/brass/target/ptx/ptx_isel.hpp` | `class PtxISel { ptx::Function lower(const Function&); }`, the intrinsic-table query API, private state |
| `src/target/ptx/ptx_isel.cpp` | driver, use analysis, register assignment, `ld.param` prologue, opcode dispatch switch, value/register helpers (`reg_of`, `materialize_pred`, `shift_amount`, `emit`) |
| `src/target/ptx/ptx_isel_alu.cpp` | constants, arithmetic, bitwise, shifts, comparisons, `select`, conversions |
| `src/target/ptx/ptx_isel_mem.cpp` | `load/store`, `vload/vstore`, `load_indexed/store_indexed` (address materialization) |
| `src/target/ptx/ptx_isel_control.cpp` | `br/br_if/ret/unreachable`, edge copies, the parallel-copy resolver |
| `src/target/ptx/ptx_isel_intrinsics.cpp` | `PtxISel::Intrinsics` (the lowering rules), the name table, plain `call` |
| `src/target/ptx_target.cpp` | thin facade: ISel -> `ptx::verify` (throws with `format_diagnostics` output) -> `ptx::print` |

Behaviour worth knowing:

- **Register mapping lives in one place.** `allocate_registers` assigns every
  MIR value (entry params, block params, instruction results) a register run
  up front, in block order: one register for scalars, a contiguous run of
  `vector_lanes()` registers of the element class for vector types (f32x4,
  f32x8, f64x2, f64x4, i32x4, ...). Comparison results get a Pred register in
  addition to their B32 result. Temporaries (narrowed shift counts, indexed
  addresses, materialized predicates, intrinsic scratch, parallel-copy
  scratch) are allocated lazily during lowering and therefore number after
  the value registers. Numbering starts at 0; there is no reserved register.
- **Comparisons** emit `setp.<cmp>.<type>` into the Pred and only emit the
  `selp.u32 r, 1, 0, p` integer materialization when the result has a use
  other than the condition operand of `br_if`/`select` (`analyze_uses`).
  Float `ne` is `setp.neu` (unordered), matching the x64 lowering; other float
  comparisons are ordered. Unsigned MIR comparisons use `.u32/.u64`, signed
  ones and `eq/ne` use `.s32/.s64`.
- **Predicate materialization** for a non-comparison `br_if`/`select`
  condition is `setp.ne.u32/u64 p, r, 0` or `setp.neu.f32/f64 p, r, 0f0`,
  emitted at the use (never cached across blocks).
- **Kernel parameters** are declared from `Function::param_types()` as
  `param_<i>` and loaded in a `$L_params` fall-through block placed before
  the MIR entry block, so an entry block with incoming edges is not
  re-entered through the loads. Load types come from the entry block
  parameter types.
- **Block arguments** become `Copy{dst, src, bit_type}` lists per edge (one
  per lane for vector values; identity copies dropped).
  `emit_parallel_copies` emits any copy whose destination no other pending
  copy still reads; when only cycles remain it moves one source into a
  scratch register of the same class and redirects its readers, which
  unblocks the cycle. For `br_if` the taken edge's copies are guarded on the
  predicate and precede the guarded `bra`; the fall-through edge's copies are
  unguarded and follow it.
- **Shift counts** that are 64-bit are narrowed with `cvt.u32.u64` into a
  fresh B32 (`shift_amount`); `shl` uses `.b32/.b64`, `lshr` `.u32/.u64`,
  `ashr` `.s32/.s64`.
- **Indexed addressing** sign-extends a 32-bit index (`cvt.s64.s32`), scales
  by `shl.b64` for 2/4/8 (or `mul.lo.s64` by an immediate otherwise), adds to
  the base and addresses `[addr + offset]`.
- **Errors:** unsupported MIR opcodes throw `runtime_error("PtxISel:
  unsupported opcode in PTX lowering: <name>")`; malformed instructions
  throw naming the opcode and the missing piece; non-void kernels throw the
  Stage 1 "cannot return values" diagnostic; a verifier failure on the
  lowered IR is reported by `PtxTarget` as a compiler bug with every
  diagnostic attached.
- Every emitted `Inst` carries `.origin(&mir_inst)` (set by `PtxISel::emit`).

Golden comparison against the string emitter (all MIR kernels in
`ml_fusion.cpp` plus the hand-built kernels in `test_gpu_execution.cpp`):
after normalizing register numbers the only differences are (1) the
`$L_params:` label, (2) the dropped always-declared `.reg .f64 %fd<1>`, and
(3) elided dead `selp.u32` materializations for loop-exit comparisons. No
kernel got longer.

**Intrinsic table** (`src/target/ptx/ptx_isel_intrinsics.cpp`):
`PtxISel::Intrinsics::table()` maps callee name -> `IntrinsicLowering`
(`void(*)(PtxISel&, const Instruction&)`). To add one: write a static rule
in `PtxISel::Intrinsics` (or instantiate the `special<SpecialReg>` /
`approx_f32<Opcode>` templates) that reads its arguments with
`isel.reg_of(inst.operand(i), "...")`, allocates scratch with
`isel.fn_->new_*()`, writes `isel.result_reg(inst)` when the call has a
result, and calls `isel.emit(...)`; then add one row per name/alias.
`PtxISel::intrinsic_names()` and `is_intrinsic()` let tests iterate the
table (`test_ptx_isel.cpp` assembles every entry with ptxas). Every alias the
string emitter accepted is preserved.

Extension points for Stage 4:

- **Shared memory:** `ptx::Function::add_shared` already exists; the ISel
  needs a rule that materializes `mov.u32 %r, <name>` for the array base and
  a state-space choice in `ptx_isel_mem.cpp` (currently every `ld/st` is
  `.global`). A per-value "address space" map next to `regs_` is the natural
  place to carry that decision.
- **`bar.sync <id>` / named barriers:** replace the `Operand::imm(0)` in
  `Intrinsics::bar_sync` with the first call operand.
- **`rcp.approx` / `div.approx` / `ex2.approx.ftz`:** `Opcode::rcp` exists;
  `div.approx` is `Inst::make(Opcode::div, f32).approx()`; both are one-line
  table rules.
- **f16 -> f32:** `cvt.f32.f16` with the f16 value in a B32 register
  (`reg_class_for(Type::f16)` is B32); load it with `ld.global.u16`.
- **v4 u32 loads:** `vec_width_for` in `ptx_isel_mem.cpp` currently accepts
  f32x4/f64x2 only; i32x4 values already get a 4-register B32 run, so adding
  the type there is all that is needed.
- **`mad.lo`:** either a builtin rule or pattern-matching `add(mul(a,b),c)`
  in `lower_binary`; the IR verifier already enforces `.lo/.hi/.wide`.

### Kernel intrinsics

Fused ML kernels are written as MIR via `KernelBuilder` helpers and lowered
through `PtxISel`. Where a kernel needs a PTX-only concept, it is an MIR
builtin with a lowering rule, not a string. Required set (from the existing
hand-written kernels): thread/block/grid ids, `shfl.sync.down`, `bar.sync`,
per-function `.shared` arrays with `ld.shared`/`st.shared`, `ex2.approx`,
`rcp.approx`, `div.approx`, `rsqrt.approx`, `f16 -> f32` conversion, `v4`
loads/stores of `f32`/`u32`, `fma.rn`, `mad.lo`.

#### Stage 4 implementation notes

Stage 5 works from this section plus `include/brass/codegen/kernel_jit.hpp`
and `src/target/ptx/ptx_isel_intrinsics.cpp`. The inventory of the four
hand-written kernel files (every mnemonic, modifier, special register and
state space they use) is covered below; nothing in them needs a string.

**Files added/changed**

| File | Concern |
| --- | --- |
| `src/target/ptx/ptx_isel_vec.cpp` | vector MIR ops -> per-lane scalar PTX |
| `src/target/ptx/ptx_isel_mem.cpp` | `vload/vstore` for every vector type (v4/v2 tuples, two tuples for 256-bit) |
| `src/target/ptx/ptx_isel_intrinsics.hpp` | private: `struct PtxISel::Intrinsics` (all rule declarations) |
| `src/target/ptx/ptx_isel_intrinsics.cpp` | the name table; special registers, math, conversions |
| `src/target/ptx/ptx_isel_intrinsics_warp.cpp` | `bar.sync`, `shfl`, `atom`, `mul.wide/hi`, `mad.lo` |
| `src/target/ptx/ptx_isel_intrinsics_mem.cpp` | shared memory, narrow loads/stores, operand helpers (`const_int`, `address_operand`, `indexed_operand`, `lane_operand`) |
| `src/codegen/kernel_builder_ptx.cpp` | `KernelBuilder` GPU helpers |
| `tests/unit/ptx_test_support.hpp` | ptxas/device helpers, `run_map` harness, host f16 conversion |
| `tests/unit/test_ptx_intrinsics.cpp` | signature table (coverage-checked against `intrinsic_names()`), on-device tests per family |
| `tests/unit/test_ptx_vector.cpp` | vector opcodes, reductions, shared memory, control-flow helpers |

**Vector lowering (`ptx_isel_vec.cpp`).** Every vector value already owns a
contiguous register run of `vector_lanes()` registers of the element class
(f32x4 -> 4 x `%f`, f64x2 -> 2 x `%fd`, i32x4 -> 4 x `%r`, i64x2 -> 2 x `%rd`,
and the 8/4-lane 256-bit types likewise). Vector ops emit one scalar
instruction per lane with the suffix the scalar op would use:

| MIR | per lane (float / integer element) |
| --- | --- |
| `vadd vsub vmin vmax` | `add/sub/min/max.f32|f64` / `.s32|s64` |
| `vmul` | `mul.f32|f64` / `mul.lo.s32|s64` |
| `vdiv` | `div.rn.f32|f64` / `div.s32|s64` |
| `vfma` | `fma.rn.f32|f64` / `mad.lo.s32|s64` |
| `vneg` | `neg.f32|f64` / `neg.s32|s64` |
| `vsqrt` | `sqrt.rn.f32|f64` (float only) |
| `vand vor vxor vnot` | `and/or/xor/not.b32|b64` |
| `vbroadcast` | `mov.b32|b64` of the scalar into each lane |
| `vextract_lane` | `mov` from lane `inst.lane()` |
| `vinsert_lane` | `mov` of every lane, the chosen one from the scalar |
| `vshuffle` | `mov`s using the x64 `shufps`/`shufpd` mask encoding (per 128-bit half: 4-lane result takes lanes 0..1 from `a`, 2..3 from `b`, 2-bit selectors; 2-lane result takes lane 0 from `a`, lane 1 from `b`, 1-bit selectors) |
| `vzero` | `mov.f32 %f, 0f00000000` / `mov.b32 %r, 0` per lane |
| `vload/vstore` | `ld/st.global.v4.<f32|u32>` or `.v2.<f64|u64>`; 256-bit types are two tuples at `+0` and `+16` |

All eight vector types (`f32x4 f64x2 i32x4 i64x2 f32x8 f64x4 i32x8 i64x4`)
are accepted everywhere (arithmetic, loads/stores, block arguments). A
`vload(i32x4)` is the `ld.global.v4.u32 {..}` header load of the Q4_K kernel;
lanes come out with `vextract_lane`. With this, `build_gemv_q8_0` /
`build_gemv_q4_k` in `ml_fusion_quant_cpu.cpp` lower and verify (they still
`call` the CPU dequantizers, so they do not assemble until Stage 5 replaces
those calls).

**Intrinsics (`PtxISel::Intrinsics::table()`).** MIR signature on the left
(`build_call(name, result_type, {args})`), PTX on the right. Arguments named
`const` must be `iconst_i32`/`iconst_i64` results (`PtxISel::const_int`);
`lane`/`id` arguments may be constants (printed as immediates) or i32/i64
registers (narrowed with `cvt.u32.u64` when 64-bit). Optional `[, off]` byte
offsets fold into `[reg + disp]` when constant, otherwise an `add.s64` is
emitted.

| Name(s) | Signature | PTX |
| --- | --- | --- |
| `ptx_tid_{x,y,z}` `ptx_ctaid_{x,y,z}` `ptx_ntid_{x,y,z}` `ptx_nctaid_{x,y,z}` | `() -> i32` | `mov.u32 %r, %tid.x` ... |
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

Things that were considered and not added: `div.full.f32` (no `.full`
modifier in the IR; `div.approx` or `div.rn` cover the kernels), an
`i32x4`-specific load intrinsic (MIR `vload` of `i32x4` already prints
`ld.global.v4.u32`), `mad.lo` pattern matching (`ptx_mad_lo_u32` exists;
ptxas fuses `mul`+`add` anyway), integer vector types beyond what MIR has.

**Shared memory model.** A `.shared` array is a per-kernel declaration
created by `ptx_shared_alloc_<T>(count)`: `PtxISel` appends a `SharedDecl`
(`.align 16`, name `smem_<index>`, unique within the function; the same
names in different `.entry` bodies are fine because PTX scopes them per
function -- verified with ptxas on a two-kernel module) and materializes the
array's shared-window address with `mov.u64 %rd, smem_<n>` into the `ptr`
result. That 64-bit value is only meaningful to the `ptx_shared_*`
intrinsics, which emit `ld.shared`/`st.shared` with a B64 base register (the
verifier accepts B32 or B64 bases for `.shared`; ptxas accepts B64 with
`.address_size 64`). Plain MIR `load`/`store`/`vload`/`vstore` are always
`.global`; there is no address-space provenance tracking, so passing a shared
pointer to `load` is a silent bug -- always use the shared intrinsics.
Integer arithmetic on the pointer (`add ptr, i64`) is fine as long as the
result is again consumed only by shared intrinsics (the round-trip test does
exactly that). Element counts must be compile-time constants (a
non-constant count throws `PtxISel: ptx_shared_alloc_* requires a positive
compile-time constant element count`). Static shared usage is the sum of the
declared arrays; the launch passes `shared_bytes = 0`.

**Narrow loads and f16.** MIR has no i8/i16 types, so 8/16-bit accesses are
intrinsics that extend into an i32; `ptx_f16_to_f32` takes such an i32 (the
Q8_0 header is `ptx_load_u16(blk)` -> `ptx_f16_to_f32`; the Q4_K header is a
`vload(i32x4)` whose lane 0 is split with `and`/`lshr` before
`ptx_f16_to_f32`, exactly as the hand-written kernel does).

**KernelBuilder helpers (`kernel_jit.hpp`, `kernel_builder_ptx.cpp`).**
Thin wrappers, each a few lines of MIR around the intrinsics above:

- constants: `const_i32(v) const_i64(v) const_f32(v)`
- indices (i32): `tid_x/y/z() ctaid_x/y/z() ntid_x/y/z() nctaid_x/y/z()
  lane_id() warp_id()` (= `tid_x >> 5`) `global_tid_x()`
- barriers: `sync()` (`bar.sync 0`), `bar_sync(id)`
- shuffles: `shfl_down_f32(v, delta|Value*) shfl_up_f32(v, delta)
  shfl_bfly_f32(v, mask) shfl_idx_f32(v, lane|Value*) shfl_down_i32
  shfl_bfly_i32 shfl_idx_i32`
- reductions: `warp_reduce_sum_f32(v)` / `warp_reduce_max_f32(v)` (the
  16/8/4/2/1 `shfl.down` butterfly; result valid in lane 0),
  `block_reduce_sum_f32(v, scratch)`: warp reduce -> lane 0 of each warp
  stores `scratch[warp]` (an `if_then`) -> `bar.sync` -> every warp reads
  `scratch[lane]` for `lane < (ntid+31)/32` (else 0), warp-reduces and
  broadcasts lane 0 with `shfl.idx`, so *every thread* returns the block
  total -> `bar.sync` so `scratch` can be reused at once. Requirements:
  `scratch` is a `shared_alloc_f32` of at least 32 elements, block size a
  multiple of 32 up to 1024, all threads reach the call. The builder is left
  in a new block (the `if_then` join), so values computed afterwards must be
  emitted after the call, which is the natural order anyway.
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
- control flow: `if_then(cond, body)` (creates `if_then`/`if_join` blocks,
  leaves the builder in the join) and `for_range(start, end, step,
  body(i))` (`head(i)`: `i < end` signed -> body -> `br head(i + step)`;
  leaves the builder in the exit block). Both accept bodies that end in
  their own terminator.

`ptx_*` names are GPU-only: `KernelJit` registers no CPU symbols for them,
so a kernel using them is a PTX-only builder (which is what the Stage 5
fused kernels are). The `fabsf/fminf/fmaxf/logf/...` aliases keep working on
both sides: real libm calls on the CPU, inline PTX on the GPU.

### Stage 5a notes (SwiGLU, AdaLN modulate x2, residual RMSNorm)

First migration batch. `emit_ptx_swiglu`, `emit_ptx_adaln_modulate` and
`emit_ptx_fused_residual_rms_norm` (+ the `emit_ptx_residual_rms_norm`
alias) now build MIR and go through `PtxTarget::emit_function`; the entry
names, parameter lists and launch contracts are unchanged, so
`test_gpu_execution.cpp` runs untouched against the MIR kernels.

**File layout**

| File | Concern |
| --- | --- |
| `src/codegen/ml_fusion_ptx_kernels.cpp` | `build_ptx_swiglu`, `build_ptx_adaln_modulate(gated)`, `build_ptx_residual_rms_norm` plus the shared row-per-block prologue (`row_block_prologue`: early `ret`, row offset, `d & ~3` with the `d % 4 != 0` -> scalar-path guard, float4/scalar loop bounds), `f32_offset`, `silu_fast` |
| `src/codegen/ml_fusion_ptx.cpp` | the three thin emitters (build -> `emit_function`) and, for now, the LayerNorm-modulate string kernel |
| `src/codegen/ml_fusion_ptx_legacy.{hpp,cpp}` | **LEGACY**: the four replaced string templates, byte-for-byte, as `legacy::legacy_ptx_{swiglu,adaln_modulate,residual_rms_norm}`. Internal header (tests include it by relative path), no library caller; deleted in Stage 6 |
| `tests/unit/test_gpu_kernel_migration.cpp` | verify + ptxas of each MIR kernel, and on-device differential runs legacy-vs-MIR over several shapes (d multiple of 4 and not, n not a multiple of the block, several rows/blocks, block sizes 32..1024) plus the host reference at the old tolerances |

**Helper added:** `KernelBuilder::for_range_reduce(start, end, step, init,
body(i, acc) -> acc')` -- `for_range` with one loop-carried value (the
RMSNorm sum of squares). The result is the head block's `acc` parameter,
which is valid in the exit block because the head dominates it, so no copy
into the exit block is emitted. `body` must return the new accumulator and
must not end in its own terminator.

**Numerics.** The kernels keep the string kernels' recipes: SiLU is
`neg; mul log2e; ex2.approx; add 1; rcp.approx; mul; mul`, the RMS is
`div.approx(sum, cvt.rn.f32.u32 d); add eps; rsqrt.approx`, the block sum
is the same 16/8/4/2/1 `shfl.down` tree over the same shared partials
(`block_reduce_sum_f32` reduces the partials in every warp and broadcasts
lane 0, instead of warp 0 writing a second shared scalar -- same values,
one shared array instead of two). Differential results (RTX 4090, ptxas
12.9): every SwiGLU and AdaLN shape agrees bit-for-bit (max relative
difference 0); RMSNorm agrees bit-for-bit on 5 of 6 shapes and to 7.5e-8
(1 ulp) on `B=4, D=300, block=256`. The tests require `<= 1e-6` relative.

**ISel quality (observed, not fixed in this batch).** Instruction counts
before ptxas: SwiGLU 76 -> 121 (48 `mov`), RMSNorm 144 -> 200 (56 `mov`).
ptxas removes all of it (the differential outputs are identical), but the
PTX is noisier than the hand-written version for these reasons, in order
of volume:

1. `vextract_lane`/`vinsert_lane` are `mov`s: a 4-lane insert chain to
   build the SwiGLU result vector is 16 `mov.f32`, plus 8 for the extracts.
   A lane-register aliasing scheme (extract = the lane's register, insert
   into a fresh vector = write the lane register directly) would remove
   them; alternatively a `vpack(s0..s3)` MIR op.
2. Constants are always materialized (`mov.b32 %r6, 2; shr.u32 %r7, %r0,
   %r6`; `mov.f32 %f11, 0f3FB8AA3B` per lane) instead of printed as
   immediates -- the string kernels use immediates everywhere. Shift
   counts, `and` masks, `add 1.0` and `vbroadcast` of a constant (4 `mov`s
   of an already-materialized constant) are the common cases. Intrinsics
   that *do* fold constants (`shfl` delta, `shared_alloc` count) leave the
   `mov.b32` of the now-dead constant behind.
3. Block-argument copies: every loop back-edge is `add.s32 %r11, %r8, %r5;
   mov.b32 %r8, %r11` and every loop entry `mov.b32 %r8, %r3; bra` -- the
   parallel-copy resolver could coalesce a copy whose source has no later
   use into the destination register.
4. `br_if` always emits `@%p bra A; bra B;` even when `B` is the next
   block, and `br` to the immediately following block is printed.
5. Special registers are re-read per use (`global_tid_x` reads `%tid.x`
   and `%ntid.x` again after `tid_x()`/`ntid_x()`); a CSE of `mov.u32 %r,
   %tid.x` within a block would fix it.
6. `x + off` address arithmetic is one `add.s64` per pointer per iteration
   (same as the hand-written kernels) -- no loss there; `load_f32_indexed`
   was avoided because it would emit `cvt.s64.s32 + shl + add` per access.

**Guidance for the next batches.**

- LayerNorm-modulate and residual-LayerNorm are the RMSNorm builder with a
  mean pass first: `row_block_prologue` + `for_range_reduce` twice
  (sum, then sum of squared deviations) + two `block_reduce_sum_f32` calls
  on the same 32-float scratch (the helper's trailing `bar.sync` makes
  reuse safe). Keep `div.approx` for the mean/variance and `rsqrt.approx`
  for rstd to preserve the 2e-3 tolerances.
- GEMV SwiGLU/residual: one block per output row, the K loop is
  `for_range_reduce` over float4s with `vfma` into an `f32x4` accumulator
  (vector block arguments work), then extract the 4 lanes, add, and
  `block_reduce_sum_f32`; `silu_fast` is in `ml_fusion_ptx_kernels.cpp`
  and can move to `KernelBuilder` if a third kernel needs it.
- Quantized GEMV: replace the `call brass_dequant_*` in
  `build_gemv_q8_0`/`build_gemv_q4_k` (`ml_fusion_quant_cpu.cpp`) with
  `load_u16`/`f16_to_f32` headers and `load_s8`/`vload(i32x4)` + `and`/`lshr`
  nibble extraction, following the string kernels in
  `ml_fusion_quant_ptx.cpp`; the existing MIR builders already lower and
  verify, so the work is device-side dequantization only.
- Keep the differential test pattern: copy the string kernel verbatim into
  the legacy file first (diff it against the original), then write the
  builder, then compare on device over shapes that hit every path
  (aligned/unaligned row, tail, multi-block, block sizes 32 and 1024).
- Each new kernels file stays under 1,000 lines: put the LayerNorm pair in
  `ml_fusion_ptx_kernels_norm.cpp`, the GEMV kernels in
  `ml_fusion_ptx_kernels_gemv.cpp`; move `row_block_prologue` / `silu_fast`
  into a small internal header when a second file needs them.

## Stages

1. Enablement: Windows CUDA loader, MinGW link fix, portable GPU tests, `ret`
   fix. On-device tests run on this machine. (done)
2. ptx IR + printer + verifier with unit tests. No integration. (done)
3. `PtxISel` replaces `PtxEmitter`; `PtxTarget::emit_function` becomes
   ISel -> verify -> print. All kernels pass ptxas and on-device tests.
   (done; see "Stage 3 implementation notes" above. `ptx_target.cpp` is now
   a 50-line facade, the string emitter is gone, and `test_ptx_isel.cpp`
   covers the parallel-copy resolver, suffix selection and the intrinsic
   table.)
4. Intrinsics and shared memory in MIR + ISel; `KernelBuilder` helpers.
   (done; see "Stage 4 implementation notes" under "Kernel intrinsics":
   vector lowering, the full intrinsic table with MIR signatures, the
   shared-memory model and the helper list. `test_ptx_intrinsics.cpp` and
   `test_ptx_vector.cpp` run every intrinsic and vector opcode on device.)
5. Migrate hand-written kernels to MIR builders (batched), with on-device
   differential tests against the legacy kernels and CPU references.
   (5a done: SwiGLU, AdaLN modulate x2, residual RMSNorm -- see "Stage 5a
   notes". Remaining: LayerNorm-modulate, residual LayerNorm, GEMV
   SwiGLU/residual, GEMV Q8_0/Q4_K.)
6. Remove string templates, decompose files, update docs.

## File size rule

Files stay under 1,000 lines. A file that grows past that is decomposed by
concern (as `x64_isel_*.cpp` is). Files over 2,000 lines are never acceptable.
