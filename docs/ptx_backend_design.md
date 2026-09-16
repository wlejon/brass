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

### Kernel intrinsics

Fused ML kernels are written as MIR via `KernelBuilder` helpers and lowered
through `PtxISel`. Where a kernel needs a PTX-only concept, it is an MIR
builtin with a lowering rule, not a string. Required set (from the existing
hand-written kernels): thread/block/grid ids, `shfl.sync.down`, `bar.sync`,
per-function `.shared` arrays with `ld.shared`/`st.shared`, `ex2.approx`,
`rcp.approx`, `div.approx`, `rsqrt.approx`, `f16 -> f32` conversion, `v4`
loads/stores of `f32`/`u32`, `fma.rn`, `mad.lo`.

## Stages

1. Enablement: Windows CUDA loader, MinGW link fix, portable GPU tests, `ret`
   fix. On-device tests run on this machine.
2. ptx IR + printer + verifier with unit tests. No integration.
3. `PtxISel` replaces `PtxEmitter`; `PtxTarget::emit_function` becomes
   ISel -> verify -> print. All kernels pass ptxas and on-device tests.
4. Intrinsics and shared memory in MIR + ISel; `KernelBuilder` helpers.
5. Migrate hand-written kernels to MIR builders (batched), with on-device
   differential tests against CPU references.
6. Remove string templates, decompose files, update docs.

## File size rule

Files stay under 1,000 lines. A file that grows past that is decomposed by
concern (as `x64_isel_*.cpp` is). Files over 2,000 lines are never acceptable.
