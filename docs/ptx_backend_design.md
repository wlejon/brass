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
   Starts from the extension points listed in the Stage 3 notes.
5. Migrate hand-written kernels to MIR builders (batched), with on-device
   differential tests against CPU references.
6. Remove string templates, decompose files, update docs.

## File size rule

Files stay under 1,000 lines. A file that grows past that is decomposed by
concern (as `x64_isel_*.cpp` is). Files over 2,000 lines are never acceptable.
