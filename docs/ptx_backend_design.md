# PTX Backend Design

Status: complete. This document is the reference for the PTX backend: the
typed PTX IR, the printer and verifier, the instruction selector, the
cleanup passes, and how they are tested. Writing kernels against the
backend -- the intrinsic table, the `KernelBuilder` GPU helpers, the shared
memory model and the fused ML kernels -- is covered in
[ptx_kernel_authoring.md](ptx_kernel_authoring.md). The "History" section
at the end records what each stage of the restructuring changed and measured.

## Background

The PTX target originally had two parallel implementations: a single-pass
walk over MIR that wrote PTX text straight into an `ostringstream` (type
suffixes, predicate materialization, block-argument copies and temporaries
all decided inline while producing strings, with no data model in which
"`ret` with an operand inside `.entry`" is a type error), and ~2,100 lines
of hand register-allocated `R"PTX(...)"` templates for the fused ML kernels,
duplicating MIR builders for several of the same kernels.

The x64 backend is shaped as MIR -> `X64ISel` -> LIR (typed instruction data
structure) -> passes -> `X64Encoder` -> bytes. The PTX backend now has the
same shape, with a text printer in place of the byte encoder; PTX has
virtual registers, no frames and no encoding, so there is no register
allocator and no `CodeBuffer`. Every fused kernel is MIR built with
`KernelBuilder`; no PTX string exists in the library.

## Pipeline

```
MIR --PtxISel--> ptx::Function --ptx::cleanup--> PtxVerifier --> PtxPrinter --> text
                 (typed PTX IR)   (copy prop, DCE,   (structural,
                                   branches, renumber)  no ptxas)
```

`PtxTarget::emit_function(fn, opts)` (`src/target/ptx_target.cpp`, a
50-line facade) runs the four steps; `emit_module` does it per function and
concatenates the bodies under one header. `PtxOptions` carries the
`.target` (`sm_arch`, default `sm_70`), the `.version` (7.0) and
`cleanup` (default true; false prints the raw ISel output, which is the same
program). A verifier failure on the lowered IR is a compiler bug and throws
with every diagnostic attached; nothing that fails verification is printed.

## ptx IR (`include/brass/target/ptx/ptx_ir.hpp`)

- `ptx::RegClass { Pred, B32, B64, F32, F64 }` with one counter per class per
  function (`Function::new_pred/new_b32/...`). Register counts are known by
  construction; `.reg` declarations are derived from the function, never
  guessed. Numbering starts at 0; there is no reserved register.
- `ptx::Type`: the suffix set (`.pred .b8 .b16 .b32 .b64 .u8 .u16 .u32 .u64
  .s8 .s16 .s32 .s64 .f16 .f32 .f64`). Three functions map MIR `Type` to a
  suffix and nothing else decides suffixes: `type_for` (data suffix:
  u32/u64/f32/f64, used for ld/st/param/cvt), `signed_type_for` (s32/s64 for
  signed arithmetic and comparisons) and `bit_type_for` (b32/b64 for
  mov/selp/bitwise/shifts). `wide_type_for` gives the `mul.wide`/`mad.wide`
  result type. `reg_class_for` maps a suffix to its register class:
  sub-32-bit types and `.f16` live in B32 registers (PTX permits wider
  registers for `ld/st/cvt` of narrow types; `ld.global.u16 %r` and
  `cvt.f32.f16 %f, %r` rely on it).
- `ptx::Opcode` for every mnemonic the backend emits: `ld st mov cvt add sub
  mul mad fma div rem neg abs min max and or xor not shl shr setp selp bra
  ret call shfl bar atom rsqrt sqrt sin cos ex2 lg2 rcp trap exit`.
- Modifiers are typed fields, not strings: rounding (`.rn .rz .rm .rp`),
  `.approx`, `.ftz`, state space (`.global .shared .param .local`), vector
  width (`.v2 .v4`), `.lo/.hi/.wide`, comparison operator, `.sync`, shuffle
  mode, `AtomOp` (printed as `atom.global.add.f32`).
- `ptx::Operand` tagged union: register (class + index), immediate (integer,
  or float with its own width `imm_f32`/`imm_f64`), address (`[reg + disp]`
  with a B32 or B64 base), vector tuple `{r0, r1, r2, r3}`, label, kernel
  param (prints `[name]`, legal only as an `ld.param` source), special
  register (`%tid.x`, `%ctaid.x`, `%ntid.x`, `%nctaid.x`, `%laneid`,
  `%warpid`, `%nwarpid`, `%smid`, `%nsmid`, `%clock`, `%clock64`,
  `%globaltimer`), and `Symbol` (the address-of-shared-array source of
  `mov.u64 %rd, smem_0` and the callee of `call`).
- `ptx::Inst { opcode, type(s), modifiers, guard predicate (optional,
  negatable), dst operands, src operands, MIR origin }`, built with the
  fluent `Inst::make(op, type).dst(..).src(..).guard(..)`.
- `ptx::Block { label, insts }` and `ptx::Function { name, params, shared
  declarations, reg counts, blocks, is_entry }`. Kernel params carry a
  `ptx::Type` and a name; the entry signature is derived from them. A
  `SharedDecl` is `.shared .align 16 .<type> name[count]`.

**Register/type compatibility** is stricter than ptxas except for bit types:
`.f32`/`.f64` operands must be F32/F64 registers and `.u*/.s*` operands must
be B32/B64, but `.b32` accepts B32 *or* F32 and `.b64` accepts B64 *or* F64.
The relaxation is required by `shfl.sync.down.b32` on f32 accumulators and
by `mov.b64` bitcasts between `%rd` and `%fd`.

**Immediate operands.** One table, `allows_immediate(op, src_index)`, is
consulted by `PtxISel::operand_of` when it turns a MIR value into a source
operand and by the verifier for every source of every instruction. It is
deliberately narrower than what ptxas accepts (ptxas takes an immediate in
every source of `setp`, `cvt` and `shfl`; probed on 12.9):

| Opcode | Immediate allowed in source |
| --- | --- |
| `mov`, `neg`, `abs`, `not`, `rsqrt`, `sqrt`, `sin`, `cos`, `ex2`, `lg2`, `rcp` | 0 |
| `add`, `sub`, `mul`, `div`, `rem`, `min`, `max`, `and`, `or`, `xor`, `shl`, `shr` | 0 or 1 |
| `selp` | 0 or 1 (2 is the predicate) |
| `mad`, `fma` | 0, 1 or 2 |
| `setp` | 1 only (a constant on the left keeps its register) |
| `st` | 1 (the stored value; the address is never an immediate) |
| `atom` | 1 and 2 (value, cas compare value) |
| `shfl` | 1, 2, 3 (delta/lane, clamp, member mask; the value never) |
| `bar` | 0, 1 |
| `call` | 1.. (arguments; 0 is the callee) |
| `ld`, `cvt`, `bra`, `ret`, `trap`, `exit` | none |

Width rule (`imm_fits`): a 64-bit slot takes any `int64_t`; a 32-bit slot
takes `-2^31 .. 2^32-1` (either reading), 16/8-bit slots likewise; `.pred`
takes none, so `mov` of a constant into a Pred is not representable.
Consequences: shift counts are `.u32` immediates when the MIR constant fits,
whatever the width of the shifted value; `mul.wide`/`mad.wide` immediates
must fit the *narrow* type (`ptx_mul_wide_u32(x, 144)` prints `mul.wide.u32
%rd, %r, 144`); `selp` folds both values; `st` and the shared/narrow store
rules fold the stored value. Integer immediates print in decimal
(`4294967295` for a full lane mask); float immediates print as `0f%08X` /
`0d%016X` and the verifier checks their width against the instruction type.

## PtxPrinter (`ptx_printer.hpp`)

A single dumb walk that makes no decisions beyond formatting: the header
(`.version`, `.target`, `.address_size 64`), the entry signature, `.reg`
declarations from the counts, `.shared` declarations, then blocks in order
(`print_body` prints only the function, for tests and diagnostics).

## PtxVerifier (`ptx_verifier.hpp`)

Runs on any platform without ptxas and returns a list of diagnostics, each
carrying the function, block label, instruction index and printed
instruction text (`format_diagnostics`). It checks:

- register index < declared count for its class,
- operand register class matches the instruction type (the compatibility
  rule above); an address base is B32 or B64 (`.shared` accepts both),
- guard is a Pred register; `setp` destination is Pred; `selp` condition is
  Pred,
- vector operand arity matches `.vN`,
- shift amounts are B32 (or a `.u32` immediate),
- an immediate appears only where `allows_immediate` permits and fits its
  slot (`imm_fits`); float immediates match the instruction width,
- the modifier requirements ptxas has for PTX 7.x: integer `mul`/`mad` need
  `.lo/.hi/.wide`, `fma` needs a rounding mode, float `div` needs `.approx`
  or a rounding mode, `shfl`/`bar` need `.sync`, `rsqrt/sin/cos/ex2/lg2`
  need `.approx`, int<->float `cvt` needs a rounding mode, `and/or/xor/shl`
  need bit types,
- `ret` inside `.entry` has no operand; every `bra` target label exists;
  `ld.param` sources name a declared param.

It does not require a block to end in a terminator: falling through to the
next block, or off the end of an `.entry`, is valid PTX (probed with ptxas)
and the cleanup passes produce it.

## PtxISel (`ptx_isel.hpp`)

Lowers one MIR `Function` to a `ptx::Function`. File layout (mirrors
`x64_isel_*.cpp`):

| File | Concern |
| --- | --- |
| `include/brass/target/ptx/ptx_isel.hpp` | `class PtxISel { ptx::Function lower(const Function&); }`, the intrinsic-table query API (`is_intrinsic`, `intrinsic_names`), private state |
| `src/target/ptx/ptx_isel.cpp` | driver, use analysis, register assignment, `ld.param` prologue, opcode dispatch, value/register helpers (`reg_of`, `operand_of`, `special_register`, `materialize_pred`, `shift_amount`, `emit`) |
| `src/target/ptx/ptx_isel_alu.cpp` | constants, arithmetic, bitwise, shifts, comparisons, `select`, conversions |
| `src/target/ptx/ptx_isel_mem.cpp` | `load/store`, `vload/vstore` (v4/v2 tuples, two tuples for 256-bit types), `load_indexed/store_indexed` |
| `src/target/ptx/ptx_isel_control.cpp` | `br/br_if/ret/unreachable`, edge copies, the parallel-copy resolver |
| `src/target/ptx/ptx_isel_vec.cpp` | vector MIR ops -> per-lane scalar PTX |
| `src/target/ptx/ptx_isel_intrinsics.hpp` | private: `struct PtxISel::Intrinsics` (all rule declarations) |
| `src/target/ptx/ptx_isel_intrinsics.cpp` | the name -> rule table; special registers, math, conversions, plain `call` |
| `src/target/ptx/ptx_isel_intrinsics_warp.cpp` | `bar.sync`, `shfl`, `atom`, `mul.wide/hi`, `mad.lo` |
| `src/target/ptx/ptx_isel_intrinsics_mem.cpp` | shared memory, narrow loads/stores, operand helpers (`const_int`, `address_operand`, `indexed_operand`, `lane_operand`) |

Rules:

- **Type suffixes** come from the `type_for`/`signed_type_for`/`bit_type_for`
  tables. Unsigned MIR ops (`udiv`, `urem`, `ult`, `lshr`, ...) emit unsigned
  PTX types; `shl` uses `.b32/.b64`, `lshr` `.u32/.u64`, `ashr` `.s32/.s64`.
- **Register mapping lives in one place.** `allocate_registers` assigns every
  MIR value (entry params, block params, instruction results) a register run
  up front, in block order: one register for scalars, a contiguous run of
  `vector_lanes()` registers of the element class for vector types (f32x4 ->
  4 x `%f`, f64x2 -> 2 x `%fd`, i32x4 -> 4 x `%r`, i64x2 -> 2 x `%rd`, and the
  8/4-lane 256-bit types likewise). Comparison results get a Pred register in
  addition to their B32 result. Temporaries (narrowed shift counts, indexed
  addresses, materialized predicates, intrinsic scratch, parallel-copy
  scratch) are allocated lazily during lowering and number after the value
  registers.
- **Constants** are emitted as `mov.b32/b64/f32/f64 %r, c` (`lower_const`),
  and `operand_of` prints the constant as an immediate wherever
  `allows_immediate` permits. ISel does not know whether every use folded,
  so the `mov` stays; cleanup's DCE removes it when nothing reads the
  register.
- **Special registers.** `special_register(s)` returns one register per
  invariant special register per function; the `mov.u32 %r, %tid.x` is
  appended to the `$L_params` prologue on first use. Invariant means
  `is_invariant(s)`: `%tid.*`, `%ntid.*`, `%ctaid.*`, `%nctaid.*`, `%laneid`,
  `%nwarpid`, `%nsmid`. `%warpid` and `%smid` are not (the PTX ISA allows
  them to change when a warp is rescheduled), nor are `%clock`, `%clock64`
  and `%globaltimer`; those are read at every use, in place, and DCE keeps
  such a `mov` even when its result is unused. `global_tid_x` composes the
  cached `%ctaid.x`/`%ntid.x`/`%tid.x` with one `mad.lo.s32`.
- **Comparisons** emit `setp.<cmp>.<type>` into the Pred and only emit the
  `selp.u32 r, 1, 0, p` integer materialization when the result has a use
  other than the condition of `br_if`/`select` (`analyze_uses`). Float `ne`
  is `setp.neu` (unordered), matching the x64 lowering; other float
  comparisons are ordered. Unsigned MIR comparisons use `.u32/.u64`, signed
  ones and `eq/ne` use `.s32/.s64`.
- **Predicate materialization** for a non-comparison `br_if`/`select`
  condition is `setp.ne.u32/u64 p, r, 0` or `setp.neu.f32/f64 p, r, 0f0`,
  emitted at the use (never cached across blocks).
- **Kernel parameters** are declared from `Function::param_types()` as
  `param_<i>` and loaded in a `$L_params` fall-through block placed before
  the MIR entry block (created on demand by `prologue_block()`, so it also
  exists for a kernel without parameters that reads `%tid.x`); an entry
  block with incoming edges is therefore not re-entered through the loads.
  Load types come from the entry block parameter types. Non-void kernels
  are rejected (`.entry` cannot return values).
- **Block arguments** become `Copy{dst, src, bit_type}` lists per edge (one
  per lane for vector values; identity copies dropped).
  `emit_parallel_copies` emits any copy whose destination no other pending
  copy still reads; when only cycles remain it moves one source into a
  scratch register of the same class and redirects its readers, which
  unblocks the cycle. For `br_if` the taken edge's copies are guarded on the
  predicate and precede the guarded `bra`; the fall-through edge's copies
  are unguarded and follow it.
- **Shift counts** that are 64-bit and not foldable are narrowed with
  `cvt.u32.u64` into a fresh B32 (`shift_amount`).
- **Indexed addressing** sign-extends a 32-bit index (`cvt.s64.s32`), scales
  by `shl.b64` for 2/4/8 (or `mul.lo.s64` by an immediate otherwise), adds to
  the base and addresses `[addr + offset]`.
- **Loads and stores** are always `.global`; shared memory is reached only
  through the `ptx_shared_*` intrinsics (see the authoring guide).
- **Errors:** unsupported MIR opcodes throw `runtime_error("PtxISel:
  unsupported opcode in PTX lowering: <name>")`; malformed instructions
  throw naming the opcode and the missing piece.
- Every emitted `Inst` carries `.origin(&mir_inst)` (set by `PtxISel::emit`).

**Vector lowering (`ptx_isel_vec.cpp`).** Vector ops emit one scalar
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
`vload(i32x4)` is the `ld.global.v4.u32 {..}` header load of the Q4_K
kernel; lanes come out with `vextract_lane`. The extract/insert `mov`s are
removed by copy propagation, so they cost nothing in the printed PTX.

**Intrinsic table.** `PtxISel::Intrinsics::table()` maps a callee name to an
`IntrinsicLowering` (`void(*)(PtxISel&, const Instruction&)`); a MIR
`build_call(name, ...)` whose name is in the table is lowered inline, any
other `call` prints as a PTX `call` to an undeclared symbol. To add one:
write a static rule in `PtxISel::Intrinsics` (or instantiate the
`special<SpecialReg>` / `approx_f32<Opcode>` templates) that reads its
arguments with `isel.reg_of(inst.operand(i), "...")` or `isel.const_int`,
allocates scratch with `isel.fn_->new_*()`, writes `isel.result_reg(inst)`
when the call has a result, and calls `isel.emit(...)`; then add one row per
name/alias, a signature row in `tests/unit/test_ptx_intrinsics.cpp` (its
coverage test fails for a table entry without a row) and a line in the
authoring guide's table. `intrinsic_names()` lets tests iterate the table.
Shared arrays are per-kernel `SharedDecl`s appended by the
`ptx_shared_alloc_*` rules (`.align 16`, `smem_<index>`, unique within the
function; the same names in different `.entry` bodies are fine because PTX
scopes them per function -- verified on a two-kernel module).

## Cleanup passes (`ptx_cleanup.hpp`, `src/target/ptx/ptx_cleanup.cpp`)

`ptx::cleanup` runs `simplify_branches`, then `propagate_copies` +
`eliminate_dead_instructions` to a fixed point, then `renumber_registers`.
Each is a public function, is a no-op on clean input (the tests run the
pipeline twice and compare the printed body) and only removes or renames;
none reorders or introduces an instruction. The IR is "SSA-ish": most
registers have exactly one def, block parameters and parallel-copy scratch
registers have several; every pass computes def/use counts first and only
rewrites registers whose def count it can reason about.

1. *Copy propagation* looks at plain copies: an unguarded `mov` with one
   register destination and one register source of the same class and no
   modifier (`mov.b32 %f0, %r3` across register files is not a copy).
   - Identity copies are deleted.
   - Case A, both registers single-def: every read of the destination
     becomes a read of the source and the `mov` is deleted. Block
     parameters (multi-def) and the parallel-copy scratch are never
     rewritten this way.
   - Case B, the source is single-def and this `mov` is its only read: the
     defining instruction earlier in the same block writes the destination
     directly (`add.s32 %r11, %r8, %r5; mov.b32 %r8, %r11` becomes
     `add.s32 %r8, %r8, %r5`). Refused when the def is guarded or has
     several destinations, when anything between the def and the `mov`
     reads or writes the destination (the swap pattern), or when a `bra`,
     `ret`, `exit` or `trap` sits between them -- a guarded `bra` there
     would make the early write visible on the taken path where the `mov`
     never ran. This removes loop back-edge and entry copies.
   - Guarded copies (the `br_if` taken-edge argument copies) are left alone.
2. *Dead instruction elimination* deletes any instruction with a
   destination none of whose registers is read anywhere, to a fixed point
   (so a constant chain disappears in one call). Side effects that keep an
   instruction: `st`, `atom`, `bar`, `call`, `ret`, `bra`, `trap`, `exit`,
   `shfl` (warp-collective; kept even when its result is unused, which is
   conservative) and a `mov` from a non-invariant special register. Loads
   are deletable.
3. *Branch simplification*: unreachable blocks (worklist from the first
   block over `bra` targets and fall-through edges; the first block is
   never removed) are dropped; a trailing `bra next` or `@p bra next` is
   dropped; `@p bra next; bra B` becomes `@!p bra B`; repeated until the
   block's tail is stable.
4. *Register renumbering* compacts each class's indices in order of the
   surviving indices and sets `reg_counts` to the number in use, so the
   printed `.reg` counts shrink with the code.

Deliberately not done (ptxas handles all of it): loop-invariant code motion
(a `vbroadcast` of a loop-invariant scalar inside a loop body is still 4
`mov.f32` per iteration), algebraic identities (`add x, 0` from a folded
zero byte offset survives), swapping `setp` operands so a constant left
operand can fold, `mad.wide` addressing for indexed shared accesses
(`cvt.s64.s32; shl.b64; add.s64` per access), and dropping an unused `shfl`.

## Testing

Every test that needs a GPU or ptxas prints a visible `[SKIP]` line and
passes when they are absent; the library has no link-time CUDA dependency
(`src/gpu/cuda_driver.cpp` loads the driver at runtime on Windows and
Linux). `tests/unit/ptx_test_support.hpp` has the shared helpers (ptxas /
device availability, `lower_ok`, `emit_checked`, `upload`/`launch`/
`download`, the `run_map` one-element-per-thread harness, host f16
conversion).

| File | Covers |
| --- | --- |
| `test_ptx_ir.cpp` | IR construction, printer formatting (exact hex floats, modifiers), every verifier rule with a hand-built failing instruction |
| `test_ptx_isel.cpp` | suffix selection, comparisons and predicate materialization, the parallel-copy resolver (swaps, cycles), indexed addressing, the original intrinsic aliases, error paths |
| `test_ptx_intrinsics.cpp` | a signature row per table entry (coverage-checked against `intrinsic_names()`), ptxas on each, on-device runs per family |
| `test_ptx_vector.cpp` | every vector opcode on device vs a host reference, warp/block reductions, shared memory round trips, `if_then`/`for_range` and the unrolled loop forms |
| `test_ptx_cleanup.cpp` | each pass on hand-built IR, immediate folding and the special-register cache, the pipeline over every intrinsic and all ten kernels (verify + ptxas, idempotence) |
| `test_ptx_target.cpp` | the facade: header, params, `ret`, options |
| `test_gpu_execution.cpp` | the CPU-style MIR kernels and every emitted fused kernel on device against host references |
| `test_gpu_kernels.cpp`, `test_gpu_kernels_quant.cpp` | the ten `build_ptx_*` kernels: verify, ptxas, recipe checks on the PTX text, on-device runs against double-precision host references over shape/block-size grids (see the authoring guide for the tolerances) |
| `test_ml_fusion.cpp` | the CPU JIT side of the fused kernels |

## History

The restructuring was done in six stages, one commit each.

**Stage 1 -- enablement.** Windows CUDA driver loader (`LoadLibrary` of
`nvcuda.dll`, `dlopen` of `libcuda.so` on Linux), a MinGW link fix, the
`ret`-inside-`.entry` fix in the old emitter, and the first on-device tests.

**Stage 2 -- typed IR, printer, verifier.** `ptx_ir.hpp`, `ptx_printer.cpp`,
`ptx_verifier.cpp` with unit tests, no integration. The register/type
compatibility relaxation for bit types, the `Symbol` operand, `atom`/`trap`/
`exit`, the three suffix tables and the PTX 7.x modifier requirements in
the verifier date from here.

**Stage 3 -- PtxISel.** Replaced the string emitter; `PtxTarget` became
ISel -> verify -> print. Golden comparison against the string emitter over
every MIR kernel: after normalizing register numbers the only differences
were the `$L_params:` label, the dropped always-declared `.reg .f64 %fd<1>`,
and elided dead `selp.u32` materializations for loop-exit comparisons. No
kernel got longer.

**Stage 4 -- intrinsics, vectors, shared memory, KernelBuilder.** The full
intrinsic table (every mnemonic, modifier, special register and state space
the hand-written kernels used), per-lane vector lowering for all eight
vector types, `.shared` arrays with `ld/st.shared`, narrow loads, f16
conversion, and the `KernelBuilder` GPU helpers (indices, barriers,
shuffles, warp/block reductions, shared memory, fast math, structured
control flow). 126 intrinsic test kernels and every vector opcode run on
device.

**Stage 5 -- kernel migration (5a/5b/5c).** The ten string kernels were
rewritten as MIR builders with unchanged entry names, parameter lists and
launch contracts, batch by batch, each batch validated by an on-device
differential test against the string kernel it replaced (RTX 4090, ptxas
12.9): 5a SwiGLU, AdaLN modulate (gated and not), residual RMSNorm -- every
shape bit-identical except one RMSNorm shape at 7.5e-8 (1 ulp); 5b
LayerNorm-modulate, residual LayerNorm, GEMV SwiGLU, GEMV residual -- the
GEMVs bit-identical on all 7 shapes, the norms bit-identical on 12 of 16
and within 1.1e-7 on the rest; 5c GEMV Q8_0 and Q4_K -- bit-identical on
all 11 shapes at block 256, <= 2.6e-6 against a double-accumulation host
reference at every block size, <= 1.9e-6 against the CPU JIT GEMVs. The
1-ulp cases were a ptxas difference, not an ISel one: the MIR kernels'
unpredicated `div.approx(var, d); add eps` is contracted by ptxas
(`--fmad=true`) into one `FFMA`, whereas the string kernels' predicated
sequence stayed `FMUL; @!P0 FADD`. Two contract fixes came out of 5c: the
string Q8_0/Q4_K kernels stepped the block index by a hard-coded 32 / 4
(correct only for a 256-thread launch); the MIR kernels derive the stride
from `ntid` and are correct for every block size that is a multiple of 8 /
64. Helpers added: `for_range_reduce`, `for_range_reduce_n`,
`block_reduce_sum_f32`.

**Stage 6a -- immediates, special-register cache, cleanup passes.** PTX
instruction counts before ptxas over the ten kernels: the string kernel,
the 5c ISel output, and the cleaned output (`mov` count in parentheses):

| Kernel | string | 5c ISel (`mov`) | 6a cleaned (`mov`) |
| --- | ---: | ---: | ---: |
| SwiGLU | 76 | 121 (48) | 73 (4) |
| AdaLN modulate | -- | 88 (22) | 69 (8) |
| AdaLN modulate, gated | -- | 98 (22) | 79 (8) |
| residual RMSNorm | 144 | 200 (56) | 144 (11) |
| LayerNorm-modulate | 219 | 332 (100) | 234 (19) |
| residual LayerNorm | 212 | 319 (94) | 223 (15) |
| GEMV SwiGLU | 146 | 245 (87) | 158 (11) |
| GEMV residual | 100 | 155 (52) | 102 (8) |
| GEMV Q8_0 | 108 | 167 (56) | 110 (6) |
| GEMV Q4_K | 156 | 237 (83) | 153 (6) |

The 126 intrinsic test kernels went from 1,684 to 633 instructions in
total. After ptxas (sm_89 SASS) the MIR norms were already smaller than
the string kernels (240 vs 312 instructions: the `s_mean`/`s_rstd` shared
round trips are gone and `rcp(d)` is CSE'd), the GEMV residual equal
(112), the GEMV SwiGLU 168 vs 160 (three extra `bar.sync` from running
`block_reduce_sum_f32` twice), Q4_K equal (152).

**Stage 6b -- string kernels deleted, Q8_0/Q4_K unrolling, docs.** The
`ml_fusion_ptx_legacy*` files and the differential tests are gone; the
kernels are checked against double-precision host references instead
(`test_gpu_kernels*.cpp`). The one performance regression of the migration
was the Q8_0 block loop: ptxas had unrolled the string kernel's loop 8x
because its stride was a literal, but keeps the MIR loop rolled because
the stride is `ntid >> 3` at runtime. `KernelBuilder::for_range*` gained
an `unroll` factor (main loop over `unroll` bodies with a guarded
remainder loop) and both quantized GEMVs use 4. SASS (sm_89, ptxas 12.9):

| Kernel | string kernel | MIR rolled | MIR unroll 2 | MIR unroll 4 | MIR unroll 8 |
| --- | --- | --- | --- | --- | --- |
| Q8_0 instructions / FFMA / LDG / regs | 432 / 32 / 32 / 27 | 112 / 4 / 4 / 23 | 192 / 12 / 12 / 34 | **264 / 20 / 20 / 39** | 392 / 36 / 36 / 40 |
| Q4_K instructions / FFMA / LDG / regs | 152 / 8 / 3 / 27 | 152 / 8 / 3 / 29 | 272 / 24 / 9 / 40 | **384 / 40 / 15 / 40** | 600 / 72 / 27 / 40 |

(FFMA/LDG counts include the remainder loop's copy of the body.) Measured
on the RTX 4090 at block 256, microseconds per launch: Q8_0 4096x4096
12.0 -> 10.7 (unroll 4; 12.4 at 8), 11008x4096 22.5 -> 21.2 (23.1 at 8);
Q4_K 4096x4096 14.5 -> 12.9, 16384x16384 175.1 -> 165.4 (165.2 at 8). The
16384x16384 Q8_0 shape is memory-bound at ~945 GB/s with every factor. No
`.maxntid`/`.reqntid` directive was needed.

## File size rule

Files stay under 1,000 lines. A file that grows past that is decomposed by
concern (as `x64_isel_*.cpp` and `ptx_isel_*.cpp` are). Files over 2,000
lines are never acceptable.
