# Brass Compiler Optimization & Semantics Specification

This document audits all optimization passes in Brass, specifying observable floating-point (IEEE-754) and integer behaviors, algebraic transformation rules, and explicit opt-in gating flags.

---

## 1. Core Principles & Semantic Laws

1. **Semantics-Changing Optimizations Are Opt-In, Never Default**:
   Any transformation that alters observable floating-point rounding, exception state, precision, or non-wrapping integer behavior is strictly disabled by default.
2. **IEEE-754 Strictness by Default**:
   Floating-point arithmetic adheres strictly to IEEE-754 single-precision (`f32`) and double-precision (`f64`) semantics:
   - Serial summation order is preserved bit-for-bit.
   - Operations are non-associative: $(a + b) + c \neq a + (b + c)$.
   - Signed zeroes (`+0.0` vs `-0.0`), infinities, and NaN payloads are maintained.
   - **NaN operands of `add`/`sub`/`mul`/`div` (`f32`, `f64`)**: when an operand is a NaN, the result is the lhs NaN if the lhs is one, else the rhs NaN, quieted (quiet bit set; sign and the rest of the payload kept). An invalid operation on non-NaN operands (`inf - inf`, `0 * inf`, `0 / 0`) gives the target's default NaN. This is x86 SSE's rule (`addsd dst, src` keeps `dst`'s NaN) and AArch64's with default-NaN off (the first NaN operand); the interpreters and the constant folder follow it through `brass/interpreter/float_arith.hpp`, since a C++ `a + b` lets the compiler commute the operands.
3. **Deterministic Integer Modulo Arithmetic**:
   Integer scalar operations execute two's complement wrapping arithmetic modulo $2^{32}$ (`i32`) and $2^{64}$ (`i64`), with associative and commutative properties valid under modular ring equivalence. Vector integer lanes operate identically per lane.
   - **Signed division overflow wraps**: `sdiv MIN, -1` is `MIN` and `smod MIN, -1` is `0` at both widths. The interpreters, the constant folder (`src/mir/int_fold`), the x64 backends (which guard `idiv` against a `-1` divisor instead of taking the `#DE` fault) and AArch64 (whose `sdiv` already wraps) all produce this result, so passes may fold it and may execute a division by a constant other than `0` speculatively.
   - **Narrow integers (`i8`, `i16`)**: an `iN` value is an N-bit two's-complement bit pattern, and every tier computes on those N bits only:
     - *Representation*: the bits above N in a register or frame slot are not part of the value. The Interpreter, the FastInterpreter and the baseline JIT keep a narrow value zero-extended; tier 2 may leave other bits above N (its producers do not mask). Every consumer whose result depends on those bits reads the operand sign- or zero-extended from N bits, so no tier depends on another's representation (a native `i8` return reaches the interpreters sign-extended, and that is the same value).
     - *Arithmetic wraps*: `add`, `sub`, `mul`, `neg`, `not`, `and`, `or`, `xor` and `shl` give the low N bits of the exact result.
     - *Signedness is the operation's*: `slt`/`sle`/`sgt`/`sge`, `sdiv`, `smod`, `ashr` and `s*_overflow` read their operands as signed N-bit values; `ult`/`ule`/`ugt`/`uge`, `udiv`, `umod`, `lshr` and `u*_overflow` as unsigned N-bit values; `eq`/`ne` compare the N-bit patterns. An overflow flag reports overflow of the N-bit operation. `clz` and `ctz` count within the N bits (N for zero); `popcnt` counts the N bits. `sdiv MIN, -1` wraps to `MIN`, and division by zero is the program error it is at `i32`.
     - *Shift amounts* of N or more give an unspecified result, as a C shift does; frontends must mask the amount.
     - *`switch`*: case values are written as signed N-bit values (the parser and the verifier reject any outside `[-2^(N-1), 2^(N-1))`), and a case is taken when the condition's N-bit pattern equals the case's: on an `i8`, case `-1` is taken for the pattern `0xFF`, and `trunc.i8 261` takes case `5`.
     - *Memory*: `load.iN` reads exactly N bits and zero-extends them; `store.iN` writes exactly N bits.
     - *Conversions*: `trunc.i8` keeps the low 8 bits; `zext_i64` of an `i8` zero-extends it. MIR has no `trunc.i16` and no narrow `sext` or `zext` result (the parser rejects `trunc.i16`, `sext.i8` and the like as unknown opcodes); an `i16` value comes from `load.i16`, a parameter, a call result or arithmetic on those.
     - *Tier 2* rejects `clz`, `ctz` and the overflow checks on narrow operands with a compile-time `UnsupportedOperation` (the function stays in a lower tier); every other narrow operation is compiled.
   - **Division by zero is a program error**: `sdiv`, `udiv`, `smod` and `umod` with a zero divisor raise an error in the interpreters and fault on x64; the result on AArch64 (`0`) is not a defined MIR result. Frontends that need a defined answer must test the divisor. Optimizers never fold such a division, never introduce one, and never hoist or speculate a division whose divisor is not a known non-zero constant.
4. **Memory SSA & Precise Alias Analysis**:
   Load elimination (RLE), store forwarding, and dead store elimination (DSE) are strictly gated by memory SSA barriers, escape analysis, and unaliased provenance proofs.

---

## 2. Pass-by-Pass Semantic Audit

### 2.1. Constant Folding (`constant_folding_pass`)
- **Pass Type**: Scalar simplification.
- **Integer Behavior**: Constant integer expressions (`iconst`, `add`, `sub`, `mul`, `and`, `or`, `xor`, `shl`, `lshr`, `ashr`, comparisons) are evaluated at compile time. Division by zero and out-of-range shifts are never folded; they remain runtime instructions.
- **Floating-Point Behavior**: Constant floating-point expressions (`fconst`, `add`, `sub`, `mul`, `div`, `neg`) are folded respecting IEEE-754 rounding, NaN payloads, signed zeroes, and infinity rules.
- **Flags**: Enabled by default via `LoopOptOptions::enable_dce` in the scalar loop pipeline.

---

### 2.2. Common Subexpression Elimination (`cse_pass`)
- **Pass Type**: Dominator-tree scoped Global Value Numbering (GVN).
- **Integer Behavior**: Replaces redundant computations with dominating equivalent value results.
- **Floating-Point Behavior**: Only eliminates byte-identical subexpressions (same opcode, identical SSA value operands). Does not commute or reassociate floating-point expressions.
- **Flags**: Enabled by default via `LoopOptOptions::enable_dce`.

---

### 2.3. Dead Code Elimination (`dead_code_elimination_pass`)
- **Pass Type**: Liveness-based dead instruction and dead induction cycle pruning.
- **Behavior**: Removes unused instructions that have no side effects (stores, calls, volatile/trapping operations, and guards are preserved).
- **Flags**: Enabled by default via `LoopOptOptions::enable_dce`.

---

### 2.4. Global Value Numbering & Memory Optimization (`gvn_function`)
- **Pass Type**: Scoped hash-based GVN with Redundant Load Elimination (RLE) and Dead Store Elimination (DSE).
- **Dominator-Tree Scoping**: Traverses basic blocks in dominator-tree pre-order, tracking value identities in scoped hash tables.
- **Redundant Load Elimination (RLE)**: Tracks available memory loads using `AvailableLoadKey` composed of:
  - `underlying_base`: Canonical base pointer resolved via `AliasAnalysis::get_underlying_base(base, dummy_offset)`.
  - `total_off`: Offset combining instruction offset and base offset (`off + dummy_offset`).
  - `mtype`: Loaded memory value type.
  - `mem_id`: Memory SSA access version identifier.
  When a dominating load or unaliased store to the same canonical address exists under an identical memory version, redundant loads are replaced with dominating values.
- **Store Forwarding & Dead Store Elimination (DSE)**: Forwards stored values to subsequent loads without intermediate memory reads, and removes stores that are unconditionally overwritten before any intervening read or clobber.
- **Memory SSA Barriers**: Function calls, safepoints, volatile memory accesses, and aliasing stores act as clobber boundaries that invalidate affected memory keys.
- **Flags**: Enabled by default via `LoopOptOptions::enable_gvn = true` / CLI `--gvn`.

---

### 2.5. GVN Partial Redundancy Elimination (`gvn_pre_pass`)
- **Pass Type**: Global Value Numbering with Partial Redundancy Elimination (GVN-PRE).
- **Syntactic & Leader-Based PRE**: Analyzes expressions partially redundant across CFG paths using canonical value leaders from GVN.
- **Dataflow Analysis**:
  - Computes `ANT_LOC` (locally anticipated), `TRANSP` (transparent to clobbers), and `AVAIL_LOC` (locally available) for candidate expressions.
  - Solves backwards anticipability and forwards availability dataflow equations to determine optimal insertion points.
- **Critical Edge Splitting**: Automatically splits critical edges (`split_critical_edges_for_pre`) by inserting synthetic predecessor blocks, preserving single-entry CFG semantics and preventing speculative execution along non-candidate paths.
- **Join Synthesis**: Inserts computations on predecessor branches and creates basic block parameters (phi nodes) at join diamonds.
- **Memory Load-PRE**: Hoists memory loads out of loop headers or across branch diamonds when memory SSA guarantees unaliased transparency along all contributing paths.
- **Flags**: Opt-in via CLI `--enable-pre` / `--enable-gvn-pre`.

---

### 2.6. Sparse Conditional Constant Propagation & Guard Elimination (`sccp_function`)
- **Pass Type**: Inter-block dataflow lattice propagation.
- **Lattice Representation**: Three-level lattice per value: $\top$ (Uninitialized), $\text{Constant}(c)$, $\bot$ (Overdefined), alongside executable flags for CFG branch edges.
- **Branch Pruning & Guard Elimination**: Folds conditional branches with constant conditions, eliminating dead blocks. Folds speculative `guard` instructions proven statically true (`cond == 1`).
- **Safe Float-to-Int Conversion Bounds**:
  - Evaluates float-to-signed-integer conversions (`fptosi_i32`, `fptosi_i32_f32`, `fptosi_i64`, `fptosi_i64_f32`).
  - Strict Boundary Checking: Validates that inputs are non-NaN (`!std::isnan`) and strictly within representable integer ranges:
    - `i32` target: checks $\text{INT32\_MIN} \le d \le \text{INT32\_MAX}$ (or $< 2147483648.0\text{f}$ for `f32`).
    - `i64` target: checks $-9223372036854775808.0 \le d < 9223372036854775808.0$.
  - Out-of-bounds or NaN inputs evaluate to $\bot$ (`LatticeValue::make_bottom(res_type)`), preventing undefined behavior or host CPU trapping during compilation.
- **Flags**: Enabled by default via `LoopOptOptions::enable_sccp = true` and `LoopOptOptions::enable_guard_elim = true` / CLI `--sccp`, `--guard-elim`.

---

### 2.7. Scalar Replacement of Aggregates (`sroa_function`)
- **Pass Type**: Memory aggregate disintegration into SSA scalar values.
- **Behavior**: Analyzes non-escaping stack allocations and structured heap memory. Disintegrates aggregates into discrete scalar fields indexed by byte offset, replacing pointer loads and stores with direct SSA values.
- **Overlapping Field Rejection**:
  - Gathers all field access intervals $[start, start + size)$.
  - For every distinct pair of fields $(1, 2)$, checks for interval overlap:
    $$\max(start_1, start_2) < \min(start_1 + size_1, start_2 + size_2)$$
  - If any two field byte ranges overlap, SROA strictly rejects the candidate allocation from promotion. This prevents corruption caused by type-punned memory, union field sharing, or unaligned partial byte slices.
- **Allocation Pruning**: When all loads from an unescaping aggregate are eliminated or dead, the backing allocation and all remaining dead stores are removed.
- **Flags**: Gated by `LoopOptOptions::enable_sroa` (default `false`) / CLI `--sroa`.

---

### 2.8. Value Range Analysis & Bounds Check Elimination (`run_bounds_check_elimination`)
- **Pass Type**: Abstract interpretation range lattice analysis.
- **Behavior**: Determines integer interval bounds `[min, max]` for loop induction variables and array indices. Removes redundant bounds check comparisons and branches when proven within valid array length.
- **Flags**: Gated by `LoopOptOptions::enable_bce` (default `false`) / CLI `--bce`.

---

### 2.9. Partial Escape Analysis & Allocation Sinking (`sink_allocations`)
- **Pass Type**: Object flow escape graph analysis and code motion.
- **Behavior**: Identifies objects that only escape along cold or exceptional execution paths. Sinks object allocation down into the cold exit paths, allowing the inline hot path to execute purely in CPU registers without GC heap allocation.
- **Flags**: Gated by `LoopOptOptions::enable_allocation_sinking` / `enable_partial_escape` / CLI `--partial-escape`, `--sink-allocations`.

---

### 2.10. Loop Invariant Code Motion (`licm_pass`)
- **Pass Type**: Loop optimization (Dominator & Loop Tree).
- **Integer & Float Behavior**: Hoists loop-invariant scalar arithmetic, constants, and proven unaliased memory reads into loop preheaders. Preserves exact serial floating-point dependencies.
- **Flags**: Enabled by default via `LoopOptOptions::enable_licm = true`.

---

### 2.11. Induction Variable Strength Reduction (`ivsr_pass`)
- **Pass Type**: Counted loop induction optimization (pipeline step `ivsr`, after `loop_vectorize` and `loop_unroll`, whose input form it would otherwise destroy).
- **Integer Behavior**: Replaces scaled `i64` induction variables in `load_indexed`/`store_indexed` addresses ($i \times C$) with an incrementing byte offset, in modular arithmetic. Only raw-pointer or `i64` bases are rewritten (a `gcref` base would become a derived pointer live across the loop). The header's exit compare moves onto the scaled variable only when that is exact: `slt`/`sle`/`ult`/`ule` with constant, non-negative start and limit, a positive step, and no overflow of the scaled limit; otherwise the compare stays on `i`.
- **Floating-Point Behavior**: **Untouched**: Float variables are strictly excluded from IVSR to prevent cumulative rounding drift.
- **Flags**: Enabled by default via `LoopOptOptions::enable_ivsr = true`.

---

### 2.12. CFG Diamond Select Simplification (`simplify_cfg_diamonds`)
- **Pass Type**: Control flow simplification.
- **Behavior**: Replaces conditional branches around single-assignment phi nodes with branchless `select` (x86-64 `cmov` / SSE `blendvpd`). Both operands must be pre-evaluated side-effect-free values.
- **Flags**: Enabled by default via `LoopOptOptions::enable_diamond_select = true`.

---

### 2.13. CFG Simplification & Dead Block Compaction (`cfg_simplify_function`)
- **Pass Type**: Control flow graph canonicalization.
- **Behavior**: Folds unconditional branch trampolines, merges single-entry single-exit block chains, and eliminates unreachable basic blocks.
- **Flags**: Enabled by default via `LoopOptOptions::enable_cfg_simplify = true` / CLI `--cfg-simplify`.

---

### 2.14. SSA Jump Threading (`run_jump_threading`)
- **Pass Type**: Path-sensitive branch elimination.
- **Behavior**: Threads conditional jumps through predecessor blocks where edge values determine the branch target, cloning target blocks when beneficial to eliminate pipeline-stalling branches.
- **Flags**: Gated by `LoopOptOptions::enable_jump_threading` (default `false`) / CLI `--jump-threading`.

---

### 2.15. Loop Unswitching (`unswitch_loops_in_function`)
- **Pass Type**: Loop invariant branch hoisting.
- **Behavior**: Duplicates loop bodies around loop-invariant conditions, hoisting branch decisions outside the loop into a single outer branch.
- **Flags**: Gated by `LoopOptOptions::enable_loop_unswitch` (default `false`) / CLI `--loop-unswitch`.

---

### 2.16. Loop Tiling & Cache Blocking (`loop_tile_pass`)
- **Pass Type**: Polyhedral loop nest transformation.
- **Behavior**:
  - Tiles multidimensional nested loops into $B \times B$ cache-friendly iteration blocks (default block size 16).
  - Strip-mines outer iteration spaces into tile loops and element loops, then applies loop interchange to ensure inner iterations access contiguous cache lines with stride-1 access patterns.
  - Improves CPU L1/L2 data cache hit rates and eliminates cache line bouncing.
  - Dependence Validation: Tiling legality is checked against loop dependence vectors, ensuring no negative distance dependences are inverted.
- **Flags**: Gated by `LoopOptOptions::enable_loop_tile` (default `false`) / CLI `--loop-tile`.

---

### 2.17. Loop Fusion & Loop Distribution (`loop_fusion_pass`, `loop_distribution_pass`)
- **Pass Type**: Polyhedral loop restructuring.
- **Loop Fusion (Jamming)**:
  - Merges two adjacent counted loops into a single loop body.
  - Legality Criteria: Loops must be directly adjacent (no intervening side-effecting code), share identical trip counts and iteration domains, and have no backwards loop-carried data dependencies ($L_2 \to L_1$).
  - Benefits: Promotes temporal cache locality by allowing consumer loops to read data while it remains warm in L1/L2 cache from the producer loop, while eliminating redundant induction variables and branches.
- **Loop Distribution (Fission)**:
  - Splits a complex loop containing independent statement groups into separate loops.
  - Isolates vectorizable or parallelizable statements from operations with dependencies or side-effects.
- **Flags**: Gated by `LoopOptOptions::enable_loop_fusion` and `enable_loop_distribution` / CLI `--enable-loop-fusion`, `--enable-loop-distribution`.

---

### 2.18. Array Contraction (`array_contraction_pass`)
- **Pass Type**: Intermediate buffer elimination.
- **Behavior**: Eliminates temporary intermediate arrays created between producer and consumer loops, converting array writes/reads into scalar SSA registers.
- **Flags**: Gated by `LoopOptOptions::enable_array_contraction` (default `false`) / CLI `--enable-array-contraction`.

---

### 2.19. Loop Dependence Analysis & Auto-Parallelization (`auto_parallelize_function`)
- **Pass Type**: Affine loop dependence analysis and multithreaded scheduling.
- **Distance & Direction Vectors**:
  - For every pair of memory accesses inside nested loops (RAW, WAR, WAW), computes iteration distance vectors ($\vec{d} = (d_1, \dots, d_m)$) and direction vectors ($\vec{D} \in \{=, <, >, \le, \ge, \neq, *\}$).
- **Alias Analysis Disambiguation**:
  - Evaluates base pointers using `AliasAnalysis::alias(base1, base2)`.
  - If pointer bases are proven unaliased (`AliasResult::NoAlias`), no loop-carried dependence exists.
  - If alias analysis returns `MayAlias` or offsets cannot be proven disjoint, conservative `DependenceDirection::Any` dependences are recorded.
- **Auto-Parallelization**:
  - When outer loop carried dependencies are absent ($\vec{d}_0 = 0$ or direction $=$), iterations are proven independent.
  - Partitions loop trip counts into contiguous chunks executed across worker threads via `brass_parallel_for` and `ParallelRuntime`.
  - Supports algebraic reductions (Sum, Product, Min, Max for `i64` and `f64`) with thread-local accumulators and tree reduction combiners.
- **Flags**: Gated by `LoopOptOptions::enable_parallel_loops` (default `false`) / CLI `--enable-parallel-loops`.

---

### 2.20. Vectorization (Counted Loop & SLP) (`loop_vectorize_pass`, `slp_vectorize_function`)
- **Pass Type**: SIMD vector code generation (SSE 128-bit and AVX2 256-bit).
- **Counted Loop Vectorization**:
  - Unrolls counted loops by vector lane factor $W$ (e.g., 4 for `f32x4` / `i32x4`, 8 for `f32x8`), emitting SIMD vector operations.
  - **Trip-Count Guard ($N \ge W$)**:
    - Constructs an explicit trip-count guard in the loop preheader:
      $$tc = \text{limit} - \text{init} \quad (\text{or } +1 \text{ for } \le)$$
    - Evaluates validity ($\text{init} < \text{limit}$) and sufficient iterations ($tc \ge W$).
    - Directs small trip counts ($tc < W$) through a conditional branch (`br_if tc_guard, vec_hdr, vec_exit`) to bypass the vector body completely and enter the scalar epilogue loop.
    - Eliminates vector underflow, buffer overrun, and memory corruption on small iterations.
  - **Scalar Epilogue**: Handles tail remainder iterations ($N \bmod W$).
- **SLP Vectorization**: Combines independent isomorphic scalar computations into parallel SIMD vector operations within basic blocks.
- **Flags**: Enabled by default via `LoopOptOptions::enable_slp = true`, `enable_vectorize = true` / CLI `--slp`, `--vectorize`.

---

### 2.21. FMA Optimization (`fma_opt_pass`)
- **Pass Type**: Fused multiply-add contraction.
- **Behavior**: Contracts separate `mul` followed by `add` into native hardware FMA instructions (`fma.f32`, `fma.f64`, `vfma`).
- **Flags**: Gated by `LoopOptOptions::enable_fma` / CLI `--enable-fma`.

---

### 2.22. F64 Demotion (`f64_demote_pass`)
- **Pass Type**: Precision narrowing optimization.
- **Behavior**: Safely narrows 64-bit double precision operations to 32-bit single precision (`f32`) when proven that inputs originate from 32-bit values and narrowing preserves exact numerical results.
- **Flags**: Enabled by default via `LoopOptOptions::enable_f64_demote = true`.

---

### 2.23. Loop Unrolling & Reduction Jam (`loop_unroll_pass`)
- **Pass Type**: Counted loop unrolling and parallel reduction jamming.
- **Integer Behavior (Default: ON)**:
  - Loops with integer accumulation (`add`) are unrolled by factor $F$ (default 4).
  - Accumulators are split into $F$ parallel sub-accumulators and combined at loop exit. Modular two's complement integer arithmetic is bit-exact identical to serial execution.
- **Floating-Point Behavior (Default: OFF / Strict IEEE-754)**:
  - **Strict Mode (Default)**: Floating-point loops preserve serial accumulation chains bit-for-bit.
  - **Opt-In Reassociation Mode (Flagged)**: $F$-way accumulator splitting and tree reduction for `f64` loops are applied only when explicitly requested.
  - **Gating Controls**:
    - `Function::set_allow_fp_reassociation(bool)`
    - `Module::set_allow_fp_reassociation(bool)`
    - `LoopOptOptions::enable_fp_reassociation = true`

---

### 2.24. Interprocedural Inlining & Speculative Devirtualization (`inline_module`, `run_speculative_devirtualization`)
- **Pass Type**: Call-graph based IPO and feedback-directed inlining.
- **Behavior**: Inlines small non-recursive functions based on instruction size heuristics. Speculative inlining devirtualizes polymorphic call sites using Type Feedback Vectors (TFV), emitting fast-path inlined guards with slow-path fallback.
- **Flags**: Gated by `--inline` and `--speculative-inlining`.

---

### 2.25. Write Barrier Elimination (`WriteBarrierElimPass`)
- **Pass Type**: Generational GC write barrier redundancy elimination.
- **Behavior**: Removes redundant `write_barrier` instructions when an object reference was already card-marked in the same basic block or when the target object is proven freshly allocated in the nursery.
- **Flags**: Gated by CLI `--wbe`, `--enable-wbe`.

---

### 2.26. Codegen Scheduling, Pipelining & LIR Peephole
- **Instruction Scheduler (`InstructionScheduler`)**: Reorders machine instructions using dependency DAG critical path analysis to minimize pipeline register stalls and latency.
- **Software Pipelining (`SoftwarePipelinePass`)**: Modulo-schedules loop iterations across pipelined functional units.
- **LIR Peephole Optimizer (`PeepholeOptimizer`)**: Post-regalloc peephole optimizations (redundant move elimination, load-after-store forwarding, dead move elimination, xor-zeroing, and branch folding).

---

## 3. Optimization Pass Summary Matrix

| Pass Name | Applies to Types | Semantic Impact | Default Status | Gating Flag / Option |
| :--- | :--- | :--- | :--- | :--- |
| **Constant Folding** | `i32`, `i64`, `f32`, `f64` | Bit-exact IEEE-754 & 2's complement | ON | `LoopOptOptions::enable_dce` |
| **CSE** | All types | Bit-exact; non-reassociating | ON | `LoopOptOptions::enable_dce` |
| **DCE** | All types | Prunes dead values & induction cycles | ON | `LoopOptOptions::enable_dce` |
| **GVN (CSE + RLE + DSE)** | All types | Memory SSA unaliased store forwarding & RLE | ON | `LoopOptOptions::enable_gvn` |
| **GVN-PRE** | All types | Hoists partial redundancies across joins & critical edges | OFF | CLI `--enable-pre` |
| **SCCP & Guard Elim** | `i32`, `i64`, `f32`, `f64` | Folds constants, true guards & safe float casts | ON | `LoopOptOptions::enable_sccp` |
| **CFG Simplification** | Control flow | Merges blocks, removes dead code | ON | `LoopOptOptions::enable_cfg_simplify` |
| **LICM** | All types | Hoists loop-invariant operations | ON | `LoopOptOptions::enable_licm` |
| **IVSR** | `i64` only | Bit-exact modular arithmetic | ON | `LoopOptOptions::enable_ivsr` |
| **Diamond Select** | All types | Branchless select (`cmov`) | ON | `LoopOptOptions::enable_diamond_select` |
| **Integer Unroll Jam** | `i32`, `i64` | Bit-exact 2's complement parallel split | ON | `LoopOptOptions::enable_unroll` |
| **FP Unroll Jam** | `f32`, `f64` | Reassociates IEEE-754 additions | **OFF** (Strict) | `LoopOptOptions::enable_fp_reassociation` |
| **SLP Vectorization** | `f32`, `f64`, `i32`, `i64` | SIMD isomorphic operation packing | ON | `LoopOptOptions::enable_slp` |
| **Loop Vectorization** | `f32`, `f64`, `i32`, `i64` | Counted loop vector unrolling with $N \ge W$ guard | ON | `LoopOptOptions::enable_vectorize` |
| **F64 Demotion** | `f64` -> `f32` | Safe precision narrowing | ON | `LoopOptOptions::enable_f64_demote` |
| **FMA Optimization** | `f32`, `f64`, vectors | Contracts mul+add to hardware FMA | OFF | `LoopOptOptions::enable_fma` |
| **SROA** | Structs, aggregates | Scalarizes memory with overlap rejection | OFF | `LoopOptOptions::enable_sroa` |
| **Range Analysis & BCE** | `i32`, `i64` | Eliminates redundant bounds checks | OFF | `LoopOptOptions::enable_bce` |
| **PEA & Alloc Sinking** | `gcref`, heap objects | Sinks allocations to cold exit paths | OFF | `LoopOptOptions::enable_allocation_sinking` |
| **Loop Unswitch** | Loops | Duplicates loop around invariant cond | OFF | `LoopOptOptions::enable_loop_unswitch` |
| **Jump Threading** | CFG branches | Eliminates redundant branch paths | OFF | `LoopOptOptions::enable_jump_threading` |
| **Loop Tiling & Cache** | Nested loops | Blocks loop nests for cache locality | OFF | `LoopOptOptions::enable_loop_tile` |
| **Loop Fusion / Fission** | Loops | Merges or splits loop iterations | OFF | `LoopOptOptions::enable_loop_fusion` / `distribution` |
| **Array Contraction** | Memory buffers | Eliminates intermediate array buffers | OFF | `LoopOptOptions::enable_array_contraction` |
| **Auto-Parallelization** | Counted loops | Affine loop dependence & parallel chunk dispatch | OFF | `LoopOptOptions::enable_parallel_loops` |
| **Inlining (Static & Spec)**| Call sites | Inlines call targets, fastpath guards | OFF | CLI `--inline`, `--speculative-inlining` |
| **Write Barrier Elim** | `gcref` stores | Eliminates redundant card markings | OFF | CLI `--wbe`, `--enable-wbe` |
| **Instruction Scheduling** | Machine LIR | DAG critical path latency scheduling | ON | `SchedOptions::enable_scheduling` |
| **Software Pipelining** | Loop kernels | Modulo scheduling for pipelined units| OFF | CLI `--software-pipeline` |
| **LIR Peephole** | Machine LIR | Dead move elimination, zeroing, fold | ON | Codegen pipeline |

---

## 4. Verification and Differential Testing

Compliance is verified through continuous differential testing:
1. **Bit-Exact Differential Tests**: Matrix computations in strict mode are compared against interpreter serial reference using deterministic pseudorandom mantissas. Exact bit equality (`std::memcmp == 0`) is verified.
2. **Bounded ULP Tolerance in Reassociation Mode**: In opt-in reassociation mode, results are verified to stay within $\le 64$ ULPs of strict reference across arbitrary odd and even matrix dimensions.
3. **Differential Fuzzing**: Differential fuzzers validate optimization passes against the unoptimized reference interpreter across random expressions, loops, and object allocations.
