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
3. **Deterministic Integer Modulo Arithmetic**:
   Integer scalar operations execute two's complement wrapping arithmetic modulo $2^{32}$ (`i32`) and $2^{64}$ (`i64`), with associative and commutative properties valid under modular ring equivalence. Vector integer lanes operate identically per lane.
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
- **Behavior**: Evaluates memory SSA versioning to forward unaliased stores to subsequent loads, eliminating redundant memory reads. Traps and volatile boundaries act as memory clobbers.
- **Flags**: Enabled by default via `LoopOptOptions::enable_gvn = true` / CLI `--gvn`.

---

### 2.5. GVN Partial Redundancy Elimination (`gvn_pre_pass`)
- **Pass Type**: Global Value Numbering with Partial Redundancy Elimination (GVN-PRE).
- **Behavior**: Identifies partially redundant expressions across CFG join diamonds and inserts computations in predecessor paths to hoist calculations into common predecessors without altering execution paths.
- **Flags**: Opt-in via CLI `--enable-pre` / `--enable-gvn-pre`.

---

### 2.6. Sparse Conditional Constant Propagation & Guard Elimination (`sccp_function`)
- **Pass Type**: Inter-block dataflow lattice propagation.
- **Behavior**: Propagates constant integers, floats, and branch conditions simultaneously. Eliminates dead CFG branches and folds speculative `guard` instructions whose condition is statically proven true (`cond == 1`).
- **Flags**: Enabled by default via `LoopOptOptions::enable_sccp = true` and `LoopOptOptions::enable_guard_elim = true` / CLI `--sccp`, `--guard-elim`.

---

### 2.7. Scalar Replacement of Aggregates (`sroa_function`)
- **Pass Type**: Memory aggregate disintegration into SSA scalar values.
- **Behavior**: Identifies non-escaping stack allocations and structured memory objects. Replaces struct fields and disjoint memory slots with discrete SSA registers, eliminating heap allocations and pointer indirection.
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
- **Pass Type**: Counted loop induction optimization.
- **Integer Behavior**: Transforms linear induction expressions ($v = i \times C + B$) into incremental additions across loop iterations in modular arithmetic.
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
- **Behavior**: Tiles multidimensional nested loops into $N \times N$ cache-friendly iteration blocks (default block size 16) and applies loop interchange to optimize CPU L1/L2 cache locality.
- **Flags**: Gated by `LoopOptOptions::enable_loop_tile` (default `false`) / CLI `--loop-tile`.

---

### 2.17. Loop Fusion & Loop Distribution (`loop_fusion_pass`, `loop_distribution_pass`)
- **Pass Type**: Polyhedral loop restructuring.
- **Behavior**: Loop fusion (jamming) merges adjacent loops with identical trip counts to increase temporal data reuse; loop distribution (fission) splits independent loop statements to expose vectorization opportunities.
- **Flags**: Gated by `LoopOptOptions::enable_loop_fusion` and `enable_loop_distribution` / CLI `--enable-loop-fusion`, `--enable-loop-distribution`.

---

### 2.18. Array Contraction (`array_contraction_pass`)
- **Pass Type**: Intermediate buffer elimination.
- **Behavior**: Eliminates temporary intermediate arrays created between producer and consumer loops, converting array writes/reads into scalar SSA registers.
- **Flags**: Gated by `LoopOptOptions::enable_array_contraction` (default `false`) / CLI `--enable-array-contraction`.

---

### 2.19. Polyhedral Auto-Parallelization (`auto_parallelize_function`)
- **Pass Type**: Multithreaded loop scheduling.
- **Behavior**: Uses polyhedral distance/direction vector dependence analysis to prove absence of loop-carried dependencies, generating task chunks executed across worker threads via `ParallelRuntime`.
- **Flags**: Gated by `LoopOptOptions::enable_parallel_loops` (default `false`) / CLI `--enable-parallel-loops`.

---

### 2.20. Vectorization (SLP & Counted Loop) (`slp_vectorize_function`, `loop_vectorize_pass`)
- **Pass Type**: SIMD vector code generation (SSE 128-bit and AVX2 256-bit).
- **Behavior**:
  - **SLP Vectorization**: Combines independent isomorphic scalar computations into parallel SIMD vector operations (`f32x4`, `f64x2`, `f32x8`, `f64x4`, `i32x4`, etc.).
  - **Loop Vectorization**: Unrolls counted loops by vector factors (e.g. 4 or 8), generating vector loads, stores, and arithmetic.
  - Floating-point reduction loops adhere strictly to IEEE serial order unless FP reassociation is explicitly permitted.
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
| **GVN (CSE + RLE + DSE)** | All types | Memory SSA unaliased store forwarding | ON | `LoopOptOptions::enable_gvn` |
| **GVN-PRE** | All types | Hoists partial redundancies across joins | OFF | CLI `--enable-pre` |
| **SCCP & Guard Elim** | `i32`, `i64`, `f32`, `f64` | Folds constants & true guards | ON | `LoopOptOptions::enable_sccp` |
| **CFG Simplification** | Control flow | Merges blocks, removes dead code | ON | `LoopOptOptions::enable_cfg_simplify` |
| **LICM** | All types | Hoists loop-invariant operations | ON | `LoopOptOptions::enable_licm` |
| **IVSR** | `i32`, `i64` only | Bit-exact modular arithmetic | ON | `LoopOptOptions::enable_ivsr` |
| **Diamond Select** | All types | Branchless select (`cmov`) | ON | `LoopOptOptions::enable_diamond_select` |
| **Integer Unroll Jam** | `i32`, `i64` | Bit-exact 2's complement parallel split | ON | `LoopOptOptions::enable_unroll` |
| **FP Unroll Jam** | `f32`, `f64` | Reassociates IEEE-754 additions | **OFF** (Strict) | `LoopOptOptions::enable_fp_reassociation` |
| **SLP Vectorization** | `f32`, `f64`, `i32`, `i64` | SIMD isomorphic operation packing | ON | `LoopOptOptions::enable_slp` |
| **Loop Vectorization** | `f32`, `f64`, `i32`, `i64` | Counted loop vector unrolling | ON | `LoopOptOptions::enable_vectorize` |
| **F64 Demotion** | `f64` -> `f32` | Safe precision narrowing | ON | `LoopOptOptions::enable_f64_demote` |
| **FMA Optimization** | `f32`, `f64`, vectors | Contracts mul+add to hardware FMA | OFF | `LoopOptOptions::enable_fma` |
| **SROA** | Structs, aggregates | Scalarizes memory into SSA registers | OFF | `LoopOptOptions::enable_sroa` |
| **Range Analysis & BCE** | `i32`, `i64` | Eliminates redundant bounds checks | OFF | `LoopOptOptions::enable_bce` |
| **PEA & Alloc Sinking** | `gcref`, heap objects | Sinks allocations to cold exit paths | OFF | `LoopOptOptions::enable_allocation_sinking` |
| **Loop Unswitch** | Loops | Duplicates loop around invariant cond | OFF | `LoopOptOptions::enable_loop_unswitch` |
| **Jump Threading** | CFG branches | Eliminates redundant branch paths | OFF | `LoopOptOptions::enable_jump_threading` |
| **Loop Tiling & Cache** | Nested loops | Blocks loop nests for cache locality | OFF | `LoopOptOptions::enable_loop_tile` |
| **Loop Fusion / Fission** | Loops | Merges or splits loop iterations | OFF | `LoopOptOptions::enable_loop_fusion` / `distribution` |
| **Array Contraction** | Memory buffers | Eliminates intermediate array buffers | OFF | `LoopOptOptions::enable_array_contraction` |
| **Auto-Parallelization** | Counted loops | Multithreaded chunk dispatch | OFF | `LoopOptOptions::enable_parallel_loops` |
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
