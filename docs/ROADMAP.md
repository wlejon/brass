# Brass Architecture and Subagent Roadmap

`brass` is a high-performance, standalone code-generation backend library for garbage-collected dynamic languages, written in C++20 with CMake and zero LLVM dependencies.

---

## High-Level Architecture

```
                                  +---------------------------------------+
                                  | Consumer (e.g. Bronze JS AOT Compiler)|
                                  +---------------------------------------+
                                                     |
                                            MIR Builder API / Text
                                                     v
                                      +-------------------------------+
                                      |     MIR Module & Verifier     |
                                      +-------------------------------+
                                            /                   \
                                           /                     \
                                          v                       v
                      +-----------------------------+   +-----------------------------+
                      |   Reference Interpreter     |   |      x64 Codegen Pipeline   |
                      |  + Mini-Cheney Moving GC    |   |                             |
                      +-----------------------------+   |  1. ISEL (MIR -> LIR)       |
                                     ^                  |  2. Linear Scan RegAlloc    |
                                     | (Differential)   |  3. Stack Maps & Frame Gen  |
                                     v                  |  4. Peephole & Encoding     |
                      +-----------------------------+   +-----------------------------+
                      |     Differential Testing    |                 |
                      |   Interpreter vs Compiled   |<----------------+ (COFF / ELF Objects)
                      +-----------------------------+                 |
                                                                      v
                                                        +-----------------------------+
                                                        | Relocatable Object Output   |
                                                        | (.obj / .o + Unwind + Maps) |
                                                        +-----------------------------+
```

---

## Milestones & Sequential Subagent Execution Plan

### Chunk 1: Foundation & MIR Core (M1.1)
- **Scope**: Core utilities (`Arena`, `StringPool`, `SmallBitSet`, `DiagnosticReporter`) + MIR Types (`i32`, `i64`, `f64`, `ptr`, `gcref`, `void`), SSA Block Parameters, Instructions, Blocks, Functions, Modules, Builder API, and Verifier.
- **Lines of Code**: ~3,500 LOC.
- **Validation**: Verifier unit tests checking malformed IR rejection, dominance, block parameter type and count checks.

### Chunk 2: MIR Canonical Text Format (M1.2)
- **Scope**: Deterministic byte-stable MIR Printer, fast recursive-descent Parser, `print(parse(x)) == x` roundtrip test suite, `brass-opt` CLI tool.
- **Lines of Code**: ~3,000 LOC.
- **Validation**: Parser round-trip tests and golden text test suite.

### Chunk 3: Reference MIR Interpreter & Mini-Cheney Moving Collector (M1.3 - The Oracle)
- **Scope**: Reference Interpreter executing all MIR operations; Mini-Cheney moving semispace GC with memory poisoning (`0xDEADBEEF`) and root rewriting.
- **Lines of Code**: ~4,000 LOC.
- **Validation**: GC stress tests, poison validation, interpreter test suite.

### Chunk 4: Target Abstraction & x64 Instruction Encoder (M2.1)
- **Scope**: Target ABI/Calling convention models (Win64 & SysV), x64 register model, complete byte-exact instruction encoder.
- **Lines of Code**: ~3,500 LOC.
- **Validation**: Byte-exact golden encoder test corpus covering all instructions and addressing modes.

### Chunk 5: Lowering Pipeline: ISEL, Linear Scan RegAlloc, Frame Layout (M2.2)
- **Scope**: Low-level IR (LIR), Instruction selection, Live range analysis, Linear Scan RegAlloc with live-range splitting, Calling convention lowering, Frame layout with prologue/epilogue.
- **Lines of Code**: ~4,000 LOC.
- **Validation**: Unit tests for ISEL patterns, live ranges, spill/split correctness, and regalloc stress tests.

### Chunk 6: Binary Emission: COFF & ELF64 Relocatable Objects + Unwind Info (M2.3)
- **Scope**: COFF emitter with Win64 SEH `.pdata`/`.xdata`, ELF64 emitter with SysV CFI `.eh_frame`, in-memory execution harness, and differential runner (Interpreter vs Native).
- **Lines of Code**: ~3,500 LOC.
- **Validation**: Linking/executing emitted objects, verifying unwinding in crash/debugger tests, differential execution tests passing.

### Chunk 7: Precise Moving GC Stack Maps & Runtime Stack Walker (M3)
- **Scope**: Safepoint stack map emitter, compact stack map binary metadata, runtime stack walker (`brass_stack_walk`), root iterator.
- **Lines of Code**: ~3,500 LOC.
- **Validation**: Running native compiled code with mini-Cheney moving collector under collection-at-every-safepoint stress mode with heap poisoning.

### Chunk 8: Speculation: Guards, Exits, State Maps, Resume Tables & Patchable Sites (M4)
- **Scope**: `guard` / `exit` lowering to out-of-line stubs, `brass_deopt_frame` state map materialization, Interior Resume Points & Resume Tables, `patchable_const` and `patchable_call` with thread-safe x64 patching protocol.
- **Lines of Code**: ~4,000 LOC.
- **Validation**: Multithreaded concurrent patching tests, deoptimization fuzz tests, differential tests.

### Chunk 9: Optimization, Benchmarks, Determinism Ratchet & Documentation (M5)
- **Scope**: LIR peephole optimizations, typed benchmark suite (numeric, matrix, pointer-chasing, GC vs shadow-stack), determinism ratchet in CI, complete documentation.
- **Lines of Code**: ~3,500 LOC.
- **Validation**: Performance bar (<= 1.3x clang -O2, >= 1.5x shadow stack), byte-identical determinism test across machines/runs.

---

## Phase II: Advanced Optimization Pipeline & Standalone Toolchain (Chunks 1–12)

- **Chunk 1: MIR Function Inlining & IPO**: Call graph analysis, heuristic inlining, devirtualization.
- **Chunk 2: 128-Bit SIMD Vector Types**: `v128` vector types, SSE/AVX lowering.
- **Chunk 3: Loop Vectorizer & SLP Packetizer**: Automatic SIMD vectorization for 3D/graphics kernels.
- **Chunk 4: Escape Analysis & SROA**: Scalar replacement of aggregates, stack allocation of non-escaping frames.
- **Chunk 5: GVN, Memory SSA & Alias Analysis**: Redundant load elimination, dead store elimination.
- **Chunk 6: Standalone AOT Linker**: PE DLL and ELF Shared Object (`.so`) writers without external linkers.
- **Chunk 7: Loop Polyhedral Tiling**: Cache blocking for multi-dimensional nested loops (`matmul`).
- **Chunk 8: SCCP, Guard Elimination & CFG Simplification**: Sparse conditional constant propagation, pruning redundant speculative guards.
- **Chunk 9: Loop Unswitching, Jump Threading & Trace Block Layout**: Fall-through block layout and branch avoidance.
- **Chunk 10: Profile-Guided Optimization (PGO)**: Minimal edge instrumentation, `.bprof` serialization, Kirchhoff's law edge flow solving.
- **Chunk 11: Machine Instruction Scheduling & Software Pipelining**: Dependency DAG scheduling, modulo loop pipelining.
- **Chunk 12: Source Maps & Symbolication**: V3 source map emission, `.brass_dbg` section, stack trace symbolicator.

---

## Phase III: Advanced Dynamic Runtime & System Capabilities (Chunks 13–16)

- **Chunk 13: Polymorphic Inline Caches (PICs), Shape Transitions, and Property Lowering [COMPLETED]**:
  - `Shape`, `ShapeRegistry`, `DynamicObject` with moving Cheney GC support.
  - Monomorphic, Polymorphic (up to 4 shapes), and Megamorphic ICs with atomic x64 dynamic patching.
  - Lowering `prop.get`, `prop.set`, `elem.get`, `elem.set`, `method.def` in Bronze IL translator.
  - 24-program Bronze corpus verification.
- **Chunk 14: Zero-Cost Hardware-Assisted Exception Handling [COMPLETED]**:
  - Win64 SEH `.pdata`/`.xdata` personality routines and Linux SysV `.gcc_except_table` / DWARF LSDA action tables.
  - MIR opcodes: `throw`, `invoke`, `landing_pad`, `resume`.
  - Lowering Bronze IL `handler`, `throw`, `exc.take`.
  - 26-program Bronze corpus verification.
- **Chunk 15: Stackless Coroutines & Resumable Frames (Async/Await & Generators) [COMPLETED]**:
  - Coroutine state machine transformation pass (`CoroTransformPass`).
  - GC-tracked resumable frames (`BrassCoroFrame`) and Interior Resume Table integration.
  - Lowering Bronze IL `create.async_machine`, `async.start`, `async.await`, `iter.open`, `iter.step`, `yield`.
  - 28-program Bronze corpus verification.
- **Chunk 16: Partial Escape Analysis (PEA) & Allocation Sinking [COMPLETED]**:
  - Path-sensitive escape analysis across control-flow graphs and dominator trees.
  - Sinking allocations from loop headers down into cold / bailout exit paths.
  - Scalarization of boxed numbers / objects in non-escaping hot loop paths.

---

## Phase IV: Enterprise JIT/AOT & High-Throughput Execution (Chunks 17–20)

- **Chunk 17: Generational Garbage Collection, Card-Table Remembered Sets, and JIT Write Barrier Elimination (WBE) [COMPLETED]**:
  - Contiguous 2-generation heap: bump-pointer Nursery (Eden) space and Tenured (Old) space.
  - 512-byte card table (`CARD_SHIFT = 9`) with 64-bit word skipping for dirty card scanning.
  - Minor GC scavenge scanning only root sets and dirty cards in Tenured space with age promotion threshold.
  - Major GC fallback evacuating all reachable objects when tenured space reaches capacity.
  - MIR `write_barrier %obj, %val` opcode with builder, verifier, printer, parser, and interpreter card-marking support.
  - Write Barrier Elimination (WBE) pass: provenance analysis removing barriers on young allocations, non-reference values, and dominating writes.
  - x64 JIT lowering to `brass_gc_write_barrier` runtime hook.
  - 29-program Bronze corpus verification (`29_generational_churn`).
- **Chunk 18: On-Stack Replacement (OSR) & Multi-Tier Execution Pipeline [COMPLETED]**:
  - `TierLevel` state machine (`Tier0_Interpreter` -> `Tier1_Baseline` -> `Tier2_Optimized`) with configurable invocation and loop backedge thresholds.
  - OSR target analysis and live-in SSA parameter mapping for candidate loop headers (`Block::is_osr_entry()`, `Opcode::osr_entry`).
  - Native secondary function prologues (`osr_entry_offset`) unpacking `OsrMigrationFrame` into physical registers and stack spill slots.
  - Bi-directional runtime `OsrCoordinator` migrating running loops from interpreter to native JIT and deoptimizing back upon speculative guard failure.
  - 30-program Bronze corpus verification (`30_osr_hot_loop`).
- **Chunk 19: Global Value Numbering with Partial Redundancy Elimination (GVN-PRE) & Critical Edge Splitting [COMPLETED]**:
  - Synthetic forwarding block injection along critical edges, redirecting branch targets and forwarding block arguments.
  - SSA-based GVN-PRE algorithm: Anticipation (DownSafe) and Availability (CanBeAvail) dataflow analyses.
  - Partial redundancy elimination at join blocks, injecting block parameters (phi nodes) and hoisting computations to missing predecessors.
  - Generalization of Loop Invariant Code Motion (LICM) and Memory Load PRE with alias protection.
  - 31-program Bronze corpus verification (`31_gvn_pre_diamonds`).
- **Chunk 20: Advanced Loop Transformations: Loop Fusion, Distribution, and Array Contraction [COMPLETED]**:
  - Loop fusion (jamming): Congruent iteration domain matching, distance vector hazard checks, CFG body merging, and shared induction variables.
  - Loop distribution (fission): Multi-loop partitioning separating vectorizable and scalar/side-effecting operations to unlock downstream SIMD auto-vectorization.
  - Array contraction: Intermediate buffer elimination forwarding stores directly to loads within loop iterations, eliminating heap allocations (0 bytes allocated in GC semispace).
  - 32-program Bronze corpus verification (`32_loop_fusion_contraction`).

---

## Phase V: Scalable Parallelism, Hardware Acceleration & Throughput (Chunks 21–23)

- **Chunk 21: AVX2 256-Bit Vector Extension & FMA3 Instruction Set [COMPLETED]**:
  - 256-bit vector types: `F32x8`, `F64x4`, `I32x8`, `I64x4` (32 bytes).
  - FMA primitives: `vfma %a, %b, %c`, scalar `fma_f32`, `fma_f64`.
  - 2-byte (`0xC5`) and 3-byte (`0xC4`) VEX prefix emitter in [`x64_encoder_vex.cpp`](file:///D:/projects/brass/src/target/x64/x64_encoder_vex.cpp) with non-destructive 3-operand encodings.
  - AVX2 arithmetic, logical, broadcast, and FMA3 instruction set lowering.
  - FMA pattern matching pass (`mul + add` -> `fma`) and 256-bit loop vectorizer expansion.
  - 33-program Bronze corpus verification (`33_avx2_fma_matmul`).
- **Chunk 22: Concurrent Background JIT Compiler Worker Threads [COMPLETED]**:
  - Thread-safe priority task queue with worker thread pool managing asynchronous background compilations.
  - Request deduplication preventing duplicate compilation of active or queued hot functions.
  - Atomic code installation via `FunctionHandle` and `CodeInstaller`, publishing native entry points with release/acquire memory barriers without pausing mutator execution.
  - 34-program Bronze corpus verification (`34_background_tiering`).
- **Chunk 23: Polyhedral Loop Dependence & Auto-Parallelization**:
  - Multi-threaded loop execution for large-trip count kernels.
  - Task scheduling runtime with work-stealing thread pool.

---

## House Rules & Code Conventions
1. **C++20**: Zero LLVM dependencies.
2. **Compiler Compatibility**: MSVC, Clang, and GCC 12 clean (no `= {}` default args which GCC 12 rejects).
3. **File Size Limit**: Keep files under 1,000 lines. Files approaching or exceeding 2,000 lines must be strictly decomposed into modular headers/sources.
4. **Validation**: Every subagent chunk must land with complete unit, differential, or golden tests.

