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

## House Rules & Code Conventions
1. **C++20**: Zero LLVM dependencies.
2. **Compiler Compatibility**: MSVC, Clang, and GCC 12 clean (no `= {}` default args which GCC 12 rejects).
3. **File Size Limit**: Keep files under 1,000 lines. Files approaching or exceeding 2,000 lines must be strictly decomposed into modular headers/sources.
4. **Validation**: Every subagent chunk must land with complete unit, differential, or golden tests.
