# brass

[![CI](https://github.com/wlejon/brass/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/ci.yml)
[![Nightly](https://github.com/wlejon/brass/actions/workflows/nightly.yml/badge.svg)](https://github.com/wlejon/brass/releases/tag/nightly)
[![CodeQL](https://github.com/wlejon/brass/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A fast, optimizing code-generation backend library and runtime for garbage-collected dynamic languages.

Standalone C++20 library with CMake.

## Key Features

1. **Precise Moving GC with Values in Registers**: First-class `gcref` tracking in SSA MIR with a hybrid register & stack map design: `gcref` pointers reside directly in native CPU registers during intra-procedural computation (achieving zero-overhead fast paths), and the linear scan allocator precisely spills them to stack slots across calls and safepoint boundaries, generating compact binary stack maps (`BSCM`) for moving GC root relocation without shadow stacks.
2. **Generational GC, Card Table & TLAB**: Two-generation GC with nursery, survivor, and tenured spaces, 512-byte card table tracking for old-to-young pointers, and lock-free thread-local allocation buffers (`TLAB`).
3. **Thread-Safe Runtime Dynamic Patching**: Dynamic patching for Inline Caches (ICs), call sites (`0xE8`), and near jumps (`0xE9`) on x64/AArch64 without stopping mutator threads, executed via thread-safe atomic 32-bit displacement patching with verified cache-line boundary alignment (`is_cache_line_safe`) and processor memory bus snooping.
4. **Speculation, Deoptimization & OSR**: Native `guard` checks with out-of-line exit stubs, captured `DeoptFrame` state maps, interior resume tables, and On-Stack Replacement (OSR) migrating interpreter loops into running JIT loops.
5. **Comprehensive Optimization Pipeline**: Global Value Numbering (GVN & GVN-PRE), Sparse Conditional Constant Propagation (SCCP), SROA, Partial Escape Analysis & Allocation Sinking (PEA), Value Range Analysis & Bounds Check Elimination (BCE), loop unswitching, tiling, fusion, and array contraction.
6. **SIMD & SLP Vectorization**: Automatic straight-line (SLP) and loop vectorization generating 128-bit SSE (`v128`) and 256-bit AVX2 (`v256`) instructions with hardware FMA contraction.
7. **Loop Dependence & Auto-Parallelization**: Affine loop dependence analysis (distance and direction vectors with alias analysis disambiguation) with multithreaded chunk execution via the parallel runtime.
8. **Coroutines & Exception Handling**: Native zero-cost exception handling (`invoke`, `landing_pad`, `throw`, `resume`) and first-class coroutine state-machine lowering (`coro_create`, `coro_suspend`, `coro_resume`).
9. **Byte-Level Determinism Ratchet**: 100% byte-for-byte deterministic emission of COFF (Win64), ELF64 (Linux/SysV), and Mach-O (macOS) relocatable object files across runs.
10. **Multi-Format Output & Standalone Linking**: Full Win64 SEH (`.pdata`/`.xdata`), Linux SysV CFI (`.eh_frame`), and built-in PE DLL, ELF `.so`, and Mach-O `.dylib` standalone linkers that produce shared libraries without external linkers.
11. **Differential Verification Oracle**: Built-in reference MIR interpreter, differential fuzzing harness, and automated Bronze IL translation suite.
12. **NVIDIA PTX & GPU Execution**: `PtxTarget` lowers MIR through `PtxISel` into a typed PTX IR, runs cleanup passes (copy propagation, dead code, branch simplification) and a structural verifier, and prints the text. GPU intrinsics (thread indices, shuffles, barriers, `.shared` arrays, approx math, f16, narrow loads, atomics) are MIR builtins with `KernelBuilder` helpers, and the fused ML kernels (RMSNorm, LayerNorm, SwiGLU, AdaLN, GEMV, Q8_0/Q4_K GEMV) are written as MIR with them -- no PTX strings. A dynamically-loaded CUDA driver runtime (Windows and Linux, no link-time CUDA dependency) JIT-compiles the PTX and launches it on device; the tests validate every kernel with `ptxas` and against host references on a real GPU.

## Performance Bars

- **Native Throughput**: Matches or beats compiled C++ baseline code across numeric loops, prime sieve, Collatz, matrix multiplication, and linked list traversals (within <= 1.8x envelope of Clang -O2 on naive scalar loops, beating Clang -O2 down to 0.28x-0.70x on vectorized SIMD kernels).
- **GC Efficiency**: Measured >= 1.25x to 3.9x faster than explicit shadow-stack tracking across GC reference benchmarks by keeping managed pointers directly in native CPU registers with zero runtime GC overhead on fast paths, spilling to stack slots only across calls and safepoints with compact binary stack map tracking.
- **Determinism**: 100% byte-identical object files verified under scrambled heap allocations.
- **Compile Throughput**: Sub-second compilation of 6,000+ functions (> 7,000 functions/sec, > 3.2 MB/sec) directly into executable native machine code.

## Building & Testing

Requires CMake (>= 3.20), Ninja, and a C++20 compiler (GCC 12+, Clang 15+, or MSVC 2022).

```bash
# Configure and build
cmake -B build -G Ninja
cmake --build build

# Run the correctness suite (unit + differential tests, one ctest per TEST_CASE)
ctest --test-dir build --output-on-failure -L correctness

# Run the performance targets and ratchet (machine-dependent)
ctest --test-dir build --output-on-failure -L perf
```

Each `TEST_CASE` runs in its own process, so a crash fails only that test.
To run tests in-process while iterating, call the binary directly:
`./build/tests/brass_unit_tests --filter=<substring>` (or `--exact=<name>`,
`--list`).

GPU tests validate every emitted kernel with `ptxas` and execute it on a real
device when a CUDA driver is present. They are skipped automatically otherwise,
and the library itself has no link-time CUDA dependency.

## Documentation

- [MIR Reference Specification](docs/mir_reference.md)
- [Optimization Passes & Semantics Specification](docs/semantics.md)
- [GC Contract & Moving Garbage Collectors](docs/gc_contract.md)
- [Speculation, Guards, Deoptimization & OSR](docs/speculation_and_deopt.md)
- [Patching Protocol & Concurrency Rules](docs/patching_protocol.md)
- [Consumer's Guide to Lowering](docs/lowering_guide.md)
- [libbrass Embedding Guide (Public C-ABI)](docs/embedding_guide.md)
- [Host Engine & Moving GC Embedding Guide](docs/embedding.md)
- [Bronze IL Translator & Runtime Integration](docs/il_translator.md)
- [PTX Backend Design](docs/ptx_backend_design.md)
- [Writing PTX Kernels with KernelBuilder](docs/ptx_kernel_authoring.md)
