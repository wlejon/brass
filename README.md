# brass

[![CI](https://github.com/wlejon/brass/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/ci.yml)
[![Nightly](https://github.com/wlejon/brass/actions/workflows/nightly.yml/badge.svg)](https://github.com/wlejon/brass/releases/tag/nightly)
[![CodeQL](https://github.com/wlejon/brass/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A fast, optimizing code-generation backend library and runtime for garbage-collected dynamic languages.

Standalone C++20 library with CMake.

## Key Features

1. **Precise Moving GC with Values in Registers**: First-class `gcref` tracking in SSA MIR and register allocator, generating compact binary stack maps and runtime stack walking for moving garbage collectors (achieving **3.9x+ speedup** over shadow stacks).
2. **Generational GC, Card Table & TLAB**: Two-generation GC with nursery, survivor, and tenured spaces, 512-byte card table tracking for old-to-young pointers, and lock-free thread-local allocation buffers (`TLAB`).
3. **Thread-Safe Runtime Patching**: Dynamic patching for Inline Caches (ICs) and call sites on x64 without stopping mutator threads, guaranteed safe via cache-line alignment and hardware memory snooping.
4. **Speculation, Deoptimization & OSR**: Native `guard` checks with out-of-line exit stubs, captured `DeoptFrame` state maps, interior resume tables, and On-Stack Replacement (OSR) migrating interpreter loops into running JIT loops.
5. **Comprehensive Optimization Pipeline**: Global Value Numbering (GVN & GVN-PRE), Sparse Conditional Constant Propagation (SCCP), SROA, Partial Escape Analysis & Allocation Sinking (PEA), Value Range Analysis & Bounds Check Elimination (BCE), loop unswitching, tiling, fusion, and array contraction.
6. **SIMD & SLP Vectorization**: Automatic straight-line (SLP) and loop vectorization generating 128-bit SSE (`v128`) and 256-bit AVX2 (`v256`) instructions with hardware FMA contraction.
7. **Polyhedral Auto-Parallelization**: Polyhedral loop dependence analysis with multithreaded chunk execution via an integrated parallel thread pool.
8. **Coroutines & Exception Handling**: Native zero-cost exception handling (`invoke`, `landing_pad`, `throw`, `resume`) and first-class coroutine state-machine lowering (`coro_create`, `coro_suspend`, `coro_resume`).
9. **Byte-Level Determinism Ratchet**: 100% byte-for-byte deterministic emission of COFF (Win64), ELF64 (Linux/SysV), and Mach-O (macOS) relocatable object files across runs.
10. **Multi-Format Output & Standalone Linking**: Full Win64 SEH (`.pdata`/`.xdata`), Linux SysV CFI (`.eh_frame`), and built-in PE DLL, ELF `.so`, and Mach-O `.dylib` standalone linkers that produce shared libraries without external linkers.
11. **Differential Verification Oracle**: Built-in reference MIR interpreter, differential fuzzing harness, and automated Bronze IL translation suite.

## Performance Bars

- **Native Throughput**: Matches or beats compiled C++ baseline code across numeric loops, prime sieve, Collatz, matrix multiplication, and linked list traversals (within <= 1.3x envelope of Clang -O2).
- **GC Efficiency**: 3.9x+ faster than explicit shadow-stack tracking by keeping managed pointers directly in native CPU registers with zero runtime GC overhead on fast paths.
- **Determinism**: 100% byte-identical object files verified under scrambled heap allocations.

## Building & Testing

Requires CMake (>= 3.20), Ninja, and a C++20 compiler (GCC 12+, Clang 15+, or MSVC 2022).

```bash
# Configure and build
cmake -B build -G Ninja
cmake --build build

# Run unit tests & determinism ratchet
ctest --test-dir build --output-on-failure

# Run performance benchmark suite
./build/tests/brass_benchmarks
```

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
