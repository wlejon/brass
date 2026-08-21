# brass

A fast code-generation backend library for GC'd dynamic languages.

Standalone C++20 library, CMake, zero LLVM dependencies.

## Key Features

1. **Precise Moving GC with Values in Registers**: First-class `gcref` tracking in SSA MIR and register allocator, generating compact binary stack maps and runtime stack walking for moving garbage collectors (achieving **3.9x+ speedup** over shadow stacks).
2. **Patchable Code Sites**: Thread-safe dynamic patching for Inline Caches (ICs) and call sites on x64 without stopping mutator threads.
3. **Speculation & Deoptimization**: Native `guard` and out-of-line side exits with state maps and interior resume tables for generic twin fallbacks.
4. **LIR Peephole Optimizer & Linear Scan**: Post-regalloc peephole optimizations (redundant move elimination, load-after-store forwarding, dead move elimination, arithmetic zeroing, and branch folding).
5. **Byte-Level Determinism Ratchet**: 100% byte-for-byte deterministic emission of COFF (Win64) and ELF64 (Linux/SysV) relocatable object files across runs.
6. **Multi-Format Output & Unwind Info**: Full Win64 SEH (`.pdata`/`.xdata`) and Linux SysV CFI (`.eh_frame`) generation.
7. **Differential Verification Oracle**: Built-in reference MIR interpreter and mini-Cheney moving collector harness for differential testing.

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
./build/tests/brass_benchmarks.exe
```

## Documentation

- [Roadmap & Milestone Architecture](docs/ROADMAP.md)
- [MIR Reference Specification](docs/mir_reference.md)
- [GC Contract & Moving Cheney Collector](docs/gc_contract.md)
- [Speculation, Guards, and Deoptimization](docs/speculation_and_deopt.md)
- [Patching Protocol & Concurrency Rules](docs/patching_protocol.md)
- [Consumer's Guide to Lowering](docs/lowering_guide.md)
