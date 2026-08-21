# brass

A fast code-generation backend library for GC'd dynamic languages.

Standalone C++20 library, CMake, zero LLVM dependency.

## Key Features

1. **Precise Moving GC with Values in Registers**: First-class `gcref` tracking in SSA MIR and register allocator, generating compact stack maps and runtime stack walking for moving garbage collectors.
2. **Patchable Code Sites**: Thread-safe dynamic patching for Inline Caches (ICs) and call sites on x64.
3. **Speculation & Deoptimization**: Native `guard` and side exits with state maps and interior resume tables for generic twin fallbacks.
4. **Fast Single-Pass & Linear-Scan Codegen**: High compile throughput targeting sub-2-second full module compilation.
5. **Multi-Format Output**: COFF (Win64) and ELF64 (Linux/SysV) relocatable object files with unwind information (`.pdata`/`.xdata` SEH and SysV CFI).
6. **Differential Verification Oracle**: Built-in reference MIR interpreter and mini-Cheney moving collector harness for differential testing and generative fuzzing.

## Building & Testing

Requires CMake (>= 3.20), Ninja, and a C++20 compiler (GCC 12+, Clang 15+, or MSVC 2022).

```bash
# Configure and build
cmake -B build -G Ninja
cmake --build build

# Run unit tests
ctest --test-dir build --output-on-failure
```

## Documentation

- [Roadmap & Milestone Architecture](docs/ROADMAP.md)
- [MIR Reference Specification](docs/mir_reference.md)
- [GC Contract & Moving Cheney Collector](docs/gc_contract.md)
- [Speculation, Guards, and Deoptimization](docs/speculation_and_deopt.md)
- [Patching Protocol & Concurrency Rules](docs/patching_protocol.md)
- [Consumer's Guide to Lowering](docs/lowering_guide.md)
