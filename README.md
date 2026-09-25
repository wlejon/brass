# brass

[![CI](https://github.com/wlejon/brass/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/ci.yml)
[![Nightly](https://github.com/wlejon/brass/actions/workflows/nightly.yml/badge.svg)](https://github.com/wlejon/brass/releases/tag/nightly)
[![CodeQL](https://github.com/wlejon/brass/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brass/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A fast, optimizing code-generation backend library and runtime for garbage-collected dynamic languages.

Standalone C++20 library with CMake.

## Key Features

1. **Precise GC references in registers**: `gcref` is a MIR type. References live in registers between GC points; the linear scan allocator spills those live across a call or safepoint to frame slots and records them in binary stack maps (`BSCM`), so a moving collector finds and rewrites them without a shadow stack. `tagged` values (NaN-boxed words that may or may not hold a reference) are tracked the same way at every tier, and the collector keeps their tag when it moves the object.
2. **One garbage-collected heap**: `gc::Heap` is generational: a copying young generation over a non-moving mark-region (Immix-style) old generation and a large-object space, inside one reserved address range per heap, with a card-marking write barrier. Object layouts and tracing are host-defined; weak references, ephemerons, finalizers and post-collection hooks are built in; each thread has its own heap. It has a verification mode and stress modes. See [docs/gc_contract.md](docs/gc_contract.md).
3. **Runtime code patching**: in-place patching of constants, direct calls and near branches on x64 and AArch64 for inline caches and call-site rebinding, with cache-line alignment rules that keep each patch a single atomic write (see [docs/patching_protocol.md](docs/patching_protocol.md)).
4. **Speculation and deoptimization**: `guard` instructions with out-of-line exit stubs, deopt state maps and resume tables that continue a failed guard's call in the interpreter.
5. **Optimization pipeline**: GVN and GVN-PRE, SCCP, SROA, partial escape analysis and allocation sinking, range analysis and bounds-check elimination, inlining, loop unswitching, unrolling, tiling, fusion, distribution and array contraction, write-barrier elimination.
6. **Vectorization**: SLP and loop vectorization to 128-bit (`v128`) and 256-bit (`v256`) vector types, with FMA contraction.
7. **Loop dependence and parallelization**: affine dependence analysis (distance and direction vectors, alias analysis) and parallel execution of independent loop iterations through the parallel runtime.
8. **Coroutines and exceptions**: exception handling (`invoke`, `landing_pad`, `throw`, `resume`) and coroutine lowering (`coro_create`, `coro_suspend`, `coro_resume`).
9. **Object files and linkers**: COFF (with Win64 `.pdata`/`.xdata`), ELF64 (with `.eh_frame`) and Mach-O relocatable objects, emitted deterministically (a test checks byte-identical output across runs), and built-in PE DLL, ELF `.so` and Mach-O `.dylib` linkers.
10. **Tiers and oracle**: a reference MIR interpreter, a register-bytecode interpreter, a baseline JIT and an optimizing JIT, with a differential fuzzer that compares them. A program's fast interpreter tiers a hot loop up in place through on-stack replacement: it requests an optimized OSR entry function for the loop header and transfers the live values into it once the code is installed.
11. **PTX backend**: `PtxTarget` lowers MIR to PTX through a typed PTX IR with cleanup passes and a verifier. GPU intrinsics (thread indices, shuffles, barriers, `.shared` arrays, approximate math, f16, atomics) are MIR builtins with `KernelBuilder` helpers. A dynamically loaded CUDA driver runtime (no link-time CUDA dependency) JIT-compiles the PTX and launches it; the tests validate kernels with `ptxas` and against host references when a GPU is present.

## Performance Tracking

Benchmarks live in `tests/benchmarks` and run under the `perf` ctest label, which also checks them against the recorded ratchet in `bench/ratchet.json`; results depend on the machine. `brass_gc_pause_bench` reports the heap's minor and full collection pauses against the size of the live old generation.

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

### Linux and AArch64 execution

The AArch64 JIT runs natively on macOS arm64 (Apple Silicon): CI and the
nightly build run the whole correctness suite on a `macos-15` arm64 runner,
and the differential fuzzer runs there as on x64
(`brass-fuzz --pipeline=all --seed=1 --iterations=1000`). Every test that
needs a native tier runs on both architectures, apart from the few that
exercise Windows x64 unwind data or need frame pointers only Apple's ABI
guarantees; each says why.

On Linux, `scripts/linux-tests.sh` builds brass and runs the correctness suite
and the differential fuzzer: natively on x86_64, and for aarch64 under
qemu-user, so the AArch64 JIT's code is executed and checked against the
interpreter on AArch64 Linux as well. It needs no root: the first run
fetches qemu-user-static and the aarch64 cross toolchain with
`apt-get download` into `~/.cache/brass-aarch64` (Debian bookworm).

```bash
# From Windows, with WSL Debian installed; from Linux, drop "wsl -e"
wsl -e bash scripts/linux-tests.sh            # setup, aarch64, then x86_64
wsl -e bash scripts/linux-tests.sh a64        # aarch64 only: build, ctest, fuzz
wsl -e bash scripts/linux-tests.sh fuzz-a64   # one step; see the script header
wsl -e env FUZZ_SEEDS=4000 bash scripts/linux-tests.sh fuzz-a64
```

The aarch64 tree cross-builds with `cmake/toolchains/aarch64-linux-gnu.cmake`,
which sets qemu as `CMAKE_CROSSCOMPILING_EMULATOR`, so test discovery and
`ctest` run the aarch64 test binary transparently. A fuzz failure's
reproducer replays with `brass-fuzz --repro=<file>`; `--dump-opt=<out>` writes
the module after the pipeline, and `--unopt-only` then replays (or, with
`--minimize=`, reduces) it comparing only the unoptimized JIT against the
interpreter, so a code generation bug is chased without any pass running.
Minimized reproducers can read memory the removed code used to initialize;
narrowing by changing which value the original returns avoids that.

GPU tests validate every emitted kernel with `ptxas` and execute it on a real
device when a CUDA driver is present. They are skipped automatically otherwise,
and the library itself has no link-time CUDA dependency.

## Documentation

- [MIR Reference Specification](docs/mir_reference.md)
- [Optimization Passes & Semantics Specification](docs/semantics.md)
- [Garbage Collection: Design and Contract](docs/gc_contract.md)
- [Speculation, Guards, Deoptimization & OSR](docs/speculation_and_deopt.md)
- [Patching Protocol & Concurrency Rules](docs/patching_protocol.md)
- [Consumer's Guide to Lowering](docs/lowering_guide.md)
- [libbrass Embedding Guide (Public C-ABI)](docs/embedding_guide.md)
- [Host Engine & Heap Embedding Guide](docs/embedding.md)
- [PTX Backend Design](docs/ptx_backend_design.md)
- [Writing PTX Kernels with KernelBuilder](docs/ptx_kernel_authoring.md)
