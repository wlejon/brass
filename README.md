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
12. **SPIR-V backend**: `SpirvTarget` lowers the same MIR kernels to SPIR-V 1.5 compute shaders for Vulkan 1.2+: a typed SPIR-V module model with a binary writer and verifier, a structurizer that recovers structured control flow from the MIR CFG, kernel pointers as buffer device addresses with parameters in a push-constant block, and the PTX intrinsic names lowered to subgroup, barrier, atomic and GLSL.std.450 operations. The Khronos SPIR-V headers are vendored (`third_party/spirv-headers`), so the target is built everywhere. A minimal dynamically loaded Vulkan runtime (`gpu/vulkan_driver`, no link-time Vulkan dependency) loads the modules as compute pipelines and launches them; the fused ML kernels compile through `MlFusionCompiler::compile_spirv`. The tests run every module through `spirv-val` when it is installed and execute the ISel features and the fused kernels on a Vulkan device when one is present.

## Performance Tracking

Benchmarks live in `tests/benchmarks` and run under the `perf` ctest label, which also checks them against the recorded ratchet in `bench/ratchet.json`; results depend on the machine. The ratios compare brass with native code built by the host compiler, so `bench/ratchet.<platform>.json` (`linux`, `windows`, `macos`), when present, overrides the keys it names on that platform. `brass_gc_pause_bench` reports the heap's minor and full collection pauses against the size of the live old generation.

The ratchet has one notion of pass: each key's number (a median over repetitions) may be worse than its golden by at most that key's margin, 10% unless `RatchetPolicy` in `tests/benchmarks/bench_ratchet.hpp` widens it for a key that is noisier between quiet runs. There are no aspiration targets; a design claim such as the stack-map GC model beating a shadow stack (`gc_model_speedup`, higher is better) or compile speed is held by its golden like any other key. The `interp_*` keys are the fast interpreter's time over the baseline JIT's on the same MIR, with the interpreter on a program of its own so nothing tiers up (native C++ as the denominator moved 10-13% when a relink shifted its alignment; re-baseline these keys when the baseline JIT's codegen changes). The suite prints the machine's CPU load before and during the run and flags a busy machine, whose numbers should be neither trusted nor recorded; on Windows it pins the benchmark thread to one logical CPU (`--cpu`, `BRASS_BENCH_CPU`; `-1` turns it off) so a run does not move between unlike cores.

```bash
build_vs/tests/Release/brass_benchmarks.exe --check-ratchet    # the gate: exit 1 on a regression
build_vs/tests/Release/brass_benchmarks.exe --update-ratchet   # record this run's medians as goldens
```

`--update-ratchet` writes every measured key to the platform overlay when one exists (so re-baselining one platform never rewrites another's goldens), otherwise to `bench/ratchet.json`. Re-baseline only on a quiet machine, from a run that already looks like the runs before it.

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
device when a CUDA driver is present; the SPIR-V tests validate with
`spirv-val` and execute on a Vulkan device when one is present
(`BRASS_VULKAN_DEVICE=<index>` picks one). They are skipped automatically
otherwise, and the library itself has no link-time CUDA or Vulkan dependency.

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
- [SPIR-V Backend Design](docs/spirv_backend_design.md)
