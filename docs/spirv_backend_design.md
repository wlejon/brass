# SPIR-V Backend Design

Status: the target, its IR, verifier and tests (stage 1 of 3). This document
is the reference for the SPIR-V compute backend: the typed SPIR-V module
model, the writer and verifier, the structurizer, the instruction selector,
the memory model and the kernel ABI. Kernels are written exactly as for PTX
([ptx_kernel_authoring.md](ptx_kernel_authoring.md)); the same MIR built
with `KernelBuilder` lowers to PTX for NVIDIA and to SPIR-V for Vulkan. The
PTX backend this one mirrors is described in
[ptx_backend_design.md](ptx_backend_design.md).

Planned next: a Vulkan runtime (`gpu/vulkan_driver`) that loads and
dispatches these modules and runs the fused ML kernels on device (stage 2),
then brotensor's trace JIT emitting through this target (stage 3).

## Pipeline

```
MIR --structurize--> structured CFG --SpirvISel--> spirv::Module --spirv::verify--> spirv::write --> words
     (loops, merges,                  (typed SPIR-V     (structural,
      forwarding blocks)               as data)          no spirv-val)
```

`SpirvTarget` (`include/brass/target/spirv_target.hpp`, a 70-line facade):

- `emit_function(fn, opts)` -> `std::vector<uint32_t>`: one kernel, one module;
- `emit_module(mod, opts)`: every function of a MIR module as an entry
  point of one SPIR-V module;
- `compile(fn, opts)` -> `SpirvKernel`: the words plus what a runtime needs
  -- entry name, push-constant layout (`params`: type, offset, size per
  parameter), `push_constant_bytes`, `shared_bytes`, `local_size`, and the
  capability and extension names the module declares;
- `dump_function(fn, opts)`: the verified module as text (`%id = OpName
  %type operands`, enumerants by name), for tests and diagnostics, with no
  external tools.

`SpirvOptions`: `local_size_x/y/z` (default 256x1x1),
`local_size_spec_constants` (default on, see the ABI), `spirv_version`
(`0x00010500` = SPIR-V 1.5 / Vulkan 1.2, or 1.6 / Vulkan 1.3; earlier
versions throw `std::invalid_argument` because PhysicalStorageBuffer and
8/16-bit storage are core only from 1.5). There is no cleanup pass: the
Vulkan driver's compiler (ACO on RADV) does copy propagation, DCE and
constant folding, and SPIR-V's SSA form has no parallel copies to clean up.

A `spirv::verify` failure is a compiler bug and throws with every
diagnostic and the module dump attached; nothing that fails verification is
returned.

## spirv IR (`include/brass/target/spirv/spirv_ir.hpp`)

- Opcodes and enumerants are the Khronos headers' (`spv::Op`,
  `spv::Capability`, `GLSLstd450*`, from `<spirv/unified1/spirv.hpp>` and
  `GLSL.std.450.h`); nothing hand-types an opcode number. CMake finds the
  headers (`BRASS_SPIRV_HEADERS_DIR`; hints: `$VULKAN_SDK/include`, a
  `SPIRV-Headers` checkout beside brass, the system include path, any
  `CMAKE_PREFIX_PATH`). Without them the SPIR-V sources and tests are left
  out and `BRASS_WITH_SPIRV` is 0.
- `spirv::Inst { op, type, result, operands, origin }`; each `Operand` is
  tagged as an id or a literal word, so the verifier can check every id.
  Strings (entry-point names, `OpName`, extensions, the extended-instruction
  import) live in dedicated module fields and are encoded only by the writer.
- `spirv::Module`: id allocation, the capability/extension sets, entry
  points, execution modes, names, decorations, globals (types, constants,
  global variables, in definition order) and functions of blocks. Types and
  constants are deduplicated by `(opcode, type, operand words)` -- a float
  constant by its bit pattern, so `0.0` and `-0.0` stay distinct; structs
  (which get decorated) and spec constants are always fresh. Requesting a
  64-bit int or float type adds `Int64` / `Float64`. `cached(key)` holds
  per-module singletons (builtin variables, the workgroup-size spec
  constants).
- `spirv::write`: the binary in the logical layout order of SPIR-V 2.4
  (header with the id bound, capabilities, extensions, imports, memory
  model, entry points, execution modes, names, annotations, globals,
  functions).
- `spirv::dump` / `to_string` / `op_name`: the text form.

### spirv::verify (`src/target/spirv/spirv_verifier.cpp`)

Diagnostics carry the function, block label, instruction index and the
printed instruction (`format_diagnostics`). It checks:

- every result id is non-zero, below the bound and defined once; every id
  operand is defined; every result type is a type;
- every block is non-empty and ends in exactly one terminator; `OpPhi`s
  come first; the entry block is not a branch target; branch targets are
  blocks of the function;
- each `OpPhi` has exactly one `(value, parent)` pair per predecessor;
- `OpSelectionMerge`/`OpLoopMerge` immediately precede a matching branch,
  name blocks of the function, and no block is the merge of two headers;
- `OpLoad`/`OpStore` through a `PhysicalStorageBuffer` pointer carry an
  `Aligned` memory operand;
- 64-bit types have their capabilities;
- every entry point names a function and lists every global variable the
  function uses in its interface (required since SPIR-V 1.4).

It does not check structured control flow, dominance or the Vulkan
environment rules; the tests run `spirv-val` for those.

## Memory model: buffer device addresses

MIR pointers are untyped 64-bit byte addresses, offset with integer
arithmetic and read at whatever element type a `load` names. The backend
keeps exactly that: a MIR `ptr` is a `u64` **buffer device address**
(`VK_KHR_buffer_device_address`, core in Vulkan 1.2; addressing model
`PhysicalStorageBuffer64`, memory model `GLSL450`). Every access computes
`ptr + offset` as an integer and converts it with `OpConvertUToPtr` to a
`PhysicalStorageBuffer` pointer of the accessed type:

- scalar `load`/`store`: `Aligned` = the element size;
- `load_indexed`/`store_indexed`: `base + sext(index) * scale + offset`
  (a constant index folds into the displacement);
- `vload`/`vstore`: one 16-byte access per part (below), `Aligned 16`;
- narrow access (`ptx_load_u8/s8/u16/s16`, `ptx_store_u8/u16`): 8/16-bit
  pointers with `StorageBuffer8BitAccess` / `StorageBuffer16BitAccess`
  (core in Vulkan 1.2) and `OpUConvert`/`OpSConvert` to and from `i32`.

Why this and not one `StorageBuffer` descriptor per pointer parameter: with
descriptors every pointer would have to be traced back to a (binding, byte
offset) pair at compile time, which fails for pointers that come through a
block parameter, a `select` or memory, and a pointer read at two element
types would need one aliased runtime-array view per type plus `OpBitcast`.
With device addresses the ISel is one-to-one with the PTX ISel (`ptr +
offset` is an `OpIAdd`, a load is a load), one pointer can be read as `f32`
in one place and `i32x4` in another, and the kernel ABI has no descriptor
sets at all. The costs: the device needs `bufferDeviceAddress` (RADV has
it), buffers are created with `SHADER_DEVICE_ADDRESS` usage, and there is
no robust-buffer bounds checking -- the same contract as CUDA.

**Vectors.** A MIR vector is one SPIR-V vector per 16 bytes ("part"):
`f32x4`/`i32x4` one 4-lane vector, `f64x2`/`i64x2` one 2-lane vector, and
the 256-bit types two parts, since shader vectors stop at 4 components.
Every vector op applies per part with native SPIR-V vector arithmetic; this
mirrors the PTX backend's two `.v4`/`.v2` tuples for 256-bit types.

**Shared memory.** `ptx_shared_alloc_<T>(count)` declares one `Workgroup`
array `T[count]` per call (count a compile-time constant, as on PTX).
Workgroup memory has no numeric address, so the `ptr` result is tracked by
*provenance* in the ISel (`SharedRef`: array, constant byte offset, dynamic
byte offset): `add` with an integer accumulates the offset, and the
`ptx_shared_*` intrinsics turn `offset / sizeof(T)` into an
`OpAccessChain` index. A shared pointer used anywhere else -- a block
argument, `select`, a store's value, a plain `load` -- is a diagnostic
naming the value. Because the arrays are typed, an access must have the
array's element size (an `i32` access of an `f32` array is an `OpBitcast`;
an 8-byte access of a 4-byte array is a diagnostic, where PTX's untyped
`.shared` allowed it); byte offsets must be multiples of the element size.

## Kernel ABI

- **Parameters**: one push-constant block (`Block`-decorated struct) with
  every kernel parameter in declaration order -- `ptr` and `i64` as `u64`,
  `i32` as `u32`, `f32`, `f64` -- each at the next offset aligned to its own
  size. `SpirvKernel::params` has the offsets; `push_constant_bytes` is
  rounded up to 4. The largest fused kernel takes 60 bytes, under Vulkan's
  guaranteed 128 (RADV allows 256). No descriptor sets.
- **Workgroup size**: with `local_size_spec_constants` (default) three
  `OpSpecConstant`s with `SpecId` 0, 1, 2 (defaults `local_size_*`) form the
  module's one `BuiltIn WorkgroupSize` composite, so the runtime picks the
  block size at pipeline creation, as a PTX launch picks it at launch;
  `ptx_ntid_*` read those constants. Off, `OpExecutionMode LocalSize` is
  fixed and `ptx_ntid_*` are plain constants. `LocalSize` is emitted either
  way (it is the default the spec constants override).
- **Grid**: `vkCmdDispatch(grid_x, grid_y, grid_z)`; `ptx_ctaid_*` is
  `WorkgroupId`, `ptx_nctaid_*` `NumWorkgroups`.
- **Shared memory**: static (`shared_bytes` reports the total); nothing is
  passed at dispatch.
- **Entry point**: `GLCompute`, named after the MIR function; void, no
  parameters.

## Structurization (`src/target/spirv/spirv_structurize.cpp`)

SPIR-V requires structured control flow: every loop has a header with
`OpLoopMerge` naming one merge block and one continue target, every
conditional branch outside a loop header has an `OpSelectionMerge`, merge
blocks are dominated by their headers and belong to one header each, and a
construct is left only through its merge, a break to the innermost loop's
merge, a continue, or a return. MIR is an arbitrary CFG with block
arguments. The structurizer recovers the structure from dominators and
natural loops (option (a) of the plan: no region markers from
`KernelBuilder`, so hand-built MIR and the output of MIR passes work too).

It builds a graph with one node per reachable MIR block plus synthetic
nodes, then:

1. **Build.** A prologue node comes first (the push-constant and builtin
   loads; the SPIR-V entry block may not be a branch target, so a MIR entry
   block with a back edge stays a normal block). A `br_if` with both edges
   to one block gets a forwarding node on its false edge, so each edge has
   its own phi parent. Terminators other than `br`, `br_if`, `ret` and
   `unreachable` are a diagnostic.
2. **Loops.** Back edges are the retreating edges of a DFS; one whose
   target does not dominate its source makes the CFG irreducible -- a
   diagnostic naming the kernel and both blocks. Each loop head `h` gets a
   **header node** in front of it (a forwarding node that takes all of
   `h`'s incoming edges and carries `OpLoopMerge`) and a **continue node**
   that takes every back edge and branches to the header. The MIR head is
   then an ordinary block inside the loop, so its `br_if` is a selection
   like any other, and multiple latches need no special case.
3. **Merges**, visiting headers in reverse postorder (outer before inner):
   - *loop*: the unique target of the loop's exit edges. A target whose
     region the header dominates and that shares no block with another
     target's region is a returning exit (`if (x) return;` in the body); it
     is absorbed into the loop construct and the remaining target is the
     merge (with every target returning, the larger region stays the
     merge). Several live targets are a diagnostic naming them; no target
     at all gives an unreachable merge block.
   - *selection*: edges to the innermost loop's continue or merge node are
     continue/break and need no merge; an edge leaving the loop any other
     way is a diagnostic (multi-level break). With two ordinary arms the
     merge is the first block in reverse postorder reachable from both
     (no back edges, not leaving the loop); with no such block (the arms
     never reconverge, e.g. one returns) the arm reaching more blocks; with
     one ordinary arm, that arm; with none (break vs continue), an
     unreachable merge block.
   - A merge another header already claimed, or one the header does not
     dominate, is replaced by a fresh **forwarding node** that takes the
     edges into it from the header's region. That is how an inner `if`
     whose arms jump straight to the outer join gets its own merge, and how
     an inner loop whose only exit is the outer loop's merge or continue
     (a multi-level break that is the loop's only exit) becomes valid.
4. **Order**: reverse postorder (dominators first), unreachable merge
   blocks last.

**Block arguments -> `OpPhi`.** Every node whose block has parameters gets
one phi per parameter (per part for vectors). A forwarding node has the
phis of the block it forwards to and passes them on; an edge from a MIR
block carries that branch's arguments to whatever node the edge ends at
after redirection. Phi operands are filled after all blocks are lowered, so
back-edge values have ids. Entry-block parameters are the prologue's loads
directly unless the entry has a back edge.

KernelBuilder's `if_then`, `for_range*` (rolled and unrolled), the
reductions and every fused kernel produce shapes the structurizer handles
without forwarding beyond the per-loop header/continue nodes.

## SpirvISel (`include/brass/target/spirv/spirv_isel.hpp`)

| File | Concern |
| --- | --- |
| `spirv_isel.cpp` | driver, push-constant parameters, phis, terminators, opcode dispatch, builtins (cached in the prologue), value access, diagnostics |
| `spirv_isel_alu.cpp` | arithmetic, signed division fixup, comparisons, `select`, conversions, `add` on shared pointers |
| `spirv_isel_mem.cpp` | device-address loads/stores (scalar, indexed, vector), shared-array element pointers |
| `spirv_isel_vec.cpp` | vector ops per part, `fmin`/`fmax`/`vmin`/`vmax`, shuffles, lane access |
| `spirv_isel_intrinsics.cpp` | the intrinsic table, the unsupported list, builtins, math, conversions, `call` |
| `spirv_isel_intrinsics_mem.cpp` | barriers, shuffles, atomics, wide multiplies, shared arrays, narrow access |
| `spirv_structurize.cpp` | the structurizer above |

Rules:

- **Types.** One integer type per width with signedness 0 (signed
  operations choose the opcode: `OpSDiv`, `OpSLessThan`,
  `OpShiftRightArithmetic`, ...); `ptr` is `u64`. `i8`/`i16` values,
  `gcref` and `tagged` are diagnostics.
- **Comparisons** produce a `bool` used directly by `br_if`/`select`; the
  `i32` 0/1 MIR value is materialized with `OpSelect` only when something
  reads it as a value (the PTX backend's `setp`/`selp` rule). Float `ne` is
  `OpFUnordNotEqual`, the other float comparisons are ordered. A
  non-comparison condition is `OpINotEqual x, 0` / `OpFUnordNotEqual x, 0`.
- **Integer division** follows MIR (x64): `MIN / -1` wraps and `MIN % -1`
  is 0, implemented by selecting a safe divisor and the wrapped result
  when the divisor is not a constant other than 0 and -1. Division by zero
  is undefined -- a compute shader has no trap (PTX traps).
- **Constants** are `OpConstant`s created on first use, so they need no
  instruction and dominate everything.
- **Errors** name the kernel and the MIR block: unsupported opcodes
  (`clz`, `ctz`, `popcnt`, the overflow ops, `switch`, `invoke`, ...),
  calls to anything that is not an intrinsic, `ret` with a value, shared
  pointers that escape, and the structurizer diagnostics above.

### Semantic differences from PTX

- Precision is Vulkan's, not PTX's `.rn`: `OpFDiv` (2.5 ULP) and
  `GLSL.std.450 Sqrt` (inherited from `1/InverseSqrt`) are not correctly
  rounded, so `ptx_sqrt_rn`, MIR `sdiv.f32` and `sqrt` are approximations
  here; `Exp`, `Exp2`, `Log`, `Log2`, `Sin`, `Cos`, `InverseSqrt` have the
  GLSL precisions, comparable to PTX `.approx`. `Fma` is fused.
- `fptosi` / `ptx_f32_to_i32` out of range are undefined (PTX saturates);
  shifts by the bit width or more are undefined (PTX clamps).
- `fmin`/`fmax` are `NMin`/`NMax` (the non-NaN operand wins, like PTX
  `min/max` and `fminf`).
- `ptx_f32_to_f16` is `PackHalf2x16(x, 0)`: the half in the low 16 bits,
  zeros above; its rounding is the implementation's.

## Intrinsic mapping

Same MIR names and signatures as the PTX table
([ptx_kernel_authoring.md](ptx_kernel_authoring.md#intrinsic-table)).

| PTX name(s) | SPIR-V |
| --- | --- |
| `ptx_tid_{x,y,z}` | `BuiltIn LocalInvocationId` component (read once, in the prologue) |
| `ptx_ctaid_{x,y,z}` / `ptx_nctaid_{x,y,z}` | `BuiltIn WorkgroupId` / `NumWorkgroups` component |
| `ptx_ntid_{x,y,z}` | the `WorkgroupSize` spec constant (or the fixed `LocalSize` value) |
| `ptx_global_tid_x`, `ptx_global_id_x` | `BuiltIn GlobalInvocationId.x` |
| `ptx_laneid`, `ptx_lane_id` | `SubgroupLocalInvocationId & 31` (`GroupNonUniform`) |
| `ptx_clock` / `ptx_clock64` | `OpReadClockKHR` subgroup scope (`ShaderClockKHR`, `SPV_KHR_shader_clock`), truncated / as is |
| `rsqrt*`, `sqrt*`, `sin*`, `cos*`, `ex2*`, `lg2*`, `exp*`, `log*` | GLSL.std.450 `InverseSqrt`, `Sqrt`, `Sin`, `Cos`, `Exp2`, `Log2`, `Exp`, `Log` |
| `ptx_rcp`, `ptx_rcp_approx` / `ptx_div_approx` | `OpFDiv 1.0, x` / `OpFDiv a, b` |
| `ptx_sqrt_rn`, `fabs*` / `fmin*`, `fmax*` | `Sqrt`, `FAbs` / `NMin`, `NMax` (type from the argument) |
| `i32_to_f32`, `ptx_{i32,u32,i64,u64}_to_f32` | `OpConvertSToF` / `OpConvertUToF` |
| `ptx_f32_to_{i32,u32}`, `ptx_f32_to_f64`, `ptx_f64_to_f32` | `OpConvertFToS` / `OpConvertFToU`, `OpFConvert` |
| `ptx_f16_to_f32` / `ptx_f32_to_f16` | `UnpackHalf2x16(x).x` / `PackHalf2x16(vec2(x, 0))` (no `Float16` capability needed) |
| `bar.sync`, `ptx_sync`, `ptx_bar_sync(const id)` | `OpControlBarrier Workgroup, Workgroup, AcquireRelease \| WorkgroupMemory \| UniformMemory` (every id is the one workgroup barrier) |
| `ptx_shfl_{down,up,bfly,xor,idx}_{f32,i32}`, `ptx_shfl_down_sync_f32` | `OpGroupNonUniformShuffle` (`GroupNonUniformShuffle`) with a computed source lane; see below |
| `ptx_atom_add_f32` | `OpAtomicFAddEXT` (`AtomicFloat32AddEXT`, `SPV_EXT_shader_atomic_float_add`) |
| `ptx_atom_add_{i32,u32,i64}`, `_min_i32`, `_max_i32`, `_exch_i32` | `OpAtomicIAdd`, `OpAtomicSMin`, `OpAtomicSMax`, `OpAtomicExchange` (device scope, relaxed; `Int64Atomics` for i64) |
| `ptx_atom_shared_add_{f32,i32}` | the same on a Workgroup element, workgroup scope |
| `ptx_mul_wide_{u32,s32}` / `ptx_mul_hi_u32` / `ptx_mad_lo_u32` | `OpUConvert`/`OpSConvert` + `OpIMul` / `OpUMulExtended` high word / `OpIMul` + `OpIAdd` |
| `ptx_shared_alloc_*`, `ptx_shared_load/store_*[_indexed]` | Workgroup arrays and `OpAccessChain` (see Shared memory) |
| `ptx_load_{u8,s8,u16,s16}`, `ptx_store_{u8,u16}` | 8/16-bit PhysicalStorageBuffer access |

**Shuffles** keep PTX semantics on 32-lane segments of the subgroup: the
source lane is computed from the lane within the segment, a source outside
it (`down` past lane 31, `up` below lane 0) yields the thread's own value
(PTX's clamp behaviour as nvcc emits it), `bfly` xors and `idx` indexes
within the segment. One `OpGroupNonUniformShuffle` with the computed
absolute index does every mode. The member mask has no SPIR-V counterpart
and is ignored. Together with `laneid & 31` this makes the KernelBuilder
warp code (`warp_reduce_*`, `block_reduce_sum_f32`, `warp_id = tid >> 5`)
correct for subgroup sizes 32 and 64, provided subgroups map to
consecutive local invocation ids (true for 1-D workgroups on AMD and
NVIDIA; forcing `requiredSubgroupSize = 32` with
`VK_EXT_subgroup_size_control` makes it exact everywhere).

**Not lowered** -- `SpirvISel::unsupported_reason()` gives the reason and
the ISel throws it, naming the kernel and block:

| Name | Why |
| --- | --- |
| `ptx_warpid`, `ptx_warp_id` | the warp's hardware scheduler slot; Vulkan does not expose it (`KernelBuilder::warp_id()` is `tid.x >> 5` and works) |
| `ptx_nwarpid` | warp slots per SM: no SPIR-V equivalent |
| `ptx_smid`, `ptx_nsmid` | SM id / count: only vendor extensions |
| `ptx_globaltimer` | a nanosecond clock; device-scope `OpReadClockKHR` counts ticks of unspecified frequency |
| `ptx_bar_sync_count` | a barrier for a subset of the block; `OpControlBarrier` waits for the whole workgroup |

## Testing

Every test that needs `spirv-val` or `spirv-dis` (SPIRV-Tools; found by
CMake with `find_program`, else on `PATH`) prints a visible `[SKIP]` and
passes without them. `tests/unit/spirv_test_support.hpp` has the helpers:
`compile_checked` (facade + `spirv::verify` + `spirv-val --target-env
vulkan1.2`, or `vulkan1.3` for SPIR-V 1.6), `disassemble`,
`require_diagnostic`, and a `Kernel` builder.

| File | Covers |
| --- | --- |
| `test_spirv_ir.cpp` | type/constant dedup, the binary layout (header, word counts, strings, 64-bit literals), the dump, every verifier rule on a hand-built failing module, and spirv-val rejecting an unstructured branch the verifier allows (proving the tool runs) |
| `test_spirv_isel.cpp` | every scalar ALU op and conversion, comparisons and their materialization, the division fixup, scalar/indexed/vector memory, every vector op on all eight vector types; if, if-else with a phi, loops with carried scalars and 256-bit vectors, nested and unrolled loops, early return and break in a loop, nested ifs sharing a join, both arms returning, a do-while latch, a loop left by return, a loop at the entry block, a `br_if` with one target; diagnostics for irreducible CFGs, multi-exit loops, unsupported opcodes/terminators/calls, non-void kernels and escaping shared pointers |
| `test_spirv_intrinsics.cpp` | a signature row per table entry (coverage-checked), each validated; every PTX intrinsic lowered or diagnosed with its reason; per-family opcodes in the spirv-dis text; shared-access size and constant-count diagnostics; the KernelBuilder warp and block reductions as a kernel |
| `test_spirv_target.cpp` | the push-constant layout, spec-constant vs fixed workgroup size, a two-kernel module, SPIR-V 1.6, the disassembly of a full kernel, and the ten fused ML kernels lowering unchanged to valid modules |

There are no on-device tests yet; `spirv-val` passing is the bar for this
stage.

## What is not done

- No Vulkan runtime and no execution (stage 2).
- `switch`, `invoke` and the other exception/coroutine terminators;
  `clz`/`ctz`/`popcnt` and the overflow ops; `i8`/`i16` values; calls to
  non-intrinsic functions.
- Loops whose exits continue into different blocks (route them through one
  exit block); multi-level breaks other than an inner loop's only exit.
- A storage-buffer-descriptor mode (device addresses only).
- Mixed-size accesses of one shared array.
- Backend-side optimisation (the driver compiler does it).

## Requirements for the Vulkan runtime (stage 2)

Device features by capability (`SpirvKernel::capabilities`):
`bufferDeviceAddress` (always), `shaderInt64` (always), `shaderFloat64`
(`Float64`), `storageBuffer8BitAccess` / `storageBuffer16BitAccess`
(narrow access), subgroup `SHUFFLE` and `BASIC` operations in the compute
stage (`GroupNonUniform*`), `shaderBufferFloat32AtomicAdd` /
`shaderSharedFloat32AtomicAdd` from `VK_EXT_shader_atomic_float`
(`AtomicFloat32AddEXT`), `shaderInt64Atomics` (`Int64Atomics`),
`shaderSubgroupClock` from `VK_KHR_shader_clock` (`ShaderClockKHR`).
Buffers need `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` and memory
allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`; the pipeline layout
has one push-constant range of `push_constant_bytes` and no descriptor
sets; the workgroup size is specialization constants 0..2.

## File size rule

As for the rest of brass: files stay under 1,000 lines, split by concern
(the ISel is split by family like `ptx_isel_*.cpp`); nothing over 2,000.
