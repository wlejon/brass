# SPIR-V Backend Design

Status: the target, its IR, verifier and tests (stage 1 of 3), and the
Vulkan runtime with on-device execution of the ISel features and the fused
ML kernels (stage 2). This document is the reference for the SPIR-V compute
backend: the typed SPIR-V module model, the writer and verifier, the
structurizer, the instruction selector, the memory model, the kernel ABI and
the runtime contract an embedder's Vulkan code must honour. Kernels are written exactly as for PTX
([ptx_kernel_authoring.md](ptx_kernel_authoring.md)); the same MIR built
with `KernelBuilder` lowers to PTX for NVIDIA and to SPIR-V for Vulkan. The
PTX backend this one mirrors is described in
[ptx_backend_design.md](ptx_backend_design.md).

Planned next: brotensor driving these kernels inside its own Vulkan runtime,
then its trace JIT emitting through this target (stage 3).

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
  `GLSL.std.450.h`); nothing hand-types an opcode number. Both headers are
  vendored, unmodified, in `third_party/spirv-headers` (MIT, Khronos), so
  `BRASS_WITH_SPIRV` is 1 in every build; `-DBRASS_SPIRV_HEADERS_DIR=<dir>`
  points at another copy (a Vulkan SDK, a SPIRV-Headers checkout), and only
  a directory without `spirv/unified1/spirv.hpp` leaves the target out.
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
- NaN and infinity results are undefined except where SPIR-V defines them
  (comparisons, `NMin`/`NMax`): the module does not declare
  `SignedZeroInfNanPreserve`, so the driver may assume finite values. On
  RADV `f64` division by NaN returns 0. Signed zeros may be lost the same
  way.
- `smod` on floats is `OpFRem`, whose Vulkan precision is that of
  `x - y * trunc(x / y)`: exact for small quotients, wrong by whole
  multiples of `y` once `x / y` loses integer precision (RADV:
  `fmod(1e10f, 3)` gives 256).
- The driver may contract a MIR `mul` feeding an `add` into an FMA (the
  instructions are not `NoContraction`), as `ptxas` may for PTX `mul`/`add`
  without a rounding modifier.
- Float **conversions** are exact: every `OpFConvert`, `OpConvertSToF`,
  `OpConvertUToF`, `OpConvertFToS` and `OpConvertFToU` is decorated
  `NoContraction`, because Mesa otherwise treats them as inexact and folds
  `fpext(fptrunc(x))` to `x` and `fpext(sitofp_f32(i))` to `sitofp_f64(i)`,
  dropping the rounding MIR specifies (found on device, see Measured).
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

## Vulkan runtime (`include/brass/gpu/vulkan_driver.hpp`)

`src/gpu/vulkan_driver.cpp` (loader, device, buffers, submission) and
`vulkan_driver_module.cpp` (pipelines, dispatch, the capability check). It is
the test and benchmark harness and the reference for an embedder's own
runtime, not a general one: one process-wide device and compute queue, one
command buffer, every submission waited on with a fence.

- **Loading.** `libvulkan.so.1` (`vulkan-1.dll`, `libvulkan.1.dylib`) is
  `dlopen`ed on first use, like `cuda_driver`; nothing links against Vulkan.
  The implementation needs the Vulkan 1.3 headers at build time
  (`__has_include(<vulkan/vulkan.h>)`); without them every entry point
  reports "unavailable" and the tests print `[SKIP]`.
- **Device.** Instance API version `min(loader, 1.3)`; the first discrete
  GPU, else the first integrated one, else any Vulkan 1.2 device;
  `BRASS_VULKAN_DEVICE=<index>` overrides. The device must have
  `bufferDeviceAddress` and `shaderInt64` (every kernel needs them);
  everything else is enabled when supported and reported in
  `VulkanDeviceCaps`.
- **Features enabled** (exactly these, not everything the device has):
  `shaderInt64`, `shaderFloat64`, `shaderInt16`; `storageBuffer16BitAccess`
  (1.1); `bufferDeviceAddress`, `storageBuffer8BitAccess`,
  `shaderBufferInt64Atomics`, `shaderSharedInt64Atomics` (1.2);
  `subgroupSizeControl` + `computeFullSubgroups` (1.3 core, or
  `VK_EXT_subgroup_size_control`); `shaderBufferFloat32AtomicAdd`,
  `shaderSharedFloat32AtomicAdd`, `shaderBufferFloat64AtomicAdd`
  (`VK_EXT_shader_atomic_float`); `shaderSubgroupClock`
  (`VK_KHR_shader_clock`). Subgroup BASIC and SHUFFLE in the compute stage
  are properties: checked, not enabled.
- **`VulkanDeviceCaps::missing_for(kernel)`** maps each name in
  `SpirvKernel::capabilities` to its feature (`Float64` -> `shaderFloat64`,
  `StorageBuffer8BitAccess` -> `storageBuffer8BitAccess`,
  `GroupNonUniformShuffle` -> compute-stage SHUFFLE, `AtomicFloat32AddEXT`
  -> `shaderBufferFloat32AtomicAdd`, `Int64Atomics` ->
  `shaderBufferInt64Atomics`, `ShaderClockKHR` -> `shaderSubgroupClock`,
  ...) and checks `push_constant_bytes` and `shared_bytes` against the
  limits; `VulkanModule::load` refuses a kernel with that list, naming the
  device. An unknown capability is refused too.
- **`VulkanBuffer`**: device-local memory allocated with
  `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`, usage `STORAGE_BUFFER |
  TRANSFER_SRC | TRANSFER_DST | SHADER_DEVICE_ADDRESS`, size rounded up to 4;
  `device_address()` is `vkGetBufferDeviceAddress`. `upload` / `download`
  go through a temporary host-visible coherent staging buffer and
  `vkCmdCopyBuffer`; `fill` / `zero` are `vkCmdFillBuffer`.
- **`VulkanModule::load(kernel, err, {subgroup_size})`**: one shader module,
  a pipeline layout with no descriptor sets and one push-constant range
  (compute stage, offset 0, `push_constant_bytes`), and one compute
  pipeline per distinct workgroup size, created on first use and cached
  (the kernel's default size is created at load, so a driver compile error
  surfaces there). The workgroup size goes in as specialization constants
  0, 1, 2 (`uint32`) unless the kernel has a fixed `LocalSize`
  (`SpirvKernel::local_size_spec_constants` false), in which case a
  dispatch must use exactly that size. `subgroup_size` (default 32) chains
  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when the device can
  require that size in compute, plus `REQUIRE_FULL_SUBGROUPS` when the
  block's x size is a multiple of it; 0 leaves the choice to the driver.
- **`launch(dispatch, args)` / `launch_1d(grid, block, {args})`**: packs the
  push-constant block from `VulkanArg`s (a `VulkanBuffer` is its device
  address; `uint64_t`/`int64_t` for `ptr`/`i64`, `uint32_t`/`int32_t` for
  `i32`, `float`, `double`), checking the count and each kind against
  `SpirvKernel::params`, then binds, pushes, dispatches `repeat` times
  (compute-to-compute barriers between) and waits on the fence. Every
  submission starts with an all-commands memory barrier and ends with one
  to host reads, because submission order alone is not a memory
  dependency. `gpu_ms` returns the GPU time from two timestamp queries.

### Runtime contract (what an embedder's Vulkan code must do)

1. Device: Vulkan 1.2+ with `bufferDeviceAddress` and `shaderInt64`
   enabled, plus whichever features of the list above the kernels'
   declared capabilities need (check them as `missing_for` does). SPIR-V
   1.6 modules only on a Vulkan 1.3 device.
2. Buffers: every buffer a kernel touches is created with
   `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (plus storage/transfer usage
   as needed) on memory allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`;
   a `ptr` argument is `vkGetBufferDeviceAddress(buffer) + byte offset`.
   Alignment: scalar accesses need their element size, `vload`/`vstore`
   (every float4 path of the fused kernels) 16 bytes -- the `Aligned`
   operands promise it and the driver compiles to it.
3. Pipeline layout: zero descriptor set layouts and exactly one
   push-constant range `{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_bytes}`.
4. Push constants: the parameters in declaration order at
   `SpirvKernel::params[i].offset` (each aligned to its own size: `ptr`/`i64`
   8 bytes, `i32`/`f32` 4, `f64` 8), little-endian, pushed with
   `vkCmdPushConstants(cmd, layout, COMPUTE, 0, push_constant_bytes, data)`.
   The fused kernels take 24 to 60 bytes.
5. Workgroup size: `VkSpecializationInfo` with map entries `{0, 0, 4}`,
   `{1, 4, 4}`, `{2, 8, 4}` (constant ids 0, 1, 2 = `uint32` x, y, z) when
   `local_size_spec_constants`; the entry point is `SpirvKernel::entry`.
6. Subgroups: require size 32 with
   `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when the device
   allows it (and `REQUIRE_FULL_SUBGROUPS` for x sizes that are multiples of
   32). The kernels are also exact at 64 (tested on RADV), so a device that
   cannot require 32 works as long as subgroups are consecutive local
   invocation ids (true for 1-D workgroups on AMD and NVIDIA).
7. Dispatch: `vkCmdDispatch(grid_x, grid_y, grid_z)` with the grids of the
   PTX launch contracts (`ml_fusion.hpp`); shared memory is static. A
   compute-to-compute memory barrier between dependent dispatches (several
   kernels update `x` in place).

## Testing

Every test that needs `spirv-val` or `spirv-dis` (SPIRV-Tools; found by
CMake with `find_program`, else on `PATH`) prints a visible `[SKIP]` and
passes without them, and every test that executes prints `[SKIP]` without a
Vulkan device. `tests/unit/spirv_test_support.hpp` has the static helpers:
`compile_checked` (facade + `spirv::verify` + `spirv-val --target-env
vulkan1.2`, or `vulkan1.3` for SPIR-V 1.6), `disassemble`,
`require_diagnostic`, and a `Kernel` builder; `spirv_exec_support.hpp` adds
`vk_ready`, `load`, `launch`, `upload`/`download` and the
one-thread-per-element `run_map`.

| File | Covers |
| --- | --- |
| `test_spirv_ir.cpp` | type/constant dedup, the binary layout (header, word counts, strings, 64-bit literals), the dump, every verifier rule on a hand-built failing module, and spirv-val rejecting an unstructured branch the verifier allows (proving the tool runs) |
| `test_spirv_isel.cpp` | every scalar ALU op and conversion, comparisons and their materialization, the division fixup, scalar/indexed/vector memory, every vector op on all eight vector types; if, if-else with a phi, loops with carried scalars and 256-bit vectors, nested and unrolled loops, early return and break in a loop, nested ifs sharing a join, both arms returning, a do-while latch, a loop left by return, a loop at the entry block, a `br_if` with one target; diagnostics for irreducible CFGs, multi-exit loops, unsupported opcodes/terminators/calls, non-void kernels and escaping shared pointers |
| `test_spirv_intrinsics.cpp` | a signature row per table entry (coverage-checked), each validated; every PTX intrinsic lowered or diagnosed with its reason; per-family opcodes in the spirv-dis text; shared-access size and constant-count diagnostics; the KernelBuilder warp and block reductions as a kernel |
| `test_spirv_target.cpp` | the push-constant layout, spec-constant vs fixed workgroup size, a two-kernel module, SPIR-V 1.6, the disassembly of a full kernel, and the fused ML kernels lowering unchanged to valid modules |
| `test_spirv_execution.cpp` | on device: the runtime (buffers, partial transfers, the capability check, every parameter type and its push-constant padding, argument count/kind errors, spec-constant and fixed workgroup sizes, SPIR-V 1.6, 3-D grids and every index builtin); i32/i64 ALU with MIN / -1, MIN % -1, shifts and constant divisors; f32/f64 ALU with NMin/NMax on NaN and RoundEven; every comparison as a value and as a condition (float `ne` unordered); conversions and bitcasts |
| `test_spirv_execution_cf.cpp` | on device: if / if-else phis, nested and unrolled loops with divergent trip counts, early return and break in a loop, nested ifs sharing a join, both arms returning, do-while, a loop left by return, a loop at the entry block with a one-target `br_if`, loop-carried 256-bit vectors, a grid-stride loop |
| `test_spirv_execution_mem.cpp` | on device: scalar loads/stores of every width (indexed, displaced, one pointer at two types), 8/16-bit access at odd offsets, all eight vector types, shared memory of every element type with barriers and pointer offsets, global and shared atomics (float add, old values), f16 unpack of all 65536 halves and pack, clock, the math and wide-multiply intrinsics |
| `test_spirv_execution_subgroup.cpp` | on device: every shuffle mode (f32/i32, constant and register deltas) against PTX semantics, and the warp and block reductions at blocks 32..1024, each at subgroup size 32, 64 and the driver's choice |
| `test_spirv_kernels.cpp` | on device: the eleven fused kernels (`MlFusionCompiler::compile_spirv`) against double-precision host references over `test_gpu_kernels.cpp`'s shapes and tolerances, and against the CPU `KernelJit` SwiGLU, AdaLN and residual RMSNorm |
| `test_spirv_kernels_quant.cpp` | on device: GEMV Q8_0 / Q4_K over `test_gpu_kernels_quant.cpp`'s shapes at subgroup sizes 32 and 64, and against the CPU `KernelJit` GEMVs |

`brass_spirv_bench` (`tests/benchmarks/bench_spirv_vulkan.cpp`, run by
hand, not a ctest) times kernels against hand-written GLSL twins in
`tests/benchmarks/spirv/*.comp`, compiled by `glslc` at build time when
CMake finds it and launched through the same runtime with the same
push-constant ABI.

## Measured

AMD Radeon 8060S (Strix Halo, gfx1151, integrated, LPDDR5X), Mesa RADV
26.2.3, Vulkan 1.4, October 2026.

**Device features.** Everything the target can declare is supported and
enabled: `bufferDeviceAddress`, `shaderInt64`, `shaderFloat64`,
`shaderInt16`, 8/16-bit storage, subgroup BASIC + SHUFFLE in compute,
`shaderBufferFloat32AtomicAdd` / `shaderSharedFloat32AtomicAdd`,
`shaderBufferInt64Atomics`, `shaderSubgroupClock`, `subgroupSizeControl`
with `computeFullSubgroups`. Push constants up to 256 bytes, 64 KiB of
shared memory, timestamp period 10.02 ns.

**Subgroup size.** RADV reports `subgroupSize` 64 and a range of 32..64,
requirable in compute. Without a required size, compute pipelines run
wave64 (`gl_SubgroupSize` 64 at blocks 64 and 256); requiring 32 or 64 is
honoured. The runtime requires 32 by default. The fused kernels and the
reductions give bit-identical results at 32 and 64 (the shuffles work on
32-lane segments, so the reduction order does not change).

**What the device found.** Every stage-1 kernel and control-flow shape ran
correctly the first time except float conversions: Mesa folded
`fpext(fptrunc(x))` and `fpext(sitofp_f32(i))`, skipping the f32 rounding.
Fixed by decorating every float conversion `NoContraction`
(`test_spirv_isel.cpp` checks the decorations, `test_spirv_execution.cpp`
the values). It also showed the precision rules listed under "Semantic
differences": `OpFRem` loses whole multiples of `y` for large quotients,
and NaN results are undefined without `SignedZeroInfNanPreserve`
(documented, not changed: matching x64 there would cost every kernel).
Accuracy of the fused kernels against the double-precision references is
far inside the CUDA tolerances: max relative error 1.4e-7 for SwiGLU, 4e-8
for AdaLN, 1.1e-7 for the norms, 1.9e-6 for the GEMVs and 2.6e-6 for the
quantized GEMVs; the in-place `x += res` outputs are exact.

**Bandwidth** (`brass_spirv_bench`, median of 5 command buffers of 20
dispatches, both at subgroup size 32):

| Kernel | Block | brass SPIR-V | GLSL (glslc -O) |
| --- | --- | --- | --- |
| SwiGLU, 64M floats (768 MiB per run) | 128 | 220.4 GB/s | 220.8 GB/s |
| | 256 | 229.7 GB/s | 230.0 GB/s |
| | 512 | 230.8 GB/s | 230.7 GB/s |
| residual RMSNorm, 4096 x 4096 (256 MiB per run) | 128 | 180.5 GB/s | 182.8 GB/s |
| | 256 | 208.5 GB/s | 208.2 GB/s |
| | 512 | 210.7 GB/s | 212.8 GB/s |
| | 1024 | 217.9 GB/s | 216.7 GB/s |

The generated kernels are within 1.5% of the GLSL ones everywhere (outputs
identical for SwiGLU, within 6e-8 for RMSNorm), about 90% of the memory's
nominal 256 GB/s: the `Aligned 16` vector accesses become 128-bit global
loads and stores as in the GLSL, so no codegen change was needed. The
RMSNorm at subgroup size 64 is 2-3% slower at blocks 256..1024 and 2%
faster at 128.

## What is not done

- `switch`, `invoke` and the other exception/coroutine terminators;
  `clz`/`ctz`/`popcnt` and the overflow ops; `i8`/`i16` values; calls to
  non-intrinsic functions.
- Loops whose exits continue into different blocks (route them through one
  exit block); multi-level breaks other than an inner loop's only exit.
- A storage-buffer-descriptor mode (device addresses only).
- Mixed-size accesses of one shared array (no fused kernel needs one).
- Backend-side optimisation (the driver compiler does it).
- An option declaring `SignedZeroInfNanPreserve`, and an exact `fmod`, for
  kernels that need x64's NaN and remainder semantics.
- The runtime is synchronous and single-queue by design; asynchronous
  submission and buffer pooling belong to the embedder's runtime.

## File size rule

As for the rest of brass: files stay under 1,000 lines, split by concern
(the ISel is split by family like `ptx_isel_*.cpp`); nothing over 2,000.
