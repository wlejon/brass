# GPU Portability & Backend-Neutral MIR Contract

Brass lowers a single GPU MIR kernel to **PTX (CUDA)**, **SPIR-V 1.5 (Vulkan compute)**, and **Metal Shading Language (MSL)** without separate per-backend kernel sources. Because the hardware and driver guarantees differ across NVIDIA warps, AMD RDNA waves, and Apple Silicon simdgroups, portable kernels must obey the invariants below.

---

## 1. Backend-Neutral `gpu_*` Intrinsics

Both canonical `ptx_*` names and backend-neutral `gpu_*` aliases are accepted by `PtxTarget`, `SpirvTarget`, `MslTarget`, and `optimize_gpu_function()` (normalized via [`brass/target/gpu_intrinsics.hpp`](../include/brass/target/gpu_intrinsics.hpp)). New backend-neutral kernels and `KernelBuilder` helpers may use `gpu_*` names while existing `ptx_*` callers remain supported with identical lowering:

| Domain | `gpu_*` Alias | Canonical Equivalent | Notes |
| --- | --- | --- | --- |
| Thread / grid indices | `gpu_tid_{x,y,z}`, `gpu_ntid_{x,y,z}`, `gpu_ctaid_{x,y,z}`, `gpu_nctaid_{x,y,z}`, `gpu_global_tid_x` | `ptx_tid_*`, `ptx_ntid_*`, `ptx_ctaid_*`, `ptx_nctaid_*`, `ptx_global_tid_x` | `ntid` / `tid` are 32-bit workgroup-local indices. |
| Subgroup lane / id | `gpu_lane_id`, `gpu_laneid`, `gpu_subgroup_id` | `ptx_lane_id`, `ptx_laneid`, `ptx_subgroup_id` | `gpu_lane_id` is always in `[0, 31]`. |
| Barriers | `gpu_sync`, `gpu_bar_sync`, `gpu_subgroup_barrier` | `ptx_sync`, `ptx_bar_sync`, `ptx_subgroup_barrier` | Must be reached uniformly by the active scope. |
| Subgroup shuffles | `gpu_shfl_{down,up,bfly,xor,idx}_{f32,i32}` | `ptx_shfl_*` | Logical 32-lane shuffle segment (see §2). |
| Shared memory | `gpu_shared_alloc_{f32,i32,f64,i64,i32x4,u16}`, `gpu_shared_{load,store}_*` | `ptx_shared_*` | Allocation count must be a positive `iconst_i32`. |
| Cooperative matrix | `gpu_coop_{alloc_a,alloc_b,alloc_acc,fill_acc,load_a,load_b,load_acc,mma,store_acc,scale_acc,add_acc,acc_len,get_acc,set_acc}` | `ptx_coop_*` | 16×16×16 FP16×FP16 → FP32 accumulator (see §4). |
| Atomics & narrow mem | `gpu_atom_[shared_]*`, `gpu_{load,store}_{u8,s8,u16,s16}` | `ptx_atom_*`, `ptx_{load,store}_*` | `i64` atomics are unavailable on Metal MSL. |
| Math & conversions | `gpu_{exp,log,ex2,lg2,rsqrt,sqrt,sqrt_rn,sin,cos,tanh,fabs,fmin,fmax,clamp,fma}`, `gpu_{f16,bf16}_*`, `gpu_mul_hi_u32`, `gpu_sdot4` | `ptx_*` | Fast approximate transcendental semantics. |

Non-portable PTX hardware slot/timer intrinsics (`ptx_warpid`, `ptx_nwarpid`, `ptx_smid`, `ptx_nsmid`, `ptx_globaltimer`, `ptx_bar_sync_count` and their `gpu_*` names) are explicitly rejected by `SpirvTarget` and `MslTarget` with clear diagnostics. Use `tid_x >> 5` (or `gpu_subgroup_id` when matching subgroup storage) instead of `warpid`.

---

## 2. Warp / Subgroup Portability Invariants

Portable kernels assume a **32-lane logical subgroup**:

- **NVIDIA CUDA (PTX):** hardware warp width is always 32 (`%laneid` in `0..31`).
- **Apple Silicon (Metal):** `[[thread_index_in_simdgroup]]` is 32 lanes wide (`simd_shuffle*`).
- **AMD / Intel / Vulkan (SPIR-V):** physical `SubgroupSize` may be 32 or 64 (RDNA wave64, Intel wave8/16/32).
  - `gpu_laneid` / `gpu_lane_id` masks `BuiltInSubgroupLocalInvocationId & 31` so lane indices remain in `0..31` on wave64.
  - Subgroup shuffles (`ptx_shfl_*` / `gpu_shfl_*`) operate within aligned 32-lane segments (`(id & ~31) | ((id + delta) & 31)` or `OpGroupNonUniformShuffleXor`), preserving reduction tree equivalence on both wave32 and wave64.
  - **Rule for kernel authors:** always allocate shared reduction scratch using `KernelBuilder::warp_id()` (`tid_x >> 5`, up to `wg / 32` slots) for cross-warp reductions rather than assuming physical hardware `subgroup_id` matches `/ 32`, and keep barriers (`gpu_sync` / `gpu_subgroup_barrier`) control-flow uniform across all threads in the workgroup/subgroup.

---

## 3. Shared Memory & Pointer Provenance

- Shared memory pointers created by `gpu_shared_alloc_*` carry static address-space provenance (`threadgroup` in MSL, `Workgroup` array in SPIR-V, `.shared` in PTX).
- Only offset a shared pointer with `add` (or pass it directly to `gpu_shared_*` / `gpu_coop_*` / `gpu_atom_shared_*`) — never `select` between a global pointer and a shared pointer or escape a shared pointer through memory.
- Keep `gpu_shared_load_*` / `gpu_shared_store_*` element types aligned with the `gpu_shared_alloc_*` element type (`SpirvISel` enforces matching array element width).

---

## 4. Cooperative Matrix (`16×16×16`) & Precision Contract

- `coop_alloc_{a,b,acc}` represents a `16×16×16` matrix multiply-accumulate tile (`A: f16`, `B: f16`, `Acc: f32`).
- Every thread in the subgroup must uniformly participate in `coop_load_*`, `coop_mma`, `coop_store_acc`, `coop_scale_acc`, and `coop_add_acc`.
- Because fragment layouts differ across NVIDIA `wmma`, Vulkan `SPV_KHR_cooperative_matrix`, and Metal `simdgroup_matrix` (4 tiled 8×8 fragments), portable kernels must never assume a hardcoded element-to-coordinate mapping for register accumulators; use `coop_acc_len`, `coop_get_acc`, and `coop_set_acc` only for pointwise transforms (scaling, masking, softmax/rowwise ops), or stage via `coop_store_acc` -> shared memory -> `coop_load_*`.
- **FP64 gating:** `double` (`f64`) is supported in PTX and optionally in SPIR-V (`Float64` capability), and **unsupported in Metal MSL** (`brotensor::interop::device_has_float64(dev)` returns `false` on Metal and queries `shader_float64` on Vulkan).
