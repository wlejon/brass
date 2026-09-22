# Patchable Sites & Thread-Safe Concurrency Protocol

Dynamic language runtimes rely on runtime code modification for Inline Caches (ICs), polymorphic call sites, on-stack replacement (OSR), and speculative deoptimization. Brass guarantees thread-safe cross-modification and code installation without requiring mutator threads to stop at global safepoints.

---

## 1. Cache-Line Alignment Safety Rules

### 1.1. Hardware Hazard of Split-Line Writes
On modern x86-64 and AArch64 processors, memory writes that cross a 64-byte hardware cache line boundary are split into multiple micro-architectural memory transactions. Such split-cache-line writes **cannot be executed atomically** by the processor memory bus.

If a mutator thread executes code across a cache-line boundary while another thread modifies it, the executing core can fetch a torn instruction (half old displacement, half new displacement). This results in invalid opcode faults (`#UD` / `SIGILL`) or branching to unmapped memory.

### 1.2. Cache-Line Safety Predicate (`is_cache_line_safe`)
Brass enforces strict cache-line boundary containment for all dynamic patches:

```cpp
inline bool is_cache_line_safe(const void* addr, size_t size) noexcept {
    if (!addr || size == 0) return true;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr);
    uintptr_t end = start + size - 1;
    return (start / 64) == (end / 64);
}
```

The patch is safe if and only if the entire modified byte span `[addr, addr + size - 1]` resides within the same 64-byte cache line:
$$\left\lfloor \frac{\text{start}}{64} \right\rfloor = \left\lfloor \frac{\text{end}}{64} \right\rfloor$$

### 1.3. Emitter Alignment Padding
During code emission, the code generator evaluates the projected byte offset of immediate constants and jump/call displacements using `compute_cache_line_padding`:
- If an immediate or displacement would cross a 64-byte boundary, the emitter inserts multi-byte NOP instructions (`0x90` or canonical multi-byte NOP sequences `0x0F 0x1F ...`) immediately preceding the instruction.
- All patchable immediate slots (`patchable_const.i32`, `patchable_const.i64`) and call displacements (`patchable_call`) are guaranteed to lie within a single cache line.

### 1.4. Runtime Safety Rejection
All runtime patching functions (`brass_patch_const32`, `brass_patch_const64`, `brass_patch_call`) verify `is_cache_line_safe` before performing any memory write. If an unaligned or cross-boundary address is detected, the operation is rejected immediately and returns `false`.

---

## 2. Instruction Displacement Patching

Brass supports dynamic in-place rewriting of direct calls and near branches on x86-64 and AArch64.

### 2.1. x86-64 Near Calls (`0xE8`) and Near Jumps (`0xE9`)

| Instruction | Opcode | Operand | Total Size |
| :--- | :--- | :--- | :--- |
| **Near Call (`call rel32`)** | `0xE8` | 32-bit signed displacement (`int32_t`) | 5 bytes |
| **Near Jump (`jmp rel32`)** | `0xE9` | 32-bit signed displacement (`int32_t`) | 5 bytes |

#### Instruction Layout & Displacement Calculation:
```
+---------------+-------------------------------+
| Opcode (0xE8) |  rel32 Displacement (4 bytes)  |
+---------------+-------------------------------+
  Byte 0          Bytes 1 .. 4
  ^               ^
  inst            disp_ptr
```
- **Displacement Offset**: The displacement bytes begin at `disp_ptr = inst + 1`.
- **Next Instruction Pointer**: On x86-64, relative displacements are calculated from the start of the next instruction:
  $$\text{next\_ip} = \text{inst} + 5$$
- **Relative Target Displacement**:
  $$\text{disp} = \text{target\_dest} - \text{next\_ip}$$
- **Reach Verification**: The calculated displacement is checked to ensure it fits within the 32-bit signed displacement envelope:
  $$\text{INT32\_MIN} \le \text{disp} \le \text{INT32\_MAX} \quad (\pm 2\text{ GB})$$

#### Atomic Modification Protocol:
1. Verify `is_cache_line_safe(disp_ptr, sizeof(int32_t))`.
2. Apply write-protection unmasking (`ScopedCodeWrite`).
3. Store the 32-bit displacement using an aligned atomic release store:
   ```cpp
   atomic_store_release(reinterpret_cast<int32_t*>(disp_ptr), static_cast<int32_t>(disp));
   ```
4. Issue a memory fence (`_mm_mfence()`).
5. Flush the instruction cache (`FlushInstructionCache` on Windows, `__builtin___clear_cache` on POSIX) to invalidate instruction prefetch queues.

### 2.2. AArch64 Direct Branch Patching (`B` / `BL`)
- Opcodes: `0x14000000` (`B`) and `0x94000000` (`BL`).
- Operand: 26-bit signed immediate word offset ($\pm 128$ MB range).
- Instruction is word-aligned (4-byte aligned).
- The complete 32-bit instruction word is updated atomically with release semantics and I-cache invalidation.

---

## 3. Concurrency Protocol

Brass allows concurrent JIT compilation, multi-tier execution, and parallel loop dispatch without global mutator locks.

### 3.1. Lock-Free Invocation Counting in `MultiTierPipeline`
Function call tracking is designed to eliminate global contention on multi-core mutator threads:
- **Lock-Free Counters**: Each function maintains atomic invocation counters:
  - `FunctionHandle::record_call()`: Atomically increments `std::atomic<uint64_t> invocation_count_` with relaxed memory order.
  - `TieringFeedback::record_invocation()`: Increments the execution counter without acquiring mutexes.
- **Pipeline Statistics**: `MultiTierPipeline` maintains atomic counters for each tier (`stats_.tier0_invocations`, `tier1_invocations`, `tier2_invocations`).
- **Threshold Triggers**: When an invocation counter hits the Tier 1 baseline threshold or Tier 2 optimization threshold, compilation is triggered or enqueued to the background worker pool without blocking the invoking mutator thread.

### 3.2. Atomic Code Installation
When a function is compiled into native machine code (either synchronously or via the background compiler), the new entry point is published atomically:

```cpp
// Publishing compiler thread:
handle->set_baseline_function(compiled_ptr);
handle->set_native_entry(compiled_ptr->entry_point());
handle->set_tier(TierLevel::Tier1_Baseline);

// Dispatching mutator thread:
void* entry = handle->native_entry(); // std::memory_order_acquire
if (entry) {
    // Jump directly to native machine code
} else {
    // Fall back to interpreter
}
```

- **Release/Acquire Synchronization**:
  - `handle->set_native_entry()` stores the entry pointer with `std::memory_order_release`.
  - `handle->native_entry()` loads the pointer with `std::memory_order_acquire`.
- **Memory Visibility**: The store-release ensures that all machine instructions, literal pools, exception tables (`.pdata`/`.eh_frame`), and binary stack maps (`BSCM`) emitted by the compiler thread are fully visible in cache before any mutator thread reads the pointer and enters the function.

### 3.3. Re-Entrant `parallel_for` Execution in `ParallelRuntime`
The parallel runtime (`ParallelRuntime`) executes loop chunks across worker threads. To prevent deadlock in nested or recursive contexts, it implements a re-entrancy protocol:

- **Hazard**: If a kernel executing inside a parallel loop invokes `brass_parallel_for` (nested parallelism), or if code running on a worker thread triggers a parallel loop, attempting to re-acquire the master dispatch lock (`dispatch_mutex_`) or recurse into the worker queue causes deadlock or worker starvation.
- **Context Tracking**: The runtime tracks execution depth using thread-local state:
  - `thread_local uint32_t t_nesting_depth`
  - `thread_local bool t_is_worker_thread`
- **Re-Entrant Fallback**:
  ```cpp
  if (t_nesting_depth > 0 || t_is_worker_thread) {
      // Execute sequentially on the current thread to prevent deadlock
      kernel(0, trip_count, context);
      return;
  }
  ```
- **Reduction State Isolation**: When running re-entrantly, thread-local reduction accumulators (`red_i64`, `red_f64`, `current_reduction_`) are preserved on the stack (`saved_i64`, `saved_f64`, `saved_red`) and restored upon exit. This ensures nested reductions do not corrupt outer reduction accumulators.
- **Concurrent Host Invocations**: Independent invocations of `parallel_for` from different external host threads are safely serialized via `dispatch_mutex_`.

---

## 4. Runtime Patching APIs

### 4.1. Low-Level C Boundary
```c
// Patch a 32-bit constant site at raw code address
bool brass_patch_const32(void* code_addr, int32_t new_val);

// Patch a 64-bit constant site at raw code address
bool brass_patch_const64(void* code_addr, int64_t new_val);

// Patch a direct call (0xE8) or jump (0xE9) site at raw code address
bool brass_patch_call(void* call_site_addr, const void* new_target);
```

### 4.2. JIT Execution Engine (`JitExecutionEngine`)
```cpp
// Patch by symbolic site name in loaded module
jit.patch_const32("ic_type_slot_42", 123);
jit.patch_const64("bias_slot_1", 0x100000000LL);
jit.patch_call("call_ic_site_1", new_target_fn_ptr);
jit.patch_call("call_ic_site_1", "optimized_fast_stub");
```

### 4.3. Embedder Host Module (`CompiledModule`)
```cpp
compiled->patch_constant("ic_type_slot_42", 123);
compiled->patch_constant("bias_slot_1", int64_t(42));
compiled->patch_call("call_ic_site_1", new_target_fn_ptr);
compiled->patch_call("call_ic_site_1", "optimized_fast_stub");
```
