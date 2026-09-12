# Patchable Sites & Thread-Safe Concurrency Protocol

Dynamic languages rely on runtime code modification for Inline Caches (ICs), polymorphic call sites, and speculative deopt triggers. Brass guarantees thread-safe cross-modification without stopping mutator threads.

---

## 1. Patchable Primitives

### `patchable_const`
An immediate operand that can be atomically modified at runtime:
```mir
%type_id = patchable_const.i32 @ic_type_slot_42, 0
```
- Lowered to `mov reg, imm32` or `movabs reg, imm64`.
- The emitter inserts NOP padding so that the immediate bytes never cross a 64-byte instruction cache line boundary.
- Patching writes the new value with a single aligned atomic store (`std::atomic_ref` / `InterlockedExchange`).
- Supported in 32-bit (`patchable_const.i32`) and 64-bit (`patchable_const.i64`).

### `patchable_call`
A direct call site whose target can be dynamically rewritten:
```mir
%res = patchable_call.i64 @call_ic_site_1, @default_stub(%arg0, %arg1)
```
- Lowered to a direct `call rel32` (5 bytes: `0xE8 <disp32>`).
- To guarantee that the 4-byte displacement does not cross a 64-byte cache line, the emitter aligns the instruction if necessary.
- Patching calculates the relative displacement from the call site to the new target address and writes it with an aligned 32-bit atomic store.

---

## 2. Concurrency Safety Rules on x64

1. **Alignment Guarantee**: All patchable operands are aligned such that the modified bytes never cross an instruction cache line (64 bytes).
2. **Atomic Store**: Modifications use aligned atomic store operations (`std::atomic<uint32_t>` or `std::atomic<uint64_t>`).
3. **Hardware Snooping**: On x64, hardware cache-coherency logic automatically snoops stores and invalidates instruction prefetch pipelines across CPU cores; execution on other worker threads remains consistent without requiring explicit I-cache flushing instructions.

---

## 3. Runtime Patching APIs

### 3.1. Low-Level C Boundary
```c
// Patch a 32-bit constant site at raw code address
bool brass_patch_const32(void* code_addr, int32_t new_val);

// Patch a 64-bit constant site at raw code address
bool brass_patch_const64(void* code_addr, int64_t new_val);

// Patch a direct call site at raw call site address
bool brass_patch_call(void* call_site_addr, const void* new_target);
```

### 3.2. JIT Execution Engine (`JitExecutionEngine`)
```cpp
// Patch by symbolic site name in loaded module
jit.patch_const32("ic_type_slot_42", 123);
jit.patch_const64("bias_slot_1", 0x100000000LL);
jit.patch_call("call_ic_site_1", new_target_fn_ptr);
jit.patch_call("call_ic_site_1", "optimized_fast_stub");
```

### 3.3. Embedder Host Module (`CompiledModule`)
```cpp
compiled->patch_constant("ic_type_slot_42", 123);
compiled->patch_constant("bias_slot_1", int64_t(42));
compiled->patch_call("call_ic_site_1", new_target_fn_ptr);
compiled->patch_call("call_ic_site_1", "optimized_fast_stub");
```
