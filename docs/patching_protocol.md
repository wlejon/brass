# Patchable Sites & Thread-Safe Concurrency Protocol

Dynamic languages rely on runtime code modification for Inline Caches (ICs) and speculative call sites. Brass guarantees thread-safe cross-modification without stopping mutator threads.

---

## 1. Patchable Primitives

### `patchable_const`
An immediate operand that can be atomically modified at runtime:
```mir
%type_id = patchable_const.i32 @ic_type_slot_42, 0
```
- Lowered to a `mov reg, imm` where the immediate offset is aligned to avoid crossing a 64-byte instruction cache line.
- Patching writes the new value with a single aligned atomic store (`std::atomic_ref` / `InterlockedExchange`).
- Supported in 32-bit (`patchable_const.i32`) and 64-bit (`patchable_const.i64` via `movabs`).

### `patchable_call`
A direct call site whose target can be dynamically rewritten:
```mir
%res = patchable_call.i64 @call_ic_site_1, @default_stub(%arg0, %arg1)
```
- Lowered to a direct `call rel32` (5 bytes: `0xE8 <disp32>`).
- To guarantee that the 4-byte displacement does not cross a 64-byte cache line, the emitter inserts NOP padding before the `call` if necessary.
- Patching uses an atomic 32-bit displacement store.

---

## 2. Concurrency Safety Rules on x64

1. **Alignment Guarantee**: All patchable operands are placed such that the modified bytes never cross an instruction cache line (64 bytes).
2. **Atomic Store**: Modifications use atomic store operations (`std::atomic<uint32_t>` or `std::atomic<uint64_t>`).
3. **Hardware Snooping**: On x64, hardware snooping maintains instruction-cache and data-cache coherence for aligned writes; execution on other worker threads remains consistent without requiring explicit I-cache flushes.

---

## 3. Runtime Patching APIs

```cpp
// Patch a 32-bit constant site by symbol name
bool brass_patch_const32(void* code_site, int32_t new_val);

// Patch a 64-bit constant site by symbol name
bool brass_patch_const64(void* code_site, int64_t new_val);

// Patch a direct call site
bool brass_patch_call(void* code_site, const void* new_target);

// High-level JIT execution engine helper
bool jit.patch_const32("ic_type_slot_42", 123);
bool jit.patch_call("call_ic_site_1", new_target_fn_ptr);
```
