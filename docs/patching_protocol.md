# Patchable Sites & Thread-Safe Concurrency Protocol

Dynamic languages rely on runtime code modification for Inline Caches (ICs) and speculative call sites. Brass guarantees thread-safe cross-modification without stopping mutator threads.

---

## 1. Patchable Primitives

### `patchable_const`
An immediate operand that can be atomically modified at runtime:
```mir
%type_id = patchable_const.i32 @ic_type_slot_42, 0
```
- Lowered to a `mov reg, imm` where the immediate offset is aligned to avoid crossing a cache-line boundary (64 bytes).
- Patching writes the new value with a single aligned atomic store (`std::atomic_ref` / `InterlockedExchange`).

### `patchable_call`
A direct call site whose target can be dynamically rewritten:
```mir
%res = patchable_call @call_ic_site_1, @default_stub(%arg0, %arg1)
```
- Lowered to a direct `call rel32` (5 bytes: `0xE8 <disp32>`).
- To guarantee that the 4-byte displacement does not cross a 64-byte cache line, the emitter inserts NOP padding before the `call` if necessary.
- Patching uses atomic 32-bit displacement store.

---

## 2. Concurrency Safety Rules on x64

1. **Alignment Guarantee**: All patchable operands are placed such that the modified bytes do not cross an instruction cache line (64 bytes).
2. **Atomic Store**: Modifications use atomic store operations.
3. **Instruction Cache Coherency**: On x64, hardware snooping maintains instruction-cache / data-cache coherence for aligned writes; execution on other worker threads remains consistent.
