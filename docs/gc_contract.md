# Brass GC Contract & Moving Garbage Collectors

Brass provides first-class support for precise moving garbage collection. Unlike conventional runtimes that force unmanaged shadow-stacks or pin objects in memory, Brass uses a hybrid register and stack map architecture: managed object references (`gcref`) reside directly in general-purpose native CPU registers during intra-procedural computation (achieving zero-overhead fast paths), and the linear scan allocator precisely spills them to stack slots across calls and safepoint boundaries, generating compact binary stack maps for moving GC root relocation.

---

## 1. Linear Scan Register Allocation Contract

The linear scan register allocator (`LinearScanAllocator`) enforces a precise allocation contract for managed pointers:

### 1.1. Intra-Procedural & Leaf Computation
- **Direct Register Residency**: During intra-procedural computations and within leaf functions (functions that make no downstream calls and contain no explicit safepoints), `gcref` values reside directly in native general-purpose registers (GPRs: `RAX`, `RCX`, `RDX`, `RBX`, `RSI`, `RDI`, `R8`–`R15` on x86-64; `X0`–`X28` on AArch64).
- **Zero GC Overhead**: No stack spills, shadow-stack pushes, or frame-pointer adjustments are emitted for managed references along pure intra-procedural paths. Register-to-register arithmetic, comparisons, field offsets, and pointer dereferences execute at full native hardware speed.

### 1.2. Call Sites & Safepoint Boundaries
- **Spanning Intervals**: During liveness analysis, the allocator determines whether a live range interval spans a function call (`inst->is_call()`) or an explicit safepoint (`inst->opcode == LirOpcode::Safepoint`).
- **Hard-Blocked Register Constraint**: In `LinearScanAllocator::get_hard_blocked_regs`, if an interval holds a `gcref` and spans a call (`interval.vreg.is_gcref && interval.spans_call`), **all** physical registers (both caller-saved and callee-saved) are marked blocked.
- **Dedicated Frame Spill**: In `LinearScanAllocator::allocate_unhandled`, call-spanning `gcref` intervals are immediately assigned a dedicated frame spill slot:
  ```cpp
  if (current->vreg.is_gcref && current->spans_call) {
      current->assigned_spill_slot = allocate_spill_slot(true, current->vreg.size);
      current->assigned_preg = PReg{};
      continue;
  }
  ```
- **Spill Slot Zero-Initialization**: Frame slots holding `gcref` values (`spill_slot_is_gcref`) are zero-initialized in function prologues. This guarantees that if a GC cycle is triggered before a slot is populated, the stack walker observes a null pointer (`0`) rather than uninitialized stack debris.
- **Site Recording**: For each call site and safepoint instruction, `inst->live_gcrefs` records every active `gcref` virtual register whose live interval covers the site, mapping directly to stack spill offsets relative to `RBP`.

### 1.3. Derived References
A *derived* reference is a `gcref` that may point inside an object rather than at its start: the result of `add` or `sub` on a `gcref`, or a `select` between values at least one of which is derived. Stack maps record every live `gcref` as an object root, and a collector that relocates an interior pointer as if it were an object start corrupts the heap. MIR therefore restricts where derived references may live, and the verifier (`verify_derived_gcrefs`) enforces it:

- A derived reference is used only in the block that defines it.
- No GC point lies between its definition and any of its uses. GC points are `safepoint`, the coroutine operations, and every call except the few runtime helpers that cannot collect (see `may_trigger_gc` in `include/brass/mir/gc_refs.hpp`).
- It is never a GC point's operand, a block argument, a return value, deopt state, the value written by a store, or the value operand of `write_barrier`.

Optimizations keep this rule by not treating derived references as ordinary pure values: LICM, CSE, GVN and PRE never hoist, merge or rematerialize them, because doing so could stretch a live range across a safepoint or out of its block. Frontends that need an interior pointer across a GC point keep the base `gcref` live and recompute the offset after the GC point.

---

## 2. Binary Stack Map Format (`BSCM`) & Return Address Lookup

Stack map tables are serialized into compact, read-only binary metadata (`.rdata` / `.rodata`) conforming to the `BSCM` specification.

### 2.1. Binary Encoding Specification

```
+-------------------------------------------------------------+
| Header:                                                     |
|   uint32_t magic = 0x4D435342 ("BSCM" in Little-Endian)     |
|   uint32_t version = 1                                      |
|   uint32_t function_count                                   |
+-------------------------------------------------------------+
| For each function (1 .. function_count):                    |
|   uint32_t name_length                                      |
|   uint8_t  function_name[name_length]                       |
|   uint64_t code_offset (relative offset in .text section)    |
|   uint32_t code_size   (total function size in bytes)       |
|   uint32_t record_count                                     |
|   +-------------------------------------------------------+ |
|   | For each stack map record (1 .. record_count):        | |
|   |   uint32_t instruction_offset (return IP - fn_base)   | |
|   |   uint32_t frame_size        (total frame in bytes)   | |
|   |   uint32_t safepoint_id                               | |
|   |   uint32_t root_count                                 | |
|   |   +-------------------------------------------------+ | |
|   |   | For each root location (1 .. root_count):       | | |
|   |   |   int32_t offset_from_rbp (e.g. -24)            | | |
|   |   |   uint8_t kind (0 = FrameSlot, 1 = CalleeSaved) | | |
|   |   |   uint8_t reg_class (RegClass::GPR)             | | |
|   |   |   uint8_t reg_code                              | | |
|   |   |   uint8_t reserved (0)                          | | |
|   |   +-------------------------------------------------+ | |
|   +-------------------------------------------------------+ |
+-------------------------------------------------------------+
```

### 2.2. Return Address Lookup
When a thread halts at a safepoint or triggers allocation:
1. The runtime captures the caller return instruction pointer (`return_ip`) from `[cur_rbp + 8]`.
2. `ModuleStackMap::find_function_by_ip(return_ip)` locates the function whose address range covers the IP:
   $$\text{fn.function\_address} \le \text{return\_ip} < \text{fn.function\_address} + \text{fn.code\_size}$$
3. `FunctionStackMap::find_record_by_ip(return_ip)` computes the relative return offset:
   $$\text{instruction\_offset} = \text{return\_ip} - \text{fn.function\_address}$$
   and retrieves the matching `StackMapRecord`.

### 2.3. Runtime Stack Walker & Visitor Pattern

The stack walker (`brass_stack_walk`) traverses active native call frames using standard RBP frame pointers:

```cpp
typedef void (*brass_root_visitor_fn)(void** root_slot, void* user_data);

size_t brass_stack_walk(
    uintptr_t top_rbp,
    uintptr_t top_return_ip,
    const ModuleStackMap& stack_maps,
    brass_root_visitor_fn visitor,
    void* user_data
);
```

#### Traversal Algorithm:
1. **Frame Pointer Chain**: Starts at `top_rbp` and `top_return_ip`. Each frame unwinds to the caller frame via:
   ```cpp
   uintptr_t next_rbp = *reinterpret_cast<const uintptr_t*>(cur_rbp);
   uintptr_t next_return_ip = *reinterpret_cast<const uintptr_t*>(cur_rbp + 8);
   ```
2. **Safety Validation**:
   - Ensures 8-byte alignment: `(cur_rbp % 8) == 0`.
   - Ensures monotonic stack growth: `next_rbp > cur_rbp`.
   - Bounds verification against platform stack limits (e.g., Windows `NtCurrentTeb()->StackLimit` and `StackBase`).
   - Caps traversal depth at 1,024 frames to prevent circular loops on corrupted stacks.
3. **Root Pointer Relocation**:
   - For every recognized frame, the walker iterates through `rec->roots`.
   - Computes mutable root address:
     $$\text{slot\_addr} = \text{cur\_rbp} + \text{root\_loc.offset\_from\_rbp}$$
   - Invokes `visitor(reinterpret_cast<void**>(slot_addr), user_data)`.
   - Moving collectors dereference `*root_slot`, copy the object to the destination generation, store a forwarding pointer in the from-space header, and write the updated to-space pointer back into `*root_slot`.

---

## 3. Generational GC & Memory Layout

Brass implements a generational moving garbage collector (`GenerationalGC`) optimized for young object churn and low-pause collection.

### 3.1. Heap Generation Layout

```
+------------------+-------------------+-------------------+-------------------------+
|     NURSERY      |   SURVIVOR-FROM   |    SURVIVOR-TO    |         TENURED         |
| (Default 512 KB) | (Default 256 KB)  | (Default 256 KB)  |     (Default 4 MB)      |
|  Bump Allocation | Minor Scavenge In | Minor Scavenge Out| Tenured / Promoted Objs |
+------------------+-------------------+-------------------+-------------------------+
```

- **Nursery Space**: High-throughput bump-pointer allocation area where new objects are initially created.
- **Survivor Semi-Spaces**: Two alternating spaces (`survivor_from_` and `survivor_to_`). Live objects surviving a minor collection are evacuated into survivor space with an incremented age counter.
- **Tenured Space**: Objects that survive past the tenuring threshold (default age 2) or when survivor space is saturated are promoted to the tenured generation.
- **Object Header (`GenGcHeader`)**: 16 bytes prepended to each object payload:
  - `size`: 32-bit payload size in bytes.
  - `pointer_mask`: 64-bit bitmap indicating which 8-byte fields contain managed references.
  - `type_tag`: 16-bit type identifier.
  - `age`: 8-bit survivor generation counter.
  - `flags`: 8-bit GC status flags.
  - `forwarding_address`: 64-bit relocated target pointer (active during evacuation).

### 3.2. Scavenge Source Filtering (`is_scavenge_source`)

During minor collection (scavenging), the collector only evacuates objects that reside in young source spaces:

```cpp
bool GenerationalGC::is_scavenge_source(uintptr_t addr) const noexcept {
    return is_in_nursery(addr) || is_in_survivor_from(addr);
}
```

#### Evacuation Safety Invariant:
When `evacuate_young_object(obj_addr)` is called on a pointer encountered during root walking or card table scanning:
1. **Source Check**: If `!is_scavenge_source(obj_addr)`, the object is already in `survivor_to` (already evacuated in the current minor cycle) or in `tenured`. The collector returns `obj_addr` immediately without copying.
2. **Forwarding Check**: If `old_hdr->forwarding_address != 0`, the object was already evacuated earlier in the current collection cycle; its forwarded address is returned.
3. **Evacuation & Aging**: Otherwise, the collector allocates destination space (in survivor space or tenured space if `age + 1 >= tenuring_threshold`), copies the object header and payload, sets `old_hdr->forwarding_address = new_payload`, and returns `new_payload`.

This filtering strictly eliminates object duplication bugs, prevents premature re-promotion, and guarantees forwarding pointer integrity.

### 3.3. Card Table Tracking (`CardTable`)

To collect young generations without scanning the entire tenured generation, Brass maintains a byte card table:
- **Card Granularity**: Each card byte covers 512 bytes of heap space:
  $$\text{card\_index} = \frac{\text{object\_address} - \text{heap\_base}}{512}$$
- **Write Barrier**: Whenever a managed pointer is written into an object field, a write barrier is executed:
  ```cpp
  void GenerationalGC::write_barrier(uintptr_t obj_addr, uintptr_t val) noexcept {
      if (is_old(obj_addr) && is_young(val)) {
          card_table_.mark_dirty(card_table_.card_index(obj_addr));
      }
  }
  ```
- **Barrier Elimination**: `WriteBarrierElimination` drops a barrier only when it can prove the barrier would do nothing: the stored value is not a pointer; the object was allocated since the last GC point by `brass_gc_alloc` with a constant payload of at most `kMaxAlwaysYoungPayloadBytes` (objects above half the nursery go straight to the tenured generation, and the nursery is never smaller than `kMinNurseryBytes`, see `include/brass/gc/gc_limits.hpp`); or an earlier barrier since the last GC point covered the same object *and the same value*. A barrier for a different value on the same object is never redundant.
- **Minor Collection Scanning**: Minor collections inspect only dirty cards in the card table. Pointers inside tenured objects residing on dirty cards are scavenged, and the card is cleared (`0`) upon completion.

### 3.4. Thread-Local Allocation Buffers (TLAB) & Synchronization

To eliminate lock contention on multi-core allocations, mutator threads allocate from thread-local buffers (`ThreadLocalAllocBuffer`):

1. **Lock-Free Fast Path**: A thread bump-allocates inside its local TLAB:
   ```cpp
   if (top_ + size <= end_) {
       uintptr_t res = top_;
       top_ += size;
       return res;
   }
   ```
2. **Synchronized Refill**: When local space is exhausted, `tlab.refill()` synchronizes with the heap via `gc->allocate_tlab`:
   - Acquires `gc_mutex_` (`std::recursive_mutex`).
   - Verifies remaining free space in the nursery/semispace.
   - Carves out a contiguous, 8-byte-aligned chunk (default 64 KB to 256 KB).
   - Updates the global allocation offset with release semantics (`std::memory_order_release`).
3. **Heap Exhaustion & Safe Safepoints**: If free space is insufficient for the minimum chunk size, `allocate_tlab` captures the caller's frame (`top_rbp`, `top_return_ip`), pauses mutators, walks roots across all registered TLABs and stacks, and triggers garbage collection before fulfilling the refill request.
4. **Retirement**: On thread termination or collection, unallocated bytes between `top` and `end` are returned to the shared allocator via `gc->retire_tlab(top, end)`.

---

## 4. Performance vs Shadow Stack Model

Traditional dynamic language runtimes maintain an explicit shadow-stack in software, pushing and popping frame structs around every call site. This incurs continuous CPU cache and memory traffic.

Brass achieves **3.9x+ faster** execution on GC-heavy call patterns by:
- Storing live GC references in native CPU registers and standard stack spill slots.
- Generating 100% out-of-band static stack map metadata in read-only sections.
- Imposing **zero runtime overhead** on normal mutator fast paths when GC does not occur.
