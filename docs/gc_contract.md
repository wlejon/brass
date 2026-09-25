# Garbage Collection: Design and Contract

brass has one garbage collector, `brass::gc::Heap` (`include/brass/gc/heap.hpp`). Generated code of every tier, both interpreters, coroutine frames and the embedding API allocate from it. A host runtime (a JavaScript engine) can make it the one heap for its own objects: object layout and tracing are host-defined, and weak references, ephemerons, finalization and post-collection hooks are part of the heap.

This document covers the compiled-code contract first (how references live in registers, stack slots and stack maps), then the heap: its design, the entry points generated code calls, and the API a host runtime uses.

---

## 1. References in Compiled Code

### 1.1. Registers and Spill Slots
A `gcref` value lives in a general-purpose register while no GC point intervenes. The linear scan allocator (`LinearScanAllocator`) gives every `gcref` interval that spans a call or a `safepoint` a dedicated frame spill slot instead of a register (`get_hard_blocked_regs` blocks every physical register for it). Frame slots that hold `gcref` values are zeroed in the prologue, so a collection that runs before a slot is written sees a null reference rather than stale stack contents. Each call site and safepoint records the `gcref` virtual registers live across it (`live_gcrefs`), which become the stack map record for its return address.

### 1.2. Derived References
A *derived* reference is a `gcref` that may point inside an object rather than at its start: the result of `add` or `sub` on a `gcref`, or a `select` between values at least one of which is derived. Stack maps record every live `gcref` as an object start, so MIR restricts where derived references may live, and the verifier (`verify_derived_gcrefs`) enforces it:

- A derived reference is used only in the block that defines it.
- No GC point lies between its definition and any of its uses. GC points are `safepoint`, the coroutine operations, and every call except the few runtime helpers that cannot collect (`may_trigger_gc` in `include/brass/mir/gc_refs.hpp`).
- It is never a GC point's operand, a block argument, a return value, deopt state, the value written by a store, or the value operand of `write_barrier`.

LICM, CSE, GVN and PRE never hoist, merge or rematerialize derived references, since that could stretch a live range across a GC point. Frontends that need an interior pointer across a GC point keep the base `gcref` live and recompute the offset after it.

The interpreters are the exception: their frames keep every value they computed, so an interpreter frame can hold a derived reference across a call it does not outlive. The interpreters report their registers with `Tracer::visit_derived`, which keeps the containing object alive and preserves the offset when it moves.

### 1.3. Stack Maps
Stack maps are serialized as read-only `BSCM` metadata:

```
Header:   uint32 magic = 0x4D435342 ("BSCM"), uint32 version = 1, uint32 function_count
Function: uint32 name_length, name bytes, uint64 code_offset, uint32 code_size, uint32 record_count
Record:   uint32 instruction_offset (return IP - function base), uint32 frame_size,
          uint32 safepoint_id, uint32 root_count
Root:     int32 offset_from_fp, uint8 kind (0 FrameSlot, 1 CalleeSaved), uint8 reg_class,
          uint8 reg_code, uint8 reserved
```

Every module brass loads registers its maps with the code registry (`code_stack_maps.hpp`). `brass_stack_walk` walks frame-pointer chains from a (frame pointer, return address) pair: for each frame whose return address a map describes, it visits `fp + offset_from_fp` for each root, then moves to the caller frame. It checks alignment, monotonic growth and the thread's stack bounds, and stops after 1,024 frames.

### 1.4. Roots of a Collection
A collection's roots are:

- the slots registered with `Heap::add_root`, and every root source (`add_root_source`): each interpreter registers one for its frames, a host runtime registers its handle tables and module/global cells;
- the generated frames above the runtime call that started the collection (`brass_gc_alloc`, `brass_gc_safepoint`, `brass_gc_collect`, `brass_coro_create`), walked through the stack maps from the caller frame the call captured;
- generated frames below re-entered interpreter code, recorded by `NativeFramesScope` (`native_frames.hpp`) at every native-to-interpreter transition brass makes;
- the suspended coroutine frames the heap allocated (`Heap::coro_frames()`).

A runtime call that may collect, made from generated code that no stack map describes, is fatal (`brass: fatal: ... no stack maps are active`), because that frame's references could not be updated. Safepoints and `brass_gc_collect` in such code do nothing.

---

## 2. The Heap

### 2.1. Layout
Each heap reserves one contiguous address range and commits memory as it grows:

```
[ eden | survivor 0 | survivor 1 ][ mature blocks ........ ][ large-object pages ........ ]
  young generation                  old generation (never moves)
```

- **Young generation.** Eden (16 MB by default, never below `kMinEdenBytes` = 64 KB) and two survivor spaces (8 MB each). Allocation bumps a pointer through eden. A minor collection copies live young objects into the other survivor space, aging them; an object that has survived `tenure_age` minor collections (default 2), or that does not fit in the survivor space, is promoted to the old generation.
- **Adaptive tenuring.** When a minor collection overflows the survivor space, the collections after it promote every survivor directly (`promote-all`), for as long as at least half of the young data collected survives. A program building long-lived data then copies it once instead of twice; one whose survivors are short-lived returns to aging at the first collection where most of eden died. A full collection leaves the mode as the last minor collection set it.
- **Mature space.** 32 KB blocks of 256-byte lines, Immix-style. Objects of up to 8 KB (header included) are allocated into runs of free lines ("holes") of partly used blocks, or into whole free blocks. Objects are never moved once there. Side bitmaps per block record object starts and marks.
- **Large-object space.** Objects above the large-object threshold (1 MB by default, never below `kMinLargeObjectBytes` = 16 KB, and never above a quarter of eden) are allocated there directly; smaller ones start young, and those that are promoted above 8 KB land there too. Each gets its own run of 4 KB pages, freed as a unit. The threshold is high because a large object allocated old stays until a full collection: a growing array's discarded backing stores would otherwise keep every young object they named alive through their dirty cards until then.
- The mature and large reservations are 4 GB each by default (`HeapConfig`), address space only.

Every object has an 8-byte header before its payload: `{uint32 size; uint16 layout; uint8 gc_bits; uint8 host_bits}`. A reference is the payload address. `host_bits` are the host's; the collector copies them and never reads them.

### 2.2. Layouts
How the collector finds an object's references is its layout, registered once in a process-wide, append-only registry (`include/brass/gc/object.hpp`):

| Kind | References |
|------|------------|
| `Leaf` | none |
| `Mask` | payload word *i* when bit *i* of the mask is set; bit 63 covers every word from 63 on |
| `Words` | every payload word |
| `Custom` | whatever the layout's `TraceFn(payload, bytes, Tracer&)` visits |

`mask_layout(mask, type_tag)` interns the `Mask` layout that `brass_gc_alloc(size, mask, tag)` uses. A layout also carries a `type_tag` (the host's label) and a name used in verification messages.

A `Custom` layout may also give a `TraceRangeFn(payload, bytes, begin, end, Tracer&)` that visits only the slots in payload bytes `[begin, end)`. With it, a minor collection rescans just the dirty cards of a large old object, and a parallel full collection splits the scan of a large object between threads; without it, both scan the whole object. In a full collection the range that begins at offset 0 also visits whatever the full trace visits outside the payload (a shape's prototype, say). `Words` and `Mask` layouts are scanned by range natively.

A *word slot* holds a reference in its low 48 bits; the high 16 bits are a tag the collector preserves when it updates the slot, so NaN-boxed values work unchanged. `HeapConfig::reference_tags` limits which tags count as references: empty means every tag, otherwise exactly the tags listed. A raw gcref has tag 0, so the embedding C API lists 0 and the `HostValue` gcref tag; a host whose references are all NaN-boxed lists only its pointer tags, and then a raw pointer or a small integer in a slot, a register an interpreter visits conservatively, or a barrier's stored value is never taken for a reference. A word whose address lies outside the heap is not a reference to it.

A `TraceFn` runs during a collection. It must not allocate, must visit the same slots each time for the same object state, and reads other objects only through slot values the tracer has written back.

### 2.3. Collections
Collections are stop-the-world for the heap's thread only; other threads' heaps keep running.

- **Minor.** Roots and the old generation's dirty cards are scanned; live young objects are copied (Cheney-style) into the survivor space or promoted. The pause depends on the surviving young data and the number of dirty cards, not on the size of the old generation.
- **Full.** Three phases. First the young generation is evacuated as a minor collection would, except that every survivor is promoted. Then the old generation is marked in place (mature marks in side bitmaps, large objects by a header bit; each scanned object marks its lines). Marking moves and allocates nothing, so once the old generation holds `parallel_mark_bytes` (16 MB by default) it runs on several threads (`mark_threads`, default half the hardware threads, at most 8): each keeps its own mark stack, claims mark bits atomically, hands work to a shared pool when another thread is idle, and scans a large object in 16 KB slices that the threads share. The ephemeron fixpoint runs between rounds on the collecting thread. Finally weak slots are settled, hooks run, and the sweep keeps only marked starts, rebuilds each block's free lines into the free and recyclable block lists, releases excess free blocks back to the OS (a contiguous run in one call), and frees dead large objects. Afterwards the young generation is empty and every card is clean.
- **Triggers.** An allocation that finds eden full runs a minor collection. Old-generation growth since the last full collection (direct old allocation plus promotion) above `max(min_full_threshold_bytes, growth_factor × live old bytes)` runs a full collection. `request_collection(kind)` asks for one at the next allocation or safepoint; `collect(kind)` runs one now.
- **Stress.** `StressMode::Minor`, `Full` or `Alternate` (minor, every eighth full) collects at every allocation and safepoint. The fast allocation path is disabled so every allocation reaches the runtime.

### 2.4. Remembered Set: Cards and the Write Barrier
The old generation is covered by a card table of 512-byte cards. A store of a young reference into an old object must be remembered:

```cpp
heap.write_barrier(object, value);          // object: the object's payload address
heap.write_barrier_interior(address, value);// address: the slot written, anywhere inside the object
heap.remember(object);                      // a bulk copy: rescan the whole object
heap.remember_range(object, begin, end);    // a bulk copy into addresses [begin, end) of it
```

A mature object is remembered by the card of its start, and a minor collection scans every object starting in a dirty card. A large object is remembered per card: `write_barrier_interior` and `remember_range` dirty only the cards holding the slots written, and the minor collection rescans just those cards (through the layout's range scan). Its first card carries the difference between the two cases, since it is also where the object starts: `write_barrier`, `remember`, or a store whose slot the barrier cannot place dirty it as "rescan the whole object", while a slot store in its own bytes marks it as a range like any other card. The barrier's fast path is two range checks and a tag check; `write_barrier_interior` looks the object up only when the store needs remembering. A minor collection cleans each card it scans and re-dirties it when the slots it covers still name young objects afterwards.

Compiled code calls `brass_gc_write_barrier(obj, val)` for MIR's `write_barrier`, which by default is `brass_default_gc_write_barrier`: the interior barrier on the thread's heap. A MIR producer must emit `write_barrier` after every store of a `gcref` into an object that may be old. The interpreters apply the barrier to every 8-byte store themselves.

`WriteBarrierElimination` drops a barrier only when it can prove it does nothing: the stored value is not a pointer; the object was allocated since the last GC point by `brass_gc_alloc` with a constant payload of at most `kMaxAlwaysYoungPayloadBytes` (such an object is always allocated young, see `include/brass/gc/gc_limits.hpp`); or an earlier barrier since the last GC point covered the same object and the same value.

### 2.5. Non-Moving Objects
Old objects never move. `allocate(bytes, layout, kAllocOld)` pretenures an object; `kAllocPinned` does too and marks it pinned. Young objects move in every collection. An object whose address must stay fixed (one native code holds without a root) is allocated with one of these flags; an existing young object cannot be pinned in place.

### 2.6. Weak References, Ephemerons, Finalization
- `Tracer::visit_weak(slot, cleared)` in a `TraceFn` is a weak slot: it does not keep its target alive; once liveness is decided the slot is updated if the target survived, else set to the word `cleared`.
- `Tracer::visit_ephemeron(key, value, cleared_key, cleared_value)` is a WeakMap entry: the value is traced only once the key is known alive by another path, iterating to a fixpoint; when the key dies both slots are cleared. A value that refers only to its own key does not keep the entry alive.
- A minor collection decides only young targets; an old target of a weak slot or ephemeron key is decided by a full collection.
- `add_post_collection_hook(hook)` runs inside every collection once liveness is decided and weak slots are settled, before memory is reclaimed. There `survivor_of(address)` answers where an object is now, or 0 if it died (addresses the collection did not collect are returned unchanged). This is where a host updates address-keyed tables and releases native handles of dead objects. A hook must not allocate on the heap.
- `add_finalizer(object, fn, context)` calls `fn(context)` once, after the collection that finds the object dead, when the heap is consistent again; the callback may allocate.

### 2.7. Threads
A heap belongs to one thread at a time; nothing in it is locked on the allocation or collection paths. Each thread has a current heap (`Heap::current()`, bound with `HeapScope`), which the runtime entry points and the interpreters use. Many heaps can live in a process, each collecting independently; the layout registry is the only shared state (lock-protected registration, lock-free lookup). A reference from one heap into another is not traced. A heap may move to another thread as long as no two threads use it at once. Collection is stop-the-world per heap. A full collection may mark on helper threads it starts and joins before it returns; they run only the collector's own marking and the layouts' trace functions, never host code otherwise, so a `TraceFn` must be safe to call from a thread other than the heap's (it touches only the object it is given and the tracer).

### 2.8. Verification and Stress
- `set_verify(true)` (or `BRASS_GC_VERIFY=1`) checks the whole heap before and after every collection: every header, every reference slot names an object of the heap, and every old object that names a young one has a dirty card. A violation stops the process with a message naming the object, its layout and the slot.
- `set_poison(true)` (`BRASS_GC_POISON=1`) fills evacuated young space and freed old memory with `0xDB` bytes.
- `BRASS_GC_STRESS=minor|full|alternate` sets the stress mode of every heap created with `read_environment` (the default).
- `BRASS_GC_LOG=1` (`HeapConfig::log`) prints one line per collection to stderr: its kind and trigger (explicit, eden full, old-generation growth, requested, stress, exhausted), its pause split into phases (roots, cards, trace, weak, hooks, sweep), the bytes and objects copied, promoted and marked, the dirty cards scanned, and the old generation's size against its full-collection threshold; the heap's destructor prints totals.
- `BRASS_GC_MARK_THREADS=n` (`HeapConfig::mark_threads`) sets the number of threads a full collection marks with; 1 marks on the collecting thread alone.
- `is_valid_object`, `find_object` (interior address to object), `check_access` (the interpreters' bounds check), `for_each_object` and `stats()` (collection counts, pause totals and maxima, promoted and live bytes) support debugging and tests.

---

## 3. Runtime Entry Points

Generated code calls these C symbols (`include/brass/gc/runtime_gc.hpp`); each acts on the calling thread's current heap:

| Symbol | Behavior |
|--------|----------|
| `brass_gc_alloc(size, mask, tag)` | A zeroed object of layout `mask_layout(mask, tag)`. Fatal with no current heap. May collect. |
| `brass_gc_safepoint()` | Collects if a collection was requested or the heap is in stress mode. |
| `brass_gc_collect()` | A full collection. |
| `brass_gc_write_barrier(obj, val)` | Default: `write_barrier_interior` on the current heap. A host may register its own under this name. |
| `brass_gc_card_table_base()`, `brass_gc_heap_base()` | The card table and the old generation's base. |
| `brass_coro_create(fn, slots, mask)` | A coroutine frame, allocated on the current heap (or unmanaged memory when the thread has none). |

On MSVC the first three and `brass_coro_create` are assembly stubs (`src/gc/gc_msvc_{x64,arm64}.asm`) that pass their caller's frame pointer and return address to a C++ bridge; elsewhere the functions read them with `__builtin_frame_address` and `__builtin_return_address`.

`Heap::allocation_buffer()` exposes the eden bump region `{top, end}` so code can allocate young objects inline: write the header at `top`, advance `top`, zero the payload, and call the runtime when the object does not fit. Stress mode pulls `end` back to `top`. `Heap::bind_allocation_buffer(buffer)` moves the region into a host-owned `{top, end}` pair (a per-thread block its generated code already addresses), carrying the current values over; the heap then bumps and resets that pair, and `nullptr` moves it back.

---

## 4. Using the Heap from a Host Runtime

The API a JavaScript engine uses, all in `brass::gc`:

```cpp
HeapConfig config;                       // sizes, tenure_age, thresholds, reference_tags, modes
Heap heap(config);
HeapScope bind(heap);                    // the thread's current heap

// Object kinds: one layout per kind, registered once per process.
LayoutDescriptor d;
d.kind = LayoutKind::Custom;
d.trace = &trace_js_object;              // visits slots with tracer.visit / visit_weak / visit_ephemeron
d.type_tag = kJsObject;
d.name = "JSObject";
const LayoutId js_object = register_layout(d);

uintptr_t obj = heap.allocate(bytes, js_object);           // young
uintptr_t code = heap.allocate(bytes, kLeafLayout, kAllocPinned);  // never moves
heap.store(obj, 1, value);                                  // store + barrier

// Roots: C++ handles and module/global cells.
heap.add_root(&cell);                                       // one slot
auto id = heap.add_root_source(&visit_handles, &handles);  // a table, visited each collection

// Weak references, WeakMap, FinalizationRegistry, native handles.
heap.add_post_collection_hook([&](Heap& h, CollectionKind) { /* h.survivor_of(...) */ });
heap.add_finalizer(obj, &release_native, native_ptr);

heap.collect(CollectionKind::Minor);    // or Full; or let allocation trigger them
```

The runtime's own interpreters and compiled code share the heap when it is bound on the thread: `Interpreter` and `FastInterpreter` constructed with no heap use the thread's current heap, register their frames as a root source, and bind their heap while they run. Generated code allocates through `brass_gc_alloc` on the current heap. A host binds its heap before it constructs interpreters or runs generated code, so everything brass allocates on its behalf (interpreter state, coroutine frames) lands on the heap the host collects.

The embedding C API (`include/brass/embedding/brass_c_api.h`) wraps the same heap: `brass_heap_create`, `brass_heap_bind`, `brass_heap_allocate`, `brass_heap_collect`, `brass_heap_add_root`, `brass_heap_write_barrier`, `brass_heap_set_stress`.

---

## 5. Limits

- Collection is stop-the-world per heap, and marking is parallel but not concurrent or incremental; a full collection's pause grows with the live old generation.
- A minor collection copies on one thread, so its pause grows with the young data that survives it: a program allocating nothing but long-lived data pays for copying a whole eden at each minor collection.
- The old generation does not compact: fragmentation in the mature space is bounded by line and block reuse but not repaired by evacuation.
- A young object cannot be pinned in place; objects that must not move are allocated old (`kAllocOld` / `kAllocPinned`).
- Compiled code calls the write barrier and `brass_gc_alloc` out of line; neither is inlined into generated code.
