# Garbage Collection: Design and Contract

brass has one garbage collector, `brass::gc::Heap` (`include/brass/gc/heap.hpp`). Generated code of every tier, both interpreters, coroutine frames and the embedding API allocate from it. A host runtime (a JavaScript engine) can make it the one heap for its own objects: object layout and tracing are host-defined, and weak references, ephemerons, finalization and post-collection hooks are part of the heap.

This document covers the compiled-code contract first (how references live in registers, stack slots and stack maps), then the heap: its design, the entry points generated code calls, and the API a host runtime uses.

---

## 1. References in Compiled Code

### 1.1. Registers and Spill Slots
A `gcref` value lives in a general-purpose register while no GC point intervenes. The linear scan allocator (`LinearScanAllocator`) gives every `gcref` interval that spans a call or a `safepoint` a dedicated frame spill slot instead of a register (`get_hard_blocked_regs` blocks every physical register for it). Frame slots that hold `gcref` values are zeroed in the prologue, so a collection that runs before a slot is written sees a null reference rather than stale stack contents. Each call site and safepoint records the `gcref` virtual registers live across it (`live_gcrefs`), which become the stack map record for its return address.

A `tagged` value follows the same rules as a `gcref`: it gets a zeroed spill slot across GC points, and it is recorded at every GC point where it is live. A tagged value is a 64-bit word that is a reference only when its high 16 bits are one of the heap's reference tags (`HeapConfig::reference_tags`). In that case its low 48 bits are the object's address. The collector visits it through `Tracer::visit`, which checks the tag, leaves a non-reference word alone, and rewrites only the low 48 bits of a reference, so the tag survives a move. `alloca.tagged` memory is zeroed in the prologue, and each of its words is a root at every GC point in the function. A value is a root only while it is live, so an object whose last use has passed can be collected even if the code is still using a raw pointer it handed out. `keep_alive %v` extends `%v`'s liveness to the point where it appears, without emitting code. Tagged values have the same roots in the interpreters (register types and tagged allocas), the baseline JIT (`tagged_slots`), tier 2 (register allocation, callee-saved spills, deopt state) and OSR (tagged live-ins).

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
          uint8 reg_code, uint8 value (0 GcRef, 1 Tagged)
```

The decoder rejects any other `value` byte. Records are sorted by instruction offset and looked up by binary search.

Every module brass loads registers its maps with the code registry (`code_stack_maps.hpp`). `brass_stack_walk` walks frame-pointer chains from a (frame pointer, return address) pair: for each frame whose return address a map describes, it visits `fp + offset_from_fp` for each root, then moves to the caller frame. Generated code always keeps the chain. Where it leads into compiled (C++) code, which need not keep one, the walk steps through the compiled frames by their unwind information until it returns to generated code: the Win64 unwind data on Windows x64, and on the other x86-64 and AArch64 hosts their DWARF CFI (`native_unwind.hpp`: the `.eh_frame` of every loaded image and of generated code, which brass registers as it loads it; libunwind on Apple platforms). On Windows ARM64 it follows the chain through compiled frames too. It checks alignment, monotonic growth and the thread's stack bounds, and walks the whole chain with no frame limit.

### 1.4. Roots of a Collection
A collection's roots are:

- the slots registered with `Heap::add_root`, and every root source (`add_root_source`): each interpreter registers one for its frames, a host runtime registers its handle tables and module/global cells;
- the generated frames above the runtime call that started the collection (`brass_gc_alloc`, `brass_gc_safepoint`, `brass_gc_collect`, `brass_coro_create`, `brass_coro_create_body`), walked through the stack maps from the caller frame the call captured;
- with `HeapConfig::walk_stack_on_host_collection`, a collection started from host code (`Heap::allocate` or `Heap::collect` called by C++ with no captured generated frame) walks the native stack from its own frame (`brass_append_stack_roots_from_here`), visiting every generated frame a stack map describes. It steps through the C++ frames by their unwind information as above, so C++ code may omit frame pointers (on Windows ARM64, which MSVC never does there, it follows the chain);
- generated frames below re-entered interpreter code, recorded by `NativeFramesScope` (`native_frames.hpp`) at every native-to-interpreter transition brass makes;
- the suspended coroutine frames the heap allocated (`Heap::coro_frames()`).

A root source is called with the collection's tracer, whose `purpose()` says which pass it serves: `Minor` (a minor collection, and also the first phase of a full one, which evacuates the young generation exactly as a minor does but promotes every survivor), `Full` (the old generation's marking) or `Verify` (`set_verify`'s walk). A `Minor` pass needs only the slots that may name a young object: an old or non-reference target is not moved or decided by it. So a source whose table is large and rarely written may keep a remembered set of its own (the slots written with a young reference since the last minor collection) and visit only those in a `Minor` pass, under these rules:

- every slot it skips holds no young reference at that moment; a store of a young reference into one must put the slot on the set before the next collection;
- after visiting a slot, `Tracer::is_young(slot)` says whether it still names a young object (a survivor that was not promoted); the slot stays on the set while that is true and may leave it once it is false. After a full collection's first phase nothing is young, so the set empties;
- a `Full` or `Verify` pass visits every slot, as a source without a set does;
- under `verifying()` (`set_verify`, `BRASS_GC_VERIFY=1`), the `Minor` pass also reads the slots it skips and stops the process if one names a young object: a store that missed the set is then a hard error rather than a dangling reference after the copy.

Weak slots (`visit_weak`) are settled after every root has been visited, so a source asks `is_young` of its weak slots in a post-collection hook, where they hold their final words.

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
- **Adaptive eden.** A minor collection's copying costs about the bytes that survive it, so after each one the eden the next may fill (its limit) is set to `minor_copy_budget_bytes` (1 MB, about 0.5 ms of copying) over the survival rate (bytes copied, survivor-space objects included, per byte of eden collected), clamped to [1 MB, eden]. The rate is the larger of what this collection saw and half the previous rate (collections of under 64 KB of eden leave it alone), so a burst of survivors bounds the very next collection, and once survival falls the limit doubles back per collection. Eden only shrinks for survivors the program keeps (a build-up), whose copying is proportional to eden. Young data that lives a while and then dies (a working set) reads as survival too whenever eden is shorter than its life, the more so the smaller eden is, and a smaller eden only adds collections and promotes what the whole eden would let die young. Two tests keep it apart. At the whole eden, a collection counts its survival only when it overflowed the survivor space or at least a quarter of what it copied was promoted out of it. While eden is shrunk, survivors age through the survivor space even after an overflow (`promote-all` waits until the whole eden is back), and each collection sees how many of the survivors the last one left there have died: a build-up keeps nearly all of them, a working set loses the share of its lifetime that one shrunk eden is. Averaged over the last several collections (each weighing 1/8, and at least 4 of them), that gives the survivors' lifetime in eden bytes; under four whole edens, eden goes back whole at once. (Four, not one: a container promoted at the shrunk eden, a growing array's store say, keeps the young objects it names alive through its dirty card after it has died, so deaths read low; the whole eden then judges for itself.) The whole eden's first 4 collections after such a restore cannot shrink it, since the shrunk eden's promotions still keep young objects alive; if the next one shrinks it anyway, the test misread a build-up and its next verdict averages twice as many collections (up to 64), so a build-up that only looks like a working set at times does not swing eden back and forth. A full collection that finds less than half of the old generation's growth since the last one alive also restores the whole eden and leaves `promote-all`. While shrunk, a build-up's collection copies about the budget out of eden and the last one's survivors again; the first collection of a burst still runs at the limit the collections before it left, and the one after it copies the survivor space that burst filled. Allocation, inline included, stops at the limit through the allocation buffer's `end`; an empty eden still takes one young object of any size. `HeapConfig::adaptive_eden = false` or `BRASS_GC_FIXED_EDEN=1` pins the limit at the whole eden.
- **Adaptive tenuring.** When a minor collection overflows the survivor space, the collections after it promote every survivor directly (`promote-all`), for as long as at least half of the young data collected survives. A program building long-lived data then copies it once instead of twice; one whose survivors are short-lived returns to aging at the first collection where most of eden died. While the adaptive eden is shrunk the mode is held off (survivors age), and a full collection leaves it as the last minor collection set it unless it restores the whole eden.
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
- **Full.** Three phases. First the young generation is evacuated as a minor collection would, except that every survivor is promoted. Then the old generation is marked in place (mature marks in side bitmaps, large objects by a header bit; each scanned object marks its lines). Marking moves and allocates nothing, so once the old generation holds `parallel_mark_bytes` (16 MB by default) it runs on several threads (`mark_threads`, default half the hardware threads, at most 8): each keeps its own mark stack, claims mark bits atomically, hands work to a shared pool when another thread is idle, and scans a large object in 16 KB slices that the threads share. The ephemeron fixpoint runs between rounds on the collecting thread. Finally weak slots are settled and hooks run. The sweep (2.11) keeps only marked starts, rebuilds each block's free lines into the free and recyclable block lists, releases excess free blocks back to the OS (a contiguous run in one call), and frees dead large objects; by default it is left pending after the collection and done lazily. Afterwards the young generation is empty and every card is clean. A full collection runs as a marking cycle (2.10) whose steps follow one another with nothing in between.
- **Triggers.** An allocation that finds eden full (up to its adaptive limit) runs a minor collection. Old-generation growth since the last full collection (direct old allocation plus promotion) above `max(min_full_threshold_bytes, growth_factor × live old bytes)` runs a full collection. `request_collection(kind)` asks for one at the next allocation or safepoint; `collect(kind)` runs one now. `pressure()` says how near each trigger is (eden used of its capacity, the adaptive limit; old-generation growth of its threshold), for a host that collects ahead of them at a moment a pause costs nothing, between a program's frames say. The finalizers of the objects a collection found dead run once it has finished, outside its pause; `stats()` reports how many ran and how long they took (`last_finalizers`, `last_finalizer_ns`), and the log prints them.
- **Stress.** `StressMode::Minor`, `Full` or `Alternate` (minor, every eighth full) collects at every allocation and safepoint. `StressMode::Incremental` keeps a marking cycle (2.10) open with the program running inside it: each allocation and safepoint starts a cycle when none is open, and otherwise runs a minor collection (every fourth step) or a mark slice with no budget, finishing the cycle after eight slices. The fast allocation path is disabled so every allocation reaches the runtime.

### 2.4. Remembered Set: Cards and the Write Barrier
The old generation is covered by a card table of 512-byte cards. A store of a young reference into an old object must be remembered:

```cpp
heap.write_barrier(object, value);          // object: the object's payload address
heap.write_barrier_interior(address, value);// address: the slot written, anywhere inside the object
heap.remember(object);                      // a bulk copy: rescan the whole object
heap.remember_range(object, begin, end);    // a bulk copy into addresses [begin, end) of it
```

A mature object is remembered by the card of its start, and a minor collection scans every object starting in a dirty card. A large object is remembered per card: `write_barrier_interior` and `remember_range` dirty only the cards holding the slots written, and the minor collection rescans just those cards (through the layout's range scan). Its first card carries the difference between the two cases, since it is also where the object starts: `write_barrier`, `remember`, or a store whose slot the barrier cannot place dirty it as "rescan the whole object", while a slot store in its own bytes marks it as a range like any other card. The barrier's fast path is two range checks and a tag check; `write_barrier_interior` looks the object up only when the store needs remembering. A minor collection cleans each card it scans and re-dirties it when the slots it covers still name young objects afterwards.

The value's range check is against `barrier_value_base()` / `barrier_value_span()`: the young generation while no marking cycle is open, the whole heap while one is (2.10). Only that range changes, so the barrier's instructions are the same either way. Outside a cycle an old value is filtered out as it always was; inside one, a store of an old reference into an old object takes the slow path and shades the value (marks it and pushes it grey) instead of dirtying a card, and a young value still dirties the card. Shading allocates nothing and collects nothing. The range changes only at a GC point (the start and the finish of a cycle), so compiled code between two GC points always sees one range. A host that writes a reference into a place the barrier does not cover but a traced object reads through (a non-moving side table the object's trace function visits, say) calls `shade_reference(word)`, which is a no-op outside a cycle.

Compiled code calls `brass_gc_write_barrier(obj, val)` for MIR's `write_barrier`, which by default is `brass_default_gc_write_barrier`: the interior barrier on the thread's heap. A MIR producer must emit `write_barrier` after every store of a `gcref` into an object that may be old. The interpreters apply the barrier to every 8-byte store themselves.

`WriteBarrierElimination` drops a barrier only when it can prove it does nothing: the stored value is not a pointer; the object was allocated since the last GC point by `brass_gc_alloc` with a constant payload of at most `kMaxAlwaysYoungPayloadBytes` (such an object is always allocated young, see `include/brass/gc/gc_limits.hpp`); or an earlier barrier since the last GC point covered the same object and the same value. Each rule holds inside a marking cycle too: a non-pointer is never shaded; a young object's stores are not barrier stores for the marker (a minor collection or the finish reaches the young object and shades what it names); and a duplicate barrier would shade a value already shaded, since the cycle cannot open or close between the two without a GC point. The second rule depends on the young allocation: an allocator that pretenured such an object would have to allocate it marked and with its card dirty, as an old allocation inside a cycle is (2.10).

### 2.5. Non-Moving Objects
Old objects never move. `allocate(bytes, layout, kAllocOld)` pretenures an object; `kAllocPinned` does too and marks it pinned. Young objects move in every collection. An object whose address must stay fixed (one native code holds without a root) is allocated with one of these flags; an existing young object cannot be pinned in place.

### 2.6. Weak References, Ephemerons, Finalization
- `Tracer::visit_weak(slot, cleared)` in a `TraceFn` is a weak slot: it does not keep its target alive; once liveness is decided the slot is updated if the target survived, else set to the word `cleared`.
- `Tracer::visit_ephemeron(key, value, cleared_key, cleared_value)` is a WeakMap entry: the value is traced only once the key is known alive by another path, iterating to a fixpoint; when the key dies both slots are cleared. A value that refers only to its own key does not keep the entry alive.
- A minor collection decides only young targets; an old target of a weak slot or ephemeron key is decided by a full collection. The tracer's range filter drops a weak slot naming an old target before it is recorded, so the weak work of a minor scales with the young targets it meets, not with every weak slot it scans. (A host root source that walks its whole table of weak slots on every collection still costs its table's size; that walk is the host's.) A minor collection inside a marking cycle hands the weak slots and ephemerons of old objects that name old targets to the cycle, which settles them at its finish, since a slot rewritten behind a card may not be met by the marker again.
- `add_post_collection_hook(hook)` runs inside every collection once liveness is decided and weak slots are settled, before memory is reclaimed. There `survivor_of(address)` answers where an object is now, or 0 if it died (addresses the collection did not collect are returned unchanged). This is where a host updates address-keyed tables and releases native handles of dead objects. A hook must not allocate on the heap.
- `add_finalizer(object, fn, context)` calls `fn(context)` once, after the collection that finds the object dead, when the heap is consistent again; the callback may allocate. `add_finalizer(object, fn, a, b)` calls `fn(a, b)`: a host that needs two words per finalizer (a payload and its destructor) registers them without allocating a record of its own. Registering is a push onto a list; a collection compacts the young list in place (survivors that were promoted move to the old list, the dead to the pending list), and a full one the old list too, so it allocates nothing per finalizer.
- Finalizers run in the order the collection found their objects dead. A finalizer that throws does not stop the ones after it. A collection a finalizer causes (its allocation fills eden, say) runs as usual, but the finalizers it finds dead are queued, not run inside the one running: they run in the same drain, after the current batch, so finalizers never nest. `last_finalizers` counts both.

### 2.7. Threads
A heap belongs to one thread at a time; nothing in it is locked on the allocation or collection paths. Each thread has a current heap (`Heap::current()`, bound with `HeapScope`), which the runtime entry points and the interpreters use. Many heaps can live in a process, each collecting independently; the layout registry is the only shared state (lock-protected registration, lock-free lookup). A reference from one heap into another is not traced. A heap may move to another thread as long as no two threads use it at once. Collection is stop-the-world per heap. A full collection may mark on helper threads it starts and joins before it returns; they run only the collector's own marking and the layouts' trace functions, never host code otherwise, so a `TraceFn` must be safe to call from a thread other than the heap's (it touches only the object it is given and the tracer).

### 2.8. Verification and Stress
- `set_verify(true)` (or `BRASS_GC_VERIFY=1`) checks the whole heap before and after every collection: every header, every reference slot names an object of the heap, and every old object that names a young one has a dirty card. A violation stops the process with a message naming the object, its layout and the slot. At a marking cycle's finish, once its marking is done, it also marks the heap afresh from the roots into a side set, stop-the-world, and stops the process if that reaches an object the cycle left unmarked (one the sweep would free while it is live).
- `set_poison(true)` (`BRASS_GC_POISON=1`) fills evacuated young space and freed old memory with `0xDB` bytes.
- `BRASS_GC_STRESS=minor|full|alternate|incremental` sets the stress mode of every heap created with `read_environment` (the default).
- `BRASS_GC_LOG=1` (`HeapConfig::log`) prints one line per collection to stderr: its kind and trigger (explicit, eden full, old-generation growth, requested, stress, exhausted, incremental for a cycle started with `start_marking`), its pause split into phases (roots, cards, trace, weak, hooks, sweep), the bytes and objects copied, promoted and marked, the dirty cards scanned, the old generation's size against its full-collection threshold, and the eden limit the collection ran at and the one it left (`eden limit 16.0M -> 3.0M`); a full collection's line ends with its cycle's pauses (`start 0.5 (sweep 0.1) mark 14.5/1 finish 1.0`: the start and the part of it spent completing the last cycle's pending sweep, the mark slices together and how many, the finish). Under lazy sweep (2.11) the full line's `sweep` phase is only the sweep's setup at the finish, and the deferred work gets a line of its own when it completes: `sweep lazy 10.1 ms in 12 slices (4.0 ms, max 0.5) | allocator 5.3 ms/4000 blocks | completed 0.8 ms | blocks 51445 swept 25614 freed | large 3 freed 1.5M | released 64.0M` (all of it, then the part done by `sweep_slice` calls, their count, total and longest, the part the allocator did and the blocks it swept, the part done by completion at the next cycle's start or a reservation's end, then the blocks swept and freed, dead large objects freed and their bytes, and bytes given back to the OS). The heap's destructor prints totals.
- `BRASS_GC_EAGER_SWEEP=1` (`HeapConfig::lazy_sweep = false`) sweeps inside the finish's pause, as before lazy sweep, for comparing against it.
- `BRASS_GC_FIXED_EDEN=1` (`HeapConfig::adaptive_eden = false`) runs every minor collection at a full eden, for comparing against the adaptive limit.
- `BRASS_GC_MARK_THREADS=n` (`HeapConfig::mark_threads`) sets the number of threads a full collection marks with; 1 marks on the collecting thread alone.
- `BRASS_GC_MARK_POOL=0` (`HeapConfig::mark_thread_pool = false`) has each parallel drain create its marker threads and join them, as before the heap kept them parked between drains; a marking cycle's sub-millisecond slices then lose most of their parallelism to it (about 0.1 ms a slice), for comparing against the pool.
- `BRASS_GC_OLD_RESERVE_MB=n` sets both old spaces' reservations (`mature_reserve_bytes`, `large_reserve_bytes`) to n MB, which makes the out-of-memory path reachable from a test.

### 2.9. Out of Memory
A collection promotes young survivors into the old generation and cannot stop half way, so each old space keeps a promotion reserve from the mutator: twice the young generation's capacity (eden plus one survivor space), at most a quarter of the space. The mutator's own old and large allocations stop short of it; only a collection takes from it. When a collection has had to, the heap is critical: the next allocation that reaches the runtime (anything but the bump fast path) runs a full collection first and, if the reserve is still breached, throws `std::bad_alloc`. Eden is empty after that collection and every structure is consistent, so the program can drop references and continue; the next full collection gives the room back. An allocation too large for its space also throws `std::bad_alloc` after a full collection. Only a program that keeps allocating live data through repeated failures can exhaust the reserve itself, which stops the process.
- `is_valid_object`, `find_object` (interior address to object), `check_access` (the interpreters' bounds check), `for_each_object` and `stats()` (collection counts, pause totals and maxima, promoted and live bytes) support debugging and tests.

### 2.10. Marking Cycle
A full collection is a marking cycle of three steps, which a host can also run as separate calls to spread the pause out:

```cpp
heap.start_marking();                 // false if a cycle is already open
while (!heap.mark_slice(budget_ns)) { // true once the grey work is gone
}
heap.finish_marking();                // then the finalizers it queued run
```

- **Start** (a pause): the roots are visited and the old objects they name are marked and pushed grey, and the write barrier's value range widens to the whole heap (2.4). The young generation is left as it is (`collect(Full)`, which runs the steps back to back, evacuates it first, as a full collection's first phase). Start first completes the previous cycle's sweep if it is still pending (2.11); the mark bitmaps are then clear: that sweep cleared every mark it kept, and a block or page handed out since starts clear.
- **Slices** (a pause each): `mark_slice(budget_ns)` drains grey work for about the budget and returns whether it is all gone. The grey work (each marker thread's stack, the shared pool of chunks, the 16 KB slices of large objects) stays on the heap between slices, and a slice resumes it where the last one stopped. A slice whose budget is at least 0.2 ms runs on the marker's threads (`mark_threads`, when the old generation holds `parallel_mark_bytes`); a shorter one on the calling thread, since starting the threads would cost too much of it. Threads look at the clock every few microseconds of marking (every 256 objects, a large-object slice counting more), so a slice overshoots its budget by about that, and always makes some progress however small the budget. When the grey work is gone the slice resolves the ephemerons whose keys have been marked since they were met, and goes on while that finds more. `collect(Full)` runs one slice with no budget.
- **Finish** (a pause): the young generation is evacuated with every survivor promoted grey and every old object a dirty card names shaded, and the roots are visited again as a full collection visits them; whatever grey work is left is drained, the barrier's range goes back to the young generation, then weak slots and ephemerons are settled, finalizers of dead objects queued, hooks run (with `survivor_of` answering from the mark bits) and the old generation's sweep begun, exactly as at the end of a full collection.

The collection is counted and logged once, at finish: `full_pause_ns_total` adds all of its pauses, `full_pause_ns_max` and `last_pause_ns` take the longest one. `marking()` says whether a cycle is open.

**The program runs between the steps.** Marking is incremental with an insertion barrier: the cycle's invariant is that no marked (black) old object names an unmarked old object the marker will not otherwise reach, except through a slot the finish rescans (a root, or a slot behind a dirty card). What keeps it while the program runs:

- **Stores.** The barrier shades an old reference stored into an old object (2.4); a young one dirties the card as always.
- **Minor collections** inside the cycle run as usual, and also shade every old object an old object names as they scan it (a dirty card's slots, which they clean so that the finish no longer sees them; the tracer's range is the heap). They do not shade what a root or a live young object names: the finish reaches those again, and shading them would keep what a root held only for a moment (a WeakRef's target on the kept-objects list until the job ends) alive in every cycle when one is open most of the time. What they promote is marked grey as it is copied: grey rather than black because a layout's trace function may visit more for a full collection's tracer than for a minor's.
- **Old allocation** (pretenured, large, or an object too big for the young generation) is marked grey and its card dirtied, so its initializing stores, made without barriers, are seen by the next minor collection or the finish.
- **Root slots** carry no barrier; the finish visits them again.
- **A full collection asked for while a cycle is open** (`collect(Full)`, a requested, stressed or exhausted one) finishes the cycle, then runs a whole collection back to back, since what the cycle kept includes what died while it ran. One asked for by old-generation growth only finishes the cycle: the growth asks for room, not for that garbage.

The marking flag (`marking()`, and the barrier range with it) changes only at a GC point: `start_marking` and `finish_marking` are each one, and so is an allocation or safepoint that runs a step. The marker threads exist only inside a slice or the finish, and shading happens only on the heap's thread outside them (the barrier, a minor collection, an allocation): nothing shades while marker threads run, and it is a hard error if something does. Shading pushes onto the first marker's stack and marks the grey work as not done, so the next slice or the finish drains it. `mark_slice` or `finish_marking` with no cycle open, and any step called from inside a collection (a trace function, root source or hook), stop the process.

### 2.11. Lazy Sweep
A marking cycle's finish settles everything that depends on liveness (weak slots, ephemerons, finalizers queued, hooks run, cards cleaned) but leaves the sweep of the old generation pending (`HeapConfig::lazy_sweep`, default on; `BRASS_GC_EAGER_SWEEP=1` turns it off). The finish's pause then costs O(1) for the sweep: every block in use becomes unswept at once by bumping a sweep epoch that each block's metadata compares against. The pending work is done by whichever comes first:

```cpp
heap.finish_marking();
while (!heap.sweep_slice(budget_ns)) {   // true once nothing is left to sweep
}
bool pending = heap.sweeping();
```

- **`sweep_slice(budget_ns)`** sweeps for about the budget (it looks at the clock every 8 blocks or large objects, and always makes progress) and returns whether the sweep is complete; with no sweep pending it returns true at once. Called from inside a collection it stops the process. `pressure()` reports `unswept_blocks` and `sweeping`.
- **The allocator**, when it needs a block with free lines or a free block and has none swept: it sweeps a few blocks (at most 4 per refill) from the low end while slices sweep from the high end, and a large allocation with no free run that fits sweeps a few large objects first. When an allocation would otherwise be refused (the promotion reserve or the reservation's end), the whole sweep is completed first, so lazy sweep never makes an allocation fail or a full collection run that the eager sweep would not have.
- **Completion**: `start_marking`, and so any full collection, completes a pending sweep first (logged as `start X (sweep Y)`).

What is lazy: mature blocks (marks to starts and lines, free blocks listed, poisoning), dead large objects (freed in address order by a cursor over the page table), and returning free blocks to the OS (a planned list of runs released a run at a time), in that order. Settlement at the finish is not: finalizers, weak slots, ephemerons and hooks see exactly what the eager sweep would.

Invariants while a sweep is pending:
- **Liveness of an unswept block is the finished cycle's mark bits**: its objects are the starts that are also marked. `is_valid_object`, `find_object`, `check_access`, `for_each_object`, verification and the card scan all read it so; a dead object's slots are never visited, so they are neither roots nor remembered-set entries. A dead large object not yet freed is likewise not an object, and all of its cards are clean.
- **Marks are cleared on sweep**: sweeping a block moves its marks into its starts and clears them; a large object's mark bit is cleared when the cursor passes it. Since the next cycle's start completes the sweep, its marks always start clear, so no second bitmap or flip is needed (a flip would have to be cleared or tested on every allocation and in the barrier instead).
- **Allocation goes only into swept space**: a recycled or free block is swept before the allocator hands out its lines, and a block taken from the free list or the frontier is stamped swept. A large object allocated ahead of the sweep's cursor is marked as it is allocated, so the cursor keeps it and clears its mark. Promotion uses the same paths, so what a minor collection promotes during a pending sweep is never swept away.
- **Accounting**: `old_used_bytes` is the cycle's marked bytes plus what has been allocated old since, until the sweep completes and the exact count takes over.
- Under `BRASS_GC_STRESS=incremental` an idle heap does sweep slices of zero budget on three of four stress steps, so pending sweeps are exercised; under `BRASS_GC_VERIFY=1` a sweep completed by slices is followed by a heap check, and the finish's verification marking checks nothing live is left unmarked, so nothing live is swept.

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
| `brass_coro_create(fn, slots, mask)` | A coroutine frame for fixed native code, allocated on the current heap (or unmanaged memory when the thread has none); `mask` bit 6 is the awaiter link and bit `i + 7` slot `i`. |
| `brass_coro_create_body(desc)` | A frame for a lowered body's descriptor (`CoroBody`), traced by the descriptor's reference map (any number of slots) plus the awaiter link. |

A frame that is old when its body resumes still receives young references: the transform emits a `write_barrier` after every spill of a reference into the frame, and the resume value is barriered as it is stored.

Generated code, interpreters and resumers hold a frame by its raw address. On a heap whose reference tags leave out tag 0 (`Heap::raw_addresses_are_references()` false) a raw address is not a reference the collector could update, so there every frame is allocated old, where it never moves; the frame registry's slots are visited as raw addresses (`Tracer::visit_address`) on every heap, and the awaiter link is stored under `Heap::raw_address_tag()` so the frame's layout traces it. A new frame is registered as a root until `brass_coro_unroot(frame)`: a host that holds the frame in a traced object of its own unroots it, and from then on the frame lives as long as that object does. The creator's arguments are stored with no barrier, so the frame's card is dirtied when it is allocated.

On MSVC the first three, `brass_coro_create` and `brass_coro_create_body` are assembly stubs (`src/gc/gc_msvc_{x64,arm64}.asm`) that pass their caller's frame pointer and return address to a C++ bridge; elsewhere the functions read them with `__builtin_frame_address` and `__builtin_return_address`.

`Heap::allocation_buffer()` exposes the eden bump region `{top, end}` so code can allocate young objects inline: write the header at `top`, advance `top`, zero the payload, and call the runtime when the object does not fit. `end` is eden's adaptive limit, which each minor collection moves, so code must reload it rather than cache it across a call that may collect. Stress mode pulls `end` back to `top`. `Heap::bind_allocation_buffer(buffer)` moves the region into a host-owned `{top, end}` pair (a per-thread block its generated code already addresses), carrying the current values over; the heap then bumps and resets that pair, and `nullptr` moves it back.

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

- Collection is stop-the-world per heap, and marking is parallel but not concurrent; a full collection's pause grows with the live old generation. Its marking can be cut into slices with the program running between them (2.10), but the finish still evacuates the young generation and revisits every root, so the finish's pause grows with them. The sweep is lazy (2.11), but a host that never calls `sweep_slice` leaves what the allocator has not swept to the next cycle's start, whose pause then includes it.
- A minor collection copies on one thread, so its pause grows with the young data that survives it. The adaptive eden bounds that for data the program keeps, after the first two collections of a burst of survivors, and only the copying: the dirty cards a minor collection scans do not shrink with eden, a young working set that dies in the survivor space is copied whole at every collection, and a program allocating nothing but long-lived data trades each long collection for many 1 MB ones, each copying about 2 MB (its eden's survivors and the last one's again).
- The old generation does not compact: fragmentation in the mature space is bounded by line and block reuse but not repaired by evacuation.
- A young object cannot be pinned in place; objects that must not move are allocated old (`kAllocOld` / `kAllocPinned`).
- Compiled code calls the write barrier and `brass_gc_alloc` out of line; neither is inlined into generated code.
