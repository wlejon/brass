// The old generation's sweep after a marking cycle, a piece at a time.
//
// A cycle's finish decides what is dead (the mark bits) and settles all that
// depends on it there: weak slots, ephemerons, finalizers and hooks see the
// mark bits before anything is freed. Freeing the memory is left to this
// sweep, so the finish's pause does not grow with the old generation's size:
//
//   - Mature blocks. Each finish bumps the sweep epoch (LazySweep::epoch),
//     which makes every block in use unswept at no cost. An unswept block
//     keeps the last cycle's marks and line marks, and its live objects are
//     exactly its marked starts: the queries (is_valid_object, find_object,
//     for_each_object, verification) and a minor collection's card scan read
//     it that way, so a dead object is never scanned (its slots may name
//     memory freed since) and never reported. Sweeping a block is what the
//     eager sweep did to it: keep the marked starts, rebuild the free lines
//     from the line marks, clear the marks, file it as free or recyclable.
//     Nothing is allocated into an unswept block: the allocator sweeps a
//     block before it takes it (from the bottom, a few per refill: the
//     allocator's share), and a block it takes off the free list is current
//     already. So promotion during the sweep never lands on an object the
//     sweep has yet to judge, and the marks the sweep clears were all set by
//     the cycle that finished: the next cycle's marks start clear without a
//     second bitmap or a flip (each block's marks are cleared by the sweep
//     that reads them, and the next cycle's start finishes the sweep first).
//   - Large objects. A cursor walks the page table; a head at or past it is
//     unswept, dead unless marked. A large object allocated past the cursor
//     while it walks is marked as it is made (note_large_allocated), so the
//     walk keeps it and clears the mark; one allocated behind is left clear.
//     The allocator advances the walk when no free run fits.
//   - Release. Once every block is swept, the free blocks past the retained
//     ones go back to the OS, highest first, a contiguous run per call.
//
// Heap::sweep_slice does the work for about a budget (the host's time
// between frames); whatever is left when the next cycle starts is finished
// there, inside its start's pause, and an old space near its end finishes it
// before refusing an allocation. HeapConfig::lazy_sweep = false
// (BRASS_GC_EAGER_SWEEP=1) does all of it inside the finish, as before.

#include "heap_mark.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>

namespace brass::gc::detail {

namespace {

constexpr size_t kRetainedFreeBlocks = 64;
constexpr uint8_t kPoisonByte = 0xDB;
// Unswept blocks one allocator refill sweeps looking for free lines before it
// takes a free block (or a fresh one from the frontier). A block's sweep is
// about a microsecond (its 1.3 KB of side metadata, mostly cache misses), so
// a minor collection that promotes into the old generation while the sweep
// is pending pays at most a few microseconds per 32 KB it fills, and a run of
// full blocks cannot turn one allocation into the whole sweep.
constexpr unsigned kAllocatorSweepBlocks = 4;
// Unswept large objects one large allocation looks at for a free run that
// fits before it grows the frontier.
constexpr unsigned kAllocatorSweepLarge = 16;
// Units of sweep work between looks at the clock in a slice: a block, a large
// object or a free run is one (about a microsecond at most); a release's
// decommit counts as a look on its own.
constexpr unsigned kClockEvery = 8;

void poison_range(uintptr_t at, size_t bytes) { std::memset(reinterpret_cast<void*>(at), kPoisonByte, bytes); }

// Sweeps unswept block `index`: what the eager sweep did to each block.
void sweep_block(HeapState& s, uint32_t index) {
    BlockMeta& meta = *s.blocks[index];
    const uintptr_t lo = s.block_base(index);
    for (size_t w = 0; w < kStartWordsPerBlock; ++w) {
        if (s.poison) {
            uint64_t dead = meta.starts[w] & ~meta.marks[w];
            while (dead) {
                const unsigned bit = ctz64(dead);
                dead &= dead - 1;
                const uintptr_t payload = lo + (w * 64 + bit) * kGranuleBytes;
                poison_range(payload - kHeaderBytes, header_of(payload)->size + kHeaderBytes);
            }
        }
        meta.starts[w] &= meta.marks[w];
        meta.marks[w] = 0;
    }
    std::memcpy(meta.line_used, meta.line_mark, kLinesPerBlock);
    std::memset(meta.line_mark, 0, kLinesPerBlock);
    size_t used = 0;  // line marks are 0 or 1, so a word's popcount counts its lines
    for (size_t l = 0; l < kLinesPerBlock; l += 8) {
        uint64_t word;
        std::memcpy(&word, meta.line_used + l, 8);
        used += popcount64(word);
    }
    LazySweep& w = s.sweep;
    meta.sweep_epoch = w.epoch;
    --w.unswept_blocks;
    ++w.blocks_swept;
    s.mature_used_lines += used;
    if (used == 0) {
        meta.in_use = false;
        s.free_blocks.push_back(index);
        ++w.blocks_freed;
        if (s.poison) poison_range(lo, kBlockBytes);
    } else if (used < kLinesPerBlock) {
        s.recyclable_blocks.push_back(index);
    }
}

// The next unswept block from the top, swept; false once none is left.
bool sweep_block_from_top(HeapState& s) {
    LazySweep& w = s.sweep;
    while (w.low < w.high) {
        const uint32_t index = --w.high;
        if (s.block_unswept(*s.blocks[index])) {
            sweep_block(s, index);
            return true;
        }
    }
    return false;
}

// One step of the large objects' walk at its cursor: the next object (freed
// if dead, its mark cleared if not) or free run. `freed_run`: the pages of
// the free run a freed object joined, else 0. False once the walk is done.
bool sweep_large_step(HeapState& s, uint32_t& freed_run) {
    LazySweep& w = s.sweep;
    freed_run = 0;
    uint32_t p = w.large_cursor;
    if (p >= s.large_frontier) {
        w.large_active = false;
        return false;
    }
    const uint32_t head_plus_one = s.large_page_head[p];
    if (head_plus_one == 0) {
        const auto it = s.large_free_runs.find(p);
        p += it != s.large_free_runs.end() ? it->second : 1;
    } else {
        const uint32_t head = head_plus_one - 1;
        auto* h = reinterpret_cast<ObjectHeader*>(s.large_lo + static_cast<uintptr_t>(head) * kPageBytes);
        // The run large_allocate gave it: header and payload in whole pages.
        const auto pages = static_cast<uint32_t>((kHeaderBytes + h->size + kPageBytes - 1) / kPageBytes);
        // A head behind the cursor is an object allocated since the walk
        // passed (live); its pages are skipped.
        if (head == p) {
            if (h->gc_bits & kGcLargeMarked) {
                h->gc_bits = static_cast<uint8_t>(h->gc_bits & ~kGcLargeMarked);
            } else {
                s.large_objects.erase(head);
                freed_run = s.free_large(head, pages);
                ++w.large_freed;
                w.large_freed_bytes += static_cast<uint64_t>(pages) * kPageBytes;
            }
        }
        p = head + pages;
    }
    w.large_cursor = p;
    if (p >= s.large_frontier) w.large_active = false;
    return true;
}

// The free blocks to give back to the OS, as the eager sweep chose them: past
// those retained, the highest first. The list is sorted from the highest
// down, so popping from its back hands out the lowest blocks first.
void plan_release(HeapState& s) {
    LazySweep& w = s.sweep;
    w.release_planned = true;
    w.release_list.clear();
    w.release_next = 0;
    std::sort(s.free_blocks.begin(), s.free_blocks.end(), std::greater<uint32_t>());
    // The retained count scales with the blocks in use, so a heap that
    // promotes steadily is not decommitting and recommitting the same blocks
    // at every full collection. The blocks one cycle's old allocation fills
    // before the next full collection are kept too, else a program allocating
    // old garbage (a runtime that allocates its objects pinned) page-faults a
    // fresh block for every 32 KiB it allocates.
    const size_t in_use = s.blocks.size() - s.free_blocks.size();
    const size_t retained = std::max({kRetainedFreeBlocks, in_use / 4,
                                      static_cast<size_t>(s.config.min_full_threshold_bytes / kBlockBytes)});
    if (s.free_blocks.size() <= retained) return;
    const size_t release = s.free_blocks.size() - retained;
    w.release_list.assign(s.free_blocks.begin(), s.free_blocks.begin() + static_cast<std::ptrdiff_t>(release));
}

// Decommits the next run of planned blocks (each still free and committed:
// the allocator may have taken one since the plan). False once none is left.
bool release_step(HeapState& s) {
    LazySweep& w = s.sweep;
    const auto releasable = [&s](uint32_t index) {
        const BlockMeta& m = *s.blocks[index];
        return !m.in_use && m.committed;
    };
    while (w.release_next < w.release_list.size()) {
        const uint32_t high = w.release_list[w.release_next++];
        if (!releasable(high)) continue;
        uint32_t low = high;
        s.blocks[low]->committed = false;
        while (w.release_next < w.release_list.size() && low > 0 && w.release_list[w.release_next] == low - 1 &&
               releasable(low - 1)) {
            ++w.release_next;
            --low;
            s.blocks[low]->committed = false;
        }
        const size_t bytes = static_cast<size_t>(high - low + 1) * kBlockBytes;
        vm_decommit(reinterpret_cast<void*>(s.block_base(low)), bytes);
        w.released_bytes += bytes;
        return true;
    }
    return false;
}

// The sweep is done: the line counts are exact again (old_used_bytes), the
// next cycle may start, and the log has its line.
void end_sweep(HeapState& s) {
    LazySweep& w = s.sweep;
    w.active = false;
    w.release_list.clear();
    w.release_next = 0;
    if (s.log && w.lazy) log_sweep(s);
}

} // namespace

bool HeapState::large_dead_pending(uint32_t head) const noexcept {
    if (!sweep.large_active || head < sweep.large_cursor) return false;
    const auto* h = reinterpret_cast<const ObjectHeader*>(large_lo + static_cast<uintptr_t>(head) * kPageBytes);
    return (h->gc_bits & kGcLargeMarked) == 0;
}

void HeapState::begin_sweep(uint64_t live_bytes) {
    LazySweep& w = sweep;
    // A cycle's start finishes the last sweep, so its finish never finds one.
    if (w.active) gc_fatal("a marking cycle finished with the last cycle's sweep still pending");
    ++w.epoch;
    w.active = true;
    w.low = 0;
    w.high = static_cast<uint32_t>(blocks.size());
    w.unswept_blocks = blocks.size() - free_blocks.size();  // free_blocks lists exactly the blocks not in use
    w.large_active = large_frontier > 0;
    w.large_cursor = 0;
    w.release_planned = false;
    w.release_list.clear();
    w.release_next = 0;
    w.live_bytes = live_bytes;
    w.collection_index = stats.minor_collections + stats.full_collections + 1;  // counted once the finish records it
    w.slice_ns = w.slice_max_ns = w.slices = 0;
    w.allocator_ns = w.allocator_blocks = w.complete_ns = 0;
    w.blocks_swept = w.blocks_freed = w.large_freed = w.large_freed_bytes = w.released_bytes = 0;
    // The regions' blocks and the recyclable ones are unswept now; the
    // allocator starts over from what the sweep hands it.
    small_region = BumpRegion{};
    medium_region = BumpRegion{};
    recyclable_blocks.clear();
    mature_used_lines = 0;  // summed again block by block as they are swept
    if (!w.lazy) {
        sweep_work(kNoDeadline);
        end_sweep(*this);
    }
}

bool HeapState::sweep_work(uint64_t deadline_ns) {
    LazySweep& w = sweep;
    unsigned work = 0;
    // True when the slice is out of time, looked at every kClockEvery units
    // (and after each `units` >= kClockEvery step): the first look comes
    // after some work, so a slice always makes progress.
    const auto spent = [&](unsigned units) {
        work += units;
        if (work < kClockEvery) return false;
        work = 0;
        return deadline_ns != kNoDeadline && gc_now_ns() >= deadline_ns;
    };
    // The large objects first: each frees the most memory per unit of work.
    uint32_t run = 0;
    while (w.large_active && sweep_large_step(*this, run)) {
        if (spent(1)) return false;
    }
    while (sweep_block_from_top(*this)) {
        if (spent(1)) return false;
    }
    if (!w.release_planned) {
        plan_release(*this);
        if (spent(kClockEvery)) return false;
    }
    while (release_step(*this)) {
        if (spent(kClockEvery)) return false;
    }
    return true;
}

void HeapState::complete_sweep() {
    if (!sweep.active) return;
    const uint64_t t0 = gc_now_ns();
    sweep_work(kNoDeadline);
    sweep.complete_ns += gc_now_ns() - t0;
    end_sweep(*this);
}

bool HeapState::sweep_blocks_for_allocation() {
    LazySweep& w = sweep;
    const uint64_t t0 = log ? gc_now_ns() : 0;
    bool found = false;
    unsigned swept = 0;
    while (w.low < w.high && swept < kAllocatorSweepBlocks) {
        const uint32_t index = w.low++;
        BlockMeta& meta = *blocks[index];
        if (!block_unswept(meta)) continue;
        const size_t recyclable = recyclable_blocks.size();
        sweep_block(*this, index);
        ++swept;
        if (!meta.in_use || recyclable_blocks.size() > recyclable) {
            found = true;
            break;
        }
    }
    w.allocator_blocks += swept;
    if (log) w.allocator_ns += gc_now_ns() - t0;
    return found;
}

bool HeapState::sweep_large_for_allocation(uint32_t pages) {
    LazySweep& w = sweep;
    const uint64_t t0 = log ? gc_now_ns() : 0;
    bool found = false;
    uint32_t run = 0;
    for (unsigned n = 0; n < kAllocatorSweepLarge && w.large_active; ++n) {
        if (!sweep_large_step(*this, run)) break;
        if (run >= pages) {
            found = true;
            break;
        }
    }
    if (log) w.allocator_ns += gc_now_ns() - t0;
    return found;
}

bool HeapState::old_in_reserve_swept() {
    if (!old_in_reserve()) return false;
    if (sweep.low >= sweep.high && !sweep.large_active) return true;
    complete_sweep();
    return old_in_reserve();
}

} // namespace brass::gc::detail

namespace brass::gc {

bool Heap::sweep_slice(uint64_t budget_ns) {
    detail::HeapState& s = *s_;
    if (s.collecting) {
        detail::gc_fatal("sweep_slice during a collection (a trace function, root source or hook called it)");
    }
    if (!s.sweep.active) return true;
    const uint64_t t0 = detail::gc_now_ns();
    const uint64_t deadline = budget_ns >= detail::kNoDeadline - t0 ? detail::kNoDeadline : t0 + budget_ns;
    const bool done = s.sweep_work(deadline);
    const uint64_t ns = detail::gc_now_ns() - t0;
    s.sweep.slice_ns += ns;
    s.sweep.slice_max_ns = std::max(s.sweep.slice_max_ns, ns);
    ++s.sweep.slices;
    if (done) {
        detail::end_sweep(s);
        // Under verification the swept heap is checked whole: a live object
        // the sweep freed shows up as a slot naming no object.
        if (s.verify) detail::verify_heap(*this, "after a lazy sweep");
    }
    return done;
}

bool Heap::sweeping() const noexcept { return s_->sweep.active; }

} // namespace brass::gc
