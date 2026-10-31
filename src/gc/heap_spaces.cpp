// The old generation's two spaces: mature blocks (mark-region: 32 KiB blocks
// of 256-byte lines, bump allocation into runs of free lines, reclaimed a
// line at a time after a full collection) and the large-object space (page
// runs, one object each, reclaimed whole). Objects in either never move.
// Their sweep after a marking cycle is heap_sweep_lazy.cpp; while it is
// pending the allocator here takes only swept blocks and free page runs,
// sweeping a few more itself when it finds none.

#include "heap_internal.hpp"

#include <algorithm>
#include <cstring>

namespace brass::gc::detail {

namespace {

// Bytes of freed large-object pages kept committed at least: a reused run
// then costs no VirtualAlloc / mprotect and no fresh page faults. Up to the
// full-collection threshold is kept, the old bytes one cycle allocates, so
// a program whose large garbage dies every cycle commits nothing in steady
// state.
constexpr size_t kRetainedLargeBytes = size_t{8} << 20;

} // namespace

void HeapState::commit_cards(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    const size_t page = os_page_bytes();
    const size_t first = ((lo - mature_lo) >> kCardShift) / page;
    const size_t last = (((hi - 1) - mature_lo) >> kCardShift) / page;
    for (size_t p = first; p <= last && p < card_pages_committed.size(); ++p) {
        if (card_pages_committed[p]) continue;
        uint8_t* at = cards + p * page;
        vm_commit(at, page);
        std::memset(at, kCardClean, page);
        card_pages_committed[p] = 1;
    }
}

void HeapState::mark_lines(BlockMeta& meta, uintptr_t block_lo, uintptr_t header, size_t total,
                           bool mark) noexcept {
    const size_t first = (header - block_lo) / kLineBytes;
    const size_t last = (header + total - 1 - block_lo) / kLineBytes;
    uint8_t* lines = mark ? meta.line_mark : meta.line_used;
    for (size_t l = first; l <= last; ++l) lines[l] = 1;
}

uint32_t HeapState::acquire_free_block() {
    // While the last cycle's sweep is pending, some unswept blocks are free
    // ones the eager sweep would have listed: a few are looked at before a
    // fresh block is taken from the frontier, and all of them before the
    // allocation is refused for the promotion reserve or the reservation's
    // end. A parallel minor collection's promotion never sweeps
    // (HeapState::parallel_promotion): it takes a free block or the frontier.
    if (sweep.low < sweep.high && !parallel_promotion) {
        if (free_blocks.empty()) sweep_blocks_for_allocation();
        const bool at_reserve = !collecting && mature_blocks_available() <= reserve_mature_blocks;
        const bool at_end = free_blocks.empty() && (blocks.size() + 1) * kBlockBytes > mature_reserve;
        if (at_reserve || at_end) complete_sweep();
    }
    // The last `reserve_mature_blocks` are a collection's (the promotion reserve).
    if (!collecting && mature_blocks_available() <= reserve_mature_blocks) return UINT32_MAX;
    uint32_t index;
    if (!free_blocks.empty()) {
        index = free_blocks.back();
        free_blocks.pop_back();
    } else {
        index = static_cast<uint32_t>(blocks.size());
        if ((static_cast<size_t>(index) + 1) * kBlockBytes > mature_reserve) return UINT32_MAX;
        blocks.push_back(std::make_unique<BlockMeta>());
        commit_cards(block_base(index), block_base(index) + kBlockBytes);
    }
    BlockMeta& meta = *blocks[index];
    if (!meta.committed) {
        vm_commit(reinterpret_cast<void*>(block_base(index)), kBlockBytes);
        meta.committed = true;
    }
    meta.in_use = true;
    // A free block holds nothing to sweep: swept as of this sweep, so the
    // pending sweep passes over it and what is allocated into it stays.
    meta.sweep_epoch = sweep.epoch;
    return index;
}

bool HeapState::refill_region(BumpRegion& region, bool whole_block) {
    if (whole_block) {
        const uint32_t index = acquire_free_block();
        if (index == UINT32_MAX) return false;
        region.block = index;
        region.cursor = block_base(index);
        region.limit = region.cursor + kBlockBytes;
        region.next_line = static_cast<uint32_t>(kLinesPerBlock);
        return true;
    }
    for (;;) {
        if (region.block != UINT32_MAX) {
            const BlockMeta& meta = *blocks[region.block];
            uint32_t line = region.next_line;
            while (line < kLinesPerBlock && meta.line_used[line]) ++line;
            if (line < kLinesPerBlock) {
                uint32_t end = line + 1;
                while (end < kLinesPerBlock && !meta.line_used[end]) ++end;
                region.cursor = block_base(region.block) + line * kLineBytes;
                region.limit = block_base(region.block) + end * kLineBytes;
                region.next_line = end;
                return true;
            }
        }
        // The next block with holes, else a free one (a single hole). With
        // the last cycle's sweep pending, holes may be in unswept blocks:
        // the allocator sweeps a few (a block it takes is always swept, so
        // nothing is allocated over an object the sweep has yet to judge).
        // (Never for a parallel minor collection's promotion: HeapState::parallel_promotion.)
        if (recyclable_blocks.empty() && sweep.low < sweep.high && !parallel_promotion) {
            (void)sweep_blocks_for_allocation();
        }
        if (!recyclable_blocks.empty()) {
            region.block = recyclable_blocks.back();
            recyclable_blocks.pop_back();
            region.next_line = 0;
            continue;
        }
        const uint32_t index = acquire_free_block();
        if (index == UINT32_MAX) {
            region = BumpRegion{};
            return false;
        }
        region.block = index;
        region.next_line = 0;
    }
}

uintptr_t HeapState::mature_allocate(size_t total) {
    BumpRegion* region = &small_region;
    if (total > kLineBytes && (small_region.limit - small_region.cursor) < total) {
        // A medium object that does not fit the current hole takes the
        // overflow region (whole free blocks) rather than skipping holes.
        region = &medium_region;
    }
    while (region->limit - region->cursor < total) {
        if (!refill_region(*region, region == &medium_region)) return 0;
    }
    const uintptr_t header = region->cursor;
    region->cursor += total;
    BlockMeta& meta = *blocks[region->block];
    const uintptr_t lo = block_base(region->block);
    const size_t granule = (header + kHeaderBytes - lo) / kGranuleBytes;
    meta.starts[granule >> 6] |= uint64_t{1} << (granule & 63);
    mark_lines(meta, lo, header, total, false);
    return header;
}

uintptr_t HeapState::large_allocate(size_t total) {
    const uint32_t pages = static_cast<uint32_t>((total + kPageBytes - 1) / kPageBytes);
    // The promotion reserve: the last `reserve_large_bytes` are a collection's.
    // Dead objects the pending sweep has not freed count as used until it
    // does, so it is finished first when they would make the difference.
    const auto over_reserve = [&] {
        return !collecting &&
               large_used_bytes + static_cast<size_t>(pages) * kPageBytes + reserve_large_bytes > large_reserve;
    };
    if (over_reserve() && sweep.large_active && !parallel_promotion) complete_sweep();
    if (over_reserve()) return 0;
    const auto take_run = [&]() -> uint32_t {
        for (auto it = large_free_runs.begin(); it != large_free_runs.end(); ++it) {
            if (it->second >= pages) {
                const uint32_t at = it->first;
                const uint32_t rest = it->second - pages;
                large_free_runs.erase(it);
                if (rest) large_free_runs.emplace(at + pages, rest);
                return at;
            }
        }
        return UINT32_MAX;
    };
    uint32_t start = take_run();
    // No free run fits: the pending sweep may free one (a few objects are
    // looked at) before the frontier grows, and must free what it can before
    // the allocation fails at the reservation's end.
    // A parallel minor collection's promotion never sweeps
    // (HeapState::parallel_promotion): it takes the frontier.
    if (start == UINT32_MAX && sweep.large_active && !parallel_promotion && sweep_large_for_allocation(pages)) {
        start = take_run();
    }
    if (start == UINT32_MAX && sweep.large_active && !parallel_promotion &&
        (static_cast<size_t>(large_frontier) + pages) * kPageBytes > large_reserve) {
        complete_sweep();
        start = take_run();
    }
    if (start == UINT32_MAX) {
        if ((static_cast<size_t>(large_frontier) + pages) * kPageBytes > large_reserve) return 0;
        start = large_frontier;
        large_frontier += pages;
        large_page_head.resize(large_frontier, 0);
        large_page_committed.resize(large_frontier, 0);
    }
    const uintptr_t header = large_lo + static_cast<uintptr_t>(start) * kPageBytes;
    const size_t bytes = static_cast<size_t>(pages) * kPageBytes;
    // Commit the run's pages a retained free run did not leave committed, a
    // range of consecutive ones at a time.
    for (uint32_t p = start; p < start + pages;) {
        if (large_page_committed[p]) {
            large_retained_bytes -= kPageBytes;
            ++p;
            continue;
        }
        uint32_t q = p;
        while (q < start + pages && !large_page_committed[q]) large_page_committed[q++] = 1;
        vm_commit(reinterpret_cast<void*>(large_lo + static_cast<uintptr_t>(p) * kPageBytes),
                  static_cast<size_t>(q - p) * kPageBytes);
        p = q;
    }
    commit_cards(header, header + bytes);
    large_objects.emplace(start, pages);
    for (uint32_t p = start; p < start + pages; ++p) large_page_head[p] = start + 1;
    large_used_bytes += bytes;
    return header;
}

uint32_t HeapState::free_large(uint32_t head, uint32_t pages) {
    const uintptr_t header = large_lo + static_cast<uintptr_t>(head) * kPageBytes;
    const size_t bytes = static_cast<size_t>(pages) * kPageBytes;
    const size_t retain_cap = std::max<size_t>(kRetainedLargeBytes, static_cast<size_t>(full_threshold));
    if (large_retained_bytes + bytes <= retain_cap) {
        large_retained_bytes += bytes;
    } else {
        vm_decommit(reinterpret_cast<void*>(header), bytes);
        for (uint32_t p = head; p < head + pages; ++p) large_page_committed[p] = 0;
    }
    for (uint32_t p = head; p < head + pages; ++p) large_page_head[p] = 0;
    large_used_bytes -= bytes;
    uint32_t start = head;
    uint32_t count = pages;
    auto next = large_free_runs.lower_bound(start);
    if (next != large_free_runs.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) {
            start = prev->first;
            count += prev->second;
            large_free_runs.erase(prev);
        }
    }
    next = large_free_runs.lower_bound(start + count);
    if (next != large_free_runs.end() && next->first == start + count) {
        count += next->second;
        large_free_runs.erase(next);
    }
    large_free_runs.emplace(start, count);
    return count;
}

uint32_t HeapState::large_head_of(uintptr_t address) const noexcept {
    const size_t page = (address - large_lo) / kPageBytes;
    if (page >= large_page_head.size()) return UINT32_MAX;
    const uint32_t v = large_page_head[page];
    return v == 0 ? UINT32_MAX : v - 1;
}

} // namespace brass::gc::detail
