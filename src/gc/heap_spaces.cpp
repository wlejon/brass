// The old generation's two spaces: mature blocks (mark-region: 32 KiB blocks
// of 256-byte lines, bump allocation into runs of free lines, reclaimed a
// line at a time after a full collection) and the large-object space (page
// runs, one object each, reclaimed whole). Objects in either never move.

#include "heap_internal.hpp"

#include <algorithm>
#include <cstring>

namespace brass::gc::detail {

namespace {

constexpr size_t kRetainedFreeBlocks = 64;
constexpr uint8_t kPoisonByte = 0xDB;

size_t os_page_bytes() { return 4096; }

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
        // The next block with holes, else a free one (a single hole).
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
    uint32_t start = UINT32_MAX;
    for (auto it = large_free_runs.begin(); it != large_free_runs.end(); ++it) {
        if (it->second >= pages) {
            start = it->first;
            const uint32_t rest = it->second - pages;
            large_free_runs.erase(it);
            if (rest) large_free_runs.emplace(start + pages, rest);
            break;
        }
    }
    if (start == UINT32_MAX) {
        if ((static_cast<size_t>(large_frontier) + pages) * kPageBytes > large_reserve) return 0;
        start = large_frontier;
        large_frontier += pages;
        large_page_head.resize(large_frontier, 0);
    }
    const uintptr_t header = large_lo + static_cast<uintptr_t>(start) * kPageBytes;
    const size_t bytes = static_cast<size_t>(pages) * kPageBytes;
    vm_commit(reinterpret_cast<void*>(header), bytes);
    commit_cards(header, header + bytes);
    large_objects.emplace(start, pages);
    for (uint32_t p = start; p < start + pages; ++p) large_page_head[p] = start + 1;
    large_used_bytes += bytes;
    return header;
}

void HeapState::free_large(uint32_t head, uint32_t pages) {
    const uintptr_t header = large_lo + static_cast<uintptr_t>(head) * kPageBytes;
    const size_t bytes = static_cast<size_t>(pages) * kPageBytes;
    vm_decommit(reinterpret_cast<void*>(header), bytes);
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
}

uint32_t HeapState::large_head_of(uintptr_t address) const noexcept {
    const size_t page = (address - large_lo) / kPageBytes;
    if (page >= large_page_head.size()) return UINT32_MAX;
    const uint32_t v = large_page_head[page];
    return v == 0 ? UINT32_MAX : v - 1;
}

uint64_t HeapState::sweep_old() {
    small_region = BumpRegion{};
    medium_region = BumpRegion{};
    free_blocks.clear();
    recyclable_blocks.clear();
    mature_used_lines = 0;

    for (size_t i = blocks.size(); i-- > 0;) {
        BlockMeta& meta = *blocks[i];
        const uint32_t index = static_cast<uint32_t>(i);
        if (!meta.in_use) {
            free_blocks.push_back(index);
            continue;
        }
        const uintptr_t lo = block_base(index);
        for (size_t w = 0; w < kStartWordsPerBlock; ++w) {
            uint64_t dead = meta.starts[w] & ~meta.marks[w];
            if (poison) {
                while (dead) {
                    const unsigned bit = ctz64(dead);
                    dead &= dead - 1;
                    const uintptr_t payload = lo + (w * 64 + bit) * kGranuleBytes;
                    const size_t size = header_of(payload)->size;
                    std::memset(reinterpret_cast<void*>(payload - kHeaderBytes), kPoisonByte, size + kHeaderBytes);
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
        mature_used_lines += used;
        if (used == 0) {
            meta.in_use = false;
            free_blocks.push_back(index);
        } else if (used < kLinesPerBlock) {
            recyclable_blocks.push_back(index);
        }
    }
    // Both lists run from the highest block down, so popping from the back
    // hands out the lowest addresses first. Free blocks past the retained
    // ones (the highest) go back to the OS; they stay on the list and are
    // committed again when reused. The retained count scales with the blocks
    // in use, so a heap that promotes steadily is not decommitting and
    // recommitting the same blocks at every full collection; runs of
    // adjacent blocks go back in one call.
    const size_t in_use = blocks.size() - free_blocks.size();
    const size_t retained = std::max(kRetainedFreeBlocks, in_use / 4);
    if (free_blocks.size() > retained) {
        const size_t release = free_blocks.size() - retained;
        size_t k = 0;
        while (k < release) {
            if (!blocks[free_blocks[k]]->committed) {
                ++k;
                continue;
            }
            const uint32_t high = free_blocks[k];
            uint32_t low = high;
            blocks[low]->committed = false;
            while (k + 1 < release && free_blocks[k + 1] == low - 1 && blocks[low - 1]->committed) {
                ++k;
                --low;
                blocks[low]->committed = false;
            }
            vm_decommit(reinterpret_cast<void*>(block_base(low)), static_cast<size_t>(high - low + 1) * kBlockBytes);
            ++k;
        }
    }
    if (poison) {
        for (uint32_t index : free_blocks) {
            if (blocks[index]->committed) {
                std::memset(reinterpret_cast<void*>(block_base(index)), kPoisonByte, kBlockBytes);
            }
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> dead_large;
    for (auto& [head, pages] : large_objects) {
        auto* header = reinterpret_cast<ObjectHeader*>(large_lo + static_cast<uintptr_t>(head) * kPageBytes);
        if (header->gc_bits & kGcLargeMarked) {
            header->gc_bits = static_cast<uint8_t>(header->gc_bits & ~kGcLargeMarked);
        } else {
            dead_large.emplace_back(head, pages);
        }
    }
    for (auto [head, pages] : dead_large) {
        large_objects.erase(head);
        free_large(head, pages);
    }
    return mature_used_lines * kLineBytes + large_used_bytes;
}

} // namespace brass::gc::detail
