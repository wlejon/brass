#include <brass/core/arena.hpp>
#include <algorithm>
#include <cstdlib>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace brass {

namespace {

// A chunk this large or larger is pages of its own from the OS, given back
// when the arena frees it. From the C heap, a freed chunk's pages stayed in
// the process: macOS's malloc keeps a freed small block dirty until the
// system runs short, so a JIT's released function bodies (Module::
// release_bodies; about 1.5 GB of 64 KB chunks in a game) never left its
// footprint. Smaller chunks (the first of a function body's arena) stay on
// the C heap, where a page would mostly be empty.
constexpr size_t kPageChunkMin = 64 * 1024;

size_t page_granularity() {
#if defined(_WIN32)
    static const size_t g = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return static_cast<size_t>(info.dwAllocationGranularity);
    }();
#else
    static const size_t g = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    return g;
}

// Rounds `size` up to what chunk_alloc gives for it.
size_t chunk_capacity(size_t size) {
    if (size < kPageChunkMin) return size;
    const size_t g = page_granularity();
    return (size + g - 1) / g * g;
}

void pages_free(uint8_t* memory, size_t capacity) {
#if defined(_WIN32)
    (void)capacity;
    VirtualFree(memory, 0, MEM_RELEASE);
#else
    munmap(memory, capacity);
#endif
}

// A few freed chunks of the smallest paged size kept on each thread for its
// next arena: a compile makes and drops many short-lived arenas of one
// 64 KB chunk, and a map and an unmap (and the pages' first touch) for each
// cost a cold smallworlds compile about 2% (70 s on an M2 Pro). At most
// kCacheChunks a thread stay dirty; the rest go back to the OS.
constexpr size_t kCacheChunks = 8;

// (An arena freed after its thread's cache, a static one at exit, finds it
// `closed` and maps and unmaps its own.)
struct ChunkCache {
    uint8_t* chunks[kCacheChunks] = {};
    size_t count = 0;
    bool closed = false;
    ~ChunkCache() {
        for (size_t i = 0; i < count; ++i) pages_free(chunks[i], kPageChunkMin);
        count = 0;
        closed = true;
    }
};

ChunkCache& chunk_cache() {
    thread_local ChunkCache cache;
    return cache;
}

uint8_t* chunk_alloc(size_t capacity) {
    if (capacity < kPageChunkMin) return static_cast<uint8_t*>(::operator new(capacity));
    if (capacity == kPageChunkMin) {
        ChunkCache& cache = chunk_cache();
        if (!cache.closed && cache.count > 0) return cache.chunks[--cache.count];
    }
#if defined(_WIN32)
    void* p = VirtualAlloc(nullptr, capacity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) throw std::bad_alloc();
#else
    void* p = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
#endif
    return static_cast<uint8_t*>(p);
}

void chunk_free(uint8_t* memory, size_t capacity) {
    if (capacity < kPageChunkMin) {
        ::operator delete(memory);
        return;
    }
    if (capacity == kPageChunkMin) {
        ChunkCache& cache = chunk_cache();
        if (!cache.closed && cache.count < kCacheChunks) {
            cache.chunks[cache.count++] = memory;
            return;
        }
    }
    pages_free(memory, capacity);
}

} // namespace

Arena::Arena(size_t default_chunk_size)
    : default_chunk_size_(default_chunk_size) {}

Arena::Arena(size_t first_chunk_size, size_t max_chunk_size)
    : default_chunk_size_(first_chunk_size), max_chunk_size_(max_chunk_size) {}

Arena::~Arena() {
    clear();
}

Arena::Arena(Arena&& other) noexcept
    : default_chunk_size_(other.default_chunk_size_),
      max_chunk_size_(other.max_chunk_size_),
      chunks_(std::move(other.chunks_)),
      current_chunk_idx_(other.current_chunk_idx_),
      bytes_allocated_(other.bytes_allocated_),
      bytes_capacity_(other.bytes_capacity_),
      cleanups_(std::move(other.cleanups_)) {
    other.current_chunk_idx_ = 0;
    other.bytes_allocated_ = 0;
    other.bytes_capacity_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        clear();
        default_chunk_size_ = other.default_chunk_size_;
        max_chunk_size_ = other.max_chunk_size_;
        chunks_ = std::move(other.chunks_);
        current_chunk_idx_ = other.current_chunk_idx_;
        bytes_allocated_ = other.bytes_allocated_;
        bytes_capacity_ = other.bytes_capacity_;
        cleanups_ = std::move(other.cleanups_);

        other.current_chunk_idx_ = 0;
        other.bytes_allocated_ = 0;
        other.bytes_capacity_ = 0;
    }
    return *this;
}

void Arena::allocate_new_chunk(size_t min_size) {
    size_t chunk_size = default_chunk_size_;
    if (max_chunk_size_ > default_chunk_size_) {
        chunk_size = std::min(max_chunk_size_, std::max(default_chunk_size_, bytes_capacity_));
    }
    chunk_size = chunk_capacity(std::max(chunk_size, min_size));
    uint8_t* mem = chunk_alloc(chunk_size);
    chunks_.push_back(Chunk{mem, chunk_size, 0});
    bytes_capacity_ += chunk_size;
    current_chunk_idx_ = chunks_.size() - 1;
}

void* Arena::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }

    if (chunks_.empty()) {
        allocate_new_chunk(size + alignment);
    }

    while (current_chunk_idx_ < chunks_.size()) {
        Chunk& chunk = chunks_[current_chunk_idx_];
        uintptr_t current_addr = reinterpret_cast<uintptr_t>(chunk.memory + chunk.used);
        uintptr_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
        size_t padding = static_cast<size_t>(aligned_addr - current_addr);

        if (chunk.used + padding + size <= chunk.capacity) {
            chunk.used += padding + size;
            bytes_allocated_ += padding + size;
            return reinterpret_cast<void*>(aligned_addr);
        }

        // Try next existing chunk if available
        if (current_chunk_idx_ + 1 < chunks_.size()) {
            current_chunk_idx_++;
            chunks_[current_chunk_idx_].used = 0;
        } else {
            break;
        }
    }

    // Need a new chunk
    allocate_new_chunk(size + alignment);
    Chunk& chunk = chunks_[current_chunk_idx_];
    uintptr_t current_addr = reinterpret_cast<uintptr_t>(chunk.memory + chunk.used);
    uintptr_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
    size_t padding = static_cast<size_t>(aligned_addr - current_addr);

    chunk.used += padding + size;
    bytes_allocated_ += padding + size;
    return reinterpret_cast<void*>(aligned_addr);
}

void* Arena::allocate(size_t size) {
    return allocate(size, alignof(std::max_align_t));
}

Arena::Marker Arena::get_marker() const noexcept {
    if (chunks_.empty()) {
        return Marker{0, 0, cleanups_.size()};
    }
    return Marker{current_chunk_idx_, chunks_[current_chunk_idx_].used, cleanups_.size()};
}

void Arena::reset_to_marker(Marker marker) {
    while (cleanups_.size() > marker.cleanup_count) {
        auto entry = cleanups_.back();
        cleanups_.pop_back();
        entry.fn(entry.obj);
    }

    if (chunks_.empty()) {
        return;
    }

    if (marker.chunk_index < chunks_.size()) {
        current_chunk_idx_ = marker.chunk_index;
        chunks_[current_chunk_idx_].used = marker.offset;
        for (size_t i = current_chunk_idx_ + 1; i < chunks_.size(); ++i) {
            chunks_[i].used = 0;
        }

        // Recalculate bytes_allocated_
        bytes_allocated_ = 0;
        for (size_t i = 0; i <= current_chunk_idx_; ++i) {
            bytes_allocated_ += chunks_[i].used;
        }
    }
}

void Arena::reset() {
    while (!cleanups_.empty()) {
        auto entry = cleanups_.back();
        cleanups_.pop_back();
        entry.fn(entry.obj);
    }
    current_chunk_idx_ = 0;
    for (auto& chunk : chunks_) {
        chunk.used = 0;
    }
    bytes_allocated_ = 0;
}

void Arena::clear() {
    while (!cleanups_.empty()) {
        auto entry = cleanups_.back();
        cleanups_.pop_back();
        entry.fn(entry.obj);
    }
    for (auto& chunk : chunks_) {
        if (chunk.memory != nullptr) {
            chunk_free(chunk.memory, chunk.capacity);
            chunk.memory = nullptr;
        }
    }
    chunks_.clear();
    current_chunk_idx_ = 0;
    bytes_allocated_ = 0;
    bytes_capacity_ = 0;
}

} // namespace brass
