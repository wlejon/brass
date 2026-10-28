#pragma once

// A per-thread pool for the short-lived containers of a compile: liveness
// sets, intervals, pass-local maps. Freed blocks go onto a free list of
// their size class and the next allocation of that size takes one back, so
// a pass that builds and drops the same tables per function never reaches
// the system allocator once the pool is warm.
//
// Why: on the eight compile threads, glibc's per-thread cache holds seven
// blocks per size, and every allocation or free past that took its arena's
// lock. A tier-2 compile of a big function makes hundreds of thousands of
// node allocations, so those locks (and the heap growth behind them: one
// mprotect per page) were a third of each compile thread's time.
//
// A block belongs to the thread that allocated it: ScratchAllocator records
// its pool, and freeing it on another thread is a fatal error rather than a
// corrupted free list. So only containers that live and die inside one
// compile on one thread may use it. When a thread's live scratch bytes fall
// to zero the pool resets, keeping kRetainedBytes of chunks for the next
// compile and returning the rest.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <new>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

class ScratchPool {
public:
    // The calling thread's pool.
    static ScratchPool& local() noexcept;

    void* allocate(size_t bytes, size_t align);
    void deallocate(void* p, size_t bytes, size_t align) noexcept;

    // For a class's operator new / delete, which has no allocator to carry
    // the pool: the block records its pool in a header, so a delete on
    // another thread is caught (a fatal error) like any other.
    static void* allocate_owned(size_t bytes);
    static void deallocate_owned(void* p, size_t bytes) noexcept;

    ScratchPool() = default;
    ~ScratchPool();
    ScratchPool(const ScratchPool&) = delete;
    ScratchPool& operator=(const ScratchPool&) = delete;

private:
    // Classes of 16-byte steps up to 256 bytes, then powers of two to
    // kMaxPooled; anything larger is the system allocator's (a large block
    // is rare, and a free list of them would hold memory for nothing).
    static constexpr size_t kMaxPooled = 64 * 1024;
    static constexpr size_t kClasses = 16 + 8;
    static constexpr size_t kChunkBytes = 256 * 1024;
    static constexpr size_t kRetainedBytes = 4 * 1024 * 1024;

    static size_t class_of(size_t bytes) noexcept;
    static size_t class_bytes(size_t cls) noexcept;
    void* refill(size_t cls);
    void reset() noexcept;

    struct FreeBlock { FreeBlock* next; };
    FreeBlock* free_[kClasses] = {};
    std::vector<void*> chunks_;   // every chunk, oldest first
    size_t chunk_in_use_ = 0;     // index of the chunk being carved
    char* bump_ = nullptr;
    char* bump_end_ = nullptr;
    size_t live_bytes_ = 0;
};

template <typename T>
class ScratchAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    ScratchAllocator() noexcept : pool_(&ScratchPool::local()) {}
    template <typename U>
    ScratchAllocator(const ScratchAllocator<U>& other) noexcept : pool_(other.pool()) {}

    T* allocate(size_t n) { return static_cast<T*>(pool_->allocate(n * sizeof(T), alignof(T))); }
    void deallocate(T* p, size_t n) noexcept { pool_->deallocate(p, n * sizeof(T), alignof(T)); }

    // A copy made on another thread draws from that thread's pool.
    ScratchAllocator select_on_container_copy_construction() const noexcept { return ScratchAllocator(); }

    ScratchPool* pool() const noexcept { return pool_; }

    template <typename U>
    bool operator==(const ScratchAllocator<U>& other) const noexcept { return pool_ == other.pool(); }

private:
    ScratchPool* pool_;
};

template <typename T>
using ScratchVector = std::vector<T, ScratchAllocator<T>>;
template <typename K, typename V, typename H = std::hash<K>, typename E = std::equal_to<K>>
using ScratchMap = std::unordered_map<K, V, H, E, ScratchAllocator<std::pair<const K, V>>>;
template <typename K, typename H = std::hash<K>, typename E = std::equal_to<K>>
using ScratchSet = std::unordered_set<K, H, E, ScratchAllocator<K>>;
template <typename K, typename V, typename C = std::less<K>>
using ScratchOrderedMap = std::map<K, V, C, ScratchAllocator<std::pair<const K, V>>>;
template <typename K, typename C = std::less<K>>
using ScratchOrderedSet = std::set<K, C, ScratchAllocator<K>>;

} // namespace brass
