#include <brass/core/scratch_alloc.hpp>

#include <cstdio>
#include <cstdlib>

namespace brass {

namespace {
thread_local ScratchPool tls_pool;
} // namespace

ScratchPool& ScratchPool::local() noexcept { return tls_pool; }

ScratchPool::~ScratchPool() {
    // Blocks still live at thread exit (a scratch container with static
    // storage, which none should be) keep their chunks: freeing them would
    // hand those blocks' memory to someone else.
    if (live_bytes_ != 0) return;
    for (void* c : chunks_) ::operator delete(c);
}

size_t ScratchPool::class_of(size_t bytes) noexcept {
    if (bytes <= 256) return bytes == 0 ? 0 : (bytes - 1) / 16;
    // 512 -> 16, 1024 -> 17, ... 64 KiB -> 23.
    size_t cls = 16;
    for (size_t cap = 512; cap < bytes; cap <<= 1) ++cls;
    return cls;
}

size_t ScratchPool::class_bytes(size_t cls) noexcept {
    return cls < 16 ? (cls + 1) * 16 : size_t{512} << (cls - 16);
}

void* ScratchPool::allocate(size_t bytes, size_t align) {
    if (bytes > kMaxPooled || align > alignof(std::max_align_t)) {
        live_bytes_ += bytes;
        return ::operator new(bytes, std::align_val_t(align));
    }
    const size_t cls = class_of(bytes);
    live_bytes_ += class_bytes(cls);
    if (FreeBlock* b = free_[cls]) {
        free_[cls] = b->next;
        return b;
    }
    return refill(cls);
}

void* ScratchPool::refill(size_t cls) {
    const size_t size = class_bytes(cls);
    if (static_cast<size_t>(bump_end_ - bump_) < size) {
        // The next chunk (the first after a reset), made when there is none.
        // The rest of the chunk left behind goes unused until the pool
        // resets; it is less than one block of the largest class.
        const size_t next = bump_ ? chunk_in_use_ + 1 : 0;
        if (next == chunks_.size()) chunks_.push_back(::operator new(kChunkBytes));
        chunk_in_use_ = next;
        bump_ = static_cast<char*>(chunks_[next]);
        bump_end_ = bump_ + kChunkBytes;
    }
    void* p = bump_;
    bump_ += size;
    return p;
}

void ScratchPool::deallocate(void* p, size_t bytes, size_t align) noexcept {
    if (this != &tls_pool) {
        std::fprintf(stderr, "brass: a scratch block was freed on a thread other than the one that allocated it\n");
        std::abort();
    }
    if (bytes > kMaxPooled || align > alignof(std::max_align_t)) {
        ::operator delete(p, std::align_val_t(align));
        live_bytes_ -= bytes;
    } else {
        const size_t cls = class_of(bytes);
        auto* b = static_cast<FreeBlock*>(p);
        b->next = free_[cls];
        free_[cls] = b;
        live_bytes_ -= class_bytes(cls);
    }
    if (live_bytes_ == 0) reset();
}

namespace {
constexpr size_t kOwnerHeader = alignof(std::max_align_t);
} // namespace

void* ScratchPool::allocate_owned(size_t bytes) {
    ScratchPool& pool = local();
    char* block = static_cast<char*>(pool.allocate(bytes + kOwnerHeader, alignof(std::max_align_t)));
    *reinterpret_cast<ScratchPool**>(block) = &pool;
    return block + kOwnerHeader;
}

void ScratchPool::deallocate_owned(void* p, size_t bytes) noexcept {
    if (!p) return;
    char* block = static_cast<char*>(p) - kOwnerHeader;
    (*reinterpret_cast<ScratchPool**>(block))->deallocate(block, bytes + kOwnerHeader, alignof(std::max_align_t));
}

void ScratchPool::reset() noexcept {
    // Nothing is live: every block is free, so the chunks start over empty.
    for (FreeBlock*& f : free_) f = nullptr;
    const size_t keep = kRetainedBytes / kChunkBytes;
    while (chunks_.size() > keep) {
        ::operator delete(chunks_.back());
        chunks_.pop_back();
    }
    chunk_in_use_ = 0;
    bump_ = nullptr;
    bump_end_ = nullptr;
}

} // namespace brass
