#pragma once

#include <brass/core/span.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <memory>
#include <type_traits>
#include <new>

namespace brass {

class Arena {
public:
    static constexpr size_t DefaultChunkSize = 64 * 1024; // 64 KB

    using CleanupFn = void (*)(void*);
    struct CleanupEntry {
        CleanupFn fn;
        void* obj;
    };

    struct Marker {
        size_t chunk_index = 0;
        size_t offset = 0;
        size_t cleanup_count = 0;
    };

    explicit Arena(size_t default_chunk_size = DefaultChunkSize);
    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    void* allocate(size_t size, size_t alignment);
    void* allocate(size_t size);

    void register_cleanup(CleanupFn fn, void* obj) {
        cleanups_.push_back(CleanupEntry{fn, obj});
    }

    [[nodiscard]] size_t cleanup_count() const noexcept { return cleanups_.size(); }

    template <typename T, typename... Args>
    T* make(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        T* obj = ::new (mem) T(std::forward<Args>(args)...);
        if constexpr (!std::is_trivially_destructible_v<T>) {
            register_cleanup([](void* ptr) {
                static_cast<T*>(ptr)->~T();
            }, obj);
        }
        return obj;
    }

    template <typename T>
    Span<T> allocate_span(size_t count) {
        if (count == 0) {
            return Span<T>();
        }
        void* mem = allocate(sizeof(T) * count, alignof(T));
        T* ptr = static_cast<T*>(mem);
        if constexpr (!std::is_trivially_default_constructible_v<T>) {
            for (size_t i = 0; i < count; ++i) {
                ::new (static_cast<void*>(ptr + i)) T();
            }
        }
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = count; i > 0; --i) {
                register_cleanup([](void* p) {
                    static_cast<T*>(p)->~T();
                }, ptr + (i - 1));
            }
        }
        return Span<T>(ptr, count);
    }

    template <typename T>
    Span<T> copy_span(Span<const T> src) {
        if (src.empty()) {
            return Span<T>();
        }
        Span<T> dst = allocate_span<T>(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] = src[i];
        }
        return dst;
    }

    template <typename T>
    Span<T> copy_span(Span<T> src) {
        return copy_span(Span<const T>(src.data(), src.size()));
    }

    template <typename T>
    Span<T> copy_span(const std::vector<T>& src) {
        return copy_span(Span<const T>(src.data(), src.size()));
    }

    Marker get_marker() const noexcept;
    void reset_to_marker(Marker marker);
    void reset();
    void clear();

    size_t bytes_allocated() const noexcept { return bytes_allocated_; }
    size_t bytes_capacity() const noexcept { return bytes_capacity_; }
    size_t chunk_count() const noexcept { return chunks_.size(); }

private:
    struct Chunk {
        uint8_t* memory = nullptr;
        size_t capacity = 0;
        size_t used = 0;
    };

    void allocate_new_chunk(size_t min_size);

    size_t default_chunk_size_;
    std::vector<Chunk> chunks_;
    size_t current_chunk_idx_ = 0;
    size_t bytes_allocated_ = 0;
    size_t bytes_capacity_ = 0;
    std::vector<CleanupEntry> cleanups_;
};

class ScopedArenaReset {
public:
    explicit ScopedArenaReset(Arena& arena)
        : arena_(arena), marker_(arena.get_marker()) {}

    ~ScopedArenaReset() {
        arena_.reset_to_marker(marker_);
    }

    ScopedArenaReset(const ScopedArenaReset&) = delete;
    ScopedArenaReset& operator=(const ScopedArenaReset&) = delete;

private:
    Arena& arena_;
    Arena::Marker marker_;
};

} // namespace brass
