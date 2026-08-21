#include <brass/core/arena.hpp>
#include <algorithm>
#include <cstdlib>

namespace brass {

Arena::Arena(size_t default_chunk_size)
    : default_chunk_size_(default_chunk_size) {}

Arena::~Arena() {
    clear();
}

Arena::Arena(Arena&& other) noexcept
    : default_chunk_size_(other.default_chunk_size_),
      chunks_(std::move(other.chunks_)),
      current_chunk_idx_(other.current_chunk_idx_),
      bytes_allocated_(other.bytes_allocated_),
      bytes_capacity_(other.bytes_capacity_) {
    other.current_chunk_idx_ = 0;
    other.bytes_allocated_ = 0;
    other.bytes_capacity_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        clear();
        default_chunk_size_ = other.default_chunk_size_;
        chunks_ = std::move(other.chunks_);
        current_chunk_idx_ = other.current_chunk_idx_;
        bytes_allocated_ = other.bytes_allocated_;
        bytes_capacity_ = other.bytes_capacity_;

        other.current_chunk_idx_ = 0;
        other.bytes_allocated_ = 0;
        other.bytes_capacity_ = 0;
    }
    return *this;
}

void Arena::allocate_new_chunk(size_t min_size) {
    size_t chunk_size = std::max(default_chunk_size_, min_size);
    uint8_t* mem = static_cast<uint8_t*>(::operator new(chunk_size));
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
        return Marker{0, 0};
    }
    return Marker{current_chunk_idx_, chunks_[current_chunk_idx_].used};
}

void Arena::reset_to_marker(Marker marker) {
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
    current_chunk_idx_ = 0;
    for (auto& chunk : chunks_) {
        chunk.used = 0;
    }
    bytes_allocated_ = 0;
}

void Arena::clear() {
    for (auto& chunk : chunks_) {
        if (chunk.memory != nullptr) {
            ::operator delete(chunk.memory);
            chunk.memory = nullptr;
        }
    }
    chunks_.clear();
    current_chunk_idx_ = 0;
    bytes_allocated_ = 0;
    bytes_capacity_ = 0;
}

} // namespace brass
