#pragma once

// An open-addressing hash map from pointers to small payloads: one array of
// slots, no node per entry.
//
// Compiler passes key per-compile tables by instruction or value pointer, a
// table rebuilt per pass and sometimes per iteration. std::unordered_map
// allocates a node for every entry, and on the eight compile threads those
// allocations (and the frees after) were a third of each thread's time,
// most of it waiting on the allocator. Objects with a per-function id use
// DenseIdMap (brass/mir/dense_id_map.hpp); this is for those without one.
//
// Keys are non-null pointers. Linear probing, at most half full; erase
// shifts the probe run back, so there are no tombstones.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace brass {

template <typename Key, typename T>
class PtrMap {
public:
    PtrMap() = default;
    explicit PtrMap(size_t expected) { reserve(expected); }

    void reserve(size_t expected) {
        size_t cap = 16;
        while (cap < expected * 2) cap <<= 1;
        if (cap > keys_.size()) rehash(cap);
    }

    // Inserts `v` for `k`; false (and nothing changes) when `k` is present.
    bool insert(const Key* k, const T& v) {
        if ((size_ + 1) * 2 > keys_.size()) rehash(keys_.empty() ? 16 : keys_.size() * 2);
        size_t i = slot_of(k);
        while (keys_[i]) {
            if (keys_[i] == k) return false;
            i = (i + 1) & mask_;
        }
        keys_[i] = k;
        vals_[i] = v;
        ++size_;
        return true;
    }

    // The payload of `k`, inserting `init` first when it is absent.
    T& get_or_insert(const Key* k, const T& init) {
        if (T* found = find(k)) return *found;
        insert(k, init);
        return *find(k);
    }
    T& operator[](const Key* k) { return get_or_insert(k, T{}); }

    T* find(const Key* k) {
        if (size_ == 0) return nullptr;
        for (size_t i = slot_of(k); keys_[i]; i = (i + 1) & mask_) {
            if (keys_[i] == k) return &vals_[i];
        }
        return nullptr;
    }
    const T* find(const Key* k) const { return const_cast<PtrMap*>(this)->find(k); }
    bool contains(const Key* k) const { return find(k) != nullptr; }
    T value_or(const Key* k, const T& absent) const {
        const T* found = find(k);
        return found ? *found : absent;
    }

    // Removes `k`; false when it was absent.
    bool erase(const Key* k) {
        if (size_ == 0) return false;
        size_t i = slot_of(k);
        while (keys_[i] != k) {
            if (!keys_[i]) return false;
            i = (i + 1) & mask_;
        }
        // Backward shift: pull each later entry of the run into the hole
        // unless its home lies cyclically in (hole, entry].
        for (size_t j = (i + 1) & mask_; keys_[j]; j = (j + 1) & mask_) {
            const size_t home = slot_of(keys_[j]);
            const bool stays = i <= j ? (i < home && home <= j) : (i < home || home <= j);
            if (stays) continue;
            keys_[i] = keys_[j];
            vals_[i] = std::move(vals_[j]);
            i = j;
        }
        keys_[i] = nullptr;
        vals_[i] = T{};
        --size_;
        return true;
    }

    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // Drops every entry and keeps the storage for the next fill.
    void clear() {
        if (size_ == 0) return;
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i]) {
                keys_[i] = nullptr;
                vals_[i] = T{};
            }
        }
        size_ = 0;
    }

    // Every (key, payload) pair, in no particular order.
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i]) fn(keys_[i], vals_[i]);
        }
    }

private:
    size_t slot_of(const Key* k) const noexcept {
        // Fibonacci hashing: the multiply spreads the pointer's middle bits
        // (the low ones are alignment) over the top, which the shift keeps.
        const uint64_t h = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(k)) * 0x9E3779B97F4A7C15ull;
        return static_cast<size_t>(h >> shift_);
    }

    void rehash(size_t cap) {
        std::vector<const Key*> old_keys(cap, nullptr);
        std::vector<T> old_vals(cap);
        old_keys.swap(keys_);
        old_vals.swap(vals_);
        mask_ = cap - 1;
        shift_ = 64 - static_cast<unsigned>(std::countr_zero(cap));
        size_ = 0;
        for (size_t i = 0; i < old_keys.size(); ++i) {
            if (old_keys[i]) insert(old_keys[i], std::move(old_vals[i]));
        }
    }

    std::vector<const Key*> keys_;
    std::vector<T> vals_;
    size_t mask_ = 0;
    unsigned shift_ = 64;
    size_t size_ = 0;
};

} // namespace brass
