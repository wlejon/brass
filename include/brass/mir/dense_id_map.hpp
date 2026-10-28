#pragma once

// A map from MIR objects that carry a per-function id (Value, BasicBlock)
// to a small payload, stored in a vector indexed by the id.
//
// Compiler passes look up every operand of every instruction and every
// block of a CFG walk, and a hash map keyed by pointer made those lookups,
// and above all the node allocated per entry, a large share of compiling a
// function: one tier-2 compile of a big function made hundreds of thousands
// of allocations, and the allocator's locks and heap growth were a third of
// every compile thread's time. Ids are dense within a function, so a vector
// serves; ids are not a proven invariant of every producer, though (a cloner
// or a parser may repeat one, and module-level values have ids of their
// own), so a second object with an id already taken goes to a hash map
// instead, and the answer stays exact either way.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brass {

template <typename Key, typename T>
class DenseIdMap {
public:
    DenseIdMap() = default;
    explicit DenseIdMap(size_t ids) { reserve(ids); }

    void reserve(size_t ids) {
        if (owner_.size() < ids) {
            owner_.resize(ids, nullptr);
            vals_.resize(ids);
        }
    }

    // Inserts `v` for `k`; false (and nothing changes) when `k` is present.
    bool insert(const Key* k, const T& v) {
        const uint32_t id = k->id();
        if (id >= owner_.size()) {
            // An id far past any function's count is no index to size by.
            if (id >= kMaxDenseId) return spill(k, v);
            reserve(grow_to(id));
        }
        const Key*& slot = owner_[id];
        if (slot == nullptr) {
            slot = k;
            vals_[id] = v;
            ++size_;
            return true;
        }
        if (slot == k) return false;
        return spill(k, v);
    }

    // The payload of `k`, inserting `init` first when it is absent.
    T& get_or_insert(const Key* k, const T& init) {
        if (T* found = find(k)) return *found;
        insert(k, init);
        return *find(k);
    }
    T& operator[](const Key* k) { return get_or_insert(k, T{}); }

    T* find(const Key* k) {
        const uint32_t id = k->id();
        if (id < owner_.size() && owner_[id] == k) return &vals_[id];
        if (spill_.empty()) return nullptr;
        auto it = spill_.find(k);
        return it == spill_.end() ? nullptr : &it->second;
    }
    const T* find(const Key* k) const { return const_cast<DenseIdMap*>(this)->find(k); }
    bool contains(const Key* k) const { return find(k) != nullptr; }
    // The payload of `k`, or `absent` when it has none.
    T value_or(const Key* k, const T& absent) const {
        const T* found = find(k);
        return found ? *found : absent;
    }

    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // Drops every entry and keeps the storage for the next fill.
    void clear() {
        std::fill(owner_.begin(), owner_.end(), nullptr);
        spill_.clear();
        size_ = 0;
    }

    // Every (key, payload) pair, in no particular order.
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (size_t i = 0; i < owner_.size(); ++i) {
            if (owner_[i]) fn(owner_[i], vals_[i]);
        }
        for (const auto& [k, v] : spill_) fn(k, v);
    }

private:
    static constexpr uint32_t kMaxDenseId = 1u << 24;

    bool spill(const Key* k, const T& v) {
        const bool fresh = spill_.emplace(k, v).second;
        if (fresh) ++size_;
        return fresh;
    }
    static size_t grow_to(uint32_t id) {
        const size_t want = static_cast<size_t>(id) + 1;
        return want < 64 ? 64 : want + want / 2;
    }
    std::vector<const Key*> owner_;
    std::vector<T> vals_;
    std::unordered_map<const Key*, T> spill_;
    size_t size_ = 0;
};

} // namespace brass
