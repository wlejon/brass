// The process-wide layout registry (object.hpp). Layouts live in fixed-size
// chunks that are never moved or freed, so a reader needs no lock: an id is
// published (the count stored with release order) only after its descriptor
// is written. Registration takes a mutex.

#include <brass/gc/object.hpp>

#include <array>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace brass::gc {

namespace {

constexpr size_t kChunkBits = 8;
constexpr size_t kChunkSize = size_t{1} << kChunkBits;
constexpr size_t kMaxLayouts = 65535;
constexpr size_t kChunks = (kMaxLayouts + kChunkSize) / kChunkSize;

struct MaskKey {
    uint64_t mask;
    uint32_t tag;
    bool operator==(const MaskKey& o) const noexcept { return mask == o.mask && tag == o.tag; }
};

struct MaskKeyHash {
    size_t operator()(const MaskKey& k) const noexcept {
        uint64_t h = k.mask * 0x9E3779B97F4A7C15ULL;
        h ^= (static_cast<uint64_t>(k.tag) + 0x632BE59BD9B4E019ULL) + (h << 6) + (h >> 2);
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

struct Registry {
    std::array<std::atomic<LayoutDescriptor*>, kChunks> chunks{};
    std::atomic<size_t> count{0};
    std::mutex mutex;
    std::unordered_map<MaskKey, LayoutId, MaskKeyHash> masks;

    Registry() {
        // Layout 0: a leaf with type tag 0.
        LayoutDescriptor leaf;
        leaf.kind = LayoutKind::Leaf;
        leaf.name = "leaf";
        append_locked(leaf);
        masks.emplace(MaskKey{0, 0}, kLeafLayout);
    }

    LayoutId append_locked(const LayoutDescriptor& d) {
        const size_t id = count.load(std::memory_order_relaxed);
        if (id >= kMaxLayouts) {
            throw std::length_error("brass::gc: the layout registry is full (65535 layouts)");
        }
        LayoutDescriptor* chunk = chunks[id >> kChunkBits].load(std::memory_order_relaxed);
        if (!chunk) {
            chunk = new LayoutDescriptor[kChunkSize];
            chunks[id >> kChunkBits].store(chunk, std::memory_order_release);
        }
        chunk[id & (kChunkSize - 1)] = d;
        count.store(id + 1, std::memory_order_release);
        return static_cast<LayoutId>(id);
    }
};

Registry& registry() {
    static Registry* r = new Registry();  // never destroyed: heaps may outlive static teardown
    return *r;
}

// A small per-thread cache in front of the mask table: brass_gc_alloc asks
// for the same few (mask, tag) pairs over and over.
struct MaskCacheEntry {
    uint64_t mask = 0;
    uint32_t tag = 0;
    LayoutId id = 0;
    bool valid = false;
};
constexpr size_t kMaskCacheSize = 16;
thread_local std::array<MaskCacheEntry, kMaskCacheSize> t_mask_cache{};

} // namespace

LayoutId register_layout(const LayoutDescriptor& descriptor) {
    if (descriptor.kind == LayoutKind::Custom && descriptor.trace == nullptr) {
        throw std::invalid_argument("brass::gc::register_layout: a Custom layout needs a trace function");
    }
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.append_locked(descriptor);
}

LayoutId mask_layout(uint64_t mask, uint32_t type_tag) {
    const size_t slot = static_cast<size_t>((mask ^ (mask >> 17) ^ (static_cast<uint64_t>(type_tag) * 31)) %
                                            kMaskCacheSize);
    MaskCacheEntry& cached = t_mask_cache[slot];
    if (cached.valid && cached.mask == mask && cached.tag == type_tag) return cached.id;

    Registry& r = registry();
    LayoutId id;
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        auto it = r.masks.find(MaskKey{mask, type_tag});
        if (it != r.masks.end()) {
            id = it->second;
        } else {
            LayoutDescriptor d;
            d.kind = mask == 0 ? LayoutKind::Leaf : LayoutKind::Mask;
            d.mask = mask;
            d.type_tag = type_tag;
            d.name = "mask";
            id = r.append_locked(d);
            r.masks.emplace(MaskKey{mask, type_tag}, id);
        }
    }
    cached = MaskCacheEntry{mask, type_tag, id, true};
    return id;
}

const LayoutDescriptor& layout_descriptor(LayoutId id) noexcept {
    const LayoutDescriptor* chunk = registry().chunks[id >> kChunkBits].load(std::memory_order_acquire);
    return chunk[id & (kChunkSize - 1)];
}

size_t layout_count() noexcept {
    return registry().count.load(std::memory_order_acquire);
}

} // namespace brass::gc
