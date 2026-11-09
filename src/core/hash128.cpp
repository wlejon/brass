#include <brass/core/hash128.hpp>

#include <cstring>

namespace brass {

namespace {

constexpr uint64_t kC1 = 0x87c37b91114253d5ULL;
constexpr uint64_t kC2 = 0x4cf5ad432745937fULL;

inline uint64_t rotl64(uint64_t x, int r) noexcept { return (x << r) | (x >> (64 - r)); }

inline uint64_t fmix64(uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// Little-endian load, so the digest does not depend on the host's order.
inline uint64_t load_le64(const unsigned char* p) noexcept {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

int hex_value(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

} // namespace

std::string Digest128::hex() const {
    static const char* kDigits = "0123456789abcdef";
    std::string s(32, '0');
    for (int i = 0; i < 16; ++i) {
        s[static_cast<size_t>(15 - i)] = kDigits[(hi >> (i * 4)) & 15];
        s[static_cast<size_t>(31 - i)] = kDigits[(lo >> (i * 4)) & 15];
    }
    return s;
}

bool Digest128::from_hex(std::string_view text, Digest128& out) noexcept {
    if (text.size() != 32) return false;
    uint64_t parts[2] = {0, 0};
    for (size_t i = 0; i < 32; ++i) {
        const int v = hex_value(text[i]);
        if (v < 0) return false;
        uint64_t& p = parts[i / 16];
        p = (p << 4) | static_cast<uint64_t>(v);
    }
    out.hi = parts[0];
    out.lo = parts[1];
    return true;
}

void Hasher128::block(const unsigned char* p) noexcept {
    uint64_t k1 = load_le64(p);
    uint64_t k2 = load_le64(p + 8);
    k1 *= kC1;
    k1 = rotl64(k1, 31);
    k1 *= kC2;
    h1_ ^= k1;
    h1_ = rotl64(h1_, 27);
    h1_ += h2_;
    h1_ = h1_ * 5 + 0x52dce729;
    k2 *= kC2;
    k2 = rotl64(k2, 33);
    k2 *= kC1;
    h2_ ^= k2;
    h2_ = rotl64(h2_, 31);
    h2_ += h1_;
    h2_ = h2_ * 5 + 0x38495ab5;
}

void Hasher128::update(const void* data, size_t size) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    length_ += size;
    if (tail_size_ > 0) {
        const size_t take = size < 16 - tail_size_ ? size : 16 - tail_size_;
        std::memcpy(tail_ + tail_size_, p, take);
        tail_size_ += take;
        p += take;
        size -= take;
        if (tail_size_ < 16) return;
        block(tail_);
        tail_size_ = 0;
    }
    while (size >= 16) {
        block(p);
        p += 16;
        size -= 16;
    }
    if (size > 0) {
        std::memcpy(tail_, p, size);
        tail_size_ = size;
    }
}

void Hasher128::update_u64(uint64_t v) noexcept {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (i * 8));
    update(b, 8);
}

Digest128 Hasher128::finish() const noexcept {
    uint64_t h1 = h1_;
    uint64_t h2 = h2_;
    const size_t rem = tail_size_;
    if (rem > 8) {
        uint64_t k2 = 0;
        for (size_t i = rem; i-- > 8;) k2 = (k2 << 8) | tail_[i];
        k2 *= kC2;
        k2 = rotl64(k2, 33);
        k2 *= kC1;
        h2 ^= k2;
    }
    if (rem > 0) {
        uint64_t k1 = 0;
        for (size_t i = rem < 8 ? rem : 8; i-- > 0;) k1 = (k1 << 8) | tail_[i];
        k1 *= kC1;
        k1 = rotl64(k1, 31);
        k1 *= kC2;
        h1 ^= k1;
    }
    h1 ^= length_;
    h2 ^= length_;
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;
    return Digest128{h1, h2};
}

Digest128 hash128(const void* data, size_t size, uint64_t seed) noexcept {
    Hasher128 h(seed);
    h.update(data, size);
    return h.finish();
}

} // namespace brass
