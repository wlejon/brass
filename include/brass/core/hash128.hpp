#pragma once
// A stable 128-bit content hash: MurmurHash3 x64/128 (public domain, Austin
// Appleby), fed incrementally. Stable means the same bytes give the same
// digest on every platform, build and run, so a digest can name a file that
// another process, or a later build, looks up. It is not a cryptographic
// hash: it separates content that differs by accident, not content an
// adversary chose to collide.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace brass {

struct Digest128 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    bool operator==(const Digest128& o) const noexcept { return lo == o.lo && hi == o.hi; }
    bool operator!=(const Digest128& o) const noexcept { return !(*this == o); }
    bool operator<(const Digest128& o) const noexcept { return hi != o.hi ? hi < o.hi : lo < o.lo; }

    // 32 lowercase hex digits, `hi` first.
    std::string hex() const;
    // The inverse of hex(); false for anything but 32 hex digits.
    static bool from_hex(std::string_view text, Digest128& out) noexcept;
};

// MurmurHash3 x64/128 over bytes given in any number of pieces: the digest of
// a sequence of update() calls is the digest of their concatenation.
class Hasher128 {
public:
    explicit Hasher128(uint64_t seed = 0) noexcept : h1_(seed), h2_(seed) {}

    void update(const void* data, size_t size) noexcept;
    void update(std::string_view bytes) noexcept { update(bytes.data(), bytes.size()); }
    // Little-endian, whatever the host.
    void update_u64(uint64_t v) noexcept;

    // The digest of everything given so far; the hasher may go on.
    Digest128 finish() const noexcept;

private:
    void block(const unsigned char* p) noexcept;

    uint64_t h1_;
    uint64_t h2_;
    uint64_t length_ = 0;
    unsigned char tail_[16] = {};
    size_t tail_size_ = 0;
};

Digest128 hash128(const void* data, size_t size, uint64_t seed = 0) noexcept;
inline Digest128 hash128(std::string_view bytes, uint64_t seed = 0) noexcept {
    return hash128(bytes.data(), bytes.size(), seed);
}

} // namespace brass
