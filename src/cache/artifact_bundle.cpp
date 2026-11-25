// Bundle export/import & key enumeration over brass::cache::ArtifactCache.
#include <brass/cache/artifact_cache.hpp>

#include "cache/artifact_format.hpp"

#include <algorithm>
#include <cstring>
#include <system_error>
#include <vector>

namespace brass::cache {

namespace {

constexpr uint32_t kBundleMagic = 0x42434142u; // "BACB"
constexpr uint32_t kBundleVersion = 1;
constexpr size_t kBundleHeaderBytes = 24;      // magic(4) + ver(4) + meta_len(8) + entry_count(8)
constexpr size_t kBundleTrailerBytes = 16;     // Murmur3-128 checksum
constexpr uint64_t kBundleSeed = 0x6261636263686b31ULL;

void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (i * 8)));
}

void append_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (i * 8)));
}

uint32_t read_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

bool is_entry_filename(const std::string& name) {
    const size_t ext_len = std::strlen(kEntryExt);
    if (name.size() != 32 + ext_len || name.compare(32, ext_len, kEntryExt) != 0) return false;
    for (size_t i = 0; i < 32; ++i) {
        const char c = name[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

void set_err(std::string* err, std::string msg) {
    if (err) *err = std::move(msg);
}

} // namespace

std::vector<CacheKey> ArtifactCache::list_keys() const {
    std::vector<CacheKey> keys;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2) || e2) continue;
        const std::string name = it->path().filename().string();
        if (!is_entry_filename(name)) continue;
        CacheKey key;
        if (ArtifactCodec::read_any(it->path(), key, nullptr).has_value() && key.valid) {
            keys.push_back(key);
        }
    }
    std::sort(keys.begin(), keys.end(), [](const CacheKey& a, const CacheKey& b) {
        if (a.id.hi != b.id.hi) return a.id.hi < b.id.hi;
        if (a.id.lo != b.id.lo) return a.id.lo < b.id.lo;
        if (a.check.hi != b.check.hi) return a.check.hi < b.check.hi;
        return a.check.lo < b.check.lo;
    });
    return keys;
}

bool ArtifactCache::export_bundle(
    const std::filesystem::path& out_path,
    std::string_view metadata,
    BundleStats* stats,
    std::string* error) const {
    BundleStats st;
    if (stats) *stats = st;

    std::vector<std::filesystem::path> entry_paths;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2) || e2) continue;
        if (is_entry_filename(it->path().filename().string())) {
            entry_paths.push_back(it->path());
        }
    }
    std::sort(entry_paths.begin(), entry_paths.end());

    struct VerifiedEntry {
        std::vector<uint8_t> raw;
    };
    std::vector<VerifiedEntry> entries;
    entries.reserve(entry_paths.size());
    for (const auto& p : entry_paths) {
        std::vector<uint8_t> raw;
        if (!detail::read_whole_file(p, raw)) continue;
        CacheKey k;
        if (!ArtifactCodec::decode(raw, k, nullptr).has_value() || !k.valid) continue;
        entries.push_back({std::move(raw)});
    }

    std::vector<uint8_t> buf;
    buf.reserve(kBundleHeaderBytes + metadata.size() + kBundleTrailerBytes + entries.size() * 256);
    append_u32(buf, kBundleMagic);
    append_u32(buf, kBundleVersion);
    append_u64(buf, static_cast<uint64_t>(metadata.size()));
    append_u64(buf, static_cast<uint64_t>(entries.size()));
    if (!metadata.empty()) {
        buf.insert(buf.end(),
                   reinterpret_cast<const uint8_t*>(metadata.data()),
                   reinterpret_cast<const uint8_t*>(metadata.data()) + metadata.size());
    }
    for (const auto& e : entries) {
        append_u64(buf, static_cast<uint64_t>(e.raw.size()));
        buf.insert(buf.end(), e.raw.begin(), e.raw.end());
    }

    Hasher128 hasher(kBundleSeed);
    hasher.update(buf.data(), buf.size());
    const Digest128 sum = hasher.finish();
    append_u64(buf, sum.lo);
    append_u64(buf, sum.hi);

    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    if (!detail::write_file_atomically(out_path, buf, error)) {
        return false;
    }

    st.entries_written = entries.size();
    st.bytes = buf.size();
    if (stats) *stats = st;
    return true;
}

bool ArtifactCache::import_bundle(
    const std::filesystem::path& in_path,
    std::string* metadata_out,
    BundleStats* stats,
    std::string* error) {
    BundleStats st;
    if (stats) *stats = st;

    std::vector<uint8_t> buf;
    if (!detail::read_whole_file(in_path, buf)) {
        set_err(error, "cannot read bundle file: " + in_path.string());
        return false;
    }
    if (buf.size() < kBundleHeaderBytes + kBundleTrailerBytes) {
        set_err(error, "bundle file is truncated");
        return false;
    }
    const size_t body_len = buf.size() - kBundleTrailerBytes;
    Hasher128 hasher(kBundleSeed);
    hasher.update(buf.data(), body_len);
    const Digest128 computed = hasher.finish();
    const Digest128 stored{read_u64(buf.data() + body_len), read_u64(buf.data() + body_len + 8)};
    if (computed != stored) {
        set_err(error, "bundle checksum mismatch");
        return false;
    }

    const uint32_t magic = read_u32(buf.data());
    const uint32_t version = read_u32(buf.data() + 4);
    const uint64_t meta_len = read_u64(buf.data() + 8);
    const uint64_t entry_count = read_u64(buf.data() + 16);
    if (magic != kBundleMagic || version != kBundleVersion) {
        set_err(error, "unrecognized bundle format or version");
        return false;
    }
    if (meta_len > body_len - kBundleHeaderBytes) {
        set_err(error, "bundle metadata length out of range");
        return false;
    }

    size_t at = kBundleHeaderBytes;
    const std::string meta(reinterpret_cast<const char*>(buf.data() + at), static_cast<size_t>(meta_len));
    at += static_cast<size_t>(meta_len);

    struct ParsedRecord {
        CacheKey key;
        std::optional<Artifact> art;
        std::vector<BlobRef> refs;
        std::span<const uint8_t> raw;
    };
    std::vector<ParsedRecord> decoded;
    decoded.reserve(static_cast<size_t>(std::min<uint64_t>(entry_count, 65536)));
    for (uint64_t i = 0; i < entry_count; ++i) {
        if (body_len - at < 8) {
            set_err(error, "bundle entry table is truncated");
            return false;
        }
        const uint64_t rec_len = read_u64(buf.data() + at);
        at += 8;
        if (rec_len > body_len - at) {
            set_err(error, "bundle entry payload is truncated");
            return false;
        }
        std::span<const uint8_t> raw(buf.data() + at, static_cast<size_t>(rec_len));
        at += static_cast<size_t>(rec_len);

        CacheKey key;
        Miss miss = Miss::None;
        auto art = ArtifactCodec::decode(raw, key, &miss);
        if (!art || !key.valid) {
            set_err(error, std::string("corrupt bundle entry: ") + to_string(miss));
            return false;
        }
        ParsedRecord rec;
        rec.key = key;
        rec.raw = raw;
        decoded.push_back(std::move(rec));
    }
    if (at != body_len) {
        set_err(error, "bundle has trailing unframed bytes");
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    for (const auto& rec : decoded) {
        const std::filesystem::path dst = path_for(rec.key);
        if (ArtifactCodec::read(dst, rec.key, nullptr).has_value()) {
            ++st.entries_skipped;
            continue;
        }
        if (!detail::write_file_atomically(dst, rec.raw, error)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.stores;
        }
        request_trim(rec.key.id.hex(), disk_bytes(rec.raw.size()));
        ++st.entries_imported;
    }

    if (metadata_out) *metadata_out = meta;
    st.bytes = buf.size();
    if (stats) *stats = st;
    return true;
}

} // namespace brass::cache
