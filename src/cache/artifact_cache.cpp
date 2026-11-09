// The cache directory: checked lookups, atomic stores, and the walk that
// holds the directory to its budget (artifact_cache.hpp,
// docs/artifact_cache.md "Size and eviction").
#include <brass/cache/artifact_cache.hpp>

#include "cache/artifact_format.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <system_error>

namespace brass::cache {

namespace {

// A write abandoned this long ago (its writer died between write and
// rename) is deleted by the next walk; a younger one is a write in progress.
constexpr auto kAbandonedAfter = std::chrono::minutes(10);

using KeyName = std::array<char, 32>;

// The key of an entry named `<32 lowercase hex digits>.bac`, or false for a
// name the cache does not write.
bool key_of_name(const std::string& name, KeyName& key) {
    const size_t ext_len = std::strlen(kEntryExt);
    if (name.size() != key.size() + ext_len || name.compare(key.size(), ext_len, kEntryExt) != 0) return false;
    for (size_t i = 0; i < key.size(); ++i) {
        const char c = name[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        key[i] = c;
    }
    return true;
}

} // namespace

ArtifactCache::ArtifactCache(std::filesystem::path dir, uint64_t max_bytes)
    : dir_(std::move(dir)), max_bytes_(max_bytes) {}

ArtifactCache::~ArtifactCache() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

std::filesystem::path ArtifactCache::path_for(const CacheKey& key) const {
    return dir_ / (key.id.hex() + kEntryExt);
}

std::optional<Artifact> ArtifactCache::get(const CacheKey& key, Miss* why) {
    const std::filesystem::path path = path_for(key);
    Miss m = Miss::None;
    std::optional<Artifact> art = ArtifactCodec::read(path, key, &m);
    if (why) *why = m;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (art) {
            ++stats_.hits;
        } else {
            ++stats_.misses;
        }
    }
    if (art) {
        // Used now: the eviction orders by this. The entry may have been
        // replaced or deleted since the read; either way nothing is lost.
        std::error_code ec;
        std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
    }
    return art;
}

bool ArtifactCache::put(const CacheKey& key, std::span<const BlobRef> blobs, std::string* error) {
    auto fail = [&] {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.store_failures;
        return false;
    };
    if (!key.valid) {
        if (error) *error = "the key is not valid";
        return fail();
    }
    const std::vector<uint8_t> bytes = ArtifactCodec::encode(key, blobs, error);
    if (bytes.empty()) return fail();
    // Each time: the directory may have been deleted behind the cache's back.
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (!detail::write_file_atomically(path_for(key), bytes, error)) return fail();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.stores;
    }
    request_trim(key.id.hex(), disk_bytes(bytes.size()));
    return true;
}

bool ArtifactCache::remove(const CacheKey& key) {
    std::error_code ec;
    return std::filesystem::remove(path_for(key), ec);
}

ArtifactCache::Stats ArtifactCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

// ---------------------------------------------------------------------------
// Trimming
//
// Per instance: what its last walk found plus what it stored since. A store
// asks for a walk when that passes the budget, or when this instance has not
// walked yet (it has not seen what other processes left). Other processes'
// stores are not in the estimate; its next walk sees them.
// ---------------------------------------------------------------------------

void ArtifactCache::request_trim(const std::string& stored_hex, uint64_t stored_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_stored_ = stored_hex;
    if (walking_) stored_during_walk_.insert(stored_hex);
    estimate_ += stored_bytes;
    if (max_bytes_ == 0) return;
    if (walked_ && estimate_ <= max_bytes_) return;
    pending_ = true;
    if (!thread_started_) {
        thread_started_ = true;
        thread_ = std::thread([this] { trim_thread(); });
    }
    cv_.notify_all();
}

void ArtifactCache::trim_thread() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        cv_.wait(lock, [&] { return pending_ || stopping_; });
        if (stopping_) break;
        pending_ = false;
        busy_ = true;
        lock.unlock();
        walk();
        lock.lock();
        busy_ = false;
        cv_.notify_all();
    }
    pending_ = false;
    cv_.notify_all();
}

void ArtifactCache::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return (!pending_ && !busy_) || stopping_; });
}

uint64_t ArtifactCache::trim() { return walk(); }

bool ArtifactCache::is_protected(const std::string& hex) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hex == last_stored_ || hex == walk_kept_ || stored_during_walk_.count(hex) != 0;
}

uint64_t ArtifactCache::walk() {
    std::lock_guard<std::mutex> one_walk(walk_mutex_);
    uint64_t estimate_before = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        walking_ = true;
        walk_kept_ = last_stored_;
        stored_during_walk_.clear();
        estimate_before = estimate_;
        ++stats_.trims;
    }

    // Each file as a fixed-size record, not a path: a full cache is tens of
    // thousands of files, and this may run on a thread of its own whose
    // allocations stay resident.
    struct Item {
        KeyName key;
        uint64_t bytes = 0;
        std::filesystem::file_time_type mtime;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        // Any of these may fail for a file deleted under the walk: skip it.
        std::error_code e2;
        if (!it->is_regular_file(e2) || e2) continue;
        const std::string name = it->path().filename().string();
        const auto mtime = it->last_write_time(e2);
        if (e2) continue;
        const uint64_t size = it->file_size(e2);
        if (e2) continue;
        if (name.find(kTmpMarker) != std::string::npos) {
            if (now - mtime > kAbandonedAfter) {
                std::filesystem::remove(it->path(), e2);
            } else {
                total += disk_bytes(size);
            }
            continue;
        }
        KeyName key;
        if (!key_of_name(name, key)) continue;   // not the cache's: left alone, not counted
        items.push_back({key, disk_bytes(size), mtime});
        total += disk_bytes(size);
    }

    uint64_t evicted = 0;
    if (max_bytes_ != 0 && total > max_bytes_) {
        const uint64_t target = max_bytes_ - max_bytes_ / 8;
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.mtime < b.mtime; });
        for (const Item& item : items) {
            if (total <= target) break;
            const std::string hex(item.key.data(), item.key.size());
            if (is_protected(hex)) continue;
            std::error_code e2;
            const bool removed = std::filesystem::remove(dir_ / (hex + kEntryExt), e2);
            // Removed now, or already gone: either way it no longer counts.
            // An error (Windows: open without delete sharing) leaves it.
            if (removed || !e2) total -= item.bytes;
            if (removed) ++evicted;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    walking_ = false;
    walk_kept_.clear();
    stored_during_walk_.clear();
    // Stores made during the walk count on top (some it saw: an
    // over-estimate only asks for the next walk sooner).
    estimate_ = total + (estimate_ > estimate_before ? estimate_ - estimate_before : 0);
    walked_ = true;
    stats_.evictions += evicted;
    return total;
}

} // namespace brass::cache
