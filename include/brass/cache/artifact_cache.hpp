#pragma once
// A content-addressed, size-bounded artifact cache on disk, shared by every
// front end that compiles through brass (docs/artifact_cache.md).
//
// A client builds a CacheKey from named fields — everything its compile
// read, plus what brass itself contributes (KeyBuilder::add_brass_build) —
// and stores under it one or more named byte blobs (IL, SPIR-V words, a
// linked image, metadata). The store is one file per key, written to a
// temporary file and renamed into place, with a header carrying a magic,
// a format version, two independent digests of the key material and a
// checksum of everything else: a missing, foreign, torn, damaged or
// colliding file is a miss, never a wrong hit. The directory is held to a
// byte budget by deleting the least recently used entries.
//
// Every operation is safe from any number of threads of any number of
// processes sharing the directory. No lock is held across processes, and
// none across a compile: a miss returns at once and the caller compiles.

#include <brass/core/hash128.hpp>
#include <brass/target/target.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace brass::cache {

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

// What a stored artifact is the result of. `id` names the file; `check`, a
// second digest of the same material under another seed, is kept in the
// file's header and compared on every read, so two keys whose ids collide
// are still told apart (a miss, not the other key's artifact).
struct CacheKey {
    Digest128 id;
    Digest128 check;
    // False when the key could not be made (KeyBuilder::add_brass_build
    // could not identify the binary brass runs in). Such a key never hits
    // and is never stored: an artifact nobody can invalidate is not cached.
    bool valid = false;

    std::string hex() const { return id.hex(); }
    bool operator==(const CacheKey& o) const noexcept {
        return id == o.id && check == o.check && valid == o.valid;
    }
    bool operator!=(const CacheKey& o) const noexcept { return !(*this == o); }
};

// Builds a CacheKey from named fields. Each field is fed as its name and its
// value, each preceded by its length, so no two different field lists hash
// the same material ("ab","c" is not "a","bc"; a renamed field is a new key).
// Order matters: add fields in a fixed order. The domain names the client and
// its key schema ("bronze.il/3"), so two clients sharing a directory never
// share a key.
class KeyBuilder {
public:
    explicit KeyBuilder(std::string_view domain);

    KeyBuilder& add(std::string_view name, std::string_view value);
    KeyBuilder& add_bytes(std::string_view name, std::span<const uint8_t> value);
    KeyBuilder& add_u64(std::string_view name, uint64_t value);
    KeyBuilder& add_bool(std::string_view name, bool value);
    KeyBuilder& add_digest(std::string_view name, const Digest128& value);
    // Another key (both its digests), e.g. a client key a layer derives from.
    KeyBuilder& add_key(std::string_view name, const CacheKey& value);

    // What brass contributes to a key of anything it compiled or will run:
    // the cache format, brass's version, the identity of the binary holding
    // brass (path, size and modification time: any rebuild of brass, or of
    // the library or executable it is linked into, is a new key space), the
    // target, and — for the host target — the CPU features codegen reads
    // (AVX, AVX2, FMA). When the binary cannot be identified the key is
    // made invalid (CacheKey::valid), so nothing is cached rather than
    // something that cannot be invalidated.
    KeyBuilder& add_brass_build(const Target& target = Target::host());

    // The same without the binary's identity: for an artifact that must
    // outlive a rebuild of the compiler as long as its ABI holds (a module
    // compiled ahead of time and shipped). The caller then owns the ABI
    // half of the key (its runtime's ABI fingerprint, its format versions).
    KeyBuilder& add_brass_codegen(const Target& target = Target::host());

    CacheKey finish() const noexcept;

private:
    void field(std::string_view name, const void* data, size_t size);

    Hasher128 id_;
    Hasher128 check_;
    bool valid_ = true;
};

// "path|size|mtime" of the executable or shared library brass's code is in,
// measured once; empty when it cannot be found.
const std::string& brass_binary_identity();

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

// One named blob to store. Names are at most 255 bytes; a name appears once
// per artifact.
struct BlobRef {
    std::string_view name;
    std::span<const uint8_t> bytes;

    BlobRef(std::string_view n, std::span<const uint8_t> b) noexcept : name(n), bytes(b) {}
    BlobRef(std::string_view n, std::string_view text) noexcept
        : name(n), bytes(reinterpret_cast<const uint8_t*>(text.data()), text.size()) {}
    BlobRef(std::string_view n, const char* text) noexcept : BlobRef(n, std::string_view(text)) {}
    BlobRef(std::string_view n, const std::string& text) noexcept : BlobRef(n, std::string_view(text)) {}
    BlobRef(std::string_view n, const std::vector<uint8_t>& b) noexcept : name(n), bytes(b.data(), b.size()) {}
    // An array of trivially copyable values (SPIR-V words), as its bytes.
    template <typename T>
    static BlobRef of(std::string_view n, std::span<const T> values) noexcept {
        return BlobRef(n, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(values.data()),
                                                   values.size_bytes()));
    }
};

// A stored artifact as read back: it owns the bytes; the views it hands out
// live as long as it does. Every blob starts 16-byte aligned, so as<T>() of
// any T up to 16-byte alignment is a view, not a copy.
class Artifact {
public:
    size_t size() const noexcept { return blobs_.size(); }
    std::string_view name(size_t i) const noexcept { return blobs_[i].name; }
    bool has(std::string_view name) const noexcept;
    // Empty for a blob that is not there.
    std::span<const uint8_t> bytes(std::string_view name) const noexcept;
    std::string_view text(std::string_view name) const noexcept;
    // The blob as an array of T; empty when absent or not a whole number of T.
    template <typename T>
    std::span<const T> as(std::string_view name) const noexcept {
        static_assert(alignof(T) <= 16, "blobs are 16-byte aligned");
        const std::span<const uint8_t> b = bytes(name);
        if (b.size() % sizeof(T) != 0) return {};
        return std::span<const T>(reinterpret_cast<const T*>(b.data()), b.size() / sizeof(T));
    }
    // What the file takes, header included.
    uint64_t file_bytes() const noexcept { return file_bytes_; }

private:
    friend struct ArtifactCodec;
    struct Entry {
        std::string_view name;
        size_t offset = 0;
        size_t size = 0;
    };
    // 16-byte units, so the buffer itself is 16-byte aligned.
    struct alignas(16) Unit { unsigned char b[16]; };
    std::vector<Unit> buffer_;
    std::vector<Entry> blobs_;
    uint64_t file_bytes_ = 0;
};

// Why a lookup missed.
enum class Miss : uint8_t {
    None,          // it hit
    InvalidKey,    // CacheKey::valid is false
    Absent,        // no file
    Unreadable,    // the file is there but could not be read
    Foreign,       // not an artifact file, or another format version
    KeyMismatch,   // an artifact of another key (an id collision)
    Truncated,     // shorter or longer than its header says
    Corrupt,       // the checksum or the blob table does not hold
};
const char* to_string(Miss miss) noexcept;

// One artifact file at a path of the caller's choosing (a record kept beside
// a module, not in a cache directory): the same format and checks as a cache
// entry, written to a temporary file beside `path` and renamed over it.
bool write_artifact_file(const std::filesystem::path& path, const CacheKey& key,
                         std::span<const BlobRef> blobs, std::string* error = nullptr);
std::optional<Artifact> read_artifact_file(const std::filesystem::path& path, const CacheKey& key,
                                           Miss* why = nullptr);

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

class ArtifactCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t stores = 0;
        uint64_t store_failures = 0;
        uint64_t evictions = 0;   // entries this instance's trims deleted
        uint64_t trims = 0;       // directory walks
    };

    // `dir` is created on the first store. `max_bytes` bounds what the
    // directory's entries take on disk (counted in whole 4 KiB units); 0
    // leaves it unbounded.
    ArtifactCache(std::filesystem::path dir, uint64_t max_bytes);
    // Waits for a trim in progress.
    ~ArtifactCache();
    ArtifactCache(const ArtifactCache&) = delete;
    ArtifactCache& operator=(const ArtifactCache&) = delete;

    const std::filesystem::path& dir() const noexcept { return dir_; }
    uint64_t max_bytes() const noexcept { return max_bytes_; }
    std::filesystem::path path_for(const CacheKey& key) const;

    // The artifact stored under `key`, or nothing with `why` set. A hit
    // marks the entry used (its modification time), which is what the
    // eviction orders by.
    std::optional<Artifact> get(const CacheKey& key, Miss* why = nullptr);

    // Stores `blobs` under `key`, replacing what is there. False (with
    // `error`) when it could not; a cache that cannot be written is a cache
    // that misses, so callers usually ignore it. When the directory may be
    // over its budget the store asks this instance's trim thread to walk
    // it: the storing thread never walks the directory.
    bool put(const CacheKey& key, std::span<const BlobRef> blobs, std::string* error = nullptr);
    bool put(const CacheKey& key, std::initializer_list<BlobRef> blobs, std::string* error = nullptr) {
        return put(key, std::span<const BlobRef>(blobs.begin(), blobs.size()), error);
    }

    // Deletes the entry; false when there was none (or it could not be).
    bool remove(const CacheKey& key);

    // Walks the directory now, on this thread: deletes writes abandoned more
    // than ten minutes ago, and when the entries take more than the budget,
    // the least recently used down to 7/8 of it. Never deletes the entry
    // this instance stored last, nor one it stores while the walk runs.
    // Returns what the entries (and writes in progress) take afterwards.
    uint64_t trim();

    // Blocks until the trim the stores asked for has finished.
    void flush();

    Stats stats() const;

private:
    void request_trim(const std::string& stored_hex, uint64_t stored_bytes);
    void trim_thread();
    uint64_t walk();
    bool is_protected(const std::string& hex) const;

    std::filesystem::path dir_;
    uint64_t max_bytes_;

    std::mutex walk_mutex_;               // one walk at a time per instance
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool thread_started_ = false;
    bool stopping_ = false;
    bool pending_ = false;
    bool busy_ = false;
    bool walked_ = false;
    uint64_t estimate_ = 0;               // last walk's total + stores since
    // What no walk of this instance deletes: the newest key it stored, the
    // newest when the walk in progress began, and those stored during it.
    std::string last_stored_;
    std::string walk_kept_;
    std::set<std::string> stored_during_walk_;
    bool walking_ = false;
    Stats stats_;
};

} // namespace brass::cache
