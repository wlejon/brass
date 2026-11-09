// Artifact files: encoding, checked reads, and the platform file operations
// the cache writes with (artifact_format.hpp).
#include "cache/artifact_format.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace brass::cache {

namespace {

constexpr uint64_t kSumSeed = 0x6d75736b63617262ULL;   // "brackusm"

void put_u32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (i * 8));
}
void put_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (i * 8));
}
uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

size_t pad16(size_t n) { return (n + 15) & ~static_cast<size_t>(15); }

Digest128 checksum(const uint8_t* header, const uint8_t* payload, size_t payload_size) {
    Hasher128 h(kSumSeed);
    h.update(header, 48);
    h.update(header + 64, kHeaderBytes - 64);
    h.update(payload, payload_size);
    return h.finish();
}

void set_error(std::string* error, std::string text) {
    if (error) *error = std::move(text);
}

// A file opened for reading that a concurrent rename over it or a delete of
// it does not disturb, and that does not stop either (Windows: shared for
// delete; POSIX: an open descriptor survives both).
class ReadHandle {
public:
    explicit ReadHandle(const std::filesystem::path& path) {
#if defined(_WIN32)
        h_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) {
            const DWORD e = GetLastError();
            absent_ = e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND;
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) absent_ = errno == ENOENT || errno == ENOTDIR;
#endif
    }
    ~ReadHandle() {
#if defined(_WIN32)
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    ReadHandle(const ReadHandle&) = delete;
    ReadHandle& operator=(const ReadHandle&) = delete;

    bool open() const {
#if defined(_WIN32)
        return h_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }
    bool absent() const { return absent_; }

    bool size(uint64_t& out) const {
#if defined(_WIN32)
        LARGE_INTEGER li;
        if (!GetFileSizeEx(h_, &li)) return false;
        out = static_cast<uint64_t>(li.QuadPart);
        return true;
#else
        struct stat st;
        if (::fstat(fd_, &st) != 0) return false;
        out = static_cast<uint64_t>(st.st_size);
        return true;
#endif
    }

    // Exactly `n` bytes, or false.
    bool read(void* dst, size_t n) const {
        auto* p = static_cast<uint8_t*>(dst);
        while (n > 0) {
#if defined(_WIN32)
            const DWORD chunk = n > 0x40000000u ? 0x40000000u : static_cast<DWORD>(n);
            DWORD got = 0;
            if (!ReadFile(h_, p, chunk, &got, nullptr) || got == 0) return false;
#else
            const ssize_t got = ::read(fd_, p, n);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return false;
#endif
            p += got;
            n -= static_cast<size_t>(got);
        }
        return true;
    }

private:
#if defined(_WIN32)
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool absent_ = false;
};

} // namespace

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

bool Artifact::has(std::string_view name) const noexcept {
    for (const Entry& e : blobs_) {
        if (e.name == name) return true;
    }
    return false;
}

std::span<const uint8_t> Artifact::bytes(std::string_view name) const noexcept {
    for (const Entry& e : blobs_) {
        if (e.name == name) {
            const auto* base = reinterpret_cast<const uint8_t*>(buffer_.data());
            return std::span<const uint8_t>(base + e.offset, e.size);
        }
    }
    return {};
}

std::string_view Artifact::text(std::string_view name) const noexcept {
    const std::span<const uint8_t> b = bytes(name);
    return std::string_view(reinterpret_cast<const char*>(b.data()), b.size());
}

const char* to_string(Miss miss) noexcept {
    switch (miss) {
        case Miss::None: return "hit";
        case Miss::InvalidKey: return "the key could not be made";
        case Miss::Absent: return "no entry";
        case Miss::Unreadable: return "the entry could not be read";
        case Miss::Foreign: return "not an artifact of this format";
        case Miss::KeyMismatch: return "an artifact of another key";
        case Miss::Truncated: return "the entry is truncated";
        case Miss::Corrupt: return "the entry is damaged";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Encoding and checked reads
// ---------------------------------------------------------------------------

std::vector<uint8_t> ArtifactCodec::encode(const CacheKey& key, std::span<const BlobRef> blobs, std::string* error) {
    size_t total = kHeaderBytes;
    for (size_t i = 0; i < blobs.size(); ++i) {
        const BlobRef& b = blobs[i];
        if (b.name.size() > kMaxNameBytes) {
            set_error(error, "blob name longer than 255 bytes");
            return {};
        }
        for (size_t j = 0; j < i; ++j) {
            if (blobs[j].name == b.name) {
                set_error(error, "blob '" + std::string(b.name) + "' given twice");
                return {};
            }
        }
        total += pad16(16 + b.name.size()) + pad16(b.bytes.size());
    }
    if (blobs.size() > 0xffffffffu) {
        set_error(error, "too many blobs");
        return {};
    }
    std::vector<uint8_t> out(total, 0);
    uint8_t* h = out.data();
    put_u32(h + 0, kMagic);
    put_u32(h + 4, kFormatVersion);
    put_u64(h + 8, key.id.lo);
    put_u64(h + 16, key.id.hi);
    put_u64(h + 24, key.check.lo);
    put_u64(h + 32, key.check.hi);
    put_u64(h + 40, total - kHeaderBytes);
    put_u32(h + 64, static_cast<uint32_t>(blobs.size()));
    size_t at = kHeaderBytes;
    for (const BlobRef& b : blobs) {
        put_u32(out.data() + at, static_cast<uint32_t>(b.name.size()));
        put_u64(out.data() + at + 8, b.bytes.size());
        if (!b.name.empty()) std::memcpy(out.data() + at + 16, b.name.data(), b.name.size());
        at += pad16(16 + b.name.size());
        if (!b.bytes.empty()) std::memcpy(out.data() + at, b.bytes.data(), b.bytes.size());
        at += pad16(b.bytes.size());
    }
    const Digest128 sum = checksum(h, out.data() + kHeaderBytes, total - kHeaderBytes);
    put_u64(h + 48, sum.lo);
    put_u64(h + 56, sum.hi);
    return out;
}

std::optional<Artifact> ArtifactCodec::read(const std::filesystem::path& path, const CacheKey& key, Miss* why) {
    auto miss = [&](Miss m) -> std::optional<Artifact> {
        if (why) *why = m;
        return std::nullopt;
    };
    if (!key.valid) return miss(Miss::InvalidKey);
    ReadHandle file(path);
    if (!file.open()) return miss(file.absent() ? Miss::Absent : Miss::Unreadable);
    uint64_t size = 0;
    if (!file.size(size)) return miss(Miss::Unreadable);

    uint8_t header[kHeaderBytes];
    if (size < kHeaderBytes) {
        if (size >= 4 && file.read(header, 4) && get_u32(header) != kMagic) return miss(Miss::Foreign);
        return miss(Miss::Truncated);
    }
    if (!file.read(header, kHeaderBytes)) return miss(Miss::Unreadable);
    if (get_u32(header) != kMagic || get_u32(header + 4) != kFormatVersion) return miss(Miss::Foreign);
    if (get_u64(header + 40) != size - kHeaderBytes) return miss(Miss::Truncated);
    const Digest128 id{get_u64(header + 8), get_u64(header + 16)};
    const Digest128 check{get_u64(header + 24), get_u64(header + 32)};
    if (id != key.id || check != key.check) return miss(Miss::KeyMismatch);
    if (size > static_cast<uint64_t>(SIZE_MAX) - 16) return miss(Miss::Unreadable);

    Artifact art;
    art.file_bytes_ = size;
    try {
        art.buffer_.resize(static_cast<size_t>((size + 15) / 16));
    } catch (const std::bad_alloc&) {
        return miss(Miss::Unreadable);
    }
    auto* base = reinterpret_cast<uint8_t*>(art.buffer_.data());
    std::memcpy(base, header, kHeaderBytes);
    const size_t payload_size = static_cast<size_t>(size - kHeaderBytes);
    if (payload_size > 0 && !file.read(base + kHeaderBytes, payload_size)) return miss(Miss::Truncated);

    const Digest128 stored{get_u64(header + 48), get_u64(header + 56)};
    if (checksum(base, base + kHeaderBytes, payload_size) != stored) return miss(Miss::Corrupt);

    // The blob table: well-formed, and covering the payload exactly.
    const uint32_t count = get_u32(header + 64);
    if (get_u32(header + 68) != 0 || get_u64(header + 72) != 0) return miss(Miss::Corrupt);
    if (count > payload_size / 16) return miss(Miss::Corrupt);
    art.blobs_.reserve(count);
    size_t at = kHeaderBytes;
    const size_t end = static_cast<size_t>(size);
    for (uint32_t i = 0; i < count; ++i) {
        if (end - at < 16) return miss(Miss::Corrupt);
        const uint32_t name_len = get_u32(base + at);
        const uint64_t data_len = get_u64(base + at + 8);
        if (name_len > kMaxNameBytes || get_u32(base + at + 4) != 0) return miss(Miss::Corrupt);
        const size_t name_at = at + 16;
        at += pad16(16 + name_len);
        if (at > end || data_len > end - at) return miss(Miss::Corrupt);
        Artifact::Entry e;
        e.name = std::string_view(reinterpret_cast<const char*>(base + name_at), name_len);
        e.offset = at;
        e.size = static_cast<size_t>(data_len);
        at += pad16(e.size);
        if (at > end) return miss(Miss::Corrupt);
        art.blobs_.push_back(e);
    }
    if (at != end) return miss(Miss::Corrupt);
    if (why) *why = Miss::None;
    return art;
}

bool write_artifact_file(const std::filesystem::path& path, const CacheKey& key, std::span<const BlobRef> blobs,
                         std::string* error) {
    if (!key.valid) {
        set_error(error, "the key is not valid");
        return false;
    }
    const std::vector<uint8_t> bytes = ArtifactCodec::encode(key, blobs, error);
    if (bytes.empty()) return false;
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    return detail::write_file_atomically(path, bytes, error);
}

std::optional<Artifact> read_artifact_file(const std::filesystem::path& path, const CacheKey& key, Miss* why) {
    return ArtifactCodec::read(path, key, why);
}

// ---------------------------------------------------------------------------
// Platform file operations
// ---------------------------------------------------------------------------

namespace detail {

uint64_t process_id() {
#if defined(_WIN32)
    return static_cast<uint64_t>(_getpid());
#else
    return static_cast<uint64_t>(::getpid());
#endif
}

uint64_t next_serial() {
    static std::atomic<uint64_t> serial{0};
    return serial.fetch_add(1, std::memory_order_relaxed);
}

std::filesystem::path temp_path_for(const std::filesystem::path& final_path) {
    std::filesystem::path tmp = final_path;
    tmp += std::string(kTmpMarker) + std::to_string(process_id()) + "-" + std::to_string(next_serial());
    return tmp;
}

bool write_new_file(const std::filesystem::path& path, std::span<const uint8_t> bytes, std::string* error,
                    bool executable) {
#if defined(_WIN32)
    (void)executable;
    // No SECURITY_ATTRIBUTES: not inheritable. CREATE_NEW: never someone
    // else's file.
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        set_error(error, "cannot create " + path.string() + ": error " + std::to_string(GetLastError()));
        return false;
    }
    size_t done = 0;
    bool ok = true;
    while (done < bytes.size()) {
        const size_t left = bytes.size() - done;
        const DWORD chunk = left > 0x40000000u ? 0x40000000u : static_cast<DWORD>(left);
        DWORD written = 0;
        if (!WriteFile(h, bytes.data() + done, chunk, &written, nullptr) || written == 0) {
            set_error(error, "cannot write " + path.string() + ": error " + std::to_string(GetLastError()));
            ok = false;
            break;
        }
        done += written;
    }
    CloseHandle(h);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, executable ? 0755 : 0644);
    if (fd < 0) {
        set_error(error, "cannot create " + path.string() + ": " + std::strerror(errno));
        return false;
    }
    size_t done = 0;
    bool ok = true;
    while (done < bytes.size()) {
        const ssize_t put = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) {
            set_error(error, "cannot write " + path.string() + ": " + std::strerror(errno));
            ok = false;
            break;
        }
        done += static_cast<size_t>(put);
    }
    if (::close(fd) != 0 && ok) {
        set_error(error, "cannot write " + path.string() + ": " + std::strerror(errno));
        ok = false;
    }
#endif
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    return ok;
}

bool write_file_atomically(const std::filesystem::path& path, std::span<const uint8_t> bytes, std::string* error) {
    const std::filesystem::path tmp = temp_path_for(path);
    if (!write_new_file(tmp, bytes, error)) return false;
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        set_error(error, "cannot rename into " + path.string() + ": " + ec.message());
        std::error_code rm;
        std::filesystem::remove(tmp, rm);
        return false;
    }
    return true;
}

bool read_whole_file(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    ReadHandle file(path);
    uint64_t size = 0;
    if (!file.open() || !file.size(size) || size > static_cast<uint64_t>(SIZE_MAX)) return false;
    out.resize(static_cast<size_t>(size));
    return size == 0 || file.read(out.data(), out.size());
}

} // namespace detail

} // namespace brass::cache
