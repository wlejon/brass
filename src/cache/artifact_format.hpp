#pragma once
// The artifact file format and the file operations under it, shared by the
// cache's parts (artifact_cache.hpp, docs/artifact_cache.md "On disk").
//
//   offset  size  field
//   0       4     magic "BRAC"
//   4       4     format version (kFormatVersion)
//   8       16    key id      (lo, hi)
//   24      16    key check   (lo, hi)
//   40      8     payload bytes (everything after the header)
//   48      16    checksum (lo, hi): Murmur3-128 of header bytes 0..48 and
//                 64..80, then the payload
//   64      4     blob count
//   68      12    zero
//   80            blobs, each: name length (u32), zero (u32), data length
//                 (u64), the name, zeros to a 16-byte offset, the data, zeros
//                 to a 16-byte offset
//
// All integers little-endian. Offsets are from the start of the file, so a
// buffer holding the file 16-byte aligned holds every blob 16-byte aligned.

#include <brass/cache/artifact_cache.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace brass::cache {

constexpr uint32_t kMagic = 0x43415242u;   // "BRAC"
constexpr uint32_t kFormatVersion = 1;
constexpr size_t kHeaderBytes = 80;
constexpr size_t kMaxNameBytes = 255;
constexpr const char* kEntryExt = ".bac";
constexpr const char* kTmpMarker = ".tmp-";

// What a file of `size` bytes takes on disk: whole 4 KiB units. Entries are
// often a few kilobytes, so counting bytes rather than units would
// under-count a full directory by a good fraction.
constexpr uint64_t kAllocUnit = 4096;
inline uint64_t disk_bytes(uint64_t size) { return (size + kAllocUnit - 1) / kAllocUnit * kAllocUnit; }

struct ArtifactCodec {
    // The whole file for `blobs` under `key`, or empty with `error` set.
    static std::vector<uint8_t> encode(const CacheKey& key, std::span<const BlobRef> blobs, std::string* error);
    // Reads and checks the file at `path`.
    static std::optional<Artifact> read(const std::filesystem::path& path, const CacheKey& key, Miss* why);
};

namespace detail {

uint64_t process_id();
// Unique within the process, across every cache instance.
uint64_t next_serial();

// `<final name><kTmpMarker><pid>-<serial>`, beside `final_path`.
std::filesystem::path temp_path_for(const std::filesystem::path& final_path);

// Creates `path` (which must not exist) with `bytes`; the handle is never
// inherited by a child process (a child spawned meanwhile cannot hold the
// file open for writing, which would make a load of it fail: ETXTBSY, or a
// sharing violation). `executable`: mode 0755 rather than 0644 on POSIX.
// False with `error` set on failure, and then no file is left behind.
bool write_new_file(const std::filesystem::path& path, std::span<const uint8_t> bytes, std::string* error,
                    bool executable = false);

// write_new_file to a temporary name beside `path`, then renamed over it, so
// a reader sees the old file or the new one, never part of one.
bool write_file_atomically(const std::filesystem::path& path, std::span<const uint8_t> bytes, std::string* error);

// Reads the whole file; false when it cannot be opened or read.
bool read_whole_file(const std::filesystem::path& path, std::vector<uint8_t>& out);

} // namespace detail

} // namespace brass::cache
