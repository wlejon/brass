#pragma once

#include <cstdint>

namespace brass::pgo {

// Magic header: 'B' 'P' 'R' 'F' (0x46525042 in little-endian)
inline constexpr uint32_t kProfileMagic = 0x46525042;
inline constexpr uint32_t kProfileVersion = 1;

#pragma pack(push, 1)
struct ProfileFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t module_hash;
    uint32_t module_name_length;
    uint32_t function_count;
};
#pragma pack(pop)

} // namespace brass::pgo
