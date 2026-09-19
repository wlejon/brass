#pragma once

// The dyld-info payloads of a Mach-O dylib: the export trie, the rebase
// opcode stream (absolute pointers dyld slides) and the bind opcode stream
// (pointers dyld fills from other images). Internal to src/target.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace brass::target::macho_dyld {

// (name with its Mach-O underscore, address relative to the image base).
std::vector<uint8_t> build_export_trie(const std::vector<std::pair<std::string, uint64_t>>& exports);

struct RebaseEntry {
    uint8_t segment = 0;    // LC_SEGMENT_64 index
    uint64_t offset = 0;    // within that segment
};

struct BindEntry {
    uint8_t segment = 0;
    uint64_t offset = 0;
    uint32_t ordinal = 0;   // 1-based LC_LOAD_DYLIB index
    std::string symbol;     // with its Mach-O underscore
    int64_t addend = 0;
};

std::vector<uint8_t> build_rebase_opcodes(std::vector<RebaseEntry> entries);
std::vector<uint8_t> build_bind_opcodes(std::vector<BindEntry> entries);

} // namespace brass::target::macho_dyld
