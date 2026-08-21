#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <span>
#include <string_view>
#include <iosfwd>

namespace brass::x64 {

struct Label {
    uint32_t id = 0;
    constexpr bool is_valid() const noexcept { return id != 0; }
    constexpr bool operator==(const Label& other) const noexcept = default;
    constexpr bool operator!=(const Label& other) const noexcept = default;
};

enum class RelocationKind : uint8_t {
    PCRel32,  // 32-bit PC-relative displacement (e.g. call/jmp target)
    Abs64,    // 64-bit absolute address (e.g. mov reg, imm64)
    SecRel32, // 32-bit section-relative offset
};

struct Relocation {
    size_t offset = 0;
    RelocationKind kind = RelocationKind::PCRel32;
    std::string symbol_name;
    int64_t addend = 0;

    bool operator==(const Relocation& other) const noexcept = default;
};

enum class FixupKind : uint8_t {
    Rel8,   // 1-byte relative displacement
    Rel32,  // 4-byte relative displacement
};

struct LabelFixup {
    size_t patch_offset = 0;
    FixupKind kind = FixupKind::Rel32;
    uint32_t label_id = 0;
};

class CodeBuffer {
public:
    CodeBuffer() = default;

    // Byte emission
    void emit8(uint8_t val) {
        bytes_.push_back(val);
    }

    void emit16(uint16_t val) {
        bytes_.push_back(static_cast<uint8_t>(val & 0xFF));
        bytes_.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    }

    void emit32(uint32_t val) {
        bytes_.push_back(static_cast<uint8_t>(val & 0xFF));
        bytes_.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
        bytes_.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
        bytes_.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
    }

    void emit64(uint64_t val) {
        for (int i = 0; i < 8; ++i) {
            bytes_.push_back(static_cast<uint8_t>((val >> (i * 8)) & 0xFF));
        }
    }

    void emit_bytes(const uint8_t* data, size_t count);
    void emit_bytes(std::span<const uint8_t> data);

    // Byte patching
    void patch8(size_t offset, uint8_t val);
    void patch32(size_t offset, uint32_t val);
    void patch64(size_t offset, uint64_t val);

    // Label management
    Label create_label();
    void bind(Label label);
    bool is_bound(Label label) const;
    size_t label_offset(Label label) const;
    void record_label_fixup(Label label, size_t patch_offset, FixupKind kind);

    // Relocations
    void add_relocation(size_t offset, RelocationKind kind, std::string symbol_name, int64_t addend = 0);
    const std::vector<Relocation>& relocations() const noexcept { return relocations_; }
    std::vector<Relocation>& relocations() noexcept { return relocations_; }

    // Multi-byte NOPs & Alignment
    void emit_nops(size_t count);
    void align(size_t alignment, size_t max_bytes_to_pad = 0);

    // Buffer inspection
    size_t size() const noexcept { return bytes_.size(); }
    bool empty() const noexcept { return bytes_.empty(); }
    const uint8_t* data() const noexcept { return bytes_.data(); }
    uint8_t* data() noexcept { return bytes_.data(); }
    const std::vector<uint8_t>& bytes() const noexcept { return bytes_; }
    std::span<const uint8_t> span() const noexcept { return {bytes_.data(), bytes_.size()}; }
    void clear();

    bool has_unresolved_labels() const;
    std::vector<uint32_t> unresolved_label_ids() const;

private:
    std::vector<uint8_t> bytes_;
    std::vector<Relocation> relocations_;
    std::vector<LabelFixup> pending_fixups_;
    std::vector<size_t> label_positions_; // indexed by label id (1-based), size_t(-1) if unbound
    uint32_t next_label_id_ = 1;
};

std::string_view to_string(RelocationKind kind) noexcept;

std::ostream& operator<<(std::ostream& os, RelocationKind kind);

} // namespace brass::x64
