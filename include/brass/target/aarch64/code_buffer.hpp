#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <span>
#include <string_view>
#include <iosfwd>

namespace brass::aarch64 {

struct Label {
    uint32_t id = 0;
    constexpr bool is_valid() const noexcept { return id != 0; }
    constexpr bool operator==(const Label& other) const noexcept = default;
    constexpr bool operator!=(const Label& other) const noexcept = default;
};

enum class RelocationKind : uint8_t {
    Call26,      // 26-bit PC-relative call (BL)
    Jump26,      // 26-bit PC-relative jump (B)
    Page21,      // ADRP: page of the symbol
    AddLo12,     // ADD (immediate): low 12 bits of the address, unscaled
    LdSt8Lo12,   // LDR/STR (unsigned immediate): low 12 bits, scaled by the access size
    LdSt16Lo12,
    LdSt32Lo12,
    LdSt64Lo12,
    LdSt128Lo12,
    GotPage21,   // ADRP: page of the symbol's GOT slot
    GotLo12,     // LDR Xt, [Xn, #imm]: the GOT slot's low 12 bits (scaled by 8)
    Abs64,       // 64-bit absolute address
};

struct Relocation {
    size_t offset = 0;
    RelocationKind kind = RelocationKind::Call26;
    std::string symbol_name;
    int64_t addend = 0;

    bool operator==(const Relocation& other) const noexcept = default;
};

enum class FixupKind : uint8_t {
    Branch26,     // 26-bit PC-relative word offset for B, BL
    CondBranch19, // 19-bit PC-relative word offset for B.cond, CBZ, CBNZ
    TestBranch14, // 14-bit PC-relative word offset for TBZ, TBNZ
    Adr21,        // 21-bit PC-relative byte offset for ADR
};

struct LabelFixup {
    size_t patch_offset = 0;
    FixupKind kind = FixupKind::Branch26;
    uint32_t label_id = 0;
    // Non-zero for a relaxable short branch (CondBranch19 / TestBranch14
    // emitted through AArch64Encoder): the site's ordinal in the emission.
    uint32_t branch_site = 0;
};

class CodeBuffer {
public:
    CodeBuffer() = default;

    // Byte & Word emission
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

    // AArch64 standard 32-bit instruction emission
    void emit_inst(uint32_t inst) {
        emit32(inst);
    }

    void emit_bytes(const uint8_t* data, size_t count);
    void emit_bytes(std::span<const uint8_t> data);

    // Patching
    void patch8(size_t offset, uint8_t val);
    void patch32(size_t offset, uint32_t val);
    void patch64(size_t offset, uint64_t val);

    // Label management
    Label create_label();
    void bind(Label label);
    bool is_bound(Label label) const;
    size_t label_offset(Label label) const;
    void record_label_fixup(Label label, size_t patch_offset, FixupKind kind, uint32_t branch_site = 0);

    // Short-branch relaxation. B.cond / CBZ / CBNZ reach +-1 MB and TBZ /
    // TBNZ +-32 KB; a forward one whose label turns out to be further away
    // cannot grow in place. Each such branch the encoder emits is a numbered
    // site; bind() records the sites that did not fit (relax_requests) and
    // leaves them unpatched, and the emitter then emits the whole function
    // again with those sites in the long form (inverted short branch over an
    // unconditional B). Emission is deterministic, so a site number names the
    // same branch in every pass. A code buffer with outstanding requests
    // holds wrong code: whoever emits into it must re-emit or refuse.
    uint32_t next_short_branch_site() noexcept { return ++short_branch_sites_; }
    bool is_long_branch_site(uint32_t site) const noexcept;
    void add_long_branch_sites(const std::vector<uint32_t>& sites);
    const std::vector<uint32_t>& relax_requests() const noexcept { return relax_requests_; }

    // Relocations
    void add_relocation(size_t offset, RelocationKind kind, std::string symbol_name, int64_t addend = 0);
    const std::vector<Relocation>& relocations() const noexcept { return relocations_; }
    std::vector<Relocation>& relocations() noexcept { return relocations_; }

    // Alignment and NOP padding (AArch64 NOP = 0xD503201F)
    void emit_nops(size_t count);
    void align(size_t alignment, size_t max_bytes_to_pad = 0);

    // Buffer inspection
    size_t size() const noexcept { return bytes_.size(); }
    bool empty() const noexcept { return bytes_.empty(); }
    const uint8_t* data() const noexcept { return bytes_.data(); }
    uint8_t* data() noexcept { return bytes_.data(); }
    std::span<const uint8_t> span() const noexcept { return { bytes_.data(), bytes_.size() }; }
    const std::vector<uint8_t>& bytes() const noexcept { return bytes_; }

    void clear();

private:
    std::vector<uint8_t> bytes_;
    std::vector<Relocation> relocations_;
    std::vector<size_t> label_positions_;
    std::vector<LabelFixup> pending_fixups_;
    uint32_t next_label_id_ = 1;
    uint32_t short_branch_sites_ = 0;
    std::vector<uint32_t> long_branch_sites_;   // sorted
    std::vector<uint32_t> relax_requests_;
};

std::string_view to_string(RelocationKind kind) noexcept;
std::string_view to_string(FixupKind kind) noexcept;

std::ostream& operator<<(std::ostream& os, RelocationKind kind);
std::ostream& operator<<(std::ostream& os, FixupKind kind);

} // namespace brass::aarch64
