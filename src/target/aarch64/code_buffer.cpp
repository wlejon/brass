#include <brass/target/aarch64/code_buffer.hpp>
#include <algorithm>
#include <cstring>
#include <ostream>
#include <stdexcept>

namespace brass::aarch64 {

std::string_view to_string(RelocationKind kind) noexcept {
    switch (kind) {
    case RelocationKind::Call26:    return "Call26";
    case RelocationKind::Jump26:    return "Jump26";
    case RelocationKind::Page21:    return "Page21";
    case RelocationKind::AddLo12:   return "AddLo12";
    case RelocationKind::LdSt8Lo12:   return "LdSt8Lo12";
    case RelocationKind::LdSt16Lo12:  return "LdSt16Lo12";
    case RelocationKind::LdSt32Lo12:  return "LdSt32Lo12";
    case RelocationKind::LdSt64Lo12:  return "LdSt64Lo12";
    case RelocationKind::LdSt128Lo12: return "LdSt128Lo12";
    case RelocationKind::GotPage21: return "GotPage21";
    case RelocationKind::GotLo12:   return "GotLo12";
    case RelocationKind::Abs64:     return "Abs64";
    default: return "unknown";
    }
}

std::string_view to_string(FixupKind kind) noexcept {
    switch (kind) {
    case FixupKind::Branch26:     return "Branch26";
    case FixupKind::CondBranch19: return "CondBranch19";
    case FixupKind::TestBranch14: return "TestBranch14";
    case FixupKind::Adr21:        return "Adr21";
    default: return "unknown";
    }
}

std::ostream& operator<<(std::ostream& os, RelocationKind kind) {
    return os << to_string(kind);
}

std::ostream& operator<<(std::ostream& os, FixupKind kind) {
    return os << to_string(kind);
}

void CodeBuffer::emit_bytes(const uint8_t* data, size_t count) {
    if (data && count > 0) {
        bytes_.insert(bytes_.end(), data, data + count);
    }
}

void CodeBuffer::emit_bytes(std::span<const uint8_t> data) {
    emit_bytes(data.data(), data.size());
}

void CodeBuffer::patch8(size_t offset, uint8_t val) {
    if (offset >= bytes_.size()) {
        throw std::out_of_range("CodeBuffer::patch8: offset out of range");
    }
    bytes_[offset] = val;
}

void CodeBuffer::patch32(size_t offset, uint32_t val) {
    if (offset + 4 > bytes_.size()) {
        throw std::out_of_range("CodeBuffer::patch32: offset out of range");
    }
    bytes_[offset + 0] = static_cast<uint8_t>(val & 0xFF);
    bytes_[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    bytes_[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    bytes_[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

void CodeBuffer::patch64(size_t offset, uint64_t val) {
    if (offset + 8 > bytes_.size()) {
        throw std::out_of_range("CodeBuffer::patch64: offset out of range");
    }
    for (int i = 0; i < 8; ++i) {
        bytes_[offset + static_cast<size_t>(i)] = static_cast<uint8_t>((val >> (i * 8)) & 0xFF);
    }
}

Label CodeBuffer::create_label() {
    uint32_t id = next_label_id_++;
    if (label_positions_.size() <= id) {
        label_positions_.resize(id + 1, static_cast<size_t>(-1));
    }
    return Label{ id };
}

void CodeBuffer::bind(Label label) {
    if (!label.is_valid() || label.id >= label_positions_.size()) {
        throw std::invalid_argument("CodeBuffer::bind: invalid label");
    }
    if (label_positions_[label.id] != static_cast<size_t>(-1)) {
        throw std::runtime_error("CodeBuffer::bind: label is already bound");
    }

    size_t target_offset = bytes_.size();
    label_positions_[label.id] = target_offset;

    // Resolve any pending forward fixups for this label
    for (auto it = pending_fixups_.begin(); it != pending_fixups_.end(); ) {
        if (it->label_id == label.id) {
            int64_t disp = static_cast<int64_t>(target_offset) - static_cast<int64_t>(it->patch_offset);

            uint32_t curr_inst = static_cast<uint32_t>(bytes_[it->patch_offset + 0]) |
                                (static_cast<uint32_t>(bytes_[it->patch_offset + 1]) << 8) |
                                (static_cast<uint32_t>(bytes_[it->patch_offset + 2]) << 16) |
                                (static_cast<uint32_t>(bytes_[it->patch_offset + 3]) << 24);

            if (it->kind == FixupKind::Branch26) {
                // 26-bit word displacement (+/- 128MB)
                int64_t word_disp = disp >> 2;
                if (word_disp < -(1 << 25) || word_disp > ((1 << 25) - 1)) {
                    throw std::runtime_error("CodeBuffer::bind: Branch26 displacement out of range");
                }
                uint32_t imm26 = static_cast<uint32_t>(word_disp) & 0x03FFFFFFu;
                curr_inst = (curr_inst & ~0x03FFFFFFu) | imm26;
                patch32(it->patch_offset, curr_inst);
            } else if (it->kind == FixupKind::CondBranch19) {
                // 19-bit word displacement (+/- 1MB)
                int64_t word_disp = disp >> 2;
                if (word_disp < -(1 << 18) || word_disp > ((1 << 18) - 1)) {
                    if (it->branch_site == 0) {
                        throw std::runtime_error("CodeBuffer::bind: CondBranch19 displacement out of range");
                    }
                    relax_requests_.push_back(it->branch_site);
                    it = pending_fixups_.erase(it);
                    continue;
                }
                uint32_t imm19 = static_cast<uint32_t>(word_disp) & 0x0007FFFFu;
                curr_inst = (curr_inst & ~(0x0007FFFFu << 5)) | (imm19 << 5);
                patch32(it->patch_offset, curr_inst);
            } else if (it->kind == FixupKind::TestBranch14) {
                // 14-bit word displacement (+/- 32KB)
                int64_t word_disp = disp >> 2;
                if (word_disp < -(1 << 13) || word_disp > ((1 << 13) - 1)) {
                    if (it->branch_site == 0) {
                        throw std::runtime_error("CodeBuffer::bind: TestBranch14 displacement out of range");
                    }
                    relax_requests_.push_back(it->branch_site);
                    it = pending_fixups_.erase(it);
                    continue;
                }
                uint32_t imm14 = static_cast<uint32_t>(word_disp) & 0x00003FFFu;
                curr_inst = (curr_inst & ~(0x00003FFFu << 5)) | (imm14 << 5);
                patch32(it->patch_offset, curr_inst);
            } else if (it->kind == FixupKind::Adr21) {
                // 21-bit byte displacement (+/- 1MB)
                if (disp < -(1 << 20) || disp > ((1 << 20) - 1)) {
                    throw std::runtime_error("CodeBuffer::bind: Adr21 displacement out of range");
                }
                uint32_t imm21 = static_cast<uint32_t>(disp) & 0x001FFFFFu;
                uint32_t immlo = imm21 & 0x3u;
                uint32_t immhi = (imm21 >> 2) & 0x0007FFFFu;
                curr_inst = (curr_inst & ~(0x3u << 29)) | (immlo << 29);
                curr_inst = (curr_inst & ~(0x0007FFFFu << 5)) | (immhi << 5);
                patch32(it->patch_offset, curr_inst);
            }

            it = pending_fixups_.erase(it);
        } else {
            ++it;
        }
    }
}

bool CodeBuffer::is_bound(Label label) const {
    if (!label.is_valid() || label.id >= label_positions_.size()) {
        return false;
    }
    return label_positions_[label.id] != static_cast<size_t>(-1);
}

size_t CodeBuffer::label_offset(Label label) const {
    if (!label.is_valid() || label.id >= label_positions_.size()) {
        throw std::invalid_argument("CodeBuffer::label_offset: invalid label");
    }
    size_t off = label_positions_[label.id];
    if (off == static_cast<size_t>(-1)) {
        throw std::runtime_error("CodeBuffer::label_offset: label not bound yet");
    }
    return off;
}

void CodeBuffer::record_label_fixup(Label label, size_t patch_offset, FixupKind kind, uint32_t branch_site) {
    if (!label.is_valid()) {
        throw std::invalid_argument("CodeBuffer::record_label_fixup: invalid label");
    }
    if (label.id >= label_positions_.size()) {
        label_positions_.resize(label.id + 1, static_cast<size_t>(-1));
    }
    pending_fixups_.push_back(LabelFixup{ patch_offset, kind, label.id, branch_site });
}

bool CodeBuffer::is_long_branch_site(uint32_t site) const noexcept {
    return std::binary_search(long_branch_sites_.begin(), long_branch_sites_.end(), site);
}

void CodeBuffer::add_long_branch_sites(const std::vector<uint32_t>& sites) {
    long_branch_sites_.insert(long_branch_sites_.end(), sites.begin(), sites.end());
    std::sort(long_branch_sites_.begin(), long_branch_sites_.end());
    long_branch_sites_.erase(std::unique(long_branch_sites_.begin(), long_branch_sites_.end()),
                             long_branch_sites_.end());
}

void CodeBuffer::add_relocation(size_t offset, RelocationKind kind, std::string symbol_name, int64_t addend) {
    relocations_.push_back(Relocation{ offset, kind, std::move(symbol_name), addend });
}

void CodeBuffer::emit_nops(size_t count) {
    // AArch64 NOP is 4 bytes: 0xD503201F
    size_t num_words = count / 4;
    for (size_t i = 0; i < num_words; ++i) {
        emit32(0xD503201Fu);
    }
    size_t rem = count % 4;
    for (size_t i = 0; i < rem; ++i) {
        emit8(0);
    }
}

void CodeBuffer::align(size_t alignment, size_t max_bytes_to_pad) {
    if (alignment <= 1) return;
    size_t current = bytes_.size();
    size_t rem = current % alignment;
    if (rem == 0) return;
    size_t pad = alignment - rem;
    if (max_bytes_to_pad > 0 && pad > max_bytes_to_pad) return;
    emit_nops(pad);
}

void CodeBuffer::clear() {
    bytes_.clear();
    relocations_.clear();
    label_positions_.clear();
    pending_fixups_.clear();
    next_label_id_ = 1;
    short_branch_sites_ = 0;
    long_branch_sites_.clear();
    relax_requests_.clear();
}

} // namespace brass::aarch64
