#include <brass/target/x64/code_buffer.hpp>
#include <stdexcept>
#include <algorithm>

namespace brass::x64 {

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
            if (it->kind == FixupKind::Rel8) {
                size_t next_off = it->patch_offset + 1;
                int64_t disp = static_cast<int64_t>(target_offset) - static_cast<int64_t>(next_off);
                if (disp < -128 || disp > 127) {
                    throw std::runtime_error("CodeBuffer::bind: Rel8 branch displacement out of range [-128, 127]");
                }
                patch8(it->patch_offset, static_cast<uint8_t>(static_cast<int8_t>(disp)));
            } else if (it->kind == FixupKind::Rel32) {
                size_t next_off = it->patch_offset + 4;
                int64_t disp = static_cast<int64_t>(target_offset) - static_cast<int64_t>(next_off);
                if (disp < INT32_MIN || disp > INT32_MAX) {
                    throw std::runtime_error("CodeBuffer::bind: Rel32 branch displacement out of 32-bit range");
                }
                patch32(it->patch_offset, static_cast<uint32_t>(static_cast<int32_t>(disp)));
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
    if (!is_bound(label)) {
        throw std::runtime_error("CodeBuffer::label_offset: label is not bound");
    }
    return label_positions_[label.id];
}

void CodeBuffer::record_label_fixup(Label label, size_t patch_offset, FixupKind kind) {
    if (!label.is_valid()) {
        throw std::invalid_argument("CodeBuffer::record_label_fixup: invalid label");
    }

    if (is_bound(label)) {
        size_t target_offset = label_positions_[label.id];
        if (kind == FixupKind::Rel8) {
            size_t next_off = patch_offset + 1;
            int64_t disp = static_cast<int64_t>(target_offset) - static_cast<int64_t>(next_off);
            if (disp < -128 || disp > 127) {
                throw std::runtime_error("CodeBuffer::record_label_fixup: Rel8 branch displacement out of range [-128, 127]");
            }
            patch8(patch_offset, static_cast<uint8_t>(static_cast<int8_t>(disp)));
        } else if (kind == FixupKind::Rel32) {
            size_t next_off = patch_offset + 4;
            int64_t disp = static_cast<int64_t>(target_offset) - static_cast<int64_t>(next_off);
            if (disp < INT32_MIN || disp > INT32_MAX) {
                throw std::runtime_error("CodeBuffer::record_label_fixup: Rel32 branch displacement out of 32-bit range");
            }
            patch32(patch_offset, static_cast<uint32_t>(static_cast<int32_t>(disp)));
        }
    } else {
        pending_fixups_.push_back(LabelFixup{ patch_offset, kind, label.id });
    }
}

void CodeBuffer::add_relocation(size_t offset, RelocationKind kind, std::string symbol_name, int64_t addend) {
    relocations_.push_back(Relocation{ offset, kind, std::move(symbol_name), addend });
}

void CodeBuffer::emit_nops(size_t count) {
    static const uint8_t nop1[] = { 0x90 };
    static const uint8_t nop2[] = { 0x66, 0x90 };
    static const uint8_t nop3[] = { 0x0F, 0x1F, 0x00 };
    static const uint8_t nop4[] = { 0x0F, 0x1F, 0x40, 0x00 };
    static const uint8_t nop5[] = { 0x0F, 0x1F, 0x44, 0x00, 0x00 };
    static const uint8_t nop6[] = { 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00 };
    static const uint8_t nop7[] = { 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00 };
    static const uint8_t nop8[] = { 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };
    static const uint8_t nop9[] = { 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };

    static const uint8_t* const nop_tables[] = {
        nullptr, nop1, nop2, nop3, nop4, nop5, nop6, nop7, nop8, nop9
    };

    while (count >= 9) {
        emit_bytes(nop9, 9);
        count -= 9;
    }
    if (count > 0 && count <= 8) {
        emit_bytes(nop_tables[count], count);
    }
}

void CodeBuffer::align(size_t alignment, size_t max_bytes_to_pad) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("CodeBuffer::align: alignment must be a power of 2");
    }
    size_t current = bytes_.size();
    size_t remainder = current % alignment;
    if (remainder == 0) return;

    size_t pad = alignment - remainder;
    if (max_bytes_to_pad > 0 && pad > max_bytes_to_pad) {
        return;
    }
    emit_nops(pad);
}

void CodeBuffer::clear() {
    bytes_.clear();
    relocations_.clear();
    pending_fixups_.clear();
    label_positions_.clear();
    next_label_id_ = 1;
}

bool CodeBuffer::has_unresolved_labels() const {
    return !pending_fixups_.empty();
}

std::vector<uint32_t> CodeBuffer::unresolved_label_ids() const {
    std::vector<uint32_t> result;
    for (const auto& f : pending_fixups_) {
        if (std::find(result.begin(), result.end(), f.label_id) == result.end()) {
            result.push_back(f.label_id);
        }
    }
    return result;
}

std::string_view to_string(RelocationKind kind) noexcept {
    switch (kind) {
    case RelocationKind::PCRel32:  return "PCRel32";
    case RelocationKind::Abs64:    return "Abs64";
    case RelocationKind::SecRel32: return "SecRel32";
    default: return "unknown";
    }
}

} // namespace brass::x64
