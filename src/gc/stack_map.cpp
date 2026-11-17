#include <brass/gc/stack_map.hpp>
#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>
#include <ostream>

namespace brass {

namespace {

void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void write_i32(std::vector<uint8_t>& buf, int32_t v) {
    write_u32(buf, static_cast<uint32_t>(v));
}

void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

void write_var(std::vector<uint8_t>& buf, uint64_t v) {
    while (v >= 0x80) {
        buf.push_back(static_cast<uint8_t>(v | 0x80));
        v >>= 7;
    }
    buf.push_back(static_cast<uint8_t>(v));
}

void write_svar(std::vector<uint8_t>& buf, int64_t v) {
    write_var(buf, (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63));
}

void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t count) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

class BinaryReader {
public:
    explicit BinaryReader(std::span<const uint8_t> data)
        : data_(data), pos_(0) {}

    bool has_bytes(size_t n) const noexcept {
        return pos_ + n <= data_.size();
    }

    bool read_u8(uint8_t& val) noexcept {
        if (!has_bytes(1)) return false;
        val = data_[pos_++];
        return true;
    }

    bool read_u16(uint16_t& val) noexcept {
        if (!has_bytes(2)) return false;
        val = static_cast<uint16_t>(data_[pos_]) |
              (static_cast<uint16_t>(data_[pos_ + 1]) << 8);
        pos_ += 2;
        return true;
    }

    bool read_u32(uint32_t& val) noexcept {
        if (!has_bytes(4)) return false;
        val = static_cast<uint32_t>(data_[pos_]) |
              (static_cast<uint32_t>(data_[pos_ + 1]) << 8) |
              (static_cast<uint32_t>(data_[pos_ + 2]) << 16) |
              (static_cast<uint32_t>(data_[pos_ + 3]) << 24);
        pos_ += 4;
        return true;
    }

    bool read_i32(int32_t& val) noexcept {
        uint32_t uval = 0;
        if (!read_u32(uval)) return false;
        val = static_cast<int32_t>(uval);
        return true;
    }

    bool read_u64(uint64_t& val) noexcept {
        if (!has_bytes(8)) return false;
        val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<uint64_t>(data_[pos_ + static_cast<size_t>(i)]) << (i * 8));
        }
        pos_ += 8;
        return true;
    }

    bool read_var(uint64_t& val) noexcept {
        val = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            uint8_t b = 0;
            if (!read_u8(b)) return false;
            val |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return true;
        }
        return false;
    }

    template <typename T>
    bool read_var_as(T& val) noexcept {
        uint64_t v = 0;
        if (!read_var(v) || v > static_cast<uint64_t>(std::numeric_limits<T>::max())) return false;
        val = static_cast<T>(v);
        return true;
    }

    bool read_svar(int64_t& val) noexcept {
        uint64_t v = 0;
        if (!read_var(v)) return false;
        val = static_cast<int64_t>(v >> 1) ^ -static_cast<int64_t>(v & 1);
        return true;
    }

    bool read_string(std::string& s, size_t len) {
        if (!has_bytes(len)) return false;
        s.assign(reinterpret_cast<const char*>(data_.data() + pos_), len);
        pos_ += len;
        return true;
    }

    bool skip(size_t len) noexcept {
        if (!has_bytes(len)) return false;
        pos_ += len;
        return true;
    }

    size_t remaining() const noexcept {
        return pos_ < data_.size() ? (data_.size() - pos_) : 0;
    }

private:
    std::span<const uint8_t> data_;
    size_t pos_;
};

} // namespace

std::string_view to_string(StackMapRootKind kind) noexcept {
    switch (kind) {
        case StackMapRootKind::FrameSlot: return "FrameSlot";
        case StackMapRootKind::CalleeSavedReg: return "CalleeSavedReg";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, StackMapRootKind kind) {
    return os << to_string(kind);
}

std::ostream& operator<<(std::ostream& os, const StackMapRootLocation& loc) {
    os << "RootLocation(" << to_string(loc.kind) << ", offset=" << loc.offset_from_rbp;
    if (loc.kind == StackMapRootKind::CalleeSavedReg) {
        os << ", reg=" << static_cast<int>(loc.reg.code);
    }
    if (loc.value == StackMapValueKind::Tagged) os << ", tagged";
    return os << ")";
}

std::ostream& operator<<(std::ostream& os, const StackMapRecord& rec) {
    return os << "StackMapRecord(offset=" << rec.instruction_offset << ", roots=" << rec.roots.size() << ")";
}

void FunctionStackMap::sort_records() {
    std::stable_sort(records.begin(), records.end(), [](const StackMapRecord& a, const StackMapRecord& b) {
        return a.instruction_offset < b.instruction_offset;
    });
    records_sorted = true;
}

const StackMapRecord* FunctionStackMap::find_record_by_offset(uint32_t call_offset) const noexcept {
    if (records_sorted) {
        auto it = std::lower_bound(records.begin(), records.end(), call_offset,
                                   [](const StackMapRecord& r, uint32_t off) { return r.instruction_offset < off; });
        return it != records.end() && it->instruction_offset == call_offset ? &*it : nullptr;
    }
    for (const auto& rec : records) {
        if (rec.instruction_offset == call_offset) {
            return &rec;
        }
    }
    return nullptr;
}

const StackMapRecord* FunctionStackMap::find_record_by_ip(uintptr_t return_ip) const noexcept {
    if (function_address == 0) return nullptr;
    if (return_ip < function_address) return nullptr;
    uint32_t offset = static_cast<uint32_t>(return_ip - function_address);
    return find_record_by_offset(offset);
}

void ModuleStackMap::add_function(FunctionStackMap fn_map) {
    for (size_t i = 0; i < functions_.size(); ++i) {
        if (functions_[i].function_name == fn_map.function_name) {
            functions_[i] = std::move(fn_map);
            return;
        }
    }
    functions_.push_back(std::move(fn_map));
}

const FunctionStackMap* ModuleStackMap::find_function_by_name(std::string_view name) const noexcept {
    for (const auto& fn : functions_) {
        if (fn.function_name == name) {
            return &fn;
        }
    }
    return nullptr;
}

const FunctionStackMap* ModuleStackMap::find_function_by_ip(uintptr_t ip) const noexcept {
    for (const auto& fn : functions_) {
        if (fn.function_address != 0) {
            if (fn.code_size > 0) {
                if (ip >= fn.function_address && ip < fn.function_address + fn.code_size) {
                    return &fn;
                }
            } else {
                // If code size is 0, check if any record matches
                if (ip >= fn.function_address) {
                    for (const auto& r : fn.records) {
                        if (fn.function_address + r.instruction_offset == ip) {
                            return &fn;
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

const StackMapRecord* ModuleStackMap::find_record(uintptr_t return_ip) const noexcept {
    for (const auto& fn : functions_) {
        if (fn.function_address != 0) {
            if (return_ip >= fn.function_address) {
                uint32_t offset = static_cast<uint32_t>(return_ip - fn.function_address);
                const auto* rec = fn.find_record_by_offset(offset);
                if (rec != nullptr) {
                    return rec;
                }
            }
        }
    }
    return nullptr;
}

void ModuleStackMap::relocate(uintptr_t text_base_address) noexcept {
    for (auto& fn : functions_) {
        fn.function_address = text_base_address + fn.code_offset;
    }
}

void ModuleStackMap::register_function_address(std::string_view name, uintptr_t address, uint32_t code_size) noexcept {
    for (auto& fn : functions_) {
        if (fn.function_name == name) {
            fn.function_address = address;
            if (code_size > 0) {
                fn.code_size = code_size;
            }
            return;
        }
    }

    FunctionStackMap new_fn;
    new_fn.function_name = std::string(name);
    new_fn.function_address = address;
    new_fn.code_size = code_size;
    functions_.push_back(std::move(new_fn));
}

// Version 3, compact: integers are LEB128 varints (signed ones zigzagged),
// and a record says only what differs from the record before it.
//   header   magic u32, version u32, function count
//   function name length + bytes, code offset, code size, slot area lo and
//            hi (signed), record count
//   record   instruction offset minus the previous record's (signed), a
//            flags byte, then what the flags name: kRecFrame its frame
//            size, kRecSafepoint its safepoint id (otherwise the previous
//            one plus one), kRecRoots its roots (otherwise the previous
//            record's): their count, and per root a tag byte (kind, value
//            kind, kRootReg when the register is not PReg()'s default, then
//            its class and code bytes) and its fp offset minus the previous
//            root's (signed).
// The first record's "previous" is offset 0, frame size 0, safepoint id
// UINT32_MAX (so an id of 0 needs no field) and no roots.
namespace {
constexpr uint8_t kRecFrame = 1;
constexpr uint8_t kRecSafepoint = 2;
constexpr uint8_t kRecRoots = 4;
constexpr uint8_t kRootKind = 1;
constexpr uint8_t kRootTagged = 2;
constexpr uint8_t kRootReg = 4;

bool default_reg(const codegen::PReg& r) noexcept {
    return r.reg_class == codegen::PReg().reg_class && r.code == codegen::PReg().code;
}
} // namespace

std::vector<uint8_t> encode_stack_maps(const ModuleStackMap& stack_maps) {
    std::vector<uint8_t> out;
    out.reserve(64);
    write_u32(out, STACK_MAP_MAGIC);
    write_u32(out, STACK_MAP_VERSION);
    write_var(out, stack_maps.functions().size());
    for (const auto& fn : stack_maps.functions()) {
        write_var(out, fn.function_name.size());
        write_bytes(out, fn.function_name.data(), fn.function_name.size());
        write_var(out, fn.code_offset);
        write_var(out, fn.code_size);
        write_svar(out, fn.slot_area_lo);
        write_svar(out, fn.slot_area_hi);
        write_var(out, fn.records.size());
        const StackMapRecord* prev = nullptr;
        uint32_t prev_offset = 0, prev_frame = 0, prev_safepoint = UINT32_MAX;
        for (const auto& rec : fn.records) {
            write_svar(out, static_cast<int64_t>(rec.instruction_offset) - static_cast<int64_t>(prev_offset));
            uint8_t flags = 0;
            if (rec.frame_size != prev_frame) flags |= kRecFrame;
            if (rec.safepoint_id != prev_safepoint + 1) flags |= kRecSafepoint;
            const bool same_roots = prev ? rec.roots == prev->roots : rec.roots.empty();
            if (!same_roots) flags |= kRecRoots;
            write_u8(out, flags);
            if (flags & kRecFrame) write_var(out, rec.frame_size);
            if (flags & kRecSafepoint) write_var(out, rec.safepoint_id);
            if (flags & kRecRoots) {
                write_var(out, rec.roots.size());
                int64_t prev_root = 0;
                for (const auto& root : rec.roots) {
                    const bool reg = !default_reg(root.reg);
                    write_u8(out, static_cast<uint8_t>((root.kind == StackMapRootKind::CalleeSavedReg ? kRootKind : 0) |
                                                       (root.value == StackMapValueKind::Tagged ? kRootTagged : 0) |
                                                       (reg ? kRootReg : 0)));
                    if (reg) {
                        write_u8(out, static_cast<uint8_t>(root.reg.reg_class));
                        write_u8(out, root.reg.code);
                    }
                    write_svar(out, root.offset_from_rbp - prev_root);
                    prev_root = root.offset_from_rbp;
                }
            }
            prev = &rec;
            prev_offset = rec.instruction_offset;
            prev_frame = rec.frame_size;
            prev_safepoint = rec.safepoint_id;
        }
    }
    return out;
}

namespace {

bool decode_v3(BinaryReader& reader, ModuleStackMap& out, uintptr_t text_base_address) {
    uint32_t fn_count = 0;
    if (!reader.read_var_as(fn_count)) return false;
    for (uint32_t f = 0; f < fn_count; ++f) {
        FunctionStackMap fn_map;
        uint32_t name_len = 0;
        if (!reader.read_var_as(name_len) || !reader.read_string(fn_map.function_name, name_len)) return false;
        int64_t lo = 0, hi = 0;
        if (!reader.read_var_as(fn_map.code_offset) || !reader.read_var_as(fn_map.code_size) ||
            !reader.read_svar(lo) || !reader.read_svar(hi)) {
            return false;
        }
        if (lo < INT32_MIN || hi > INT32_MAX || hi < lo) return false;
        fn_map.slot_area_lo = static_cast<int32_t>(lo);
        fn_map.slot_area_hi = static_cast<int32_t>(hi);
        if (text_base_address > 0) fn_map.function_address = text_base_address + fn_map.code_offset;
        uint32_t rec_count = 0;
        if (!reader.read_var_as(rec_count) || rec_count > reader.remaining()) return false;
        fn_map.records.reserve(rec_count);
        int64_t offset = 0;
        uint32_t frame = 0, safepoint = UINT32_MAX;
        for (uint32_t r = 0; r < rec_count; ++r) {
            StackMapRecord rec;
            int64_t delta = 0;
            uint8_t flags = 0;
            if (!reader.read_svar(delta) || !reader.read_u8(flags)) return false;
            offset += delta;
            if (offset < 0 || offset > UINT32_MAX) return false;
            rec.instruction_offset = static_cast<uint32_t>(offset);
            if ((flags & kRecFrame) && !reader.read_var_as(frame)) return false;
            safepoint += 1;
            if ((flags & kRecSafepoint) && !reader.read_var_as(safepoint)) return false;
            rec.frame_size = frame;
            rec.safepoint_id = safepoint;
            if (flags & kRecRoots) {
                uint32_t root_count = 0;
                if (!reader.read_var_as(root_count) || root_count > reader.remaining()) return false;
                rec.roots.reserve(root_count);
                int64_t root_offset = 0;
                for (uint32_t k = 0; k < root_count; ++k) {
                    StackMapRootLocation root;
                    uint8_t tag = 0;
                    if (!reader.read_u8(tag) || tag > (kRootKind | kRootTagged | kRootReg)) return false;
                    root.kind = (tag & kRootKind) ? StackMapRootKind::CalleeSavedReg : StackMapRootKind::FrameSlot;
                    root.value = (tag & kRootTagged) ? StackMapValueKind::Tagged : StackMapValueKind::GcRef;
                    if (tag & kRootReg) {
                        uint8_t reg_class = 0;
                        if (!reader.read_u8(reg_class) || !reader.read_u8(root.reg.code)) return false;
                        root.reg.reg_class = static_cast<codegen::RegClass>(reg_class);
                    }
                    int64_t d = 0;
                    if (!reader.read_svar(d)) return false;
                    root_offset += d;
                    if (root_offset < INT32_MIN || root_offset > INT32_MAX) return false;
                    root.offset_from_rbp = static_cast<int32_t>(root_offset);
                    rec.roots.push_back(root);
                }
            } else if (!fn_map.records.empty()) {
                rec.roots = fn_map.records.back().roots;
            }
            fn_map.records.push_back(std::move(rec));
        }
        out.add_function(std::move(fn_map));
    }
    return true;
}

} // namespace

bool decode_stack_maps_into(std::span<const uint8_t> data, ModuleStackMap& out, uintptr_t text_base_address) {
    out.clear();
    BinaryReader reader(data);

    uint32_t magic = 0;
    if (!reader.read_u32(magic) || magic != STACK_MAP_MAGIC) {
        return false;
    }

    uint32_t version = 0;
    if (!reader.read_u32(version) || version < 1 || version > STACK_MAP_VERSION) {
        return false;
    }
    if (version == 3) return decode_v3(reader, out, text_base_address);

    uint32_t fn_count = 0;
    if (!reader.read_u32(fn_count)) {
        return false;
    }

    for (uint32_t f = 0; f < fn_count; ++f) {
        FunctionStackMap fn_map;
        uint32_t name_len = 0;
        if (!reader.read_u32(name_len)) return false;
        if (!reader.read_string(fn_map.function_name, name_len)) return false;

        uint64_t code_offset = 0;
        if (!reader.read_u64(code_offset)) return false;
        fn_map.code_offset = static_cast<uint32_t>(code_offset);

        if (!reader.read_u32(fn_map.code_size)) return false;
        if (version >= 2) {
            if (!reader.read_i32(fn_map.slot_area_lo) || !reader.read_i32(fn_map.slot_area_hi)) return false;
            if (fn_map.slot_area_hi < fn_map.slot_area_lo) return false;
        }

        if (text_base_address > 0) {
            fn_map.function_address = text_base_address + fn_map.code_offset;
        }

        uint32_t rec_count = 0;
        if (!reader.read_u32(rec_count)) return false;

        for (uint32_t r = 0; r < rec_count; ++r) {
            StackMapRecord rec;
            if (!reader.read_u32(rec.instruction_offset)) return false;
            if (!reader.read_u32(rec.frame_size)) return false;
            if (!reader.read_u32(rec.safepoint_id)) return false;

            uint32_t root_count = 0;
            if (!reader.read_u32(root_count)) return false;

            for (uint32_t k = 0; k < root_count; ++k) {
                StackMapRootLocation root;
                if (!reader.read_i32(root.offset_from_rbp)) return false;
                uint8_t kind = 0, reg_class = 0, reg_code = 0, value = 0;
                if (!reader.read_u8(kind)) return false;
                if (!reader.read_u8(reg_class)) return false;
                if (!reader.read_u8(reg_code)) return false;
                if (!reader.read_u8(value)) return false;
                if (kind > 1 || value > 1) return false;

                root.kind = static_cast<StackMapRootKind>(kind);
                root.reg.reg_class = static_cast<codegen::RegClass>(reg_class);
                root.reg.code = reg_code;
                root.value = static_cast<StackMapValueKind>(value);
                rec.roots.push_back(root);
            }

            fn_map.records.push_back(std::move(rec));
        }

        out.add_function(std::move(fn_map));
    }

    return true;
}

ModuleStackMap decode_stack_maps(std::span<const uint8_t> data, uintptr_t text_base_address) {
    ModuleStackMap res;
    decode_stack_maps_into(data, res, text_base_address);
    return res;
}

bool peek_stack_maps(std::span<const uint8_t> data, uint32_t& function_count, uint32_t& first_record_count) {
    function_count = 0;
    first_record_count = 0;
    BinaryReader reader(data);
    uint32_t magic = 0, version = 0;
    if (!reader.read_u32(magic) || magic != STACK_MAP_MAGIC) return false;
    if (!reader.read_u32(version) || version < 1 || version > STACK_MAP_VERSION) return false;
    if (version == 3) {
        if (!reader.read_var_as(function_count)) return false;
        if (function_count == 0) return true;
        uint32_t name_len = 0, code_size = 0, code_offset = 0;
        int64_t area = 0;
        if (!reader.read_var_as(name_len) || !reader.skip(name_len)) return false;
        if (!reader.read_var_as(code_offset) || !reader.read_var_as(code_size)) return false;
        if (!reader.read_svar(area) || !reader.read_svar(area)) return false;
        return reader.read_var_as(first_record_count);
    }
    if (!reader.read_u32(function_count)) return false;
    if (function_count == 0) return true;
    // The first function: its name, code offset and size, then its count.
    uint32_t name_len = 0, code_size = 0;
    uint64_t code_offset = 0;
    if (!reader.read_u32(name_len) || !reader.skip(name_len)) return false;
    if (!reader.read_u64(code_offset) || !reader.read_u32(code_size)) return false;
    if (version >= 2 && !reader.skip(8)) return false;
    return reader.read_u32(first_record_count);
}

} // namespace brass
