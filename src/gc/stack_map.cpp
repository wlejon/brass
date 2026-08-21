#include <brass/gc/stack_map.hpp>
#include <cstring>
#include <algorithm>

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

    bool read_string(std::string& s, size_t len) {
        if (!has_bytes(len)) return false;
        s.assign(reinterpret_cast<const char*>(data_.data() + pos_), len);
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
    return os << ")";
}

std::ostream& operator<<(std::ostream& os, const StackMapRecord& rec) {
    return os << "StackMapRecord(offset=" << rec.instruction_offset << ", roots=" << rec.roots.size() << ")";
}

const StackMapRecord* FunctionStackMap::find_record_by_offset(uint32_t call_offset) const noexcept {
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

std::vector<uint8_t> encode_stack_maps(const ModuleStackMap& stack_maps) {
    std::vector<uint8_t> out;
    out.reserve(256);

    // 1. Header
    write_u32(out, STACK_MAP_MAGIC);
    write_u32(out, STACK_MAP_VERSION);
    write_u32(out, static_cast<uint32_t>(stack_maps.functions().size()));

    // 2. Function Records
    for (const auto& fn : stack_maps.functions()) {
        uint32_t name_len = static_cast<uint32_t>(fn.function_name.size());
        write_u32(out, name_len);
        if (name_len > 0) {
            write_bytes(out, fn.function_name.data(), name_len);
        }
        write_u64(out, static_cast<uint64_t>(fn.code_offset));
        write_u32(out, fn.code_size);
        write_u32(out, static_cast<uint32_t>(fn.records.size()));

        for (const auto& rec : fn.records) {
            write_u32(out, rec.instruction_offset);
            write_u32(out, rec.frame_size);
            write_u32(out, rec.safepoint_id);
            write_u32(out, static_cast<uint32_t>(rec.roots.size()));

            for (const auto& root : rec.roots) {
                write_i32(out, root.offset_from_rbp);
                write_u8(out, static_cast<uint8_t>(root.kind));
                write_u8(out, static_cast<uint8_t>(root.reg.reg_class));
                write_u8(out, root.reg.code);
                write_u8(out, 0); // reserved
            }
        }
    }

    return out;
}

bool decode_stack_maps_into(std::span<const uint8_t> data, ModuleStackMap& out, uintptr_t text_base_address) {
    out.clear();
    BinaryReader reader(data);

    uint32_t magic = 0;
    if (!reader.read_u32(magic) || magic != STACK_MAP_MAGIC) {
        return false;
    }

    uint32_t version = 0;
    if (!reader.read_u32(version) || version != STACK_MAP_VERSION) {
        return false;
    }

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
                uint8_t kind = 0, reg_class = 0, reg_code = 0, res = 0;
                if (!reader.read_u8(kind)) return false;
                if (!reader.read_u8(reg_class)) return false;
                if (!reader.read_u8(reg_code)) return false;
                if (!reader.read_u8(res)) return false;

                root.kind = static_cast<StackMapRootKind>(kind);
                root.reg.reg_class = static_cast<codegen::RegClass>(reg_class);
                root.reg.code = reg_code;
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

} // namespace brass
