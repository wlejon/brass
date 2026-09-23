#include <brass/object/coff_writer.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brass::object {

namespace {

void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

void write_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t count) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

uint32_t get_coff_section_characteristics(const Section& sec) {
    uint32_t flags = 0;
    if (has_flag(sec.flags, SectionFlags::Execute)) {
        flags |= coff::IMAGE_SCN_CNT_CODE | coff::IMAGE_SCN_MEM_EXECUTE | coff::IMAGE_SCN_MEM_READ;
        flags |= coff::IMAGE_SCN_ALIGN_16BYTES;
    } else {
        flags |= coff::IMAGE_SCN_CNT_INITIALIZED_DATA | coff::IMAGE_SCN_MEM_READ;
        if (has_flag(sec.flags, SectionFlags::Write)) {
            flags |= coff::IMAGE_SCN_MEM_WRITE;
        }
        if (has_flag(sec.flags, SectionFlags::Discardable)) {
            flags |= coff::IMAGE_SCN_MEM_DISCARDABLE;
            flags |= coff::IMAGE_SCN_ALIGN_4BYTES;
        } else {
            // Loaded data carries 8-byte pointer slots and 16-byte constants;
            // an in-section align_to() means nothing unless the linker places
            // the section itself at that alignment. The field is log2(align)+1.
            uint32_t align = sec.alignment < 4 ? 4 : (sec.alignment > 8192 ? 8192 : sec.alignment);
            uint32_t log2 = 0;
            while ((1u << (log2 + 1)) <= align) ++log2;
            flags |= (log2 + 1) << 20;
        }
    }
    return flags;
}

[[noreturn]] void bad_reloc(RelocKind kind, bool is_aarch64) {
    throw std::runtime_error("COFF writer: relocation kind " + std::to_string(static_cast<int>(kind)) +
                             " has no " + (is_aarch64 ? "ARM64" : "AMD64") + " COFF equivalent");
}

uint16_t to_coff_reloc_type(RelocKind kind, bool is_aarch64) {
    if (is_aarch64) {
        switch (kind) {
            case RelocKind::Plt32:       return coff::IMAGE_REL_ARM64_BRANCH26;
            case RelocKind::PCRel32:     return coff::IMAGE_REL_ARM64_REL32;
            case RelocKind::AdrPage21:   return coff::IMAGE_REL_ARM64_PAGE21;
            // ADD takes the unscaled low 12 bits (12A); a load/store the low
            // 12 bits scaled by its access size, which the linker reads from
            // the instruction (12L).
            case RelocKind::AddLo12:     return coff::IMAGE_REL_ARM64_PAGEOFFSET_12A;
            case RelocKind::LdSt8Lo12:
            case RelocKind::LdSt16Lo12:
            case RelocKind::LdSt32Lo12:
            case RelocKind::LdSt64Lo12:
            case RelocKind::LdSt128Lo12: return coff::IMAGE_REL_ARM64_PAGEOFFSET_12L;
            case RelocKind::SecRel32:    return coff::IMAGE_REL_ARM64_SECREL;
            case RelocKind::Abs64:       return coff::IMAGE_REL_ARM64_ADDR64;
            case RelocKind::Addr32NB:    return coff::IMAGE_REL_ARM64_ADDR32NB;
            case RelocKind::SecIdx:      return coff::IMAGE_REL_ARM64_SECTION;
            case RelocKind::Abs32:       return coff::IMAGE_REL_ARM64_ADDR32;
            // COFF has no GOT: materialize_got_slots rewrites these first.
            case RelocKind::GotPage21:
            case RelocKind::GotLo12:
            case RelocKind::GotPCRel32:
                bad_reloc(kind, true);
        }
        bad_reloc(kind, true);
    }
    switch (kind) {
        case RelocKind::AdrPage21:
        case RelocKind::AddLo12:
        case RelocKind::LdSt8Lo12:
        case RelocKind::LdSt16Lo12:
        case RelocKind::LdSt32Lo12:
        case RelocKind::LdSt64Lo12:
        case RelocKind::LdSt128Lo12:
        case RelocKind::GotPage21:
        case RelocKind::GotLo12:
        case RelocKind::GotPCRel32:   // none left after materialize_got_slots
            bad_reloc(kind, false);
        case RelocKind::PCRel32:
        case RelocKind::Plt32:
            return coff::IMAGE_REL_AMD64_REL32;
        case RelocKind::Abs64:
            return coff::IMAGE_REL_AMD64_ADDR64;
        case RelocKind::Addr32NB:
            return coff::IMAGE_REL_AMD64_ADDR32NB;
        case RelocKind::SecRel32:
            return coff::IMAGE_REL_AMD64_SECREL;
        case RelocKind::SecIdx:
            return coff::IMAGE_REL_AMD64_SECTION;
        case RelocKind::Abs32:
            return coff::IMAGE_REL_AMD64_ADDR32;
    }
    return coff::IMAGE_REL_AMD64_REL32;
}

struct CoffSymbolRecord {
    std::string name;
    uint32_t value = 0;
    int16_t section_number = 0; // 1-based, 0 = UNDEF
    uint16_t type = 0;          // 0x20 for DT_FCN
    uint8_t storage_class = 2;  // 2 = EXTERNAL, 3 = STATIC
};

} // namespace

CoffWriter::CoffWriter(const ObjectFile& obj)
    : obj_(obj) {}

std::vector<uint8_t> CoffWriter::write() {
    ObjectFile working_obj = obj_;
    // COFF has no GOT-relative relocation for a system linker to bind, so
    // the object carries its own slots: an ADDR64 word per undefined symbol
    // in .rdata (a base relocation in the image, like a vtable's), the load
    // a REL32 onto it. Loads of the object's own symbols become `lea`s.
    materialize_got_slots(working_obj, ".rdata", SectionKind::RoData,
                          SectionFlags::Read | SectionFlags::Alloc);

    // Generate Win64 SEH tables if functions exist
    if (!working_obj.functions.empty()) {
        working_obj.get_or_create_section(
            ".xdata",
            SectionKind::XData,
            SectionFlags::Read | SectionFlags::Alloc,
            4
        );
        working_obj.get_or_create_section(
            ".pdata",
            SectionKind::PData,
            SectionFlags::Read | SectionFlags::Alloc,
            4
        );
        Section* pdata_sec = working_obj.get_section(".pdata");
        Section* xdata_sec = working_obj.get_section(".xdata");
        if (pdata_sec && xdata_sec) {
            CoffUnwindBuilder::build_unwind_info(working_obj, *pdata_sec, *xdata_sec);
        }
    }

    // Generate Win64 CodeView debug info (.debug$S and .debug$T)
    if ((!working_obj.debug_tables.empty() || working_obj.debug_context.file_count() > 0) &&
        !working_obj.get_section(".debug$S")) {
        debug::CodeViewEmitter::emit(working_obj);
    }

    // Build String Table and Symbol Table
    std::vector<uint8_t> string_table;
    std::unordered_map<std::string, uint32_t> str_offsets;

    auto get_or_add_string = [&](const std::string& s) -> uint32_t {
        if (s.size() <= 8) return 0;
        auto it = str_offsets.find(s);
        if (it != str_offsets.end()) return it->second;
        uint32_t off = 4 + static_cast<uint32_t>(string_table.size());
        str_offsets[s] = off;
        string_table.insert(string_table.end(), s.begin(), s.end());
        string_table.push_back(0);
        return off;
    };

    std::vector<CoffSymbolRecord> coff_symbols;
    std::unordered_map<std::string, uint32_t> sym_index_map;

    // 1. Section symbols (Static)
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        CoffSymbolRecord rec;
        rec.name = sec.name;
        rec.value = 0;
        rec.section_number = static_cast<int16_t>(i + 1);
        rec.type = 0;
        rec.storage_class = coff::IMAGE_SYM_CLASS_STATIC;
        uint32_t idx = static_cast<uint32_t>(coff_symbols.size());
        sym_index_map[sec.name] = idx;
        coff_symbols.push_back(rec);
        get_or_add_string(sec.name);
    }

    // 2. Symbols from ObjectFile
    for (const auto& sym : working_obj.symbols) {
        if (sym.type == SymbolType::Section) continue;
        CoffSymbolRecord rec;
        rec.name = sym.name;
        rec.value = static_cast<uint32_t>(sym.value);
        rec.section_number = (sym.section_index >= 0) ? static_cast<int16_t>(sym.section_index + 1) : 0;
        rec.type = (sym.type == SymbolType::Function) ? coff::IMAGE_SYM_DTYPE_FUNCTION : 0;
        rec.storage_class = (sym.binding == SymbolBinding::Local) ? coff::IMAGE_SYM_CLASS_STATIC : coff::IMAGE_SYM_CLASS_EXTERNAL;

        uint32_t idx = static_cast<uint32_t>(coff_symbols.size());
        sym_index_map[sym.name] = idx;
        coff_symbols.push_back(rec);
        get_or_add_string(sym.name);
    }

    // Prepare Section Headers layout
    uint16_t num_sections = static_cast<uint16_t>(working_obj.sections.size());
    uint32_t file_header_size = 20;
    uint32_t section_headers_size = num_sections * 40;
    uint32_t cur_file_offset = file_header_size + section_headers_size;

    struct SectionLayout {
        uint32_t raw_data_offset = 0;
        uint32_t raw_data_size = 0;
        uint32_t reloc_offset = 0;
        uint16_t num_relocs = 0;
    };
    std::vector<SectionLayout> layouts(num_sections);

    // 1. Raw section data
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (!sec.data.empty()) {
            layouts[i].raw_data_offset = cur_file_offset;
            layouts[i].raw_data_size = static_cast<uint32_t>(sec.data.size());
            cur_file_offset += layouts[i].raw_data_size;
        }
    }

    // 2. Relocations for each section
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        if (!sec.relocations.empty()) {
            layouts[i].reloc_offset = cur_file_offset;
            const bool overflow = sec.relocations.size() >= 0xFFFF;
            layouts[i].num_relocs = overflow ? 0xFFFF : static_cast<uint16_t>(sec.relocations.size());
            const uint32_t total_records = overflow ? static_cast<uint32_t>(sec.relocations.size() + 1)
                                                    : static_cast<uint32_t>(sec.relocations.size());
            cur_file_offset += total_records * 10;
        }
    }

    // 3. Symbol table offset
    uint32_t sym_table_offset = cur_file_offset;
    uint32_t num_symbols = static_cast<uint32_t>(coff_symbols.size());

    // Begin building binary buffer
    std::vector<uint8_t> out;
    out.reserve(sym_table_offset + num_symbols * 18 + string_table.size() + 4);

    // Write IMAGE_FILE_HEADER (20 bytes)
    uint16_t machine = working_obj.target.is_aarch64() ? coff::IMAGE_FILE_MACHINE_ARM64 : coff::IMAGE_FILE_MACHINE_AMD64;
    write_u16(out, machine);
    write_u16(out, num_sections);
    write_u32(out, 0); // TimeDateStamp
    write_u32(out, sym_table_offset);
    write_u32(out, num_symbols);
    write_u16(out, 0); // SizeOfOptionalHeader
    write_u16(out, 0); // Characteristics

    // Write IMAGE_SECTION_HEADER entries (40 bytes each)
    for (size_t i = 0; i < working_obj.sections.size(); ++i) {
        const auto& sec = working_obj.sections[i];
        const auto& lay = layouts[i];

        // 8 bytes Name
        uint8_t name_buf[8] = {0};
        if (sec.name.size() <= 8) {
            std::memcpy(name_buf, sec.name.data(), sec.name.size());
        } else {
            uint32_t str_off = get_or_add_string(sec.name);
            std::string slash_str = "/" + std::to_string(str_off);
            std::memcpy(name_buf, slash_str.data(), std::min(slash_str.size(), size_t(8)));
        }
        write_bytes(out, name_buf, 8);

        write_u32(out, 0); // VirtualSize
        write_u32(out, 0); // VirtualAddress
        write_u32(out, lay.raw_data_size);
        write_u32(out, lay.raw_data_offset);
        write_u32(out, lay.reloc_offset);
        write_u32(out, 0); // PointerToLinenumbers
        write_u16(out, lay.num_relocs);
        write_u16(out, 0); // NumberOfLinenumbers
        uint32_t chars = get_coff_section_characteristics(sec);
        if (sec.relocations.size() >= 0xFFFF) {
            chars |= coff::IMAGE_SCN_LNK_NRELOC_OVFL;
        }
        write_u32(out, chars);
    }

    // Patch inline addends into section data for COFF relocations.
    // Relocation addends follow the S + A - P convention (a rip-relative
    // displacement at the end of its instruction carries A = -4), while an
    // AMD64 REL32 is already measured from the end of its 4-byte field, so the
    // inline value it needs is A + 4.
    const bool rel32_from_field_end = !working_obj.target.is_aarch64();
    for (auto& sec : working_obj.sections) {
        for (const auto& r : sec.relocations) {
            if (!rel32_from_field_end && (a64::is_instruction_kind(r.kind) || r.kind == RelocKind::Plt32)) {
                // ARM64 COFF relocations carry the addend in the instruction's
                // own immediate (ADRP: bytes; ADD: bytes; LDR/STR: scaled).
                if (r.addend == 0) continue;
                uint32_t inst = 0;
                if (r.offset + 4 > sec.data.size()) {
                    throw std::runtime_error("COFF writer: relocation outside section " + sec.name);
                }
                std::memcpy(&inst, sec.data.data() + r.offset, 4);
                if (!a64::encode_implicit_addend(r.kind, inst, r.addend)) {
                    throw std::runtime_error("COFF writer: addend " + std::to_string(r.addend) + " against '" +
                                             r.symbol_name + "' does not fit the ARM64 instruction's immediate");
                }
                std::memcpy(sec.data.data() + r.offset, &inst, 4);
                continue;
            }
            if (rel32_from_field_end &&
                (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::Plt32)) {
                if (r.addend + 4 != 0 && r.offset + 4 <= sec.data.size()) {
                    uint32_t current_val = 0;
                    std::memcpy(&current_val, sec.data.data() + r.offset, sizeof(uint32_t));
                    if (current_val == 0) {
                        uint32_t inline_val = static_cast<uint32_t>(static_cast<int32_t>(r.addend + 4));
                        std::memcpy(sec.data.data() + r.offset, &inline_val, sizeof(uint32_t));
                    }
                }
                continue;
            }
            if (r.addend != 0) {
                if (r.kind == RelocKind::Addr32NB || r.kind == RelocKind::PCRel32 ||
                    r.kind == RelocKind::SecRel32 || r.kind == RelocKind::Abs32) {
                    if (r.offset + 4 <= sec.data.size()) {
                        uint32_t current_val = 0;
                        std::memcpy(&current_val, sec.data.data() + r.offset, sizeof(uint32_t));
                        if (current_val == 0) {
                            uint32_t addend_val = static_cast<uint32_t>(r.addend);
                            std::memcpy(sec.data.data() + r.offset, &addend_val, sizeof(uint32_t));
                        }
                    }
                } else if (r.kind == RelocKind::Abs64) {
                    if (r.offset + 8 <= sec.data.size()) {
                        uint64_t current_val = 0;
                        std::memcpy(&current_val, sec.data.data() + r.offset, sizeof(uint64_t));
                        if (current_val == 0) {
                            uint64_t addend_val = static_cast<uint64_t>(r.addend);
                            std::memcpy(sec.data.data() + r.offset, &addend_val, sizeof(uint64_t));
                        }
                    }
                }
            }
        }
    }

    // Write Raw Section Data
    for (const auto& sec : working_obj.sections) {
        if (!sec.data.empty()) {
            write_bytes(out, sec.data.data(), sec.data.size());
        }
    }

    // Write Section Relocations (10 bytes each)
    for (const auto& sec : working_obj.sections) {
        if (sec.relocations.empty()) continue;
        if (sec.relocations.size() >= 0xFFFF) {
            // In case of overflow, write actual relocation count as first relocation (+1 for synthetic entry)
            write_u32(out, static_cast<uint32_t>(sec.relocations.size() + 1));
            write_u32(out, 0);
            write_u16(out, 0);
        }
        for (const auto& r : sec.relocations) {
            write_u32(out, static_cast<uint32_t>(r.offset));
            uint32_t sym_idx = 0;
            auto it = sym_index_map.find(r.symbol_name);
            if (it != sym_index_map.end()) {
                sym_idx = it->second;
            }
            write_u32(out, sym_idx);
            write_u16(out, to_coff_reloc_type(r.kind, working_obj.target.is_aarch64()));
        }
    }

    // Write Symbol Table (18 bytes each)
    for (const auto& sym : coff_symbols) {
        uint8_t name_buf[8] = {0};
        if (sym.name.size() <= 8) {
            std::memcpy(name_buf, sym.name.data(), sym.name.size());
        } else {
            uint32_t off = str_offsets[sym.name];
            // First 4 bytes are zeroes, next 4 bytes are string table offset
            name_buf[0] = 0;
            name_buf[1] = 0;
            name_buf[2] = 0;
            name_buf[3] = 0;
            name_buf[4] = static_cast<uint8_t>(off & 0xFF);
            name_buf[5] = static_cast<uint8_t>((off >> 8) & 0xFF);
            name_buf[6] = static_cast<uint8_t>((off >> 16) & 0xFF);
            name_buf[7] = static_cast<uint8_t>((off >> 24) & 0xFF);
        }
        write_bytes(out, name_buf, 8);
        write_u32(out, sym.value);
        write_u16(out, static_cast<uint16_t>(sym.section_number));
        write_u16(out, sym.type);
        write_u8(out, sym.storage_class);
        write_u8(out, 0); // NumberOfAuxSymbols
    }

    // Write String Table
    uint32_t total_str_size = static_cast<uint32_t>(string_table.size() + 4);
    write_u32(out, total_str_size);
    if (!string_table.empty()) {
        write_bytes(out, string_table.data(), string_table.size());
    }

    return out;
}

bool CoffWriter::write_to_file(const std::string& path) {
    auto data = write();
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

std::vector<uint8_t> emit_coff_object(const ObjectFile& obj) {
    CoffWriter writer(obj);
    return writer.write();
}

} // namespace brass::object
