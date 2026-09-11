#include <brass/object/coff_writer.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <fstream>
#include <cstring>
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
        }
        flags |= coff::IMAGE_SCN_ALIGN_4BYTES;
    }
    return flags;
}

uint16_t to_coff_reloc_type(RelocKind kind) {
    switch (kind) {
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
    write_u16(out, coff::IMAGE_FILE_MACHINE_AMD64);
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
            write_u16(out, to_coff_reloc_type(r.kind));
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
