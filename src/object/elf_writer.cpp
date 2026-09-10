#include <brass/object/elf_writer.hpp>
#include <brass/debug/dwarf_emitter.hpp>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <algorithm>

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

void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

void write_i64(std::vector<uint8_t>& buf, int64_t v) {
    write_u64(buf, static_cast<uint64_t>(v));
}

void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t count) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

void align_buf(std::vector<uint8_t>& buf, size_t align) {
    if (align <= 1) return;
    size_t rem = buf.size() % align;
    if (rem != 0) {
        buf.resize(buf.size() + (align - rem), 0);
    }
}

uint32_t to_elf_reloc_type(RelocKind kind) {
    switch (kind) {
        case RelocKind::PCRel32: return elf::R_X86_64_PC32;
        case RelocKind::Plt32:   return elf::R_X86_64_PLT32;
        case RelocKind::Abs64:   return elf::R_X86_64_64;
        case RelocKind::Abs32:   return elf::R_X86_64_32;
        case RelocKind::Addr32NB: return elf::R_X86_64_32;
        case RelocKind::SecRel32: return elf::R_X86_64_32;
        case RelocKind::SecIdx:   return elf::R_X86_64_NONE;
    }
    return elf::R_X86_64_PC32;
}

uint64_t to_elf_section_flags(const Section& sec) {
    uint64_t flags = 0;
    if (has_flag(sec.flags, SectionFlags::Alloc)) {
        flags |= elf::SHF_ALLOC;
    }
    if (has_flag(sec.flags, SectionFlags::Execute)) {
        flags |= elf::SHF_EXECINSTR | elf::SHF_ALLOC;
    }
    if (has_flag(sec.flags, SectionFlags::Write)) {
        flags |= elf::SHF_WRITE;
    }
    return flags;
}

uint32_t to_elf_section_type(const Section& sec) {
    if (sec.kind == SectionKind::Bss) return elf::SHT_NOBITS;
    return elf::SHT_PROGBITS;
}

struct ElfShdrEntry {
    std::string name;
    uint32_t sh_name = 0;
    uint32_t sh_type = elf::SHT_NULL;
    uint64_t sh_flags = 0;
    uint64_t sh_addr = 0;
    uint64_t sh_offset = 0;
    uint64_t sh_size = 0;
    uint32_t sh_link = 0;
    uint32_t sh_info = 0;
    uint64_t sh_addralign = 1;
    uint64_t sh_entsize = 0;
    std::vector<uint8_t> data;
};

struct ElfSymEntry {
    std::string name;
    uint32_t st_name = 0;
    uint8_t st_info = 0;
    uint8_t st_other = 0;
    uint16_t st_shndx = 0;
    uint64_t st_value = 0;
    uint64_t st_size = 0;
};

} // namespace

ElfWriter::ElfWriter(const ObjectFile& obj)
    : obj_(obj) {}

std::vector<uint8_t> ElfWriter::write() {
    ObjectFile working_obj = obj_;

    // Generate SysV DWARF .eh_frame
    if (!working_obj.functions.empty()) {
        working_obj.get_or_create_section(
            ".eh_frame",
            SectionKind::EhFrame,
            SectionFlags::Read | SectionFlags::Alloc,
            8
        );
        Section* eh_frame_sec = working_obj.get_section(".eh_frame");
        if (eh_frame_sec) {
            ElfCfiBuilder::build_eh_frame(working_obj, *eh_frame_sec);
        }
    }

    // Generate SysV DWARF debug info (.debug_line, .debug_info, .debug_abbrev, .debug_str)
    if ((!working_obj.debug_tables.empty() || working_obj.debug_context.file_count() > 0) &&
        !working_obj.get_section(".debug_line")) {
        debug::DwarfEmitter::emit(working_obj);
    }

    std::vector<ElfShdrEntry> elf_sections;

    // 0. NULL section
    {
        ElfShdrEntry null_sec;
        null_sec.name = "";
        null_sec.sh_type = elf::SHT_NULL;
        elf_sections.push_back(std::move(null_sec));
    }

    // 1. Program sections from ObjectFile
    std::unordered_map<std::string, uint32_t> sec_name_to_shndx;
    for (const auto& sec : working_obj.sections) {
        ElfShdrEntry s;
        s.name = sec.name;
        s.sh_type = to_elf_section_type(sec);
        s.sh_flags = to_elf_section_flags(sec);
        s.sh_addralign = sec.alignment;
        s.data = sec.data;
        s.sh_size = s.data.size();
        uint32_t shndx = static_cast<uint32_t>(elf_sections.size());
        sec_name_to_shndx[sec.name] = shndx;
        elf_sections.push_back(std::move(s));
    }

    // 2. Build Symbol Table (.symtab) and String Table (.strtab)
    std::vector<uint8_t> strtab;
    strtab.push_back(0); // 0th byte is null

    auto add_str = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(strtab.size());
        strtab.insert(strtab.end(), s.begin(), s.end());
        strtab.push_back(0);
        return off;
    };

    std::vector<ElfSymEntry> local_syms;
    std::vector<ElfSymEntry> global_syms;

    // Symbol 0: NULL symbol
    ElfSymEntry null_sym;
    null_sym.name = "";
    local_syms.push_back(null_sym);

    // Section symbols (STT_SECTION, STB_LOCAL)
    for (size_t i = 1; i < elf_sections.size(); ++i) {
        ElfSymEntry sym;
        sym.name = elf_sections[i].name;
        sym.st_name = 0; // Section symbols have no name in strtab
        sym.st_info = (elf::STB_LOCAL << 4) | (elf::STT_SECTION & 0xF);
        sym.st_shndx = static_cast<uint16_t>(i);
        sym.st_value = 0;
        sym.st_size = 0;
        local_syms.push_back(sym);
    }

    // Other symbols from ObjectFile
    for (const auto& sym : working_obj.symbols) {
        if (sym.type == SymbolType::Section) continue;
        ElfSymEntry esym;
        esym.name = sym.name;
        esym.st_name = add_str(sym.name);

        uint8_t binding = (sym.binding == SymbolBinding::Local) ? elf::STB_LOCAL : elf::STB_GLOBAL;
        uint8_t type = (sym.type == SymbolType::Function) ? elf::STT_FUNC : (sym.type == SymbolType::Object ? elf::STT_OBJECT : elf::STT_NOTYPE);
        esym.st_info = (binding << 4) | (type & 0xF);

        if (sym.section_index >= 0 && sym.section_index < static_cast<int32_t>(working_obj.sections.size())) {
            const auto& sec_name = working_obj.sections[sym.section_index].name;
            esym.st_shndx = static_cast<uint16_t>(sec_name_to_shndx[sec_name]);
        } else {
            esym.st_shndx = 0; // SHN_UNDEF
        }

        esym.st_value = sym.value;
        esym.st_size = sym.size;

        if (binding == elf::STB_LOCAL) {
            local_syms.push_back(esym);
        } else {
            global_syms.push_back(esym);
        }
    }

    // Merge symbols: locals first, then globals
    uint32_t first_global_idx = static_cast<uint32_t>(local_syms.size());
    std::vector<ElfSymEntry> all_symbols = std::move(local_syms);
    all_symbols.insert(all_symbols.end(), global_syms.begin(), global_syms.end());

    std::unordered_map<std::string, uint32_t> sym_name_to_idx;
    for (size_t i = 0; i < all_symbols.size(); ++i) {
        if (!all_symbols[i].name.empty()) {
            sym_name_to_idx[all_symbols[i].name] = static_cast<uint32_t>(i);
        }
    }

    // 3. Create Relocation Sections (.rela.<sec>)
    for (const auto& sec : working_obj.sections) {
        if (sec.relocations.empty()) continue;

        ElfShdrEntry rela_sec;
        rela_sec.name = ".rela" + sec.name;
        rela_sec.sh_type = elf::SHT_RELA;
        rela_sec.sh_flags = elf::SHF_INFO_LINK;
        rela_sec.sh_addralign = 8;
        rela_sec.sh_entsize = 24;
        rela_sec.sh_info = sec_name_to_shndx[sec.name]; // target section index

        for (const auto& r : sec.relocations) {
            uint32_t sym_idx = 0;
            auto it = sym_name_to_idx.find(r.symbol_name);
            if (it != sym_name_to_idx.end()) {
                sym_idx = it->second;
            }
            uint32_t r_type = to_elf_reloc_type(r.kind);
            uint64_t r_info = (static_cast<uint64_t>(sym_idx) << 32) | (static_cast<uint64_t>(r_type) & 0xFFFFFFFFULL);
            int64_t addend = r.addend;
            if (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::Plt32) {
                // In standard x86_64 ELF rela, PC-relative call displacement fixup has addend -4
                if (addend == 0) addend = -4;
            }

            write_u64(rela_sec.data, static_cast<uint64_t>(r.offset));
            write_u64(rela_sec.data, r_info);
            write_i64(rela_sec.data, addend);
        }

        rela_sec.sh_size = rela_sec.data.size();
        elf_sections.push_back(std::move(rela_sec));
    }

    // 4. .symtab section
    uint32_t symtab_shndx = static_cast<uint32_t>(elf_sections.size());
    uint32_t strtab_shndx = symtab_shndx + 1;
    uint32_t shstrtab_shndx = symtab_shndx + 2;

    {
        ElfShdrEntry symtab_sec;
        symtab_sec.name = ".symtab";
        symtab_sec.sh_type = elf::SHT_SYMTAB;
        symtab_sec.sh_flags = 0;
        symtab_sec.sh_addralign = 8;
        symtab_sec.sh_entsize = 24;
        symtab_sec.sh_link = strtab_shndx;
        symtab_sec.sh_info = first_global_idx;

        for (const auto& s : all_symbols) {
            write_u32(symtab_sec.data, s.st_name);
            write_u8(symtab_sec.data, s.st_info);
            write_u8(symtab_sec.data, s.st_other);
            write_u16(symtab_sec.data, s.st_shndx);
            write_u64(symtab_sec.data, s.st_value);
            write_u64(symtab_sec.data, s.st_size);
        }
        symtab_sec.sh_size = symtab_sec.data.size();
        elf_sections.push_back(std::move(symtab_sec));
    }

    // Set sh_link for all .rela.* sections to symtab_shndx
    for (auto& s : elf_sections) {
        if (s.sh_type == elf::SHT_RELA) {
            s.sh_link = symtab_shndx;
        }
    }

    // 5. .strtab section
    {
        ElfShdrEntry strtab_sec;
        strtab_sec.name = ".strtab";
        strtab_sec.sh_type = elf::SHT_STRTAB;
        strtab_sec.sh_flags = 0;
        strtab_sec.sh_addralign = 1;
        strtab_sec.data = std::move(strtab);
        strtab_sec.sh_size = strtab_sec.data.size();
        elf_sections.push_back(std::move(strtab_sec));
    }

    // 6. .shstrtab section
    std::vector<uint8_t> shstrtab;
    shstrtab.push_back(0); // 0th byte is null

    auto add_shstr = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(shstrtab.size());
        shstrtab.insert(shstrtab.end(), s.begin(), s.end());
        shstrtab.push_back(0);
        return off;
    };

    // Calculate section header name offsets
    for (auto& s : elf_sections) {
        s.sh_name = add_shstr(s.name);
    }
    uint32_t shstrtab_name_off = add_shstr(".shstrtab");

    {
        ElfShdrEntry shstrtab_sec;
        shstrtab_sec.name = ".shstrtab";
        shstrtab_sec.sh_name = shstrtab_name_off;
        shstrtab_sec.sh_type = elf::SHT_STRTAB;
        shstrtab_sec.sh_flags = 0;
        shstrtab_sec.sh_addralign = 1;
        shstrtab_sec.data = std::move(shstrtab);
        shstrtab_sec.sh_size = shstrtab_sec.data.size();
        elf_sections.push_back(std::move(shstrtab_sec));
    }

    // Layout binary file
    uint64_t cur_offset = 64; // sizeof(Elf64_Ehdr)
    for (auto& s : elf_sections) {
        if (s.sh_type == elf::SHT_NULL || s.data.empty()) {
            s.sh_offset = 0;
            continue;
        }
        if (s.sh_addralign > 1) {
            cur_offset = (cur_offset + (s.sh_addralign - 1)) & ~(s.sh_addralign - 1);
        }
        s.sh_offset = cur_offset;
        cur_offset += s.sh_size;
    }

    // Align section header table to 8 bytes
    cur_offset = (cur_offset + 7) & ~7ULL;
    uint64_t e_shoff = cur_offset;
    uint16_t e_shnum = static_cast<uint16_t>(elf_sections.size());

    // Build binary output
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(e_shoff + e_shnum * 64));

    // Write Elf64_Ehdr (64 bytes)
    // e_ident (16 bytes)
    out.push_back(0x7F);
    out.push_back('E');
    out.push_back('L');
    out.push_back('F');
    out.push_back(elf::ELFCLASS64);
    out.push_back(elf::ELFDATA2LSB);
    out.push_back(elf::EV_CURRENT);
    out.push_back(elf::ELFOSABI_SYSV);
    out.resize(16, 0);

    write_u16(out, elf::ET_REL);
    write_u16(out, elf::EM_X86_64);
    write_u32(out, elf::EV_CURRENT);
    write_u64(out, 0); // e_entry
    write_u64(out, 0); // e_phoff
    write_u64(out, e_shoff);
    write_u32(out, 0); // e_flags
    write_u16(out, 64); // e_ehsize
    write_u16(out, 0);  // e_phentsize
    write_u16(out, 0);  // e_phnum
    write_u16(out, 64); // e_shentsize
    write_u16(out, e_shnum);
    write_u16(out, static_cast<uint16_t>(shstrtab_shndx));

    // Write Section Data
    for (const auto& s : elf_sections) {
        if (s.sh_type == elf::SHT_NULL || s.data.empty()) continue;
        align_buf(out, static_cast<size_t>(s.sh_addralign));
        write_bytes(out, s.data.data(), s.data.size());
    }

    // Write Section Header Table
    align_buf(out, 8);
    for (const auto& s : elf_sections) {
        write_u32(out, s.sh_name);
        write_u32(out, s.sh_type);
        write_u64(out, s.sh_flags);
        write_u64(out, s.sh_addr);
        write_u64(out, s.sh_offset);
        write_u64(out, s.sh_size);
        write_u32(out, s.sh_link);
        write_u32(out, s.sh_info);
        write_u64(out, s.sh_addralign);
        write_u64(out, s.sh_entsize);
    }

    return out;
}

bool ElfWriter::write_to_file(const std::string& path) {
    auto data = write();
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

std::vector<uint8_t> emit_elf_object(const ObjectFile& obj) {
    ElfWriter writer(obj);
    return writer.write();
}

} // namespace brass::object
