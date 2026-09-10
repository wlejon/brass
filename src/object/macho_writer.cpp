#include <brass/object/macho_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <bit>

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

void write_fixed_string(std::vector<uint8_t>& buf, std::string_view str, size_t fixed_len) {
    size_t copy_len = std::min(str.size(), fixed_len);
    buf.insert(buf.end(), str.begin(), str.begin() + copy_len);
    if (copy_len < fixed_len) {
        buf.resize(buf.size() + (fixed_len - copy_len), 0);
    }
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

uint32_t align_to_log2(uint32_t align) {
    if (align <= 1) return 0;
    return static_cast<uint32_t>(std::countr_zero(align));
}

std::string to_macho_symbol_name(const std::string& name) {
    if (name.empty()) return name;
    if (name[0] == '_') return name;
    return "_" + name;
}

struct MachOSectionEntry {
    std::string sectname;
    std::string segname;
    uint32_t align_pow2 = 4;
    uint32_t flags = macho::S_REGULAR;
    std::vector<uint8_t> data;
    std::vector<ObjectRelocation> relocations;
    uint64_t addr = 0;
    uint32_t offset = 0;
    uint32_t reloff = 0;
    uint32_t nreloc = 0;
};

struct MachOSymbolEntry {
    std::string original_name;
    std::string macho_name;
    uint32_t n_strx = 0;
    uint8_t n_type = 0;
    uint8_t n_sect = 0;
    uint16_t n_desc = 0;
    uint64_t n_value = 0;
    SymbolType type = SymbolType::Function;
    SymbolBinding binding = SymbolBinding::Global;
};

} // namespace

MachOWriter::MachOWriter(const ObjectFile& obj)
    : obj_(obj) {}

std::vector<uint8_t> MachOWriter::write() {
    ObjectFile working_obj = obj_;

    // Generate DWARF .eh_frame for unwinding
    if (!working_obj.functions.empty()) {
        working_obj.get_or_create_section(
            ".eh_frame",
            SectionKind::EhFrame,
            SectionFlags::Read | SectionFlags::Alloc,
            8
        );
        Section* eh_frame_sec = working_obj.get_section(".eh_frame");
        if (eh_frame_sec && eh_frame_sec->data.empty()) {
            ElfCfiBuilder::build_eh_frame(working_obj, *eh_frame_sec);
        }
    }

    // 1. Map ObjectFile sections to Mach-O sections
    std::vector<MachOSectionEntry> macho_sections;
    std::unordered_map<std::string, uint32_t> sec_name_to_idx;

    for (const auto& sec : working_obj.sections) {
        MachOSectionEntry entry;
        entry.align_pow2 = align_to_log2(sec.alignment);
        entry.data = sec.data;
        entry.relocations = sec.relocations;

        if (sec.name == ".text" || sec.kind == SectionKind::Text) {
            entry.sectname = "__text";
            entry.segname = "__TEXT";
            entry.flags = macho::S_REGULAR | macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS;
            if (entry.align_pow2 < 4) entry.align_pow2 = 4;
        } else if (sec.name == ".rodata" || sec.name == ".rdata" || sec.name == "__const" || sec.kind == SectionKind::RoData) {
            entry.sectname = "__const";
            entry.segname = "__TEXT";
            entry.flags = macho::S_REGULAR;
            if (entry.align_pow2 < 4) entry.align_pow2 = 4;
        } else if (sec.name == ".data" || sec.kind == SectionKind::Data) {
            entry.sectname = "__data";
            entry.segname = "__DATA";
            entry.flags = macho::S_REGULAR;
        } else if (sec.name == ".bss" || sec.kind == SectionKind::Bss) {
            entry.sectname = "__bss";
            entry.segname = "__DATA";
            entry.flags = macho::S_ZEROFILL;
        } else if (sec.name == ".eh_frame" || sec.kind == SectionKind::EhFrame) {
            entry.sectname = "__eh_frame";
            entry.segname = "__TEXT";
            entry.flags = macho::S_REGULAR;
            if (entry.align_pow2 < 3) entry.align_pow2 = 3;
        } else if (sec.name.rfind(".brass_dbg", 0) == 0 || sec.name.rfind(".debug", 0) == 0) {
            entry.sectname = "__" + sec.name.substr(1);
            entry.segname = "__DWARF";
            entry.flags = macho::S_REGULAR;
        } else if (size_t comma = sec.name.find(','); comma != std::string::npos) {
            entry.segname = sec.name.substr(0, comma);
            entry.sectname = sec.name.substr(comma + 1);
            entry.flags = macho::S_REGULAR;
        } else {
            entry.sectname = sec.name.substr(0, 16);
            entry.segname = has_flag(sec.flags, SectionFlags::Write) ? "__DATA" : "__TEXT";
            entry.flags = macho::S_REGULAR;
        }

        uint32_t idx = static_cast<uint32_t>(macho_sections.size());
        sec_name_to_idx[sec.name] = idx;
        macho_sections.push_back(std::move(entry));
    }

    // 2. Build Symbol Table & String Table
    std::vector<uint8_t> strtab;
    strtab.push_back(0); // 0th byte is null string

    auto add_str = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(strtab.size());
        strtab.insert(strtab.end(), s.begin(), s.end());
        strtab.push_back(0);
        return off;
    };

    std::vector<MachOSymbolEntry> local_syms;
    std::vector<MachOSymbolEntry> extdef_syms;
    std::vector<MachOSymbolEntry> undef_syms;

    for (const auto& sym : working_obj.symbols) {
        if (sym.type == SymbolType::Section) continue;

        MachOSymbolEntry msym;
        msym.original_name = sym.name;
        msym.macho_name = to_macho_symbol_name(sym.name);
        msym.type = sym.type;
        msym.binding = sym.binding;

        bool is_local = (sym.binding == SymbolBinding::Local);
        bool is_defined = (sym.section_index >= 0 && sym.section_index < static_cast<int32_t>(working_obj.sections.size()));

        if (is_defined) {
            const auto& sec_name = working_obj.sections[static_cast<size_t>(sym.section_index)].name;
            uint32_t macho_sec_idx = sec_name_to_idx[sec_name];
            msym.n_sect = static_cast<uint8_t>(macho_sec_idx + 1); // 1-based section number
            msym.n_value = sym.value;
            if (is_local) {
                msym.n_type = macho::N_SECT;
                local_syms.push_back(std::move(msym));
            } else {
                msym.n_type = macho::N_SECT | macho::N_EXT;
                extdef_syms.push_back(std::move(msym));
            }
        } else {
            msym.n_sect = macho::NO_SECT;
            msym.n_value = 0;
            msym.n_type = macho::N_UNDF | macho::N_EXT;
            undef_syms.push_back(std::move(msym));
        }
    }

    // Ensure all symbols referenced by relocations are present
    auto find_or_add_reloc_sym = [&](const std::string& name) {
        for (const auto& s : local_syms) {
            if (s.original_name == name || s.macho_name == name) return;
        }
        for (const auto& s : extdef_syms) {
            if (s.original_name == name || s.macho_name == name) return;
        }
        for (const auto& s : undef_syms) {
            if (s.original_name == name || s.macho_name == name) return;
        }
        MachOSymbolEntry msym;
        msym.original_name = name;
        msym.macho_name = to_macho_symbol_name(name);
        msym.n_sect = macho::NO_SECT;
        msym.n_value = 0;
        msym.n_type = macho::N_UNDF | macho::N_EXT;
        msym.binding = SymbolBinding::Global;
        msym.type = SymbolType::Function;
        undef_syms.push_back(std::move(msym));
    };

    for (const auto& sec : macho_sections) {
        for (const auto& r : sec.relocations) {
            if (!r.symbol_name.empty()) {
                find_or_add_reloc_sym(r.symbol_name);
            }
        }
    }

    // Combine symbol partitions: locals, then external definitions, then undefined
    std::vector<MachOSymbolEntry> all_symbols;
    all_symbols.reserve(local_syms.size() + extdef_syms.size() + undef_syms.size());

    uint32_t ilocalsym = 0;
    uint32_t nlocalsym = static_cast<uint32_t>(local_syms.size());
    uint32_t iextdefsym = nlocalsym;
    uint32_t nextdefsym = static_cast<uint32_t>(extdef_syms.size());
    uint32_t iundefsym = iextdefsym + nextdefsym;
    uint32_t nundefsym = static_cast<uint32_t>(undef_syms.size());

    all_symbols.insert(all_symbols.end(), local_syms.begin(), local_syms.end());
    all_symbols.insert(all_symbols.end(), extdef_syms.begin(), extdef_syms.end());
    all_symbols.insert(all_symbols.end(), undef_syms.begin(), undef_syms.end());

    std::unordered_map<std::string, uint32_t> sym_name_to_idx;
    for (size_t i = 0; i < all_symbols.size(); ++i) {
        all_symbols[i].n_strx = add_str(all_symbols[i].macho_name);
        sym_name_to_idx[all_symbols[i].original_name] = static_cast<uint32_t>(i);
        sym_name_to_idx[all_symbols[i].macho_name] = static_cast<uint32_t>(i);
    }

    // 3. Compute Section Data Offsets and Relocations
    uint32_t nsects = static_cast<uint32_t>(macho_sections.size());
    uint32_t segment_cmd_size = 72 + nsects * 80;
    uint32_t symtab_cmd_size = 24;
    uint32_t dysymtab_cmd_size = 80;
    uint32_t sizeofcmds = segment_cmd_size + symtab_cmd_size + dysymtab_cmd_size;

    uint32_t cur_file_offset = 32 + sizeofcmds; // mach_header_64 (32 bytes) + cmds
    cur_file_offset = (cur_file_offset + 15) & ~15u; // 16-byte align section data

    uint64_t cur_vmaddr = 0;
    for (auto& s : macho_sections) {
        uint32_t sec_align = 1u << s.align_pow2;
        if (sec_align > 1) {
            cur_file_offset = (cur_file_offset + (sec_align - 1)) & ~(sec_align - 1);
            cur_vmaddr = (cur_vmaddr + (sec_align - 1)) & ~(sec_align - 1);
        }
        s.offset = cur_file_offset;
        s.addr = cur_vmaddr;
        cur_file_offset += static_cast<uint32_t>(s.data.size());
        cur_vmaddr += s.data.size();
    }

    // Relocations offset
    for (auto& s : macho_sections) {
        if (!s.relocations.empty()) {
            cur_file_offset = (cur_file_offset + 7) & ~7u;
            s.reloff = cur_file_offset;
            s.nreloc = static_cast<uint32_t>(s.relocations.size());
            cur_file_offset += s.nreloc * 8; // 8 bytes per relocation_info
        } else {
            s.reloff = 0;
            s.nreloc = 0;
        }
    }

    // Symbol Table offset
    cur_file_offset = (cur_file_offset + 7) & ~7u;
    uint32_t symoff = cur_file_offset;
    uint32_t nsyms = static_cast<uint32_t>(all_symbols.size());
    cur_file_offset += nsyms * 16; // 16 bytes per nlist_64

    // String Table offset
    uint32_t stroff = cur_file_offset;
    uint32_t strsize = static_cast<uint32_t>(strtab.size());
    cur_file_offset += strsize;

    // 4. Build Output Binary
    std::vector<uint8_t> out;
    out.reserve(cur_file_offset);

    // Write mach_header_64 (32 bytes)
    write_u32(out, macho::MH_MAGIC_64);
    write_u32(out, static_cast<uint32_t>(macho::CPU_TYPE_X86_64));
    write_u32(out, static_cast<uint32_t>(macho::CPU_SUBTYPE_X86_64_ALL));
    write_u32(out, macho::MH_OBJECT);
    write_u32(out, 3); // ncmds
    write_u32(out, sizeofcmds);
    write_u32(out, macho::MH_SUBSECTIONS_VIA_SYMBOLS); // flags
    write_u32(out, 0); // reserved

    // Write LC_SEGMENT_64
    uint64_t seg_vmsize = cur_vmaddr;
    uint64_t seg_fileoff = macho_sections.empty() ? 0 : macho_sections[0].offset;
    uint64_t seg_filesize = macho_sections.empty() ? 0 : (cur_file_offset - seg_fileoff);

    write_u32(out, macho::LC_SEGMENT_64);
    write_u32(out, segment_cmd_size);
    write_fixed_string(out, "", 16); // Empty segname for MH_OBJECT
    write_u64(out, 0); // vmaddr
    write_u64(out, seg_vmsize);
    write_u64(out, seg_fileoff);
    write_u64(out, seg_filesize);
    write_u32(out, macho::VM_PROT_READ | macho::VM_PROT_WRITE | macho::VM_PROT_EXECUTE); // maxprot
    write_u32(out, macho::VM_PROT_READ | macho::VM_PROT_WRITE | macho::VM_PROT_EXECUTE); // initprot
    write_u32(out, nsects);
    write_u32(out, 0); // flags

    // Write section_64 entries (80 bytes each)
    for (const auto& s : macho_sections) {
        write_fixed_string(out, s.sectname, 16);
        write_fixed_string(out, s.segname, 16);
        write_u64(out, s.addr);
        write_u64(out, s.data.size());
        write_u32(out, s.offset);
        write_u32(out, s.align_pow2);
        write_u32(out, s.reloff);
        write_u32(out, s.nreloc);
        write_u32(out, s.flags);
        write_u32(out, 0); // reserved1
        write_u32(out, 0); // reserved2
        write_u32(out, 0); // reserved3
    }

    // Write LC_SYMTAB (24 bytes)
    write_u32(out, macho::LC_SYMTAB);
    write_u32(out, symtab_cmd_size);
    write_u32(out, symoff);
    write_u32(out, nsyms);
    write_u32(out, stroff);
    write_u32(out, strsize);

    // Write LC_DYSYMTAB (80 bytes)
    write_u32(out, macho::LC_DYSYMTAB);
    write_u32(out, dysymtab_cmd_size);
    write_u32(out, ilocalsym);
    write_u32(out, nlocalsym);
    write_u32(out, iextdefsym);
    write_u32(out, nextdefsym);
    write_u32(out, iundefsym);
    write_u32(out, nundefsym);
    // TOC, module table, ext/indirect/loc reloc offsets (all 0 for MH_OBJECT)
    for (int i = 0; i < 12; ++i) {
        write_u32(out, 0);
    }

    // Write Section Data
    for (const auto& s : macho_sections) {
        align_buf(out, 1u << s.align_pow2);
        if (!s.data.empty()) {
            write_bytes(out, s.data.data(), s.data.size());
        }
    }

    // Write Relocations
    for (const auto& s : macho_sections) {
        if (s.relocations.empty()) continue;
        align_buf(out, 8);
        for (const auto& r : s.relocations) {
            uint32_t sym_idx = 0;
            auto it = sym_name_to_idx.find(r.symbol_name);
            if (it != sym_name_to_idx.end()) {
                sym_idx = it->second;
            }

            int32_t r_address = static_cast<int32_t>(r.offset);
            uint32_t r_pcrel = 0;
            uint32_t r_length = 2; // 4 bytes by default
            uint32_t r_extern = 1;
            uint32_t r_type = macho::X86_64_RELOC_BRANCH;

            bool is_func = (sym_idx < all_symbols.size() && all_symbols[sym_idx].type == SymbolType::Function);

            if (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::Plt32) {
                r_pcrel = 1;
                r_length = 2;
                r_type = is_func ? macho::X86_64_RELOC_BRANCH : macho::X86_64_RELOC_SIGNED;
            } else if (r.kind == RelocKind::SecRel32) {
                r_pcrel = 1;
                r_length = 2;
                r_type = macho::X86_64_RELOC_SIGNED;
            } else if (r.kind == RelocKind::Abs64) {
                r_pcrel = 0;
                r_length = 3; // 8 bytes
                r_type = macho::X86_64_RELOC_UNSIGNED;
            } else if (r.kind == RelocKind::Abs32 || r.kind == RelocKind::Addr32NB) {
                r_pcrel = 0;
                r_length = 2;
                r_type = macho::X86_64_RELOC_UNSIGNED;
            }

            uint32_t word2 = (sym_idx & 0x00FFFFFF)
                           | ((r_pcrel & 0x1) << 24)
                           | ((r_length & 0x3) << 25)
                           | ((r_extern & 0x1) << 27)
                           | ((r_type & 0xF) << 28);

            write_u32(out, static_cast<uint32_t>(r_address));
            write_u32(out, word2);
        }
    }

    // Write Symbol Table (nlist_64 entries, 16 bytes each)
    align_buf(out, 8);
    for (const auto& s : all_symbols) {
        write_u32(out, s.n_strx);
        write_u8(out, s.n_type);
        write_u8(out, s.n_sect);
        write_u16(out, s.n_desc);
        write_u64(out, s.n_value);
    }

    // Write String Table
    write_bytes(out, strtab.data(), strtab.size());

    return out;
}

bool MachOWriter::write_to_file(const std::string& path) {
    auto data = write();
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

std::vector<uint8_t> emit_macho_object(const ObjectFile& obj) {
    MachOWriter writer(obj);
    return writer.write();
}

} // namespace brass::object
