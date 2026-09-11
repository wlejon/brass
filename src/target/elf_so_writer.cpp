#include <brass/target/elf_so_writer.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace brass::target {

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
    const auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

uint64_t align_up_u64(uint64_t val, uint64_t align) {
    if (align <= 1) return val;
    uint64_t rem = val % align;
    return rem == 0 ? val : val + (align - rem);
}

uint32_t elf_hash(const char* name) {
    uint32_t h = 0;
    uint32_t g = 0;
    while (*name) {
        h = (h << 4) + static_cast<uint8_t>(*name++);
        g = h & 0xF0000000u;
        if (g) h ^= (g >> 24);
        h &= ~g;
    }
    return h;
}

struct ElfShdr {
    std::string name;
    uint32_t sh_name = 0;
    uint32_t sh_type = elf64::SHT_NULL;
    uint64_t sh_flags = 0;
    uint64_t sh_addr = 0;
    uint64_t sh_offset = 0;
    uint64_t sh_size = 0;
    uint32_t sh_link = 0;
    uint32_t sh_info = 0;
    uint64_t sh_addralign = 1;
    uint64_t sh_entsize = 0;
    std::vector<uint8_t> data;
    std::vector<object::ObjectRelocation> relocations;
};

struct DynSymEntry {
    std::string name;
    uint32_t st_name = 0;
    uint8_t st_info = 0;
    uint8_t st_other = 0;
    uint16_t st_shndx = 0;
    uint64_t st_value = 0;
    uint64_t st_size = 0;
};

} // namespace

ElfSoWriter::ElfSoWriter(const object::ObjectFile& obj, const ElfSoOptions& options)
    : obj_(obj), options_(options) {}

ElfSoWriter::ElfSoWriter(const object::ObjectFile& obj)
    : obj_(obj), options_(ElfSoOptions()) {}

std::vector<uint8_t> ElfSoWriter::write() {
    object::ObjectFile working_obj = obj_;

    // 1. Identify symbols to export
    std::vector<std::string> export_names;
    if (options_.export_all_functions) {
        for (const auto& sym : working_obj.symbols) {
            if (sym.type == object::SymbolType::Function &&
                sym.binding == object::SymbolBinding::Global &&
                sym.section_index != object::SECTION_UNDEF) {
                export_names.push_back(sym.name);
            }
        }
    }
    for (const auto& exp : options_.explicit_exports) {
        if (std::find(export_names.begin(), export_names.end(), exp) == export_names.end()) {
            export_names.push_back(exp);
        }
    }

    // 2. Build .dynstr and list of dynamic symbols
    std::vector<uint8_t> dynstr_data;
    dynstr_data.push_back(0); // Index 0 is empty string

    auto add_dynstr = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(dynstr_data.size());
        write_bytes(dynstr_data, s.c_str(), s.size() + 1);
        return off;
    };

    uint32_t soname_offset = 0;
    if (!options_.soname.empty()) {
        soname_offset = add_dynstr(options_.soname);
    }

    std::vector<DynSymEntry> dynsyms;
    // Index 0: NULL symbol
    dynsyms.push_back({});

    for (const auto& name : export_names) {
        const auto* sym = working_obj.find_symbol(name);
        DynSymEntry entry;
        entry.name = name;
        entry.st_name = add_dynstr(name);
        uint8_t bind = elf64::STB_GLOBAL;
        uint8_t type = (sym && sym->type == object::SymbolType::Object) ? elf64::STT_OBJECT : elf64::STT_FUNC;
        entry.st_info = static_cast<uint8_t>((bind << 4) | (type & 0xF));
        entry.st_other = 0; // Default visibility
        entry.st_size = sym ? sym->size : 0;
        entry.st_value = sym ? sym->value : 0;
        dynsyms.push_back(entry);
    }

    // 3. Build .hash table (ELF DT_HASH)
    uint32_t num_dynsym = static_cast<uint32_t>(dynsyms.size());
    uint32_t nbucket = std::max(1u, num_dynsym | 1u);
    uint32_t nchain = num_dynsym;

    std::vector<uint32_t> bucket(nbucket, 0);
    std::vector<uint32_t> chain(nchain, 0);

    for (uint32_t i = 1; i < num_dynsym; ++i) {
        uint32_t h = elf_hash(dynsyms[i].name.c_str()) % nbucket;
        chain[i] = bucket[h];
        bucket[h] = i;
    }

    std::vector<uint8_t> hash_data;
    write_u32(hash_data, nbucket);
    write_u32(hash_data, nchain);
    for (uint32_t b : bucket) write_u32(hash_data, b);
    for (uint32_t c : chain) write_u32(hash_data, c);

    // 4. Create Section List
    // Sections:
    // [0] NULL
    // [1] .hash
    // [2] .dynsym
    // [3] .dynstr
    // [4] .text
    // [5] .rodata (if any)
    // [6] .dynamic
    // [7] .rela.dyn (if any)
    // [8] .data (if any)
    // [9] .shstrtab
    std::vector<ElfShdr> sections;
    sections.push_back({}); // NULL section

    // .hash
    {
        ElfShdr s;
        s.name = ".hash";
        s.sh_type = elf64::SHT_HASH;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 8;
        s.sh_entsize = 4;
        s.data = std::move(hash_data);
        sections.push_back(std::move(s));
    }

    // .dynsym
    {
        ElfShdr s;
        s.name = ".dynsym";
        s.sh_type = elf64::SHT_DYNSYM;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 8;
        s.sh_entsize = 24;
        s.data.resize(dynsyms.size() * 24, 0);
        s.sh_info = 1; // 1 local symbol (index 0)
        sections.push_back(std::move(s));
    }

    // .dynstr
    {
        ElfShdr s;
        s.name = ".dynstr";
        s.sh_type = elf64::SHT_STRTAB;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 1;
        s.data = std::move(dynstr_data);
        sections.push_back(std::move(s));
    }

    // .text
    uint32_t text_sec_idx = 0;
    {
        ElfShdr s;
        s.name = ".text";
        s.sh_type = elf64::SHT_PROGBITS;
        s.sh_flags = elf64::SHF_ALLOC | elf64::SHF_EXECINSTR;
        s.sh_addralign = 16;
        if (const auto* ts = working_obj.get_section(".text")) {
            s.data = ts->data;
            s.relocations = ts->relocations;
        }
        text_sec_idx = static_cast<uint32_t>(sections.size());
        sections.push_back(std::move(s));
    }

    // .rodata
    if (const auto* rs = working_obj.get_section(".rodata")) {
        ElfShdr s;
        s.name = ".rodata";
        s.sh_type = elf64::SHT_PROGBITS;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 16;
        s.data = rs->data;
        s.relocations = rs->relocations;
        sections.push_back(std::move(s));
    } else if (const auto* rds = working_obj.get_section(".rdata")) {
        ElfShdr s;
        s.name = ".rodata";
        s.sh_type = elf64::SHT_PROGBITS;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 16;
        s.data = rds->data;
        s.relocations = rds->relocations;
        sections.push_back(std::move(s));
    }

    // .dynamic placeholder
    uint32_t dynamic_sec_idx = static_cast<uint32_t>(sections.size());
    {
        ElfShdr s;
        s.name = ".dynamic";
        s.sh_type = elf64::SHT_DYNAMIC;
        s.sh_flags = elf64::SHF_ALLOC | elf64::SHF_WRITE;
        s.sh_addralign = 8;
        s.sh_entsize = 16;
        // Allocate space for up to 12 dynamic entries (16 bytes each)
        s.data.resize(12 * 16, 0);
        sections.push_back(std::move(s));
    }

    // .rela.dyn placeholder
    uint32_t rela_sec_idx = static_cast<uint32_t>(sections.size());
    {
        ElfShdr s;
        s.name = ".rela.dyn";
        s.sh_type = elf64::SHT_RELA;
        s.sh_flags = elf64::SHF_ALLOC;
        s.sh_addralign = 8;
        s.sh_entsize = 24;
        sections.push_back(std::move(s));
    }

    // .data
    if (const auto* ds = working_obj.get_section(".data")) {
        ElfShdr s;
        s.name = ".data";
        s.sh_type = elf64::SHT_PROGBITS;
        s.sh_flags = elf64::SHF_ALLOC | elf64::SHF_WRITE;
        s.sh_addralign = 8;
        s.data = ds->data;
        s.relocations = ds->relocations;
        sections.push_back(std::move(s));
    }

    // .shstrtab placeholder
    uint32_t shstrtab_sec_idx = static_cast<uint32_t>(sections.size());
    {
        ElfShdr s;
        s.name = ".shstrtab";
        s.sh_type = elf64::SHT_STRTAB;
        s.sh_flags = 0;
        s.sh_addralign = 1;
        sections.push_back(std::move(s));
    }

    // Update section links
    sections[1].sh_link = 2; // .hash -> .dynsym
    sections[2].sh_link = 3; // .dynsym -> .dynstr
    sections[dynamic_sec_idx].sh_link = 3; // .dynamic -> .dynstr
    sections[rela_sec_idx].sh_link = 2; // .rela.dyn -> .dynsym

    // 5. Layout Segments and Compute Offsets / VAddrs
    constexpr uint64_t PAGE_SIZE = 0x1000;
    constexpr uint64_t PHDR_COUNT = 6; // PT_PHDR, PT_LOAD(R), PT_LOAD(RX), PT_LOAD(RW), PT_DYNAMIC, PT_GNU_STACK

    uint64_t ehdr_size = 64;
    uint64_t phdrs_size = PHDR_COUNT * 56;
    uint64_t headers_end = ehdr_size + phdrs_size;

    // Segment 1 (PT_LOAD R): Headers, .hash, .dynsym, .dynstr, .rodata
    // Segment 2 (PT_LOAD RX): .text
    // Segment 3 (PT_LOAD RW): .dynamic, .rela.dyn, .data
    uint64_t cur_offset = headers_end;
    uint64_t cur_vaddr = headers_end;

    auto align_section = [&](ElfShdr& s) {
        cur_offset = align_up_u64(cur_offset, s.sh_addralign);
        cur_vaddr = align_up_u64(cur_vaddr, s.sh_addralign);
        s.sh_offset = cur_offset;
        s.sh_addr = cur_vaddr;
        s.sh_size = s.data.size();
        cur_offset += s.sh_size;
        cur_vaddr += s.sh_size;
    };

    // Seg 1 sections: .hash, .dynsym, .dynstr
    align_section(sections[1]);
    align_section(sections[2]);
    align_section(sections[3]);

    // Check if .rodata exists
    for (size_t i = 4; i < sections.size(); ++i) {
        if (sections[i].name == ".rodata") {
            align_section(sections[i]);
            break;
        }
    }

    uint64_t seg1_filesz = cur_offset;
    uint64_t seg1_memsz = cur_vaddr;

    // Seg 2 (RX): align to next page
    cur_offset = align_up_u64(cur_offset, PAGE_SIZE);
    cur_vaddr = cur_offset; // Keep vaddr == offset for shared libraries
    uint64_t seg2_vaddr = cur_vaddr;
    uint64_t seg2_offset = cur_offset;

    align_section(sections[text_sec_idx]);

    uint64_t seg2_filesz = cur_offset - seg2_offset;
    uint64_t seg2_memsz = cur_vaddr - seg2_vaddr;

    // Seg 3 (RW): align to next page
    cur_offset = align_up_u64(cur_offset, PAGE_SIZE);
    cur_vaddr = cur_offset;
    uint64_t seg3_vaddr = cur_vaddr;
    uint64_t seg3_offset = cur_offset;

    align_section(sections[dynamic_sec_idx]);
    align_section(sections[rela_sec_idx]);
    for (size_t i = 4; i < sections.size(); ++i) {
        if (sections[i].name == ".data") {
            align_section(sections[i]);
            break;
        }
    }

    uint64_t seg3_filesz = cur_offset - seg3_offset;
    uint64_t seg3_memsz = cur_vaddr - seg3_vaddr;

    // 6. Update DynSym entries with computed section addresses
    for (size_t i = 1; i < dynsyms.size(); ++i) {
        auto& ds = dynsyms[i];
        const auto* sym = working_obj.find_symbol(ds.name);
        if (sym && sym->section_index >= 0) {
            const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
            for (size_t s_idx = 1; s_idx < sections.size(); ++s_idx) {
                if (sections[s_idx].name == src_sec.name || (src_sec.name == ".text" && sections[s_idx].name == ".text")) {
                    ds.st_shndx = static_cast<uint16_t>(s_idx);
                    ds.st_value = sections[s_idx].sh_addr + sym->value;
                    break;
                }
            }
        }
    }

    // Write updated dynsym table to .dynsym section data
    std::vector<uint8_t>& dynsym_buf = sections[2].data;
    dynsym_buf.clear();
    for (const auto& ds : dynsyms) {
        write_u32(dynsym_buf, ds.st_name);
        write_u8(dynsym_buf, ds.st_info);
        write_u8(dynsym_buf, ds.st_other);
        write_u16(dynsym_buf, ds.st_shndx);
        write_u64(dynsym_buf, ds.st_value);
        write_u64(dynsym_buf, ds.st_size);
    }
    sections[2].sh_size = dynsym_buf.size();

    // 7. Resolve Relocations
    std::vector<uint8_t> rela_data;
    for (auto& sec : sections) {
        for (const auto& r : sec.relocations) {
            uint64_t target_vaddr = 0;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                for (const auto& s : sections) {
                    if (s.name == src_sec.name) {
                        target_vaddr = s.sh_addr + sym->value;
                        break;
                    }
                }
            }

            uint64_t reloc_vaddr = sec.sh_addr + r.offset;

            if (r.kind == object::RelocKind::PCRel32 || r.kind == object::RelocKind::Plt32) {
                int64_t disp = static_cast<int64_t>(target_vaddr + static_cast<uint64_t>(r.addend)) - static_cast<int64_t>(reloc_vaddr + 4);
                int32_t disp32 = static_cast<int32_t>(disp);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &disp32, 4);
                }
            } else if (r.kind == object::RelocKind::Abs64) {
                // Emit R_X86_64_RELATIVE relocation into .rela.dyn
                // r_offset (8), r_info (8), r_addend (8)
                write_u64(rela_data, reloc_vaddr);
                uint64_t r_info = elf64::R_X86_64_RELATIVE;
                write_u64(rela_data, r_info);
                write_i64(rela_data, static_cast<int64_t>(target_vaddr + static_cast<uint64_t>(r.addend)));
            }
        }
    }

    sections[rela_sec_idx].data = std::move(rela_data);
    sections[rela_sec_idx].sh_size = sections[rela_sec_idx].data.size();

    // 8. Populate .dynamic section
    std::vector<uint8_t>& dyn_buf = sections[dynamic_sec_idx].data;
    dyn_buf.clear();

    auto add_dyn = [&](int64_t tag, uint64_t val) {
        write_i64(dyn_buf, tag);
        write_u64(dyn_buf, val);
    };

    add_dyn(elf64::DT_HASH, sections[1].sh_addr);
    add_dyn(elf64::DT_STRTAB, sections[3].sh_addr);
    add_dyn(elf64::DT_SYMTAB, sections[2].sh_addr);
    add_dyn(elf64::DT_STRSZ, sections[3].sh_size);
    add_dyn(elf64::DT_SYMENT, 24);
    if (soname_offset > 0) {
        add_dyn(elf64::DT_SONAME, soname_offset);
    }
    if (sections[rela_sec_idx].sh_size > 0) {
        add_dyn(elf64::DT_RELA, sections[rela_sec_idx].sh_addr);
        add_dyn(elf64::DT_RELASZ, sections[rela_sec_idx].sh_size);
        add_dyn(elf64::DT_RELAENT, 24);
    }
    add_dyn(elf64::DT_NULL, 0);

    sections[dynamic_sec_idx].sh_size = dyn_buf.size();

    // 9. Build .shstrtab section names
    std::vector<uint8_t>& shstrtab = sections[shstrtab_sec_idx].data;
    shstrtab.clear();
    shstrtab.push_back(0); // Index 0 is empty string

    for (auto& s : sections) {
        if (s.name.empty()) {
            s.sh_name = 0;
        } else {
            s.sh_name = static_cast<uint32_t>(shstrtab.size());
            write_bytes(shstrtab, s.name.c_str(), s.name.size() + 1);
        }
    }
    sections[shstrtab_sec_idx].sh_size = shstrtab.size();

    // Place .shstrtab at the end of file (non-alloc)
    sections[shstrtab_sec_idx].sh_offset = cur_offset;
    sections[shstrtab_sec_idx].sh_addr = 0;
    cur_offset += sections[shstrtab_sec_idx].sh_size;

    // Section header table offset
    cur_offset = align_up_u64(cur_offset, 8);
    uint64_t shoff = cur_offset;

    // 10. Assemble ELF64 binary buffer
    std::vector<uint8_t> out;
    out.reserve(shoff + sections.size() * 64);

    // ELF64 Header (64 bytes)
    write_u8(out, 0x7F); write_u8(out, 'E'); write_u8(out, 'L'); write_u8(out, 'F');
    write_u8(out, elf64::ELFCLASS64);
    write_u8(out, elf64::ELFDATA2LSB);
    write_u8(out, elf64::EV_CURRENT);
    write_u8(out, elf64::ELFOSABI_SYSV);
    for (int i = 0; i < 8; ++i) write_u8(out, 0); // e_ident padding

    write_u16(out, elf64::ET_DYN); // e_type
    write_u16(out, elf64::EM_X86_64); // e_machine
    write_u32(out, elf64::EV_CURRENT); // e_version
    write_u64(out, 0); // e_entry
    write_u64(out, 64); // e_phoff (immediately after Ehdr)
    write_u64(out, shoff); // e_shoff
    write_u32(out, 0); // e_flags
    write_u16(out, 64); // e_ehsize
    write_u16(out, 56); // e_phentsize
    write_u16(out, static_cast<uint16_t>(PHDR_COUNT)); // e_phnum
    write_u16(out, 64); // e_shentsize
    write_u16(out, static_cast<uint16_t>(sections.size())); // e_shnum
    write_u16(out, static_cast<uint16_t>(shstrtab_sec_idx)); // e_shstrndx

    // Program Header Table (5 phdrs, 56 bytes each)
    auto write_phdr = [&](uint32_t type, uint32_t flags, uint64_t off, uint64_t vaddr, uint64_t filesz, uint64_t memsz, uint64_t align) {
        write_u32(out, type);
        write_u32(out, flags);
        write_u64(out, off);
        write_u64(out, vaddr);
        write_u64(out, vaddr); // paddr
        write_u64(out, filesz);
        write_u64(out, memsz);
        write_u64(out, align);
    };

    // 0: PT_PHDR
    write_phdr(elf64::PT_PHDR, elf64::PF_R, 64, 64, phdrs_size, phdrs_size, 8);
    // 1: PT_LOAD (R): Seg 1
    write_phdr(elf64::PT_LOAD, elf64::PF_R, 0, 0, seg1_filesz, seg1_memsz, PAGE_SIZE);
    // 2: PT_LOAD (RX): Seg 2 (.text)
    write_phdr(elf64::PT_LOAD, elf64::PF_R | elf64::PF_X, seg2_offset, seg2_vaddr, seg2_filesz, seg2_memsz, PAGE_SIZE);
    // 3: PT_LOAD (RW): Seg 3 (.dynamic, .rela.dyn, .data)
    write_phdr(elf64::PT_LOAD, elf64::PF_R | elf64::PF_W, seg3_offset, seg3_vaddr, seg3_filesz, seg3_memsz, PAGE_SIZE);
    // 4: PT_DYNAMIC
    uint64_t dyn_off = sections[dynamic_sec_idx].sh_offset;
    uint64_t dyn_vaddr = sections[dynamic_sec_idx].sh_addr;
    uint64_t dyn_sz = sections[dynamic_sec_idx].sh_size;
    write_phdr(elf64::PT_DYNAMIC, elf64::PF_R | elf64::PF_W, dyn_off, dyn_vaddr, dyn_sz, dyn_sz, 8);
    // 5: PT_GNU_STACK
    write_phdr(elf64::PT_GNU_STACK, elf64::PF_R | elf64::PF_W, 0, 0, 0, 0, 8);

    // Section Data
    for (size_t i = 1; i < sections.size(); ++i) {
        const auto& s = sections[i];
        if (out.size() < s.sh_offset) {
            out.resize(s.sh_offset, 0);
        }
        write_bytes(out, s.data.data(), s.data.size());
    }

    // Section Header Table (64 bytes per section)
    if (out.size() < shoff) {
        out.resize(shoff, 0);
    }

    for (const auto& s : sections) {
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

bool ElfSoWriter::write_to_file(const std::string& path) {
    auto data = write();
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

std::vector<uint8_t> ElfSoWriter::emit(const object::ObjectFile& obj, const ElfSoOptions& options) {
    ElfSoWriter writer(obj, options);
    return writer.write();
}

std::vector<uint8_t> ElfSoWriter::emit(const object::ObjectFile& obj) {
    return emit(obj, ElfSoOptions());
}

} // namespace brass::target
