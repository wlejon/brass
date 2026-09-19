#include <brass/target/elf_so_writer.hpp>
#include "image_file.hpp"
#include "image_util.hpp"
#include "import_plan.hpp"
#include <algorithm>
#include <cstring>

// An ELF64 shared object from one object file, with no lazy binding.
//
// Layout, offsets equal to virtual addresses throughout:
//
//   PT_LOAD R    headers, .hash, .dynsym, .dynstr, .rela.dyn
//   PT_LOAD RX   .text, .plt
//   PT_LOAD RW   .rodata, .got, .dynamic  | page |  .data
//                `--------- PT_GNU_RELRO ---------'
//
// .rodata is in the RELRO part of the writable segment rather than in the
// read-only one because it carries absolute pointers (function descriptors,
// tables of code addresses) that the dynamic linker has to relocate: a
// RELATIVE relocation into a read-only page is a fault at load. The loader
// makes the region read-only again once it has applied them, which is what
// a system linker's .data.rel.ro is.
//
// Imports take one GOT slot each, bound at load time through a GLOB_DAT
// relocation, and one .plt stub that jumps through the slot. There is no
// PLT0, no JUMP_SLOT and no lazy resolver: every reference is resolved before
// the first instruction of the module runs, which is also what a data
// reference (an R_X86_64_64 against the symbol) requires anyway.

namespace brass::target {

using namespace brass::target::image;

namespace {

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
    // The object section this one was copied from, for symbol lookup.
    std::string source;
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

constexpr uint64_t PAGE_SIZE = 0x1000;
constexpr uint64_t PHDR_COUNT = 7; // PHDR, LOAD R, LOAD RX, LOAD RW, DYNAMIC, GNU_RELRO, GNU_STACK
constexpr uint64_t PLT_STUB_SIZE = 16;

} // namespace

ElfSoWriter::ElfSoWriter(const object::ObjectFile& obj, const ElfSoOptions& options)
    : obj_(obj), options_(options) {}

ElfSoWriter::ElfSoWriter(const object::ObjectFile& obj)
    : obj_(obj), options_(ElfSoOptions()) {}

std::vector<uint8_t> ElfSoWriter::write() {
    error_.clear();
    object::ObjectFile working_obj = obj_;
    const bool aarch64 = working_obj.target.is_aarch64();

    // 1. Exports
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
    for (const auto& exp : export_names) {
        const auto* sym = working_obj.find_symbol(exp);
        if (!sym || sym->section_index < 0) {
            error_ = "cannot export '" + exp + "': the object does not define it";
            return {};
        }
    }

    // 2. Imports: the referenced undefined symbols of the placed sections.
    const bool has_rodata = working_obj.get_section(".rodata") != nullptr;
    const std::string ro_source = has_rodata ? ".rodata" : ".rdata";
    imports::Plan imports;
    if (!imports::plan(working_obj, {".text", ro_source, ".data"}, options_.imports, imports, error_)) {
        return {};
    }

    // 3. .dynstr and the dynamic symbols: null, imports (undefined), exports
    std::vector<uint8_t> dynstr_data;
    dynstr_data.push_back(0);
    auto add_dynstr = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(dynstr_data.size());
        write_cstring(dynstr_data, s);
        return off;
    };

    std::vector<uint32_t> needed_offsets;
    for (const auto& lib : imports.libraries) needed_offsets.push_back(add_dynstr(lib));
    const uint32_t soname_offset = options_.soname.empty() ? 0 : add_dynstr(options_.soname);
    std::string runpath;
    for (const auto& p : options_.rpaths) runpath += (runpath.empty() ? "" : ":") + p;
    const uint32_t runpath_offset = runpath.empty() ? 0 : add_dynstr(runpath);

    std::vector<DynSymEntry> dynsyms;
    dynsyms.push_back({});
    for (const auto& imp : imports.symbols) {
        DynSymEntry entry;
        entry.name = imp.name;
        entry.st_name = add_dynstr(imp.name);
        entry.st_info = static_cast<uint8_t>((elf64::STB_GLOBAL << 4) | elf64::STT_NOTYPE);
        entry.st_shndx = elf64::SHN_UNDEF;
        dynsyms.push_back(entry);
    }
    const size_t first_export_dynsym = dynsyms.size();
    for (const auto& name : export_names) {
        const auto* sym = working_obj.find_symbol(name);
        DynSymEntry entry;
        entry.name = name;
        entry.st_name = add_dynstr(name);
        const uint8_t type = (sym->type == object::SymbolType::Object) ? elf64::STT_OBJECT : elf64::STT_FUNC;
        entry.st_info = static_cast<uint8_t>((elf64::STB_GLOBAL << 4) | (type & 0xF));
        entry.st_size = sym->size;
        dynsyms.push_back(entry);
    }
    auto import_dynsym_index = [&](size_t import_index) { return static_cast<uint32_t>(1 + import_index); };

    // 4. .hash (SysV)
    const uint32_t num_dynsym = static_cast<uint32_t>(dynsyms.size());
    const uint32_t nbucket = std::max(1u, num_dynsym | 1u);
    std::vector<uint32_t> bucket(nbucket, 0);
    std::vector<uint32_t> chain(num_dynsym, 0);
    for (uint32_t i = 1; i < num_dynsym; ++i) {
        const uint32_t h = elf_hash(dynsyms[i].name.c_str()) % nbucket;
        chain[i] = bucket[h];
        bucket[h] = i;
    }
    std::vector<uint8_t> hash_data;
    write_u32(hash_data, nbucket);
    write_u32(hash_data, num_dynsym);
    for (uint32_t b : bucket) write_u32(hash_data, b);
    for (uint32_t c : chain) write_u32(hash_data, c);

    // 5. Sections, with every size that layout needs already known
    size_t rela_count = imports.symbols.size();   // one GLOB_DAT per import
    // An absolute address inside .text (x64's movabs) is a relocation into
    // the executable segment; DT_TEXTREL is what tells the loader to make it
    // writable for the duration, and without the tag the write is a fault.
    bool text_relocs = false;
    for (const auto& sec : working_obj.sections) {
        if (sec.name != ".text" && sec.name != ro_source && sec.name != ".data") continue;
        for (const auto& r : sec.relocations) {
            if (r.kind == object::RelocKind::Abs64) {
                ++rela_count;
                if (sec.name == ".text") text_relocs = true;
            }
        }
    }
    size_t dynamic_count = imports.libraries.size() + 5 + 1;   // NEEDED*, HASH..SYMENT, NULL
    if (soname_offset) ++dynamic_count;
    if (runpath_offset) ++dynamic_count;
    if (rela_count) dynamic_count += 3;
    if (text_relocs) ++dynamic_count;            // TEXTREL
    if (!imports.empty() || text_relocs) ++dynamic_count;   // FLAGS
    if (!imports.empty()) ++dynamic_count;       // FLAGS_1

    std::vector<ElfShdr> sections;
    auto add = [&](const char* name, uint32_t type, uint64_t flags, uint64_t align, uint64_t entsize,
                   std::vector<uint8_t> data) -> uint32_t {
        ElfShdr s;
        s.name = name;
        s.sh_type = type;
        s.sh_flags = flags;
        s.sh_addralign = align;
        s.sh_entsize = entsize;
        s.data = std::move(data);
        sections.push_back(std::move(s));
        return static_cast<uint32_t>(sections.size() - 1);
    };
    sections.push_back({});
    const uint32_t hash_idx = add(".hash", elf64::SHT_HASH, elf64::SHF_ALLOC, 8, 4, std::move(hash_data));
    const uint32_t dynsym_idx = add(".dynsym", elf64::SHT_DYNSYM, elf64::SHF_ALLOC, 8, 24,
                                    std::vector<uint8_t>(dynsyms.size() * 24, 0));
    const uint32_t dynstr_idx = add(".dynstr", elf64::SHT_STRTAB, elf64::SHF_ALLOC, 1, 0, std::move(dynstr_data));
    const uint32_t rela_idx = add(".rela.dyn", elf64::SHT_RELA, elf64::SHF_ALLOC, 8, 24,
                                  std::vector<uint8_t>(rela_count * 24, 0));
    const uint32_t text_idx = add(".text", elf64::SHT_PROGBITS, elf64::SHF_ALLOC | elf64::SHF_EXECINSTR, 16, 0, {});
    if (const auto* ts = working_obj.get_section(".text")) {
        sections[text_idx].data = ts->data;
        sections[text_idx].relocations = ts->relocations;
        sections[text_idx].source = ".text";
    }
    uint32_t plt_idx = 0;
    if (!imports.empty()) {
        plt_idx = add(".plt", elf64::SHT_PROGBITS, elf64::SHF_ALLOC | elf64::SHF_EXECINSTR, 16, PLT_STUB_SIZE,
                      std::vector<uint8_t>(imports.symbols.size() * PLT_STUB_SIZE, 0));
    }
    uint32_t rodata_idx = 0;
    if (const auto* rs = working_obj.get_section(ro_source)) {
        rodata_idx = add(".rodata", elf64::SHT_PROGBITS, elf64::SHF_ALLOC | elf64::SHF_WRITE, 16, 0, rs->data);
        sections[rodata_idx].relocations = rs->relocations;
        sections[rodata_idx].source = ro_source;
    }
    uint32_t got_idx = 0;
    if (!imports.empty()) {
        got_idx = add(".got", elf64::SHT_PROGBITS, elf64::SHF_ALLOC | elf64::SHF_WRITE, 8, 8,
                      std::vector<uint8_t>(imports.symbols.size() * 8, 0));
    }
    const uint32_t dynamic_idx = add(".dynamic", elf64::SHT_DYNAMIC, elf64::SHF_ALLOC | elf64::SHF_WRITE, 8, 16,
                                     std::vector<uint8_t>(dynamic_count * 16, 0));
    uint32_t data_idx = 0;
    if (const auto* ds = working_obj.get_section(".data")) {
        data_idx = add(".data", elf64::SHT_PROGBITS, elf64::SHF_ALLOC | elf64::SHF_WRITE, 16, 0, ds->data);
        sections[data_idx].relocations = ds->relocations;
        sections[data_idx].source = ".data";
    }
    const uint32_t shstrtab_idx = add(".shstrtab", elf64::SHT_STRTAB, 0, 1, 0, {});

    sections[hash_idx].sh_link = dynsym_idx;
    sections[dynsym_idx].sh_link = dynstr_idx;
    sections[dynsym_idx].sh_info = 1;
    sections[rela_idx].sh_link = dynsym_idx;
    sections[dynamic_idx].sh_link = dynstr_idx;

    // 6. Layout: offset == vaddr, three loadable segments
    uint64_t cur = 64 + PHDR_COUNT * 56;
    auto place = [&](uint32_t idx) {
        ElfShdr& s = sections[idx];
        cur = align_up(cur, s.sh_addralign);
        s.sh_offset = cur;
        s.sh_addr = cur;
        s.sh_size = s.data.size();
        cur += s.sh_size;
    };
    place(hash_idx);
    place(dynsym_idx);
    place(dynstr_idx);
    place(rela_idx);
    const uint64_t seg_r_end = cur;

    cur = align_up(cur, PAGE_SIZE);
    const uint64_t seg_rx_start = cur;
    place(text_idx);
    if (plt_idx) place(plt_idx);
    const uint64_t seg_rx_end = cur;

    cur = align_up(cur, PAGE_SIZE);
    const uint64_t seg_rw_start = cur;
    if (rodata_idx) place(rodata_idx);
    if (got_idx) place(got_idx);
    place(dynamic_idx);
    const uint64_t relro_end = align_up(cur, PAGE_SIZE);
    if (data_idx) {
        cur = relro_end;
        place(data_idx);
    }
    const uint64_t seg_rw_end = cur;

    // 7a. Dynamic symbol values
    auto section_for = [&](const std::string& source) -> const ElfShdr* {
        for (const auto& s : sections) {
            if (!s.source.empty() && s.source == source) return &s;
        }
        return nullptr;
    };
    auto placed_index_for = [&](const std::string& source) -> uint16_t {
        for (size_t i = 0; i < sections.size(); ++i) {
            if (!sections[i].source.empty() && sections[i].source == source) return static_cast<uint16_t>(i);
        }
        return elf64::SHN_UNDEF;
    };
    for (size_t i = first_export_dynsym; i < dynsyms.size(); ++i) {
        auto& ds = dynsyms[i];
        const auto* sym = working_obj.find_symbol(ds.name);
        const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
        const ElfShdr* placed = section_for(src_sec.name);
        if (!placed) {
            error_ = "cannot export '" + ds.name + "': its section " + src_sec.name + " is not placed in the image";
            return {};
        }
        ds.st_shndx = placed_index_for(src_sec.name);
        ds.st_value = placed->sh_addr + sym->value;
    }
    std::vector<uint8_t>& dynsym_buf = sections[dynsym_idx].data;
    dynsym_buf.clear();
    for (const auto& ds : dynsyms) {
        write_u32(dynsym_buf, ds.st_name);
        write_u8(dynsym_buf, ds.st_info);
        write_u8(dynsym_buf, ds.st_other);
        write_u16(dynsym_buf, ds.st_shndx);
        write_u64(dynsym_buf, ds.st_value);
        write_u64(dynsym_buf, ds.st_size);
    }

    // 7b. The PLT stubs and their GOT slots
    std::vector<uint8_t> rela_data;
    auto add_rela = [&](uint64_t offset, uint32_t type, uint32_t sym, int64_t addend) {
        write_u64(rela_data, offset);
        write_u64(rela_data, (static_cast<uint64_t>(sym) << 32) | type);
        write_i64(rela_data, addend);
    };
    auto stub_vaddr = [&](size_t import_index) {
        return sections[plt_idx].sh_addr + import_index * PLT_STUB_SIZE;
    };
    if (!imports.empty()) {
        std::vector<uint8_t>& plt = sections[plt_idx].data;
        plt.clear();
        for (const auto& imp : imports.symbols) {
            const uint64_t slot = sections[got_idx].sh_addr + imp.index * 8;
            const uint64_t at = stub_vaddr(imp.index);
            if (aarch64) {
                write_u32(plt, aarch64_adrp(16, at, slot));
                write_u32(plt, aarch64_ldr_x_uoff(17, 16, slot));
                write_u32(plt, aarch64_br(17));
                write_u32(plt, 0xD503201Fu);   // nop
            } else {
                write_u8(plt, 0xFF);
                write_u8(plt, 0x25);
                write_u32(plt, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int64_t>(slot) - static_cast<int64_t>(at + 6))));
                for (int i = 0; i < 10; ++i) write_u8(plt, 0xCC);
            }
            add_rela(slot, aarch64 ? elf64::R_AARCH64_GLOB_DAT : elf64::R_X86_64_GLOB_DAT,
                     import_dynsym_index(imp.index), 0);
        }
    }

    // 7c. Relocations
    for (auto& sec : sections) {
        if (sec.source.empty()) continue;
        for (const auto& r : sec.relocations) {
            uint64_t target_vaddr = 0;
            const imports::Imported* imp = nullptr;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                const ElfShdr* placed = section_for(src_sec.name);
                if (!placed) {
                    error_ = "relocation against '" + r.symbol_name + "' names section " + src_sec.name +
                             ", which is not placed in the image";
                    return {};
                }
                target_vaddr = placed->sh_addr + sym->value;
            } else if (const ElfShdr* placed = section_for(r.symbol_name)) {
                target_vaddr = placed->sh_addr;
            } else if ((imp = imports.find(r.symbol_name)) != nullptr) {
                target_vaddr = stub_vaddr(imp->index);
            } else {
                error_ = "unresolved symbol '" + r.symbol_name + "' referenced from " + sec.name;
                return {};
            }

            const uint64_t reloc_vaddr = sec.sh_addr + r.offset;
            const bool fits4 = r.offset + 4 <= sec.data.size();
            if (r.kind == object::RelocKind::Abs64) {
                if (r.offset + 8 > sec.data.size()) continue;
                if (imp) {
                    // Bound to the symbol itself at load: the pointer a data
                    // word holds is the callee's real address, not the stub.
                    patch_u64(sec.data, r.offset, 0);
                    add_rela(reloc_vaddr, aarch64 ? elf64::R_AARCH64_ABS64 : elf64::R_X86_64_64,
                             import_dynsym_index(imp->index), r.addend);
                } else {
                    const uint64_t unslid = target_vaddr + static_cast<uint64_t>(r.addend);
                    patch_u64(sec.data, r.offset, unslid);
                    add_rela(reloc_vaddr, aarch64 ? elf64::R_AARCH64_RELATIVE : elf64::R_X86_64_RELATIVE, 0,
                             static_cast<int64_t>(unslid));
                }
            } else if (!fits4) {
                continue;
            } else if (aarch64) {
                uint32_t inst = read_u32(sec.data, r.offset);
                if (r.kind == object::RelocKind::Plt32) {
                    const int64_t disp = static_cast<int64_t>(target_vaddr) + r.addend - static_cast<int64_t>(reloc_vaddr);
                    inst = (inst & 0xFC000000u) | (static_cast<uint32_t>(disp >> 2) & 0x03FFFFFFu);
                } else if (r.kind == object::RelocKind::PCRel32) {
                    inst = (inst & 0x9F00001Fu) | (aarch64_adrp(0, reloc_vaddr, target_vaddr) & 0x60FFFFE0u);
                } else if (r.kind == object::RelocKind::SecRel32) {
                    inst = (inst & 0xFFC003FFu) | (static_cast<uint32_t>(target_vaddr & 0xFFFu) << 10);
                } else if (r.kind == object::RelocKind::Abs32 || r.kind == object::RelocKind::Addr32NB) {
                    inst = static_cast<uint32_t>(target_vaddr + static_cast<uint64_t>(r.addend));
                } else {
                    continue;
                }
                patch_u32(sec.data, r.offset, inst);
            } else if (r.kind == object::RelocKind::PCRel32 || r.kind == object::RelocKind::Plt32) {
                // disp = S + A - P, the addend carrying the instruction tail
                // (-4 for a call or lea) exactly as the JIT applies it.
                const int64_t disp = static_cast<int64_t>(target_vaddr) + r.addend - static_cast<int64_t>(reloc_vaddr);
                patch_u32(sec.data, r.offset, static_cast<uint32_t>(static_cast<int32_t>(disp)));
            }
        }
    }
    if (rela_data.size() != rela_count * 24) {
        error_ = "internal: dynamic relocation count changed during resolution";
        return {};
    }
    sections[rela_idx].data = std::move(rela_data);

    // 8. .dynamic
    std::vector<uint8_t>& dyn_buf = sections[dynamic_idx].data;
    dyn_buf.clear();
    auto add_dyn = [&](int64_t tag, uint64_t val) {
        write_i64(dyn_buf, tag);
        write_u64(dyn_buf, val);
    };
    for (uint32_t off : needed_offsets) add_dyn(elf64::DT_NEEDED, off);
    if (soname_offset) add_dyn(elf64::DT_SONAME, soname_offset);
    if (runpath_offset) add_dyn(elf64::DT_RUNPATH, runpath_offset);
    add_dyn(elf64::DT_HASH, sections[hash_idx].sh_addr);
    add_dyn(elf64::DT_STRTAB, sections[dynstr_idx].sh_addr);
    add_dyn(elf64::DT_SYMTAB, sections[dynsym_idx].sh_addr);
    add_dyn(elf64::DT_STRSZ, sections[dynstr_idx].sh_size);
    add_dyn(elf64::DT_SYMENT, 24);
    if (rela_count) {
        add_dyn(elf64::DT_RELA, sections[rela_idx].sh_addr);
        add_dyn(elf64::DT_RELASZ, sections[rela_idx].sh_size);
        add_dyn(elf64::DT_RELAENT, 24);
    }
    if (text_relocs) add_dyn(elf64::DT_TEXTREL, 0);
    if (!imports.empty() || text_relocs) {
        add_dyn(elf64::DT_FLAGS, (imports.empty() ? 0 : elf64::DF_BIND_NOW) |
                                     (text_relocs ? elf64::DF_TEXTREL : 0));
    }
    if (!imports.empty()) add_dyn(elf64::DT_FLAGS_1, elf64::DF_1_NOW);
    add_dyn(elf64::DT_NULL, 0);
    if (dyn_buf.size() != dynamic_count * 16) {
        error_ = "internal: dynamic entry count changed during emission";
        return {};
    }

    // 9. .shstrtab, non-alloc, after the last segment
    std::vector<uint8_t>& shstrtab = sections[shstrtab_idx].data;
    shstrtab.push_back(0);
    for (auto& s : sections) {
        if (s.name.empty()) continue;
        s.sh_name = static_cast<uint32_t>(shstrtab.size());
        write_cstring(shstrtab, s.name);
    }
    sections[shstrtab_idx].sh_size = shstrtab.size();
    sections[shstrtab_idx].sh_offset = cur;
    cur += shstrtab.size();
    const uint64_t shoff = align_up(cur, 8);

    // 10. Assemble
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(shoff + sections.size() * 64));

    write_u8(out, 0x7F); write_u8(out, 'E'); write_u8(out, 'L'); write_u8(out, 'F');
    write_u8(out, elf64::ELFCLASS64);
    write_u8(out, elf64::ELFDATA2LSB);
    write_u8(out, elf64::EV_CURRENT);
    write_u8(out, elf64::ELFOSABI_SYSV);
    for (int i = 0; i < 8; ++i) write_u8(out, 0);
    write_u16(out, elf64::ET_DYN);
    write_u16(out, aarch64 ? elf64::EM_AARCH64 : elf64::EM_X86_64);
    write_u32(out, elf64::EV_CURRENT);
    write_u64(out, 0);      // e_entry
    write_u64(out, 64);     // e_phoff
    write_u64(out, shoff);  // e_shoff
    write_u32(out, 0);      // e_flags
    write_u16(out, 64);     // e_ehsize
    write_u16(out, 56);     // e_phentsize
    write_u16(out, static_cast<uint16_t>(PHDR_COUNT));
    write_u16(out, 64);     // e_shentsize
    write_u16(out, static_cast<uint16_t>(sections.size()));
    write_u16(out, static_cast<uint16_t>(shstrtab_idx));

    auto write_phdr = [&](uint32_t type, uint32_t flags, uint64_t off, uint64_t vaddr, uint64_t filesz,
                          uint64_t memsz, uint64_t align) {
        write_u32(out, type);
        write_u32(out, flags);
        write_u64(out, off);
        write_u64(out, vaddr);
        write_u64(out, vaddr);
        write_u64(out, filesz);
        write_u64(out, memsz);
        write_u64(out, align);
    };
    write_phdr(elf64::PT_PHDR, elf64::PF_R, 64, 64, PHDR_COUNT * 56, PHDR_COUNT * 56, 8);
    write_phdr(elf64::PT_LOAD, elf64::PF_R, 0, 0, seg_r_end, seg_r_end, PAGE_SIZE);
    write_phdr(elf64::PT_LOAD, elf64::PF_R | elf64::PF_X, seg_rx_start, seg_rx_start,
               seg_rx_end - seg_rx_start, seg_rx_end - seg_rx_start, PAGE_SIZE);
    write_phdr(elf64::PT_LOAD, elf64::PF_R | elf64::PF_W, seg_rw_start, seg_rw_start,
               seg_rw_end - seg_rw_start, seg_rw_end - seg_rw_start, PAGE_SIZE);
    write_phdr(elf64::PT_DYNAMIC, elf64::PF_R | elf64::PF_W, sections[dynamic_idx].sh_offset,
               sections[dynamic_idx].sh_addr, sections[dynamic_idx].sh_size, sections[dynamic_idx].sh_size, 8);
    write_phdr(elf64::PT_GNU_RELRO, elf64::PF_R, seg_rw_start, seg_rw_start, relro_end - seg_rw_start,
               relro_end - seg_rw_start, 1);
    write_phdr(elf64::PT_GNU_STACK, elf64::PF_R | elf64::PF_W, 0, 0, 0, 0, 16);

    for (size_t i = 1; i < sections.size(); ++i) {
        const auto& s = sections[i];
        if (out.size() < s.sh_offset) out.resize(static_cast<size_t>(s.sh_offset), 0);
        write_bytes(out, s.data.data(), s.data.size());
    }
    if (out.size() < shoff) out.resize(static_cast<size_t>(shoff), 0);
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
    if (data.empty()) return false;
    return image::write_image_file(path, data, &error_);
}

std::vector<uint8_t> ElfSoWriter::emit(const object::ObjectFile& obj, const ElfSoOptions& options,
                                       std::string* error_out) {
    ElfSoWriter writer(obj, options);
    std::vector<uint8_t> out = writer.write();
    if (error_out) *error_out = writer.error();
    return out;
}

std::vector<uint8_t> ElfSoWriter::emit(const object::ObjectFile& obj, const ElfSoOptions& options) {
    return emit(obj, options, nullptr);
}

std::vector<uint8_t> ElfSoWriter::emit(const object::ObjectFile& obj) {
    return emit(obj, ElfSoOptions(), nullptr);
}

} // namespace brass::target
