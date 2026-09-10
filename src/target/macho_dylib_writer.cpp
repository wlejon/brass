#include <brass/target/macho_dylib_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <string_view>

namespace brass::target {

using namespace brass::object;

namespace {

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

std::string to_macho_symbol_name(const std::string& name) {
    if (name.empty()) return name;
    if (name[0] == '_') return name;
    return "_" + name;
}

void encode_uleb128(std::vector<uint8_t>& buf, uint64_t val) {
    while (true) {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if (val != 0) {
            byte |= 0x80;
        }
        buf.push_back(byte);
        if (val == 0) break;
    }
}

size_t uleb128_len(uint64_t val) {
    size_t len = 0;
    while (true) {
        val >>= 7;
        len++;
        if (val == 0) break;
    }
    return len;
}

struct ExportNode {
    std::string edge;
    bool is_terminal = false;
    uint64_t address = 0;
    uint32_t offset = 0;
    std::vector<std::unique_ptr<ExportNode>> children;

    void add(std::string_view name, uint64_t addr) {
        if (name.empty()) {
            is_terminal = true;
            address = addr;
            return;
        }
        for (auto& child : children) {
            size_t match = 0;
            while (match < name.size() && match < child->edge.size() && name[match] == child->edge[match]) {
                match++;
            }
            if (match > 0) {
                if (match == child->edge.size()) {
                    child->add(name.substr(match), addr);
                    return;
                }
                auto split = std::make_unique<ExportNode>();
                split->edge = child->edge.substr(0, match);
                child->edge = child->edge.substr(match);
                std::string_view remainder = name.substr(match);
                if (remainder.empty()) {
                    split->is_terminal = true;
                    split->address = addr;
                } else {
                    auto new_child = std::make_unique<ExportNode>();
                    new_child->edge = std::string(remainder);
                    new_child->is_terminal = true;
                    new_child->address = addr;
                    split->children.push_back(std::move(new_child));
                }
                split->children.push_back(std::move(child));
                child = std::move(split);
                return;
            }
        }
        auto new_child = std::make_unique<ExportNode>();
        new_child->edge = std::string(name);
        new_child->is_terminal = true;
        new_child->address = addr;
        children.push_back(std::move(new_child));
    }

    size_t compute_size() const {
        size_t sz = 0;
        if (is_terminal) {
            size_t term_data_len = uleb128_len(0) + uleb128_len(address);
            sz += uleb128_len(term_data_len) + term_data_len;
        } else {
            sz += uleb128_len(0);
        }
        sz += 1; // child count
        for (const auto& ch : children) {
            sz += ch->edge.size() + 1; // null-terminated edge label
            sz += uleb128_len(ch->offset);
        }
        return sz;
    }

    void collect(std::vector<ExportNode*>& order) {
        order.push_back(this);
        for (auto& ch : children) {
            ch->collect(order);
        }
    }

    void serialize(std::vector<uint8_t>& buf) const {
        if (is_terminal) {
            std::vector<uint8_t> term_data;
            encode_uleb128(term_data, 0); // flags: EXPORT_SYMBOL_FLAGS_KIND_REGULAR
            encode_uleb128(term_data, address);
            encode_uleb128(buf, term_data.size());
            buf.insert(buf.end(), term_data.begin(), term_data.end());
        } else {
            encode_uleb128(buf, 0);
        }
        buf.push_back(static_cast<uint8_t>(children.size()));
        for (const auto& ch : children) {
            buf.insert(buf.end(), ch->edge.begin(), ch->edge.end());
            buf.push_back(0);
            encode_uleb128(buf, ch->offset);
        }
    }
};

std::vector<uint8_t> build_export_trie(const std::vector<std::pair<std::string, uint64_t>>& exports) {
    if (exports.empty()) return {};

    ExportNode root;
    for (const auto& [name, addr] : exports) {
        root.add(name, addr);
    }

    std::vector<ExportNode*> order;
    root.collect(order);

    for (int iter = 0; iter < 10; ++iter) {
        uint32_t cur_off = 0;
        bool changed = false;
        for (auto* node : order) {
            if (node->offset != cur_off) {
                node->offset = cur_off;
                changed = true;
            }
            cur_off += static_cast<uint32_t>(node->compute_size());
        }
        if (!changed) break;
    }

    std::vector<uint8_t> trie_bytes;
    for (const auto* node : order) {
        node->serialize(trie_bytes);
    }
    return trie_bytes;
}

} // namespace

MachODylibWriter::MachODylibWriter(const ObjectFile& obj, const MachODylibOptions& options)
    : obj_(obj), options_(options) {}

MachODylibWriter::MachODylibWriter(const ObjectFile& obj)
    : obj_(obj), options_() {}

std::vector<uint8_t> MachODylibWriter::write() {
    ObjectFile working_obj = obj_;

    // Determine exported functions
    auto should_export = [&](const std::string& name) -> bool {
        if (options_.export_all_functions) return true;
        for (const auto& exp : options_.explicit_exports) {
            if (exp == name || (!name.empty() && name[0] == '_' && name.substr(1) == exp)) {
                return true;
            }
        }
        return false;
    };

    // Gather sections
    Section* text_sec = working_obj.get_section(".text");
    Section dummy_text;
    if (!text_sec) {
        text_sec = &dummy_text;
    }

    Section* const_sec = working_obj.get_section(".rodata");
    if (!const_sec) const_sec = working_obj.get_section(".rdata");
    if (!const_sec) const_sec = working_obj.get_section("__const");
    for (auto& s : working_obj.sections) {
        if (s.kind == SectionKind::RoData && !const_sec) {
            const_sec = &s;
            break;
        }
    }

    Section* data_sec = working_obj.get_section(".data");
    for (auto& s : working_obj.sections) {
        if (s.kind == SectionKind::Data && !data_sec) {
            data_sec = &s;
            break;
        }
    }

    Section* bss_sec = working_obj.get_section(".bss");

    // Sections in __TEXT
    uint32_t text_num_sections = 1; // __text
    if (const_sec && !const_sec->data.empty()) {
        text_num_sections++; // __const
    }

    // Sections in __DATA
    uint32_t data_num_sections = 0;
    if (data_sec && !data_sec->data.empty()) data_num_sections++;
    if (bss_sec && !bss_sec->data.empty()) data_num_sections++;

    bool has_data_segment = (data_num_sections > 0);

    // Compute load commands sizes
    uint32_t seg_text_cmdsize = 72 + text_num_sections * 80;
    uint32_t seg_data_cmdsize = has_data_segment ? (72 + data_num_sections * 80) : 0;
    uint32_t seg_linkedit_cmdsize = 72;

    std::string install_name = options_.install_name.empty() ? "brass_module.dylib" : options_.install_name;
    uint32_t id_dylib_cmdsize = 24 + static_cast<uint32_t>((install_name.size() + 1 + 7) & ~7u);

    std::string load_dylib_name = "/usr/lib/libSystem.B.dylib";
    uint32_t load_dylib_cmdsize = 24 + static_cast<uint32_t>((load_dylib_name.size() + 1 + 7) & ~7u);

    uint32_t dyld_info_cmdsize = 48; // LC_DYLD_INFO_ONLY
    uint32_t symtab_cmdsize = 24;    // LC_SYMTAB
    uint32_t dysymtab_cmdsize = 80;  // LC_DYSYMTAB
    uint32_t build_version_cmdsize = 24; // LC_BUILD_VERSION
    uint32_t uuid_cmdsize = 24;          // LC_UUID

    uint32_t ncmds = 7 + (has_data_segment ? 1 : 0) + 2; // + LC_BUILD_VERSION + LC_UUID
    uint32_t sizeofcmds = seg_text_cmdsize + seg_data_cmdsize + seg_linkedit_cmdsize +
                          id_dylib_cmdsize + load_dylib_cmdsize + dyld_info_cmdsize +
                          symtab_cmdsize + dysymtab_cmdsize + build_version_cmdsize + uuid_cmdsize;

    // Layout __TEXT segment
    uint32_t header_and_cmds = 32 + sizeofcmds;
    // Leave padding after load commands so codesign can append LC_CODE_SIGNATURE without clobbering __text
    uint32_t text_fileoff = (header_and_cmds + 256 + 15) & ~15u;
    if (text_fileoff < 1024) text_fileoff = 1024;
    uint64_t text_vaddr = options_.image_base + text_fileoff;
    uint64_t text_size = text_sec->data.size();

    uint32_t cur_text_end = text_fileoff + static_cast<uint32_t>(text_size);

    uint32_t const_fileoff = 0;
    uint64_t const_vaddr = 0;
    uint64_t const_size = 0;
    if (const_sec && !const_sec->data.empty()) {
        const_fileoff = (cur_text_end + 15) & ~15u;
        const_vaddr = options_.image_base + const_fileoff;
        const_size = const_sec->data.size();
        cur_text_end = const_fileoff + static_cast<uint32_t>(const_size);
    }

    constexpr uint32_t PAGE_SIZE = 4096;
    uint32_t text_filesize = (cur_text_end + (PAGE_SIZE - 1)) & ~(PAGE_SIZE - 1);
    uint64_t text_vmsize = text_filesize;

    // Layout __DATA segment (if needed)
    uint32_t data_fileoff = text_filesize;
    uint64_t data_vaddr = options_.image_base + data_fileoff;
    uint32_t data_filesize = 0;
    uint64_t data_vmsize = 0;
    uint32_t cur_data_end = data_fileoff;

    uint32_t data_sec_fileoff = 0;
    uint64_t data_sec_vaddr = 0;
    uint64_t data_sec_size = 0;

    if (has_data_segment) {
        if (data_sec && !data_sec->data.empty()) {
            data_sec_fileoff = (cur_data_end + 15) & ~15u;
            data_sec_vaddr = options_.image_base + data_sec_fileoff;
            data_sec_size = data_sec->data.size();
            cur_data_end = data_sec_fileoff + static_cast<uint32_t>(data_sec_size);
        }
        data_filesize = (cur_data_end - data_fileoff + (PAGE_SIZE - 1)) & ~(PAGE_SIZE - 1);
        data_vmsize = data_filesize;
    }

    // Layout __LINKEDIT segment
    uint32_t linkedit_fileoff = has_data_segment ? (data_fileoff + data_filesize) : text_filesize;
    uint64_t linkedit_vaddr = options_.image_base + linkedit_fileoff;

    // Resolve internal relocations
    auto resolve_section_relocs = [&](Section& sec, uint64_t sec_vaddr) {
        for (const auto& r : sec.relocations) {
            uint64_t target_vaddr = 0;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                if (src_sec.name == ".text" || src_sec.kind == SectionKind::Text) {
                    target_vaddr = text_vaddr + sym->value;
                } else if (src_sec.name == ".rodata" || src_sec.name == ".rdata" || src_sec.name == "__const" || src_sec.kind == SectionKind::RoData) {
                    target_vaddr = const_vaddr + sym->value;
                } else if (src_sec.name == ".data" || src_sec.kind == SectionKind::Data) {
                    target_vaddr = data_sec_vaddr + sym->value;
                }
            }

            uint64_t reloc_vaddr = sec_vaddr + r.offset;

            if (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::Plt32) {
                int64_t disp = static_cast<int64_t>(target_vaddr + static_cast<uint64_t>(r.addend)) - static_cast<int64_t>(reloc_vaddr + 4);
                int32_t disp32 = static_cast<int32_t>(disp);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &disp32, 4);
                }
            } else if (r.kind == RelocKind::SecRel32) {
                uint32_t val32 = static_cast<uint32_t>(static_cast<int64_t>(sym ? sym->value : 0) + r.addend);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val32, 4);
                }
            } else if (r.kind == RelocKind::Abs64) {
                uint64_t val64 = target_vaddr + static_cast<uint64_t>(r.addend);
                if (r.offset + 8 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val64, 8);
                }
            } else if (r.kind == RelocKind::Abs32 || r.kind == RelocKind::Addr32NB) {
                uint32_t val32 = static_cast<uint32_t>(target_vaddr + static_cast<uint64_t>(r.addend));
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val32, 4);
                }
            }
        }
    };

    resolve_section_relocs(*text_sec, text_vaddr);
    if (const_sec && !const_sec->data.empty()) {
        resolve_section_relocs(*const_sec, const_vaddr);
    }
    if (data_sec && !data_sec->data.empty()) {
        resolve_section_relocs(*data_sec, data_sec_vaddr);
    }

    // Build Exports & Symbols
    std::vector<std::pair<std::string, uint64_t>> exported_symbols;
    std::vector<std::string> local_symbol_names;
    std::vector<uint64_t> local_symbol_addrs;

    for (const auto& fn : working_obj.functions) {
        uint64_t fn_vaddr = text_vaddr + fn.text_offset;
        uint64_t fn_offset_from_base = fn_vaddr - options_.image_base;
        std::string macho_name = to_macho_symbol_name(fn.name);

        if (should_export(fn.name)) {
            exported_symbols.emplace_back(macho_name, fn_offset_from_base);
            if (macho_name != fn.name) {
                exported_symbols.emplace_back(fn.name, fn_offset_from_base);
            }
        } else {
            local_symbol_names.push_back(macho_name);
            local_symbol_addrs.push_back(fn_vaddr);
        }
    }

    // Generate Export Trie
    std::vector<uint8_t> export_trie = build_export_trie(exported_symbols);
    uint32_t export_fileoff = linkedit_fileoff;
    uint32_t export_size = static_cast<uint32_t>(export_trie.size());

    // Build String Table and Symbol Table (nlist_64)
    std::vector<uint8_t> strtab;
    strtab.push_back(0); // 0th byte

    auto add_to_strtab = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(strtab.size());
        strtab.insert(strtab.end(), s.begin(), s.end());
        strtab.push_back(0);
        return off;
    };

    struct Nlist64 {
        uint32_t n_strx = 0;
        uint8_t n_type = 0;
        uint8_t n_sect = 0;
        uint16_t n_desc = 0;
        uint64_t n_value = 0;
    };

    std::vector<Nlist64> local_nlists;
    for (size_t i = 0; i < local_symbol_names.size(); ++i) {
        Nlist64 nl;
        nl.n_strx = add_to_strtab(local_symbol_names[i]);
        nl.n_type = macho::N_SECT; // Local defined
        nl.n_sect = 1; // 1-based (__text)
        nl.n_desc = 0;
        nl.n_value = local_symbol_addrs[i];
        local_nlists.push_back(nl);
    }

    std::vector<Nlist64> extdef_nlists;
    for (const auto& [name, off_from_base] : exported_symbols) {
        Nlist64 nl;
        nl.n_strx = add_to_strtab(name);
        nl.n_type = macho::N_SECT | macho::N_EXT; // External defined
        nl.n_sect = 1; // 1-based (__text)
        nl.n_desc = 0;
        nl.n_value = options_.image_base + off_from_base;
        extdef_nlists.push_back(nl);
    }

    uint32_t ilocalsym = 0;
    uint32_t nlocalsym = static_cast<uint32_t>(local_nlists.size());
    uint32_t iextdefsym = nlocalsym;
    uint32_t nextdefsym = static_cast<uint32_t>(extdef_nlists.size());
    uint32_t iundefsym = iextdefsym + nextdefsym;
    uint32_t nundefsym = 0;

    std::vector<Nlist64> all_nlists;
    all_nlists.insert(all_nlists.end(), local_nlists.begin(), local_nlists.end());
    all_nlists.insert(all_nlists.end(), extdef_nlists.begin(), extdef_nlists.end());

    // __LINKEDIT layout
    uint32_t cur_linkedit_off = export_fileoff + export_size;
    cur_linkedit_off = (cur_linkedit_off + 7) & ~7u;
    uint32_t symoff = cur_linkedit_off;
    uint32_t nsyms = static_cast<uint32_t>(all_nlists.size());
    cur_linkedit_off += nsyms * 16;

    uint32_t stroff = cur_linkedit_off;
    uint32_t strsize = static_cast<uint32_t>(strtab.size());
    cur_linkedit_off += strsize;

    uint32_t linkedit_filesize = cur_linkedit_off - linkedit_fileoff;
    uint64_t linkedit_vmsize = (linkedit_filesize + (PAGE_SIZE - 1)) & ~(PAGE_SIZE - 1);

    // Build binary buffer
    std::vector<uint8_t> out(cur_linkedit_off, 0);

    // 1. Write mach_header_64 (32 bytes)
    auto patch_u32 = [](uint8_t* p, uint32_t v) {
        p[0] = static_cast<uint8_t>(v & 0xFF);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    };

    uint8_t* hdr_ptr = out.data();
    patch_u32(hdr_ptr + 0, macho::MH_MAGIC_64);
    patch_u32(hdr_ptr + 4, static_cast<uint32_t>(macho::CPU_TYPE_X86_64));
    patch_u32(hdr_ptr + 8, static_cast<uint32_t>(macho::CPU_SUBTYPE_X86_64_ALL));
    patch_u32(hdr_ptr + 12, macho::MH_DYLIB);
    patch_u32(hdr_ptr + 16, ncmds);
    patch_u32(hdr_ptr + 20, sizeofcmds);
    patch_u32(hdr_ptr + 24, macho::MH_NOUNDEFS | macho::MH_DYLDLINK | macho::MH_TWOLEVEL);
    patch_u32(hdr_ptr + 28, 0); // reserved

    // 2. Load commands
    std::vector<uint8_t> cmds;

    // LC_SEGMENT_64 __TEXT
    write_u32(cmds, macho::LC_SEGMENT_64);
    write_u32(cmds, seg_text_cmdsize);
    write_fixed_string(cmds, "__TEXT", 16);
    write_u64(cmds, options_.image_base);
    write_u64(cmds, text_vmsize);
    write_u64(cmds, 0); // fileoff
    write_u64(cmds, text_filesize);
    write_u32(cmds, macho::VM_PROT_READ | macho::VM_PROT_EXECUTE); // maxprot
    write_u32(cmds, macho::VM_PROT_READ | macho::VM_PROT_EXECUTE); // initprot
    write_u32(cmds, text_num_sections);
    write_u32(cmds, 0); // flags

    // section_64 __text
    write_fixed_string(cmds, "__text", 16);
    write_fixed_string(cmds, "__TEXT", 16);
    write_u64(cmds, text_vaddr);
    write_u64(cmds, text_size);
    write_u32(cmds, text_fileoff);
    write_u32(cmds, 4); // align 2^4 = 16
    write_u32(cmds, 0); // reloff
    write_u32(cmds, 0); // nreloc
    write_u32(cmds, macho::S_REGULAR | macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS);
    write_u32(cmds, 0); // reserved1
    write_u32(cmds, 0); // reserved2
    write_u32(cmds, 0); // reserved3

    // section_64 __const (if present)
    if (const_sec && !const_sec->data.empty()) {
        write_fixed_string(cmds, "__const", 16);
        write_fixed_string(cmds, "__TEXT", 16);
        write_u64(cmds, const_vaddr);
        write_u64(cmds, const_size);
        write_u32(cmds, const_fileoff);
        write_u32(cmds, 4); // align 16
        write_u32(cmds, 0); // reloff
        write_u32(cmds, 0); // nreloc
        write_u32(cmds, macho::S_REGULAR);
        write_u32(cmds, 0);
        write_u32(cmds, 0);
        write_u32(cmds, 0);
    }

    // LC_SEGMENT_64 __DATA (if present)
    if (has_data_segment) {
        write_u32(cmds, macho::LC_SEGMENT_64);
        write_u32(cmds, seg_data_cmdsize);
        write_fixed_string(cmds, "__DATA", 16);
        write_u64(cmds, data_vaddr);
        write_u64(cmds, data_vmsize);
        write_u64(cmds, data_fileoff);
        write_u64(cmds, data_filesize);
        write_u32(cmds, macho::VM_PROT_READ | macho::VM_PROT_WRITE);
        write_u32(cmds, macho::VM_PROT_READ | macho::VM_PROT_WRITE);
        write_u32(cmds, data_num_sections);
        write_u32(cmds, 0);

        if (data_sec && !data_sec->data.empty()) {
            write_fixed_string(cmds, "__data", 16);
            write_fixed_string(cmds, "__DATA", 16);
            write_u64(cmds, data_sec_vaddr);
            write_u64(cmds, data_sec_size);
            write_u32(cmds, data_sec_fileoff);
            write_u32(cmds, 4);
            write_u32(cmds, 0);
            write_u32(cmds, 0);
            write_u32(cmds, macho::S_REGULAR);
            write_u32(cmds, 0);
            write_u32(cmds, 0);
            write_u32(cmds, 0);
        }
    }

    // LC_SEGMENT_64 __LINKEDIT
    write_u32(cmds, macho::LC_SEGMENT_64);
    write_u32(cmds, seg_linkedit_cmdsize);
    write_fixed_string(cmds, "__LINKEDIT", 16);
    write_u64(cmds, linkedit_vaddr);
    write_u64(cmds, linkedit_vmsize);
    write_u64(cmds, linkedit_fileoff);
    write_u64(cmds, linkedit_filesize);
    write_u32(cmds, macho::VM_PROT_READ);
    write_u32(cmds, macho::VM_PROT_READ);
    write_u32(cmds, 0); // nsects
    write_u32(cmds, 0);

    // LC_ID_DYLIB
    write_u32(cmds, macho::LC_ID_DYLIB);
    write_u32(cmds, id_dylib_cmdsize);
    write_u32(cmds, 24); // name offset
    write_u32(cmds, 1);  // timestamp
    write_u32(cmds, 0x00010000); // current version 1.0.0
    write_u32(cmds, 0x00010000); // compat version 1.0.0
    write_bytes(cmds, install_name.c_str(), install_name.size() + 1);
    align_buf(cmds, 8);

    // LC_LOAD_DYLIB (/usr/lib/libSystem.B.dylib)
    write_u32(cmds, macho::LC_LOAD_DYLIB);
    write_u32(cmds, load_dylib_cmdsize);
    write_u32(cmds, 24); // name offset
    write_u32(cmds, 2);  // timestamp
    write_u32(cmds, 0x05470000); // current version 1351.0.0
    write_u32(cmds, 0x00010000); // compat version 1.0.0
    write_bytes(cmds, load_dylib_name.c_str(), load_dylib_name.size() + 1);
    align_buf(cmds, 8);

    // LC_DYLD_INFO_ONLY
    write_u32(cmds, macho::LC_DYLD_INFO_ONLY);
    write_u32(cmds, dyld_info_cmdsize);
    write_u32(cmds, 0); // rebase_off
    write_u32(cmds, 0); // rebase_size
    write_u32(cmds, 0); // bind_off
    write_u32(cmds, 0); // bind_size
    write_u32(cmds, 0); // weak_bind_off
    write_u32(cmds, 0); // weak_bind_size
    write_u32(cmds, 0); // lazy_bind_off
    write_u32(cmds, 0); // lazy_bind_size
    write_u32(cmds, export_fileoff);
    write_u32(cmds, export_size);

    // LC_SYMTAB
    write_u32(cmds, macho::LC_SYMTAB);
    write_u32(cmds, symtab_cmdsize);
    write_u32(cmds, symoff);
    write_u32(cmds, nsyms);
    write_u32(cmds, stroff);
    write_u32(cmds, strsize);

    // LC_DYSYMTAB
    write_u32(cmds, macho::LC_DYSYMTAB);
    write_u32(cmds, dysymtab_cmdsize);
    write_u32(cmds, ilocalsym);
    write_u32(cmds, nlocalsym);
    write_u32(cmds, iextdefsym);
    write_u32(cmds, nextdefsym);
    write_u32(cmds, iundefsym);
    write_u32(cmds, nundefsym);
    for (int i = 0; i < 12; ++i) {
        write_u32(cmds, 0);
    }

    // LC_BUILD_VERSION
    write_u32(cmds, macho::LC_BUILD_VERSION);
    write_u32(cmds, build_version_cmdsize);
    write_u32(cmds, 1); // platform 1 = PLATFORM_MACOS
    write_u32(cmds, 0x000b0000); // minos 11.0
    write_u32(cmds, 0x000e0000); // sdk 14.0
    write_u32(cmds, 0); // ntools

    // LC_UUID (24 bytes)
    constexpr uint32_t LC_UUID = 0x1b;
    write_u32(cmds, LC_UUID);
    write_u32(cmds, uuid_cmdsize);
    uint8_t uuid_bytes[16] = {
        0x42, 0x52, 0x41, 0x53, 0x53, 0x2d, 0x44, 0x59,
        0x4c, 0x49, 0x42, 0x2d, 0x58, 0x36, 0x34, 0x01
    };
    write_bytes(cmds, uuid_bytes, 16);

    // Copy load commands into output
    std::memcpy(out.data() + 32, cmds.data(), cmds.size());

    // Write __text section data
    if (!text_sec->data.empty()) {
        std::memcpy(out.data() + text_fileoff, text_sec->data.data(), text_sec->data.size());
    }

    // Write __const section data
    if (const_sec && !const_sec->data.empty()) {
        std::memcpy(out.data() + const_fileoff, const_sec->data.data(), const_sec->data.size());
    }

    // Write __data section data
    if (data_sec && !data_sec->data.empty()) {
        std::memcpy(out.data() + data_sec_fileoff, data_sec->data.data(), data_sec->data.size());
    }

    // Write export trie
    if (!export_trie.empty()) {
        std::memcpy(out.data() + export_fileoff, export_trie.data(), export_trie.size());
    }

    // Write symbol table
    uint8_t* sym_ptr = out.data() + symoff;
    for (const auto& nl : all_nlists) {
        uint32_t strx = nl.n_strx;
        uint8_t type = nl.n_type;
        uint8_t sect = nl.n_sect;
        uint16_t desc = nl.n_desc;
        uint64_t val = nl.n_value;

        std::memcpy(sym_ptr + 0, &strx, 4);
        sym_ptr[4] = type;
        sym_ptr[5] = sect;
        std::memcpy(sym_ptr + 6, &desc, 2);
        std::memcpy(sym_ptr + 8, &val, 8);
        sym_ptr += 16;
    }

    // Write string table
    std::memcpy(out.data() + stroff, strtab.data(), strtab.size());

    return out;
}

bool MachODylibWriter::write_to_file(const std::string& path) {
    auto data = write();
    if (data.empty()) return false;
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    file.close();

#if defined(__APPLE__)
    std::string cmd = "codesign -s - -f \"" + path + "\" > /dev/null 2>&1";
    int res = std::system(cmd.c_str());
    (void)res;
#endif

    return true;
}

std::vector<uint8_t> MachODylibWriter::emit(const ObjectFile& obj, const MachODylibOptions& options) {
    MachODylibWriter writer(obj, options);
    return writer.write();
}

std::vector<uint8_t> MachODylibWriter::emit(const ObjectFile& obj) {
    MachODylibWriter writer(obj);
    return writer.write();
}

} // namespace brass::target
