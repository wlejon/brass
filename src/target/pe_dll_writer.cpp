#include <brass/target/pe_dll_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
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

void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t count) {
    const auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + count);
}

uint32_t align_up_u32(uint32_t val, uint32_t align) {
    if (align <= 1) return val;
    uint32_t rem = val % align;
    return rem == 0 ? val : val + (align - rem);
}

const uint8_t DOS_STUB[64] = {
    0x0E, 0x1F, 0xBA, 0x0E, 0x00, 0xB4, 0x09, 0xCD, 0x21, 0xB8, 0x01, 0x4C, 0xCD, 0x21,
    'T', 'h', 'i', 's', ' ', 'p', 'r', 'o', 'g', 'r', 'a', 'm', ' ', 'c', 'a', 'n', 'n', 'o', 't', ' ',
    'b', 'e', ' ', 'r', 'u', 'n', ' ', 'i', 'n', ' ', 'D', 'O', 'S', ' ', 'm', 'o', 'd', 'e', '.', '\r', '\r', '\n', '$',
    0, 0, 0, 0, 0, 0, 0
};

struct PeSectionMeta {
    std::string name;
    uint32_t virtual_size = 0;
    uint32_t rva = 0;
    uint32_t raw_size = 0;
    uint32_t file_offset = 0;
    uint32_t characteristics = 0;
    std::vector<uint8_t> data;
    std::vector<object::ObjectRelocation> relocations;
};

struct ExportEntry {
    std::string name;
    uint32_t function_rva = 0;
    uint32_t ordinal = 0;
};

} // namespace

PeDllWriter::PeDllWriter(const object::ObjectFile& obj, const PeDllOptions& options)
    : obj_(obj), options_(options) {}

PeDllWriter::PeDllWriter(const object::ObjectFile& obj)
    : obj_(obj), options_(PeDllOptions()) {}

std::vector<uint8_t> PeDllWriter::write() {
    object::ObjectFile working_obj = obj_;

    // 1. Build SEH unwind tables (.pdata & .xdata) if functions exist
    if (!working_obj.functions.empty()) {
        working_obj.get_or_create_section(
            ".xdata",
            object::SectionKind::XData,
            object::SectionFlags::Read | object::SectionFlags::Alloc,
            4
        );
        working_obj.get_or_create_section(
            ".pdata",
            object::SectionKind::PData,
            object::SectionFlags::Read | object::SectionFlags::Alloc,
            4
        );
        object::Section* pdata_sec = working_obj.get_section(".pdata");
        object::Section* xdata_sec = working_obj.get_section(".xdata");
        if (pdata_sec && xdata_sec) {
            object::CoffUnwindBuilder::build_unwind_info(working_obj, *pdata_sec, *xdata_sec);
        }
    }

    // 2. Identify symbols to export
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
    // Strictly lexicographically sort export names for binary search in GetProcAddress
    std::sort(export_names.begin(), export_names.end());

    // 3. Collect and organize sections
    // Order: .text, .rdata (if any), .xdata (if any), .pdata (if any), .data (if any), .edata, .reloc
    std::vector<PeSectionMeta> sections;

    auto add_section = [&](std::string name, uint32_t chars, std::vector<uint8_t> d, std::vector<object::ObjectRelocation> r) {
        PeSectionMeta m;
        m.name = std::move(name);
        m.characteristics = chars;
        m.virtual_size = static_cast<uint32_t>(d.size());
        m.data = std::move(d);
        m.relocations = std::move(r);
        sections.push_back(std::move(m));
    };

    // .text
    if (const auto* s = working_obj.get_section(".text")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_CODE | object::coff::IMAGE_SCN_MEM_EXECUTE | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".text", chars, s->data, s->relocations);
    }
    // .rdata
    if (const auto* s = working_obj.get_section(".rdata")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".rdata", chars, s->data, s->relocations);
    } else if (const auto* ro = working_obj.get_section(".rodata")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".rdata", chars, ro->data, ro->relocations);
    }
    // .xdata
    if (const auto* s = working_obj.get_section(".xdata")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".xdata", chars, s->data, s->relocations);
    }
    // .pdata
    if (const auto* s = working_obj.get_section(".pdata")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".pdata", chars, s->data, s->relocations);
    }
    // .data
    if (const auto* s = working_obj.get_section(".data")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ | object::coff::IMAGE_SCN_MEM_WRITE;
        add_section(".data", chars, s->data, s->relocations);
    }

    // Allocate placeholder .edata
    size_t num_exports = export_names.size();
    std::vector<uint8_t> edata_data;
    if (num_exports > 0) {
        // Compute size: 40 bytes IMAGE_EXPORT_DIRECTORY + N*4 (EAT) + N*4 (ENT) + N*2 (Ordinals) + strings
        size_t est_size = 40 + num_exports * 4 + num_exports * 4 + num_exports * 2 + options_.module_name.size() + 1;
        for (const auto& n : export_names) {
            est_size += n.size() + 1;
        }
        edata_data.resize(est_size, 0);
        uint32_t chars = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
        add_section(".edata", chars, std::move(edata_data), {});
    }

    // Placeholder .reloc section
    PeSectionMeta reloc_meta;
    reloc_meta.name = ".reloc";
    reloc_meta.characteristics = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ | 0x42000040; // DISCARDABLE
    sections.push_back(std::move(reloc_meta));

    // 4. Compute RVAs and file offsets for headers and sections
    constexpr uint32_t SECTION_ALIGN = 0x1000;
    constexpr uint32_t FILE_ALIGN = 0x200;

    uint32_t dos_header_size = 64;
    uint32_t dos_stub_size = sizeof(DOS_STUB);
    uint32_t pe_sig_size = 4;
    uint32_t file_header_size = 20;
    uint32_t optional_header_size = 240;
    uint32_t section_headers_size = static_cast<uint32_t>(sections.size() * 40);

    uint32_t total_headers_size = dos_header_size + dos_stub_size + pe_sig_size +
                                  file_header_size + optional_header_size + section_headers_size;
    uint32_t size_of_headers = align_up_u32(total_headers_size, FILE_ALIGN);

    uint32_t cur_rva = SECTION_ALIGN;
    uint32_t cur_file_offset = size_of_headers;

    for (size_t i = 0; i < sections.size(); ++i) {
        auto& sec = sections[i];
        sec.rva = cur_rva;
        sec.file_offset = cur_file_offset;
        sec.raw_size = align_up_u32(sec.virtual_size, FILE_ALIGN);
        cur_rva = align_up_u32(cur_rva + std::max(sec.virtual_size, 1u), SECTION_ALIGN);
        cur_file_offset += sec.raw_size;
    }

    // Helper to find section meta by name
    auto find_meta = [&](std::string_view name) -> PeSectionMeta* {
        for (auto& s : sections) {
            if (s.name == name) return &s;
        }
        return nullptr;
    };

    // 5. Populate Export Directory in .edata
    PeSectionMeta* edata_meta = find_meta(".edata");
    uint32_t export_dir_rva = 0;
    uint32_t export_dir_size = 0;

    if (edata_meta && num_exports > 0) {
        edata_meta->data.clear();
        export_dir_rva = edata_meta->rva;

        uint32_t eat_offset = 40;
        uint32_t ent_offset = eat_offset + static_cast<uint32_t>(num_exports * 4);
        uint32_t ord_offset = ent_offset + static_cast<uint32_t>(num_exports * 4);
        uint32_t strings_offset = ord_offset + static_cast<uint32_t>(num_exports * 2);

        edata_meta->data.resize(strings_offset, 0);

        // Strings
        std::vector<uint32_t> name_rvas;
        name_rvas.reserve(num_exports);

        uint32_t mod_name_rva = edata_meta->rva + static_cast<uint32_t>(edata_meta->data.size());
        write_bytes(edata_meta->data, options_.module_name.c_str(), options_.module_name.size() + 1);

        for (const auto& name : export_names) {
            uint32_t nrva = edata_meta->rva + static_cast<uint32_t>(edata_meta->data.size());
            name_rvas.push_back(nrva);
            write_bytes(edata_meta->data, name.c_str(), name.size() + 1);
        }

        export_dir_size = static_cast<uint32_t>(edata_meta->data.size());
        edata_meta->virtual_size = export_dir_size;
        edata_meta->raw_size = align_up_u32(export_dir_size, FILE_ALIGN);

        // Populate IMAGE_EXPORT_DIRECTORY (40 bytes)
        uint8_t* p = edata_meta->data.data();
        // Characteristics (0), TimeDateStamp (0), Major (0), Minor (0) -> 12 bytes 0
        // Name RVA (offset 12)
        std::memcpy(p + 12, &mod_name_rva, 4);
        uint32_t base = 1;
        std::memcpy(p + 16, &base, 4);
        uint32_t n_funcs = static_cast<uint32_t>(num_exports);
        std::memcpy(p + 20, &n_funcs, 4);
        std::memcpy(p + 24, &n_funcs, 4);
        uint32_t eat_rva = edata_meta->rva + eat_offset;
        uint32_t ent_rva = edata_meta->rva + ent_offset;
        uint32_t ord_rva = edata_meta->rva + ord_offset;
        std::memcpy(p + 28, &eat_rva, 4);
        std::memcpy(p + 32, &ent_rva, 4);
        std::memcpy(p + 36, &ord_rva, 4);

        // Populate EAT, ENT, Ordinals
        for (size_t i = 0; i < num_exports; ++i) {
            const std::string& exp_name = export_names[i];
            uint32_t fn_rva = 0;
            const auto* sym = working_obj.find_symbol(exp_name);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                PeSectionMeta* m = find_meta(src_sec.name);
                if (m) {
                    fn_rva = m->rva + static_cast<uint32_t>(sym->value);
                }
            }
            std::memcpy(p + eat_offset + i * 4, &fn_rva, 4);
            std::memcpy(p + ent_offset + i * 4, &name_rvas[i], 4);
            uint16_t ord = static_cast<uint16_t>(i);
            std::memcpy(p + ord_offset + i * 2, &ord, 2);
        }
    }

    // 6. Resolve all relocations across sections
    std::vector<uint32_t> abs_reloc_rvas;

    for (auto& sec : sections) {
        if (sec.name == ".reloc") continue;
        for (const auto& r : sec.relocations) {
            uint32_t target_rva = 0;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                PeSectionMeta* sm = find_meta(src_sec.name);
                if (sm) {
                    target_rva = sm->rva + static_cast<uint32_t>(sym->value);
                }
            } else {
                PeSectionMeta* sm = find_meta(r.symbol_name);
                if (sm) {
                    target_rva = sm->rva;
                }
            }

            uint32_t reloc_rva = sec.rva + static_cast<uint32_t>(r.offset);

            if (r.kind == object::RelocKind::PCRel32 || r.kind == object::RelocKind::Plt32) {
                int64_t disp = static_cast<int64_t>(target_rva + r.addend) - static_cast<int64_t>(reloc_rva + 4);
                int32_t disp32 = static_cast<int32_t>(disp);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &disp32, 4);
                }
            } else if (r.kind == object::RelocKind::Addr32NB) {
                uint32_t val32 = static_cast<uint32_t>(target_rva + r.addend);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val32, 4);
                }
            } else if (r.kind == object::RelocKind::SecRel32) {
                uint32_t val32 = static_cast<uint32_t>((sym ? sym->value : 0) + r.addend);
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val32, 4);
                }
            } else if (r.kind == object::RelocKind::Abs64) {
                uint64_t val64 = options_.image_base + target_rva + static_cast<uint64_t>(r.addend);
                if (r.offset + 8 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val64, 8);
                }
                abs_reloc_rvas.push_back(reloc_rva);
            } else if (r.kind == object::RelocKind::Abs32) {
                uint32_t val32 = static_cast<uint32_t>(options_.image_base + target_rva + static_cast<uint64_t>(r.addend));
                if (r.offset + 4 <= sec.data.size()) {
                    std::memcpy(sec.data.data() + r.offset, &val32, 4);
                }
            }
        }
    }

    // 7. Generate .reloc section
    PeSectionMeta* reloc_meta_ptr = find_meta(".reloc");
    if (reloc_meta_ptr) {
        reloc_meta_ptr->data.clear();
        if (abs_reloc_rvas.empty()) {
            // Emit single dummy base relocation block (12 bytes: 8 header + 2 dummy absolute entries)
            write_u32(reloc_meta_ptr->data, SECTION_ALIGN);
            write_u32(reloc_meta_ptr->data, 12);
            write_u16(reloc_meta_ptr->data, 0); // IMAGE_REL_BASED_ABSOLUTE
            write_u16(reloc_meta_ptr->data, 0);
        } else {
            std::map<uint32_t, std::vector<uint16_t>> pages;
            for (uint32_t rva : abs_reloc_rvas) {
                uint32_t page_rva = rva & ~0xFFFu;
                uint16_t off = static_cast<uint16_t>(rva & 0xFFFu);
                pages[page_rva].push_back(off);
            }

            for (auto& [page_rva, offsets] : pages) {
                std::sort(offsets.begin(), offsets.end());
                uint32_t num_entries = static_cast<uint32_t>(offsets.size());
                uint32_t block_size = 8 + num_entries * 2;
                if (num_entries % 2 != 0) {
                    block_size += 2; // Pad to 4 bytes
                }

                write_u32(reloc_meta_ptr->data, page_rva);
                write_u32(reloc_meta_ptr->data, block_size);
                for (uint16_t off : offsets) {
                    uint16_t entry = static_cast<uint16_t>((pe::IMAGE_REL_BASED_DIR64 << 12) | off);
                    write_u16(reloc_meta_ptr->data, entry);
                }
                if (num_entries % 2 != 0) {
                    write_u16(reloc_meta_ptr->data, 0);
                }
            }
        }

        reloc_meta_ptr->virtual_size = static_cast<uint32_t>(reloc_meta_ptr->data.size());
        reloc_meta_ptr->raw_size = align_up_u32(reloc_meta_ptr->virtual_size, FILE_ALIGN);
    }

    // Recompute total image virtual size
    uint32_t size_of_image = SECTION_ALIGN;
    for (const auto& sec : sections) {
        size_of_image = align_up_u32(sec.rva + std::max(sec.virtual_size, 1u), SECTION_ALIGN);
    }

    // 8. Assemble binary DLL output
    std::vector<uint8_t> out;
    out.reserve(size_of_headers + cur_file_offset);

    // MS-DOS 2.0 Header (64 bytes)
    write_u16(out, pe::IMAGE_DOS_SIGNATURE); // e_magic = 0x5A4D
    write_u16(out, 0x0090); // e_cblp
    write_u16(out, 0x0003); // e_cp
    write_u16(out, 0x0000); // e_crlc
    write_u16(out, 0x0004); // e_cparhdr
    write_u16(out, 0x0000); // e_minalloc
    write_u16(out, 0xFFFF); // e_maxalloc
    write_u16(out, 0x0000); // e_ss
    write_u16(out, 0x00B8); // e_sp
    write_u16(out, 0x0000); // e_csum
    write_u16(out, 0x0000); // e_ip
    write_u16(out, 0x0000); // e_cs
    write_u16(out, 0x0040); // e_lfarlc
    write_u16(out, 0x0000); // e_ovno
    for (int i = 0; i < 4; ++i) write_u16(out, 0); // e_res[4]
    write_u16(out, 0); // e_oemid
    write_u16(out, 0); // e_oeminfo
    for (int i = 0; i < 10; ++i) write_u16(out, 0); // e_res2[10]
    uint32_t pe_offset = dos_header_size + dos_stub_size; // 0x80
    write_u32(out, pe_offset); // e_lfanew

    // MS-DOS 2.0 Stub (64 bytes)
    write_bytes(out, DOS_STUB, sizeof(DOS_STUB));

    // PE Signature ("PE\0\0" = 0x00004550)
    write_u32(out, pe::IMAGE_NT_SIGNATURE);

    // IMAGE_FILE_HEADER (20 bytes)
    write_u16(out, pe::IMAGE_FILE_MACHINE_AMD64);
    write_u16(out, static_cast<uint16_t>(sections.size()));
    write_u32(out, 0); // TimeDateStamp
    write_u32(out, 0); // PointerToSymbolTable
    write_u32(out, 0); // NumberOfSymbols
    write_u16(out, static_cast<uint16_t>(optional_header_size));
    uint16_t file_chars = pe::IMAGE_FILE_EXECUTABLE_IMAGE | pe::IMAGE_FILE_DLL | pe::IMAGE_FILE_LARGE_ADDRESS_AWARE;
    write_u16(out, file_chars);

    // IMAGE_OPTIONAL_HEADER64 (240 bytes)
    write_u16(out, pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC); // 0x020B
    write_u8(out, 14); // MajorLinkerVersion
    write_u8(out, 0);  // MinorLinkerVersion

    PeSectionMeta* text_meta = find_meta(".text");
    uint32_t size_of_code = text_meta ? text_meta->raw_size : 0;
    uint32_t base_of_code = text_meta ? text_meta->rva : SECTION_ALIGN;

    uint32_t size_of_init_data = 0;
    for (const auto& s : sections) {
        if (s.name != ".text") size_of_init_data += s.raw_size;
    }

    write_u32(out, size_of_code);
    write_u32(out, size_of_init_data);
    write_u32(out, 0); // SizeOfUninitializedData
    write_u32(out, 0); // AddressOfEntryPoint (DLL without DllMain)
    write_u32(out, base_of_code);

    write_u64(out, options_.image_base);
    write_u32(out, SECTION_ALIGN);
    write_u32(out, FILE_ALIGN);
    write_u16(out, 6); // MajorOperatingSystemVersion
    write_u16(out, 0); // MinorOperatingSystemVersion
    write_u16(out, 0); // MajorImageVersion
    write_u16(out, 0); // MinorImageVersion
    write_u16(out, 6); // MajorSubsystemVersion
    write_u16(out, 0); // MinorSubsystemVersion
    write_u32(out, 0); // Win32VersionValue
    write_u32(out, size_of_image);
    write_u32(out, size_of_headers);
    write_u32(out, 0); // CheckSum
    write_u16(out, pe::IMAGE_SUBSYSTEM_WINDOWS_CUI);
    uint16_t dll_chars = pe::IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE |
                         pe::IMAGE_DLLCHARACTERISTICS_NX_COMPAT |
                         pe::IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA;
    write_u16(out, dll_chars);

    write_u64(out, 0x100000); // SizeOfStackReserve
    write_u64(out, 0x1000);   // SizeOfStackCommit
    write_u64(out, 0x100000); // SizeOfHeapReserve
    write_u64(out, 0x1000);   // SizeOfHeapCommit
    write_u32(out, 0);        // LoaderFlags
    write_u32(out, 16);       // NumberOfRvaAndSizes

    // 16 Data Directories (8 bytes each: RVA, Size)
    PeSectionMeta* pdata_meta = find_meta(".pdata");
    uint32_t pdata_rva = pdata_meta ? pdata_meta->rva : 0;
    uint32_t pdata_size = pdata_meta ? pdata_meta->virtual_size : 0;

    PeSectionMeta* reloc_sec = find_meta(".reloc");
    uint32_t reloc_rva = reloc_sec ? reloc_sec->rva : 0;
    uint32_t reloc_size = reloc_sec ? reloc_sec->virtual_size : 0;

    // [0] Export
    write_u32(out, export_dir_rva);
    write_u32(out, export_dir_size);
    // [1] Import
    write_u32(out, 0); write_u32(out, 0);
    // [2] Resource
    write_u32(out, 0); write_u32(out, 0);
    // [3] Exception (.pdata)
    write_u32(out, pdata_rva);
    write_u32(out, pdata_size);
    // [4] Security
    write_u32(out, 0); write_u32(out, 0);
    // [5] Base Relocation (.reloc)
    write_u32(out, reloc_rva);
    write_u32(out, reloc_size);
    // [6..15]
    for (int i = 6; i < 16; ++i) {
        write_u32(out, 0);
        write_u32(out, 0);
    }

    // Section Headers (40 bytes each)
    for (const auto& sec : sections) {
        uint8_t name_buf[8] = {0};
        std::memcpy(name_buf, sec.name.data(), std::min(sec.name.size(), size_t(8)));
        write_bytes(out, name_buf, 8);
        write_u32(out, sec.virtual_size);
        write_u32(out, sec.rva);
        write_u32(out, sec.raw_size);
        write_u32(out, sec.file_offset);
        write_u32(out, 0); // PointerToRelocations
        write_u32(out, 0); // PointerToLinenumbers
        write_u16(out, 0); // NumberOfRelocations
        write_u16(out, 0); // NumberOfLinenumbers
        write_u32(out, sec.characteristics);
    }

    // Pad headers to size_of_headers
    if (out.size() < size_of_headers) {
        out.resize(size_of_headers, 0);
    }

    // Write Section Raw Data
    for (const auto& sec : sections) {
        size_t current_file_pos = out.size();
        (void)current_file_pos;
        write_bytes(out, sec.data.data(), sec.data.size());
        if (sec.data.size() < sec.raw_size) {
            out.resize(out.size() + (sec.raw_size - sec.data.size()), 0);
        }
    }

    return out;
}

bool PeDllWriter::write_to_file(const std::string& path) {
    auto data = write();
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

std::vector<uint8_t> PeDllWriter::emit(const object::ObjectFile& obj, const PeDllOptions& options) {
    PeDllWriter writer(obj, options);
    return writer.write();
}

std::vector<uint8_t> PeDllWriter::emit(const object::ObjectFile& obj) {
    return emit(obj, PeDllOptions());
}

} // namespace brass::target
