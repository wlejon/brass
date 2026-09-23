#include <brass/target/pe_dll_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include "image_file.hpp"
#include "image_util.hpp"
#include "pe_imports.hpp"
#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>

namespace brass::target {

using namespace brass::target::image;

namespace {

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

constexpr uint32_t SECTION_ALIGN = 0x1000;
constexpr uint32_t FILE_ALIGN = 0x200;
constexpr uint32_t SCN_RDATA = object::coff::IMAGE_SCN_CNT_INITIALIZED_DATA | object::coff::IMAGE_SCN_MEM_READ;
constexpr uint32_t SCN_DATA = SCN_RDATA | object::coff::IMAGE_SCN_MEM_WRITE;

} // namespace

PeDllWriter::PeDllWriter(const object::ObjectFile& obj, const PeDllOptions& options)
    : obj_(obj), options_(options) {}

PeDllWriter::PeDllWriter(const object::ObjectFile& obj)
    : obj_(obj), options_(PeDllOptions()) {}

std::vector<uint8_t> PeDllWriter::write() {
    error_.clear();
    object::ObjectFile working_obj = obj_;
    const bool aarch64 = working_obj.target.is_aarch64();
    // Loads of the object's own symbols become `lea`s; what is left loads
    // an import's IAT slot.
    object::relax_got_loads(working_obj);

    // 1. Build SEH unwind tables (.pdata & .xdata) if functions exist
    if (!working_obj.functions.empty()) {
        working_obj.get_or_create_section(".xdata", object::SectionKind::XData,
                                          object::SectionFlags::Read | object::SectionFlags::Alloc, 4);
        working_obj.get_or_create_section(".pdata", object::SectionKind::PData,
                                          object::SectionFlags::Read | object::SectionFlags::Alloc, 4);
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
    for (const auto& exp : export_names) {
        const auto* sym = working_obj.find_symbol(exp);
        if (!sym || sym->section_index < 0) {
            error_ = "cannot export '" + exp + "': the object does not define it";
            return {};
        }
    }

    // 3. Plan the imports: every undefined symbol a placed section references
    // must come from a library, or the image is refused here.
    pe_imports::Table imports;
    imports.aarch64 = aarch64;
    const bool has_rdata = working_obj.get_section(".rdata") != nullptr;
    const std::vector<std::string_view> placed = {".text", has_rdata ? ".rdata" : ".rodata",
                                                  ".xdata", ".pdata", ".data"};
    if (!imports::plan(working_obj, placed, options_.imports, imports.plan, error_)) return {};

    // 4. Collect and organize sections
    // Order: .text, .rdata (if any), .xdata (if any), .pdata (if any), .data (if any), .idata, .edata, .reloc
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

    if (const auto* s = working_obj.get_section(".text")) {
        uint32_t chars = object::coff::IMAGE_SCN_CNT_CODE | object::coff::IMAGE_SCN_MEM_EXECUTE | object::coff::IMAGE_SCN_MEM_READ;
        std::vector<uint8_t> text = s->data;
        // The import thunks live after the last function; .pdata was built
        // above off the functions alone, so nothing describes them and
        // nothing needs to (a thunk never owns a frame).
        imports.append_thunks(text);
        add_section(".text", chars, std::move(text), s->relocations);
    } else if (!imports.empty()) {
        error_ = "the object imports symbols but has no .text section to hold their thunks";
        return {};
    }
    if (const auto* s = working_obj.get_section(".rdata")) {
        add_section(".rdata", SCN_RDATA, s->data, s->relocations);
    } else if (const auto* ro = working_obj.get_section(".rodata")) {
        add_section(".rdata", SCN_RDATA, ro->data, ro->relocations);
    }
    if (const auto* s = working_obj.get_section(".xdata")) {
        add_section(".xdata", SCN_RDATA, s->data, s->relocations);
    }
    if (const auto* s = working_obj.get_section(".pdata")) {
        add_section(".pdata", SCN_RDATA, s->data, s->relocations);
    }
    if (const auto* s = working_obj.get_section(".data")) {
        add_section(".data", SCN_DATA, s->data, s->relocations);
    }
    // Placeholder .idata, sized now and filled once RVAs exist. Writable:
    // the loader stores the resolved addresses into the IAT inside it.
    if (!imports.empty()) {
        add_section(".idata", SCN_DATA, std::vector<uint8_t>(imports.idata_size(), 0), {});
    }

    // Placeholder .edata
    size_t num_exports = export_names.size();
    if (num_exports > 0) {
        // 40 bytes IMAGE_EXPORT_DIRECTORY + N*4 (EAT) + N*4 (ENT) + N*2 (Ordinals) + strings
        size_t est_size = 40 + num_exports * 4 + num_exports * 4 + num_exports * 2 + options_.module_name.size() + 1;
        for (const auto& n : export_names) est_size += n.size() + 1;
        add_section(".edata", SCN_RDATA, std::vector<uint8_t>(est_size, 0), {});
    }

    // Placeholder .reloc section
    PeSectionMeta reloc_meta;
    reloc_meta.name = ".reloc";
    reloc_meta.characteristics = SCN_RDATA | object::coff::IMAGE_SCN_MEM_DISCARDABLE;
    sections.push_back(std::move(reloc_meta));

    // 5. Compute RVAs and file offsets for headers and sections
    const uint32_t dos_header_size = 64;
    const uint32_t dos_stub_size = sizeof(DOS_STUB);
    const uint32_t optional_header_size = 240;
    const uint32_t section_headers_size = static_cast<uint32_t>(sections.size() * 40);
    const uint32_t total_headers_size = dos_header_size + dos_stub_size + 4 + 20 +
                                        optional_header_size + section_headers_size;
    const uint32_t size_of_headers = static_cast<uint32_t>(align_up(total_headers_size, FILE_ALIGN));

    uint32_t cur_rva = SECTION_ALIGN;
    uint32_t cur_file_offset = size_of_headers;
    for (auto& sec : sections) {
        sec.rva = cur_rva;
        sec.file_offset = cur_file_offset;
        sec.raw_size = static_cast<uint32_t>(align_up(sec.virtual_size, FILE_ALIGN));
        cur_rva = static_cast<uint32_t>(align_up(cur_rva + std::max(sec.virtual_size, 1u), SECTION_ALIGN));
        cur_file_offset += sec.raw_size;
    }

    auto find_meta = [&](std::string_view name) -> PeSectionMeta* {
        for (auto& s : sections) {
            if (s.name == name) return &s;
        }
        return nullptr;
    };

    // 6. Fill .idata and patch the thunks now that the IAT has an address
    PeSectionMeta* text_meta = find_meta(".text");
    if (PeSectionMeta* idata_meta = find_meta(".idata")) {
        imports.emit(idata_meta->data, idata_meta->rva, text_meta->data, text_meta->rva);
        idata_meta->virtual_size = static_cast<uint32_t>(idata_meta->data.size());
        idata_meta->raw_size = static_cast<uint32_t>(align_up(idata_meta->virtual_size, FILE_ALIGN));
    }

    // 7. Populate Export Directory in .edata
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

        std::vector<uint32_t> name_rvas;
        name_rvas.reserve(num_exports);
        uint32_t mod_name_rva = edata_meta->rva + static_cast<uint32_t>(edata_meta->data.size());
        write_cstring(edata_meta->data, options_.module_name);
        for (const auto& name : export_names) {
            name_rvas.push_back(edata_meta->rva + static_cast<uint32_t>(edata_meta->data.size()));
            write_cstring(edata_meta->data, name);
        }

        export_dir_size = static_cast<uint32_t>(edata_meta->data.size());
        edata_meta->virtual_size = export_dir_size;
        edata_meta->raw_size = static_cast<uint32_t>(align_up(export_dir_size, FILE_ALIGN));

        // IMAGE_EXPORT_DIRECTORY: Characteristics, TimeDateStamp, Major, Minor stay 0.
        patch_u32(edata_meta->data, 12, mod_name_rva);
        patch_u32(edata_meta->data, 16, 1);   // Base ordinal
        patch_u32(edata_meta->data, 20, static_cast<uint32_t>(num_exports));
        patch_u32(edata_meta->data, 24, static_cast<uint32_t>(num_exports));
        patch_u32(edata_meta->data, 28, edata_meta->rva + eat_offset);
        patch_u32(edata_meta->data, 32, edata_meta->rva + ent_offset);
        patch_u32(edata_meta->data, 36, edata_meta->rva + ord_offset);

        for (size_t i = 0; i < num_exports; ++i) {
            const auto* sym = working_obj.find_symbol(export_names[i]);
            const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
            PeSectionMeta* m = find_meta(src_sec.name == ".rodata" ? ".rdata" : src_sec.name);
            const uint32_t fn_rva = m ? m->rva + static_cast<uint32_t>(sym->value) : 0;
            patch_u32(edata_meta->data, eat_offset + i * 4, fn_rva);
            patch_u32(edata_meta->data, ent_offset + i * 4, name_rvas[i]);
            const uint16_t ord = static_cast<uint16_t>(i);
            std::memcpy(edata_meta->data.data() + ord_offset + i * 2, &ord, 2);
        }
    }

    // 8. Resolve all relocations across sections
    std::vector<uint32_t> abs_reloc_rvas;

    for (auto& sec : sections) {
        if (sec.name == ".reloc") continue;
        for (const auto& r : sec.relocations) {
            uint32_t target_rva = 0;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            const bool got_load = r.kind == object::RelocKind::GotPCRel32 || object::a64::is_got_kind(r.kind);
            if (sym && sym->section_index >= 0) {
                const auto& src_sec = working_obj.sections[static_cast<size_t>(sym->section_index)];
                PeSectionMeta* sm = find_meta(src_sec.name == ".rodata" ? ".rdata" : src_sec.name);
                if (!sm) {
                    error_ = "relocation against '" + r.symbol_name + "' names section " +
                             src_sec.name + ", which is not placed in the image";
                    return {};
                }
                target_rva = sm->rva + static_cast<uint32_t>(sym->value);
            } else if (PeSectionMeta* sm = find_meta(r.symbol_name == ".rodata" ? ".rdata" : r.symbol_name)) {
                target_rva = sm->rva;
            } else if (const auto* imp = imports.plan.find(r.symbol_name)) {
                // A GOT load reads the IAT slot, which holds the import's
                // real address; every other reference takes the thunk.
                target_rva = got_load ? imports.slot_rvas[imp->index]
                                      : imports.thunk_rva(imp->index, text_meta->rva);
            } else {
                error_ = "unresolved symbol '" + r.symbol_name + "' referenced from " + sec.name;
                return {};
            }
            if (got_load && !imports.plan.find(r.symbol_name)) {
                error_ = "internal: GOT load of '" + r.symbol_name + "' survived relaxation without an import";
                return {};
            }

            const uint32_t reloc_rva = sec.rva + static_cast<uint32_t>(r.offset);
            if (aarch64 && (r.kind == object::RelocKind::Plt32 || object::a64::is_instruction_kind(r.kind))) {
                // B/BL, ADRP (page arithmetic: RVAs keep the page offsets of
                // the VAs, the image base being 64 KB aligned), ADD and the
                // access-size-scaled LDR/STR offsets. A GOT pair reads the
                // import's IAT slot.
                if (r.offset + 4 > sec.data.size()) {
                    error_ = "relocation against '" + r.symbol_name + "' lies outside " + sec.name;
                    return {};
                }
                if (got_load && r.addend != 0) {
                    error_ = "GOT load of '" + r.symbol_name + "' carries an addend";
                    return {};
                }
                uint32_t inst = read_u32(sec.data, r.offset);
                const uint64_t value = static_cast<uint64_t>(static_cast<int64_t>(target_rva) + r.addend);
                const std::string err = object::a64::patch(r.kind, inst, reloc_rva, value);
                if (!err.empty()) {
                    error_ = "relocation against '" + r.symbol_name + "' in " + sec.name + ": " + err;
                    return {};
                }
                patch_u32(sec.data, r.offset, inst);
            } else if (r.kind == object::RelocKind::PCRel32 ||
                       (!aarch64 && (r.kind == object::RelocKind::Plt32 || got_load))) {
                if (r.offset + 4 > sec.data.size()) continue;
                // The addend already accounts for the instruction's tail
                // (-4 for a call/lea displacement), as the JIT applies it:
                // disp = S + A - P.
                int64_t disp = static_cast<int64_t>(target_rva) + r.addend - static_cast<int64_t>(reloc_rva);
                patch_u32(sec.data, r.offset, static_cast<uint32_t>(static_cast<int32_t>(disp)));
            } else if (r.kind == object::RelocKind::Addr32NB) {
                if (r.offset + 4 <= sec.data.size()) {
                    patch_u32(sec.data, r.offset, static_cast<uint32_t>(target_rva + r.addend));
                }
            } else if (r.kind == object::RelocKind::SecRel32) {
                if (r.offset + 4 <= sec.data.size()) {
                    patch_u32(sec.data, r.offset, static_cast<uint32_t>((sym ? sym->value : 0) + r.addend));
                }
            } else if (r.kind == object::RelocKind::Abs64) {
                if (r.offset + 8 <= sec.data.size()) {
                    patch_u64(sec.data, r.offset, options_.image_base + target_rva + static_cast<uint64_t>(r.addend));
                }
                abs_reloc_rvas.push_back(reloc_rva);
            } else if (r.kind == object::RelocKind::Abs32) {
                if (r.offset + 4 <= sec.data.size()) {
                    patch_u32(sec.data, r.offset, static_cast<uint32_t>(options_.image_base + target_rva + static_cast<uint64_t>(r.addend)));
                }
            } else {
                error_ = "relocation against '" + r.symbol_name + "' in " + sec.name + " has a kind (" +
                         std::to_string(static_cast<int>(r.kind)) + ") a " + (aarch64 ? "ARM64" : "x64") +
                         " PE image cannot resolve";
                return {};
            }
        }
    }

    // 9. Generate .reloc section
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
                pages[rva & ~0xFFFu].push_back(static_cast<uint16_t>(rva & 0xFFFu));
            }
            for (auto& [page_rva, offsets] : pages) {
                std::sort(offsets.begin(), offsets.end());
                uint32_t num_entries = static_cast<uint32_t>(offsets.size());
                uint32_t block_size = 8 + num_entries * 2;
                if (num_entries % 2 != 0) block_size += 2; // Pad to 4 bytes
                write_u32(reloc_meta_ptr->data, page_rva);
                write_u32(reloc_meta_ptr->data, block_size);
                for (uint16_t off : offsets) {
                    write_u16(reloc_meta_ptr->data, static_cast<uint16_t>((pe::IMAGE_REL_BASED_DIR64 << 12) | off));
                }
                if (num_entries % 2 != 0) write_u16(reloc_meta_ptr->data, 0);
            }
        }
        reloc_meta_ptr->virtual_size = static_cast<uint32_t>(reloc_meta_ptr->data.size());
        reloc_meta_ptr->raw_size = static_cast<uint32_t>(align_up(reloc_meta_ptr->virtual_size, FILE_ALIGN));
    }

    uint32_t size_of_image = SECTION_ALIGN;
    for (const auto& sec : sections) {
        size_of_image = static_cast<uint32_t>(align_up(sec.rva + std::max(sec.virtual_size, 1u), SECTION_ALIGN));
    }

    // 10. Assemble binary DLL output
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
    write_u32(out, dos_header_size + dos_stub_size); // e_lfanew = 0x80

    write_bytes(out, DOS_STUB, sizeof(DOS_STUB));
    write_u32(out, pe::IMAGE_NT_SIGNATURE);

    // IMAGE_FILE_HEADER (20 bytes)
    write_u16(out, aarch64 ? pe::IMAGE_FILE_MACHINE_ARM64 : pe::IMAGE_FILE_MACHINE_AMD64);
    write_u16(out, static_cast<uint16_t>(sections.size()));
    write_u32(out, 0); // TimeDateStamp
    write_u32(out, 0); // PointerToSymbolTable
    write_u32(out, 0); // NumberOfSymbols
    write_u16(out, static_cast<uint16_t>(optional_header_size));
    write_u16(out, pe::IMAGE_FILE_EXECUTABLE_IMAGE | pe::IMAGE_FILE_DLL | pe::IMAGE_FILE_LARGE_ADDRESS_AWARE);

    // IMAGE_OPTIONAL_HEADER64 (240 bytes)
    write_u16(out, pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC); // 0x020B
    write_u8(out, 14); // MajorLinkerVersion
    write_u8(out, 0);  // MinorLinkerVersion

    uint32_t size_of_init_data = 0;
    for (const auto& s : sections) {
        if (s.name != ".text") size_of_init_data += s.raw_size;
    }
    write_u32(out, text_meta ? text_meta->raw_size : 0);   // SizeOfCode
    write_u32(out, size_of_init_data);
    write_u32(out, 0); // SizeOfUninitializedData
    write_u32(out, 0); // AddressOfEntryPoint (DLL without DllMain)
    write_u32(out, text_meta ? text_meta->rva : SECTION_ALIGN);   // BaseOfCode

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
    write_u16(out, pe::IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE |
                   pe::IMAGE_DLLCHARACTERISTICS_NX_COMPAT |
                   pe::IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA);

    write_u64(out, 0x100000); // SizeOfStackReserve
    write_u64(out, 0x1000);   // SizeOfStackCommit
    write_u64(out, 0x100000); // SizeOfHeapReserve
    write_u64(out, 0x1000);   // SizeOfHeapCommit
    write_u32(out, 0);        // LoaderFlags
    write_u32(out, 16);       // NumberOfRvaAndSizes

    // 16 Data Directories (8 bytes each: RVA, Size)
    PeSectionMeta* pdata_meta = find_meta(".pdata");
    PeSectionMeta* reloc_sec = find_meta(".reloc");
    struct Dir { uint32_t rva, size; };
    Dir dirs[16] = {};
    dirs[pe::IMAGE_DIRECTORY_ENTRY_EXPORT] = {export_dir_rva, export_dir_size};
    dirs[pe::IMAGE_DIRECTORY_ENTRY_IMPORT] = {imports.directory_rva, imports.directory_size};
    dirs[pe::IMAGE_DIRECTORY_ENTRY_EXCEPTION] = {pdata_meta ? pdata_meta->rva : 0, pdata_meta ? pdata_meta->virtual_size : 0};
    dirs[pe::IMAGE_DIRECTORY_ENTRY_BASERELOC] = {reloc_sec ? reloc_sec->rva : 0, reloc_sec ? reloc_sec->virtual_size : 0};
    dirs[pe::IMAGE_DIRECTORY_ENTRY_IAT] = {imports.iat_rva, imports.iat_size};
    for (const Dir& d : dirs) {
        write_u32(out, d.rva);
        write_u32(out, d.size);
    }

    // Section Headers (40 bytes each)
    for (const auto& sec : sections) {
        write_fixed_string(out, sec.name, 8);
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

    if (out.size() < size_of_headers) out.resize(size_of_headers, 0);

    for (const auto& sec : sections) {
        write_bytes(out, sec.data.data(), sec.data.size());
        if (sec.data.size() < sec.raw_size) {
            out.resize(out.size() + (sec.raw_size - sec.data.size()), 0);
        }
    }

    return out;
}

bool PeDllWriter::write_to_file(const std::string& path) {
    auto data = write();
    if (data.empty()) return false;
    return image::write_image_file(path, data, &error_);
}

std::vector<uint8_t> PeDllWriter::emit(const object::ObjectFile& obj, const PeDllOptions& options,
                                       std::string* error_out) {
    PeDllWriter writer(obj, options);
    std::vector<uint8_t> out = writer.write();
    if (error_out) *error_out = writer.error();
    return out;
}

std::vector<uint8_t> PeDllWriter::emit(const object::ObjectFile& obj, const PeDllOptions& options) {
    return emit(obj, options, nullptr);
}

std::vector<uint8_t> PeDllWriter::emit(const object::ObjectFile& obj) {
    return emit(obj, PeDllOptions(), nullptr);
}

} // namespace brass::target
