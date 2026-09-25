#include <brass/object/macho_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#if defined(__APPLE__)
#include <Availability.h>
#endif

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
    std::vector<std::pair<int32_t, uint32_t>> encoded_relocs;   // r_address, r_info word
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

uint32_t reloc_info(uint32_t symbolnum, uint32_t pcrel, uint32_t length, uint32_t is_extern, uint32_t type) {
    return (symbolnum & 0x00FFFFFFu) | ((pcrel & 1u) << 24) | ((length & 3u) << 25) |
           ((is_extern & 1u) << 27) | ((type & 0xFu) << 28);
}

[[noreturn]] void bad_reloc(const ObjectRelocation& r, const char* why) {
    throw std::runtime_error("Mach-O writer: relocation against '" + r.symbol_name + "' (kind " +
                             std::to_string(static_cast<int>(r.kind)) + "): " + why);
}

// Appends the relocation_info entries for `r` to `s.encoded_relocs`.
void encode_macho_reloc(MachOSectionEntry& s, const ObjectRelocation& r, bool is_aarch64,
                        const std::unordered_map<std::string, uint32_t>& sec_name_to_idx,
                        const std::unordered_map<std::string, uint32_t>& sym_name_to_idx,
                        const std::vector<MachOSymbolEntry>& all_symbols) {
    uint32_t sym_idx = 0;
    uint32_t r_extern = 1;
    auto sec_it = sec_name_to_idx.find(r.symbol_name);
    if (sec_it != sec_name_to_idx.end()) {
        r_extern = 0;
        sym_idx = sec_it->second + 1; // 1-based section number
    } else {
        auto it = sym_name_to_idx.find(r.symbol_name);
        if (it != sym_name_to_idx.end()) {
            sym_idx = it->second;
        }
    }
    const int32_t r_address = static_cast<int32_t>(r.offset);

    if (!is_aarch64) {
        const bool is_func = r_extern && (sym_idx < all_symbols.size() && all_symbols[sym_idx].type == SymbolType::Function);
        uint32_t r_pcrel = 0;
        uint32_t r_length = 2;
        uint32_t r_type = macho::X86_64_RELOC_BRANCH;
        if (r.kind == RelocKind::PCRel32) {
            r_pcrel = 1;
            r_type = macho::X86_64_RELOC_SIGNED;
        } else if (r.kind == RelocKind::Plt32) {
            r_pcrel = 1;
            r_type = is_func ? macho::X86_64_RELOC_BRANCH : macho::X86_64_RELOC_SIGNED;
        } else if (r.kind == RelocKind::SecRel32) {
            r_pcrel = 1;
            r_type = macho::X86_64_RELOC_SIGNED;
        } else if (r.kind == RelocKind::GotPCRel32) {
            r_pcrel = 1;
            r_type = macho::X86_64_RELOC_GOT_LOAD;
        } else if (r.kind == RelocKind::Abs64) {
            r_length = 3; // 8 bytes
            r_type = macho::X86_64_RELOC_UNSIGNED;
        } else if (r.kind == RelocKind::Abs32 || r.kind == RelocKind::Addr32NB) {
            r_type = macho::X86_64_RELOC_UNSIGNED;
        } else if (a64::is_instruction_kind(r.kind)) {
            bad_reloc(r, "an AArch64 relocation in an x86-64 object");
        }
        s.encoded_relocs.emplace_back(r_address, reloc_info(sym_idx, r_pcrel, r_length, r_extern, r_type));
        return;
    }

    // ARM64: the instruction relocations take their addend from a preceding
    // ARM64_RELOC_ADDEND (24-bit signed, in r_symbolnum); UNSIGNED takes it
    // from the bytes it relocates.
    uint32_t r_type = 0;
    uint32_t r_pcrel = 0;
    uint32_t r_length = 2;
    bool addend_entry = false;
    switch (r.kind) {
        case RelocKind::Plt32:
            r_type = macho::ARM64_RELOC_BRANCH26; r_pcrel = 1; addend_entry = true; break;
        case RelocKind::AdrPage21:
            r_type = macho::ARM64_RELOC_PAGE21; r_pcrel = 1; addend_entry = true; break;
        // ld64 reads the access size of a PAGEOFF12 from the instruction
        // (ADD: unscaled; LDR/STR: scaled by the size it decodes).
        case RelocKind::AddLo12:
        case RelocKind::LdSt8Lo12:
        case RelocKind::LdSt16Lo12:
        case RelocKind::LdSt32Lo12:
        case RelocKind::LdSt64Lo12:
        case RelocKind::LdSt128Lo12:
            r_type = macho::ARM64_RELOC_PAGEOFF12; addend_entry = true; break;
        case RelocKind::GotPage21:
            if (r.addend != 0) bad_reloc(r, "a GOT load cannot carry an addend");
            r_type = macho::ARM64_RELOC_GOT_LOAD_PAGE21; r_pcrel = 1; break;
        case RelocKind::GotLo12:
            if (r.addend != 0) bad_reloc(r, "a GOT load cannot carry an addend");
            r_type = macho::ARM64_RELOC_GOT_LOAD_PAGEOFF12; break;
        case RelocKind::Abs64:
        case RelocKind::Abs32: {
            r_type = macho::ARM64_RELOC_UNSIGNED;
            r_length = r.kind == RelocKind::Abs64 ? 3 : 2;
            const size_t width = r.kind == RelocKind::Abs64 ? 8 : 4;
            if (r.addend != 0) {
                if (r.offset + width > s.data.size()) bad_reloc(r, "outside its section");
                const uint64_t a = static_cast<uint64_t>(r.addend);
                std::memcpy(s.data.data() + r.offset, &a, width);
            }
            break;
        }
        case RelocKind::PCRel32:
        case RelocKind::SecRel32:
        case RelocKind::Addr32NB:
        case RelocKind::SecIdx:
        case RelocKind::GotPCRel32:
            bad_reloc(r, "no ARM64 Mach-O equivalent");
    }
    if (addend_entry && r.addend != 0) {
        if (r.addend < -(int64_t(1) << 23) || r.addend >= (int64_t(1) << 23)) {
            bad_reloc(r, "addend does not fit ARM64_RELOC_ADDEND's 24 bits");
        }
        s.encoded_relocs.emplace_back(
            r_address, reloc_info(static_cast<uint32_t>(r.addend), 0, 2, 0, macho::ARM64_RELOC_ADDEND));
    }
    s.encoded_relocs.emplace_back(r_address, reloc_info(sym_idx, r_pcrel, r_length, r_extern, r_type));
}

} // namespace

std::optional<uint32_t> MachOBuildVersion::parse_version(std::string_view text) {
    uint32_t parts[3] = {0, 0, 0};
    size_t n = 0;
    size_t i = 0;
    while (true) {
        if (n == 3 || i >= text.size() || text[i] < '0' || text[i] > '9') return std::nullopt;
        uint32_t v = 0;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            v = v * 10 + static_cast<uint32_t>(text[i] - '0');
            if (v > 0xFFFF) return std::nullopt;
            ++i;
        }
        parts[n++] = v;
        if (i == text.size()) break;
        if (text[i] != '.') return std::nullopt;
        ++i;
    }
    if (parts[1] > 0xFF || parts[2] > 0xFF) return std::nullopt;
    return parts[0] << 16 | parts[1] << 8 | parts[2];
}

namespace {

// A version as Availability.h writes it from 10.10 / iOS 10 on: decimal
// XXYYZZ (110000 is 11.0). 0 if brass is not built for that platform.
constexpr uint32_t availability_version(long v) {
    return v <= 0 ? 0u
                  : static_cast<uint32_t>((v / 10000) << 16 | ((v / 100) % 100) << 8 | (v % 100));
}

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
constexpr uint32_t kBuiltMacosMin = availability_version(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__);
#else
constexpr uint32_t kBuiltMacosMin = 0;
#endif
#if defined(__APPLE__) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED)
constexpr uint32_t kBuiltMacosSdk = availability_version(__MAC_OS_X_VERSION_MAX_ALLOWED);
#else
constexpr uint32_t kBuiltMacosSdk = 0;
#endif
#if defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
constexpr uint32_t kBuiltIosMin = availability_version(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__);
#else
constexpr uint32_t kBuiltIosMin = 0;
#endif
#if defined(__APPLE__) && defined(__IPHONE_OS_VERSION_MAX_ALLOWED)
constexpr uint32_t kBuiltIosSdk = availability_version(__IPHONE_OS_VERSION_MAX_ALLOWED);
#else
constexpr uint32_t kBuiltIosSdk = 0;
#endif

uint32_t env_version(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return 0;
    return MachOBuildVersion::parse_version(v).value_or(0);
}

} // namespace

MachOBuildVersion MachOBuildVersion::resolve(const Target& target, const MachOBuildVersion& requested) {
    MachOBuildVersion out = requested;
    const bool macos = out.platform == macho::PLATFORM_MACOS;
    if (out.minos == 0) {
        out.minos = env_version(macos ? "MACOSX_DEPLOYMENT_TARGET" : "IPHONEOS_DEPLOYMENT_TARGET");
    }
    if (out.minos == 0) out.minos = macos ? kBuiltMacosMin : kBuiltIosMin;
    if (out.minos == 0) out.minos = !macos ? 0x000E0000u : target.is_aarch64() ? 0x000B0000u : 0x000A0F00u;
    if (macos && target.is_aarch64() && out.minos < 0x000B0000u) out.minos = 0x000B0000u;
    if (out.sdk == 0) out.sdk = macos ? kBuiltMacosSdk : kBuiltIosSdk;
    if (out.sdk < out.minos) out.sdk = out.minos;
    return out;
}

MachOWriter::MachOWriter(const ObjectFile& obj)
    : obj_(obj), build_version_(MachOBuildVersion::resolve(obj.target, MachOBuildVersion{})) {}

MachOWriter::MachOWriter(const ObjectFile& obj, const MachOBuildVersion& build_version)
    : obj_(obj), build_version_(MachOBuildVersion::resolve(obj.target, build_version)) {}

std::vector<uint8_t> MachOWriter::write() {
    ObjectFile working_obj = obj_;
    // Loads of the object's own symbols become `lea`s; the rest are
    // GOT_LOAD relocations for ld64 to bind (and relax if it can).
    relax_got_loads(working_obj);

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
            entry.segname = sec.relocations.empty() ? "__TEXT" : "__DATA";
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
            if (entry.data.size() >= 4 &&
                entry.data[entry.data.size() - 4] == 0 &&
                entry.data[entry.data.size() - 3] == 0 &&
                entry.data[entry.data.size() - 2] == 0 &&
                entry.data[entry.data.size() - 1] == 0) {
                entry.data.resize(entry.data.size() - 4);
            }
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
        if (sec_name_to_idx.count(name)) return;
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
    uint32_t build_version_cmd_size = 24;
    uint32_t sizeofcmds = segment_cmd_size + build_version_cmd_size + symtab_cmd_size + dysymtab_cmd_size;

    uint32_t cur_file_offset = 32 + sizeofcmds; // mach_header_64 (32 bytes) + cmds
    cur_file_offset = (cur_file_offset + 15) & ~15u; // 16-byte align section data

    uint64_t cur_vmaddr = 0;
    for (auto& s : macho_sections) {
        uint32_t sec_align = 1u << s.align_pow2;
        if (sec_align > 1) {
            cur_file_offset = (cur_file_offset + (sec_align - 1)) & ~(sec_align - 1);
            cur_vmaddr = (cur_vmaddr + (sec_align - 1)) & ~static_cast<uint64_t>(sec_align - 1);
        }
        s.offset = cur_file_offset;
        s.addr = cur_vmaddr;
        cur_file_offset += static_cast<uint32_t>(s.data.size());
        cur_vmaddr += s.data.size();
    }

    uint64_t seg_vmsize = cur_vmaddr;
    uint64_t seg_fileoff = macho_sections.empty() ? 0 : macho_sections[0].offset;
    uint64_t seg_filesize = macho_sections.empty() ? 0 : (cur_file_offset - seg_fileoff);

    for (auto& sym : all_symbols) {
        if (sym.n_sect >= 1 && sym.n_sect <= macho_sections.size()) {
            sym.n_value += macho_sections[sym.n_sect - 1].addr;
        }
    }

    // Pre-resolve PC-relative displacements in __eh_frame and clear relocations
    for (auto& s : macho_sections) {
        if (s.sectname == "__eh_frame") {
            uint64_t text_addr = 0;
            for (const auto& other : macho_sections) {
                if (other.sectname == "__text") {
                    text_addr = other.addr;
                    break;
                }
            }
            for (const auto& r : s.relocations) {
                int64_t target_addr = static_cast<int64_t>(text_addr) + r.addend;
                int64_t cur_addr = static_cast<int64_t>(s.addr + r.offset);
                int32_t disp = static_cast<int32_t>(target_addr - cur_addr);
                if (r.offset + 4 <= s.data.size()) {
                    s.data[r.offset + 0] = static_cast<uint8_t>(disp & 0xFF);
                    s.data[r.offset + 1] = static_cast<uint8_t>((disp >> 8) & 0xFF);
                    s.data[r.offset + 2] = static_cast<uint8_t>((disp >> 16) & 0xFF);
                    s.data[r.offset + 3] = static_cast<uint8_t>((disp >> 24) & 0xFF);
                }
            }
            s.relocations.clear();
        }
    }

    // Encode the relocation_info entries (an ARM64 addend is a separate
    // ARM64_RELOC_ADDEND entry in front of the one it modifies, so the count
    // is only known once they are encoded).
    const bool is_aarch64 = working_obj.target.is_aarch64();
    for (auto& s : macho_sections) {
        for (const auto& r : s.relocations) {
            encode_macho_reloc(s, r, is_aarch64, sec_name_to_idx, sym_name_to_idx, all_symbols);
        }
    }

    // Relocations offset
    for (auto& s : macho_sections) {
        if (!s.encoded_relocs.empty()) {
            cur_file_offset = (cur_file_offset + 7) & ~7u;
            s.reloff = cur_file_offset;
            s.nreloc = static_cast<uint32_t>(s.encoded_relocs.size());
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
    uint32_t cpu_type = working_obj.target.is_aarch64() ? static_cast<uint32_t>(macho::CPU_TYPE_ARM64) : static_cast<uint32_t>(macho::CPU_TYPE_X86_64);
    uint32_t cpu_subtype = working_obj.target.is_aarch64() ? static_cast<uint32_t>(macho::CPU_SUBTYPE_ARM64_ALL) : static_cast<uint32_t>(macho::CPU_SUBTYPE_X86_64_ALL);
    write_u32(out, cpu_type);
    write_u32(out, cpu_subtype);
    write_u32(out, macho::MH_OBJECT);
    write_u32(out, 4); // ncmds
    write_u32(out, sizeofcmds);
    write_u32(out, macho::MH_SUBSECTIONS_VIA_SYMBOLS); // flags
    write_u32(out, 0); // reserved

    // Write LC_SEGMENT_64
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

    // Write LC_BUILD_VERSION (24 bytes, no tools), where clang puts it
    write_u32(out, macho::LC_BUILD_VERSION);
    write_u32(out, build_version_cmd_size);
    write_u32(out, build_version_.platform);
    write_u32(out, build_version_.minos);
    write_u32(out, build_version_.sdk);
    write_u32(out, 0); // ntools

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
        align_buf(out, static_cast<size_t>(1ULL << s.align_pow2));
        if (!s.data.empty()) {
            write_bytes(out, s.data.data(), s.data.size());
        }
    }

    // Write Relocations
    for (const auto& s : macho_sections) {
        if (s.encoded_relocs.empty()) continue;
        align_buf(out, 8);
        for (const auto& [r_address, word2] : s.encoded_relocs) {
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
