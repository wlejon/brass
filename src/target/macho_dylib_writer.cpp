#include <brass/target/macho_dylib_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include "image_file.hpp"
#include "image_util.hpp"
#include "import_plan.hpp"
#include "macho_dyld_info.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string_view>

// A Mach-O dylib from one object file.
//
//   __TEXT        __text, __stubs
//   __DATA_CONST  __got, __const          (SG_READ_ONLY: writable while dyld
//                                          binds, read-only afterwards)
//   __DATA        __data
//   __LINKEDIT    rebase + bind opcodes, export trie, symbol table,
//                 indirect symbol table, string table
//
// Read-only data goes to __DATA_CONST rather than __TEXT because it carries
// absolute pointers dyld has to slide and bind, and __TEXT is never
// writable. Imports take one non-lazy pointer in __got, bound at load time,
// and one stub in __stubs that jumps through it; a data reference to an
// import is a bind entry on the word itself.

namespace brass::target {

using namespace brass::object;
using namespace brass::target::image;

namespace {

constexpr uint32_t LC_RPATH = 0x8000001c;
constexpr uint32_t LC_UUID = 0x1b;
constexpr uint32_t SG_READ_ONLY = 0x10;
constexpr uint32_t S_NON_LAZY_SYMBOL_POINTERS = 0x6;
constexpr uint32_t S_SYMBOL_STUBS = 0x8;

std::string macho_name(const std::string& name) {
    if (name.empty() || name[0] == '_') return name;
    return "_" + name;
}

// A load command carrying a trailing string: the fixed part, the string
// with its terminator, then padding to an 8-byte boundary.
uint32_t name_cmd_size(uint32_t header, const std::string& name) {
    return static_cast<uint32_t>(align_up(header + name.size() + 1, 8));
}

// One section of the image, wherever it landed.
struct Placed {
    std::string source;      // object section name it came from ("" for synthetic)
    std::string sectname;
    std::string segname;
    uint8_t segment = 0;     // LC_SEGMENT_64 index
    uint8_t ordinal = 0;     // 1-based section number
    uint64_t vaddr = 0;
    uint32_t fileoff = 0;
    uint32_t align_log2 = 4;
    uint32_t flags = macho::S_REGULAR;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    std::vector<uint8_t> data;
    std::vector<ObjectRelocation> relocations;
};

struct Segment {
    std::string name;
    uint64_t vmaddr = 0;
    uint64_t vmsize = 0;
    uint32_t fileoff = 0;
    uint32_t filesize = 0;
    int32_t prot = 0;
    uint32_t flags = 0;
    std::vector<size_t> sections;   // indices into the Placed list
};

struct Nlist64 {
    uint32_t n_strx = 0;
    uint8_t n_type = 0;
    uint8_t n_sect = 0;
    uint16_t n_desc = 0;
    uint64_t n_value = 0;
};

} // namespace

MachODylibWriter::MachODylibWriter(const ObjectFile& obj, const MachODylibOptions& options)
    : obj_(obj), options_(options) {}

MachODylibWriter::MachODylibWriter(const ObjectFile& obj)
    : obj_(obj), options_() {}

std::vector<uint8_t> MachODylibWriter::write() {
    error_.clear();
    ObjectFile working_obj = obj_;
    const bool aarch64 = working_obj.target.is_aarch64();
    const uint32_t page_size = aarch64 ? 16384u : 4096u;
    // Loads of the object's own symbols become `lea`s; what is left loads
    // an import's __got entry. That is what keeps __TEXT free of absolute
    // words: dyld slides the image without writing into it.
    relax_got_loads(working_obj);
    const uint64_t base = options_.image_base;

    // ---- the object's sections ---------------------------------------------
    const Section* text_sec = working_obj.get_section(".text");
    const Section* const_sec = working_obj.get_section(".rodata");
    if (!const_sec) const_sec = working_obj.get_section(".rdata");
    if (!const_sec) const_sec = working_obj.get_section("__const");
    if (!const_sec) {
        for (const auto& s : working_obj.sections) {
            if (s.kind == SectionKind::RoData) { const_sec = &s; break; }
        }
    }
    const Section* data_sec = working_obj.get_section(".data");
    if (!data_sec) {
        for (const auto& s : working_obj.sections) {
            if (s.kind == SectionKind::Data) { data_sec = &s; break; }
        }
    }
    const bool has_const = const_sec && !const_sec->data.empty();
    const bool has_data = data_sec && !data_sec->data.empty();

    // ---- exports and imports -----------------------------------------------
    auto should_export = [&](const ObjectSymbol& sym) -> bool {
        if (sym.section_index < 0 || sym.binding != SymbolBinding::Global) return false;
        if (options_.export_all_functions && sym.type == SymbolType::Function) return true;
        for (const auto& exp : options_.explicit_exports) {
            if (exp == sym.name || macho_name(exp) == sym.name || exp == macho_name(sym.name)) return true;
        }
        return false;
    };
    for (const auto& exp : options_.explicit_exports) {
        bool found = false;
        for (const auto& sym : working_obj.symbols) {
            if (should_export(sym) && (exp == sym.name || macho_name(exp) == sym.name || exp == macho_name(sym.name))) {
                found = true;
                break;
            }
        }
        if (!found) {
            error_ = "cannot export '" + exp + "': the object does not define it";
            return {};
        }
    }

    std::vector<std::string_view> placed_names = {".text"};
    if (has_const) placed_names.push_back(const_sec->name);
    if (has_data) placed_names.push_back(data_sec->name);
    imports::Plan imports;
    if (!imports::plan(working_obj, placed_names, options_.imports, imports, error_)) return {};
    const size_t nimports = imports.symbols.size();
    const uint32_t stub_size = aarch64 ? 12u : 6u;

    // ---- sections and segments ---------------------------------------------
    std::vector<Placed> placed;
    std::vector<Segment> segments;
    auto add_segment = [&](const char* name, int32_t prot, uint32_t flags) -> size_t {
        Segment s;
        s.name = name;
        s.prot = prot;
        s.flags = flags;
        segments.push_back(std::move(s));
        return segments.size() - 1;
    };
    auto add_section = [&](size_t seg, const char* sectname, uint32_t flags, uint32_t align_log2,
                           const Section* src) -> size_t {
        Placed p;
        p.segname = segments[seg].name;
        p.sectname = sectname;
        p.segment = static_cast<uint8_t>(seg);
        p.flags = flags;
        p.align_log2 = align_log2;
        if (src) {
            p.source = src->name;
            p.data = src->data;
            p.relocations = src->relocations;
        }
        placed.push_back(std::move(p));
        segments[seg].sections.push_back(placed.size() - 1);
        placed.back().ordinal = static_cast<uint8_t>(placed.size());
        return placed.size() - 1;
    };

    const size_t text_seg = add_segment("__TEXT", macho::VM_PROT_READ | macho::VM_PROT_EXECUTE, 0);
    Section dummy_text;
    add_section(text_seg, "__text",
                macho::S_REGULAR | macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS,
                4, text_sec ? text_sec : &dummy_text);
    size_t stubs_idx = SIZE_MAX;
    if (nimports) {
        stubs_idx = add_section(text_seg, "__stubs",
                                S_SYMBOL_STUBS | macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS,
                                aarch64 ? 2 : 1, nullptr);
        placed[stubs_idx].reserved1 = 0;           // first indirect symbol
        placed[stubs_idx].reserved2 = stub_size;
        placed[stubs_idx].data.assign(nimports * stub_size, 0);
    }
    size_t got_idx = SIZE_MAX;
    if (nimports || has_const) {
        const size_t const_seg =
            add_segment("__DATA_CONST", macho::VM_PROT_READ | macho::VM_PROT_WRITE, SG_READ_ONLY);
        if (nimports) {
            got_idx = add_section(const_seg, "__got", S_NON_LAZY_SYMBOL_POINTERS, 3, nullptr);
            placed[got_idx].reserved1 = static_cast<uint32_t>(nimports);   // after the stubs' entries
            placed[got_idx].data.assign(nimports * 8, 0);
        }
        if (has_const) add_section(const_seg, "__const", macho::S_REGULAR, 4, const_sec);
    }
    if (has_data) {
        const size_t data_seg = add_segment("__DATA", macho::VM_PROT_READ | macho::VM_PROT_WRITE, 0);
        add_section(data_seg, "__data", macho::S_REGULAR, 4, data_sec);
    }
    const size_t linkedit_seg = add_segment("__LINKEDIT", macho::VM_PROT_READ, 0);

    // ---- load command sizes ------------------------------------------------
    const std::string install_name = options_.install_name.empty() ? "brass_module.dylib" : options_.install_name;
    // libSystem is loaded whether or not anything binds to it, unless the
    // imports already name it — once per library is all dyld expects.
    const std::string libsystem = "/usr/lib/libSystem.B.dylib";
    const bool add_libsystem =
        std::find(imports.libraries.begin(), imports.libraries.end(), libsystem) == imports.libraries.end();
    uint32_t sizeofcmds = 0;
    uint32_t ncmds = 0;
    for (const auto& seg : segments) {
        sizeofcmds += 72 + 80 * static_cast<uint32_t>(seg.sections.size());
        ++ncmds;
    }
    sizeofcmds += name_cmd_size(24, install_name); ++ncmds;
    for (const auto& lib : imports.libraries) { sizeofcmds += name_cmd_size(24, lib); ++ncmds; }
    if (add_libsystem) { sizeofcmds += name_cmd_size(24, libsystem); ++ncmds; }
    for (const auto& rp : options_.rpaths) { sizeofcmds += name_cmd_size(12, rp); ++ncmds; }
    sizeofcmds += 48 + 24 + 80 + 24 + 24;   // DYLD_INFO_ONLY, SYMTAB, DYSYMTAB, BUILD_VERSION, UUID
    ncmds += 5;

    // ---- layout ------------------------------------------------------------
    // Room after the load commands so codesign can append LC_CODE_SIGNATURE
    // without moving __text.
    uint32_t cursor = std::max(1024u, static_cast<uint32_t>(align_up(32 + sizeofcmds + 256, 16)));
    for (size_t s = 0; s < segments.size(); ++s) {
        Segment& seg = segments[s];
        if (s == linkedit_seg) break;
        seg.fileoff = (s == text_seg) ? 0 : cursor;
        seg.vmaddr = base + seg.fileoff;
        for (size_t idx : seg.sections) {
            Placed& p = placed[idx];
            cursor = static_cast<uint32_t>(align_up(cursor, uint64_t{1} << p.align_log2));
            p.fileoff = cursor;
            p.vaddr = base + cursor;
            cursor += static_cast<uint32_t>(p.data.size());
        }
        cursor = static_cast<uint32_t>(align_up(cursor, page_size));
        seg.filesize = cursor - seg.fileoff;
        seg.vmsize = seg.filesize;
    }
    segments[linkedit_seg].fileoff = cursor;
    segments[linkedit_seg].vmaddr = base + cursor;

    auto placed_for = [&](std::string_view source) -> const Placed* {
        for (const auto& p : placed) {
            if (!p.source.empty() && p.source == source) return &p;
        }
        return nullptr;
    };
    auto symbol_vaddr = [&](const ObjectSymbol& sym, uint64_t& out) -> bool {
        const auto& src_sec = working_obj.sections[static_cast<size_t>(sym.section_index)];
        const Placed* p = placed_for(src_sec.name);
        if (!p) return false;
        out = p->vaddr + sym.value;
        return true;
    };
    auto stub_vaddr = [&](size_t i) { return placed[stubs_idx].vaddr + i * stub_size; };
    auto got_vaddr = [&](size_t i) { return placed[got_idx].vaddr + i * 8; };

    // ---- stubs and GOT binds -----------------------------------------------
    std::vector<macho_dyld::RebaseEntry> rebases;
    std::vector<macho_dyld::BindEntry> binds;
    auto ordinal_of = [&](size_t library) { return static_cast<uint32_t>(library + 1); };
    for (const auto& imp : imports.symbols) {
        Placed& stubs = placed[stubs_idx];
        const size_t at = imp.index * stub_size;
        const uint64_t pc = stub_vaddr(imp.index);
        const uint64_t slot = got_vaddr(imp.index);
        if (aarch64) {
            patch_u32(stubs.data, at + 0, aarch64_adrp(16, pc, slot));
            patch_u32(stubs.data, at + 4, aarch64_ldr_x_uoff(16, 16, slot));
            patch_u32(stubs.data, at + 8, aarch64_br(16));
        } else {
            stubs.data[at] = 0xFF;
            stubs.data[at + 1] = 0x25;
            patch_u32(stubs.data, at + 2, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int64_t>(slot) - static_cast<int64_t>(pc + 6))));
        }
        binds.push_back({placed[got_idx].segment, slot - segments[placed[got_idx].segment].vmaddr,
                         ordinal_of(imp.library), macho_name(imp.name), 0});
    }

    // ---- relocations -------------------------------------------------------
    for (Placed& p : placed) {
        if (p.source.empty()) continue;
        for (const auto& r : p.relocations) {
            uint64_t target = 0;
            const imports::Imported* imp = nullptr;
            const auto* sym = working_obj.find_symbol(r.symbol_name);
            const bool got_load = r.kind == RelocKind::GotPCRel32;
            if (sym && sym->section_index >= 0) {
                if (!symbol_vaddr(*sym, target)) {
                    error_ = "relocation against '" + r.symbol_name + "' names a section that is not placed in the image";
                    return {};
                }
            } else if (const Placed* ps = placed_for(r.symbol_name)) {
                target = ps->vaddr;
            } else if ((imp = imports.find(r.symbol_name)) != nullptr) {
                // A GOT load reads the __got entry dyld binds, which is the
                // import's own address; every other reference takes the stub.
                target = got_load ? got_vaddr(imp->index) : stub_vaddr(imp->index);
            } else {
                error_ = "unresolved symbol '" + r.symbol_name + "' referenced from " + p.source;
                return {};
            }
            if (got_load && !imp) {
                error_ = "internal: GOT load of '" + r.symbol_name + "' survived relaxation without an import";
                return {};
            }

            const uint64_t at = p.vaddr + r.offset;
            const bool fits4 = r.offset + 4 <= p.data.size();
            if (r.kind == RelocKind::Abs64) {
                if (r.offset + 8 > p.data.size()) continue;
                if (p.segment == text_seg) {
                    error_ = "absolute pointer to '" + r.symbol_name + "' in __TEXT cannot be slid by dyld; the code must be position independent";
                    return {};
                }
                const uint64_t seg_off = at - segments[p.segment].vmaddr;
                if (imp) {
                    patch_u64(p.data, r.offset, 0);
                    binds.push_back({p.segment, seg_off, ordinal_of(imp->library), macho_name(imp->name), r.addend});
                } else {
                    patch_u64(p.data, r.offset, target + static_cast<uint64_t>(r.addend));
                    rebases.push_back({p.segment, seg_off});
                }
            } else if (!fits4) {
                continue;
            } else if (aarch64) {
                uint32_t inst = read_u32(p.data, r.offset);
                if (r.kind == RelocKind::Plt32) {
                    const int64_t disp = static_cast<int64_t>(target) + r.addend - static_cast<int64_t>(at);
                    inst = (inst & 0xFC000000u) | (static_cast<uint32_t>(disp >> 2) & 0x03FFFFFFu);
                } else if (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::AdrPage21) {
                    inst = (inst & 0x9F00001Fu) | (aarch64_adrp(0, at, target) & 0x60FFFFE0u);
                } else if (r.kind == RelocKind::SecRel32) {
                    inst = (inst & 0xFFC003FFu) | (static_cast<uint32_t>(target & 0xFFFu) << 10);
                } else if (r.kind == RelocKind::Abs32 || r.kind == RelocKind::Addr32NB) {
                    inst = static_cast<uint32_t>(target + static_cast<uint64_t>(r.addend));
                } else {
                    continue;
                }
                patch_u32(p.data, r.offset, inst);
            } else if (r.kind == RelocKind::PCRel32 || r.kind == RelocKind::Plt32 || got_load) {
                // disp = S + A - P, the addend carrying the instruction tail.
                const int64_t disp = static_cast<int64_t>(target) + r.addend - static_cast<int64_t>(at);
                patch_u32(p.data, r.offset, static_cast<uint32_t>(static_cast<int32_t>(disp)));
            } else if (r.kind == RelocKind::SecRel32) {
                patch_u32(p.data, r.offset, static_cast<uint32_t>(static_cast<int64_t>(sym ? sym->value : 0) + r.addend));
            } else if (r.kind == RelocKind::Abs32 || r.kind == RelocKind::Addr32NB) {
                patch_u32(p.data, r.offset, static_cast<uint32_t>(target + static_cast<uint64_t>(r.addend)));
            }
        }
    }

    // ---- symbols -----------------------------------------------------------
    std::vector<uint8_t> strtab;
    strtab.push_back(0);
    auto add_str = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        uint32_t off = static_cast<uint32_t>(strtab.size());
        write_cstring(strtab, s);
        return off;
    };

    std::vector<std::pair<std::string, uint64_t>> exported;
    std::vector<Nlist64> locals, extdefs, undefs;
    for (const auto& sym : working_obj.symbols) {
        if (sym.section_index < 0) continue;
        uint64_t vaddr = 0;
        if (!symbol_vaddr(sym, vaddr)) continue;
        const auto& src_sec = working_obj.sections[static_cast<size_t>(sym.section_index)];
        const Placed* ps = placed_for(src_sec.name);
        Nlist64 nl;
        nl.n_sect = ps->ordinal;
        nl.n_value = vaddr;
        if (should_export(sym)) {
            const std::string underscored = macho_name(sym.name);
            exported.emplace_back(underscored, vaddr - base);
            if (underscored != sym.name) exported.emplace_back(sym.name, vaddr - base);
            nl.n_strx = add_str(underscored);
            nl.n_type = macho::N_SECT | macho::N_EXT;
            extdefs.push_back(nl);
        } else if (sym.type == SymbolType::Function) {
            nl.n_strx = add_str(macho_name(sym.name));
            nl.n_type = macho::N_SECT;
            locals.push_back(nl);
        }
    }
    for (const auto& imp : imports.symbols) {
        Nlist64 nl;
        nl.n_strx = add_str(macho_name(imp.name));
        nl.n_type = macho::N_UNDF | macho::N_EXT;
        nl.n_sect = macho::NO_SECT;
        nl.n_desc = static_cast<uint16_t>(ordinal_of(imp.library) << 8);
        undefs.push_back(nl);
    }
    const uint32_t nlocalsym = static_cast<uint32_t>(locals.size());
    const uint32_t nextdefsym = static_cast<uint32_t>(extdefs.size());
    const uint32_t iundefsym = nlocalsym + nextdefsym;
    const uint32_t nundefsym = static_cast<uint32_t>(undefs.size());
    std::vector<Nlist64> all_nlists;
    all_nlists.insert(all_nlists.end(), locals.begin(), locals.end());
    all_nlists.insert(all_nlists.end(), extdefs.begin(), extdefs.end());
    all_nlists.insert(all_nlists.end(), undefs.begin(), undefs.end());
    // The indirect symbol table: the stubs' entries, then the GOT's, each
    // naming the undefined symbol of the import at that position.
    std::vector<uint8_t> indirect;
    for (int pass = 0; pass < 2 && nimports; ++pass) {
        for (size_t i = 0; i < nimports; ++i) write_u32(indirect, iundefsym + static_cast<uint32_t>(i));
    }

    // ---- __LINKEDIT --------------------------------------------------------
    const std::vector<uint8_t> rebase_ops = macho_dyld::build_rebase_opcodes(std::move(rebases));
    const std::vector<uint8_t> bind_ops = macho_dyld::build_bind_opcodes(std::move(binds));
    const std::vector<uint8_t> export_trie = macho_dyld::build_export_trie(exported);

    uint32_t le = segments[linkedit_seg].fileoff;
    const uint32_t rebase_off = le; le += static_cast<uint32_t>(rebase_ops.size());
    const uint32_t bind_off = le; le += static_cast<uint32_t>(bind_ops.size());
    const uint32_t export_off = le; le += static_cast<uint32_t>(export_trie.size());
    le = static_cast<uint32_t>(align_up(le, 8));
    const uint32_t symoff = le; le += static_cast<uint32_t>(all_nlists.size() * 16);
    const uint32_t indirectoff = le; le += static_cast<uint32_t>(indirect.size());
    le = static_cast<uint32_t>(align_up(le, 8));
    const uint32_t stroff = le; le += static_cast<uint32_t>(strtab.size());
    segments[linkedit_seg].filesize = le - segments[linkedit_seg].fileoff;
    segments[linkedit_seg].vmsize = align_up(segments[linkedit_seg].filesize, page_size);

    // ---- assemble ----------------------------------------------------------
    std::vector<uint8_t> out(le, 0);
    patch_u32(out, 0, macho::MH_MAGIC_64);
    patch_u32(out, 4, static_cast<uint32_t>(aarch64 ? macho::CPU_TYPE_ARM64 : macho::CPU_TYPE_X86_64));
    patch_u32(out, 8, static_cast<uint32_t>(aarch64 ? macho::CPU_SUBTYPE_ARM64_ALL : macho::CPU_SUBTYPE_X86_64_ALL));
    patch_u32(out, 12, macho::MH_DYLIB);
    patch_u32(out, 16, ncmds);
    patch_u32(out, 20, sizeofcmds);
    patch_u32(out, 24, macho::MH_DYLDLINK | macho::MH_TWOLEVEL | (nimports ? 0u : macho::MH_NOUNDEFS));
    patch_u32(out, 28, 0);

    std::vector<uint8_t> cmds;
    for (const auto& seg : segments) {
        write_u32(cmds, macho::LC_SEGMENT_64);
        write_u32(cmds, 72 + 80 * static_cast<uint32_t>(seg.sections.size()));
        write_fixed_string(cmds, seg.name, 16);
        write_u64(cmds, seg.vmaddr);
        write_u64(cmds, seg.vmsize);
        write_u64(cmds, seg.fileoff);
        write_u64(cmds, seg.filesize);
        write_u32(cmds, static_cast<uint32_t>(seg.prot));   // maxprot
        write_u32(cmds, static_cast<uint32_t>(seg.prot));   // initprot
        write_u32(cmds, static_cast<uint32_t>(seg.sections.size()));
        write_u32(cmds, seg.flags);
        for (size_t idx : seg.sections) {
            const Placed& p = placed[idx];
            write_fixed_string(cmds, p.sectname, 16);
            write_fixed_string(cmds, p.segname, 16);
            write_u64(cmds, p.vaddr);
            write_u64(cmds, p.data.size());
            write_u32(cmds, p.fileoff);
            write_u32(cmds, p.align_log2);
            write_u32(cmds, 0);   // reloff
            write_u32(cmds, 0);   // nreloc
            write_u32(cmds, p.flags);
            write_u32(cmds, p.reserved1);
            write_u32(cmds, p.reserved2);
            write_u32(cmds, 0);   // reserved3
        }
    }
    auto dylib_cmd = [&](uint32_t cmd, const std::string& name, uint32_t timestamp, uint32_t version) {
        write_u32(cmds, cmd);
        write_u32(cmds, name_cmd_size(24, name));
        write_u32(cmds, 24);
        write_u32(cmds, timestamp);
        write_u32(cmds, version);
        write_u32(cmds, 0x00010000);   // compatibility version 1.0.0
        write_cstring(cmds, name);
        pad_to(cmds, 8);
    };
    dylib_cmd(macho::LC_ID_DYLIB, install_name, 1, 0x00010000);
    // Ordinals count LC_LOAD_DYLIBs in order: the import libraries first, so
    // library i is ordinal i+1, then libSystem.
    for (const auto& lib : imports.libraries) {
        dylib_cmd(macho::LC_LOAD_DYLIB, lib, 2, lib == libsystem ? 0x05470000 : 0x00010000);
    }
    if (add_libsystem) dylib_cmd(macho::LC_LOAD_DYLIB, libsystem, 2, 0x05470000);
    for (const auto& rp : options_.rpaths) {
        write_u32(cmds, LC_RPATH);
        write_u32(cmds, name_cmd_size(12, rp));
        write_u32(cmds, 12);
        write_cstring(cmds, rp);
        pad_to(cmds, 8);
    }

    write_u32(cmds, macho::LC_DYLD_INFO_ONLY);
    write_u32(cmds, 48);
    write_u32(cmds, rebase_ops.empty() ? 0 : rebase_off);
    write_u32(cmds, static_cast<uint32_t>(rebase_ops.size()));
    write_u32(cmds, bind_ops.empty() ? 0 : bind_off);
    write_u32(cmds, static_cast<uint32_t>(bind_ops.size()));
    write_u32(cmds, 0); write_u32(cmds, 0);   // weak bind
    write_u32(cmds, 0); write_u32(cmds, 0);   // lazy bind
    write_u32(cmds, export_trie.empty() ? 0 : export_off);
    write_u32(cmds, static_cast<uint32_t>(export_trie.size()));

    write_u32(cmds, macho::LC_SYMTAB);
    write_u32(cmds, 24);
    write_u32(cmds, symoff);
    write_u32(cmds, static_cast<uint32_t>(all_nlists.size()));
    write_u32(cmds, stroff);
    write_u32(cmds, static_cast<uint32_t>(strtab.size()));

    write_u32(cmds, macho::LC_DYSYMTAB);
    write_u32(cmds, 80);
    write_u32(cmds, 0);            // ilocalsym
    write_u32(cmds, nlocalsym);
    write_u32(cmds, nlocalsym);    // iextdefsym
    write_u32(cmds, nextdefsym);
    write_u32(cmds, iundefsym);
    write_u32(cmds, nundefsym);
    for (int i = 0; i < 6; ++i) write_u32(cmds, 0);   // toc, modtab, extrefsyms
    write_u32(cmds, indirect.empty() ? 0 : indirectoff);
    write_u32(cmds, static_cast<uint32_t>(indirect.size() / 4));
    for (int i = 0; i < 4; ++i) write_u32(cmds, 0);   // extrel, locrel

    write_u32(cmds, macho::LC_BUILD_VERSION);
    write_u32(cmds, 24);
    write_u32(cmds, 1);            // PLATFORM_MACOS
    write_u32(cmds, 0x000b0000);   // minos 11.0
    write_u32(cmds, 0x000e0000);   // sdk 14.0
    write_u32(cmds, 0);            // ntools

    write_u32(cmds, LC_UUID);
    write_u32(cmds, 24);
    const uint8_t uuid_bytes[16] = {0x42, 0x52, 0x41, 0x53, 0x53, 0x2d, 0x44, 0x59,
                                    0x4c, 0x49, 0x42, 0x2d, 0x58, 0x36, 0x34, 0x01};
    write_bytes(cmds, uuid_bytes, 16);

    if (cmds.size() != sizeofcmds) {
        error_ = "internal: load command size mismatch";
        return {};
    }
    std::memcpy(out.data() + 32, cmds.data(), cmds.size());

    for (const Placed& p : placed) {
        if (!p.data.empty()) std::memcpy(out.data() + p.fileoff, p.data.data(), p.data.size());
    }
    if (!rebase_ops.empty()) std::memcpy(out.data() + rebase_off, rebase_ops.data(), rebase_ops.size());
    if (!bind_ops.empty()) std::memcpy(out.data() + bind_off, bind_ops.data(), bind_ops.size());
    if (!export_trie.empty()) std::memcpy(out.data() + export_off, export_trie.data(), export_trie.size());
    uint8_t* sym_ptr = out.data() + symoff;
    for (const auto& nl : all_nlists) {
        std::memcpy(sym_ptr + 0, &nl.n_strx, 4);
        sym_ptr[4] = nl.n_type;
        sym_ptr[5] = nl.n_sect;
        std::memcpy(sym_ptr + 6, &nl.n_desc, 2);
        std::memcpy(sym_ptr + 8, &nl.n_value, 8);
        sym_ptr += 16;
    }
    if (!indirect.empty()) std::memcpy(out.data() + indirectoff, indirect.data(), indirect.size());
    std::memcpy(out.data() + stroff, strtab.data(), strtab.size());

    return out;
}

bool MachODylibWriter::write_to_file(const std::string& path) {
    auto data = write();
    if (data.empty()) return false;
    if (!image::write_image_file(path, data, &error_)) return false;

#if defined(__APPLE__)
    std::string cmd = "codesign -s - -f \"" + path + "\" > /dev/null 2>&1";
    int res = std::system(cmd.c_str());
    (void)res;
#endif

    return true;
}

std::vector<uint8_t> MachODylibWriter::emit(const ObjectFile& obj, const MachODylibOptions& options,
                                            std::string* error_out) {
    MachODylibWriter writer(obj, options);
    std::vector<uint8_t> out = writer.write();
    if (error_out) *error_out = writer.error();
    return out;
}

std::vector<uint8_t> MachODylibWriter::emit(const ObjectFile& obj, const MachODylibOptions& options) {
    return emit(obj, options, nullptr);
}

std::vector<uint8_t> MachODylibWriter::emit(const ObjectFile& obj) {
    return emit(obj, MachODylibOptions(), nullptr);
}

} // namespace brass::target
