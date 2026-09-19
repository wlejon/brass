// The import tables of the three shared-image writers: a module that calls
// a function it does not define links against a library that provides it,
// and links to nothing without one.

#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <brass/target/pe_dll_writer.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::target;

namespace {

uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// twice(n) = n + n
std::unique_ptr<Module> build_provider() {
    auto mod = std::make_unique<Module>("aot_import_provider");
    Builder b(*mod);
    Function* fn = mod->create_function("twice", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    b.build_ret(b.build_add(n, n));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    return mod;
}

// quad(n) = twice(twice(n)), with twice imported.
std::unique_ptr<Module> build_consumer() {
    auto mod = std::make_unique<Module>("aot_import_consumer");
    mod->add_external_symbol("twice");
    Builder b(*mod);
    Function* fn = mod->create_function("quad", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* once = b.build_call("twice", Type::i64(), {n});
    Value* again = b.build_call("twice", Type::i64(), {once});
    b.build_ret(again);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    return mod;
}

LinkerOptions consumer_options(const std::string& library) {
    LinkerOptions opts;
    opts.export_all_functions = true;
    opts.imports.push_back({library, {"twice"}});
    return opts;
}

} // namespace

TEST_CASE("AOT Linker - Unresolved Symbol Is a Link Error On Every Format") {
    auto consumer = build_consumer();
    const Target targets[] = {Target::x64_windows(), Target::x64_linux(), Target::x64_macos(),
                              Target::aarch64_linux(), Target::aarch64_macos(), Target::aarch64_windows()};
    for (const Target& t : targets) {
        object::ObjectFile obj = object::compile_module_to_object(*consumer, t);
        LinkerOptions opts;
        opts.export_all_functions = true;
        std::string err;
        std::vector<uint8_t> image = AotLinker::link(obj, opts, &err);
        CHECK(image.empty());
        CHECK(err.find("twice") != std::string::npos);

        // A library that does not list the symbol is no better.
        opts.imports.push_back({"other.dll", {"unrelated"}});
        err.clear();
        image = AotLinker::link(obj, opts, &err);
        CHECK(image.empty());
        CHECK(err.find("twice") != std::string::npos);
    }
}

TEST_CASE("AOT Linker - PE Import Directory, IAT and Thunks") {
    auto consumer = build_consumer();
    object::ObjectFile obj = object::compile_module_to_object(*consumer, Target::x64_windows());
    std::string err;
    std::vector<uint8_t> dll = AotLinker::link(obj, consumer_options("aot_import_provider.dll"), &err);
    if (dll.empty()) std::fprintf(stderr, "PE link failed: %s\n", err.c_str());
    REQUIRE(!dll.empty());
    CHECK(err.empty());

    const uint32_t pe_off = rd32(dll.data() + 0x3C);
    const uint8_t* file_hdr = dll.data() + pe_off + 4;
    const uint16_t nsections = rd16(file_hdr + 2);
    const uint16_t opt_size = rd16(file_hdr + 16);
    const uint8_t* opt = file_hdr + 20;
    const uint8_t* dirs = opt + 112;
    const uint32_t import_rva = rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IMPORT * 8);
    const uint32_t import_size = rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IMPORT * 8 + 4);
    const uint32_t iat_rva = rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IAT * 8);
    const uint32_t iat_size = rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IAT * 8 + 4);
    REQUIRE(import_rva > 0);
    CHECK_EQ(import_size, 40u);   // one descriptor + the null terminator
    REQUIRE(iat_rva > 0);
    CHECK_EQ(iat_size, 16u);      // one slot + the null terminator

    // RVA -> file offset through the section table.
    struct Sec { uint32_t va, vsize, raw, rawsize; std::string name; };
    std::vector<Sec> secs;
    const uint8_t* sh = opt + opt_size;
    for (uint16_t i = 0; i < nsections; ++i, sh += 40) {
        char name[9] = {0};
        std::memcpy(name, sh, 8);
        secs.push_back({rd32(sh + 12), rd32(sh + 8), rd32(sh + 20), rd32(sh + 16), name});
    }
    auto to_off = [&](uint32_t rva) -> const uint8_t* {
        for (const auto& s : secs) {
            if (rva >= s.va && rva < s.va + std::max(s.vsize, s.rawsize)) return dll.data() + s.raw + (rva - s.va);
        }
        return nullptr;
    };
    bool has_idata = false, has_text = false;
    for (const auto& s : secs) {
        if (s.name == ".idata") has_idata = true;
        if (s.name == ".text") has_text = true;
    }
    CHECK(has_idata);
    CHECK(has_text);

    const uint8_t* desc = to_off(import_rva);
    REQUIRE(desc != nullptr);
    const uint32_t ilt_rva = rd32(desc + 0);
    const uint32_t name_rva = rd32(desc + 12);
    const uint32_t first_thunk = rd32(desc + 16);
    CHECK_EQ(first_thunk, iat_rva);
    const uint8_t* libname = to_off(name_rva);
    REQUIRE(libname != nullptr);
    CHECK_EQ(std::string(reinterpret_cast<const char*>(libname)), "aot_import_provider.dll");

    const uint8_t* ilt = to_off(ilt_rva);
    const uint8_t* iat = to_off(iat_rva);
    REQUIRE(ilt != nullptr);
    REQUIRE(iat != nullptr);
    const uint64_t entry = rd64(ilt);
    CHECK_EQ(entry, rd64(iat));          // unbound: IAT mirrors the ILT
    CHECK_EQ(rd64(ilt + 8), 0ULL);       // terminator
    CHECK((entry >> 63) == 0);           // by name, not ordinal
    const uint8_t* hint_name = to_off(static_cast<uint32_t>(entry));
    REQUIRE(hint_name != nullptr);
    CHECK_EQ(std::string(reinterpret_cast<const char*>(hint_name + 2)), "twice");

    // The thunk lives in .text and is `jmp qword ptr [rip + disp32]` onto
    // the IAT slot. Find it by scanning the text section.
    const Sec* text = nullptr;
    for (const auto& s : secs) if (s.name == ".text") text = &s;
    REQUIRE(text != nullptr);
    bool found_thunk = false;
    for (uint32_t at = 0; at + 6 <= text->rawsize; at += 2) {
        const uint8_t* p = dll.data() + text->raw + at;
        if (p[0] != 0xFF || p[1] != 0x25) continue;
        const int32_t disp = static_cast<int32_t>(rd32(p + 2));
        const uint64_t target = static_cast<uint64_t>(static_cast<int64_t>(text->va + at + 6) + disp);
        if (target == iat_rva) { found_thunk = true; break; }
    }
    CHECK(found_thunk);

    // .idata is writable so the loader can fill the IAT.
    for (const auto& s : secs) {
        if (s.name != ".idata") continue;
        const uint8_t* hdr = opt + opt_size;
        for (uint16_t i = 0; i < nsections; ++i, hdr += 40) {
            if (std::strncmp(reinterpret_cast<const char*>(hdr), ".idata", 8) == 0) {
                CHECK((rd32(hdr + 36) & brass::object::coff::IMAGE_SCN_MEM_WRITE) != 0);
            }
        }
    }
}

TEST_CASE("AOT Linker - ELF DT_NEEDED, Undefined dynsym and GLOB_DAT Relocation") {
    auto consumer = build_consumer();
    object::ObjectFile obj = object::compile_module_to_object(*consumer, Target::x64_linux());
    LinkerOptions opts = consumer_options("libaot_import_provider.so");
    opts.rpaths.push_back("$ORIGIN");
    std::string err;
    std::vector<uint8_t> so = AotLinker::link(obj, opts, &err);
    if (so.empty()) std::fprintf(stderr, "ELF link failed: %s\n", err.c_str());
    REQUIRE(!so.empty());
    CHECK(err.empty());

    const uint64_t shoff = rd64(so.data() + 40);
    const uint16_t shentsize = rd16(so.data() + 58);
    const uint16_t shnum = rd16(so.data() + 60);
    const uint16_t shstrndx = rd16(so.data() + 62);
    REQUIRE(shentsize == 64);
    auto shdr = [&](uint16_t i) { return so.data() + shoff + i * 64ULL; };
    const uint8_t* shstr = so.data() + rd64(shdr(shstrndx) + 24);
    const uint8_t* dynamic = nullptr; uint64_t dynamic_size = 0;
    const uint8_t* dynstr = nullptr;
    const uint8_t* dynsym = nullptr; uint64_t dynsym_size = 0;
    const uint8_t* rela = nullptr; uint64_t rela_size = 0;
    bool has_plt = false, has_got = false;
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* h = shdr(i);
        const std::string name(reinterpret_cast<const char*>(shstr + rd32(h)));
        const uint32_t type = rd32(h + 4);
        const uint8_t* data = so.data() + rd64(h + 24);
        const uint64_t size = rd64(h + 32);
        if (type == elf64::SHT_DYNAMIC) { dynamic = data; dynamic_size = size; }
        else if (name == ".dynstr") dynstr = data;
        else if (type == elf64::SHT_DYNSYM) { dynsym = data; dynsym_size = size; }
        else if (type == elf64::SHT_RELA) { rela = data; rela_size = size; }
        else if (name == ".plt") has_plt = true;
        else if (name == ".got") has_got = true;
    }
    REQUIRE(dynamic != nullptr);
    REQUIRE(dynstr != nullptr);
    REQUIRE(dynsym != nullptr);
    REQUIRE(rela != nullptr);
    CHECK(has_plt);
    CHECK(has_got);

    bool needed = false, runpath = false, jmprel_or_rela = false, bind_now = false;
    for (uint64_t off = 0; off + 16 <= dynamic_size; off += 16) {
        const int64_t tag = static_cast<int64_t>(rd64(dynamic + off));
        const uint64_t val = rd64(dynamic + off + 8);
        if (tag == elf64::DT_NEEDED && std::string(reinterpret_cast<const char*>(dynstr + val)) == "libaot_import_provider.so") needed = true;
        if (tag == elf64::DT_RUNPATH && std::string(reinterpret_cast<const char*>(dynstr + val)) == "$ORIGIN") runpath = true;
        if (tag == elf64::DT_RELA) jmprel_or_rela = true;
        if (tag == elf64::DT_FLAGS && (val & elf64::DF_BIND_NOW)) bind_now = true;
        if (tag == elf64::DT_NULL) break;
    }
    CHECK(needed);
    CHECK(runpath);
    CHECK(jmprel_or_rela);
    CHECK(bind_now);

    // An undefined dynsym for the import, and a GLOB_DAT against it.
    uint32_t twice_index = 0;
    for (uint64_t i = 0; i * 24 < dynsym_size; ++i) {
        const uint8_t* s = dynsym + i * 24;
        if (std::string(reinterpret_cast<const char*>(dynstr + rd32(s))) == "twice") {
            twice_index = static_cast<uint32_t>(i);
            CHECK_EQ(rd16(s + 6), elf64::SHN_UNDEF);
            CHECK_EQ(rd64(s + 8), 0ULL);
        }
    }
    REQUIRE(twice_index > 0);
    bool glob_dat = false;
    for (uint64_t off = 0; off + 24 <= rela_size; off += 24) {
        const uint64_t info = rd64(rela + off + 8);
        if ((info >> 32) == twice_index && (info & 0xFFFFFFFFu) == elf64::R_X86_64_GLOB_DAT) glob_dat = true;
    }
    CHECK(glob_dat);
}

TEST_CASE("AOT Linker - Mach-O LC_LOAD_DYLIB, LC_RPATH and Undefined Symbols") {
    auto consumer = build_consumer();
    object::ObjectFile obj = object::compile_module_to_object(*consumer, Target::x64_macos());
    LinkerOptions opts = consumer_options("@rpath/libaot_import_provider.dylib");
    opts.rpaths.push_back("@loader_path");
    std::string err;
    std::vector<uint8_t> dylib = AotLinker::link(obj, opts, &err);
    if (dylib.empty()) std::fprintf(stderr, "Mach-O link failed: %s\n", err.c_str());
    REQUIRE(!dylib.empty());
    CHECK(err.empty());

    using namespace brass::object;
    const uint32_t ncmds = rd32(dylib.data() + 16);
    const uint32_t flags = rd32(dylib.data() + 24);
    CHECK((flags & macho::MH_NOUNDEFS) == 0);

    std::vector<std::string> loaded;
    std::vector<std::string> rpaths;
    bool has_stubs = false, has_got = false;
    uint32_t bind_size = 0;
    uint32_t symoff = 0, nsyms = 0, stroff = 0, iundefsym = 0, nundefsym = 0;
    const uint8_t* p = dylib.data() + 32;
    for (uint32_t c = 0; c < ncmds; ++c) {
        const uint32_t cmd = rd32(p);
        const uint32_t size = rd32(p + 4);
        if (cmd == macho::LC_LOAD_DYLIB) {
            loaded.emplace_back(reinterpret_cast<const char*>(p + rd32(p + 8)));
        } else if (cmd == 0x8000001c) {   // LC_RPATH
            rpaths.emplace_back(reinterpret_cast<const char*>(p + rd32(p + 8)));
        } else if (cmd == macho::LC_SEGMENT_64) {
            const uint32_t nsects = rd32(p + 64);
            for (uint32_t s = 0; s < nsects; ++s) {
                const uint8_t* sect = p + 72 + s * 80;
                if (std::strncmp(reinterpret_cast<const char*>(sect), "__stubs", 16) == 0) has_stubs = true;
                if (std::strncmp(reinterpret_cast<const char*>(sect), "__got", 16) == 0) has_got = true;
            }
        } else if (cmd == macho::LC_DYLD_INFO_ONLY) {
            bind_size = rd32(p + 20);
        } else if (cmd == macho::LC_SYMTAB) {
            symoff = rd32(p + 8); nsyms = rd32(p + 12); stroff = rd32(p + 16);
        } else if (cmd == macho::LC_DYSYMTAB) {
            iundefsym = rd32(p + 24); nundefsym = rd32(p + 28);
        }
        p += size;
    }
    REQUIRE_EQ(loaded.size(), size_t(2));
    CHECK_EQ(loaded[0], "@rpath/libaot_import_provider.dylib");
    CHECK_EQ(loaded[1], "/usr/lib/libSystem.B.dylib");
    REQUIRE_EQ(rpaths.size(), size_t(1));
    CHECK_EQ(rpaths[0], "@loader_path");
    CHECK(has_stubs);
    CHECK(has_got);
    CHECK(bind_size > 0);
    REQUIRE_EQ(nundefsym, 1u);
    REQUIRE(iundefsym < nsyms);
    const uint8_t* undef = dylib.data() + symoff + iundefsym * 16;
    CHECK_EQ(std::string(reinterpret_cast<const char*>(dylib.data() + stroff + rd32(undef))), "_twice");
    CHECK_EQ(undef[4], uint8_t(macho::N_UNDF | macho::N_EXT));
    CHECK_EQ(rd16(undef + 6) >> 8, 1);   // bound to LC_LOAD_DYLIB ordinal 1
}

TEST_CASE("AOT Linker - Native Roundtrip Through an Import") {
    if (!Target::host().is_windows() && !Target::host().is_macos()) return;
    const std::string ext = Target::host().is_windows() ? ".dll" : ".dylib";
    const std::string provider_name = Target::host().is_windows() ? "aot_import_provider.dll"
                                                                  : "libaot_import_provider.dylib";
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path provider_path = dir / provider_name;
    const std::filesystem::path consumer_path = dir / ("aot_import_consumer" + ext);
    std::error_code ec;
    std::filesystem::remove(provider_path, ec);
    std::filesystem::remove(consumer_path, ec);

    auto provider = build_provider();
    LinkerOptions popts;
    popts.export_all_functions = true;
    std::string err;
    REQUIRE(AotLinker::link_to_file(object::compile_module_to_object(*provider, Target::host()),
                                    provider_path.string(), popts, &err));

    auto consumer = build_consumer();
    LinkerOptions copts = consumer_options(Target::host().is_windows() ? provider_name : "@rpath/" + provider_name);
    copts.rpaths.push_back("@loader_path");
    REQUIRE(AotLinker::link_to_file(object::compile_module_to_object(*consumer, Target::host()),
                                    consumer_path.string(), copts, &err));

    // The provider is loaded first so the consumer's dependency resolves
    // from the already-loaded image rather than from the search path.
    auto provider_lib = DynamicLibrary::open(provider_path.string(), &err);
    REQUIRE(provider_lib != nullptr);
    auto consumer_lib = DynamicLibrary::open(consumer_path.string(), &err);
    if (!consumer_lib) std::fprintf(stderr, "open consumer failed: %s\n", err.c_str());
    REQUIRE(consumer_lib != nullptr);
    using Fn = int64_t (*)(int64_t);
    auto quad = consumer_lib->get_function<Fn>("quad");
    REQUIRE(quad != nullptr);
    CHECK_EQ(quad(3), 12);
    CHECK_EQ(quad(-5), -20);

    consumer_lib.reset();
    provider_lib.reset();
    std::filesystem::remove(consumer_path, ec);
    std::filesystem::remove(provider_path, ec);
}
