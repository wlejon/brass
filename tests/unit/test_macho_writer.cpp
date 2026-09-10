#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/target/macho_dylib_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <cstring>
#include <vector>
#include <string>

using namespace brass;
using namespace brass::object;
using namespace brass::target;

namespace {

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] | (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

} // namespace

TEST_CASE("Mach-O Writer - Header and Section Table Layout") {
    Module mod("test_macho");
    Function* fn = mod.create_function("simple_macho_add", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_macos());
    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);

    CHECK(!macho_bytes.empty());
    CHECK(macho_bytes.size() >= 32);

    // 1. mach_header_64 (32 bytes)
    const uint8_t* hdr = macho_bytes.data();
    uint32_t magic = read_u32(hdr + 0);
    int32_t cputype = static_cast<int32_t>(read_u32(hdr + 4));
    int32_t cpusubtype = static_cast<int32_t>(read_u32(hdr + 8));
    uint32_t filetype = read_u32(hdr + 12);
    uint32_t ncmds = read_u32(hdr + 16);
    uint32_t sizeofcmds = read_u32(hdr + 20);

    CHECK_EQ(magic, macho::MH_MAGIC_64);
    CHECK_EQ(cputype, macho::CPU_TYPE_X86_64);
    CHECK_EQ(cpusubtype, macho::CPU_SUBTYPE_X86_64_ALL);
    CHECK_EQ(filetype, macho::MH_OBJECT);
    CHECK_EQ(ncmds, 3u); // LC_SEGMENT_64, LC_SYMTAB, LC_DYSYMTAB
    CHECK(sizeofcmds > 0);

    // 2. Read LC_SEGMENT_64
    const uint8_t* cmd_ptr = hdr + 32;
    uint32_t seg_cmd = read_u32(cmd_ptr + 0);
    uint32_t seg_cmdsize = read_u32(cmd_ptr + 4);
    CHECK_EQ(seg_cmd, macho::LC_SEGMENT_64);

    uint32_t nsects = read_u32(cmd_ptr + 64);
    CHECK(nsects >= 2u); // __text and __eh_frame (and __const if stack maps present)

    bool found_text = false;
    bool found_eh_frame = false;

    const uint8_t* sec_ptr = cmd_ptr + 72;
    for (uint32_t i = 0; i < nsects; ++i) {
        char sectname[17] = {0};
        char segname[17] = {0};
        std::memcpy(sectname, sec_ptr + i * 80, 16);
        std::memcpy(segname, sec_ptr + i * 80 + 16, 16);

        uint64_t size = read_u64(sec_ptr + i * 80 + 40);
        uint32_t offset = read_u32(sec_ptr + i * 80 + 48);
        uint32_t align_pow2 = read_u32(sec_ptr + i * 80 + 52);
        uint32_t flags = read_u32(sec_ptr + i * 80 + 64);

        if (std::strcmp(sectname, "__text") == 0) {
            found_text = true;
            CHECK_EQ(std::string(segname), "__TEXT");
            CHECK(size > 0);
            CHECK(offset >= 32 + sizeofcmds);
            CHECK_EQ(align_pow2, 4u); // 2^4 = 16-byte aligned
            CHECK((flags & macho::S_ATTR_PURE_INSTRUCTIONS) != 0);
        } else if (std::strcmp(sectname, "__eh_frame") == 0) {
            found_eh_frame = true;
            CHECK_EQ(std::string(segname), "__TEXT");
            CHECK(size > 0);
            CHECK(align_pow2 >= 3u); // At least 8-byte aligned
        }
    }

    CHECK(found_text);
    CHECK(found_eh_frame);

    // 3. Read LC_SYMTAB
    cmd_ptr += seg_cmdsize;
    uint32_t symtab_cmd = read_u32(cmd_ptr + 0);
    uint32_t symtab_cmdsize = read_u32(cmd_ptr + 4);
    CHECK_EQ(symtab_cmd, macho::LC_SYMTAB);
    CHECK_EQ(symtab_cmdsize, 24u);

    uint32_t symoff = read_u32(cmd_ptr + 8);
    uint32_t nsyms = read_u32(cmd_ptr + 12);
    uint32_t stroff = read_u32(cmd_ptr + 16);
    uint32_t strsize = read_u32(cmd_ptr + 20);

    CHECK(symoff > 0);
    CHECK(nsyms > 0);
    CHECK(stroff > symoff);
    CHECK(strsize > 0);

    // Verify symbol names have leading underscore
    const char* strtab = reinterpret_cast<const char*>(macho_bytes.data() + stroff);
    bool found_macho_fn = false;

    for (uint32_t i = 0; i < nsyms; ++i) {
        const uint8_t* sym = macho_bytes.data() + symoff + i * 16;
        uint32_t n_strx = read_u32(sym + 0);
        uint8_t n_type = sym[4];
        uint8_t n_sect = sym[5];

        if (n_strx < strsize) {
            const char* name = strtab + n_strx;
            if (std::strcmp(name, "_simple_macho_add") == 0) {
                found_macho_fn = true;
                CHECK_EQ(n_type, macho::N_SECT | macho::N_EXT);
                CHECK_EQ(n_sect, 1u); // 1st section is __text
            }
        }
    }

    CHECK(found_macho_fn);

    // 4. Read LC_DYSYMTAB
    cmd_ptr += symtab_cmdsize;
    uint32_t dysymtab_cmd = read_u32(cmd_ptr + 0);
    uint32_t dysymtab_cmdsize = read_u32(cmd_ptr + 4);
    CHECK_EQ(dysymtab_cmd, macho::LC_DYSYMTAB);
    CHECK_EQ(dysymtab_cmdsize, 80u);
}

TEST_CASE("Mach-O Writer - Relocations for Function Calls and Branches") {
    Module mod("test_macho_relocs");
    Function* f1 = mod.create_function("macho_callee", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("macho_caller", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* two = b.build_iconst_i64(2);
        Value* res = b.build_mul(x, two);
        b.build_ret(res);
        f1->rebuild_cfg_predecessors();
        CHECK(verify_function(*f1));
    }

    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("macho_callee", Type::i64(), {x});
        b.build_ret(call_res);
        f2->rebuild_cfg_predecessors();
        CHECK(verify_function(*f2));
    }

    ObjectFile obj = compile_module_to_object(mod, Target::x64_macos());
    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);

    const uint8_t* hdr = macho_bytes.data();
    const uint8_t* cmd_ptr = hdr + 32;
    uint32_t nsects = read_u32(cmd_ptr + 64);

    bool found_branch_reloc = false;
    const uint8_t* sec_ptr = cmd_ptr + 72;
    for (uint32_t i = 0; i < nsects; ++i) {
        char sectname[17] = {0};
        std::memcpy(sectname, sec_ptr + i * 80, 16);

        if (std::strcmp(sectname, "__text") == 0) {
            uint32_t reloff = read_u32(sec_ptr + i * 80 + 56);
            uint32_t nreloc = read_u32(sec_ptr + i * 80 + 60);

            if (nreloc > 0 && reloff > 0) {
                const uint8_t* reloc_ptr = macho_bytes.data() + reloff;
                for (uint32_t r = 0; r < nreloc; ++r) {
                    uint32_t r_info = read_u32(reloc_ptr + r * 8 + 4);
                    uint32_t r_pcrel = (r_info >> 24) & 1;
                    uint32_t r_length = (r_info >> 25) & 3;
                    uint32_t r_extern = (r_info >> 27) & 1;
                    uint32_t r_type = (r_info >> 28) & 0xF;

                    if (r_type == macho::X86_64_RELOC_BRANCH) {
                        found_branch_reloc = true;
                        CHECK_EQ(r_pcrel, 1u);
                        CHECK_EQ(r_length, 2u); // 4-byte displacement
                        CHECK_EQ(r_extern, 1u);
                    }
                }
            }
        }
    }

    CHECK(found_branch_reloc);
}

TEST_CASE("Mach-O Writer - RoData Stack Maps Placement in __TEXT,__const") {
    Module mod("test_macho_const");
    Function* fn = mod.create_function("fn_stack_map", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_ret(x);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_macos());
    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);

    const uint8_t* hdr = macho_bytes.data();
    const uint8_t* cmd_ptr = hdr + 32;
    uint32_t nsects = read_u32(cmd_ptr + 64);

    bool found_const_sec = false;
    const uint8_t* sec_ptr = cmd_ptr + 72;
    for (uint32_t i = 0; i < nsects; ++i) {
        char sectname[17] = {0};
        char segname[17] = {0};
        std::memcpy(sectname, sec_ptr + i * 80, 16);
        std::memcpy(segname, sec_ptr + i * 80 + 16, 16);

        if (std::strcmp(sectname, "__const") == 0) {
            found_const_sec = true;
            CHECK_EQ(std::string(segname), "__TEXT");
        }
    }

    CHECK(found_const_sec);
}

TEST_CASE("Mach-O Dylib Writer - Direct Emission Structure") {
    Module mod("test_macho_dylib");
    Function* fn = mod.create_function("dylib_mult", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* prod = b.build_mul(a, c);
    b.build_ret(prod);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    LinkerOptions opts;
    opts.format = OutputFormat::MacOSMachODylib;
    opts.soname = "libdylib_mult.dylib";
    opts.export_all_functions = true;

    std::vector<uint8_t> dylib_bytes = AotLinker::link(mod, Target::x64_macos(), opts);
    REQUIRE(dylib_bytes.size() >= 4096);

    const uint8_t* hdr = dylib_bytes.data();
    uint32_t magic = read_u32(hdr + 0);
    int32_t cputype = static_cast<int32_t>(read_u32(hdr + 4));
    uint32_t filetype = read_u32(hdr + 12);
    uint32_t ncmds = read_u32(hdr + 16);
    uint32_t flags = read_u32(hdr + 24);

    CHECK_EQ(magic, macho::MH_MAGIC_64);
    CHECK_EQ(cputype, macho::CPU_TYPE_X86_64);
    CHECK_EQ(filetype, macho::MH_DYLIB);
    CHECK(ncmds >= 7u);
    CHECK((flags & macho::MH_NOUNDEFS) != 0);
    CHECK((flags & macho::MH_DYLDLINK) != 0);
    CHECK((flags & macho::MH_TWOLEVEL) != 0);

    // Verify load commands
    const uint8_t* p = hdr + 32;
    bool has_text_seg = false;
    bool has_linkedit_seg = false;
    bool has_id_dylib = false;
    bool has_load_dylib = false;
    bool has_dyld_info = false;
    bool has_symtab = false;
    bool has_dysymtab = false;

    for (uint32_t c = 0; c < ncmds; ++c) {
        uint32_t cmd = read_u32(p + 0);
        uint32_t cmdsize = read_u32(p + 4);

        if (cmd == macho::LC_SEGMENT_64) {
            char segname[17] = {0};
            std::memcpy(segname, p + 8, 16);
            if (std::strcmp(segname, "__TEXT") == 0) has_text_seg = true;
            if (std::strcmp(segname, "__LINKEDIT") == 0) has_linkedit_seg = true;
        } else if (cmd == macho::LC_ID_DYLIB) {
            has_id_dylib = true;
        } else if (cmd == macho::LC_LOAD_DYLIB) {
            has_load_dylib = true;
        } else if (cmd == macho::LC_DYLD_INFO_ONLY) {
            has_dyld_info = true;
        } else if (cmd == macho::LC_SYMTAB) {
            has_symtab = true;
        } else if (cmd == macho::LC_DYSYMTAB) {
            has_dysymtab = true;
        }

        p += cmdsize;
    }

    CHECK(has_text_seg);
    CHECK(has_linkedit_seg);
    CHECK(has_id_dylib);
    CHECK(has_load_dylib);
    CHECK(has_dyld_info);
    CHECK(has_symtab);
    CHECK(has_dysymtab);
}
