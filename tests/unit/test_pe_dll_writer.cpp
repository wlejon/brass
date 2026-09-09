#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/pe_dll_writer.hpp>
#include <cstring>
#include <vector>
#include <string>

using namespace brass;
using namespace brass::target;

namespace {

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

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

TEST_CASE("PE DLL Writer - Header Validation and Magic Signatures") {
    Module mod("test_pe_hdr");
    Function* fn = mod.create_function("simple_add", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_windows());
    PeDllOptions opts;
    opts.module_name = "test_pe_hdr.dll";
    opts.image_base = 0x180000000ULL;
    std::vector<uint8_t> dll = PeDllWriter::emit(obj, opts);

    REQUIRE(dll.size() >= 0x400);

    // 1. DOS Header
    uint16_t dos_magic = read_u16(dll.data());
    CHECK_EQ(dos_magic, pe::IMAGE_DOS_SIGNATURE); // 0x5A4D "MZ"
    uint32_t pe_offset = read_u32(dll.data() + 0x3C);
    CHECK_EQ(pe_offset, 0x80u);

    // 2. PE Signature
    const uint8_t* pe_hdr = dll.data() + pe_offset;
    uint32_t pe_sig = read_u32(pe_hdr);
    CHECK_EQ(pe_sig, pe::IMAGE_NT_SIGNATURE); // 0x00004550 "PE\0\0"

    // 3. IMAGE_FILE_HEADER (20 bytes at pe_hdr + 4)
    const uint8_t* file_hdr = pe_hdr + 4;
    uint16_t machine = read_u16(file_hdr + 0);
    uint16_t num_sections = read_u16(file_hdr + 2);
    uint16_t opt_hdr_size = read_u16(file_hdr + 16);
    uint16_t characteristics = read_u16(file_hdr + 18);

    CHECK_EQ(machine, pe::IMAGE_FILE_MACHINE_AMD64); // 0x8664
    CHECK(num_sections >= 3); // .text, .edata, .reloc
    CHECK_EQ(opt_hdr_size, uint16_t(240)); // 0xF0
    CHECK((characteristics & pe::IMAGE_FILE_EXECUTABLE_IMAGE) != 0);
    CHECK((characteristics & pe::IMAGE_FILE_DLL) != 0);
    CHECK((characteristics & pe::IMAGE_FILE_LARGE_ADDRESS_AWARE) != 0);

    // 4. IMAGE_OPTIONAL_HEADER64 (240 bytes at file_hdr + 20)
    const uint8_t* opt_hdr = file_hdr + 20;
    uint16_t opt_magic = read_u16(opt_hdr + 0);
    CHECK_EQ(opt_magic, pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC); // 0x020B (PE32+)

    uint64_t image_base = read_u64(opt_hdr + 24);
    CHECK_EQ(image_base, 0x180000000ULL);

    uint32_t sec_align = read_u32(opt_hdr + 32);
    uint32_t file_align = read_u32(opt_hdr + 36);
    CHECK_EQ(sec_align, 0x1000u);
    CHECK_EQ(file_align, 0x200u);

    uint16_t dll_chars = read_u16(opt_hdr + 70);
    CHECK((dll_chars & pe::IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE) != 0);
    CHECK((dll_chars & pe::IMAGE_DLLCHARACTERISTICS_NX_COMPAT) != 0);
    CHECK((dll_chars & pe::IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA) != 0);

    uint32_t num_rva_sizes = read_u32(opt_hdr + 108);
    CHECK_EQ(num_rva_sizes, 16u);

    // Data Directories
    const uint8_t* data_dirs = opt_hdr + 112;
    uint32_t export_rva = read_u32(data_dirs + 0);
    uint32_t export_size = read_u32(data_dirs + 4);
    CHECK(export_rva > 0);
    CHECK(export_size > 0);

    uint32_t reloc_rva = read_u32(data_dirs + 5 * 8);
    uint32_t reloc_size = read_u32(data_dirs + 5 * 8 + 4);
    CHECK(reloc_rva > 0);
    CHECK(reloc_size > 0);
}

TEST_CASE("PE DLL Writer - Export Directory Formatting and Ordinal Mapping") {
    Module mod("test_pe_exports");
    std::vector<std::string> names = {"zebra_fn", "alpha_fn", "mike_fn", "bravo_fn"};

    for (const auto& name : names) {
        Function* f = mod.create_function(name, Type::i64(), {Type::i64()});
        Builder b(mod);
        b.set_function(f);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* one = b.build_iconst_i64(1);
        Value* res = b.build_add(x, one);
        b.build_ret(res);
        f->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f));
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_windows());
    PeDllOptions opts;
    opts.module_name = "math_funcs.dll";
    std::vector<uint8_t> dll = PeDllWriter::emit(obj, opts);

    // Locate section headers
    uint32_t pe_offset = read_u32(dll.data() + 0x3C);
    const uint8_t* file_hdr = dll.data() + pe_offset + 4;
    uint16_t num_sections = read_u16(file_hdr + 2);
    const uint8_t* opt_hdr = file_hdr + 20;
    const uint8_t* sec_hdrs = opt_hdr + 240;

    uint32_t export_rva = read_u32(opt_hdr + 112);
    uint32_t export_size = read_u32(opt_hdr + 116);
    CHECK(export_rva > 0);
    CHECK(export_size > 0);

    // Find section containing export_rva
    uint32_t edata_file_off = 0;
    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        uint32_t sec_rva = read_u32(sec + 12);
        uint32_t sec_raw_size = read_u32(sec + 16);
        uint32_t sec_file_off = read_u32(sec + 20);
        if (export_rva >= sec_rva && export_rva < sec_rva + sec_raw_size) {
            edata_file_off = sec_file_off + (export_rva - sec_rva);
            break;
        }
    }
    REQUIRE(edata_file_off > 0);

    // Inspect IMAGE_EXPORT_DIRECTORY (40 bytes)
    const uint8_t* exp_dir = dll.data() + edata_file_off;
    uint32_t mod_name_rva = read_u32(exp_dir + 12);
    uint32_t base = read_u32(exp_dir + 16);
    uint32_t num_funcs = read_u32(exp_dir + 20);
    uint32_t num_names = read_u32(exp_dir + 24);
    uint32_t eat_rva = read_u32(exp_dir + 28);
    uint32_t ent_rva = read_u32(exp_dir + 32);
    uint32_t ord_rva = read_u32(exp_dir + 36);

    CHECK_EQ(base, 1u);
    CHECK_EQ(num_funcs, 4u);
    CHECK_EQ(num_names, 4u);
    CHECK(eat_rva > export_rva);
    CHECK(ent_rva > export_rva);
    CHECK(ord_rva > export_rva);

    auto rva_to_ptr = [&](uint32_t rva) -> const uint8_t* {
        for (uint16_t i = 0; i < num_sections; ++i) {
            const uint8_t* sec = sec_hdrs + i * 40;
            uint32_t s_rva = read_u32(sec + 12);
            uint32_t s_raw_size = read_u32(sec + 16);
            uint32_t s_file_off = read_u32(sec + 20);
            if (rva >= s_rva && rva < s_rva + s_raw_size) {
                return dll.data() + s_file_off + (rva - s_rva);
            }
        }
        return nullptr;
    };

    // Verify module name string
    const char* mod_name = reinterpret_cast<const char*>(rva_to_ptr(mod_name_rva));
    REQUIRE(mod_name != nullptr);
    CHECK_EQ(std::string(mod_name), "math_funcs.dll");

    // Verify ENT is lexicographically sorted!
    std::vector<std::string> sorted_names;
    for (uint32_t i = 0; i < num_names; ++i) {
        uint32_t name_rva = read_u32(rva_to_ptr(ent_rva + i * 4));
        const char* s = reinterpret_cast<const char*>(rva_to_ptr(name_rva));
        REQUIRE(s != nullptr);
        sorted_names.push_back(std::string(s));
    }

    std::vector<std::string> expected_sorted = {"alpha_fn", "bravo_fn", "mike_fn", "zebra_fn"};
    REQUIRE_EQ(sorted_names.size(), expected_sorted.size());
    for (size_t i = 0; i < expected_sorted.size(); ++i) {
        CHECK_EQ(sorted_names[i], expected_sorted[i]);
    }

    // Verify Ordinals and EAT function RVAs
    for (uint32_t i = 0; i < num_names; ++i) {
        uint16_t ord = read_u16(rva_to_ptr(ord_rva + i * 2));
        CHECK(ord < num_funcs);
        uint32_t fn_rva = read_u32(rva_to_ptr(eat_rva + ord * 4));
        CHECK(fn_rva >= 0x1000u); // In .text section
    }
}

TEST_CASE("PE DLL Writer - Base Relocation Page Alignment and Blocks") {
    Module mod("test_pe_reloc");
    Function* fn = mod.create_function("reloc_fn", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* c10 = b.build_iconst_i64(10);
    Value* res = b.build_add(x, c10);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> dll = PeDllWriter::emit(obj);

    uint32_t pe_offset = read_u32(dll.data() + 0x3C);
    const uint8_t* opt_hdr = dll.data() + pe_offset + 24;
    const uint8_t* sec_hdrs = opt_hdr + 240;
    uint16_t num_sections = read_u16(dll.data() + pe_offset + 6);

    uint32_t reloc_rva = read_u32(opt_hdr + 112 + 5 * 8);
    uint32_t reloc_size = read_u32(opt_hdr + 116 + 5 * 8);
    CHECK(reloc_rva > 0);
    CHECK(reloc_size >= 12); // At least one base relocation block

    // Find .reloc file offset
    uint32_t reloc_file_off = 0;
    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sec, 8);
        if (std::strcmp(name, ".reloc") == 0) {
            reloc_file_off = read_u32(sec + 20);
            break;
        }
    }
    REQUIRE(reloc_file_off > 0);

    const uint8_t* reloc_ptr = dll.data() + reloc_file_off;
    uint32_t page_rva = read_u32(reloc_ptr + 0);
    uint32_t block_size = read_u32(reloc_ptr + 4);

    CHECK_EQ(page_rva % 0x1000u, 0u); // 4KB page aligned!
    CHECK_EQ(block_size % 4u, 0u);    // 4-byte block size aligned!
    CHECK(block_size >= 8u);
}

TEST_CASE("PE DLL Writer - Win64 SEH .pdata and .xdata Exception Directory") {
    Module mod("test_pe_seh");
    mod.add_external_symbol("target_callee");
    Function* fn = mod.create_function("fn_with_seh", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* call_res = b.build_call("target_callee", Type::i64(), {x});
    b.build_ret(call_res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> dll = PeDllWriter::emit(obj);

    uint32_t pe_offset = read_u32(dll.data() + 0x3C);
    const uint8_t* opt_hdr = dll.data() + pe_offset + 24;
    const uint8_t* sec_hdrs = opt_hdr + 240;
    uint16_t num_sections = read_u16(dll.data() + pe_offset + 6);

    uint32_t pdata_rva = read_u32(opt_hdr + 112 + 3 * 8);
    uint32_t pdata_size = read_u32(opt_hdr + 116 + 3 * 8);

    CHECK(pdata_rva > 0);
    CHECK_EQ(pdata_size, 12u); // 1 function = 1 RUNTIME_FUNCTION (12 bytes)

    // Locate .pdata in file
    uint32_t pdata_file_off = 0;
    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sec, 8);
        if (std::strcmp(name, ".pdata") == 0) {
            pdata_file_off = read_u32(sec + 20);
            break;
        }
    }
    REQUIRE(pdata_file_off > 0);

    const uint8_t* pdata_ptr = dll.data() + pdata_file_off;
    uint32_t begin_addr = read_u32(pdata_ptr + 0);
    uint32_t end_addr = read_u32(pdata_ptr + 4);
    uint32_t unwind_info_addr = read_u32(pdata_ptr + 8);

    CHECK(begin_addr >= 0x1000u);
    CHECK(end_addr > begin_addr);
    CHECK(unwind_info_addr > 0);
}
