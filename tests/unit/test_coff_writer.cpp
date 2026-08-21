#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <cstring>

using namespace brass;
using namespace brass::object;

namespace {

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] | (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

} // namespace

TEST_CASE("COFF Writer - Header and Section Table Layout") {
    Module mod("test_coff");
    Function* fn = mod.create_function("simple_add", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);

    CHECK(!coff_bytes.empty());
    CHECK(coff_bytes.size() >= 20);

    // 1. IMAGE_FILE_HEADER
    const uint8_t* hdr = coff_bytes.data();
    uint16_t machine = read_u16(hdr + 0);
    uint16_t num_sections = read_u16(hdr + 2);
    uint32_t sym_ptr = read_u32(hdr + 8);
    uint32_t num_syms = read_u32(hdr + 12);
    uint16_t opt_hdr_size = read_u16(hdr + 16);

    CHECK_EQ(machine, coff::IMAGE_FILE_MACHINE_AMD64);
    CHECK(num_sections >= 3); // .text, .xdata, .pdata
    CHECK_EQ(opt_hdr_size, uint16_t(0));
    CHECK(sym_ptr > 0);
    CHECK(num_syms > 0);

    // 2. Section Headers
    const uint8_t* sec_hdrs = hdr + 20;
    bool found_text = false;
    bool found_pdata = false;
    bool found_xdata = false;

    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sec, 8);

        uint32_t raw_size = read_u32(sec + 16);
        uint32_t raw_ptr = read_u32(sec + 20);
        uint32_t chars = read_u32(sec + 36);

        if (std::strcmp(name, ".text") == 0) {
            found_text = true;
            CHECK(raw_size > 0);
            CHECK(raw_ptr >= static_cast<uint32_t>(20 + num_sections * 40));
            CHECK((chars & coff::IMAGE_SCN_CNT_CODE) != 0);
            CHECK((chars & coff::IMAGE_SCN_MEM_EXECUTE) != 0);
            CHECK((chars & coff::IMAGE_SCN_MEM_READ) != 0);
        } else if (std::strcmp(name, ".pdata") == 0) {
            found_pdata = true;
            CHECK_EQ(raw_size, uint32_t(12)); // 1 function = 1 RUNTIME_FUNCTION (12 bytes)
            CHECK((chars & coff::IMAGE_SCN_CNT_INITIALIZED_DATA) != 0);
        } else if (std::strcmp(name, ".xdata") == 0) {
            found_xdata = true;
            CHECK(raw_size >= 4); // At least 4-byte UNWIND_INFO header
            CHECK((chars & coff::IMAGE_SCN_CNT_INITIALIZED_DATA) != 0);
        }
    }

    CHECK(found_text);
    CHECK(found_pdata);
    CHECK(found_xdata);
}

TEST_CASE("COFF Writer - Win64 SEH .pdata & .xdata Unwind Info") {
    Module mod("test_unwind");
    Function* fn = mod.create_function("fn_with_frame", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    
    // Create multiple calls to force stack spill / frame allocation
    Value* c1 = b.build_iconst_i64(1);
    Value* c2 = b.build_iconst_i64(2);
    Value* a1 = b.build_add(x, c1);
    Value* a2 = b.build_add(x, c2);
    Value* sum = b.build_add(a1, a2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);

    const uint8_t* hdr = coff_bytes.data();
    uint16_t num_sections = read_u16(hdr + 2);
    const uint8_t* sec_hdrs = hdr + 20;

    const uint8_t* xdata_ptr = nullptr;
    uint32_t xdata_size = 0;
    const uint8_t* pdata_ptr = nullptr;
    uint32_t pdata_size = 0;

    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sec, 8);
        uint32_t raw_size = read_u32(sec + 16);
        uint32_t raw_ptr = read_u32(sec + 20);

        if (std::strcmp(name, ".xdata") == 0) {
            xdata_ptr = hdr + raw_ptr;
            xdata_size = raw_size;
        } else if (std::strcmp(name, ".pdata") == 0) {
            pdata_ptr = hdr + raw_ptr;
            pdata_size = raw_size;
        }
    }

    CHECK(pdata_ptr != nullptr);
    CHECK(xdata_ptr != nullptr);
    CHECK(xdata_size >= 4);
    CHECK_EQ(pdata_size, uint32_t(12));

    // Inspect .xdata UNWIND_INFO
    uint8_t ver_flags = xdata_ptr[0];
    uint8_t prolog_sz = xdata_ptr[1];
    uint8_t cnt_codes = xdata_ptr[2];
    uint8_t frame_reg = xdata_ptr[3];

    CHECK_EQ(ver_flags & 0x07, 1); // Version 1
    CHECK_EQ(ver_flags >> 3, 0);   // UNW_FLAG_NHANDLER = 0
    CHECK(prolog_sz > 0);
    CHECK(cnt_codes >= 2); // At least push rbp and set_fpreg
    CHECK_EQ(frame_reg & 0x0F, 5); // RBP
}

TEST_CASE("COFF Writer - Relocations for Function Calls") {
    Module mod("test_relocs");
    Function* f1 = mod.create_function("target_func", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("caller_func", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* one = b.build_iconst_i64(1);
        Value* res = b.build_add(x, one);
        b.build_ret(res);
        f1->rebuild_cfg_predecessors();
        CHECK(verify_function(*f1));
    }

    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("target_func", Type::i64(), {x});
        b.build_ret(call_res);
        f2->rebuild_cfg_predecessors();
        CHECK(verify_function(*f2));
    }

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);

    const uint8_t* hdr = coff_bytes.data();
    uint16_t num_sections = read_u16(hdr + 2);
    const uint8_t* sec_hdrs = hdr + 20;

    bool found_text_reloc = false;
    for (uint16_t i = 0; i < num_sections; ++i) {
        const uint8_t* sec = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sec, 8);
        if (std::strcmp(name, ".text") == 0) {
            uint32_t reloc_ptr = read_u32(sec + 24);
            uint16_t num_relocs = read_u16(sec + 32);

            CHECK(num_relocs >= 1);
            const uint8_t* reloc = hdr + reloc_ptr;
            uint16_t r_type = read_u16(reloc + 8);
            CHECK_EQ(r_type, coff::IMAGE_REL_AMD64_REL32);
            found_text_reloc = true;
        }
    }
    CHECK(found_text_reloc);
}
