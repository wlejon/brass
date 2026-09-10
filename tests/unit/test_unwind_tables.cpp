#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>

using namespace brass;
using namespace brass::runtime;
using namespace brass::object;

TEST_CASE("Unwind Tables - FunctionExceptionTable Scope Lookup") {
    FunctionExceptionTable table("test_fn", 0, 100);
    table.add_scope(10, 20, 50);
    table.add_scope(30, 45, 80);

    CHECK(table.has_scopes());
    CHECK_EQ(table.scopes().size(), 2u);

    // If call is at 15 (return address is 20):
    // find_scope with return IP = 20: func_ip_offset = 20, call_ip = 19 -> matches [10, 20)
    const ExceptionScopeEntry* s1 = table.find_scope(20);
    REQUIRE(s1 != nullptr);
    CHECK_EQ(s1->begin_offset, 10u);
    CHECK_EQ(s1->end_offset, 20u);
    CHECK_EQ(s1->landing_pad_offset, 50u);

    // Call at 35 (return address 40): matches [30, 45)
    const ExceptionScopeEntry* s2 = table.find_scope(40);
    REQUIRE(s2 != nullptr);
    CHECK_EQ(s2->landing_pad_offset, 80u);

    // Unprotected offset
    const ExceptionScopeEntry* s_none = table.find_scope(25);
    CHECK(s_none == nullptr);
}

TEST_CASE("Unwind Tables - Win64 SEH Scope Table Emitter") {
    FunctionExceptionTable table("win64_fn", 0, 200);
    table.add_scope(0x10, 0x25, 0x80);
    table.add_scope(0x40, 0x60, 0xC0);

    Section xdata;
    emit_win64_seh_scope_table(xdata, table);

    // DWORD count = 2
    // then 2 scopes of 3 DWORDs = 6 DWORDs. Total = 7 * 4 = 28 bytes.
    REQUIRE_EQ(xdata.data.size(), 28u);

    auto read_u32 = [&](size_t off) -> uint32_t {
        return static_cast<uint32_t>(xdata.data[off]) |
               (static_cast<uint32_t>(xdata.data[off + 1]) << 8) |
               (static_cast<uint32_t>(xdata.data[off + 2]) << 16) |
               (static_cast<uint32_t>(xdata.data[off + 3]) << 24);
    };

    CHECK_EQ(read_u32(0), 2u);        // ScopeCount
    CHECK_EQ(read_u32(4), 0x10u);     // Scope 0 begin
    CHECK_EQ(read_u32(8), 0x25u);     // Scope 0 end
    CHECK_EQ(read_u32(12), 0x80u);    // Scope 0 landing pad
    CHECK_EQ(read_u32(16), 0x40u);    // Scope 1 begin
    CHECK_EQ(read_u32(20), 0x60u);    // Scope 1 end
    CHECK_EQ(read_u32(24), 0xC0u);    // Scope 1 landing pad
}

TEST_CASE("Unwind Tables - COFF UNW_FLAG_EHANDLER & Personality Relocation") {
    ObjectFile obj;
    Section& text_sec = obj.get_or_create_section(".text", SectionKind::Text, SectionFlags::Read | SectionFlags::Execute);
    text_sec.emit32(0x90909090); // 4 dummy nops

    CompiledFunctionInfo cfi;
    cfi.name = "seh_fn";
    cfi.text_offset = 0;
    cfi.text_size = 4;
    cfi.frame_info.is_leaf = false;
    cfi.frame_info.total_frame_size = 32;
    cfi.exception_table.set_function_name("seh_fn");
    cfi.exception_table.add_scope(1, 3, 4);
    obj.functions.push_back(std::move(cfi));

    Section& pdata_sec = obj.get_or_create_section(".pdata", SectionKind::Data, SectionFlags::Read);
    Section& xdata_sec = obj.get_or_create_section(".xdata", SectionKind::Data, SectionFlags::Read);

    CoffUnwindBuilder::build_unwind_info(obj, pdata_sec, xdata_sec);

    // Check UNWIND_INFO header has UNW_FLAG_EHANDLER: 0x01 | 0x08 = 0x09
    REQUIRE(!xdata_sec.data.empty());
    uint8_t version_flags = xdata_sec.data[0];
    CHECK_EQ(version_flags & 0x08, 0x08); // UNW_FLAG_EHANDLER bit set

    // Check relocation to brass_seh_personality exists
    bool found_personality_reloc = false;
    for (const auto& r : xdata_sec.relocations) {
        if (r.symbol_name == "brass_seh_personality") {
            found_personality_reloc = true;
            break;
        }
    }
    CHECK(found_personality_reloc);
}

TEST_CASE("Unwind Tables - SysV DWARF LSDA Emitter") {
    FunctionExceptionTable table("sysv_fn", 0, 150);
    table.add_scope(0x10, 0x20, 0x70);

    Section lsda_sec;
    emit_sysv_lsda(lsda_sec, table);

    REQUIRE(lsda_sec.data.size() >= 10u);

    // Byte 0: DW_EH_PE_omit (0xFF)
    CHECK_EQ(lsda_sec.data[0], 0xFF);
    // Byte 1: DW_EH_PE_omit (0xFF)
    CHECK_EQ(lsda_sec.data[1], 0xFF);

    // Search for cs_lp = 0x70 in call site records
    bool found_lp = false;
    for (size_t i = 0; i + 4 <= lsda_sec.data.size(); ++i) {
        uint32_t val = static_cast<uint32_t>(lsda_sec.data[i]) |
                       (static_cast<uint32_t>(lsda_sec.data[i + 1]) << 8) |
                       (static_cast<uint32_t>(lsda_sec.data[i + 2]) << 16) |
                       (static_cast<uint32_t>(lsda_sec.data[i + 3]) << 24);
        if (val == 0x70u) {
            found_lp = true;
            break;
        }
    }
    CHECK(found_lp);
}
