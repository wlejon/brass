#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <brass/debug/debug_section.hpp>
#include <cstring>

using namespace brass;
using namespace brass::debug;
using namespace brass::object;

namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] |
                                 (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

} // namespace

TEST_CASE("CodeView Emitter - Section Flags and Types (.debug$T)") {
    Section t_sec;
    t_sec.name = ".debug$T";
    t_sec.kind = SectionKind::Custom;
    t_sec.flags = SectionFlags::Read | SectionFlags::Discardable;
    t_sec.alignment = 4;
    CodeViewEmitter::emit_debug_t(t_sec);

    REQUIRE(t_sec.data.size() >= 4);
    const uint8_t* p = t_sec.data.data();
    uint32_t sig = read_u32_le(p);
    CHECK_EQ(sig, codeview::CV_SIGNATURE_C13);

    size_t off = 4;
    // Record 1: LF_ARGLIST (0x1201)
    REQUIRE(off + 4 <= t_sec.data.size());
    uint16_t rec1_len = read_u16_le(p + off);
    uint16_t rec1_kind = read_u16_le(p + off + 2);
    CHECK_EQ(rec1_len, uint16_t(6));
    CHECK_EQ(rec1_kind, codeview::LF_ARGLIST);
    off += 2 + rec1_len;

    // Record 2: LF_PROCEDURE (0x1008) -> returns T_INT8
    REQUIRE(off + 4 <= t_sec.data.size());
    uint16_t rec2_len = read_u16_le(p + off);
    uint16_t rec2_kind = read_u16_le(p + off + 2);
    uint32_t rec2_ret = read_u32_le(p + off + 4);
    CHECK_EQ(rec2_len, uint16_t(14));
    CHECK_EQ(rec2_kind, codeview::LF_PROCEDURE);
    CHECK_EQ(rec2_ret, codeview::T_INT8);
    off += 2 + rec2_len;

    // Record 3: LF_PROCEDURE (0x1008) -> returns T_REAL64
    REQUIRE(off + 4 <= t_sec.data.size());
    uint16_t rec3_len = read_u16_le(p + off);
    uint16_t rec3_kind = read_u16_le(p + off + 2);
    uint32_t rec3_ret = read_u32_le(p + off + 4);
    CHECK_EQ(rec3_len, uint16_t(14));
    CHECK_EQ(rec3_kind, codeview::LF_PROCEDURE);
    CHECK_EQ(rec3_ret, codeview::T_REAL64);
    off += 2 + rec3_len;

    // Record 4: LF_PROCEDURE (0x1008) -> returns T_INT4
    REQUIRE(off + 4 <= t_sec.data.size());
    uint16_t rec4_len = read_u16_le(p + off);
    uint16_t rec4_kind = read_u16_le(p + off + 2);
    uint32_t rec4_ret = read_u32_le(p + off + 4);
    CHECK_EQ(rec4_len, uint16_t(14));
    CHECK_EQ(rec4_kind, codeview::LF_PROCEDURE);
    CHECK_EQ(rec4_ret, codeview::T_INT4);
    off += 2 + rec4_len;

    // Record 5: LF_PROCEDURE (0x1008) -> returns T_64PVOID
    REQUIRE(off + 4 <= t_sec.data.size());
    uint16_t rec5_len = read_u16_le(p + off);
    uint16_t rec5_kind = read_u16_le(p + off + 2);
    uint32_t rec5_ret = read_u32_le(p + off + 4);
    CHECK_EQ(rec5_len, uint16_t(14));
    CHECK_EQ(rec5_kind, codeview::LF_PROCEDURE);
    CHECK_EQ(rec5_ret, codeview::T_64PVOID);
}

TEST_CASE("CodeView Emitter - Subsections in .debug$S") {
    DebugContext ctx;
    ctx.get_or_add_file("calc.brass");

    std::vector<FunctionDebugTable> tables;
    FunctionDebugTable tbl("sum_and_mul", 0x60);
    tbl.set_decl_file(1);
    tbl.set_decl_line(10);
    tbl.set_prologue_size(16);
    tbl.add_line_entry(0x0, DebugLoc(1, 10, 1));
    tbl.add_line_entry(0x10, DebugLoc(1, 12, 4));
    tbl.add_line_entry(0x28, DebugLoc(1, 15, 2));

    DebugVariable v;
    v.name = "total";
    v.is_parameter = false;
    v.stack_offset = -16;
    v.decl_file = 1;
    v.decl_line = 11;
    tbl.add_variable(v);

    tables.push_back(tbl);

    std::vector<CompiledFunctionInfo> functions;
    CompiledFunctionInfo finfo;
    finfo.name = "sum_and_mul";
    finfo.text_offset = 0;
    finfo.text_size = 0x60;
    functions.push_back(finfo);

    Section s_sec;
    s_sec.name = ".debug$S";
    s_sec.kind = SectionKind::Custom;
    s_sec.flags = SectionFlags::Read | SectionFlags::Discardable;
    s_sec.alignment = 4;
    CodeViewEmitter::emit_debug_s(ctx, tables, functions, s_sec);

    REQUIRE(s_sec.data.size() >= 4);
    const uint8_t* p = s_sec.data.data();
    uint32_t sig = read_u32_le(p);
    CHECK_EQ(sig, codeview::CV_SIGNATURE_C13);

    bool found_stringtable = false;
    bool found_filechksms = false;
    bool found_lines = false;
    bool found_symbols = false;

    size_t off = 4;
    while (off + 8 <= s_sec.data.size()) {
        uint32_t sub_kind = read_u32_le(p + off);
        uint32_t sub_len = read_u32_le(p + off + 4);
        off += 8;

        if (sub_kind == codeview::DEBUG_S_STRINGTABLE) {
            found_stringtable = true;
            std::string st(reinterpret_cast<const char*>(p + off), sub_len);
            CHECK(st.find("calc.brass") != std::string::npos);
        } else if (sub_kind == codeview::DEBUG_S_FILECHKSMS) {
            found_filechksms = true;
        } else if (sub_kind == codeview::DEBUG_S_LINES) {
            found_lines = true;
            REQUIRE(sub_len >= 24);
            // Lines subsection header: offset, section index, flags, code size
            uint32_t code_off = read_u32_le(p + off);
            CHECK_EQ(code_off, 0u);
            uint32_t code_sz = read_u32_le(p + off + 8);
            CHECK_EQ(code_sz, 0x60u);
        } else if (sub_kind == codeview::DEBUG_S_SYMBOLS) {
            found_symbols = true;
            // Scan symbol records inside subsection
            size_t sym_off = off;
            size_t sym_end = off + sub_len;
            bool found_proc = false;
            bool found_regrel = false;
            bool found_proc_end = false;

            while (sym_off + 4 <= sym_end) {
                uint16_t r_len = read_u16_le(p + sym_off);
                uint16_t r_kind = read_u16_le(p + sym_off + 2);
                if (r_kind == codeview::S_GPROC32) found_proc = true;
                if (r_kind == codeview::S_REGREL32) {
                    found_regrel = true;
                    // Register is at sym_off + 12 (len:2, kind:2, off:4, type:4 -> reg:2)
                    uint16_t reg = read_u16_le(p + sym_off + 12);
                    CHECK_EQ(reg, codeview::CV_AMD64_RBP);
                }
                if (r_kind == codeview::S_PROC_ID_END) found_proc_end = true;

                sym_off += 2 + r_len;
            }
            CHECK(found_proc);
            CHECK(found_regrel);
            CHECK(found_proc_end);
        }

        off += sub_len;
        if (off % 4 != 0) {
            off += (4 - (off % 4));
        }
    }

    CHECK(found_stringtable);
    CHECK(found_filechksms);
    CHECK(found_lines);
    CHECK(found_symbols);
}

TEST_CASE("CodeView Emitter - Full COFF Object Integration") {
    Module mod("test_cv_mod");
    Function* fn = mod.create_function("cv_func", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.build_iconst_i64(100);
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);
    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);

    REQUIRE(coff_bytes.size() >= 20);
    const uint8_t* hdr = coff_bytes.data();
    uint16_t num_sec = read_u16_le(hdr + 2);

    const uint8_t* sec_hdrs = hdr + 20;
    bool found_debug_s = false;
    bool found_debug_t = false;

    for (uint16_t i = 0; i < num_sec; ++i) {
        const uint8_t* sh = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sh, 8);
        uint32_t raw_sz = read_u32_le(sh + 16);
        uint16_t num_relocs = read_u16_le(sh + 32);
        uint32_t flags = read_u32_le(sh + 36);

        if (std::strcmp(name, ".debug$S") == 0) {
            found_debug_s = true;
            CHECK(raw_sz > 0);
            CHECK(num_relocs > 0);
            // Characteristics: IMAGE_SCN_MEM_READ (0x40000000) | IMAGE_SCN_MEM_DISCARDABLE (0x02000000)
            // | IMAGE_SCN_CNT_INITIALIZED_DATA (0x00000040) | IMAGE_SCN_ALIGN_4BYTES (0x00300000) = 0x42300040
            CHECK_EQ(flags, 0x42300040u);
        }
        if (std::strcmp(name, ".debug$T") == 0) {
            found_debug_t = true;
            CHECK(raw_sz > 0);
            CHECK_EQ(flags, 0x42300040u);
        }
    }

    CHECK(found_debug_s);
    CHECK(found_debug_t);
}