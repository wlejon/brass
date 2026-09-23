#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <brass/debug/debug_section.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>

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

CompiledFunctionInfo make_fn(const char* name, Type ret, std::vector<Type> params) {
    CompiledFunctionInfo f;
    f.name = name;
    f.return_type = ret;
    f.param_types = std::move(params);
    f.text_size = 0x10;
    return f;
}

// A parsed type record of .debug$T: its leaf kind and the bytes after it.
struct TypeRecord {
    uint16_t kind = 0;
    std::vector<uint8_t> body;
};

// Parses .debug$T records by type index; checks each record's 4-byte padding.
std::map<uint32_t, TypeRecord> parse_types(const std::vector<uint8_t>& data) {
    std::map<uint32_t, TypeRecord> types;
    REQUIRE(data.size() >= 4);
    CHECK_EQ(read_u32_le(data.data()), codeview::CV_SIGNATURE_C13);
    size_t off = 4;
    uint32_t idx = codeview::FIRST_TYPE_INDEX;
    while (off < data.size()) {
        REQUIRE(off + 4 <= data.size());
        uint16_t len = read_u16_le(data.data() + off);
        CHECK_EQ((len + 2u) % 4u, 0u);
        REQUIRE(off + 2 + len <= data.size());
        TypeRecord rec;
        rec.kind = read_u16_le(data.data() + off + 2);
        rec.body.assign(data.begin() + off + 4, data.begin() + off + 2 + len);
        types[idx++] = std::move(rec);
        off += 2 + len;
    }
    return types;
}

// Checks that `proc_index` is an LF_PROCEDURE returning `ret` with `params`.
void check_procedure(const std::map<uint32_t, TypeRecord>& types, uint32_t proc_index,
                     uint32_t ret, const std::vector<uint32_t>& params) {
    auto pit = types.find(proc_index);
    REQUIRE(pit != types.end());
    const TypeRecord& proc = pit->second;
    REQUIRE_EQ(proc.kind, codeview::LF_PROCEDURE);
    REQUIRE(proc.body.size() >= 12);
    CHECK_EQ(read_u32_le(proc.body.data()), ret);
    CHECK_EQ(uint32_t(read_u16_le(proc.body.data() + 6)), uint32_t(params.size()));
    auto ait = types.find(read_u32_le(proc.body.data() + 8));
    REQUIRE(ait != types.end());
    const TypeRecord& args = ait->second;
    REQUIRE_EQ(args.kind, codeview::LF_ARGLIST);
    REQUIRE(args.body.size() >= 4 + 4 * params.size());
    REQUIRE_EQ(read_u32_le(args.body.data()), uint32_t(params.size()));
    for (size_t i = 0; i < params.size(); ++i) {
        CHECK_EQ(read_u32_le(args.body.data() + 4 + 4 * i), params[i]);
    }
}

// A section of a COFF object, with its relocations resolved to symbol names.
struct CoffSection {
    std::vector<uint8_t> data;
    std::map<uint32_t, std::pair<uint16_t, std::string>> relocs; // offset -> (type, symbol)
};

std::map<std::string, CoffSection> parse_coff(const std::vector<uint8_t>& bytes) {
    REQUIRE(bytes.size() >= 20);
    const uint8_t* b = bytes.data();
    uint16_t num_sec = read_u16_le(b + 2);
    uint32_t sym_ptr = read_u32_le(b + 8);
    uint32_t num_syms = read_u32_le(b + 12);
    size_t str_ptr = sym_ptr + size_t(num_syms) * 18;
    REQUIRE(str_ptr <= bytes.size());

    std::vector<std::string> sym_names(num_syms);
    for (uint32_t i = 0; i < num_syms; ++i) {
        const uint8_t* s = b + sym_ptr + size_t(i) * 18;
        if (read_u32_le(s) == 0) {
            sym_names[i] = reinterpret_cast<const char*>(b + str_ptr + read_u32_le(s + 4));
        } else {
            char name[9] = {0};
            std::memcpy(name, s, 8);
            sym_names[i] = name;
        }
        i += s[17]; // skip aux records
    }

    std::map<std::string, CoffSection> sections;
    for (uint16_t i = 0; i < num_sec; ++i) {
        const uint8_t* sh = b + 20 + size_t(i) * 40;
        char name[9] = {0};
        std::memcpy(name, sh, 8);
        CoffSection sec;
        uint32_t raw_size = read_u32_le(sh + 16);
        uint32_t raw_ptr = read_u32_le(sh + 20);
        REQUIRE(size_t(raw_ptr) + raw_size <= bytes.size());
        sec.data.assign(b + raw_ptr, b + raw_ptr + raw_size);
        uint32_t reloc_ptr = read_u32_le(sh + 24);
        uint16_t num_relocs = read_u16_le(sh + 32);
        for (uint16_t r = 0; r < num_relocs; ++r) {
            const uint8_t* rp = b + reloc_ptr + size_t(r) * 10;
            uint32_t sym = read_u32_le(rp + 4);
            REQUIRE(sym < num_syms);
            sec.relocs[read_u32_le(rp)] = {read_u16_le(rp + 8), sym_names[sym]};
        }
        sections[name] = std::move(sec);
    }
    return sections;
}

constexpr uint16_t kRelAmd64Section = 0x000A;
constexpr uint16_t kRelAmd64SecRel = 0x000B;

// Checks that a code reference at `off` (SECREL + SECTION) targets `symbol`.
void check_code_ref(const CoffSection& sec, uint32_t off, const std::string& symbol) {
    auto r_off = sec.relocs.find(off);
    auto r_seg = sec.relocs.find(off + 4);
    REQUIRE(r_off != sec.relocs.end());
    REQUIRE(r_seg != sec.relocs.end());
    CHECK_EQ(r_off->second.first, kRelAmd64SecRel);
    CHECK_EQ(r_off->second.second, symbol);
    CHECK_EQ(r_seg->second.first, kRelAmd64Section);
    CHECK_EQ(r_seg->second.second, symbol);
}

} // namespace

TEST_CASE("CodeView Emitter - Procedure types from MIR signatures (.debug$T)") {
    std::vector<CompiledFunctionInfo> functions = {
        make_fn("a", Type::i32(), {Type::i64(), Type::f64()}),
        make_fn("b", Type::void_type(), {}),
        make_fn("c", Type::i32(), {Type::i64(), Type::f64()}),
        make_fn("d", Type::f32(), {Type(TypeKind::F32x4), Type::ptr()}),
    };

    Section t_sec;
    CodeViewEmitter::emit_debug_t(functions, t_sec);
    auto types = parse_types(t_sec.data);

    CodeViewTypeTable table = CodeViewEmitter::build_type_table(functions);
    REQUIRE_EQ(table.proc_types.size(), size_t(4));
    // Identical signatures share one record.
    CHECK_EQ(table.proc_types[0], table.proc_types[2]);
    CHECK(table.proc_types[0] != table.proc_types[1]);

    check_procedure(types, table.proc_types[0], codeview::T_INT4, {codeview::T_INT8, codeview::T_REAL64});
    check_procedure(types, table.proc_types[1], codeview::T_VOID, {});

    // A vector parameter is an LF_ARRAY of its lanes.
    auto dit = types.find(table.proc_types[3]);
    REQUIRE(dit != types.end());
    uint32_t d_args = read_u32_le(dit->second.body.data() + 8);
    uint32_t vec_type = read_u32_le(types.at(d_args).body.data() + 4);
    check_procedure(types, table.proc_types[3], codeview::T_REAL32, {vec_type, codeview::T_64PVOID});
    const TypeRecord& arr = types.at(vec_type);
    REQUIRE_EQ(arr.kind, codeview::LF_ARRAY);
    CHECK_EQ(read_u32_le(arr.body.data()), codeview::T_REAL32);
    CHECK_EQ(read_u32_le(arr.body.data() + 4), codeview::T_UQUAD);
    CHECK_EQ(read_u16_le(arr.body.data() + 8), uint16_t(16));
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
                // S_GPROC32 is closed by S_END (S_PROC_ID_END pairs with S_GPROC32_ID)
                CHECK(r_kind != codeview::S_PROC_ID_END);
                if (r_kind == codeview::S_END) found_proc_end = true;

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

TEST_CASE("CodeView Emitter - Multi-function COFF object is well-formed") {
    // Three functions with distinct signatures. The module registers no
    // source file, so the file is named after the module.
    Module mod("cv_multi");
    Builder b(mod);
    {
        Function* fn = mod.create_function("cv_add", Type::i64(), {Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, y));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }
    {
        Function* fn = mod.create_function("cv_twice", Type::f64(), {Type::f64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::f64());
        b.build_ret(b.build_fadd(x, x));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }
    {
        Function* fn = mod.create_function("cv_seven", Type::i32(), {});
        b.set_function(fn);
        b.append_block("entry");
        b.build_ret(b.build_iconst_i32(7));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    auto sections = parse_coff(coff_bytes);
    REQUIRE(sections.count(".debug$S") == 1);
    REQUIRE(sections.count(".debug$T") == 1);
    const CoffSection& ds = sections.at(".debug$S");
    auto types = parse_types(sections.at(".debug$T").data);

    struct Expected {
        uint32_t ret;
        std::vector<uint32_t> params;
    };
    std::map<std::string, Expected> expected = {
        {"cv_add", {codeview::T_INT8, {codeview::T_INT8, codeview::T_INT8}}},
        {"cv_twice", {codeview::T_REAL64, {codeview::T_REAL64}}},
        {"cv_seven", {codeview::T_INT4, {}}},
    };

    const uint8_t* p = ds.data.data();
    REQUIRE(ds.data.size() >= 4);
    CHECK_EQ(read_u32_le(p), codeview::CV_SIGNATURE_C13);

    std::string string_table;
    std::set<std::string> line_table_fns;
    std::set<std::string> proc_fns;
    size_t off = 4;
    while (off < ds.data.size()) {
        REQUIRE(off + 8 <= ds.data.size());
        uint32_t sub_kind = read_u32_le(p + off);
        uint32_t sub_len = read_u32_le(p + off + 4);
        off += 8;
        REQUIRE(off + sub_len <= ds.data.size());

        if (sub_kind == codeview::DEBUG_S_STRINGTABLE) {
            string_table.assign(reinterpret_cast<const char*>(p + off), sub_len);
        } else if (sub_kind == codeview::DEBUG_S_LINES) {
            // Each line table is keyed by the symbol its header is relocated
            // against, so each function needs its own.
            auto r = ds.relocs.find(static_cast<uint32_t>(off));
            REQUIRE(r != ds.relocs.end());
            check_code_ref(ds, static_cast<uint32_t>(off), r->second.second);
            CHECK(line_table_fns.insert(r->second.second).second);
        } else if (sub_kind == codeview::DEBUG_S_SYMBOLS) {
            size_t sym_off = off;
            size_t sym_end = off + sub_len;
            int depth = 0;
            while (sym_off < sym_end) {
                REQUIRE(sym_off + 4 <= sym_end);
                uint16_t r_len = read_u16_le(p + sym_off);
                uint16_t r_kind = read_u16_le(p + sym_off + 2);
                REQUIRE(sym_off + 2 + r_len <= sym_end);
                if (r_kind == codeview::S_GPROC32) {
                    CHECK_EQ(depth, 0);
                    ++depth;
                    // len kind parent end next codesize dbgstart dbgend type off seg flags name
                    const uint8_t* rec = p + sym_off;
                    std::string name(reinterpret_cast<const char*>(rec + 39));
                    REQUIRE(expected.count(name) == 1);
                    check_procedure(types, read_u32_le(rec + 28), expected.at(name).ret, expected.at(name).params);
                    check_code_ref(ds, static_cast<uint32_t>(sym_off + 32), name);
                    CHECK(proc_fns.insert(name).second);
                } else if (r_kind == codeview::S_END) {
                    CHECK_EQ(depth, 1);
                    --depth;
                } else {
                    CHECK(r_kind != codeview::S_PROC_ID_END);
                }
                sym_off += 2 + r_len;
            }
            CHECK_EQ(depth, 0);
        }
        off += sub_len;
        if (off % 4 != 0) off += 4 - (off % 4);
    }

    CHECK_EQ(proc_fns.size(), size_t(3));
    CHECK_EQ(line_table_fns.size(), size_t(3));
    for (const auto& name : line_table_fns) CHECK(expected.count(name) == 1);
    CHECK(string_table.find(std::string("cv_multi") + '\0') != std::string::npos);
    CHECK(string_table.find("source.js") == std::string::npos);

    // Optional: llvm-readobj must also parse it, when it is on PATH.
#if defined(_WIN32)
    const char* null_dev = "NUL";
#else
    const char* null_dev = "/dev/null";
#endif
    std::string quiet = std::string(" >") + null_dev + " 2>&1";
    if (std::system(("llvm-readobj --version" + quiet).c_str()) == 0) {
        std::filesystem::path path = brass::test::scratch_dir() / "cv_multi.obj";
        {
            std::ofstream f(path, std::ios::binary);
            REQUIRE(static_cast<bool>(f));
            f.write(reinterpret_cast<const char*>(coff_bytes.data()),
                    static_cast<std::streamsize>(coff_bytes.size()));
        }
        std::string cmd = "llvm-readobj --codeview \"" + path.string() + "\"" + quiet;
        CHECK_EQ(std::system(cmd.c_str()), 0);
    } else {
        std::cout << "  [SKIP] llvm-readobj not on PATH: external CodeView parse not run\n";
    }
}
