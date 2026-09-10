#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/debug/dwarf_emitter.hpp>
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

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

} // namespace

TEST_CASE("DWARF - ULEB128 and SLEB128 round-trip encoding") {
    std::vector<uint8_t> buf;

    std::vector<uint64_t> uvals = {0, 1, 63, 64, 127, 128, 255, 300, 16384, 0xDEADBEEF, 0x123456789ABCDEF0ULL};
    for (uint64_t val : uvals) {
        buf.clear();
        encode_uleb128(buf, val);
        const uint8_t* ptr = buf.data();
        const uint8_t* end = buf.data() + buf.size();
        uint64_t decoded = decode_uleb128(ptr, end);
        CHECK_EQ(decoded, val);
        CHECK_EQ(ptr, end);
    }

    std::vector<int64_t> svals = {0, 1, -1, 63, -64, 64, -65, 127, -128, 128, -129, -500, 500, -0x12345678LL, 0x12345678LL};
    for (int64_t val : svals) {
        buf.clear();
        encode_sleb128(buf, val);
        const uint8_t* ptr = buf.data();
        const uint8_t* end = buf.data() + buf.size();
        int64_t decoded = decode_sleb128(ptr, end);
        CHECK_EQ(decoded, val);
        CHECK_EQ(ptr, end);
    }
}

TEST_CASE("DWARF Line Program - Header and State Machine Execution") {
    DebugContext ctx;
    uint32_t fid = ctx.get_or_add_file("src/test.brass");
    CHECK_EQ(fid, 1u);

    std::vector<FunctionDebugTable> tables;
    FunctionDebugTable tbl("my_func", 0x40);
    tbl.add_line_entry(0x0, DebugLoc(1, 10, 1));
    tbl.add_line_entry(0x8, DebugLoc(1, 12, 5));
    tbl.add_line_entry(0x14, DebugLoc(1, 15, 3));
    tbl.add_line_entry(0x28, DebugLoc(1, 20, 1));
    tables.push_back(tbl);

    std::vector<CompiledFunctionInfo> functions;
    CompiledFunctionInfo finfo;
    finfo.name = "my_func";
    finfo.text_offset = 0;
    finfo.text_size = 0x40;
    functions.push_back(finfo);

    Section line_sec;
    DwarfLineEmitter emitter;
    emitter.emit(ctx, tables, functions, line_sec);

    REQUIRE(line_sec.data.size() >= 30);
    const uint8_t* data = line_sec.data.data();

    uint32_t unit_length = read_u32_le(data + 0);
    CHECK_EQ(unit_length + 4, static_cast<uint32_t>(line_sec.data.size()));

    uint16_t version = read_u16_le(data + 4);
    CHECK_EQ(version, uint16_t(4));

    uint32_t header_length = read_u32_le(data + 6);
    CHECK(header_length > 0);

    uint8_t min_insn_len = data[10];
    CHECK_EQ(min_insn_len, uint8_t(1));

    uint8_t max_ops = data[11];
    CHECK_EQ(max_ops, uint8_t(1));

    uint8_t default_is_stmt = data[12];
    CHECK_EQ(default_is_stmt, uint8_t(1));

    int8_t line_base = static_cast<int8_t>(data[13]);
    CHECK_EQ(line_base, int8_t(-5));

    uint8_t line_range = data[14];
    CHECK_EQ(line_range, uint8_t(14));

    uint8_t opcode_base = data[15];
    CHECK_EQ(opcode_base, uint8_t(13));

    const uint8_t* prog_ptr = data + 10 + header_length;
    const uint8_t* prog_end = data + line_sec.data.size();

    struct Row {
        uint64_t address;
        uint32_t line;
        uint32_t column;
        uint32_t file;
    };
    std::vector<Row> decoded_matrix;

    uint64_t cur_addr = 0;
    uint32_t cur_line = 1;
    uint32_t cur_col = 0;
    uint32_t cur_file = 1;

    while (prog_ptr < prog_end) {
        uint8_t op = *prog_ptr++;
        if (op == 0) {
            uint64_t len = decode_uleb128(prog_ptr, prog_end);
            const uint8_t* sub_end = prog_ptr + len;
            uint8_t sub_op = *prog_ptr++;
            if (sub_op == dwarf::DW_LNE_set_address) {
                cur_addr = read_u64_le(prog_ptr);
                prog_ptr += 8;
            } else if (sub_op == dwarf::DW_LNE_end_sequence) {
                decoded_matrix.push_back({cur_addr, cur_line, cur_col, cur_file});
                cur_addr = 0;
                cur_line = 1;
                cur_col = 0;
                cur_file = 1;
            }
            prog_ptr = sub_end;
        } else if (op < opcode_base) {
            switch (op) {
            case dwarf::DW_LNS_copy:
                decoded_matrix.push_back({cur_addr, cur_line, cur_col, cur_file});
                break;
            case dwarf::DW_LNS_advance_pc:
                cur_addr += decode_uleb128(prog_ptr, prog_end) * min_insn_len;
                break;
            case dwarf::DW_LNS_advance_line:
                cur_line = static_cast<uint32_t>(static_cast<int64_t>(cur_line) + decode_sleb128(prog_ptr, prog_end));
                break;
            case dwarf::DW_LNS_set_file:
                cur_file = static_cast<uint32_t>(decode_uleb128(prog_ptr, prog_end));
                break;
            case dwarf::DW_LNS_set_column:
                cur_col = static_cast<uint32_t>(decode_uleb128(prog_ptr, prog_end));
                break;
            default:
                break;
            }
        } else {
            uint8_t adj = op - opcode_base;
            uint64_t addr_adv = (adj / line_range) * min_insn_len;
            int64_t line_adv = line_base + static_cast<int64_t>(adj % line_range);
            cur_addr += addr_adv;
            cur_line = static_cast<uint32_t>(static_cast<int64_t>(cur_line) + line_adv);
            decoded_matrix.push_back({cur_addr, cur_line, cur_col, cur_file});
        }
    }

    REQUIRE(decoded_matrix.size() >= 4);
    CHECK_EQ(decoded_matrix[0].address, 0u);
    CHECK_EQ(decoded_matrix[0].line, 10u);
    CHECK_EQ(decoded_matrix[0].column, 1u);

    CHECK_EQ(decoded_matrix[1].address, 8u);
    CHECK_EQ(decoded_matrix[1].line, 12u);
    CHECK_EQ(decoded_matrix[1].column, 5u);

    CHECK_EQ(decoded_matrix[2].address, 0x14u);
    CHECK_EQ(decoded_matrix[2].line, 15u);
    CHECK_EQ(decoded_matrix[2].column, 3u);

    CHECK_EQ(decoded_matrix[3].address, 0x28u);
    CHECK_EQ(decoded_matrix[3].line, 20u);
    CHECK_EQ(decoded_matrix[3].column, 1u);
}

TEST_CASE("DWARF Info Program - Compilation Unit and Subprograms DIE Tree") {
    DebugContext ctx;
    ctx.get_or_add_file("main.brass");

    std::vector<FunctionDebugTable> tables;
    FunctionDebugTable tbl("calculate", 0x50);
    tbl.set_decl_file(1);
    tbl.set_decl_line(5);
    tbl.set_prologue_size(8);

    DebugVariable p1;
    p1.name = "arg_x";
    p1.is_parameter = true;
    p1.stack_offset = 16;
    p1.decl_file = 1;
    p1.decl_line = 5;
    tbl.add_variable(p1);

    DebugVariable v1;
    v1.name = "loc_temp";
    v1.is_parameter = false;
    v1.stack_offset = -8;
    v1.decl_file = 1;
    v1.decl_line = 7;
    tbl.add_variable(v1);

    tables.push_back(tbl);

    std::vector<CompiledFunctionInfo> functions;
    CompiledFunctionInfo finfo;
    finfo.name = "calculate";
    finfo.text_offset = 0;
    finfo.text_size = 0x50;
    functions.push_back(finfo);

    Section info_sec;
    Section abbrev_sec;
    Section str_sec;

    DwarfOptions opts;
    opts.version = 4;
    opts.producer = "brass 1.0 test";
    opts.comp_dir = "/test/dir";

    DwarfInfoEmitter emitter(opts);
    emitter.emit(ctx, tables, functions, info_sec, abbrev_sec, str_sec, 0x50);

    CHECK(!str_sec.data.empty());
    std::string str_content(reinterpret_cast<const char*>(str_sec.data.data()), str_sec.data.size());
    CHECK(str_content.find("brass 1.0 test") != std::string::npos);
    CHECK(str_content.find("calculate") != std::string::npos);
    CHECK(str_content.find("arg_x") != std::string::npos);
    CHECK(str_content.find("loc_temp") != std::string::npos);

    CHECK(!abbrev_sec.data.empty());
    const uint8_t* a_ptr = abbrev_sec.data.data();
    const uint8_t* a_end = a_ptr + abbrev_sec.data.size();

    uint64_t code1 = decode_uleb128(a_ptr, a_end);
    CHECK_EQ(code1, 1u);
    uint64_t tag1 = decode_uleb128(a_ptr, a_end);
    CHECK_EQ(tag1, static_cast<uint64_t>(dwarf::DW_TAG_compile_unit));
    uint8_t child1 = *a_ptr++;
    CHECK_EQ(child1, dwarf::DW_CHILDREN_yes);

    CHECK(!info_sec.data.empty());
    uint32_t ulen = read_u32_le(info_sec.data.data());
    CHECK_EQ(ulen + 4, static_cast<uint32_t>(info_sec.data.size()));
    uint16_t ver = read_u16_le(info_sec.data.data() + 4);
    CHECK_EQ(ver, uint16_t(4));
    uint8_t addr_sz = info_sec.data[10];
    CHECK_EQ(addr_sz, uint8_t(8));

    CHECK(!info_sec.relocations.empty());
    bool has_abbrev_reloc = false;
    bool has_str_reloc = false;
    bool has_text_reloc = false;
    for (const auto& r : info_sec.relocations) {
        if (r.symbol_name == ".debug_abbrev") has_abbrev_reloc = true;
        if (r.symbol_name == ".debug_str") has_str_reloc = true;
        if (r.symbol_name == ".text" || r.symbol_name == "calculate") has_text_reloc = true;
    }
    CHECK(has_abbrev_reloc);
    CHECK(has_str_reloc);
    CHECK(has_text_reloc);
}

TEST_CASE("DWARF Emitter - Full Object Integration") {
    Module mod("test_dwarf_mod");
    Function* fn = mod.create_function("func_dwarf", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p = b.add_block_param(entry, Type::i32());
    Value* c = b.build_iconst_i32(42);
    Value* res = b.build_add(p, c);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_linux());
    DwarfOptions opts;
    DwarfEmitter::emit(obj, opts);

    const Section* line_sec = obj.get_section(".debug_line");
    const Section* info_sec = obj.get_section(".debug_info");
    const Section* abbrev_sec = obj.get_section(".debug_abbrev");
    const Section* str_sec = obj.get_section(".debug_str");

    REQUIRE(line_sec != nullptr);
    REQUIRE(info_sec != nullptr);
    REQUIRE(abbrev_sec != nullptr);
    REQUIRE(str_sec != nullptr);

    CHECK(!line_sec->data.empty());
    CHECK(!info_sec->data.empty());
    CHECK(!abbrev_sec->data.empty());
    CHECK(!str_sec->data.empty());
}