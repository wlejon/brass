#include <brass/debug/dwarf_emitter.hpp>
#include <algorithm>
#include <unordered_map>
#include <cstring>

namespace brass::debug {

namespace {

void patch_u32(std::vector<uint8_t>& buf, size_t offset, uint32_t val) {
    if (offset + 4 <= buf.size()) {
        buf[offset + 0] = static_cast<uint8_t>(val & 0xFF);
        buf[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        buf[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        buf[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    }
}

void write_string(std::vector<uint8_t>& buf, const std::string& str) {
    buf.insert(buf.end(), str.begin(), str.end());
    buf.push_back(0);
}

} // namespace

void encode_uleb128(std::vector<uint8_t>& buf, uint64_t val) {
    do {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        if (val != 0) byte |= 0x80;
        buf.push_back(byte);
    } while (val != 0);
}

void encode_sleb128(std::vector<uint8_t>& buf, int64_t val) {
    bool more = true;
    while (more) {
        uint8_t byte = static_cast<uint8_t>(val & 0x7F);
        val >>= 7;
        bool sign_bit = (byte & 0x40) != 0;
        if ((val == 0 && !sign_bit) || (val == -1 && sign_bit)) {
            more = false;
        } else {
            byte |= 0x80;
        }
        buf.push_back(byte);
    }
}

uint64_t decode_uleb128(const uint8_t*& ptr, const uint8_t* end) {
    uint64_t result = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        result |= (static_cast<uint64_t>(byte & 0x7F) << shift);
        if ((byte & 0x80) == 0) return result;
        shift += 7;
        if (shift >= 64) break;
    }
    return result;
}

int64_t decode_sleb128(const uint8_t*& ptr, const uint8_t* end) {
    int64_t result = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        result |= (static_cast<int64_t>(byte & 0x7F) << shift);
        shift += 7;
        if ((byte & 0x80) == 0) {
            if (shift < 64 && (byte & 0x40) != 0) {
                result |= (~0ULL << shift);
            }
            return result;
        }
        if (shift >= 64) break;
    }
    return result;
}

DwarfLineEmitter::DwarfLineEmitter(DwarfOptions opts)
    : opts_(std::move(opts)) {}

void DwarfLineEmitter::emit(
    const DebugContext& ctx,
    const std::vector<FunctionDebugTable>& tables,
    const std::vector<object::CompiledFunctionInfo>& functions,
    object::Section& line_sec
) {
    size_t unit_start = line_sec.data.size();

    // 1. Prologue
    line_sec.emit32(0); // Placeholder for unit_length (offset 0)
    line_sec.emit16(opts_.version); // DWARF version 4
    line_sec.emit32(0); // Placeholder for header_length (offset 6)

    size_t header_body_start = line_sec.data.size();

    line_sec.emit8(1);  // minimum_instruction_length
    line_sec.emit8(1);  // maximum_operations_per_instruction (for DWARF 4)
    line_sec.emit8(1);  // default_is_stmt
    line_sec.emit8(static_cast<uint8_t>(dwarf::DWARF_LINE_BASE));
    line_sec.emit8(dwarf::DWARF_LINE_RANGE);
    line_sec.emit8(dwarf::DWARF_OPCODE_BASE);

    // Standard opcode lengths (12 opcodes)
    uint8_t std_opcode_lens[12] = {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1};
    for (int i = 0; i < 12; ++i) {
        line_sec.emit8(std_opcode_lens[i]);
    }

    // Directory table
    write_string(line_sec.data, opts_.comp_dir.empty() ? "." : opts_.comp_dir);
    line_sec.emit8(0); // End of directory table

    // File table
    if (ctx.file_count() > 0) {
        for (const auto& file_path : ctx.files()) {
            write_string(line_sec.data, file_path);
            encode_uleb128(line_sec.data, 1); // dir_index = 1
            encode_uleb128(line_sec.data, 0); // mtime = 0
            encode_uleb128(line_sec.data, 0); // length = 0
        }
    } else {
        write_string(line_sec.data, "source.js");
        encode_uleb128(line_sec.data, 1);
        encode_uleb128(line_sec.data, 0);
        encode_uleb128(line_sec.data, 0);
    }
    line_sec.emit8(0); // End of file table

    size_t header_body_end = line_sec.data.size();
    uint32_t header_len = static_cast<uint32_t>(header_body_end - header_body_start);
    patch_u32(line_sec.data, unit_start + 6, header_len);

    // 2. Line Number Programs
    std::unordered_map<std::string, const FunctionDebugTable*> table_map;
    for (const auto& t : tables) {
        table_map[t.function_name()] = &t;
    }

    uint64_t cur_pc = 0;
    uint32_t cur_line = 1;
    uint32_t cur_col = 0;
    uint32_t cur_file = 1;

    for (const auto& fn : functions) {
        uint64_t fn_start = fn.text_offset;
        uint64_t fn_end = fn_start + fn.text_size;

        // Set address to function start
        line_sec.emit8(0);
        encode_uleb128(line_sec.data, 9);
        line_sec.emit8(dwarf::DW_LNE_set_address);

        object::ObjectRelocation r;
        r.offset = line_sec.data.size();
        r.kind = object::RelocKind::Abs64;
        r.symbol_name = ".text";
        r.addend = static_cast<int64_t>(fn_start);
        line_sec.relocations.push_back(r);

        line_sec.emit64(fn_start);

        cur_pc = fn_start;
        cur_line = 1;
        cur_col = 0;
        cur_file = 1;

        auto it = table_map.find(fn.name);
        const FunctionDebugTable* tbl = (it != table_map.end()) ? it->second : nullptr;

        std::vector<DebugLineEntry> entries;
        if (tbl && !tbl->line_entries().empty()) {
            entries = tbl->line_entries();
            std::sort(entries.begin(), entries.end(), [](const DebugLineEntry& a, const DebugLineEntry& b) {
                return a.code_offset < b.code_offset;
            });
        } else {
            entries.push_back({0, DebugLoc(1, 1, 1)});
        }

        for (const auto& entry : entries) {
            uint64_t target_pc = fn_start + entry.code_offset;
            uint32_t target_line = entry.loc.line > 0 ? entry.loc.line : 1;
            uint32_t target_col = entry.loc.column;
            uint32_t target_file = entry.loc.file_id > 0 ? entry.loc.file_id : 1;

            if (target_file != cur_file) {
                line_sec.emit8(dwarf::DW_LNS_set_file);
                encode_uleb128(line_sec.data, target_file);
                cur_file = target_file;
            }

            if (target_col != cur_col) {
                line_sec.emit8(dwarf::DW_LNS_set_column);
                encode_uleb128(line_sec.data, target_col);
                cur_col = target_col;
            }

            int64_t line_adv = static_cast<int64_t>(target_line) - static_cast<int64_t>(cur_line);
            uint64_t pc_adv = (target_pc >= cur_pc) ? (target_pc - cur_pc) : 0;

            int64_t line_offset = line_adv - dwarf::DWARF_LINE_BASE;
            if (line_offset >= 0 && line_offset < dwarf::DWARF_LINE_RANGE) {
                uint64_t special_op = static_cast<uint64_t>(line_offset) + (pc_adv * dwarf::DWARF_LINE_RANGE) + dwarf::DWARF_OPCODE_BASE;
                if (special_op <= 255) {
                    line_sec.emit8(static_cast<uint8_t>(special_op));
                    cur_pc = target_pc;
                    cur_line = target_line;
                    continue;
                }
            }

            if (line_adv != 0) {
                line_sec.emit8(dwarf::DW_LNS_advance_line);
                encode_sleb128(line_sec.data, line_adv);
                cur_line = target_line;
            }

            if (pc_adv > 0) {
                int64_t lo = 0 - dwarf::DWARF_LINE_BASE;
                uint64_t special_op2 = static_cast<uint64_t>(lo) + (pc_adv * dwarf::DWARF_LINE_RANGE) + dwarf::DWARF_OPCODE_BASE;
                if (special_op2 <= 255) {
                    line_sec.emit8(static_cast<uint8_t>(special_op2));
                    cur_pc = target_pc;
                } else {
                    line_sec.emit8(dwarf::DW_LNS_advance_pc);
                    encode_uleb128(line_sec.data, pc_adv);
                    line_sec.emit8(dwarf::DW_LNS_copy);
                    cur_pc = target_pc;
                }
            } else {
                line_sec.emit8(dwarf::DW_LNS_copy);
            }
        }

        // End of sequence for this function
        if (cur_pc < fn_end) {
            line_sec.emit8(dwarf::DW_LNS_advance_pc);
            encode_uleb128(line_sec.data, fn_end - cur_pc);
            cur_pc = fn_end;
        }

        line_sec.emit8(0);
        encode_uleb128(line_sec.data, 1);
        line_sec.emit8(dwarf::DW_LNE_end_sequence);
    }

    uint32_t total_line_len = static_cast<uint32_t>((line_sec.data.size() - unit_start) - 4);
    patch_u32(line_sec.data, unit_start, total_line_len);
}

std::vector<uint8_t> DwarfLineEmitter::emit_standalone_line_table(
    const DebugContext& ctx,
    const std::vector<FunctionDebugTable>& tables,
    uint16_t version
) {
    DwarfOptions opts;
    opts.version = version;
    DwarfLineEmitter emitter(opts);

    object::Section sec;
    std::vector<object::CompiledFunctionInfo> functions;
    for (size_t i = 0; i < tables.size(); ++i) {
        object::CompiledFunctionInfo cfi;
        cfi.name = tables[i].function_name();
        cfi.text_offset = i * 0x1000;
        cfi.text_size = tables[i].code_size() > 0 ? tables[i].code_size() : 64;
        functions.push_back(std::move(cfi));
    }

    emitter.emit(ctx, tables, functions, sec);
    return sec.data;
}

DwarfInfoEmitter::DwarfInfoEmitter(DwarfOptions opts)
    : opts_(std::move(opts)) {}

void DwarfInfoEmitter::emit(
    const DebugContext& ctx,
    const std::vector<FunctionDebugTable>& tables,
    const std::vector<object::CompiledFunctionInfo>& functions,
    object::Section& info_sec,
    object::Section& abbrev_sec,
    object::Section& str_sec,
    uint64_t total_text_size
) {
    // 1. Setup .debug_str
    if (str_sec.data.empty()) {
        str_sec.emit8(0);
    }

    std::unordered_map<std::string, uint32_t> str_offsets;
    auto get_or_add_str = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        auto it = str_offsets.find(s);
        if (it != str_offsets.end()) return it->second;
        uint32_t off = static_cast<uint32_t>(str_sec.data.size());
        write_string(str_sec.data, s);
        str_offsets[s] = off;
        return off;
    };

    uint32_t producer_off = get_or_add_str(opts_.producer);
    uint32_t comp_dir_off = get_or_add_str(opts_.comp_dir.empty() ? "." : opts_.comp_dir);
    std::string main_file = ctx.file_count() > 0 ? ctx.files()[0] : "source.js";
    uint32_t main_file_off = get_or_add_str(main_file);

    // 2. Setup .debug_abbrev
    // Code 1: DW_TAG_compile_unit (children = yes)
    encode_uleb128(abbrev_sec.data, 1);
    encode_uleb128(abbrev_sec.data, dwarf::DW_TAG_compile_unit);
    abbrev_sec.emit8(dwarf::DW_CHILDREN_yes);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_producer);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_language);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data2);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_name);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_comp_dir);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_low_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_addr);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_high_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data8);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_stmt_list);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_sec_offset);
    encode_uleb128(abbrev_sec.data, 0);
    encode_uleb128(abbrev_sec.data, 0);

    // Code 2: DW_TAG_subprogram (children = yes)
    encode_uleb128(abbrev_sec.data, 2);
    encode_uleb128(abbrev_sec.data, dwarf::DW_TAG_subprogram);
    abbrev_sec.emit8(dwarf::DW_CHILDREN_yes);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_name);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_file);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data1);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_line);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data4);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_low_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_addr);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_high_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data8);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_frame_base);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_exprloc);
    encode_uleb128(abbrev_sec.data, 0);
    encode_uleb128(abbrev_sec.data, 0);

    // Code 3: DW_TAG_subprogram (children = no)
    encode_uleb128(abbrev_sec.data, 3);
    encode_uleb128(abbrev_sec.data, dwarf::DW_TAG_subprogram);
    abbrev_sec.emit8(dwarf::DW_CHILDREN_no);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_name);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_file);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data1);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_line);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data4);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_low_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_addr);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_high_pc);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data8);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_frame_base);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_exprloc);
    encode_uleb128(abbrev_sec.data, 0);
    encode_uleb128(abbrev_sec.data, 0);

    // Code 4: DW_TAG_formal_parameter (children = no)
    encode_uleb128(abbrev_sec.data, 4);
    encode_uleb128(abbrev_sec.data, dwarf::DW_TAG_formal_parameter);
    abbrev_sec.emit8(dwarf::DW_CHILDREN_no);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_name);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_file);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data1);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_line);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data4);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_location);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_exprloc);
    encode_uleb128(abbrev_sec.data, 0);
    encode_uleb128(abbrev_sec.data, 0);

    // Code 5: DW_TAG_variable (children = no)
    encode_uleb128(abbrev_sec.data, 5);
    encode_uleb128(abbrev_sec.data, dwarf::DW_TAG_variable);
    abbrev_sec.emit8(dwarf::DW_CHILDREN_no);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_name);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_strp);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_file);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data1);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_decl_line);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_data4);
    encode_uleb128(abbrev_sec.data, dwarf::DW_AT_location);
    encode_uleb128(abbrev_sec.data, dwarf::DW_FORM_exprloc);
    encode_uleb128(abbrev_sec.data, 0);
    encode_uleb128(abbrev_sec.data, 0);

    // Terminating 0 byte for abbrev table
    abbrev_sec.emit8(0);

    // 3. Setup .debug_info
    size_t info_start = info_sec.data.size();
    info_sec.emit32(0); // Placeholder for unit_length
    info_sec.emit16(opts_.version); // DWARF version 4

    // Relocation for debug_abbrev_offset (offset 6)
    object::ObjectRelocation r_abbrev;
    r_abbrev.offset = info_sec.data.size();
    r_abbrev.kind = object::RelocKind::SecRel32;
    r_abbrev.symbol_name = ".debug_abbrev";
    r_abbrev.addend = 0;
    info_sec.relocations.push_back(r_abbrev);
    info_sec.emit32(0);

    info_sec.emit8(8); // Address size = 8 bytes

    // CU DIE (Code 1)
    encode_uleb128(info_sec.data, 1);

    // DW_AT_producer (strp)
    object::ObjectRelocation r_prod;
    r_prod.offset = info_sec.data.size();
    r_prod.kind = object::RelocKind::Abs32;
    r_prod.symbol_name = ".debug_str";
    r_prod.addend = static_cast<int64_t>(producer_off);
    info_sec.relocations.push_back(r_prod);
    info_sec.emit32(producer_off);

    // DW_AT_language (data2)
    info_sec.emit16(dwarf::DW_LANG_C99);

    // DW_AT_name (strp)
    object::ObjectRelocation r_name;
    r_name.offset = info_sec.data.size();
    r_name.kind = object::RelocKind::Abs32;
    r_name.symbol_name = ".debug_str";
    r_name.addend = static_cast<int64_t>(main_file_off);
    info_sec.relocations.push_back(r_name);
    info_sec.emit32(main_file_off);

    // DW_AT_comp_dir (strp)
    object::ObjectRelocation r_dir;
    r_dir.offset = info_sec.data.size();
    r_dir.kind = object::RelocKind::Abs32;
    r_dir.symbol_name = ".debug_str";
    r_dir.addend = static_cast<int64_t>(comp_dir_off);
    info_sec.relocations.push_back(r_dir);
    info_sec.emit32(comp_dir_off);

    // DW_AT_low_pc (addr)
    object::ObjectRelocation r_low_pc;
    r_low_pc.offset = info_sec.data.size();
    r_low_pc.kind = object::RelocKind::Abs64;
    r_low_pc.symbol_name = ".text";
    r_low_pc.addend = 0;
    info_sec.relocations.push_back(r_low_pc);
    info_sec.emit64(0);

    // DW_AT_high_pc (data8)
    info_sec.emit64(total_text_size);

    // DW_AT_stmt_list (sec_offset)
    object::ObjectRelocation r_stmt;
    r_stmt.offset = info_sec.data.size();
    r_stmt.kind = object::RelocKind::SecRel32;
    r_stmt.symbol_name = ".debug_line";
    r_stmt.addend = 0;
    info_sec.relocations.push_back(r_stmt);
    info_sec.emit32(0);

    // Subprograms
    std::unordered_map<std::string, const FunctionDebugTable*> table_map;
    for (const auto& t : tables) {
        table_map[t.function_name()] = &t;
    }

    for (const auto& fn : functions) {
        auto it = table_map.find(fn.name);
        const FunctionDebugTable* tbl = (it != table_map.end()) ? it->second : nullptr;

        bool has_children = tbl && !tbl->variables().empty();
        uint32_t fn_name_off = get_or_add_str(fn.name);

        encode_uleb128(info_sec.data, has_children ? 2 : 3);

        // DW_AT_name
        object::ObjectRelocation r_fn_name;
        r_fn_name.offset = info_sec.data.size();
        r_fn_name.kind = object::RelocKind::Abs32;
        r_fn_name.symbol_name = ".debug_str";
        r_fn_name.addend = static_cast<int64_t>(fn_name_off);
        info_sec.relocations.push_back(r_fn_name);
        info_sec.emit32(fn_name_off);

        // DW_AT_decl_file
        uint8_t d_file = static_cast<uint8_t>(tbl ? tbl->decl_file() : 1);
        info_sec.emit8(d_file > 0 ? d_file : 1);

        // DW_AT_decl_line
        uint32_t d_line = tbl ? tbl->decl_line() : 1;
        info_sec.emit32(d_line > 0 ? d_line : 1);

        // DW_AT_low_pc
        object::ObjectRelocation r_fn_low;
        r_fn_low.offset = info_sec.data.size();
        r_fn_low.kind = object::RelocKind::Abs64;
        r_fn_low.symbol_name = ".text";
        r_fn_low.addend = static_cast<int64_t>(fn.text_offset);
        info_sec.relocations.push_back(r_fn_low);
        info_sec.emit64(fn.text_offset);

        // DW_AT_high_pc
        info_sec.emit64(fn.text_size);

        // DW_AT_frame_base: DW_FORM_exprloc (1 byte, DW_OP_reg6)
        encode_uleb128(info_sec.data, 1);
        info_sec.emit8(dwarf::DW_OP_reg6);

        if (has_children) {
            for (const auto& var : tbl->variables()) {
                encode_uleb128(info_sec.data, var.is_parameter ? 4 : 5);

                uint32_t var_name_off = get_or_add_str(var.name);
                object::ObjectRelocation r_var_name;
                r_var_name.offset = info_sec.data.size();
                r_var_name.kind = object::RelocKind::Abs32;
                r_var_name.symbol_name = ".debug_str";
                r_var_name.addend = static_cast<int64_t>(var_name_off);
                info_sec.relocations.push_back(r_var_name);
                info_sec.emit32(var_name_off);

                info_sec.emit8(static_cast<uint8_t>(var.decl_file > 0 ? var.decl_file : 1));
                info_sec.emit32(var.decl_line > 0 ? var.decl_line : 1);

                // DW_AT_location: DW_OP_fbreg <stack_offset>
                std::vector<uint8_t> loc_expr;
                loc_expr.push_back(dwarf::DW_OP_fbreg);
                encode_sleb128(loc_expr, var.stack_offset);

                encode_uleb128(info_sec.data, loc_expr.size());
                info_sec.emit_bytes(loc_expr.data(), loc_expr.size());
            }

            // End of subprogram children
            info_sec.emit8(0);
        }
    }

    // End of CU children
    info_sec.emit8(0);

    uint32_t total_info_len = static_cast<uint32_t>((info_sec.data.size() - info_start) - 4);
    patch_u32(info_sec.data, info_start, total_info_len);
}

void DwarfEmitter::emit(object::ObjectFile& obj, const DwarfOptions& opts) {
    obj.get_or_create_section(".debug_line", object::SectionKind::Custom, object::SectionFlags::Read, 1);
    obj.get_or_create_section(".debug_abbrev", object::SectionKind::Custom, object::SectionFlags::Read, 1);
    obj.get_or_create_section(".debug_info", object::SectionKind::Custom, object::SectionFlags::Read, 1);
    obj.get_or_create_section(".debug_str", object::SectionKind::Custom, object::SectionFlags::Read, 1);

    auto* line_sec = obj.get_section(".debug_line");
    auto* abbrev_sec = obj.get_section(".debug_abbrev");
    auto* info_sec = obj.get_section(".debug_info");
    auto* str_sec = obj.get_section(".debug_str");

    uint64_t total_text_size = 0;
    const auto* text_sec = obj.get_section(".text");
    if (text_sec) {
        total_text_size = text_sec->data.size();
    }

    if (line_sec) {
        DwarfLineEmitter line_emitter(opts);
        line_emitter.emit(obj.debug_context, obj.debug_tables, obj.functions, *line_sec);
    }

    if (info_sec && abbrev_sec && str_sec) {
        DwarfInfoEmitter info_emitter(opts);
        info_emitter.emit(obj.debug_context, obj.debug_tables, obj.functions, *info_sec, *abbrev_sec, *str_sec, total_text_size);
    }
}

} // namespace brass::debug
