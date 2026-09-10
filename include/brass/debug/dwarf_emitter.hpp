#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/debug/debug_section.hpp>
#include <brass/object/object_writer.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace brass::debug {

namespace dwarf {
    // DWARF Standard Opcodes
    constexpr uint8_t DW_LNS_copy               = 0x01;
    constexpr uint8_t DW_LNS_advance_pc         = 0x02;
    constexpr uint8_t DW_LNS_advance_line       = 0x03;
    constexpr uint8_t DW_LNS_set_file           = 0x04;
    constexpr uint8_t DW_LNS_set_column         = 0x05;
    constexpr uint8_t DW_LNS_negate_stmt        = 0x06;
    constexpr uint8_t DW_LNS_set_basic_block    = 0x07;
    constexpr uint8_t DW_LNS_const_add_pc       = 0x08;
    constexpr uint8_t DW_LNS_fixed_advance_pc   = 0x09;
    constexpr uint8_t DW_LNS_set_prologue_end   = 0x0a;
    constexpr uint8_t DW_LNS_set_epilogue_begin = 0x0b;
    constexpr uint8_t DW_LNS_set_isa            = 0x0c;

    // Extended Opcodes
    constexpr uint8_t DW_LNE_end_sequence = 0x01;
    constexpr uint8_t DW_LNE_set_address  = 0x02;

    // Line program constants
    constexpr int8_t  DWARF_LINE_BASE   = -5;
    constexpr uint8_t DWARF_LINE_RANGE  = 14;
    constexpr uint8_t DWARF_OPCODE_BASE = 13;

    // Tags
    constexpr uint16_t DW_TAG_formal_parameter = 0x0005;
    constexpr uint16_t DW_TAG_compile_unit     = 0x0011;
    constexpr uint16_t DW_TAG_subprogram       = 0x002e;
    constexpr uint16_t DW_TAG_variable         = 0x0034;

    // Children flags
    constexpr uint8_t DW_CHILDREN_no  = 0x00;
    constexpr uint8_t DW_CHILDREN_yes = 0x01;

    // Attributes
    constexpr uint16_t DW_AT_location   = 0x0002;
    constexpr uint16_t DW_AT_name       = 0x0003;
    constexpr uint16_t DW_AT_stmt_list  = 0x0010;
    constexpr uint16_t DW_AT_low_pc     = 0x0011;
    constexpr uint16_t DW_AT_high_pc    = 0x0012;
    constexpr uint16_t DW_AT_language   = 0x0013;
    constexpr uint16_t DW_AT_comp_dir   = 0x001b;
    constexpr uint16_t DW_AT_producer   = 0x0025;
    constexpr uint16_t DW_AT_decl_file  = 0x003a;
    constexpr uint16_t DW_AT_decl_line  = 0x003b;
    constexpr uint16_t DW_AT_frame_base = 0x0040;

    // Forms
    constexpr uint16_t DW_FORM_addr       = 0x0001;
    constexpr uint16_t DW_FORM_data2      = 0x0005;
    constexpr uint16_t DW_FORM_data4      = 0x0006;
    constexpr uint16_t DW_FORM_data8      = 0x0007;
    constexpr uint16_t DW_FORM_string     = 0x0008;
    constexpr uint16_t DW_FORM_data1      = 0x000b;
    constexpr uint16_t DW_FORM_strp       = 0x000e;
    constexpr uint16_t DW_FORM_udata      = 0x000f;
    constexpr uint16_t DW_FORM_ref4       = 0x0013;
    constexpr uint16_t DW_FORM_sec_offset = 0x0017;
    constexpr uint16_t DW_FORM_exprloc    = 0x0018;

    // Register / Location Expression Opcodes
    constexpr uint8_t DW_OP_reg6   = 0x56; // %rbp in x86_64
    constexpr uint8_t DW_OP_fbreg  = 0x91;

    // Languages
    constexpr uint16_t DW_LANG_C99 = 0x000c;
}

// ULEB128 & SLEB128 encoding & decoding
void encode_uleb128(std::vector<uint8_t>& buf, uint64_t val);
void encode_sleb128(std::vector<uint8_t>& buf, int64_t val);
uint64_t decode_uleb128(const uint8_t*& ptr, const uint8_t* end);
int64_t decode_sleb128(const uint8_t*& ptr, const uint8_t* end);

struct DwarfOptions {
    uint16_t version = 4;
    std::string comp_dir = ".";
    std::string producer = "brass 1.0";
};

class DwarfLineEmitter {
public:
    explicit DwarfLineEmitter(DwarfOptions opts = {});

    void emit(
        const DebugContext& ctx,
        const std::vector<FunctionDebugTable>& tables,
        const std::vector<object::CompiledFunctionInfo>& functions,
        object::Section& line_sec
    );

    static std::vector<uint8_t> emit_standalone_line_table(
        const DebugContext& ctx,
        const std::vector<FunctionDebugTable>& tables,
        uint16_t version = 4
    );

private:
    DwarfOptions opts_;
};

class DwarfInfoEmitter {
public:
    explicit DwarfInfoEmitter(DwarfOptions opts = {});

    void emit(
        const DebugContext& ctx,
        const std::vector<FunctionDebugTable>& tables,
        const std::vector<object::CompiledFunctionInfo>& functions,
        object::Section& info_sec,
        object::Section& abbrev_sec,
        object::Section& str_sec,
        uint64_t total_text_size
    );

private:
    DwarfOptions opts_;
};

class DwarfEmitter {
public:
    static void emit(object::ObjectFile& obj, const DwarfOptions& opts = {});
};

} // namespace brass::debug
