#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/debug/debug_section.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace brass::debug {

namespace codeview {
    // CodeView Signature
    constexpr uint32_t CV_SIGNATURE_C13 = 4;

    // Subsection Kinds
    constexpr uint32_t DEBUG_S_SYMBOLS     = 0xF1;
    constexpr uint32_t DEBUG_S_LINES       = 0xF2;
    constexpr uint32_t DEBUG_S_STRINGTABLE = 0xF3;
    constexpr uint32_t DEBUG_S_FILECHKSMS  = 0xF4;
    constexpr uint32_t DEBUG_S_FRAMEDATA   = 0xF5;

    // Symbol Record Kinds
    constexpr uint16_t S_END         = 0x0006;
    constexpr uint16_t S_FRAMEPROC   = 0x1012;
    constexpr uint16_t S_LPROC32     = 0x110F;
    constexpr uint16_t S_GPROC32     = 0x1110;
    constexpr uint16_t S_REGREL32    = 0x1111;
    constexpr uint16_t S_LPROC32_ID  = 0x1146;
    constexpr uint16_t S_GPROC32_ID  = 0x1147;
    constexpr uint16_t S_PROC_ID_END = 0x114F;

    // Basic Builtin Type Indices
    constexpr uint32_t T_NOTYPE  = 0x0000;
    constexpr uint32_t T_VOID    = 0x0003;
    constexpr uint32_t T_UQUAD   = 0x0023; // 64-bit unsigned (array index type)
    constexpr uint32_t T_REAL32  = 0x0040; // 32-bit IEEE float
    constexpr uint32_t T_REAL64  = 0x0041; // 64-bit IEEE float
    constexpr uint32_t T_INT1    = 0x0068; // 8-bit signed int
    constexpr uint32_t T_INT2    = 0x0072; // 16-bit signed int
    constexpr uint32_t T_INT4    = 0x0074; // 32-bit signed int
    constexpr uint32_t T_INT8    = 0x0076; // 64-bit signed int
    constexpr uint32_t T_64PVOID = 0x0603; // 64-bit void pointer

    // First non-builtin type index
    constexpr uint32_t FIRST_TYPE_INDEX = 0x1000;

    // Type Leaf Kinds
    constexpr uint16_t LF_PROCEDURE = 0x1008;
    constexpr uint16_t LF_ARGLIST   = 0x1201;
    constexpr uint16_t LF_ARRAY     = 0x1503;
    constexpr uint8_t  LF_PAD0      = 0xF0;

    // Registers
    constexpr uint16_t CV_AMD64_RBP = 334;
    constexpr uint16_t CV_AMD64_RSP = 335;
    constexpr uint16_t CV_ARM64_FP  = 79;
    constexpr uint16_t CV_ARM64_SP  = 81;
}

struct CodeViewOptions {
    bool emit_symbols = true;
    bool emit_lines = true;
    bool is_aarch64 = false;
    // The file name used when the debug context has no source file
    // (see debug_primary_file_name). emit() takes it from the object.
    std::string module_name;
};

// The .debug$T type records for a set of functions: one LF_PROCEDURE (with
// its LF_ARGLIST, and LF_ARRAY for vector types) per distinct MIR signature.
struct CodeViewTypeTable {
    std::vector<uint8_t> records;      // type records, without the C13 signature
    std::vector<uint32_t> proc_types;  // LF_PROCEDURE type index per function
};

class CodeViewEmitter {
public:
    // Symbols and line tables refer to each function through relocations
    // against its own symbol (fn.name), which must be defined in the object.
    static void emit(object::ObjectFile& obj, const CodeViewOptions& opts = {});

    static CodeViewTypeTable build_type_table(const std::vector<object::CompiledFunctionInfo>& functions);

    static void emit_debug_s(
        const DebugContext& ctx,
        const std::vector<FunctionDebugTable>& tables,
        const std::vector<object::CompiledFunctionInfo>& functions,
        object::Section& debug_s_sec,
        const CodeViewOptions& opts = {}
    );

    static void emit_debug_t(const std::vector<object::CompiledFunctionInfo>& functions,
                             object::Section& debug_t_sec);
};

} // namespace brass::debug
