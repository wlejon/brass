#pragma once

#include <brass/object/reloc_kind.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/code_buffer.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/mir/module.hpp>
#include <brass/gc/stack_map.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>

namespace brass::object {

enum class SectionKind : uint8_t {
    Text,
    Data,
    RoData,
    PData,
    XData,
    EhFrame,
    Bss,
    Custom
};

enum class SectionFlags : uint32_t {
    None        = 0,
    Read        = 1 << 0,
    Write       = 1 << 1,
    Execute     = 1 << 2,
    Alloc       = 1 << 3,
    Discardable = 1 << 4,
};

inline SectionFlags operator|(SectionFlags a, SectionFlags b) {
    return static_cast<SectionFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline SectionFlags operator&(SectionFlags a, SectionFlags b) {
    return static_cast<SectionFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool has_flag(SectionFlags mask, SectionFlags flag) {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(flag)) != 0;
}

struct ObjectRelocation {
    size_t offset = 0;              // Offset in section data where relocation applies
    RelocKind kind = RelocKind::PCRel32;
    std::string symbol_name;        // Target symbol name
    int64_t addend = 0;             // Addend
    uint32_t symbol_index = 0;      // Index in symbol table (populated during emission)
};

enum class SymbolBinding : uint8_t {
    Local,
    Global,
    Weak
};

enum class SymbolType : uint8_t {
    Notype,
    Function,
    Object,
    Section
};

constexpr int32_t SECTION_UNDEF = -1;

struct ObjectSymbol {
    std::string name;
    int32_t section_index = SECTION_UNDEF; // 0-based section index, or SECTION_UNDEF (-1) for external
    uint64_t value = 0;                    // Section offset or symbol value
    uint64_t size = 0;                     // Symbol size
    SymbolBinding binding = SymbolBinding::Global;
    SymbolType type = SymbolType::Function;
};

struct Section {
    std::string name;
    SectionKind kind = SectionKind::Custom;
    SectionFlags flags = SectionFlags::Read;
    uint32_t alignment = 16;
    std::vector<uint8_t> data;
    std::vector<ObjectRelocation> relocations;

    void emit8(uint8_t v) { data.push_back(v); }
    void emit16(uint16_t v) {
        data.push_back(static_cast<uint8_t>(v & 0xFF));
        data.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }
    void emit32(uint32_t v) {
        data.push_back(static_cast<uint8_t>(v & 0xFF));
        data.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        data.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        data.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    }
    void emit64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            data.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
        }
    }
    void emit_bytes(const uint8_t* ptr, size_t count) {
        data.insert(data.end(), ptr, ptr + count);
    }
    void emit_bytes(std::span<const uint8_t> bytes) {
        data.insert(data.end(), bytes.begin(), bytes.end());
    }
    void align_to(size_t align) {
        if (align <= 1) return;
        size_t rem = data.size() % align;
        if (rem != 0) {
            data.resize(data.size() + (align - rem), 0);
        }
    }
};

#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/exception.hpp>

struct CompiledFunctionInfo {
    std::string name;
    size_t text_offset = 0;
    size_t text_size = 0;
    size_t prologue_size = 0;
    size_t osr_entry_offset = 0;
    codegen::FrameInfo frame_info;
    CallingConvention cc;
    std::vector<codegen::SafepointRecord> safepoints;
    FunctionStackMap stack_map;
    runtime::FunctionResumeTable resume_table;
    std::vector<runtime::PatchSite> patch_sites;
    runtime::FunctionExceptionTable exception_table;
};

struct ObjectFile {
    Target target = Target::host();
    std::vector<Section> sections;
    std::vector<ObjectSymbol> symbols;
    std::vector<CompiledFunctionInfo> functions;
    ModuleStackMap stack_maps;
    runtime::ResumeTableRegistry resume_tables;
    runtime::PatchRegistry patch_sites;
    std::vector<FunctionDebugTable> debug_tables;
    runtime::ExceptionTableRegistry exception_tables;
    DebugContext debug_context;

    Section* get_section(std::string_view name);
    const Section* get_section(std::string_view name) const;
    int32_t get_section_index(std::string_view name) const;

    Section& get_or_create_section(std::string_view name, SectionKind kind, SectionFlags flags, uint32_t alignment);
    Section& get_or_create_section(std::string_view name, SectionKind kind, SectionFlags flags);

    // A symbol is unique by name: add_symbol replaces an existing entry of
    // the same name in place (its index is what relocations hold) and
    // appends otherwise. Both it and find_symbol go through the name index
    // below rather than a scan of `symbols` — at ~19k symbols the scans
    // were a quarter of a second per module.
    uint32_t add_symbol(ObjectSymbol sym);
    const ObjectSymbol* find_symbol(std::string_view name) const;
    ObjectSymbol* find_symbol(std::string_view name);

private:
    struct NameHash {
        using is_transparent = void;
        size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    };
    // name -> index into `symbols`. `symbols` is public and a caller may
    // append to it directly, so the index is checked against the vector's
    // size on every use and rebuilt when it has fallen behind; a rename in
    // place is the one edit it cannot see, and nothing performs one.
    mutable std::unordered_map<std::string, uint32_t, NameHash, std::equal_to<>> symbol_index_;
    void sync_symbol_index() const;
};

class ModuleCompiler {
public:
    explicit ModuleCompiler(const Target& target);
    ModuleCompiler(const Target& target, const CallingConvention& cc);

    void set_sched_options(const codegen::SchedOptions& opts) { sched_opts_ = opts; }
    const codegen::SchedOptions& sched_options() const noexcept { return sched_opts_; }

    void set_enable_trace_layout(bool enable) { enable_trace_layout_ = enable; }
    bool enable_trace_layout() const noexcept { return enable_trace_layout_; }

    // The per-function MIR optimizer (loop opts, GVN, SCCP, DCE, unrolling,
    // vectorization) that runs on every function before instruction
    // selection. Off is the baseline tier: straight ISel of the MIR as given,
    // for a host that wants the code now rather than good.
    void set_enable_mir_opts(bool enable) { enable_mir_opts_ = enable; }
    bool enable_mir_opts() const noexcept { return enable_mir_opts_; }

    ObjectFile compile(const Module& mod);

private:
    Target target_;
    CallingConvention cc_;
    codegen::SchedOptions sched_opts_;
    bool enable_trace_layout_ = false;
    bool enable_mir_opts_ = true;
};

ObjectFile compile_module_to_object(const Module& mod, const Target& target, const codegen::SchedOptions& sched_opts);
ObjectFile compile_module_to_object(const Module& mod, const Target& target);

// Turns every GotPCRel32 load of a symbol the object itself defines (or of
// one of its sections) into `lea r64, [rip + symbol]` — the opcode byte
// before the displacement goes from 8B to 8D, the relocation becomes a
// PCRel32 with the same addend — and leaves the loads of undefined symbols
// for the placer's GOT. On AArch64 the GotPage21 / GotLo12 pair of a defined
// symbol becomes AdrPage21 / AddLo12, the `ldr x, [x, #lo12]` rewritten to
// `add x, x, #lo12` (a GotLo12 on any other instruction throws: the object
// is malformed). Run by the image writers on their
// working copy, once the object is complete: a host may define data symbols
// after code generation, and it is only then that "defined" is known. The
// return value is the number of loads left, every one against an undefined
// symbol. With `local_only`, only loads of local (non-preemptible) symbols
// are relaxed: a relocatable ELF object keeps GOT loads of its global
// symbols, which a shared-library link may preempt, and lets the system
// linker relax them (REX_GOTPCRELX) when it can.
size_t relax_got_loads(ObjectFile& obj, bool local_only = false);

// For an object that is going to a system linker without a GOT of its own
// (COFF): after relax_got_loads, every remaining load takes an 8-byte slot
// in `slot_section`, an Abs64 relocation binds the slot to its symbol, and
// the load is rewritten as a PCRel32 against a local symbol at the slot (on
// AArch64: AdrPage21 + LdSt64Lo12 against it).
// Distinct symbols get distinct slots; repeated loads share one.
void materialize_got_slots(ObjectFile& obj, std::string_view slot_section, SectionKind kind,
                           SectionFlags flags);

} // namespace brass::object
