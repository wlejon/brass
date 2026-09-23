#include <brass/debug/codeview_emitter.hpp>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <cstring>

namespace brass::debug {

namespace {

void write_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

void write_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void patch_u16(std::vector<uint8_t>& buf, size_t offset, uint16_t v) {
    if (offset + 2 <= buf.size()) {
        buf[offset + 0] = static_cast<uint8_t>(v & 0xFF);
        buf[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    }
}

void patch_u32(std::vector<uint8_t>& buf, size_t offset, uint32_t v) {
    if (offset + 4 <= buf.size()) {
        buf[offset + 0] = static_cast<uint8_t>(v & 0xFF);
        buf[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        buf[offset + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        buf[offset + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    }
}

void write_str(std::vector<uint8_t>& buf, const std::string& str) {
    buf.insert(buf.end(), str.begin(), str.end());
    buf.push_back(0);
}

void pad_to_4(std::vector<uint8_t>& buf) {
    size_t rem = buf.size() % 4;
    if (rem != 0) {
        buf.resize(buf.size() + (4 - rem), 0);
    }
}

// Builds the .debug$T records, deduplicating identical records.
class TypeTableBuilder {
public:
    // Returns the type index of the record `body` (leaf kind + fields).
    uint32_t intern(const std::vector<uint8_t>& body) {
        auto it = index_of_.find(body);
        if (it != index_of_.end()) return it->second;
        // Record length (u16) excludes itself; records are padded to 4 bytes
        // with LF_PAD bytes (0xF0 | bytes remaining).
        size_t padded = body.size();
        while ((padded + 2) % 4 != 0) ++padded;
        if (padded > 0xFFFF) throw std::runtime_error("CodeView: type record too long");
        write_u16(records_, static_cast<uint16_t>(padded));
        records_.insert(records_.end(), body.begin(), body.end());
        for (size_t rem = padded - body.size(); rem > 0; --rem) {
            write_u8(records_, static_cast<uint8_t>(codeview::LF_PAD0 | rem));
        }
        uint32_t idx = next_index_++;
        index_of_.emplace(body, idx);
        return idx;
    }

    // The CodeView type index of a MIR value type.
    uint32_t value_type(Type t) {
        switch (t.kind()) {
            case TypeKind::I8:    return codeview::T_INT1;
            case TypeKind::I16:   return codeview::T_INT2;
            case TypeKind::I32:   return codeview::T_INT4;
            case TypeKind::I64:   return codeview::T_INT8;
            case TypeKind::F32:   return codeview::T_REAL32;
            case TypeKind::F64:   return codeview::T_REAL64;
            case TypeKind::Ptr:
            case TypeKind::GCRef: return codeview::T_64PVOID;
            case TypeKind::Void:  return codeview::T_VOID;
            default: break;
        }
        if (!t.is_vector()) {
            throw std::runtime_error("CodeView: no type mapping for MIR type kind " +
                                     std::to_string(static_cast<int>(t.kind())));
        }
        // A vector is an array of its lanes.
        std::vector<uint8_t> body;
        write_u16(body, codeview::LF_ARRAY);
        write_u32(body, value_type(t.element_type()));
        write_u32(body, codeview::T_UQUAD);
        write_u16(body, static_cast<uint16_t>(t.size_in_bytes())); // numeric leaf < 0x8000
        write_u8(body, 0);                                          // empty name
        return intern(body);
    }

    uint32_t procedure(Type ret, const std::vector<Type>& params) {
        if (params.size() > 0xFFFF) throw std::runtime_error("CodeView: too many parameters");
        std::vector<uint8_t> args;
        write_u16(args, codeview::LF_ARGLIST);
        write_u32(args, static_cast<uint32_t>(params.size()));
        for (Type p : params) {
            if (p.is_void()) throw std::runtime_error("CodeView: void parameter type");
            write_u32(args, value_type(p));
        }
        uint32_t arglist = intern(args);

        std::vector<uint8_t> proc;
        write_u16(proc, codeview::LF_PROCEDURE);
        write_u32(proc, value_type(ret));
        write_u8(proc, 0); // calling convention: near C
        write_u8(proc, 0); // function options
        write_u16(proc, static_cast<uint16_t>(params.size()));
        write_u32(proc, arglist);
        return intern(proc);
    }

    std::vector<uint8_t> take_records() { return std::move(records_); }

private:
    std::vector<uint8_t> records_;
    std::map<std::vector<uint8_t>, uint32_t> index_of_;
    uint32_t next_index_ = codeview::FIRST_TYPE_INDEX;
};

// A relocation of a 4-byte field (SECREL) and the 2-byte field after it
// (SECTION) against a function's own symbol; the stored values stay 0.
void add_code_ref_relocs(std::vector<object::ObjectRelocation>& relocs, size_t offset,
                         const std::string& fn_symbol) {
    object::ObjectRelocation r_off;
    r_off.offset = offset;
    r_off.kind = object::RelocKind::SecRel32;
    r_off.symbol_name = fn_symbol;
    r_off.addend = 0;
    relocs.push_back(r_off);

    object::ObjectRelocation r_seg;
    r_seg.offset = offset + 4;
    r_seg.kind = object::RelocKind::SecIdx;
    r_seg.symbol_name = fn_symbol;
    r_seg.addend = 0;
    relocs.push_back(r_seg);
}

} // namespace

CodeViewTypeTable CodeViewEmitter::build_type_table(
    const std::vector<object::CompiledFunctionInfo>& functions
) {
    TypeTableBuilder builder;
    CodeViewTypeTable table;
    table.proc_types.reserve(functions.size());
    for (const auto& fn : functions) {
        table.proc_types.push_back(builder.procedure(fn.return_type, fn.param_types));
    }
    table.records = builder.take_records();
    return table;
}

void CodeViewEmitter::emit_debug_s(
    const DebugContext& ctx,
    const std::vector<FunctionDebugTable>& tables,
    const std::vector<object::CompiledFunctionInfo>& functions,
    object::Section& debug_s_sec,
    const CodeViewOptions& opts
) {
    // 1. Signature
    debug_s_sec.emit32(codeview::CV_SIGNATURE_C13);

    // 2. String Table (DEBUG_S_STRINGTABLE = 0xF3)
    std::vector<uint8_t> str_payload;
    str_payload.push_back(0); // 0th offset is empty string

    std::unordered_map<std::string, uint32_t> str_offsets;
    auto get_or_add_str = [&](const std::string& s) -> uint32_t {
        if (s.empty()) return 0;
        auto it = str_offsets.find(s);
        if (it != str_offsets.end()) return it->second;
        uint32_t off = static_cast<uint32_t>(str_payload.size());
        write_str(str_payload, s);
        str_offsets[s] = off;
        return off;
    };

    // File ids are 1-based; with no registered source file, the one file is
    // the module itself.
    std::vector<std::string> file_names = ctx.files();
    if (file_names.empty()) {
        file_names.push_back(debug_primary_file_name(ctx, opts.module_name));
    }
    std::vector<uint32_t> file_str_offsets(file_names.size() + 1, 0);
    for (size_t i = 0; i < file_names.size(); ++i) {
        file_str_offsets[i + 1] = get_or_add_str(file_names[i]);
    }

    // Pre-intern function and variable names
    for (const auto& fn : functions) {
        get_or_add_str(fn.name);
    }
    for (const auto& tbl : tables) {
        for (const auto& var : tbl.variables()) {
            get_or_add_str(var.name);
        }
    }

    debug_s_sec.emit32(codeview::DEBUG_S_STRINGTABLE);
    debug_s_sec.emit32(static_cast<uint32_t>(str_payload.size()));
    debug_s_sec.emit_bytes(str_payload.data(), str_payload.size());
    debug_s_sec.align_to(4);

    // 3. File Checksums (DEBUG_S_FILECHKSMS = 0xF4)
    std::vector<uint8_t> chk_payload;
    std::unordered_map<uint32_t, uint32_t> file_chk_offsets;

    for (uint32_t i = 1; i <= file_names.size(); ++i) {
        file_chk_offsets[i] = static_cast<uint32_t>(chk_payload.size());
        write_u32(chk_payload, file_str_offsets[i]);
        write_u8(chk_payload, 0); // Checksum length = 0
        write_u8(chk_payload, 0); // Checksum type = 0 (None)
        pad_to_4(chk_payload);
    }

    debug_s_sec.emit32(codeview::DEBUG_S_FILECHKSMS);
    debug_s_sec.emit32(static_cast<uint32_t>(chk_payload.size()));
    debug_s_sec.emit_bytes(chk_payload.data(), chk_payload.size());
    debug_s_sec.align_to(4);

    // Map function name to FunctionDebugTable
    std::unordered_map<std::string, const FunctionDebugTable*> table_map;
    for (const auto& t : tables) {
        table_map[t.function_name()] = &t;
    }

    // 4. Lines (DEBUG_S_LINES = 0xF2)
    if (opts.emit_lines) {
        for (const auto& fn : functions) {
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

            uint32_t file_id = 1;
            if (!entries.empty() && entries.front().loc.file_id > 0) {
                file_id = entries.front().loc.file_id;
            }
            auto chk_it = file_chk_offsets.find(file_id);
            if (chk_it == file_chk_offsets.end()) {
                throw std::runtime_error("CodeView: function '" + fn.name + "' refers to unknown file id " +
                                         std::to_string(file_id));
            }
            uint32_t chk_off = chk_it->second;

            uint32_t num_lines = static_cast<uint32_t>(entries.size());
            uint32_t block_size = 12 + num_lines * 8;
            uint32_t lines_sub_len = 12 + block_size;

            debug_s_sec.emit32(codeview::DEBUG_S_LINES);
            debug_s_sec.emit32(lines_sub_len);

            size_t subsec_start = debug_s_sec.data.size();

            // Header code offset and segment, relocated against the function's
            // own symbol. Readers key each line table by that symbol, so
            // relocating every table against .text makes them collide.
            add_code_ref_relocs(debug_s_sec.relocations, subsec_start, fn.name);

            // Header (12 bytes)
            debug_s_sec.emit32(0); // Code offset (relocated)
            debug_s_sec.emit16(0); // Segment index (relocated)
            debug_s_sec.emit16(0); // Flags (no column info)
            debug_s_sec.emit32(static_cast<uint32_t>(fn.text_size));

            // File Block (12 bytes + num_lines * 8)
            debug_s_sec.emit32(chk_off);
            debug_s_sec.emit32(num_lines);
            debug_s_sec.emit32(block_size);

            for (const auto& entry : entries) {
                debug_s_sec.emit32(entry.code_offset);
                uint32_t line_num = entry.loc.line > 0 ? entry.loc.line : 1;
                uint32_t line_flags = (line_num & 0x00FFFFFF) | 0x80000000; // Statement bit = 1
                debug_s_sec.emit32(line_flags);
            }

            debug_s_sec.align_to(4);
        }
    }

    // 5. Symbols (DEBUG_S_SYMBOLS = 0xF1)
    if (opts.emit_symbols && !functions.empty()) {
        std::vector<uint8_t> sym_payload;
        std::vector<object::ObjectRelocation> sym_relocs;
        const std::vector<uint32_t> proc_types = build_type_table(functions).proc_types;

        for (size_t fi = 0; fi < functions.size(); ++fi) {
            const auto& fn = functions[fi];
            auto it = table_map.find(fn.name);
            const FunctionDebugTable* tbl = (it != table_map.end()) ? it->second : nullptr;

            // S_GPROC32 (0x1110)
            size_t proc_rec_start = sym_payload.size();
            write_u16(sym_payload, 0); // Length placeholder
            write_u16(sym_payload, codeview::S_GPROC32);
            write_u32(sym_payload, 0); // pParent
            size_t pend_offset = sym_payload.size();
            write_u32(sym_payload, 0); // pEnd placeholder
            write_u32(sym_payload, 0); // pNext
            write_u32(sym_payload, static_cast<uint32_t>(fn.text_size));
            write_u32(sym_payload, static_cast<uint32_t>(fn.prologue_size));
            write_u32(sym_payload, static_cast<uint32_t>(fn.text_size));
            write_u32(sym_payload, proc_types[fi]); // LF_PROCEDURE type index

            // codeOffset and codeSegment, relocated against the function symbol
            add_code_ref_relocs(sym_relocs, sym_payload.size(), fn.name);
            write_u32(sym_payload, 0);
            write_u16(sym_payload, 0);
            write_u8(sym_payload, 0); // Flags
            write_str(sym_payload, fn.name);
            pad_to_4(sym_payload);
            patch_u16(sym_payload, proc_rec_start, static_cast<uint16_t>((sym_payload.size() - proc_rec_start) - 2));

            // Variables (S_REGREL32 = 0x1111)
            if (tbl) {
                for (const auto& var : tbl->variables()) {
                    size_t var_rec_start = sym_payload.size();
                    write_u16(sym_payload, 0); // Length placeholder
                    write_u16(sym_payload, codeview::S_REGREL32);
                    write_u32(sym_payload, static_cast<uint32_t>(var.stack_offset));
                    uint32_t tid = var.type_index > 0 ? var.type_index : codeview::T_INT8;
                    write_u32(sym_payload, tid);
                    write_u16(sym_payload, opts.is_aarch64 ? codeview::CV_ARM64_FP : codeview::CV_AMD64_RBP);
                    write_str(sym_payload, var.name);
                    pad_to_4(sym_payload);
                    patch_u16(sym_payload, var_rec_start, static_cast<uint16_t>((sym_payload.size() - var_rec_start) - 2));
                }
            }

            // S_END closes the S_GPROC32 scope (S_PROC_ID_END pairs with S_GPROC32_ID)
            uint32_t proc_end_offset = static_cast<uint32_t>(sym_payload.size());
            write_u16(sym_payload, 2);
            write_u16(sym_payload, codeview::S_END);
            patch_u32(sym_payload, pend_offset, proc_end_offset);
        }

        debug_s_sec.emit32(codeview::DEBUG_S_SYMBOLS);
        debug_s_sec.emit32(static_cast<uint32_t>(sym_payload.size()));

        size_t sec_offset_base = debug_s_sec.data.size();
        for (auto& r : sym_relocs) {
            r.offset += sec_offset_base;
            debug_s_sec.relocations.push_back(r);
        }

        debug_s_sec.emit_bytes(sym_payload.data(), sym_payload.size());
        debug_s_sec.align_to(4);
    }
}

void CodeViewEmitter::emit_debug_t(
    const std::vector<object::CompiledFunctionInfo>& functions,
    object::Section& debug_t_sec
) {
    debug_t_sec.emit32(codeview::CV_SIGNATURE_C13);
    CodeViewTypeTable table = build_type_table(functions);
    debug_t_sec.emit_bytes(table.records.data(), table.records.size());
}

void CodeViewEmitter::emit(object::ObjectFile& obj, const CodeViewOptions& opts) {
    obj.get_or_create_section(
        ".debug$S",
        object::SectionKind::Custom,
        object::SectionFlags::Read | object::SectionFlags::Discardable,
        4
    );

    obj.get_or_create_section(
        ".debug$T",
        object::SectionKind::Custom,
        object::SectionFlags::Read | object::SectionFlags::Discardable,
        4
    );

    auto* s_sec = obj.get_section(".debug$S");
    auto* t_sec = obj.get_section(".debug$T");

    if (!s_sec || !t_sec) {
        throw std::runtime_error("CodeView: could not create .debug$S/.debug$T");
    }
    for (const auto& fn : obj.functions) {
        const object::ObjectSymbol* sym = obj.find_symbol(fn.name);
        if (!sym || sym->section_index == object::SECTION_UNDEF) {
            throw std::runtime_error("CodeView: function '" + fn.name + "' has no defined symbol to relocate against");
        }
    }
    CodeViewOptions effective_opts = opts;
    if (obj.target.is_aarch64()) {
        effective_opts.is_aarch64 = true;
    }
    if (effective_opts.module_name.empty()) {
        effective_opts.module_name = obj.module_name;
    }
    emit_debug_s(obj.debug_context, obj.debug_tables, obj.functions, *s_sec, effective_opts);
    emit_debug_t(obj.functions, *t_sec);
}

} // namespace brass::debug
