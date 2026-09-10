#include <brass/debug/codeview_emitter.hpp>
#include <algorithm>
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

} // namespace

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

    std::vector<uint32_t> file_str_offsets;
    if (ctx.file_count() > 0) {
        file_str_offsets.resize(ctx.file_count() + 1, 0);
        for (uint32_t i = 1; i <= ctx.file_count(); ++i) {
            file_str_offsets[i] = get_or_add_str(ctx.get_file(i));
        }
    } else {
        file_str_offsets.push_back(0);
        file_str_offsets.push_back(get_or_add_str("source.js"));
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

    size_t count_files = (ctx.file_count() > 0) ? ctx.file_count() : 1;
    for (uint32_t i = 1; i <= count_files; ++i) {
        file_chk_offsets[i] = static_cast<uint32_t>(chk_payload.size());
        uint32_t s_off = (i < file_str_offsets.size()) ? file_str_offsets[i] : file_str_offsets.back();
        write_u32(chk_payload, s_off);
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
            uint32_t chk_off = 0;
            auto chk_it = file_chk_offsets.find(file_id);
            if (chk_it != file_chk_offsets.end()) {
                chk_off = chk_it->second;
            }

            uint32_t num_lines = static_cast<uint32_t>(entries.size());
            uint32_t block_size = 12 + num_lines * 8;
            uint32_t lines_sub_len = 12 + block_size;

            debug_s_sec.emit32(codeview::DEBUG_S_LINES);
            debug_s_sec.emit32(lines_sub_len);

            size_t subsec_start = debug_s_sec.data.size();

            // Relocations for Header: code_offset (SecRel32) and code_segment (SecIdx)
            object::ObjectRelocation r_off;
            r_off.offset = subsec_start + 0;
            r_off.kind = object::RelocKind::SecRel32;
            r_off.symbol_name = ".text";
            r_off.addend = static_cast<int64_t>(fn.text_offset);
            debug_s_sec.relocations.push_back(r_off);

            object::ObjectRelocation r_seg;
            r_seg.offset = subsec_start + 4;
            r_seg.kind = object::RelocKind::SecIdx;
            r_seg.symbol_name = ".text";
            r_seg.addend = 0;
            debug_s_sec.relocations.push_back(r_seg);

            // Header (12 bytes)
            debug_s_sec.emit32(static_cast<uint32_t>(fn.text_offset));
            debug_s_sec.emit16(0); // Segment index placeholder
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

        for (const auto& fn : functions) {
            auto it = table_map.find(fn.name);
            const FunctionDebugTable* tbl = (it != table_map.end()) ? it->second : nullptr;

            // S_GPROC32 (0x1110)
            size_t proc_rec_start = sym_payload.size();
            write_u16(sym_payload, 0); // Length placeholder
            write_u16(sym_payload, codeview::S_GPROC32);
            write_u32(sym_payload, 0); // pParent
            write_u32(sym_payload, 0); // pEnd
            write_u32(sym_payload, 0); // pNext
            write_u32(sym_payload, static_cast<uint32_t>(fn.text_size));
            write_u32(sym_payload, static_cast<uint32_t>(fn.prologue_size));
            write_u32(sym_payload, static_cast<uint32_t>(fn.text_size));
            write_u32(sym_payload, 0x1001); // LF_PROCEDURE type index

            // Relocations for proc codeOffset and codeSegment
            object::ObjectRelocation r_proc_off;
            r_proc_off.offset = sym_payload.size();
            r_proc_off.kind = object::RelocKind::SecRel32;
            r_proc_off.symbol_name = ".text";
            r_proc_off.addend = static_cast<int64_t>(fn.text_offset);
            sym_relocs.push_back(r_proc_off);

            write_u32(sym_payload, static_cast<uint32_t>(fn.text_offset));

            object::ObjectRelocation r_proc_seg;
            r_proc_seg.offset = sym_payload.size();
            r_proc_seg.kind = object::RelocKind::SecIdx;
            r_proc_seg.symbol_name = ".text";
            r_proc_seg.addend = 0;
            sym_relocs.push_back(r_proc_seg);

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
                    write_u16(sym_payload, codeview::CV_AMD64_RBP);
                    write_str(sym_payload, var.name);
                    pad_to_4(sym_payload);
                    patch_u16(sym_payload, var_rec_start, static_cast<uint16_t>((sym_payload.size() - var_rec_start) - 2));
                }
            }

            // End marker S_PROC_ID_END (0x114F)
            write_u16(sym_payload, 2);
            write_u16(sym_payload, codeview::S_PROC_ID_END);
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

void CodeViewEmitter::emit_debug_t(object::Section& debug_t_sec) {
    debug_t_sec.emit32(codeview::CV_SIGNATURE_C13);

    // Type Record 1: LF_ARGLIST (0x1201) at index 0x1000
    debug_t_sec.emit16(6); // Record length (6 bytes: leaf 2, count 4)
    debug_t_sec.emit16(codeview::LF_ARGLIST);
    debug_t_sec.emit32(0); // Argument count = 0

    // Type Record 2: LF_PROCEDURE (0x1008) at index 0x1001 (returns T_INT8)
    debug_t_sec.emit16(14);
    debug_t_sec.emit16(codeview::LF_PROCEDURE);
    debug_t_sec.emit32(codeview::T_INT8); // Return type
    debug_t_sec.emit8(0);                 // Calling convention (near C)
    debug_t_sec.emit8(0);                 // Attributes
    debug_t_sec.emit16(0);                // Parameter count
    debug_t_sec.emit32(0x1000);           // ArgList type index

    // Type Record 3: LF_PROCEDURE (0x1008) at index 0x1002 (returns T_REAL64)
    debug_t_sec.emit16(14);
    debug_t_sec.emit16(codeview::LF_PROCEDURE);
    debug_t_sec.emit32(codeview::T_REAL64);
    debug_t_sec.emit8(0);
    debug_t_sec.emit8(0);
    debug_t_sec.emit16(0);
    debug_t_sec.emit32(0x1000);

    // Type Record 4: LF_PROCEDURE (0x1008) at index 0x1003 (returns T_INT4)
    debug_t_sec.emit16(14);
    debug_t_sec.emit16(codeview::LF_PROCEDURE);
    debug_t_sec.emit32(codeview::T_INT4);
    debug_t_sec.emit8(0);
    debug_t_sec.emit8(0);
    debug_t_sec.emit16(0);
    debug_t_sec.emit32(0x1000);

    // Type Record 5: LF_PROCEDURE (0x1008) at index 0x1004 (returns T_64PVOID)
    debug_t_sec.emit16(14);
    debug_t_sec.emit16(codeview::LF_PROCEDURE);
    debug_t_sec.emit32(codeview::T_64PVOID);
    debug_t_sec.emit8(0);
    debug_t_sec.emit8(0);
    debug_t_sec.emit16(0);
    debug_t_sec.emit32(0x1000);

    debug_t_sec.align_to(4);
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

    if (s_sec && t_sec) {
        emit_debug_s(obj.debug_context, obj.debug_tables, obj.functions, *s_sec, opts);
        emit_debug_t(*t_sec);
    }
}

} // namespace brass::debug
