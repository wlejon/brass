#include <brass/object/coff_writer.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace brass::object {

using namespace brass::x64;

struct UnwindOpSlot {
    uint8_t code_offset = 0;
    uint8_t unwind_op = 0;
    uint8_t op_info = 0;
    uint16_t extra_slot1 = 0;
    uint16_t extra_slot2 = 0;
    int num_slots = 1; // 1, 2, or 3
};

// ARM64 unwind codes for a brass AArch64 frame, in the order the unwinder
// applies them (the reverse of the prologue). AArch64FrameLayout lays a
// frame out as
//
//   CFA - (total - outgoing)  FP, LR            <- x29
//                             callee GPRs, callee FPRs, spills, locals
//   SP                        outgoing arguments
//
// and saves the callee registers relative to x29 after setting it, which
// no canonical (packed) prologue does. What is described is the equivalent
//
//   sub sp, sp, #(total - outgoing)     alloc
//   stp x29, lr, [sp]                   save_fplr 0
//   stp/str callee regs, [sp, #16 + ..] save_regp / save_reg / save_fregp / save_freg
//   mov x29, sp                         set_fp
//   sub sp, sp, #outgoing               alloc
//
// whose effect at every instruction of the body is the same: set_fp makes the
// unwinder take SP from x29, so the register offsets stay small whatever the
// size of the outgoing area. It is not the emitted prologue instruction for
// instruction, so an unwind from inside the prologue or an epilogue (an
// asynchronous fault there, not a call) is not described exactly.
static std::vector<uint8_t> aarch64_unwind_codes(const codegen::FrameInfo& frame) {
    std::vector<uint8_t> codes;
    auto alloc = [&](uint64_t bytes) {
        if (bytes == 0) return;
        const uint64_t units = bytes / 16;
        if (units < 32) {
            codes.push_back(static_cast<uint8_t>(units));                        // alloc_s
        } else if (units < 2048) {
            codes.push_back(static_cast<uint8_t>(0xC0 | (units >> 8)));          // alloc_m
            codes.push_back(static_cast<uint8_t>(units & 0xFF));
        } else if (units < (uint64_t{1} << 24)) {
            codes.push_back(0xE0);                                               // alloc_l
            codes.push_back(static_cast<uint8_t>(units >> 16));
            codes.push_back(static_cast<uint8_t>(units >> 8));
            codes.push_back(static_cast<uint8_t>(units & 0xFF));
        } else {
            throw std::runtime_error("ARM64 unwind: a frame of " + std::to_string(bytes) +
                                     " bytes exceeds what alloc_l can describe");
        }
    };
    if (frame.is_leaf) {
        codes.push_back(0xE4);   // end
        return codes;
    }
    const uint64_t total = frame.total_frame_size;
    const uint64_t outgoing = (frame.outgoing_arg_space + 15) & ~uint64_t{15};
    if (total % 16 != 0 || outgoing > total || total - outgoing < 16) {
        throw std::runtime_error("ARM64 unwind: frame layout is not 16-byte aligned around FP/LR");
    }
    auto saved_gprs = aarch64::AArch64FrameLayout::get_saved_callee_gprs(frame);
    auto saved_fprs = aarch64::AArch64FrameLayout::get_saved_callee_fprs(frame);

    alloc(outgoing);
    codes.push_back(0xE1);   // set_fp

    // Register saves, last first. Offsets from x29, in 8-byte units.
    auto reg_codes = [&](auto& regs, size_t first_offset, int base, bool fp) {
        std::vector<std::vector<uint8_t>> groups;
        size_t i = 0;
        while (i < regs.size()) {
            const int r = static_cast<int>(regs[i]);
            const uint32_t z = static_cast<uint32_t>((first_offset + i * 8) / 8);
            if (z > 63) throw std::runtime_error("ARM64 unwind: callee save offset out of range");
            const bool pair = i + 1 < regs.size() && static_cast<int>(regs[i + 1]) == r + 1;
            const uint32_t x = static_cast<uint32_t>(r - base);
            uint16_t code = 0;
            if (fp) {
                code = static_cast<uint16_t>((pair ? 0xD800u : 0xDC00u) | (x << 6) | z);   // save_fregp / save_freg
            } else {
                code = static_cast<uint16_t>((pair ? 0xC800u : 0xD000u) | (x << 6) | z);   // save_regp / save_reg
            }
            groups.push_back({static_cast<uint8_t>(code >> 8), static_cast<uint8_t>(code & 0xFF)});
            i += pair ? 2 : 1;
        }
        for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
            codes.insert(codes.end(), it->begin(), it->end());
        }
    };
    reg_codes(saved_fprs, 16 + saved_gprs.size() * 8, 8, true);
    reg_codes(saved_gprs, 16, 19, false);

    codes.push_back(0x40);   // save_fplr, [sp + 0]
    alloc(total - outgoing);
    codes.push_back(0xE4);   // end
    return codes;
}

static void build_aarch64_unwind_info(ObjectFile& obj) {
    Section* pdata_sec = obj.get_section(".pdata");
    Section* xdata_sec = obj.get_section(".xdata");
    if (!pdata_sec || !xdata_sec) return;

    for (const auto& fn : obj.functions) {
        bool has_ehandler = fn.exception_table.has_scopes();
        size_t fn_len_words = fn.text_size / 4;
        if (fn_len_words > 0x3FFFF) {
            throw std::runtime_error("ARM64 unwind: function '" + fn.name + "' is longer than one .xdata record covers");
        }

        // .pdata entries are 8 bytes: BeginAddress, then either packed
        // unwind data (low bits 01) or the RVA of an .xdata record (00).
        //
        // A leaf has no prologue at all, which the packed form says with
        // every field but FunctionLength zero: no saved registers, CR 00
        // (LR not saved), FrameSize 0. A framed function gets a full record:
        // brass saves its callee registers above FP/LR, which no packed
        // (canonical) prologue does.
        if (fn.frame_info.is_leaf && !has_ehandler && fn_len_words <= 0x7FF) {
            pdata_sec->align_to(4);
            size_t pdata_offset = pdata_sec->data.size();
            pdata_sec->emit32(0);
            ObjectRelocation r0;
            r0.offset = pdata_offset;
            r0.kind = RelocKind::Addr32NB;
            r0.symbol_name = fn.name;
            r0.addend = 0;
            pdata_sec->relocations.push_back(std::move(r0));
            pdata_sec->emit32(0x1u | (static_cast<uint32_t>(fn_len_words) << 2));
        } else {
            xdata_sec->align_to(4);
            size_t xdata_offset = xdata_sec->data.size();

            std::vector<uint8_t> unwind_codes = aarch64_unwind_codes(fn.frame_info);
            while (unwind_codes.size() % 4 != 0) {
                unwind_codes.push_back(0xE3);   // nop padding after end
            }

            uint32_t code_words = static_cast<uint32_t>(unwind_codes.size() / 4);
            if (code_words > 31) {
                throw std::runtime_error("ARM64 unwind: too many unwind codes for '" + fn.name + "'");
            }
            // FunctionLength, Vers 0, X, E 0 with no epilog scopes: the
            // epilogues are not described, the body is.
            uint32_t header = static_cast<uint32_t>(fn_len_words & 0x3FFFF);
            if (has_ehandler) {
                header |= (1u << 20); // X = 1
            }
            header |= ((code_words & 0x1Fu) << 27);

            xdata_sec->emit32(header);
            for (uint8_t b : unwind_codes) {
                xdata_sec->emit8(b);
            }

            if (has_ehandler) {
                if (!obj.find_symbol("brass_seh_personality")) {
                    ObjectSymbol s;
                    s.name = "brass_seh_personality";
                    s.section_index = SECTION_UNDEF;
                    s.binding = SymbolBinding::Global;
                    s.type = SymbolType::Function;
                    obj.add_symbol(std::move(s));
                }
                size_t handler_off = xdata_sec->data.size();
                xdata_sec->emit32(0);
                ObjectRelocation r;
                r.offset = handler_off;
                r.kind = RelocKind::Addr32NB;
                r.symbol_name = "brass_seh_personality";
                r.addend = 0;
                xdata_sec->relocations.push_back(std::move(r));

                runtime::emit_win64_seh_scope_table(*xdata_sec, fn.exception_table, fn.name);
            }

            // Emit RUNTIME_FUNCTION in .pdata (8 bytes for ARM64)
            pdata_sec->align_to(4);
            size_t pdata_offset = pdata_sec->data.size();

            // Word 0: BeginAddress
            pdata_sec->emit32(0);
            ObjectRelocation r0;
            r0.offset = pdata_offset + 0;
            r0.kind = RelocKind::Addr32NB;
            r0.symbol_name = fn.name;
            r0.addend = 0;
            pdata_sec->relocations.push_back(std::move(r0));

            // Word 1: UnwindInfoAddress (Addr32NB to .xdata, low 2 bits = 00b)
            pdata_sec->emit32(static_cast<uint32_t>(xdata_offset));
            ObjectRelocation r1;
            r1.offset = pdata_offset + 4;
            r1.kind = RelocKind::Addr32NB;
            r1.symbol_name = ".xdata";
            r1.addend = static_cast<int64_t>(xdata_offset);
            pdata_sec->relocations.push_back(std::move(r1));
        }
    }
}

void CoffUnwindBuilder::build_unwind_info(
    ObjectFile& obj,
    Section& pdata_sec_param,
    Section& xdata_sec_param
) {
    (void)pdata_sec_param;
    (void)xdata_sec_param;

    // Ensure section symbols exist for relocations
    if (!obj.find_symbol(".text")) {
        ObjectSymbol s;
        s.name = ".text";
        s.section_index = obj.get_section_index(".text");
        s.value = 0;
        s.size = 0;
        s.binding = SymbolBinding::Local;
        s.type = SymbolType::Section;
        obj.add_symbol(std::move(s));
    }
    if (!obj.find_symbol(".xdata")) {
        ObjectSymbol s;
        s.name = ".xdata";
        s.section_index = obj.get_section_index(".xdata");
        s.value = 0;
        s.size = 0;
        s.binding = SymbolBinding::Local;
        s.type = SymbolType::Section;
        obj.add_symbol(std::move(s));
    }

    Section* pdata_sec = obj.get_section(".pdata");
    Section* xdata_sec = obj.get_section(".xdata");
    if (!pdata_sec || !xdata_sec) return;

    if (obj.target.is_aarch64()) {
        build_aarch64_unwind_info(obj);
        return;
    }

    for (const auto& fn : obj.functions) {
        xdata_sec->align_to(4);
        size_t xdata_offset = xdata_sec->data.size();

        std::vector<UnwindOpSlot> ops;
        uint8_t cur_offset = 0;
        uint8_t frame_reg_offset = 0;

        if (!fn.frame_info.is_leaf) {
            frame_reg_offset = 0; // FrameReg = 0 (RSP-based tracking with UWOP_ALLOC & UWOP_PUSH_NONVOL)

            // 1. push rbp (1 byte: 0x55)
            cur_offset += 1;
            {
                UnwindOpSlot op;
                op.code_offset = cur_offset;
                op.unwind_op = coff::UWOP_PUSH_NONVOL;
                op.op_info = 5; // RBP
                op.num_slots = 1;
                ops.push_back(op);
            }

            // 2. mov rbp, rsp (3 bytes: 0x48 0x89 0xE5)
            cur_offset += 3;

            // 3. sub rsp, total_frame_size
            size_t frame_sz = fn.frame_info.total_frame_size;
            if (frame_sz > 0) {
                if (frame_sz <= 127) {
                    cur_offset += 4; // 48 83 EC imm8
                } else {
                    cur_offset += 7; // 48 81 EC imm32
                }

                UnwindOpSlot op;
                op.code_offset = cur_offset;
                if (frame_sz <= 128) {
                    op.unwind_op = coff::UWOP_ALLOC_SMALL;
                    op.op_info = static_cast<uint8_t>((frame_sz - 8) / 8);
                    op.num_slots = 1;
                } else if (frame_sz <= 512 * 1024 - 8) {
                    op.unwind_op = coff::UWOP_ALLOC_LARGE;
                    op.op_info = 0;
                    op.extra_slot1 = static_cast<uint16_t>(frame_sz / 8);
                    op.num_slots = 2;
                } else {
                    op.unwind_op = coff::UWOP_ALLOC_LARGE;
                    op.op_info = 1;
                    op.extra_slot1 = static_cast<uint16_t>(frame_sz & 0xFFFF);
                    op.extra_slot2 = static_cast<uint16_t>((frame_sz >> 16) & 0xFFFF);
                    op.num_slots = 3;
                }
                ops.push_back(op);
            }

            // 4. Saved callee-saved GPRs
            auto saved_gprs = X64FrameLayout::get_saved_callee_gprs(fn.frame_info);
            for (size_t i = 0; i < saved_gprs.size(); ++i) {
                GPR g = saved_gprs[i];
                int32_t disp_from_rbp = static_cast<int32_t>((i + 1) * 8);
                int32_t disp_from_rsp = static_cast<int32_t>(frame_sz - disp_from_rbp);
                disp_from_rbp <= 127 ? (cur_offset = static_cast<uint8_t>(cur_offset + 4))
                                     : (cur_offset = static_cast<uint8_t>(cur_offset + 7));

                UnwindOpSlot op;
                op.code_offset = cur_offset;
                op.unwind_op = coff::UWOP_SAVE_NONVOL;
                op.op_info = static_cast<uint8_t>(g);
                op.extra_slot1 = static_cast<uint16_t>(disp_from_rsp / 8);
                op.num_slots = 2;
                ops.push_back(op);
            }

            // 5. Saved callee-saved XMMs
            auto saved_xmms = X64FrameLayout::get_saved_callee_xmms(fn.frame_info);
            size_t gpr_bytes = (saved_gprs.size() * 8 + 15) & ~size_t(15);
            for (size_t i = 0; i < saved_xmms.size(); ++i) {
                XMM x = saved_xmms[i];
                int32_t disp_from_rbp = static_cast<int32_t>(gpr_bytes + (i + 1) * 16);
                int32_t disp_from_rsp = static_cast<int32_t>(frame_sz - disp_from_rbp);
                bool is_ext = (static_cast<uint8_t>(x) >= 8);
                disp_from_rbp <= 127 ? (cur_offset = static_cast<uint8_t>(cur_offset + (is_ext ? 5 : 4)))
                                     : (cur_offset = static_cast<uint8_t>(cur_offset + (is_ext ? 8 : 7)));

                UnwindOpSlot op;
                op.code_offset = cur_offset;
                op.unwind_op = coff::UWOP_SAVE_XMM128;
                op.op_info = static_cast<uint8_t>(x);
                op.extra_slot1 = static_cast<uint16_t>(disp_from_rsp / 16);
                op.num_slots = 2;
                ops.push_back(op);
            }
        }

        // Total 16-bit slots
        size_t total_slots = 0;
        for (const auto& op : ops) {
            total_slots += static_cast<size_t>(op.num_slots);
        }

        // Emit UNWIND_INFO Header (4 bytes)
        bool has_ehandler = fn.exception_table.has_scopes();
        uint8_t version_flags = static_cast<uint8_t>(0x01 | (has_ehandler ? 0x08 : 0x00)); // Version = 1, UNW_FLAG_EHANDLER
        uint8_t prolog_sz = static_cast<uint8_t>(fn.prologue_size > 0 ? fn.prologue_size : cur_offset);
        uint8_t cnt_codes = static_cast<uint8_t>(total_slots);

        xdata_sec->emit8(version_flags);
        xdata_sec->emit8(prolog_sz);
        xdata_sec->emit8(cnt_codes);
        xdata_sec->emit8(frame_reg_offset);

        // Win64 SEH requires UNWIND_CODE entries in REVERSE order of prologue execution
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            const auto& op = *it;
            // Primary code slot: (OpInfo << 12) | (UnwindOp << 8) | CodeOffset
            uint16_t primary = static_cast<uint16_t>(op.code_offset) |
                               (static_cast<uint16_t>(op.unwind_op & 0xF) << 8) |
                               (static_cast<uint16_t>(op.op_info & 0xF) << 12);
            xdata_sec->emit16(primary);
            if (op.num_slots >= 2) {
                xdata_sec->emit16(op.extra_slot1);
            }
            if (op.num_slots >= 3) {
                xdata_sec->emit16(op.extra_slot2);
            }
        }

        // Pad to DWORD (4-byte) alignment
        if (total_slots % 2 != 0) {
            xdata_sec->emit16(0);
        }

        if (has_ehandler) {
            if (!obj.find_symbol("brass_seh_personality")) {
                ObjectSymbol s;
                s.name = "brass_seh_personality";
                s.section_index = SECTION_UNDEF;
                s.binding = SymbolBinding::Global;
                s.type = SymbolType::Function;
                obj.add_symbol(std::move(s));
            }
            size_t handler_off = xdata_sec->data.size();
            xdata_sec->emit32(0);
            ObjectRelocation r;
            r.offset = handler_off;
            r.kind = RelocKind::Addr32NB;
            r.symbol_name = "brass_seh_personality";
            r.addend = 0;
            xdata_sec->relocations.push_back(std::move(r));

            runtime::emit_win64_seh_scope_table(*xdata_sec, fn.exception_table, fn.name);
        }

        // A RUNTIME_FUNCTION (12 bytes): [begin, end) of this function, and
        // its UNWIND_INFO; into .pdata, or into a chained UNWIND_INFO.
        auto emit_runtime_function = [&](Section& sec, uint32_t begin, uint32_t end, size_t info_offset) {
            size_t off = sec.data.size();
            const int64_t addends[3] = {begin, end, static_cast<int64_t>(info_offset)};
            for (int k = 0; k < 3; ++k) {
                sec.emit32(static_cast<uint32_t>(addends[k]));
                ObjectRelocation r;
                r.offset = off + static_cast<size_t>(k) * 4;
                r.kind = RelocKind::Addr32NB;
                r.symbol_name = k < 2 ? fn.name : std::string(".xdata");
                r.addend = addends[k];
                sec.relocations.push_back(std::move(r));
            }
        };

        // Guard exits run with RSP below the prologue's frame. Each gets a
        // chained entry that undoes that allocation and then the primary's
        // prologue; the code between them gets a chained entry with no codes
        // of its own (the primary entry's prologue offsets would misread it).
        const auto& regions = fn.stack_adjust_regions;
        const uint32_t fn_end = static_cast<uint32_t>(fn.text_size);
        const uint32_t primary_end = regions.empty() ? fn_end : regions.front().begin;
        auto emit_chained_info = [&](uint32_t alloc_bytes) {
            xdata_sec->align_to(4);
            size_t off = xdata_sec->data.size();
            std::vector<uint16_t> codes;
            if (alloc_bytes > 0) {
                if (alloc_bytes % 8 != 0) {
                    throw std::runtime_error("x64 unwind: guard exit allocation of '" + fn.name + "' is not 8-aligned");
                }
                if (alloc_bytes <= 128) {
                    codes.push_back(static_cast<uint16_t>((coff::UWOP_ALLOC_SMALL << 8) |
                                                          (((alloc_bytes - 8) / 8) << 12)));
                } else if (alloc_bytes <= 512 * 1024 - 8) {
                    codes.push_back(static_cast<uint16_t>(coff::UWOP_ALLOC_LARGE << 8));
                    codes.push_back(static_cast<uint16_t>(alloc_bytes / 8));
                } else {
                    codes.push_back(static_cast<uint16_t>((coff::UWOP_ALLOC_LARGE << 8) | (1u << 12)));
                    codes.push_back(static_cast<uint16_t>(alloc_bytes & 0xFFFF));
                    codes.push_back(static_cast<uint16_t>(alloc_bytes >> 16));
                }
            }
            xdata_sec->emit8(static_cast<uint8_t>(0x01 | (0x04 << 3))); // Version 1, UNW_FLAG_CHAININFO
            xdata_sec->emit8(0);                                        // no prologue: every code applies
            xdata_sec->emit8(static_cast<uint8_t>(codes.size()));
            xdata_sec->emit8(0);
            for (uint16_t c : codes) xdata_sec->emit16(c);
            if (codes.size() % 2 != 0) xdata_sec->emit16(0);
            emit_runtime_function(*xdata_sec, 0, primary_end, xdata_offset);
            return off;
        };

        pdata_sec->align_to(4);
        emit_runtime_function(*pdata_sec, 0, primary_end, xdata_offset);
        uint32_t cursor = primary_end;
        for (size_t i = 0; i < regions.size(); ++i) {
            const auto& reg = regions[i];
            if (reg.begin < cursor || reg.end < reg.begin || reg.end > fn_end) {
                throw std::runtime_error("x64 unwind: guard exit regions of '" + fn.name + "' overlap or are out of order");
            }
            if (reg.begin > cursor) emit_runtime_function(*pdata_sec, cursor, reg.begin, emit_chained_info(0));
            if (reg.end > reg.begin) emit_runtime_function(*pdata_sec, reg.begin, reg.end, emit_chained_info(reg.bytes));
            cursor = reg.end;
        }
        if (cursor < fn_end && !regions.empty()) {
            emit_runtime_function(*pdata_sec, cursor, fn_end, emit_chained_info(0));
        }
    }
}

} // namespace brass::object
