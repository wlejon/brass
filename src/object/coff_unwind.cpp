#include <brass/object/coff_writer.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <vector>
#include <algorithm>

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

static void build_aarch64_unwind_info(ObjectFile& obj) {
    Section* pdata_sec = obj.get_section(".pdata");
    Section* xdata_sec = obj.get_section(".xdata");
    if (!pdata_sec || !xdata_sec) return;

    for (const auto& fn : obj.functions) {
        bool has_ehandler = fn.exception_table.has_scopes();
        auto saved_gprs = aarch64::AArch64FrameLayout::get_saved_callee_gprs(fn.frame_info);
        auto saved_fprs = aarch64::AArch64FrameLayout::get_saved_callee_fprs(fn.frame_info);
        size_t frame_sz = fn.frame_info.total_frame_size;
        size_t fn_len_words = fn.text_size / 4;

        // On ARM64 Windows COFF SEH, .pdata entries are 8 bytes:
        // Word 0: BeginAddress (Addr32NB to fn.name)
        // Word 1: UnwindData (Packed unwind data if flag=01, or Addr32NB to .xdata if flag=00)
        bool can_pack = !has_ehandler && (frame_sz <= 2032) && (fn_len_words <= 0x7FF);

        if (can_pack) {
            pdata_sec->align_to(4);
            size_t pdata_offset = pdata_sec->data.size();

            // Word 0: BeginAddress
            pdata_sec->emit32(static_cast<uint32_t>(fn.text_offset));
            ObjectRelocation r0;
            r0.offset = pdata_offset + 0;
            r0.kind = RelocKind::Addr32NB;
            r0.symbol_name = fn.name;
            r0.addend = 0;
            pdata_sec->relocations.push_back(std::move(r0));

            // Word 1: Packed Unwind Data
            uint32_t packed = 0x1u; // Flag = 01b
            packed |= (static_cast<uint32_t>(fn_len_words & 0x7FF) << 2); // FunctionLength (11 bits)
            packed |= (0x1u << 13); // Ret = 01b (canonical ret)
            // H = bit 15 = 0
            packed |= (static_cast<uint32_t>(saved_gprs.size() & 0xF) << 16); // RegI (4 bits)
            packed |= (static_cast<uint32_t>(saved_fprs.size() & 0x7) << 20); // RegF (3 bits)
            packed |= (static_cast<uint32_t>(fn.frame_info.is_leaf ? 0 : 1) << 23); // CR (2 bits, 01b = chained)
            packed |= (static_cast<uint32_t>((frame_sz / 16) & 0x7F) << 25); // FrameSize (7 bits)

            pdata_sec->emit32(packed);
        } else {
            // Full .xdata record
            xdata_sec->align_to(4);
            size_t xdata_offset = xdata_sec->data.size();

            std::vector<uint8_t> unwind_codes;
            if (fn.frame_info.is_leaf) {
                unwind_codes.push_back(0xE6); // end
            } else {
                if (frame_sz <= 128) {
                    uint8_t imm = static_cast<uint8_t>(frame_sz / 16);
                    unwind_codes.push_back(0xC8 | ((imm > 0 ? imm - 1 : 0) & 0x07));
                } else {
                    uint8_t imm = static_cast<uint8_t>((frame_sz / 16) & 0x7F);
                    unwind_codes.push_back(imm);
                    unwind_codes.push_back(0xE1); // set_fp
                }
                unwind_codes.push_back(0xE6); // end
            }

            while (unwind_codes.size() % 4 != 0) {
                unwind_codes.push_back(0xE6);
            }

            uint32_t code_words = static_cast<uint32_t>(unwind_codes.size() / 4);
            uint32_t header = static_cast<uint32_t>(fn_len_words & 0x3FFFF);
            if (has_ehandler) {
                header |= (1u << 20); // X = 1
            }
            header |= (1u << 21); // E = 1 (single epilogue)
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

                runtime::emit_win64_seh_scope_table(*xdata_sec, fn.exception_table);
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

            runtime::emit_win64_seh_scope_table(*xdata_sec, fn.exception_table);
        }

        // Emit RUNTIME_FUNCTION in .pdata (12 bytes)
        pdata_sec->align_to(4);
        size_t pdata_offset = pdata_sec->data.size();

        // 1. BeginAddress
        pdata_sec->emit32(0);
        ObjectRelocation r0;
        r0.offset = pdata_offset + 0;
        r0.kind = RelocKind::Addr32NB;
        r0.symbol_name = fn.name;
        r0.addend = 0;
        pdata_sec->relocations.push_back(std::move(r0));

        // 2. EndAddress
        pdata_sec->emit32(static_cast<uint32_t>(fn.text_size));
        ObjectRelocation r1;
        r1.offset = pdata_offset + 4;
        r1.kind = RelocKind::Addr32NB;
        r1.symbol_name = fn.name;
        r1.addend = static_cast<int64_t>(fn.text_size);
        pdata_sec->relocations.push_back(std::move(r1));

        // 3. UnwindInfoAddress
        pdata_sec->emit32(static_cast<uint32_t>(xdata_offset));
        ObjectRelocation r2;
        r2.offset = pdata_offset + 8;
        r2.kind = RelocKind::Addr32NB;
        r2.symbol_name = ".xdata";
        r2.addend = static_cast<int64_t>(xdata_offset);
        pdata_sec->relocations.push_back(std::move(r2));
    }
}

} // namespace brass::object
