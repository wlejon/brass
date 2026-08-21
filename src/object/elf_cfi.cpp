#include <brass/object/elf_writer.hpp>
#include <brass/target/x64/x64_frame.hpp>
#include <vector>
#include <algorithm>

namespace brass::object {

using namespace brass::x64;

namespace {

uint8_t to_dwarf_gpr(GPR reg) {
    switch (reg) {
        case GPR::RAX: return 0;
        case GPR::RDX: return 1;
        case GPR::RCX: return 2;
        case GPR::RBX: return 3;
        case GPR::RSI: return 4;
        case GPR::RDI: return 5;
        case GPR::RBP: return 6;
        case GPR::RSP: return 7;
        case GPR::R8:  return 8;
        case GPR::R9:  return 9;
        case GPR::R10: return 10;
        case GPR::R11: return 11;
        case GPR::R12: return 12;
        case GPR::R13: return 13;
        case GPR::R14: return 14;
        case GPR::R15: return 15;
        default: return 0;
    }
}

void patch_u32_le(std::vector<uint8_t>& buf, size_t offset, uint32_t val) {
    buf[offset + 0] = static_cast<uint8_t>(val & 0xFF);
    buf[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    buf[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    buf[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

} // namespace

void ElfCfiBuilder::build_eh_frame(
    ObjectFile& obj,
    Section& eh_frame_sec
) {
    eh_frame_sec.align_to(8);

    // 1. Common Information Entry (CIE)
    size_t cie_start = eh_frame_sec.data.size();
    eh_frame_sec.emit32(0); // Length placeholder
    eh_frame_sec.emit32(0); // CIE ID = 0 (for .eh_frame)
    eh_frame_sec.emit8(1);  // Version = 1
    
    // Augmentation: "zR\0"
    eh_frame_sec.emit8('z');
    eh_frame_sec.emit8('R');
    eh_frame_sec.emit8(0);

    eh_frame_sec.emit8(1);    // Code alignment factor = 1 (ULEB128)
    eh_frame_sec.emit8(0x78); // Data alignment factor = -8 (SLEB128)
    eh_frame_sec.emit8(16);   // Return address register = 16 (RIP) (ULEB128)
    eh_frame_sec.emit8(1);    // Augmentation data length = 1 (ULEB128)
    eh_frame_sec.emit8(elf::DW_EH_PE_pcrel | elf::DW_EH_PE_sdata4); // 0x1B

    // Initial instructions:
    // DW_CFA_def_cfa (RSP, 8)
    eh_frame_sec.emit8(elf::DW_CFA_def_cfa);
    eh_frame_sec.emit8(7); // RSP
    eh_frame_sec.emit8(8); // offset 8

    // DW_CFA_offset (RIP, 1 * -8 = -8)
    eh_frame_sec.emit8(elf::DW_CFA_offset | 16);
    eh_frame_sec.emit8(1);

    eh_frame_sec.align_to(8);
    uint32_t cie_len = static_cast<uint32_t>(eh_frame_sec.data.size() - cie_start - 4);
    patch_u32_le(eh_frame_sec.data, cie_start, cie_len);

    // 2. Frame Description Entries (FDE) for each function
    for (const auto& fn : obj.functions) {
        size_t fde_start = eh_frame_sec.data.size();
        eh_frame_sec.emit32(0); // Length placeholder
        
        // CIE Pointer: offset from this field to CIE start
        uint32_t cie_pointer = static_cast<uint32_t>(fde_start + 4 - cie_start);
        eh_frame_sec.emit32(cie_pointer);

        // PC Begin (4 bytes, DW_EH_PE_pcrel | DW_EH_PE_sdata4)
        size_t pc_begin_offset = eh_frame_sec.data.size();
        eh_frame_sec.emit32(static_cast<uint32_t>(fn.text_offset));

        ObjectRelocation r;
        r.offset = pc_begin_offset;
        r.kind = RelocKind::PCRel32;
        r.symbol_name = fn.name;
        r.addend = 0;
        eh_frame_sec.relocations.push_back(std::move(r));

        // PC Range (4 bytes)
        eh_frame_sec.emit32(static_cast<uint32_t>(fn.text_size));

        // Augmentation Data Length = 0 (ULEB128)
        eh_frame_sec.emit8(0);

        if (!fn.frame_info.is_leaf) {
            // Call Frame Instructions:
            // After push rbp (1 byte)
            eh_frame_sec.emit8(elf::DW_CFA_advance_loc | 1);
            eh_frame_sec.emit8(elf::DW_CFA_def_cfa_offset);
            eh_frame_sec.emit8(16);
            eh_frame_sec.emit8(elf::DW_CFA_offset | 6); // RBP
            eh_frame_sec.emit8(2); // offset 2 * -8 = -16

            // After mov rbp, rsp (3 bytes)
            eh_frame_sec.emit8(elf::DW_CFA_advance_loc | 3);
            eh_frame_sec.emit8(elf::DW_CFA_def_cfa_register);
            eh_frame_sec.emit8(6); // RBP

            // Callee-saved GPRs (if any)
            auto saved_gprs = X64FrameLayout::get_saved_callee_gprs(fn.frame_info);
            for (size_t i = 0; i < saved_gprs.size(); ++i) {
                uint8_t dreg = to_dwarf_gpr(saved_gprs[i]);
                uint8_t factored = static_cast<uint8_t>(i + 1); // [rbp - (i+1)*8]
                eh_frame_sec.emit8(elf::DW_CFA_offset | dreg);
                eh_frame_sec.emit8(factored);
            }
        }

        eh_frame_sec.align_to(8);
        uint32_t fde_len = static_cast<uint32_t>(eh_frame_sec.data.size() - fde_start - 4);
        patch_u32_le(eh_frame_sec.data, fde_start, fde_len);
    }

    // 3. Terminator: 4 zero bytes
    eh_frame_sec.emit32(0);
}

} // namespace brass::object
