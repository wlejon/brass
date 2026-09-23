// x64 baseline tier: unwind data for the fixed prologue
//
//     push rbp ; mov rbp, rsp ; sub rsp, frame ; [mov [rbp-8], r13]
//
// appended after the code so the OS unwinder can walk a baseline frame when
// a C++ exception thrown by a runtime helper passes through it. The body
// moves RSP around calls, so both formats unwind through RBP.
#include "baseline_emit_internal.hpp"
#include <stdexcept>

namespace brass::codegen {

namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void patch_u32(std::vector<uint8_t>& out, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) out[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

void align_to(std::vector<uint8_t>& out, size_t align, uint8_t fill) {
    while (out.size() % align) out.push_back(fill);
}

uint32_t checked_u32(size_t v) {
    if (v > 0xFFFFFFFFu) throw std::runtime_error("x64 baseline: function too large for its unwind data");
    return static_cast<uint32_t>(v);
}

// Win64 UNWIND_INFO + one RUNTIME_FUNCTION; returns the RUNTIME_FUNCTION's
// offset. Codes are listed in unwind order (descending prologue offset).
size_t append_win64(std::vector<uint8_t>& image, const X64BaselinePrologue& p, uint32_t code_size) {
    constexpr uint8_t kPushNonvol = 0, kAllocLarge = 1, kAllocSmall = 2, kSetFpreg = 3, kSaveNonvol = 4;
    constexpr uint8_t kRbp = 5, kR13 = 13;
    const uint32_t prolog_end = p.r13_save_end ? p.r13_save_end : p.alloc_end;
    if (prolog_end > 255) throw std::runtime_error("x64 baseline: prologue too long for UNWIND_INFO");

    std::vector<uint16_t> codes;
    auto code = [&](uint32_t off, uint8_t op, uint8_t info) {
        codes.push_back(static_cast<uint16_t>(off | (op << 8) | (info << 12)));
    };
    auto alloc = [&](uint32_t off, uint32_t size) {
        if (size == 0) return;
        if (size <= 128) {
            code(off, kAllocSmall, static_cast<uint8_t>(size / 8 - 1));
        } else if (size <= 512 * 1024 - 8) {
            code(off, kAllocLarge, 0);
            codes.push_back(static_cast<uint16_t>(size / 8));
        } else {
            code(off, kAllocLarge, 1);
            codes.push_back(static_cast<uint16_t>(size & 0xFFFF));
            codes.push_back(static_cast<uint16_t>(size >> 16));
        }
    };

    const uint32_t frame = static_cast<uint32_t>(p.frame_size);
    uint8_t frame_offset = 0;
    if (!p.r13_save_end) {
        // RBP = the RSP after the push; unwinding restores RSP from it.
        alloc(p.alloc_end, frame);
        code(p.mov_end, kSetFpreg, 0);
    } else {
        // R13 is saved at [rbp-8], below RBP, but a save offset is unsigned
        // from the frame base (RBP - 16*FrameOffset). Described as if the
        // frame pointer were set 16 bytes into the allocation, so the base
        // is RBP-16 and R13 sits at base+8. Every code before SET_FPREG in
        // unwind order is overridden by it, so the split is unobservable.
        frame_offset = 1;
        alloc(p.alloc_end, frame - 16);
        code(p.mov_end, kSetFpreg, 0);
        code(p.mov_end, kSaveNonvol, kR13);
        codes.push_back(1);                  // [base + 1*8]
        alloc(p.mov_end, 16);
    }
    code(p.push_end, kPushNonvol, kRbp);

    align_to(image, 4, 0xCC);
    const size_t info_off = image.size();
    image.push_back(1);                                        // version 1, no flags
    image.push_back(static_cast<uint8_t>(prolog_end));
    image.push_back(static_cast<uint8_t>(codes.size()));
    image.push_back(static_cast<uint8_t>(kRbp | (frame_offset << 4)));
    for (uint16_t c : codes) {
        image.push_back(static_cast<uint8_t>(c));
        image.push_back(static_cast<uint8_t>(c >> 8));
    }
    if (codes.size() % 2) { image.push_back(0); image.push_back(0); }

    const size_t rf_off = image.size();
    put_u32(image, 0);
    put_u32(image, code_size);
    put_u32(image, checked_u32(info_off));
    return rf_off;
}

void put_uleb(std::vector<uint8_t>& out, uint32_t v) {
    do {
        uint8_t b = v & 0x7F;
        v >>= 7;
        if (v) b |= 0x80;
        out.push_back(b);
    } while (v);
}

void advance(std::vector<uint8_t>& out, uint32_t delta) {
    if (delta == 0) return;
    if (delta < 64) out.push_back(static_cast<uint8_t>(0x40 | delta)); // DW_CFA_advance_loc
    else { out.push_back(0x02); out.push_back(static_cast<uint8_t>(delta)); } // advance_loc1
}

// A CIE, one FDE and the zero terminator; returns the CIE's offset.
size_t append_eh_frame(std::vector<uint8_t>& image, const X64BaselinePrologue& p, uint32_t code_size) {
    constexpr uint8_t kRbp = 6, kR13 = 13, kRa = 16, kRsp = 7;
    align_to(image, 8, 0xCC);
    const size_t cie = image.size();
    put_u32(image, 0);                      // length, patched
    put_u32(image, 0);                      // CIE id
    image.push_back(1);                     // version
    image.push_back('z'); image.push_back('R'); image.push_back(0);
    put_uleb(image, 1);                     // code alignment
    image.push_back(0x78);                  // data alignment -8 (sleb)
    image.push_back(kRa);
    put_uleb(image, 1);                     // augmentation data length
    image.push_back(0x1B);                  // FDE pointers: pcrel | sdata4
    image.push_back(0x0C); image.push_back(kRsp); image.push_back(8); // def_cfa rsp+8
    image.push_back(0x80 | kRa); image.push_back(1);                  // ra at cfa-8
    align_to(image, 8, 0);
    patch_u32(image, cie, checked_u32(image.size() - cie - 4));

    const size_t fde = image.size();
    put_u32(image, 0);                                             // length, patched
    put_u32(image, checked_u32(image.size() - cie));               // CIE pointer
    const size_t pc_field = image.size();
    put_u32(image, static_cast<uint32_t>(-static_cast<int64_t>(pc_field))); // pcrel to offset 0
    put_u32(image, code_size);
    put_uleb(image, 0);                                            // augmentation data length
    advance(image, p.push_end);
    image.push_back(0x0E); put_uleb(image, 16);                    // def_cfa_offset 16
    image.push_back(0x80 | kRbp); put_uleb(image, 2);              // rbp at cfa-16
    advance(image, p.mov_end - p.push_end);
    image.push_back(0x0D); put_uleb(image, kRbp);                  // def_cfa_register rbp
    if (p.r13_save_end) {
        advance(image, p.r13_save_end - p.mov_end);
        image.push_back(0x80 | kR13); put_uleb(image, 3);          // r13 at cfa-24 = rbp-8
    }
    align_to(image, 8, 0);
    patch_u32(image, fde, checked_u32(image.size() - fde - 4));
    put_u32(image, 0);                                             // terminator
    return cie;
}

} // namespace

size_t append_x64_baseline_unwind(std::vector<uint8_t>& image, const X64BaselinePrologue& p,
                                  uint32_t code_size, bool windows) {
    return windows ? append_win64(image, p, code_size) : append_eh_frame(image, p, code_size);
}

} // namespace brass::codegen
