// AArch64 baseline tier: unwind data for the fixed prologue
//
//     stp x29, x30, [sp, #-16]! ; mov x29, sp ; sub sp, sp, #frame
//     [str x28, [x29, #-8]]
//
// appended after the code so the unwinder can walk a baseline frame when a
// C++ exception thrown by a runtime helper passes through it. sp is fixed
// after the prologue, but the CFA is described through x29 anyway, as the
// frame record is what every walker follows.
#include "aarch64_baseline_emit_internal.hpp"
#include <stdexcept>

namespace brass::aarch64 {

namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void patch_u32(std::vector<uint8_t>& out, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) out[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

void put_uleb(std::vector<uint8_t>& out, uint64_t v) {
    do {
        uint8_t b = v & 0x7F;
        v >>= 7;
        if (v) b |= 0x80;
        out.push_back(b);
    } while (v);
}

void align_to(std::vector<uint8_t>& out, size_t align) {
    while (out.size() % align) out.push_back(0);
}

uint32_t checked_u32(size_t v) {
    if (v > 0xFFFFFFFFu) throw std::runtime_error("aarch64 baseline: function too large for its unwind data");
    return static_cast<uint32_t>(v);
}

// `delta` bytes of code, in the CIE's 4-byte code alignment units.
void advance(std::vector<uint8_t>& out, uint32_t delta) {
    const uint32_t units = delta / 4;
    if (units == 0) return;
    if (units < 64) out.push_back(static_cast<uint8_t>(0x40 | units)); // DW_CFA_advance_loc
    else if (units < 256) { out.push_back(0x02); out.push_back(static_cast<uint8_t>(units)); } // advance_loc1
    else { out.push_back(0x03); out.push_back(static_cast<uint8_t>(units)); out.push_back(static_cast<uint8_t>(units >> 8)); }
}

} // namespace

size_t append_aarch64_baseline_eh_frame(std::vector<uint8_t>& image, uint32_t code_size,
                                        uint32_t stp_end, uint32_t mov_end, uint32_t tls_save_end) {
    constexpr uint8_t kX28 = 28, kFp = 29, kLr = 30, kSp = 31;
    align_to(image, 8);
    const size_t cie = image.size();
    put_u32(image, 0);                      // length, patched
    put_u32(image, 0);                      // CIE id
    image.push_back(1);                     // version
    image.push_back('z'); image.push_back('R'); image.push_back(0);
    put_uleb(image, 4);                     // code alignment
    image.push_back(0x78);                  // data alignment -8 (sleb)
    image.push_back(kLr);                   // return address column
    put_uleb(image, 1);                     // augmentation data length
    image.push_back(0x1B);                  // FDE pointers: pcrel | sdata4
    image.push_back(0x0C); image.push_back(kSp); image.push_back(0); // def_cfa sp+0
    align_to(image, 8);
    patch_u32(image, cie, checked_u32(image.size() - cie - 4));

    const size_t fde = image.size();
    put_u32(image, 0);                                             // length, patched
    put_u32(image, checked_u32(image.size() - cie));               // CIE pointer
    const size_t pc_field = image.size();
    put_u32(image, static_cast<uint32_t>(-static_cast<int64_t>(pc_field))); // pcrel to offset 0
    put_u32(image, code_size);
    put_uleb(image, 0);                                            // augmentation data length
    advance(image, stp_end);
    image.push_back(0x0E); put_uleb(image, 16);                    // def_cfa_offset 16
    image.push_back(0x80 | kFp); put_uleb(image, 2);               // x29 at cfa-16
    image.push_back(0x80 | kLr); put_uleb(image, 1);               // x30 at cfa-8
    advance(image, mov_end - stp_end);
    image.push_back(0x0D); put_uleb(image, kFp);                   // def_cfa_register x29
    if (tls_save_end) {
        advance(image, tls_save_end - mov_end);
        image.push_back(0x80 | kX28); put_uleb(image, 3);          // x28 at cfa-24 = fp-8
    }
    align_to(image, 8);
    patch_u32(image, fde, checked_u32(image.size() - fde - 4));
    put_u32(image, 0);                                             // terminator
    return cie;
}

} // namespace brass::aarch64
