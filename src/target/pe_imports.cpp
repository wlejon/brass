#include "pe_imports.hpp"
#include "image_util.hpp"

namespace brass::target::pe_imports {

using namespace brass::target::image;

namespace {

constexpr uint32_t kDescriptorSize = 20;   // IMAGE_IMPORT_DESCRIPTOR

// Size of one hint/name entry: u16 hint, the name, NUL, padded to even.
size_t hint_name_size(const std::string& name) {
    return static_cast<size_t>(align_up(2 + name.size() + 1, 2));
}

} // namespace

uint32_t Table::idata_size() const {
    const auto groups = plan.by_library();
    size_t size = (plan.libraries.size() + 1) * kDescriptorSize;
    size = static_cast<size_t>(align_up(size, 8));
    // ILT then IAT, each one 8-byte entry per symbol plus a terminator, per
    // library, laid out contiguously so the IAT directory covers one range.
    size_t table_entries = 0;
    for (const auto& g : groups) table_entries += g.size() + 1;
    size += table_entries * 8 * 2;
    for (const auto& s : plan.symbols) size += hint_name_size(s.name);
    for (const auto& lib : plan.libraries) size += lib.size() + 1;
    return static_cast<uint32_t>(size);
}

void Table::append_thunks(std::vector<uint8_t>& text) {
    if (plan.empty()) return;
    thunk_stride = aarch64 ? 16 : 8;
    pad_to(text, 16);
    thunk_base = text.size();
    for (size_t i = 0; i < plan.symbols.size(); ++i) {
        if (aarch64) {
            // Patched in emit: adrp x16, iat ; ldr x16, [x16, #lo12] ; br x16 ; nop
            write_u32(text, 0x90000010u);
            write_u32(text, 0xF9400210u);
            write_u32(text, aarch64_br(16));
            write_u32(text, 0xD503201Fu);
        } else {
            // jmp qword ptr [rip + disp32], then int3 padding to 8.
            write_u8(text, 0xFF);
            write_u8(text, 0x25);
            write_u32(text, 0);
            write_u8(text, 0xCC);
            write_u8(text, 0xCC);
        }
    }
}

void Table::emit(std::vector<uint8_t>& idata, uint32_t idata_rva, std::vector<uint8_t>& text,
                 uint32_t text_rva) {
    idata.clear();
    if (plan.empty()) return;
    const auto groups = plan.by_library();
    const size_t nlibs = plan.libraries.size();

    // Offsets within .idata.
    const size_t dir_off = 0;
    size_t cursor = static_cast<size_t>(align_up((nlibs + 1) * kDescriptorSize, 8));
    std::vector<size_t> ilt_off(nlibs), iat_off(nlibs);
    for (size_t l = 0; l < nlibs; ++l) {
        ilt_off[l] = cursor;
        cursor += (groups[l].size() + 1) * 8;
    }
    const size_t iat_begin = cursor;
    for (size_t l = 0; l < nlibs; ++l) {
        iat_off[l] = cursor;
        cursor += (groups[l].size() + 1) * 8;
    }
    const size_t iat_end = cursor;
    std::vector<size_t> name_off(plan.symbols.size());
    for (const auto& s : plan.symbols) {
        name_off[s.index] = cursor;
        cursor += hint_name_size(s.name);
    }
    std::vector<size_t> lib_off(nlibs);
    for (size_t l = 0; l < nlibs; ++l) {
        lib_off[l] = cursor;
        cursor += plan.libraries[l].size() + 1;
    }
    idata.resize(cursor, 0);

    // Import directory: one descriptor per library, then an all-zero one.
    for (size_t l = 0; l < nlibs; ++l) {
        const size_t d = dir_off + l * kDescriptorSize;
        patch_u32(idata, d + 0, idata_rva + static_cast<uint32_t>(ilt_off[l]));   // OriginalFirstThunk
        patch_u32(idata, d + 4, 0);                                               // TimeDateStamp
        patch_u32(idata, d + 8, 0);                                               // ForwarderChain
        patch_u32(idata, d + 12, idata_rva + static_cast<uint32_t>(lib_off[l]));  // Name
        patch_u32(idata, d + 16, idata_rva + static_cast<uint32_t>(iat_off[l]));  // FirstThunk
    }

    // Lookup and address tables: both name the hint/name entry (bit 63 clear
    // = import by name). The loader overwrites the IAT copy with addresses.
    for (size_t l = 0; l < nlibs; ++l) {
        for (size_t k = 0; k < groups[l].size(); ++k) {
            const size_t sym = groups[l][k];
            const uint64_t entry = idata_rva + static_cast<uint64_t>(name_off[sym]);
            patch_u64(idata, ilt_off[l] + k * 8, entry);
            patch_u64(idata, iat_off[l] + k * 8, entry);
        }
    }

    for (const auto& s : plan.symbols) {
        const size_t off = name_off[s.index];
        idata[off] = 0;       // hint
        idata[off + 1] = 0;
        std::memcpy(idata.data() + off + 2, s.name.data(), s.name.size());
    }
    for (size_t l = 0; l < nlibs; ++l) {
        std::memcpy(idata.data() + lib_off[l], plan.libraries[l].data(), plan.libraries[l].size());
    }

    directory_rva = idata_rva;
    directory_size = static_cast<uint32_t>((nlibs + 1) * kDescriptorSize);
    iat_rva = idata_rva + static_cast<uint32_t>(iat_begin);
    iat_size = static_cast<uint32_t>(iat_end - iat_begin);

    // The thunks: each jumps through its own IAT slot.
    for (size_t l = 0; l < nlibs; ++l) {
        for (size_t k = 0; k < groups[l].size(); ++k) {
            const size_t sym = groups[l][k];
            const uint32_t slot_rva = idata_rva + static_cast<uint32_t>(iat_off[l] + k * 8);
            const size_t thunk = thunk_base + sym * thunk_stride;
            const uint32_t thunk_at = text_rva + static_cast<uint32_t>(thunk);
            if (aarch64) {
                patch_u32(text, thunk + 0, aarch64_adrp(16, thunk_at, slot_rva));
                patch_u32(text, thunk + 4, aarch64_ldr_x_uoff(16, 16, slot_rva));
            } else {
                const int64_t disp = static_cast<int64_t>(slot_rva) - static_cast<int64_t>(thunk_at + 6);
                patch_u32(text, thunk + 2, static_cast<uint32_t>(static_cast<int32_t>(disp)));
            }
        }
    }
}

} // namespace brass::target::pe_imports
