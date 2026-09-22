#include <brass/target/aarch64/aarch64_logical_imm.hpp>
#include <bit>

namespace brass::aarch64 {

bool encode_logical_immediate(uint64_t val, bool is_64bit, uint32_t& n, uint32_t& immr, uint32_t& imms) noexcept {
    const uint32_t size = is_64bit ? 64u : 32u;
    if (!is_64bit) {
        val &= 0xFFFFFFFFULL;
    }
    if (val == 0 || (is_64bit && val == ~0ULL) || (!is_64bit && val == 0xFFFFFFFFULL)) {
        return false;
    }

    // Find the smallest repeating element size (2, 4, 8, 16, 32, 64).
    uint32_t esize = size;
    while (esize > 2u) {
        uint32_t half = esize / 2u;
        uint64_t mask = (1ULL << half) - 1ULL;
        if ((val & mask) != ((val >> half) & mask)) {
            break;
        }
        esize = half;
    }

    uint64_t emask = (esize == 64u) ? ~0ULL : ((1ULL << esize) - 1ULL);
    uint64_t elem = val & emask;

    // An element is valid if it is a cyclic rotation of a contiguous run of 1s
    // of length 1 <= run_len < esize.
    bool found = false;
    uint32_t immr_found = 0;
    uint32_t run_len_found = 0;

    for (uint32_t r = 0; r < esize; ++r) {
        uint64_t rot_elem;
        if (r == 0) {
            rot_elem = elem;
        } else {
            rot_elem = (((elem << r) & emask) | (elem >> (esize - r))) & emask;
        }
        // rot_elem must be of the form (1 << run_len) - 1 with 1 <= run_len < esize
        if ((rot_elem & (rot_elem + 1ULL)) == 0 && rot_elem != 0 && rot_elem != emask) {
            uint32_t run_len = static_cast<uint32_t>(std::popcount(rot_elem));
            immr_found = r;
            run_len_found = run_len;
            found = true;
            break;
        }
    }

    if (!found) {
        return false;
    }

    n = (esize == 64u) ? 1u : 0u;
    uint32_t prefix = (esize < 64u) ? ((~(esize * 2u - 1u)) & 0x3Fu) : 0u;
    imms = prefix | (run_len_found - 1u);
    immr = immr_found;
    return true;
}

} // namespace brass::aarch64
