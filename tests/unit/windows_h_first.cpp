// <windows.h> before <brass/brass.hpp>, without NOMINMAX or
// WIN32_LEAN_AND_MEAN, as an embedding host commonly writes it. The umbrella
// header once failed here: the COFF/PE writers' IMAGE_* constants collide
// with <winnt.h> macros, and std::min / std::max calls with the min/max
// macros. Building this file is most of the test; main checks that both the
// Windows macros and brass's constants are still usable afterwards.

#include <windows.h>
#include <brass/brass.hpp>

#include <cstdio>

int main() {
    int failures = 0;
    // The Windows macros survive the brass headers.
    static_assert(IMAGE_SCN_MEM_READ == 0x40000000, "winnt.h macro restored");
    static_assert(IMAGE_FILE_MACHINE_AMD64 == 0x8664, "winnt.h macro restored");
    volatile int two = 2;
    volatile int three = 3;
    if (max(two, three) != 3 || min(two, three) != 2) ++failures;

    // brass headers that call std::min / std::max still work.
    brass::BitSet a(10);
    brass::BitSet b(70);
    a.set(3);
    b.set(3);
    b.set(65);
    a &= b;
    if (!a.test(3)) ++failures;
    a |= b;
    if (!a.test(65)) ++failures;

    if (failures != 0) {
        std::printf("FAIL: %d\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
