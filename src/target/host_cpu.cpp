#include <brass/target/host_cpu.hpp>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#include <immintrin.h>
#endif

namespace brass {

namespace {

struct HostCpu {
    bool avx2 = false;
    bool fma = false;
};

HostCpu measure() noexcept {
    HostCpu h;
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    h.avx2 = __builtin_cpu_supports("avx2") != 0;
    h.fma = __builtin_cpu_supports("fma") != 0;
#elif defined(_MSC_VER)
    int info[4] = {};
    __cpuid(info, 0);
    const int max_leaf = info[0];
    if (max_leaf < 1) return h;
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    // The OS saves the XMM and YMM state across context switches.
    if (!osxsave || !avx || (_xgetbv(0) & 0x6ULL) != 0x6ULL) return h;
    h.fma = fma;
    if (max_leaf >= 7) {
        __cpuidex(info, 7, 0);
        h.avx2 = (info[1] & (1 << 5)) != 0;
    }
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    h.fma = true;
#endif
    return h;
}

const HostCpu& host() noexcept {
    static const HostCpu h = measure();
    return h;
}

} // namespace

bool host_has_avx2() noexcept { return host().avx2; }
bool host_has_fma() noexcept { return host().fma; }

} // namespace brass
