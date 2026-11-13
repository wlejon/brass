// The filter of Tracer::visit_run (tracer.hpp).
//
// A collection's trace is mostly one loop: an object's words visited one
// after another, few of them naming anything the collection moves or marks.
// In a minor collection a slot of a survivor, or of an old object on a dirty
// card, holds a number, a constant or an old reference nine times in ten. So
// a run is filtered before any slot is handled: four slots tested against
// the tracer's range in one compare.

#include <brass/gc/tracer.hpp>
#include <brass/target/host_cpu.hpp>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define BRASS_GC_RUN_AVX2 1
#else
#define BRASS_GC_RUN_AVX2 0
#endif

namespace brass::gc::detail {

#if BRASS_GC_RUN_AVX2

const bool run_filter_avx2 = host_has_avx2();

// A leaf: no frame, nothing spilled, the three constants built per call.
__attribute__((target("avx2"))) size_t run_groups_outside_avx2(const uint64_t* slots, size_t groups, uintptr_t lo,
                                                                uintptr_t span) noexcept {
    // address - lo < span, unsigned, as a signed compare: both sides biased
    // by 2^63, the bias folded into the two constants.
    constexpr uint64_t kBias = uint64_t{1} << 63;
    const __m256i mask = _mm256_set1_epi64x(static_cast<long long>(kAddressMask));
    const __m256i low = _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(lo) + kBias));
    const __m256i limit = _mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(span) + kBias));
    size_t g = 0;
    for (; g < groups; ++g) {
        const __m256i words = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(slots + g * 4));
        const __m256i offset = _mm256_sub_epi64(_mm256_and_si256(words, mask), low);
        const __m256i inside = _mm256_cmpgt_epi64(limit, offset);
        if (!_mm256_testz_si256(inside, inside)) break;
    }
    return g;
}

#else

const bool run_filter_avx2 = false;

size_t run_groups_outside_avx2(const uint64_t*, size_t, uintptr_t, uintptr_t) noexcept { return 0; }

#endif

} // namespace brass::gc::detail
