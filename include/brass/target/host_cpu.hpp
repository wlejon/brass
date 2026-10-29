#pragma once
// What the CPU running this process supports, for choosing what code to
// emit for Target::host(). Each is measured once.

namespace brass {

// AVX2 usable here: the CPU has AVX and AVX2 and the OS saves the YMM state
// (XCR0). False on every non-x64 host.
bool host_has_avx2() noexcept;

// FMA3 usable here: on x64 the CPU's FMA bit with the OS saving YMM state;
// always true on aarch64, where fused multiply-add is base ISA.
bool host_has_fma() noexcept;

} // namespace brass
