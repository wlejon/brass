// The quantized GEMV kernels (Q8_0, Q4_K) through SpirvTarget on a Vulkan
// device, against host references that dequantize exactly as the CPU
// dequantizers do (double accumulation) and against the CPU KernelJit GEMVs,
// over the shapes and the 1e-4 tolerance test_gpu_kernels_quant.cpp uses on
// CUDA: random weights (fixed seed, f16 scales in a sane range, every 6-bit
// scale field), n above the grid (those rows stay 0), blocks 32/64..1024,
// k counts that exercise the unrolled loop's remainder. Each shape also runs
// with subgroup size 64 when the device can require it. [SKIP] without a
// Vulkan device.

#include "spirv_exec_support.hpp"
#include "ptx_test_support.hpp"   // f16 host conversions

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>

using namespace brass;
using namespace spvexec;
using codegen::GpuKernel;
using codegen::KernelFunction;
using codegen::MlFusionCompiler;
using ptxtest::f16_to_f32_host;
using ptxtest::f32_to_f16_host;

namespace {

constexpr float kReferenceTol = 1e-4f;

struct Q8Block { uint16_t d; int8_t qs[32]; };
struct Q4KBlock { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(Q8Block) == 34, "Q8_0 block is 34 bytes");
static_assert(sizeof(Q4KBlock) == 144, "Q4_K super-block is 144 bytes");

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (static_cast<float>(next() & 0xFFFFFF) / 16777216.0f); }
    uint8_t byte() { return static_cast<uint8_t>(next() >> 24); }
};

std::vector<Q8Block> random_q8(uint32_t n, uint32_t k, Rng& rng) {
    std::vector<Q8Block> w(static_cast<size_t>(n) * (k / 32));
    for (Q8Block& blk : w) {
        blk.d = f32_to_f16_host(rng.uniform(0.004f, 0.05f));
        for (int8_t& q : blk.qs) q = static_cast<int8_t>(static_cast<int>(rng.byte()) - 128);
    }
    return w;
}

std::vector<Q4KBlock> random_q4k(uint32_t n, uint32_t k, Rng& rng) {
    std::vector<Q4KBlock> w(static_cast<size_t>(n) * (k / 256));
    for (Q4KBlock& blk : w) {
        blk.d = f32_to_f16_host(rng.uniform(0.002f, 0.02f));
        blk.dmin = f32_to_f16_host(rng.uniform(0.001f, 0.01f));
        for (uint8_t& s : blk.scales) s = rng.byte();
        for (uint8_t& q : blk.qs) q = rng.byte();
    }
    return w;
}

std::vector<float> random_x(uint32_t k, Rng& rng) {
    std::vector<float> x(k);
    for (float& v : x) v = rng.uniform(-1.0f, 1.0f);
    return x;
}

void host_dequant_q8(const Q8Block& blk, float* out32) {
    float d = f16_to_f32_host(blk.d);
    for (int i = 0; i < 32; ++i) out32[i] = static_cast<float>(blk.qs[i]) * d;
}

void host_dequant_q4k(const Q4KBlock& blk, float* out256) {
    float d = f16_to_f32_host(blk.d), dmin = f16_to_f32_host(blk.dmin);
    const uint8_t* scales = blk.scales;
    uint8_t sc[8], m[8];
    for (int j = 0; j < 8; ++j) {
        if (j < 4) { sc[j] = scales[j] & 0x3F; m[j] = scales[j + 4] & 0x3F; }
        else {
            sc[j] = static_cast<uint8_t>((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            m[j] = static_cast<uint8_t>((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
    }
    for (int p = 0; p < 4; ++p) {
        int lo = 2 * p, hi = 2 * p + 1;
        float wlo = static_cast<float>(sc[lo]) * d, whi = static_cast<float>(sc[hi]) * d;
        float blo = static_cast<float>(m[lo]) * dmin, bhi = static_cast<float>(m[hi]) * dmin;
        for (int l = 0; l < 32; ++l) {
            uint8_t byte = blk.qs[p * 32 + l];
            out256[lo * 32 + l] = wlo * static_cast<float>(byte & 0x0F) - blo;
            out256[hi * 32 + l] = whi * static_cast<float>((byte >> 4) & 0x0F) - bhi;
        }
    }
}

template <typename Block, size_t BlockK, void (*Dequant)(const Block&, float*)>
std::vector<float> host_gemv(const std::vector<Block>& w, const std::vector<float>& x, uint32_t n, uint32_t k, uint32_t grid) {
    std::vector<float> y(n, 0.0f);
    uint32_t bpr = k / static_cast<uint32_t>(BlockK);
    float tmp[BlockK];
    for (uint32_t r = 0; r < std::min(n, grid); ++r) {
        double acc = 0.0;
        for (uint32_t b = 0; b < bpr; ++b) {
            Dequant(w[static_cast<size_t>(r) * bpr + b], tmp);
            for (size_t i = 0; i < BlockK; ++i) acc += static_cast<double>(tmp[i]) * static_cast<double>(x[b * BlockK + i]);
        }
        y[r] = static_cast<float>(acc);
    }
    return y;
}

// Device run: the weights as raw bytes (the blocks are packed, 2-byte aligned).
template <typename Block>
std::vector<float> run_quant(const VulkanModule& m, const std::vector<Block>& w, const std::vector<float>& x, uint32_t n,
                             uint32_t k, uint32_t grid, uint32_t block) {
    std::vector<uint8_t> raw(w.size() * sizeof(Block));
    std::memcpy(raw.data(), w.data(), raw.size());
    VulkanBuffer dw = upload(raw), dx = upload(x), dy = zeros<float>(n);
    launch(m, grid, block, {dw, dx, dy, n, k});
    return download<float>(dy, n);
}

void check_rows(const char* what, const std::vector<float>& got, const std::vector<float>& ref, uint32_t grid) {
    float worst = 0.0f;
    for (size_t r = 0; r < got.size(); ++r) {
        if (r >= grid) { CHECK(got[r] == 0.0f); continue; }
        float d = std::fabs(got[r] - ref[r]) / (1.0f + std::fabs(ref[r]));
        if (!(d <= kReferenceTol)) std::fprintf(stderr, "%s: row %zu device %.9g host %.9g\n", what, r, got[r], ref[r]);
        worst = std::max(worst, d);
    }
    std::printf("    %-52s max rel diff %.3g\n", what, static_cast<double>(worst));
    CHECK(worst <= kReferenceTol);
}

struct QuantShape { uint32_t n, k, grid, block; };
const QuantShape kQ8Shapes[] = {
    {1, 32, 1, 256}, {3, 64, 3, 256}, {17, 96, 17, 256}, {64, 1024, 64, 256}, {3, 4096, 3, 256}, {17, 1024, 8, 256},
    {3, 96, 3, 32}, {64, 64, 64, 32}, {17, 1024, 17, 128}, {4, 4096, 4, 1024}, {1, 32, 1, 1024},
    {5, 1120, 5, 256}, {2, 2336, 2, 128}, {3, 8192, 3, 256},
};
const QuantShape kQ4KShapes[] = {
    {1, 256, 1, 256}, {3, 512, 3, 256}, {17, 1024, 17, 256}, {64, 4096, 64, 256}, {17, 4096, 8, 256},
    {3, 512, 3, 64}, {64, 256, 64, 64}, {17, 1024, 17, 128}, {4, 4096, 4, 1024}, {1, 256, 1, 1024},
    {5, 1280, 5, 256}, {2, 2816, 2, 128}, {3, 8192, 3, 256},
};

std::vector<uint32_t> subgroup_sizes() {
    std::vector<uint32_t> s = {32};
    const brass::gpu::VulkanDeviceCaps& c = caps();
    if (c.can_require_subgroup_size_in_compute && c.max_subgroup_size >= 64 && c.min_subgroup_size <= 64) s.push_back(64);
    return s;
}

} // namespace

TEST_CASE("SPIR-V kernels - GEMV Q8_0 matches the host reference across shapes and subgroup sizes") {
    if (!vk_ready()) return;
    MlFusionCompiler c;
    target::SpirvKernel sk = c.compile_spirv(GpuKernel::gemv_q8_0);
    CHECK(spvtest::spirv_val(sk.words));
    for (uint32_t sg : subgroup_sizes()) {
        VulkanModuleOptions mo;
        mo.subgroup_size = sg;
        VulkanModule m = load_kernel(sk, mo);
        Rng rng(0x5C0DE8u);
        for (QuantShape s : kQ8Shapes) {
            std::vector<Q8Block> w = random_q8(s.n, s.k, rng);
            std::vector<float> x = random_x(s.k, rng);
            char what[96];
            std::snprintf(what, sizeof(what), "gemv_q8_0 n=%u k=%u grid=%u block=%u sg=%u", s.n, s.k, s.grid, s.block, sg);
            check_rows(what, run_quant(m, w, x, s.n, s.k, s.grid, s.block),
                       host_gemv<Q8Block, 32, host_dequant_q8>(w, x, s.n, s.k, s.grid), s.grid);
        }
    }
}

TEST_CASE("SPIR-V kernels - GEMV Q4_K matches the host reference across shapes and subgroup sizes") {
    if (!vk_ready()) return;
    MlFusionCompiler c;
    target::SpirvKernel sk = c.compile_spirv(GpuKernel::gemv_q4_k);
    CHECK(spvtest::spirv_val(sk.words));
    for (uint32_t sg : subgroup_sizes()) {
        VulkanModuleOptions mo;
        mo.subgroup_size = sg;
        VulkanModule m = load_kernel(sk, mo);
        Rng rng(0x5C0DE4u);
        for (QuantShape s : kQ4KShapes) {
            std::vector<Q4KBlock> w = random_q4k(s.n, s.k, rng);
            std::vector<float> x = random_x(s.k, rng);
            char what[96];
            std::snprintf(what, sizeof(what), "gemv_q4_k n=%u k=%u grid=%u block=%u sg=%u", s.n, s.k, s.grid, s.block, sg);
            check_rows(what, run_quant(m, w, x, s.n, s.k, s.grid, s.block),
                       host_gemv<Q4KBlock, 256, host_dequant_q4k>(w, x, s.n, s.k, s.grid), s.grid);
        }
    }
}

TEST_CASE("SPIR-V kernels - GEMV Q8_0 / Q4_K agree with the CPU JIT GEMVs") {
    if (!vk_ready()) return;
    MlFusionCompiler c;
    Rng rng(0xC0DEC0DEu);
    {
        const uint32_t n = 6, k = 1024;
        std::vector<Q8Block> w = random_q8(n, k, rng);
        std::vector<float> x = random_x(k, rng);
        KernelFunction kfn = c.compile_gemv_q8_0();
        REQUIRE(kfn.is_valid());
        std::vector<float> cpu(n, 0.0f);
        kfn.as<MlFusionCompiler::GemvQ8_0Fn>()(w.data(), x.data(), cpu.data(), n, k);
        VulkanModule m = load_kernel(c.compile_spirv(GpuKernel::gemv_q8_0));
        check_rows("gemv_q8_0 n=6 k=1024 vs CPU JIT", run_quant(m, w, x, n, k, n, 256), cpu, n);
    }
    {
        const uint32_t n = 5, k = 2048;
        std::vector<Q4KBlock> w = random_q4k(n, k, rng);
        std::vector<float> x = random_x(k, rng);
        KernelFunction kfn = c.compile_gemv_q4_k();
        REQUIRE(kfn.is_valid());
        std::vector<float> cpu(n, 0.0f);
        kfn.as<MlFusionCompiler::GemvQ4_KFn>()(w.data(), x.data(), cpu.data(), n, k);
        VulkanModule m = load_kernel(c.compile_spirv(GpuKernel::gemv_q4_k));
        check_rows("gemv_q4_k n=5 k=2048 vs CPU JIT", run_quant(m, w, x, n, k, n, 256), cpu, n);
    }
}
