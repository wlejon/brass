// The quantized GEMV GPU kernels built as MIR (build_ptx_gemv_q8_0,
// build_ptx_gemv_q4_k) checked on device against host references that
// dequantize exactly as the CPU dequantizers (brass_dequant_q8_0_block /
// brass_dequant_q4k_block) do, with double accumulation, and against the CPU
// JIT GEMVs (compile_gemv_q8_0 / compile_gemv_q4_k). Random quantized weights
// (fixed seed, f16 scales in a sane range, every 6-bit scale field and both
// nibble packings exercised) and random x over several n / k / block sizes,
// including n larger than the grid (those rows stay 0) and blocks 32/64 up
// to 1024. Required 1e-4 relative; the observed maximum is printed per shape.
// The kernels must also pass ptx::verify and ptxas and keep their recipe.
// Visible [SKIP] lines without ptxas / CUDA.

#include "ptx_test_support.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace brass;
using brass::codegen::KernelFunction;
using brass::codegen::MlFusionCompiler;
using namespace ptxtest;

namespace {

constexpr float kReferenceTol = 1e-4f;   // GPU vs host / CPU JIT reference

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]) / (1.0f + std::fabs(b[i]));
        worst = std::max(worst, d);
    }
    return worst;
}

// Lower + verify + ptxas for a MIR kernel; returns the PTX text.
std::string checked_mir_ptx(Function* fn) {
    lower_ok(*fn);
    return emit_checked(*fn);
}

// ---------------------------------------------------------------------------
// Block formats (as the CPU dequantizers read them) and random generators
// ---------------------------------------------------------------------------

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
        for (int8_t& q : blk.qs) q = static_cast<int8_t>(static_cast<int>(rng.byte()) - 128); // -128..127
    }
    return w;
}

std::vector<Q4KBlock> random_q4k(uint32_t n, uint32_t k, Rng& rng) {
    std::vector<Q4KBlock> w(static_cast<size_t>(n) * (k / 256));
    for (Q4KBlock& blk : w) {
        blk.d = f32_to_f16_host(rng.uniform(0.002f, 0.02f));
        blk.dmin = f32_to_f16_host(rng.uniform(0.001f, 0.01f));
        for (uint8_t& s : blk.scales) s = rng.byte();   // every 6-bit field and both packings exercised
        for (uint8_t& q : blk.qs) q = rng.byte();
    }
    return w;
}

std::vector<float> random_x(uint32_t k, Rng& rng) {
    std::vector<float> x(k);
    for (float& v : x) v = rng.uniform(-1.0f, 1.0f);
    return x;
}

// Host dequantization, transcribed from brass_dequant_q8_0_block / brass_dequant_q4k_block.
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

// ---------------------------------------------------------------------------
// Device runner: y is zero-initialised so rows beyond the grid stay 0
// ---------------------------------------------------------------------------

template <typename Block>
std::vector<float> run_quant(const std::string& ptx, const char* entry, const std::vector<Block>& w,
                             const std::vector<float>& x, uint32_t n, uint32_t k, uint32_t grid, uint32_t block) {
    CudaBuffer dw = upload(w), dx = upload(x);
    CudaBuffer dy = CudaBuffer::alloc(n * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* pw = dw.device_ptr(); void* px = dx.device_ptr(); void* py = dy.device_ptr();
    uint32_t nn = n, kk = k;
    launch(ptx, entry, grid, block, {&pw, &px, &py, &nn, &kk});
    return download<float>(dy, n);
}

void check_reference(const char* what, const std::vector<float>& got, const std::vector<float>& ref, uint32_t grid) {
    float d = max_rel_diff(got, ref);
    std::printf("    %-44s max rel diff vs reference %.3g\n", what, static_cast<double>(d));
    CHECK(d <= kReferenceTol);
    for (size_t r = 0; r < got.size(); ++r) {
        if (r >= grid) { CHECK(got[r] == 0.0f); continue; }
        if (!near(got[r], ref[r], kReferenceTol)) {
            std::fprintf(stderr, "%s: row %zu gpu %.9g host %.9g\n", what, r, got[r], ref[r]);
            break;
        }
    }
}

struct QuantShape { uint32_t n, k, grid, block; };

// n in {1, 3, 17, 64} (17 rows on an 8-block grid: rows beyond the grid stay 0),
// k over the multiples the kernels require -- including counts that are not a
// multiple of the unrolled stride, so the remainder loop runs -- and blocks
// 32/64 .. 1024 (256 is what the runtime launches).
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

} // namespace

// ---------------------------------------------------------------------------
// Static checks: the kernels verify, assemble, and keep the recipe
// ---------------------------------------------------------------------------

TEST_CASE("GPU kernels - quantized GEMV kernels verify, assemble and keep the recipe") {
    MlFusionCompiler c;
    Module m8("k_gemv_q8_0"), m4("k_gemv_q4_k");
    std::string q8 = checked_mir_ptx(c.build_ptx_gemv_q8_0(m8));
    std::string q4 = checked_mir_ptx(c.build_ptx_gemv_q4_k(m4));

    auto has = [](const std::string& ptx, const char* needle) {
        bool ok = ptx.find(needle) != std::string::npos;
        if (!ok) std::cerr << "missing " << needle << "\n" << ptx;
        CHECK(ok);
    };
    auto count = [](const std::string& ptx, const char* needle) {
        size_t n = 0;
        for (size_t p = ptx.find(needle); p != std::string::npos; p = ptx.find(needle, p + 1)) ++n;
        return n;
    };
    for (const std::string* p : {&q8, &q4}) {
        has(*p, ".param .u64"); has(*p, ".param .u32");
        has(*p, "cvt.f32.f16"); has(*p, "ld.global.v4.f32"); has(*p, "fma.rn.f32");
        has(*p, "shfl.sync.down.b32"); has(*p, "bar.sync 0;"); has(*p, ".shared .align 16 .f32 smem_0[32]");
        has(*p, "st.global.f32"); has(*p, "%ntid.x"); has(*p, "%ctaid.x");
        CHECK(p->find("st.global.v4") == std::string::npos);   // one scalar output per block
        CHECK(p->find("call ") == std::string::npos);          // no CPU dequantizer calls
        CHECK(p->find("div.") == std::string::npos);
    }
    // Q8_0: f16 header, two u16 loads for the 4 int8 (2-byte aligned blocks), sign
    // extension by shl/shr.s32, cvt.rn.f32.s32 * d then fma. The block loop is
    // unrolled kQ8Unroll times plus a remainder copy of the body.
    has(q8, ".entry fused_gemv_q8_0_kernel(");
    has(q8, "ld.global.u16"); has(q8, "shr.s32"); has(q8, "cvt.rn.f32.s32"); has(q8, "mul.f32");
    const size_t q8_bodies = MlFusionCompiler::kQ8Unroll + 1;
    CHECK(count(q8, "ld.global.u16") == 3 * q8_bodies);
    CHECK(count(q8, "cvt.rn.f32.s32") == 4 * q8_bodies);
    CHECK(count(q8, "ld.global.v4.f32") == q8_bodies);
    CHECK(q8.find("ld.global.s8") == std::string::npos);
    CHECK(q8.find("ld.global.u8") == std::string::npos);
    // Q4_K: v4.u32 header, u32 nibble word, 6-bit scale/min select, wscale * nib - wmin.
    has(q4, ".entry fused_gemv_q4_k_kernel(");
    has(q4, "ld.global.v4.u32"); has(q4, "cvt.rn.f32.u32"); has(q4, "selp.b32"); has(q4, "neg.f32");
    const size_t q4_bodies = MlFusionCompiler::kQ4KUnroll + 1;
    CHECK(count(q4, "cvt.f32.f16") == 2 * q4_bodies);
    CHECK(count(q4, "cvt.rn.f32.u32") == 6 * q4_bodies);   // sc, m and the four nibbles
    CHECK(count(q4, "fma.rn.f32") == 8 * q4_bodies);       // four dequant + four dot fmas
    CHECK(count(q4, "ld.global.v4.u32") == q4_bodies);
    CHECK(q4.find("ld.global.u8") == std::string::npos);

    // The public emitters return exactly these kernels.
    CHECK(c.emit_ptx_fused_gemv_q8_0() == q8);
    CHECK(c.emit_ptx_fused_gemv_q4_k() == q4);
}

// ---------------------------------------------------------------------------
// On-device runs against the host reference
// ---------------------------------------------------------------------------

TEST_CASE("GPU kernels - GEMV Q8_0 matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_gemv_q8_0();
    Rng rng(0x5C0DE8u);

    for (QuantShape s : kQ8Shapes) {
        std::vector<Q8Block> w = random_q8(s.n, s.k, rng);
        std::vector<float> x = random_x(s.k, rng);
        std::vector<float> got = run_quant(ptx, "fused_gemv_q8_0_kernel", w, x, s.n, s.k, s.grid, s.block);
        std::vector<float> ref = host_gemv<Q8Block, 32, host_dequant_q8>(w, x, s.n, s.k, s.grid);
        char what[96];
        std::snprintf(what, sizeof(what), "gemv_q8_0 n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, got, ref, s.grid);
    }
}

TEST_CASE("GPU kernels - GEMV Q4_K matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_gemv_q4_k();
    Rng rng(0x5C0DE4u);

    for (QuantShape s : kQ4KShapes) {
        std::vector<Q4KBlock> w = random_q4k(s.n, s.k, rng);
        std::vector<float> x = random_x(s.k, rng);
        std::vector<float> got = run_quant(ptx, "fused_gemv_q4_k_kernel", w, x, s.n, s.k, s.grid, s.block);
        std::vector<float> ref = host_gemv<Q4KBlock, 256, host_dequant_q4k>(w, x, s.n, s.k, s.grid);
        char what[96];
        std::snprintf(what, sizeof(what), "gemv_q4_k n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, got, ref, s.grid);
    }
}

// ---------------------------------------------------------------------------
// Cross check against the CPU JIT GEMVs (the reference block layout)
// ---------------------------------------------------------------------------

TEST_CASE("GPU kernels - GEMV Q8_0 / Q4_K agree with the CPU JIT GEMVs") {
    if (!gpu_ready()) return;
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
        std::vector<float> gpu = run_quant(c.emit_ptx_fused_gemv_q8_0(), "fused_gemv_q8_0_kernel", w, x, n, k, n, 256);
        check_reference("gemv_q8_0 n=6 k=1024 vs CPU JIT", gpu, cpu, n);
    }
    {
        const uint32_t n = 5, k = 2048;
        std::vector<Q4KBlock> w = random_q4k(n, k, rng);
        std::vector<float> x = random_x(k, rng);
        KernelFunction kfn = c.compile_gemv_q4_k();
        REQUIRE(kfn.is_valid());
        std::vector<float> cpu(n, 0.0f);
        kfn.as<MlFusionCompiler::GemvQ4_KFn>()(w.data(), x.data(), cpu.data(), n, k);
        std::vector<float> gpu = run_quant(c.emit_ptx_fused_gemv_q4_k(), "fused_gemv_q4_k_kernel", w, x, n, k, n, 256);
        check_reference("gemv_q4_k n=5 k=2048 vs CPU JIT", gpu, cpu, n);
    }
}
