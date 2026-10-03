// The fused ML GPU kernels lowered through SpirvTarget
// (MlFusionCompiler::compile_spirv) and run on a Vulkan device, checked
// against double-precision host references over the shapes and with the
// tolerances test_gpu_kernels.cpp uses for the same kernels on CUDA (row
// lengths a multiple of 4 and not, element counts not a multiple of the
// block, rows above the grid, blocks 32..1024), and against the CPU KernelJit
// kernels where brass has one (SwiGLU, AdaLN, residual RMSNorm). The
// quantized GEMVs are in test_spirv_kernels_quant.cpp. [SKIP] without a
// Vulkan device.

#include "spirv_exec_support.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>
#include <map>

using namespace brass;
using namespace spvexec;
using codegen::GpuKernel;
using codegen::KernelFunction;
using codegen::MlFusionCompiler;

namespace {

// test_gpu_kernels.cpp's tolerances: exact float arithmetic (AdaLN, the
// in-place x += res) at 1e-5 / 1e-6; the SiLU and the statistics at 2e-3
// (Vulkan's exp2 / division / inverseSqrt are approximations, like PTX's
// .approx recipes); the GEMV dot products at 1e-4 and 5e-3 (SiLU of a dot).
constexpr float kExactTol = 1e-5f;
constexpr float kInPlaceTol = 1e-6f;
constexpr float kApproxTol = 2e-3f;
constexpr float kGemvTol = 1e-4f;
constexpr float kGemvSwigluTol = 5e-3f;

void check_reference(const char* what, const std::vector<float>& got, const std::vector<double>& ref, float tol) {
    REQUIRE(got.size() == ref.size());
    float worst = 0.0f;
    size_t first_bad = got.size();
    for (size_t i = 0; i < got.size(); ++i) {
        float r = static_cast<float>(ref[i]);
        float d = std::fabs(got[i] - r) / (1.0f + std::fabs(r));
        if (!(d <= tol) && first_bad == got.size()) first_bad = i;
        worst = std::max(worst, std::isnan(d) ? INFINITY : d);
    }
    std::printf("    %-46s max rel diff %.3g (tol %.0e)\n", what, static_cast<double>(worst), static_cast<double>(tol));
    if (first_bad < got.size())
        std::fprintf(stderr, "%s: element %zu device %.9g host %.9g\n", what, first_bad, got[first_bad], ref[first_bad]);
    CHECK(worst <= tol);
}

std::vector<float> pattern(size_t n, float scale, float bias, unsigned mod) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i % mod) * scale + bias;
    return v;
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }

void host_ln_stats(const float* row, uint32_t D, float eps, double& mean, double& rstd) {
    mean = 0;
    for (uint32_t i = 0; i < D; ++i) mean += row[i];
    mean /= D;
    double var = 0;
    for (uint32_t i = 0; i < D; ++i) { double d = row[i] - mean; var += d * d; }
    var /= D;
    rstd = 1.0 / std::sqrt(var + eps);
}

// One VulkanModule per kernel id, compiled (+ spirv-val) and loaded once.
const VulkanModule& module(GpuKernel k) {
    static std::map<GpuKernel, VulkanModule> cache;
    auto it = cache.find(k);
    if (it == cache.end()) {
        MlFusionCompiler c;
        Module mod("k");
        target::SpirvKernel sk = spvtest::compile_checked(*c.build_gpu_kernel(mod, k));
        it = cache.emplace(k, load_kernel(sk)).first;
    }
    return it->second;
}

} // namespace

TEST_CASE("SPIR-V kernels - all eleven compile, validate and match the PTX contract") {
    MlFusionCompiler c;
    for (GpuKernel k : codegen::kAllGpuKernels) {
        target::SpirvKernel sk = c.compile_spirv(k);
        CHECK(sk.entry == codegen::gpu_kernel_entry(k));
        Module mod("m");
        Function* fn = c.build_gpu_kernel(mod, k);
        CHECK(sk.entry == std::string(fn->name()));
        CHECK_EQ(sk.params.size(), fn->param_types().size());
        CHECK(sk.push_constant_bytes <= 128u);
        CHECK(sk.local_size_spec_constants);
        CHECK(spvtest::spirv_val(sk.words));
        std::printf("    %-36s push %2u B, shared %3u B, caps:", sk.entry.c_str(), sk.push_constant_bytes, sk.shared_bytes);
        for (const std::string& cap : sk.capabilities) std::printf(" %s", cap.c_str());
        std::printf("\n");
    }
}

TEST_CASE("SPIR-V kernels - SwiGLU and packed SwiGLU match the host reference across shapes") {
    if (!vk_ready()) return;
    struct Shape { uint32_t n, grid, block; };
    for (Shape s : {Shape{64, 1, 64}, Shape{101, 1, 64}, Shape{1000, 3, 128}, Shape{4099, 4, 256}, Shape{7, 2, 32},
                    Shape{2050, 1, 1024}}) {
        std::vector<float> g(s.n), u(s.n);
        for (uint32_t i = 0; i < s.n; ++i) {
            g[i] = (static_cast<float>(i % 211) - 105.0f) * 0.08f;
            u[i] = static_cast<float>(i % 8) * 0.5f - 1.0f;
        }
        VulkanBuffer dg = upload(g), du = upload(u), dout = zeros<float>(s.n);
        launch(module(GpuKernel::swiglu), s.grid, s.block, {dg, du, dout, s.n});
        std::vector<double> ref(s.n);
        for (uint32_t i = 0; i < s.n; ++i) ref[i] = silu(g[i]) * u[i];
        char what[96];
        std::snprintf(what, sizeof(what), "swiglu n=%u grid=%u block=%u", s.n, s.grid, s.block);
        check_reference(what, download<float>(dout, s.n), ref, kApproxTol);
    }
    struct PShape { uint32_t b, d, grid, block; };
    for (PShape s : {PShape{1, 64, 1, 64}, PShape{4, 100, 2, 128}, PShape{3, 101, 1, 64}, PShape{8, 1152, 4, 256},
                     PShape{2, 7, 2, 32}, PShape{1, 4096, 8, 1024}, PShape{5, 12, 1, 1024}}) {
        std::vector<float> x(static_cast<size_t>(s.b) * 2 * s.d);
        std::vector<double> ref(static_cast<size_t>(s.b) * s.d);
        for (uint32_t r = 0; r < s.b; ++r) {
            float* row = &x[static_cast<size_t>(r) * 2 * s.d];
            for (uint32_t col = 0; col < s.d; ++col) {
                uint32_t e = r * s.d + col;
                row[col] = (static_cast<float>(e % 211) - 105.0f) * 0.08f;
                row[s.d + col] = static_cast<float>(e % 8) * 0.5f - 1.0f;
            }
            for (uint32_t col = 0; col < s.d; ++col) ref[static_cast<size_t>(r) * s.d + col] = silu(row[col]) * row[s.d + col];
        }
        VulkanBuffer dx = upload(x), dy = zeros<float>(s.b * s.d);
        launch(module(GpuKernel::swiglu_packed), s.grid, s.block, {dx, dy, s.b, s.d});
        char what[96];
        std::snprintf(what, sizeof(what), "swiglu_packed b=%u d=%u grid=%u block=%u", s.b, s.d, s.grid, s.block);
        check_reference(what, download<float>(dy, s.b * s.d), ref, kApproxTol);
    }
}

TEST_CASE("SPIR-V kernels - AdaLN modulate matches the host reference (gated and ungated)") {
    if (!vk_ready()) return;
    for (bool gated : {false, true}) {
        struct Shape { uint32_t L, D, block; };
        for (Shape s : {Shape{2, 35, 128}, Shape{3, 64, 128}, Shape{5, 130, 128}, Shape{4, 8, 64}, Shape{2, 1024, 256},
                        Shape{3, 33, 32}, Shape{2, 2052, 1024}}) {
            std::vector<float> x(s.L * s.D);
            for (uint32_t r = 0; r < s.L; ++r)
                for (uint32_t i = 0; i < s.D; ++i) x[r * s.D + i] = static_cast<float>((r + i) % 11) * 0.1f - 0.3f;
            std::vector<float> scale = pattern(s.D, 0.05f, 0.0f, 4), shift = pattern(s.D, -0.1f, 0.0f, 3);
            std::vector<float> gate = pattern(s.D, 0.1f, 0.5f, 5);
            VulkanBuffer dx = upload(x), ds = upload(scale), dh = upload(shift), dgt = upload(gate);
            VulkanBuffer dy = zeros<float>(s.L * s.D);
            if (gated) launch(module(GpuKernel::adaln_modulate_gated), s.L, s.block, {dx, ds, dh, dgt, dy, s.L, s.D});
            else launch(module(GpuKernel::adaln_modulate), s.L, s.block, {dx, ds, dh, dy, s.L, s.D});
            std::vector<double> ref(s.L * s.D);
            for (uint32_t r = 0; r < s.L; ++r) {
                for (uint32_t i = 0; i < s.D; ++i) {
                    double v = static_cast<double>(x[r * s.D + i]) * (1.0 + scale[i]) + shift[i];
                    ref[r * s.D + i] = gated ? v * gate[i] : v;
                }
            }
            char what[96];
            std::snprintf(what, sizeof(what), "adaln%s L=%u D=%u block=%u", gated ? "_gated" : "", s.L, s.D, s.block);
            check_reference(what, download<float>(dy, s.L * s.D), ref, kExactTol);
        }
    }
}

TEST_CASE("SPIR-V kernels - residual RMSNorm matches the host reference (y and in-place x)") {
    if (!vk_ready()) return;
    float eps = 1e-5f;
    struct Shape { uint32_t B, D, block; };
    for (Shape s : {Shape{3, 37, 128}, Shape{2, 64, 128}, Shape{4, 300, 256}, Shape{1, 1024, 128}, Shape{2, 50, 32},
                    Shape{3, 96, 1024}, Shape{2, 4096, 512}}) {
        std::vector<float> x(s.B * s.D), res(s.B * s.D);
        for (uint32_t r = 0; r < s.B; ++r) {
            for (uint32_t i = 0; i < s.D; ++i) {
                x[r * s.D + i] = static_cast<float>((r + i) % 7) * 0.25f - 0.5f;
                res[r * s.D + i] = static_cast<float>((r * 3 + i) % 5) * 0.3f;
            }
        }
        std::vector<float> gamma = pattern(s.D, 0.1f, 1.0f, 3);
        VulkanBuffer dx = upload(x), dres = upload(res), dg = upload(gamma), dy = zeros<float>(s.B * s.D);
        launch(module(GpuKernel::residual_rms_norm), s.B, s.block, {dx, dres, dg, dy, s.B, s.D, eps});
        std::vector<double> ref_y(s.B * s.D), ref_x(s.B * s.D);
        for (uint32_t r = 0; r < s.B; ++r) {
            double sum = 0;
            for (uint32_t i = 0; i < s.D; ++i) {
                double v = static_cast<double>(x[r * s.D + i]) + res[r * s.D + i];
                ref_x[r * s.D + i] = v;
                sum += v * v;
            }
            double rrms = 1.0 / std::sqrt(sum / s.D + eps);
            for (uint32_t i = 0; i < s.D; ++i) ref_y[r * s.D + i] = ref_x[r * s.D + i] * gamma[i] * rrms;
        }
        char what[96];
        std::snprintf(what, sizeof(what), "rms B=%u D=%u block=%u", s.B, s.D, s.block);
        check_reference(what, download<float>(dy, s.B * s.D), ref_y, kApproxTol);
        std::snprintf(what, sizeof(what), "rms B=%u D=%u block=%u (in-place x)", s.B, s.D, s.block);
        check_reference(what, download<float>(dx, s.B * s.D), ref_x, kInPlaceTol);
    }
}

namespace {
struct NormShape { uint32_t rows, D, block; };
const NormShape kNormShapes[] = {
    {1, 8, 32}, {4, 8, 512}, {3, 37, 128}, {2, 37, 1024}, {4, 300, 256}, {1, 300, 64}, {2, 1024, 1024}, {3, 1024, 128},
    {2, 4096, 256},
};
} // namespace

TEST_CASE("SPIR-V kernels - LayerNorm modulate and residual LayerNorm match the host reference") {
    if (!vk_ready()) return;
    float eps = 1e-5f;
    for (NormShape s : kNormShapes) {
        std::vector<float> x(s.rows * s.D), res(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r)
            for (uint32_t i = 0; i < s.D; ++i) {
                x[r * s.D + i] = static_cast<float>((r * 5 + i) % 9) * 0.2f - 0.8f;
                res[r * s.D + i] = static_cast<float>((r * 2 + i) % 5) * 0.1f;
            }
        std::vector<float> gamma = pattern(s.D, 0.1f, 1.0f, 3), beta = pattern(s.D, 0.05f, 0.0f, 2);
        std::vector<float> scale = pattern(s.D, 0.02f, 0.0f, 4), shift = pattern(s.D, -0.03f, 0.0f, 3);
        // LayerNorm + modulate
        VulkanBuffer dx = upload(x), dg = upload(gamma), db = upload(beta), dsc = upload(scale), dsh = upload(shift);
        VulkanBuffer dy = zeros<float>(s.rows * s.D);
        launch(module(GpuKernel::layernorm_modulate), s.rows, s.block, {dx, dg, db, dsc, dsh, dy, s.rows, s.D, eps});
        std::vector<double> ref(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r) {
            double mean, rstd;
            host_ln_stats(&x[r * s.D], s.D, eps, mean, rstd);
            for (uint32_t i = 0; i < s.D; ++i)
                ref[r * s.D + i] = ((x[r * s.D + i] - mean) * rstd * gamma[i] + beta[i]) * (1.0 + scale[i]) + shift[i];
        }
        char what[96];
        std::snprintf(what, sizeof(what), "lnmod R=%u D=%u block=%u", s.rows, s.D, s.block);
        check_reference(what, download<float>(dy, s.rows * s.D), ref, kApproxTol);

        // residual LayerNorm: x += res in place, then LN of the float sum
        VulkanBuffer dx2 = upload(x), dres = upload(res), dy2 = zeros<float>(s.rows * s.D);
        launch(module(GpuKernel::residual_layernorm), s.rows, s.block, {dx2, dres, dg, db, dy2, s.rows, s.D, eps});
        std::vector<double> ref_y(s.rows * s.D), ref_x(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r) {
            std::vector<float> v(s.D);
            for (uint32_t i = 0; i < s.D; ++i) ref_x[r * s.D + i] = v[i] = x[r * s.D + i] + res[r * s.D + i];
            double mean, rstd;
            host_ln_stats(v.data(), s.D, eps, mean, rstd);
            for (uint32_t i = 0; i < s.D; ++i) ref_y[r * s.D + i] = (v[i] - mean) * rstd * gamma[i] + beta[i];
        }
        std::snprintf(what, sizeof(what), "res_ln B=%u D=%u block=%u", s.rows, s.D, s.block);
        check_reference(what, download<float>(dy2, s.rows * s.D), ref_y, kApproxTol);
        std::snprintf(what, sizeof(what), "res_ln B=%u D=%u block=%u (in-place x)", s.rows, s.D, s.block);
        check_reference(what, download<float>(dx2, s.rows * s.D), ref_x, kInPlaceTol);
    }
}

TEST_CASE("SPIR-V kernels - GEMV SwiGLU and GEMV residual match the host reference") {
    if (!vk_ready()) return;
    struct Shape { uint32_t n, k, grid, block; };
    for (Shape s : {Shape{1, 8, 1, 64}, Shape{4, 64, 4, 128}, Shape{3, 37, 3, 32}, Shape{5, 1027, 5, 256},
                    Shape{2, 1024, 2, 1024}, Shape{7, 100, 4, 128}, Shape{6, 2048, 6, 512}, Shape{9, 4096, 7, 256}}) {
        std::vector<float> wg(s.n * s.k), wu(s.n * s.k), wd(s.n * s.k), x(s.k), res(s.n);
        for (uint32_t i = 0; i < s.k; ++i) x[i] = static_cast<float>(i % 5) * 0.25f - 0.5f;
        for (uint32_t r = 0; r < s.n; ++r) {
            for (uint32_t i = 0; i < s.k; ++i) {
                wg[r * s.k + i] = static_cast<float>((r + i) % 9) * 0.1f - 0.4f;
                wu[r * s.k + i] = static_cast<float>((r * 3 + i) % 7) * 0.15f - 0.5f;
                wd[r * s.k + i] = static_cast<float>((r * 2 + i) % 6) * 0.2f - 0.6f;
            }
            res[r] = 0.1f * static_cast<float>(r + 1);
        }
        auto dot = [&](const std::vector<float>& w, uint32_t r) {
            double d = 0;
            for (uint32_t i = 0; i < s.k; ++i) d += static_cast<double>(w[r * s.k + i]) * x[i];
            return d;
        };
        VulkanBuffer dwg = upload(wg), dwu = upload(wu), dwd = upload(wd), dx = upload(x), dres = upload(res);
        VulkanBuffer dy = zeros<float>(s.n), dy2 = zeros<float>(s.n);
        launch(module(GpuKernel::gemv_swiglu), s.grid, s.block, {dwg, dwu, dx, dy, s.n, s.k});
        launch(module(GpuKernel::gemv_residual), s.grid, s.block, {dwd, dx, dres, dy2, s.n, s.k});
        std::vector<double> ref(s.n, 0.0), ref2(s.n, 0.0);   // rows beyond the grid stay 0
        for (uint32_t r = 0; r < std::min(s.n, s.grid); ++r) {
            ref[r] = silu(dot(wg, r)) * dot(wu, r);
            ref2[r] = dot(wd, r) + res[r];
        }
        char what[96];
        std::snprintf(what, sizeof(what), "gemv_swiglu n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, download<float>(dy, s.n), ref, kGemvSwigluTol);
        std::snprintf(what, sizeof(what), "gemv_res n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, download<float>(dy2, s.n), ref2, kGemvTol);
    }
}

TEST_CASE("SPIR-V kernels - SwiGLU, AdaLN and residual RMSNorm agree with the CPU KernelJit kernels") {
    if (!vk_ready()) return;
    MlFusionCompiler c;
    const uint32_t n = 4096 + 3;
    std::vector<float> g(n), u(n);
    for (uint32_t i = 0; i < n; ++i) {
        g[i] = (static_cast<float>(i % 97) - 48.0f) * 0.11f;
        u[i] = static_cast<float>(i % 13) * 0.2f - 1.1f;
    }
    {
        KernelFunction kf = c.compile_swiglu();
        REQUIRE(kf.is_valid());
        std::vector<float> cpu(n);
        kf.as<codegen::FusedSwiGLUFn>()(g.data(), u.data(), cpu.data(), n);
        VulkanBuffer dg = upload(g), du = upload(u), dout = zeros<float>(n);
        launch(module(GpuKernel::swiglu), 8, 256, {dg, du, dout, n});
        check_reference("swiglu n=4099 vs CPU JIT", download<float>(dout, n), std::vector<double>(cpu.begin(), cpu.end()),
                        kApproxTol);
    }
    for (bool gated : {false, true}) {
        KernelFunction kf = c.compile_adaln_modulate(gated);
        REQUIRE(kf.is_valid());
        std::vector<float> sc = pattern(n, 0.03f, -0.1f, 7), sh = pattern(n, 0.02f, 0.05f, 5), gt = pattern(n, 0.1f, 0.4f, 6);
        std::vector<float> cpu(n);
        if (gated) kf.as<codegen::FusedAdaLNModulateGatedFn>()(g.data(), sc.data(), sh.data(), gt.data(), cpu.data(), n);
        else kf.as<codegen::FusedAdaLNModulateFn>()(g.data(), sc.data(), sh.data(), cpu.data(), n);
        VulkanBuffer dx = upload(g), ds = upload(sc), dh = upload(sh), dgt = upload(gt), dy = zeros<float>(n);
        uint32_t one = 1;
        if (gated) launch(module(GpuKernel::adaln_modulate_gated), 1, 256, {dx, ds, dh, dgt, dy, one, n});
        else launch(module(GpuKernel::adaln_modulate), 1, 256, {dx, ds, dh, dy, one, n});
        check_reference(gated ? "adaln_gated n=4099 vs CPU JIT" : "adaln n=4099 vs CPU JIT", download<float>(dy, n),
                        std::vector<double>(cpu.begin(), cpu.end()), kExactTol);
    }
    {
        KernelFunction kf = c.compile_residual_rms_norm();
        REQUIRE(kf.is_valid());
        std::vector<float> res = u, w = pattern(n, 0.05f, 0.9f, 4), cpu(n);
        codegen::run_residual_rms_norm(kf.as<codegen::FusedResidualRmsNormFn>(), g.data(), res.data(), w.data(), cpu.data(), n);
        VulkanBuffer dx = upload(g), dres = upload(u), dw = upload(w), dy = zeros<float>(n);
        uint32_t one = 1;
        launch(module(GpuKernel::residual_rms_norm), 1, 512, {dx, dres, dw, dy, one, n, 1e-5f});
        check_reference("rms n=4099 vs CPU JIT", download<float>(dy, n), std::vector<double>(cpu.begin(), cpu.end()),
                        kApproxTol);
        check_reference("rms n=4099 in-place vs CPU JIT residual", download<float>(dx, n),
                        std::vector<double>(res.begin(), res.end()), kInPlaceTol);
    }
}
