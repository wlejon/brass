// Stage 5a kernel migration: the MIR-built GPU kernels (build_ptx_swiglu,
// build_ptx_adaln_modulate, build_ptx_residual_rms_norm) against the
// hand-written PTX they replaced (ml_fusion_ptx_legacy.cpp). Both sides run
// on device with identical inputs over several shapes (row length a multiple
// of 4 and not, element counts that are not a multiple of the block size,
// several rows/blocks, several block sizes) and must agree to a tight
// tolerance: the kernels use the same approx instructions in the same order,
// so the expected difference is 0 ulp. The new kernels must also pass
// ptx::verify and ptxas. Visible [SKIP] lines without ptxas / CUDA.

#include "ptx_test_support.hpp"
#include "../../src/codegen/ml_fusion_ptx_legacy.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>
#include <cstdio>

using namespace brass;
using brass::codegen::MlFusionCompiler;
using namespace ptxtest;

namespace {

// Tolerance for old-vs-new agreement (relative, see `near`). Observed: 0.
constexpr float kMigrationTol = 1e-6f;

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]) / (1.0f + std::fabs(b[i]));
        worst = std::max(worst, d);
    }
    return worst;
}

void check_agree(const char* what, const std::vector<float>& legacy, const std::vector<float>& mir) {
    float d = max_rel_diff(legacy, mir);
    std::printf("    %-44s max rel diff %.3g\n", what, static_cast<double>(d));
    CHECK(d <= kMigrationTol);
    for (size_t i = 0; i < legacy.size(); ++i) {
        if (!near(mir[i], legacy[i], kMigrationTol)) {
            std::fprintf(stderr, "%s: element %zu legacy %.9g mir %.9g\n", what, i, legacy[i], mir[i]);
            break;
        }
    }
}

// Lower + verify + ptxas for a MIR kernel; returns the PTX text.
std::string checked_mir_ptx(Function* fn) {
    lower_ok(*fn);
    return emit_checked(*fn);
}

std::vector<float> pattern(size_t n, float scale, float bias, unsigned mod) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i % mod) * scale + bias;
    return v;
}

// ---------------------------------------------------------------------------
// SwiGLU: out = silu(gate) * up, grid-stride
// ---------------------------------------------------------------------------

std::vector<float> run_swiglu(const std::string& ptx, const std::vector<float>& g, const std::vector<float>& u,
                              uint32_t grid, uint32_t block) {
    uint32_t n = static_cast<uint32_t>(g.size());
    CudaBuffer dg = upload(g), du = upload(u);
    CudaBuffer dout = CudaBuffer::alloc(n * 4);
    REQUIRE(dout.valid() && dout.zero());
    void* pg = dg.device_ptr(); void* pu = du.device_ptr(); void* po = dout.device_ptr();
    launch(ptx, "fused_swiglu_kernel", grid, block, {&pg, &pu, &po, &n});
    return download<float>(dout, n);
}

// ---------------------------------------------------------------------------
// AdaLN modulate: y[row] = x[row] * (1 + scale) + shift [* gate], block per row
// ---------------------------------------------------------------------------

std::vector<float> run_adaln(const std::string& ptx, bool gated, uint32_t L, uint32_t D, uint32_t block,
                             const std::vector<float>& x, const std::vector<float>& scale,
                             const std::vector<float>& shift, const std::vector<float>& gate) {
    CudaBuffer dx = upload(x), ds = upload(scale), dh = upload(shift), dgt = upload(gate);
    CudaBuffer dy = CudaBuffer::alloc(L * D * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* px = dx.device_ptr(); void* ps = ds.device_ptr(); void* ph = dh.device_ptr();
    void* pg = dgt.device_ptr(); void* py = dy.device_ptr();
    uint32_t l = L, d = D;
    if (gated) launch(ptx, "fused_adaln_modulate_gated_kernel", L, block, {&px, &ps, &ph, &pg, &py, &l, &d});
    else       launch(ptx, "fused_adaln_modulate_kernel", L, block, {&px, &ps, &ph, &py, &l, &d});
    return download<float>(dy, L * D);
}

// ---------------------------------------------------------------------------
// Residual RMSNorm: x += res (in place), y = x * gamma * rrms, block per row.
// Returns y followed by the updated x.
// ---------------------------------------------------------------------------

std::vector<float> run_rms(const std::string& ptx, uint32_t B, uint32_t D, uint32_t block, float eps,
                           const std::vector<float>& x, const std::vector<float>& res, const std::vector<float>& gamma) {
    CudaBuffer dx = upload(x), dres = upload(res), dg = upload(gamma);
    CudaBuffer dy = CudaBuffer::alloc(B * D * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* px = dx.device_ptr(); void* pres = dres.device_ptr(); void* pg = dg.device_ptr(); void* py = dy.device_ptr();
    uint32_t b = B, d = D;
    float feps = eps;
    launch(ptx, "fused_residual_rms_norm_kernel", B, block, {&px, &pres, &pg, &py, &b, &d, &feps});
    std::vector<float> out = download<float>(dy, B * D);
    std::vector<float> xs = download<float>(dx, B * D);
    out.insert(out.end(), xs.begin(), xs.end());
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Static checks: the MIR kernels verify, assemble, and keep the fast recipe
// ---------------------------------------------------------------------------

TEST_CASE("GPU migration - MIR kernels verify, assemble and keep the approx recipe") {
    MlFusionCompiler c;
    Module ms("mig_swiglu"), ma("mig_adaln"), mg("mig_adaln_g"), mr("mig_rms");
    std::string swiglu = checked_mir_ptx(c.build_ptx_swiglu(ms));
    std::string adaln = checked_mir_ptx(c.build_ptx_adaln_modulate(ma, false));
    std::string adaln_g = checked_mir_ptx(c.build_ptx_adaln_modulate(mg, true));
    std::string rms = checked_mir_ptx(c.build_ptx_residual_rms_norm(mr));

    auto has = [](const std::string& ptx, const char* needle) {
        bool ok = ptx.find(needle) != std::string::npos;
        if (!ok) std::cerr << "missing " << needle << "\n" << ptx;
        CHECK(ok);
    };
    has(swiglu, ".entry fused_swiglu_kernel(");
    has(swiglu, "ex2.approx.f32"); has(swiglu, "rcp.approx.f32"); has(swiglu, "0f3FB8AA3B");
    has(swiglu, "ld.global.v4.f32"); has(swiglu, "st.global.v4.f32"); has(swiglu, "%nctaid.x");
    has(adaln, ".entry fused_adaln_modulate_kernel(");
    has(adaln, "fma.rn.f32"); has(adaln, "ld.global.v4.f32"); has(adaln, "st.global.v4.f32");
    has(adaln_g, ".entry fused_adaln_modulate_gated_kernel(");
    has(rms, ".entry fused_residual_rms_norm_kernel(");
    has(rms, "rsqrt.approx.f32"); has(rms, "div.approx.f32"); has(rms, "shfl.sync.down.b32");
    has(rms, "bar.sync 0;"); has(rms, ".shared .align 16 .f32 smem_0[32]");
    CHECK(rms.find("call ") == std::string::npos); // no CPU helper calls remain

    // The public emitters now return exactly these kernels.
    CHECK(c.emit_ptx_swiglu() == swiglu);
    CHECK(c.emit_ptx_adaln_modulate(false) == adaln);
    CHECK(c.emit_ptx_adaln_modulate(true) == adaln_g);
    CHECK(c.emit_ptx_fused_residual_rms_norm() == rms);
    CHECK(c.emit_ptx_residual_rms_norm() == rms);

    // Legacy strings still assemble too (they are the reference side below).
    if (ptxas_available()) {
        CHECK(ptxas_assembles(codegen::legacy::legacy_ptx_swiglu()));
        CHECK(ptxas_assembles(codegen::legacy::legacy_ptx_adaln_modulate(false)));
        CHECK(ptxas_assembles(codegen::legacy::legacy_ptx_adaln_modulate(true)));
        CHECK(ptxas_assembles(codegen::legacy::legacy_ptx_residual_rms_norm()));
    }
}

// ---------------------------------------------------------------------------
// Differential: legacy string kernel vs MIR kernel on device
// ---------------------------------------------------------------------------

TEST_CASE("GPU migration - SwiGLU legacy vs MIR agree across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string mir = c.emit_ptx_swiglu();
    std::string legacy = codegen::legacy::legacy_ptx_swiglu();

    struct Shape { uint32_t n, grid, block; };
    for (Shape s : {Shape{64, 1, 64}, Shape{101, 1, 64}, Shape{1000, 3, 128}, Shape{4099, 4, 256}, Shape{7, 2, 32}}) {
        std::vector<float> g(s.n), u(s.n);
        for (uint32_t i = 0; i < s.n; ++i) {
            g[i] = (static_cast<float>(i % 211) - 105.0f) * 0.08f;
            u[i] = static_cast<float>(i % 8) * 0.5f - 1.0f;
        }
        std::vector<float> a = run_swiglu(legacy, g, u, s.grid, s.block);
        std::vector<float> b = run_swiglu(mir, g, u, s.grid, s.block);
        char what[96];
        std::snprintf(what, sizeof(what), "swiglu n=%u grid=%u block=%u", s.n, s.grid, s.block);
        check_agree(what, a, b);
        // and both against the host reference at the tolerance test_gpu_execution.cpp uses
        for (uint32_t i = 0; i < s.n; ++i) {
            float ref = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
            CHECK(near(b[i], ref, 2e-3f));
        }
    }
}

TEST_CASE("GPU migration - AdaLN modulate legacy vs MIR agree across shapes (gated and ungated)") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    for (bool gated : {false, true}) {
        std::string mir = c.emit_ptx_adaln_modulate(gated);
        std::string legacy = codegen::legacy::legacy_ptx_adaln_modulate(gated);

        struct Shape { uint32_t L, D, block; };
        for (Shape s : {Shape{2, 35, 128}, Shape{3, 64, 128}, Shape{5, 130, 128}, Shape{4, 8, 64}, Shape{2, 1024, 256}, Shape{3, 33, 32}}) {
            std::vector<float> x(s.L * s.D);
            for (uint32_t r = 0; r < s.L; ++r)
                for (uint32_t i = 0; i < s.D; ++i) x[r * s.D + i] = static_cast<float>((r + i) % 11) * 0.1f - 0.3f;
            std::vector<float> scale = pattern(s.D, 0.05f, 0.0f, 4);
            std::vector<float> shift = pattern(s.D, -0.1f, 0.0f, 3);
            std::vector<float> gate = pattern(s.D, 0.1f, 0.5f, 5);
            std::vector<float> a = run_adaln(legacy, gated, s.L, s.D, s.block, x, scale, shift, gate);
            std::vector<float> b = run_adaln(mir, gated, s.L, s.D, s.block, x, scale, shift, gate);
            char what[96];
            std::snprintf(what, sizeof(what), "adaln%s L=%u D=%u block=%u", gated ? "_gated" : "", s.L, s.D, s.block);
            check_agree(what, a, b);
            for (uint32_t r = 0; r < s.L; ++r) {
                for (uint32_t i = 0; i < s.D; ++i) {
                    float ref = x[r * s.D + i] * (1.0f + scale[i]) + shift[i];
                    if (gated) ref *= gate[i];
                    CHECK(near(b[r * s.D + i], ref, 1e-5f));
                }
            }
        }
    }
}

TEST_CASE("GPU migration - residual RMSNorm legacy vs MIR agree across shapes (y and in-place x)") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string mir = c.emit_ptx_fused_residual_rms_norm();
    std::string legacy = codegen::legacy::legacy_ptx_residual_rms_norm();
    float eps = 1e-5f;

    struct Shape { uint32_t B, D, block; };
    for (Shape s : {Shape{3, 37, 128}, Shape{2, 64, 128}, Shape{4, 300, 256}, Shape{1, 1024, 128}, Shape{2, 50, 32}, Shape{3, 96, 1024}}) {
        std::vector<float> x(s.B * s.D), res(s.B * s.D);
        for (uint32_t r = 0; r < s.B; ++r) {
            for (uint32_t i = 0; i < s.D; ++i) {
                x[r * s.D + i] = static_cast<float>((r + i) % 7) * 0.25f - 0.5f;
                res[r * s.D + i] = static_cast<float>((r * 3 + i) % 5) * 0.3f;
            }
        }
        std::vector<float> gamma = pattern(s.D, 0.1f, 1.0f, 3);
        std::vector<float> a = run_rms(legacy, s.B, s.D, s.block, eps, x, res, gamma);
        std::vector<float> b = run_rms(mir, s.B, s.D, s.block, eps, x, res, gamma);
        char what[96];
        std::snprintf(what, sizeof(what), "rms B=%u D=%u block=%u (y, x)", s.B, s.D, s.block);
        check_agree(what, a, b);
        // host reference for the MIR side at the test_gpu_execution.cpp tolerance
        for (uint32_t r = 0; r < s.B; ++r) {
            float sum = 0;
            for (uint32_t i = 0; i < s.D; ++i) { float v = x[r * s.D + i] + res[r * s.D + i]; sum += v * v; }
            float rrms = 1.0f / std::sqrt(sum / static_cast<float>(s.D) + eps);
            for (uint32_t i = 0; i < s.D; ++i) {
                float v = x[r * s.D + i] + res[r * s.D + i];
                CHECK(near(b[r * s.D + i], v * gamma[i] * rrms, 2e-3f));
                CHECK(near(b[s.B * s.D + r * s.D + i], v, 1e-6f));
            }
        }
    }
}
