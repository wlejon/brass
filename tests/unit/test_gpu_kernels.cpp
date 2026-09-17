// The fused ML GPU kernels built as MIR (build_ptx_swiglu,
// build_ptx_adaln_modulate, build_ptx_residual_rms_norm,
// build_ptx_layernorm_modulate, build_ptx_residual_layernorm,
// build_ptx_gemv_swiglu, build_ptx_gemv_residual) checked against
// double-precision host references on device over several shapes: row length
// a multiple of 4 and not (the float4 path and the scalar path), element
// counts that are not a multiple of the block size, rows above the grid,
// block sizes 32..1024. The kernels must also pass ptx::verify and ptxas and
// keep their approx recipes (ex2/rcp/div/rsqrt.approx), so the tolerances are
// the ones test_gpu_execution.cpp uses for the same kernels. Visible [SKIP]
// lines without ptxas / CUDA. The quantized GEMVs are in test_gpu_kernels_quant.cpp.

#include "ptx_test_support.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <algorithm>
#include <cstdio>

using namespace brass;
using brass::codegen::MlFusionCompiler;
using namespace ptxtest;

namespace {

// Tolerances (relative, see `near`), per recipe: exact float arithmetic
// (AdaLN, the in-place x += res) at 1e-5 / 1e-6; ex2.approx + rcp.approx
// SiLU at 2e-3; div.approx + rsqrt.approx statistics at 2e-3; the GEMV dot
// products at 1e-4 (residual) and 5e-3 (SiLU of a large dot product).
constexpr float kExactTol = 1e-5f;
constexpr float kInPlaceTol = 1e-6f;
constexpr float kApproxTol = 2e-3f;
constexpr float kGemvTol = 1e-4f;
constexpr float kGemvSwigluTol = 5e-3f;

float max_rel_diff(const std::vector<float>& got, const std::vector<double>& ref) {
    REQUIRE(got.size() == ref.size());
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float r = static_cast<float>(ref[i]);
        worst = std::max(worst, std::fabs(got[i] - r) / (1.0f + std::fabs(r)));
    }
    return worst;
}

// Prints the observed maximum relative difference and checks every element.
void check_reference(const char* what, const std::vector<float>& got, const std::vector<double>& ref, float tol) {
    float d = max_rel_diff(got, ref);
    std::printf("    %-44s max rel diff %.3g (tol %.0e)\n", what, static_cast<double>(d), static_cast<double>(tol));
    CHECK(d <= tol);
    for (size_t i = 0; i < got.size(); ++i) {
        if (!near(got[i], static_cast<float>(ref[i]), tol)) {
            std::fprintf(stderr, "%s: element %zu gpu %.9g host %.9g\n", what, i, got[i], ref[i]);
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

double silu(double g) { return g / (1.0 + std::exp(-g)); }

// Host LayerNorm statistics of one row (two-pass, like the kernels).
void host_ln_stats(const float* row, uint32_t D, float eps, double& mean, double& rstd) {
    mean = 0;
    for (uint32_t i = 0; i < D; ++i) mean += row[i];
    mean /= D;
    double var = 0;
    for (uint32_t i = 0; i < D; ++i) { double d = row[i] - mean; var += d * d; }
    var /= D;
    rstd = 1.0 / std::sqrt(var + eps);
}

// ---------------------------------------------------------------------------
// Device runners. Outputs are zero-initialised so rows beyond the grid stay 0.
// ---------------------------------------------------------------------------

// SwiGLU: out = silu(gate) * up, grid-stride
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

// Packed SwiGLU: x is [b, 2d] (gate | up per row), y = silu(gate) * up is [b, d], grid-stride
std::vector<float> run_swiglu_packed(const std::string& ptx, const std::vector<float>& x, uint32_t b, uint32_t d,
                                     uint32_t grid, uint32_t block) {
    CudaBuffer dx = upload(x);
    CudaBuffer dy = CudaBuffer::alloc(b * d * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* px = dx.device_ptr(); void* py = dy.device_ptr();
    uint32_t bb = b, dd = d;
    launch(ptx, "fused_swiglu_packed_kernel", grid, block, {&px, &py, &bb, &dd});
    return download<float>(dy, b * d);
}

// AdaLN modulate: y[row] = x[row] * (1 + scale) + shift [* gate], block per row
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

// Residual RMSNorm: x += res (in place), y = x * gamma * rrms, block per row.
// Returns y followed by the updated x.
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

// LayerNorm + modulate: y = ((x - mean) * rstd * gamma + beta) * (1 + scale) + shift, block per row.
std::vector<float> run_lnmod(const std::string& ptx, uint32_t R, uint32_t D, uint32_t block, float eps,
                             const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta,
                             const std::vector<float>& scale, const std::vector<float>& shift) {
    CudaBuffer dx = upload(x), dg = upload(gamma), db = upload(beta), dsc = upload(scale), dsh = upload(shift);
    CudaBuffer dy = CudaBuffer::alloc(R * D * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* px = dx.device_ptr(); void* pg = dg.device_ptr(); void* pb = db.device_ptr();
    void* psc = dsc.device_ptr(); void* psh = dsh.device_ptr(); void* py = dy.device_ptr();
    uint32_t r = R, d = D;
    float feps = eps;
    launch(ptx, "fused_layernorm_modulate_kernel", R, block, {&px, &pg, &pb, &psc, &psh, &py, &r, &d, &feps});
    return download<float>(dy, R * D);
}

// Residual LayerNorm: x += res (in place), y = (x - mean) * rstd * gamma + beta.
// Returns y followed by the updated x.
std::vector<float> run_res_ln(const std::string& ptx, uint32_t B, uint32_t D, uint32_t block, float eps,
                              const std::vector<float>& x, const std::vector<float>& res,
                              const std::vector<float>& gamma, const std::vector<float>& beta) {
    CudaBuffer dx = upload(x), dres = upload(res), dg = upload(gamma), db = upload(beta);
    CudaBuffer dy = CudaBuffer::alloc(B * D * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* px = dx.device_ptr(); void* pres = dres.device_ptr(); void* pg = dg.device_ptr();
    void* pb = db.device_ptr(); void* py = dy.device_ptr();
    uint32_t b = B, d = D;
    float feps = eps;
    launch(ptx, "fused_residual_layernorm_kernel", B, block, {&px, &pres, &pg, &pb, &py, &b, &d, &feps});
    std::vector<float> out = download<float>(dy, B * D);
    std::vector<float> xs = download<float>(dx, B * D);
    out.insert(out.end(), xs.begin(), xs.end());
    return out;
}

// GEMV SwiGLU: y[row] = silu(w_gate[row] . x) * (w_up[row] . x), block per row.
std::vector<float> run_gemv_swiglu(const std::string& ptx, uint32_t n, uint32_t k, uint32_t grid, uint32_t block,
                                   const std::vector<float>& wg, const std::vector<float>& wu, const std::vector<float>& x) {
    CudaBuffer dwg = upload(wg), dwu = upload(wu), dx = upload(x);
    CudaBuffer dy = CudaBuffer::alloc(n * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* pg = dwg.device_ptr(); void* pu = dwu.device_ptr(); void* px = dx.device_ptr(); void* py = dy.device_ptr();
    uint32_t nn = n, kk = k;
    launch(ptx, "fused_gemv_swiglu_kernel", grid, block, {&pg, &pu, &px, &py, &nn, &kk});
    return download<float>(dy, n);
}

// GEMV residual: y[row] = w_down[row] . x + res[row], block per row.
std::vector<float> run_gemv_res(const std::string& ptx, uint32_t n, uint32_t k, uint32_t grid, uint32_t block,
                                const std::vector<float>& wd, const std::vector<float>& x, const std::vector<float>& res) {
    CudaBuffer dwd = upload(wd), dx = upload(x), dres = upload(res);
    CudaBuffer dy = CudaBuffer::alloc(n * 4);
    REQUIRE(dy.valid() && dy.zero());
    void* pw = dwd.device_ptr(); void* px = dx.device_ptr(); void* pr = dres.device_ptr(); void* py = dy.device_ptr();
    uint32_t nn = n, kk = k;
    launch(ptx, "fused_gemv_residual_kernel", grid, block, {&pw, &px, &pr, &py, &nn, &kk});
    return download<float>(dy, n);
}

} // namespace

// ---------------------------------------------------------------------------
// Static checks: the kernels verify, assemble, and keep the approx recipe
// ---------------------------------------------------------------------------

TEST_CASE("GPU kernels - SwiGLU, AdaLN and RMSNorm verify, assemble and keep the approx recipe") {
    MlFusionCompiler c;
    Module ms("k_swiglu"), msp("k_swiglu_packed"), ma("k_adaln"), mg("k_adaln_g"), mr("k_rms");
    std::string swiglu = checked_mir_ptx(c.build_ptx_swiglu(ms));
    std::string swiglu_p = checked_mir_ptx(c.build_ptx_swiglu_packed(msp));
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
    has(swiglu_p, ".entry fused_swiglu_packed_kernel(");
    has(swiglu_p, "ex2.approx.f32"); has(swiglu_p, "rcp.approx.f32"); has(swiglu_p, "0f3FB8AA3B");
    has(swiglu_p, "ld.global.v4.f32"); has(swiglu_p, "st.global.v4.f32"); has(swiglu_p, "%nctaid.x");
    has(swiglu_p, "div.u32"); // row = e / d
    has(adaln, ".entry fused_adaln_modulate_kernel(");
    has(adaln, "fma.rn.f32"); has(adaln, "ld.global.v4.f32"); has(adaln, "st.global.v4.f32");
    has(adaln_g, ".entry fused_adaln_modulate_gated_kernel(");
    has(rms, ".entry fused_residual_rms_norm_kernel(");
    has(rms, "rsqrt.approx.f32"); has(rms, "div.approx.f32"); has(rms, "shfl.sync.down.b32");
    has(rms, "bar.sync 0;"); has(rms, ".shared .align 16 .f32 smem_0[32]");
    CHECK(rms.find("call ") == std::string::npos); // no CPU helper calls

    // The public emitters return exactly these kernels.
    CHECK(c.emit_ptx_swiglu() == swiglu);
    CHECK(c.emit_ptx_swiglu_packed() == swiglu_p);
    CHECK(c.emit_ptx_adaln_modulate(false) == adaln);
    CHECK(c.emit_ptx_adaln_modulate(true) == adaln_g);
    CHECK(c.emit_ptx_fused_residual_rms_norm() == rms);
    CHECK(c.emit_ptx_residual_rms_norm() == rms);
}

TEST_CASE("GPU kernels - LayerNorm and GEMV kernels verify, assemble and keep the recipe") {
    MlFusionCompiler c;
    Module ml("k_lnmod"), mr("k_res_ln"), mg("k_gemv_swiglu"), md("k_gemv_res");
    std::string lnmod = checked_mir_ptx(c.build_ptx_layernorm_modulate(ml));
    std::string res_ln = checked_mir_ptx(c.build_ptx_residual_layernorm(mr));
    std::string gemv_sw = checked_mir_ptx(c.build_ptx_gemv_swiglu(mg));
    std::string gemv_res = checked_mir_ptx(c.build_ptx_gemv_residual(md));

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
    for (const std::string* p : {&lnmod, &res_ln}) {
        has(*p, "rsqrt.approx.f32"); has(*p, "div.approx.f32"); has(*p, "shfl.sync.down.b32");
        has(*p, "bar.sync 0;"); has(*p, ".shared .align 16 .f32 smem_0[32]");
        has(*p, "ld.global.v4.f32"); has(*p, "st.global.v4.f32"); has(*p, "fma.rn.f32"); has(*p, "sub.f32");
        has(*p, "cvt.rn.f32.u32");
        CHECK(count(*p, "div.approx.f32") == 2); // mean and variance, two-pass
        CHECK(count(*p, "rsqrt.approx.f32") == 1);
        CHECK(p->find("call ") == std::string::npos);
    }
    has(lnmod, ".entry fused_layernorm_modulate_kernel(");
    has(lnmod, "0f3F800000"); // scale + 1
    has(res_ln, ".entry fused_residual_layernorm_kernel(");
    for (const std::string* p : {&gemv_sw, &gemv_res}) {
        has(*p, "shfl.sync.down.b32"); has(*p, "bar.sync 0;"); has(*p, ".shared .align 16 .f32 smem_0[32]");
        has(*p, "ld.global.v4.f32"); has(*p, "ld.global.f32"); has(*p, "fma.rn.f32"); has(*p, "st.global.f32");
        CHECK(p->find("st.global.v4") == std::string::npos); // one scalar output per block
        CHECK(p->find("call ") == std::string::npos);
    }
    has(gemv_sw, ".entry fused_gemv_swiglu_kernel(");
    has(gemv_sw, "ex2.approx.f32"); has(gemv_sw, "rcp.approx.f32"); has(gemv_sw, "max.f32"); has(gemv_sw, "min.f32");
    has(gemv_sw, "0fBFB8AA3B"); has(gemv_sw, "0fC2B00000"); has(gemv_sw, "0f42B00000"); // -log2e, -88, +88 clamps
    has(gemv_res, ".entry fused_gemv_residual_kernel(");
    CHECK(gemv_res.find("ex2.approx") == std::string::npos);

    // The public emitters return exactly these kernels.
    CHECK(c.emit_ptx_fused_layernorm_modulate() == lnmod);
    CHECK(c.emit_ptx_fused_residual_layernorm() == res_ln);
    CHECK(c.emit_ptx_fused_gemv_swiglu() == gemv_sw);
    CHECK(c.emit_ptx_fused_gemv_residual() == gemv_res);
}

// ---------------------------------------------------------------------------
// On-device runs against double-precision host references
// ---------------------------------------------------------------------------

TEST_CASE("GPU kernels - SwiGLU matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_swiglu();

    struct Shape { uint32_t n, grid, block; };
    for (Shape s : {Shape{64, 1, 64}, Shape{101, 1, 64}, Shape{1000, 3, 128}, Shape{4099, 4, 256}, Shape{7, 2, 32},
                    Shape{2050, 1, 1024}}) {
        std::vector<float> g(s.n), u(s.n);
        for (uint32_t i = 0; i < s.n; ++i) {
            g[i] = (static_cast<float>(i % 211) - 105.0f) * 0.08f;
            u[i] = static_cast<float>(i % 8) * 0.5f - 1.0f;
        }
        std::vector<float> got = run_swiglu(ptx, g, u, s.grid, s.block);
        std::vector<double> ref(s.n);
        for (uint32_t i = 0; i < s.n; ++i) ref[i] = silu(g[i]) * u[i];
        char what[96];
        std::snprintf(what, sizeof(what), "swiglu n=%u grid=%u block=%u", s.n, s.grid, s.block);
        check_reference(what, got, ref, kApproxTol);
    }
}

TEST_CASE("GPU kernels - packed SwiGLU matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_swiglu_packed();

    // d % 4 == 0 (float4 path) and not (all scalar); b * d not a multiple of
    // the grid's thread count; more threads than outputs; a single row.
    struct Shape { uint32_t b, d, grid, block; };
    for (Shape s : {Shape{1, 64, 1, 64}, Shape{4, 100, 2, 128}, Shape{3, 101, 1, 64}, Shape{8, 1152, 4, 256},
                    Shape{2, 7, 2, 32}, Shape{1, 4096, 8, 1024}, Shape{5, 12, 1, 1024}}) {
        std::vector<float> x(static_cast<size_t>(s.b) * 2 * s.d);
        for (uint32_t r = 0; r < s.b; ++r) {
            for (uint32_t col = 0; col < s.d; ++col) {
                uint32_t e = r * s.d + col;
                x[static_cast<size_t>(r) * 2 * s.d + col] = (static_cast<float>(e % 211) - 105.0f) * 0.08f;
                x[static_cast<size_t>(r) * 2 * s.d + s.d + col] = static_cast<float>(e % 8) * 0.5f - 1.0f;
            }
        }
        std::vector<float> got = run_swiglu_packed(ptx, x, s.b, s.d, s.grid, s.block);
        std::vector<double> ref(static_cast<size_t>(s.b) * s.d);
        for (uint32_t r = 0; r < s.b; ++r) {
            for (uint32_t col = 0; col < s.d; ++col) {
                const float* row = &x[static_cast<size_t>(r) * 2 * s.d];
                ref[static_cast<size_t>(r) * s.d + col] = silu(row[col]) * row[s.d + col];
            }
        }
        char what[96];
        std::snprintf(what, sizeof(what), "swiglu_packed b=%u d=%u grid=%u block=%u", s.b, s.d, s.grid, s.block);
        check_reference(what, got, ref, kApproxTol);
    }
}

TEST_CASE("GPU kernels - AdaLN modulate matches the host reference across shapes (gated and ungated)") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    for (bool gated : {false, true}) {
        std::string ptx = c.emit_ptx_adaln_modulate(gated);

        struct Shape { uint32_t L, D, block; };
        for (Shape s : {Shape{2, 35, 128}, Shape{3, 64, 128}, Shape{5, 130, 128}, Shape{4, 8, 64}, Shape{2, 1024, 256},
                        Shape{3, 33, 32}, Shape{2, 2052, 1024}}) {
            std::vector<float> x(s.L * s.D);
            for (uint32_t r = 0; r < s.L; ++r)
                for (uint32_t i = 0; i < s.D; ++i) x[r * s.D + i] = static_cast<float>((r + i) % 11) * 0.1f - 0.3f;
            std::vector<float> scale = pattern(s.D, 0.05f, 0.0f, 4);
            std::vector<float> shift = pattern(s.D, -0.1f, 0.0f, 3);
            std::vector<float> gate = pattern(s.D, 0.1f, 0.5f, 5);
            std::vector<float> got = run_adaln(ptx, gated, s.L, s.D, s.block, x, scale, shift, gate);
            std::vector<double> ref(s.L * s.D);
            for (uint32_t r = 0; r < s.L; ++r) {
                for (uint32_t i = 0; i < s.D; ++i) {
                    double v = static_cast<double>(x[r * s.D + i]) * (1.0 + scale[i]) + shift[i];
                    ref[r * s.D + i] = gated ? v * gate[i] : v;
                }
            }
            char what[96];
            std::snprintf(what, sizeof(what), "adaln%s L=%u D=%u block=%u", gated ? "_gated" : "", s.L, s.D, s.block);
            check_reference(what, got, ref, kExactTol);
        }
    }
}

TEST_CASE("GPU kernels - residual RMSNorm matches the host reference across shapes (y and in-place x)") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_residual_rms_norm();
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
        std::vector<float> got = run_rms(ptx, s.B, s.D, s.block, eps, x, res, gamma);
        std::vector<float> y(got.begin(), got.begin() + s.B * s.D);
        std::vector<float> xs(got.begin() + s.B * s.D, got.end());
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
        check_reference(what, y, ref_y, kApproxTol);
        std::snprintf(what, sizeof(what), "rms B=%u D=%u block=%u (in-place x)", s.B, s.D, s.block);
        check_reference(what, xs, ref_x, kInPlaceTol);
    }
}

namespace {
struct NormShape { uint32_t rows, D, block; };
// D a multiple of 4 and not (8, 300, 1024, 4096 vs 37), rows 1..4, blocks 32..1024.
const NormShape kNormShapes[] = {
    {1, 8, 32}, {4, 8, 512}, {3, 37, 128}, {2, 37, 1024}, {4, 300, 256}, {1, 300, 64}, {2, 1024, 1024}, {3, 1024, 128},
    {2, 4096, 256},
};
} // namespace

TEST_CASE("GPU kernels - LayerNorm modulate matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_layernorm_modulate();
    float eps = 1e-5f;

    for (NormShape s : kNormShapes) {
        std::vector<float> x(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r)
            for (uint32_t i = 0; i < s.D; ++i) x[r * s.D + i] = static_cast<float>((r * 5 + i) % 9) * 0.2f - 0.8f;
        std::vector<float> gamma = pattern(s.D, 0.1f, 1.0f, 3);
        std::vector<float> beta = pattern(s.D, 0.05f, 0.0f, 2);
        std::vector<float> scale = pattern(s.D, 0.02f, 0.0f, 4);
        std::vector<float> shift = pattern(s.D, -0.03f, 0.0f, 3);
        std::vector<float> got = run_lnmod(ptx, s.rows, s.D, s.block, eps, x, gamma, beta, scale, shift);
        std::vector<double> ref(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r) {
            double mean, rstd;
            host_ln_stats(&x[r * s.D], s.D, eps, mean, rstd);
            for (uint32_t i = 0; i < s.D; ++i) {
                double ln = (x[r * s.D + i] - mean) * rstd * gamma[i] + beta[i];
                ref[r * s.D + i] = ln * (1.0 + scale[i]) + shift[i];
            }
        }
        char what[96];
        std::snprintf(what, sizeof(what), "lnmod R=%u D=%u block=%u", s.rows, s.D, s.block);
        check_reference(what, got, ref, kApproxTol);
    }
}

TEST_CASE("GPU kernels - residual LayerNorm matches the host reference across shapes (y and in-place x)") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_residual_layernorm();
    float eps = 1e-5f;

    for (NormShape s : kNormShapes) {
        std::vector<float> x(s.rows * s.D), res(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r) {
            for (uint32_t i = 0; i < s.D; ++i) {
                x[r * s.D + i] = static_cast<float>((r + i) % 9) * 0.2f - 0.8f;
                res[r * s.D + i] = static_cast<float>((r * 2 + i) % 5) * 0.1f;
            }
        }
        std::vector<float> gamma = pattern(s.D, 0.1f, 1.0f, 3);
        std::vector<float> beta = pattern(s.D, 0.05f, 0.0f, 4);
        std::vector<float> got = run_res_ln(ptx, s.rows, s.D, s.block, eps, x, res, gamma, beta);
        std::vector<float> y(got.begin(), got.begin() + s.rows * s.D);
        std::vector<float> xs(got.begin() + s.rows * s.D, got.end());
        std::vector<double> ref_y(s.rows * s.D), ref_x(s.rows * s.D);
        for (uint32_t r = 0; r < s.rows; ++r) {
            // The kernel computes x + res in float, then the statistics of that row.
            std::vector<float> v(s.D);
            for (uint32_t i = 0; i < s.D; ++i) {
                v[i] = x[r * s.D + i] + res[r * s.D + i];
                ref_x[r * s.D + i] = v[i];
            }
            double mean, rstd;
            host_ln_stats(v.data(), s.D, eps, mean, rstd);
            for (uint32_t i = 0; i < s.D; ++i) ref_y[r * s.D + i] = (v[i] - mean) * rstd * gamma[i] + beta[i];
        }
        char what[96];
        std::snprintf(what, sizeof(what), "res_ln B=%u D=%u block=%u", s.rows, s.D, s.block);
        check_reference(what, y, ref_y, kApproxTol);
        std::snprintf(what, sizeof(what), "res_ln B=%u D=%u block=%u (in-place x)", s.rows, s.D, s.block);
        check_reference(what, xs, ref_x, kInPlaceTol);
    }
}

namespace {
struct GemvShape { uint32_t n, k, grid, block; };
// k a multiple of 4 and not (37, 1027), n below and above the grid, blocks 32..1024.
const GemvShape kGemvShapes[] = {
    {1, 8, 1, 64}, {4, 64, 4, 128}, {3, 37, 3, 32}, {5, 1027, 5, 256}, {2, 1024, 2, 1024}, {7, 100, 4, 128}, {6, 2048, 6, 512},
    {9, 4096, 7, 256},
};

void gemv_inputs(const GemvShape& s, std::vector<float>& wg, std::vector<float>& wu, std::vector<float>& wd,
                 std::vector<float>& x, std::vector<float>& res) {
    wg.assign(s.n * s.k, 0.0f); wu.assign(s.n * s.k, 0.0f); wd.assign(s.n * s.k, 0.0f);
    x.assign(s.k, 0.0f); res.assign(s.n, 0.0f);
    for (uint32_t i = 0; i < s.k; ++i) x[i] = static_cast<float>(i % 5) * 0.25f - 0.5f;
    for (uint32_t r = 0; r < s.n; ++r) {
        for (uint32_t i = 0; i < s.k; ++i) {
            wg[r * s.k + i] = static_cast<float>((r + i) % 9) * 0.1f - 0.4f;
            wu[r * s.k + i] = static_cast<float>((r * 3 + i) % 7) * 0.15f - 0.5f;
            wd[r * s.k + i] = static_cast<float>((r * 2 + i) % 6) * 0.2f - 0.6f;
        }
        res[r] = 0.1f * static_cast<float>(r + 1);
    }
}

double host_dot(const float* w, const float* x, uint32_t k) {
    double d = 0;
    for (uint32_t i = 0; i < k; ++i) d += static_cast<double>(w[i]) * x[i];
    return d;
}
} // namespace

TEST_CASE("GPU kernels - GEMV SwiGLU matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_gemv_swiglu();

    for (GemvShape s : kGemvShapes) {
        std::vector<float> wg, wu, wd, x, res;
        gemv_inputs(s, wg, wu, wd, x, res);
        std::vector<float> got = run_gemv_swiglu(ptx, s.n, s.k, s.grid, s.block, wg, wu, x);
        std::vector<double> ref(s.n, 0.0); // rows beyond the grid stay 0
        for (uint32_t r = 0; r < std::min(s.n, s.grid); ++r) {
            ref[r] = silu(host_dot(&wg[r * s.k], x.data(), s.k)) * host_dot(&wu[r * s.k], x.data(), s.k);
        }
        char what[96];
        std::snprintf(what, sizeof(what), "gemv_swiglu n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, got, ref, kGemvSwigluTol);
        for (uint32_t r = s.grid; r < s.n; ++r) CHECK(got[r] == 0.0f);
    }
}

TEST_CASE("GPU kernels - GEMV residual matches the host reference across shapes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_fused_gemv_residual();

    for (GemvShape s : kGemvShapes) {
        std::vector<float> wg, wu, wd, x, res;
        gemv_inputs(s, wg, wu, wd, x, res);
        std::vector<float> got = run_gemv_res(ptx, s.n, s.k, s.grid, s.block, wd, x, res);
        std::vector<double> ref(s.n, 0.0);
        for (uint32_t r = 0; r < std::min(s.n, s.grid); ++r) ref[r] = host_dot(&wd[r * s.k], x.data(), s.k) + res[r];
        char what[96];
        std::snprintf(what, sizeof(what), "gemv_res n=%u k=%u grid=%u block=%u", s.n, s.k, s.grid, s.block);
        check_reference(what, got, ref, kGemvTol);
        for (uint32_t r = s.grid; r < s.n; ++r) CHECK(got[r] == 0.0f);
    }
}
