// Real GPU execution tests for the PTX code generator.
//
// These tests do two things:
//   1. Validate that every kernel Brass emits assembles with `ptxas`
//      (skipped when ptxas is not installed).
//   2. JIT-compile the PTX through the CUDA driver and execute it on a real
//      GPU, comparing against a CPU reference (skipped when no device).
//
// All GPU tests degrade to a no-op pass when CUDA is unavailable so that the
// suite still runs on machines without a GPU.

#include "test_framework.hpp"

#include <brass/codegen/ml_fusion.hpp>
#include <brass/gpu/cuda_driver.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/brass_c_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::gpu;
using namespace brass::target;

namespace {

// Skips are printed so that a suite passing without touching a device is
// distinguishable from one that ran.
void report_skip(const char* what) {
    std::cout << "  [SKIP] " << what << "\n" << std::flush;
}

bool gpu_ready() {
    static const bool ok = cuda_available();
    if (!ok) report_skip(("CUDA unavailable (" + cuda_last_error() + "): test not executed on device").c_str());
    return ok;
}

// Portable null-device redirection for std::system (cmd.exe has no /dev/null).
const char* null_device() {
#if defined(_WIN32)
    return "NUL";
#else
    return "/dev/null";
#endif
}

std::string quiet(const std::string& cmd) {
    return cmd + " >" + null_device() + " 2>&1";
}

std::filesystem::path scratch_path(const char* name) {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = std::filesystem::current_path();
    return dir / name;
}

bool ptxas_available() {
    static const bool ok = (std::system(quiet("ptxas --version").c_str()) == 0);
    if (!ok) report_skip("ptxas not on PATH: assembly not validated");
    return ok;
}

bool ptxas_assembles(const std::string& ptx, const char* arch) {
    std::filesystem::path in = scratch_path("brass_gpu_ptx_check.ptx");
    std::filesystem::path out = scratch_path("brass_gpu_ptx_check.cubin");
    {
        std::ofstream f(in, std::ios::binary);
        if (!f) return false;
        f << ptx;
    }
    std::string cmd = std::string("ptxas -arch=") + arch + " \"" + in.string() + "\" -o \"" + out.string() + "\"";
    bool ok = std::system(quiet(cmd).c_str()) == 0;
    std::error_code ec;
    std::filesystem::remove(out, ec);
    return ok;
}

bool near(float a, float b, float eps) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(b));
}

// Launch helper. args are pointers to the argument values in parameter order.
bool gpu_launch(const CudaModule& mod, const char* entry, uint32_t grid, uint32_t block,
                std::vector<void*> args, std::string* err) {
    return mod.launch_1d(entry, grid, block, args.data(), 0, err);
}

std::string mir_rms_ptx() {
    MlFusionCompiler c;
    Module mod("mir_rms");
    Function* fn = c.build_residual_rms_norm(mod);
    return PtxTarget::emit_function(*fn);
}
std::string mir_swiglu_ptx() {
    MlFusionCompiler c;
    Module mod("mir_swiglu");
    Function* fn = c.build_swiglu(mod);
    return PtxTarget::emit_function(*fn);
}
std::string mir_adaln_ptx(bool gated) {
    MlFusionCompiler c;
    Module mod(gated ? "mir_adaln_g" : "mir_adaln");
    Function* fn = c.build_adaln_modulate(
        mod, gated, gated ? "fused_adaln_modulate_gated" : "fused_adaln_modulate");
    return PtxTarget::emit_function(*fn);
}

// ---------------------------------------------------------------------------
// ptxas validation (no GPU required)
// ---------------------------------------------------------------------------

TEST_CASE("GPU - every emitted kernel assembles with ptxas") {
    if (!ptxas_available()) return;
    MlFusionCompiler c;
    std::vector<std::pair<std::string, std::string>> kernels = {
        { "rms", c.emit_ptx_fused_residual_rms_norm() },
        { "lnmod", c.emit_ptx_fused_layernorm_modulate() },
        { "swiglu", c.emit_ptx_swiglu() },
        { "adaln", c.emit_ptx_adaln_modulate(false) },
        { "adaln_gated", c.emit_ptx_adaln_modulate(true) },
        { "q8dot", c.emit_ptx_q8_dot() },
        { "blockq8", c.emit_ptx_block_q8_dot() },
        { "gemv_swiglu", c.emit_ptx_fused_gemv_swiglu() },
        { "gemv_res", c.emit_ptx_fused_gemv_residual() },
        { "gemv_q8", c.emit_ptx_fused_gemv_q8_0() },
        { "gemv_q4k", c.emit_ptx_fused_gemv_q4_k() },
        { "res_ln", c.emit_ptx_fused_residual_layernorm() },
        { "mir_rms", mir_rms_ptx() },
        { "mir_swiglu", mir_swiglu_ptx() },
        { "mir_adaln", mir_adaln_ptx(false) },
        { "mir_adaln_gated", mir_adaln_ptx(true) },
    };
    for (auto& [name, ptx] : kernels) {
        bool ok = ptxas_assembles(ptx, "sm_70");
        if (!ok) std::fprintf(stderr, "ptxas rejected kernel: %s\n", name.c_str());
        CHECK(ok);
    }
}

// ---------------------------------------------------------------------------
// MIR -> PTX -> GPU (the actual JIT codegen path), single-thread scalar kernels
// ---------------------------------------------------------------------------

TEST_CASE("GPU - MIR residual RMSNorm executes") {
    if (!gpu_ready()) return;
    std::string err;
    CudaModule mod = CudaModule::load(mir_rms_ptx(), &err);
    REQUIRE(mod.valid());

    uint64_t n = 64;
    float eps = 1e-5f;
    std::vector<float> x(n), res(n), w(n), out(n, 0.0f), ref_res(n), ref_out(n);
    float sum = 0;
    for (uint64_t i = 0; i < n; ++i) {
        x[i] = float(i % 7) * 0.25f - 0.5f;
        res[i] = float(i % 5) * 0.3f;
        w[i] = 1.0f + float(i % 3) * 0.1f;
        ref_res[i] = x[i] + res[i];
        sum += ref_res[i] * ref_res[i];
    }
    float inv_n = 1.0f / float(n);
    float inv_rms = 1.0f / std::sqrt(sum * inv_n + eps);
    for (uint64_t i = 0; i < n; ++i) ref_out[i] = ref_res[i] * inv_rms * w[i];

    CudaBuffer dx = CudaBuffer::alloc(n * 4), dres = CudaBuffer::alloc(n * 4);
    CudaBuffer dw = CudaBuffer::alloc(n * 4), dout = CudaBuffer::alloc(n * 4);
    REQUIRE(dx.valid() && dres.valid() && dw.valid() && dout.valid());
    dx.upload(x.data(), n * 4); dres.upload(res.data(), n * 4); dw.upload(w.data(), n * 4);

    void* px = dx.device_ptr(); void* pres = dres.device_ptr();
    void* pw = dw.device_ptr(); void* po = dout.device_ptr();
    float feps = eps, finv = inv_n;
    std::vector<void*> args = { &px, &pres, &pw, &po, &n, &feps, &finv };
    REQUIRE(gpu_launch(mod, "fused_residual_rms_norm", 1, 1, args, &err));

    std::vector<float> got_res(n), got_out(n);
    REQUIRE(dres.download(got_res.data(), n * 4));
    REQUIRE(dout.download(got_out.data(), n * 4));
    for (uint64_t i = 0; i < n; ++i) {
        CHECK(near(got_res[i], ref_res[i], 1e-4f));
        CHECK(near(got_out[i], ref_out[i], 1e-4f));
    }
}

TEST_CASE("GPU - MIR SwiGLU executes") {
    if (!gpu_ready()) return;
    std::string err;
    CudaModule mod = CudaModule::load(mir_swiglu_ptx(), &err);
    REQUIRE(mod.valid());

    uint64_t n = 64;
    std::vector<float> g(n), u(n), out(n, 0.0f), ref(n);
    for (uint64_t i = 0; i < n; ++i) {
        g[i] = (float(i) - 32.0f) * 0.1f;
        u[i] = float(i % 8) * 0.5f - 1.0f;
        ref[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    }
    CudaBuffer dg = CudaBuffer::alloc(n * 4), du = CudaBuffer::alloc(n * 4), dout = CudaBuffer::alloc(n * 4);
    dg.upload(g.data(), n * 4); du.upload(u.data(), n * 4);
    void* pg = dg.device_ptr(); void* pu = du.device_ptr(); void* po = dout.device_ptr();
    uint64_t nn = n;
    std::vector<void*> args = { &pg, &pu, &po, &nn };
    REQUIRE(gpu_launch(mod, "fused_swiglu", 1, 1, args, &err));
    std::vector<float> got(n);
    REQUIRE(dout.download(got.data(), n * 4));
    for (uint64_t i = 0; i < n; ++i) CHECK(near(got[i], ref[i], 1e-3f));
}

TEST_CASE("GPU - MIR AdaLN modulate executes (gated and ungated)") {
    if (!gpu_ready()) return;
    uint64_t n = 48;
    std::vector<float> x(n), scale(n), shift(n), gate(n), ref(n);
    for (uint64_t i = 0; i < n; ++i) {
        x[i] = float(i) * 0.1f;
        scale[i] = 0.05f * float(i % 4);
        shift[i] = -0.1f * float(i % 3);
        gate[i] = 0.5f + float(i % 5) * 0.1f;
        ref[i] = x[i] * (1.0f + scale[i]) + shift[i];
    }
    uint64_t nn = n;
    {   // ungated
        std::string err;
        CudaModule mod = CudaModule::load(mir_adaln_ptx(false), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(n * 4), ds = CudaBuffer::alloc(n * 4);
        CudaBuffer dh = CudaBuffer::alloc(n * 4), dout = CudaBuffer::alloc(n * 4);
        dx.upload(x.data(), n * 4); ds.upload(scale.data(), n * 4); dh.upload(shift.data(), n * 4);
        void* px = dx.device_ptr(); void* ps = ds.device_ptr();
        void* ph = dh.device_ptr(); void* po = dout.device_ptr();
        std::vector<void*> args = { &px, &ps, &ph, &po, &nn };
        REQUIRE(gpu_launch(mod, "fused_adaln_modulate", 1, 1, args, &err));
        std::vector<float> got(n);
        REQUIRE(dout.download(got.data(), n * 4));
        for (uint64_t i = 0; i < n; ++i) CHECK(near(got[i], ref[i], 1e-5f));
    }
    {   // gated
        std::string err;
        CudaModule mod = CudaModule::load(mir_adaln_ptx(true), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(n * 4), ds = CudaBuffer::alloc(n * 4);
        CudaBuffer dh = CudaBuffer::alloc(n * 4), dg = CudaBuffer::alloc(n * 4), dout = CudaBuffer::alloc(n * 4);
        dx.upload(x.data(), n * 4); ds.upload(scale.data(), n * 4);
        dh.upload(shift.data(), n * 4); dg.upload(gate.data(), n * 4);
        void* px = dx.device_ptr(); void* ps = ds.device_ptr(); void* ph = dh.device_ptr();
        void* pg = dg.device_ptr(); void* po = dout.device_ptr();
        std::vector<void*> args = { &px, &ps, &ph, &pg, &po, &nn };
        REQUIRE(gpu_launch(mod, "fused_adaln_modulate_gated", 1, 1, args, &err));
        std::vector<float> got(n);
        REQUIRE(dout.download(got.data(), n * 4));
        for (uint64_t i = 0; i < n; ++i) CHECK(near(got[i], ref[i] * gate[i], 1e-5f));
    }
}

TEST_CASE("GPU - MIR block Q8_0 dot executes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string err;
    CudaModule mod = CudaModule::load(c.emit_ptx_block_q8_dot(), &err);
    REQUIRE(mod.valid());

    struct BlockQ8 { float d; int8_t qs[32]; };
    uint64_t nb = 4;
    std::vector<BlockQ8> a(nb), b(nb);
    float ref = 0;
    for (uint64_t blk = 0; blk < nb; ++blk) {
        a[blk].d = 0.01f * float(blk + 1);
        b[blk].d = 0.02f * float(blk + 2);
        int32_t acc = 0;
        for (int j = 0; j < 32; ++j) {
            a[blk].qs[j] = int8_t((blk * 32 + j * 5) % 256 - 128);
            b[blk].qs[j] = int8_t((blk * 32 + j * 9) % 256 - 128);
            acc += int32_t(a[blk].qs[j]) * int32_t(b[blk].qs[j]);
        }
        ref += float(acc) * a[blk].d * b[blk].d;
    }
    CudaBuffer da = CudaBuffer::alloc(nb * sizeof(BlockQ8));
    CudaBuffer db = CudaBuffer::alloc(nb * sizeof(BlockQ8));
    CudaBuffer dout = CudaBuffer::alloc(4);
    da.upload(a.data(), nb * sizeof(BlockQ8));
    db.upload(b.data(), nb * sizeof(BlockQ8));
    void* pa = da.device_ptr(); void* pb = db.device_ptr(); void* po = dout.device_ptr();
    uint64_t nn = nb;
    std::vector<void*> args = { &pa, &pb, &po, &nn };
    REQUIRE(gpu_launch(mod, "fused_block_q8_dot", 1, 1, args, &err));
    float got = 0;
    REQUIRE(dout.download(&got, 4));
    CHECK(near(got, ref, 1e-4f));
}

// ---------------------------------------------------------------------------
// Hand-written fast kernels: execute with real grids and compare to reference
// ---------------------------------------------------------------------------

TEST_CASE("GPU - fused residual RMSNorm fast kernel executes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string err;
    CudaModule mod = CudaModule::load(c.emit_ptx_fused_residual_rms_norm(), &err);
    REQUIRE(mod.valid());

    uint32_t B = 3, D = 37; // D not a multiple of 4 exercises the tail
    float eps = 1e-5f;
    std::vector<float> x(B * D), res(B * D), gamma(D), y(B * D, 0.0f);
    std::vector<float> ref(B * D);
    for (uint32_t r = 0; r < B; ++r) {
        for (uint32_t i = 0; i < D; ++i) {
            x[r * D + i] = float((r + i) % 7) * 0.25f - 0.5f;
            res[r * D + i] = float((r * 3 + i) % 5) * 0.3f;
            gamma[i] = 1.0f + float(i % 3) * 0.1f;
        }
        float sum = 0;
        for (uint32_t i = 0; i < D; ++i) { float v = x[r * D + i] + res[r * D + i]; sum += v * v; }
        float rrms = 1.0f / std::sqrt(sum / float(D) + eps);
        for (uint32_t i = 0; i < D; ++i) {
            float v = x[r * D + i] + res[r * D + i];
            ref[r * D + i] = v * gamma[i] * rrms;
        }
    }
    CudaBuffer dx = CudaBuffer::alloc(B * D * 4), dres = CudaBuffer::alloc(B * D * 4);
    CudaBuffer dg = CudaBuffer::alloc(D * 4), dy = CudaBuffer::alloc(B * D * 4);
    dx.upload(x.data(), B * D * 4); dres.upload(res.data(), B * D * 4);
    dg.upload(gamma.data(), D * 4);
    void* px = dx.device_ptr(); void* pres = dres.device_ptr(); void* pg = dg.device_ptr(); void* py = dy.device_ptr();
    float feps = eps;
    std::vector<void*> args = { &px, &pres, &pg, &py, &B, &D, &feps };
    REQUIRE(gpu_launch(mod, "fused_residual_rms_norm_kernel", B, 128, args, &err));
    std::vector<float> got(B * D);
    REQUIRE(dy.download(got.data(), B * D * 4));
    for (size_t i = 0; i < got.size(); ++i) CHECK(near(got[i], ref[i], 2e-3f));
}

TEST_CASE("GPU - fused LayerNorm + Modulate fast kernel executes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string err;
    CudaModule mod = CudaModule::load(c.emit_ptx_fused_layernorm_modulate(), &err);
    REQUIRE(mod.valid());

    uint32_t R = 2, D = 35;
    float eps = 1e-5f;
    std::vector<float> x(R * D), gamma(D), beta(D), scale(D), shift(D), y(R * D, 0.0f), ref(R * D);
    for (uint32_t i = 0; i < D; ++i) {
        gamma[i] = 1.0f + float(i % 3) * 0.1f;
        beta[i] = 0.05f * float(i % 2);
        scale[i] = 0.02f * float(i % 4);
        shift[i] = -0.03f * float(i % 3);
    }
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t i = 0; i < D; ++i) x[r * D + i] = float((r + i) % 9) * 0.2f - 0.8f;
        float mean = 0;
        for (uint32_t i = 0; i < D; ++i) mean += x[r * D + i];
        mean /= float(D);
        float var = 0;
        for (uint32_t i = 0; i < D; ++i) { float d = x[r * D + i] - mean; var += d * d; }
        var /= float(D);
        float rstd = 1.0f / std::sqrt(var + eps);
        for (uint32_t i = 0; i < D; ++i) {
            float ln = (x[r * D + i] - mean) * rstd * gamma[i] + beta[i];
            ref[r * D + i] = ln * (1.0f + scale[i]) + shift[i];
        }
    }
    CudaBuffer dx = CudaBuffer::alloc(R * D * 4), dg = CudaBuffer::alloc(D * 4), db = CudaBuffer::alloc(D * 4);
    CudaBuffer dsc = CudaBuffer::alloc(D * 4), dsh = CudaBuffer::alloc(D * 4), dy = CudaBuffer::alloc(R * D * 4);
    dx.upload(x.data(), R * D * 4); dg.upload(gamma.data(), D * 4); db.upload(beta.data(), D * 4);
    dsc.upload(scale.data(), D * 4); dsh.upload(shift.data(), D * 4);
    void* px = dx.device_ptr(); void* pg = dg.device_ptr(); void* pb = db.device_ptr();
    void* psc = dsc.device_ptr(); void* psh = dsh.device_ptr(); void* py = dy.device_ptr();
    float feps = eps;
    std::vector<void*> args = { &px, &pg, &pb, &psc, &psh, &py, &R, &D, &feps };
    REQUIRE(gpu_launch(mod, "fused_layernorm_modulate_kernel", R, 128, args, &err));
    std::vector<float> got(R * D);
    REQUIRE(dy.download(got.data(), R * D * 4));
    for (size_t i = 0; i < got.size(); ++i) CHECK(near(got[i], ref[i], 2e-3f));
}

TEST_CASE("GPU - fused SwiGLU fast kernel executes with tail") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string err;
    CudaModule mod = CudaModule::load(c.emit_ptx_swiglu(), &err);
    REQUIRE(mod.valid());

    uint32_t n = 101; // not a multiple of 4 -> exercises scalar tail
    std::vector<float> g(n), u(n), out(n, 0.0f), ref(n);
    for (uint32_t i = 0; i < n; ++i) {
        g[i] = (float(i) - 50.0f) * 0.08f;
        u[i] = float(i % 8) * 0.5f - 1.0f;
        ref[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    }
    CudaBuffer dg = CudaBuffer::alloc(n * 4), du = CudaBuffer::alloc(n * 4), dout = CudaBuffer::alloc(n * 4);
    dg.upload(g.data(), n * 4); du.upload(u.data(), n * 4);
    void* pg = dg.device_ptr(); void* pu = du.device_ptr(); void* po = dout.device_ptr();
    std::vector<void*> args = { &pg, &pu, &po, &n };
    REQUIRE(gpu_launch(mod, "fused_swiglu_kernel", 1, 64, args, &err));
    std::vector<float> got(n);
    REQUIRE(dout.download(got.data(), n * 4));
    for (uint32_t i = 0; i < n; ++i) CHECK(near(got[i], ref[i], 2e-3f));
}

TEST_CASE("GPU - fused AdaLN modulate fast kernel executes (gated and ungated)") {
    if (!gpu_ready()) return;
    uint32_t L = 2, D = 35;
    std::vector<float> x(L * D), scale(D), shift(D), gate(D), ref(L * D), refg(L * D);
    for (uint32_t i = 0; i < D; ++i) {
        scale[i] = 0.05f * float(i % 4);
        shift[i] = -0.1f * float(i % 3);
        gate[i] = 0.5f + float(i % 5) * 0.1f;
    }
    for (uint32_t r = 0; r < L; ++r) {
        for (uint32_t i = 0; i < D; ++i) {
            x[r * D + i] = float((r + i) % 11) * 0.1f;
            float v = x[r * D + i] * (1.0f + scale[i]) + shift[i];
            ref[r * D + i] = v;
            refg[r * D + i] = v * gate[i];
        }
    }
    {   // ungated
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_adaln_modulate(false), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(L * D * 4), ds = CudaBuffer::alloc(D * 4);
        CudaBuffer dh = CudaBuffer::alloc(D * 4), dy = CudaBuffer::alloc(L * D * 4);
        dx.upload(x.data(), L * D * 4); ds.upload(scale.data(), D * 4); dh.upload(shift.data(), D * 4);
        void* px = dx.device_ptr(); void* ps = ds.device_ptr(); void* ph = dh.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &px, &ps, &ph, &py, &L, &D };
        REQUIRE(gpu_launch(mod, "fused_adaln_modulate_kernel", L, 128, args, &err));
        std::vector<float> got(L * D);
        REQUIRE(dy.download(got.data(), L * D * 4));
        for (size_t i = 0; i < got.size(); ++i) CHECK(near(got[i], ref[i], 1e-5f));
    }
    {   // gated - the previously-broken shift load is exercised here
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_adaln_modulate(true), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(L * D * 4), ds = CudaBuffer::alloc(D * 4), dh = CudaBuffer::alloc(D * 4);
        CudaBuffer dgt = CudaBuffer::alloc(D * 4), dy = CudaBuffer::alloc(L * D * 4);
        dx.upload(x.data(), L * D * 4); ds.upload(scale.data(), D * 4);
        dh.upload(shift.data(), D * 4); dgt.upload(gate.data(), D * 4);
        void* px = dx.device_ptr(); void* ps = ds.device_ptr(); void* ph = dh.device_ptr();
        void* pg = dgt.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &px, &ps, &ph, &pg, &py, &L, &D };
        REQUIRE(gpu_launch(mod, "fused_adaln_modulate_gated_kernel", L, 128, args, &err));
        std::vector<float> got(L * D);
        REQUIRE(dy.download(got.data(), L * D * 4));
        for (size_t i = 0; i < got.size(); ++i) CHECK(near(got[i], refg[i], 1e-5f));
    }
}

TEST_CASE("GPU - fused GEMV SwiGLU/Residual fast kernels execute with tail") {
    if (!gpu_ready()) return;
    uint32_t N = 3, K = 70; // K not a multiple of 4 exercises the scalar tail
    std::vector<float> x(K), wg(N * K), wu(N * K), wd(N * K), res(N);
    std::vector<float> ref(N), refd(N);
    for (uint32_t i = 0; i < K; ++i) x[i] = float(i % 7) * 0.1f - 0.3f;
    for (uint32_t r = 0; r < N; ++r) {
        float g = 0, u = 0, d = 0;
        for (uint32_t i = 0; i < K; ++i) {
            wg[r * K + i] = float((r * 5 + i) % 9) * 0.1f - 0.4f;
            wu[r * K + i] = float((r * 3 + i) % 7) * 0.15f - 0.5f;
            wd[r * K + i] = float((r * 2 + i) % 6) * 0.2f - 0.6f;
            g += wg[r * K + i] * x[i];
            u += wu[r * K + i] * x[i];
            d += wd[r * K + i] * x[i];
        }
        ref[r] = (g / (1.0f + std::exp(-g))) * u;
        res[r] = 0.1f * float(r + 1);
        refd[r] = d + res[r];
    }
    {   // swiglu
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_fused_gemv_swiglu(), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(K * 4), dwg = CudaBuffer::alloc(N * K * 4);
        CudaBuffer dwu = CudaBuffer::alloc(N * K * 4), dy = CudaBuffer::alloc(N * 4);
        dx.upload(x.data(), K * 4); dwg.upload(wg.data(), N * K * 4); dwu.upload(wu.data(), N * K * 4);
        void* px = dx.device_ptr(); void* pg = dwg.device_ptr(); void* pu = dwu.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &pg, &pu, &px, &py, &N, &K };
        REQUIRE(gpu_launch(mod, "fused_gemv_swiglu_kernel", N, 128, args, &err));
        std::vector<float> got(N);
        REQUIRE(dy.download(got.data(), N * 4));
        for (uint32_t r = 0; r < N; ++r) CHECK(near(got[r], ref[r], 5e-3f));
    }
    {   // residual
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_fused_gemv_residual(), &err);
        REQUIRE(mod.valid());
        CudaBuffer dx = CudaBuffer::alloc(K * 4), dwd = CudaBuffer::alloc(N * K * 4);
        CudaBuffer dr = CudaBuffer::alloc(N * 4), dy = CudaBuffer::alloc(N * 4);
        dx.upload(x.data(), K * 4); dwd.upload(wd.data(), N * K * 4); dr.upload(res.data(), N * 4);
        void* px = dx.device_ptr(); void* pw = dwd.device_ptr(); void* pr = dr.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &pw, &px, &pr, &py, &N, &K };
        REQUIRE(gpu_launch(mod, "fused_gemv_residual_kernel", N, 128, args, &err));
        std::vector<float> got(N);
        REQUIRE(dy.download(got.data(), N * 4));
        for (uint32_t r = 0; r < N; ++r) CHECK(near(got[r], refd[r], 1e-4f));
    }
}

TEST_CASE("GPU - fused GEMV Q8_0 and Q4_K fast kernels execute") {
    if (!gpu_ready()) return;
    {   // Q8_0: fp16 scale + 32 int8 per block
        struct Q8Block { uint16_t d; int8_t qs[32]; };
        uint32_t N = 4, K = 128, BPR = K / 32;
        std::vector<Q8Block> w(N * BPR);
        std::vector<float> x(K), ref(N, 0.0f);
        for (uint32_t i = 0; i < K; ++i) x[i] = float(i % 11) * 0.1f - 0.5f;
        for (uint32_t r = 0; r < N; ++r) {
            float acc = 0;
            for (uint32_t b = 0; b < BPR; ++b) {
                Q8Block& blk = w[r * BPR + b];
                blk.d = 0x3800; // 0.5 in fp16
                for (int j = 0; j < 32; ++j) {
                    blk.qs[j] = int8_t((r * 17 + b * 7 + j * 3) % 25 - 12);
                    acc += 0.5f * float(blk.qs[j]) * x[b * 32 + j];
                }
            }
            ref[r] = acc;
        }
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_fused_gemv_q8_0(), &err);
        REQUIRE(mod.valid());
        CudaBuffer dw = CudaBuffer::alloc(N * BPR * sizeof(Q8Block)), dx = CudaBuffer::alloc(K * 4), dy = CudaBuffer::alloc(N * 4);
        dw.upload(w.data(), N * BPR * sizeof(Q8Block)); dx.upload(x.data(), K * 4);
        void* pw = dw.device_ptr(); void* px = dx.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &pw, &px, &py, &N, &K };
        REQUIRE(gpu_launch(mod, "fused_gemv_q8_0_kernel", N, 256, args, &err));
        std::vector<float> got(N);
        REQUIRE(dy.download(got.data(), N * 4));
        for (uint32_t r = 0; r < N; ++r) CHECK(near(got[r], ref[r], 1e-3f));
    }
    {   // Q4_K: one 256-element super-block per row
        struct Q4KBlock {
            uint16_t d, dmin;
            uint8_t scales[12];
            uint8_t qs[128];
        };
        uint32_t N = 2, K = 256, BPR = K / 256;
        std::vector<Q4KBlock> w(N * BPR);
        std::vector<float> x(K), ref(N, 0.0f);
        for (uint32_t i = 0; i < K; ++i) x[i] = float(i % 7) * 0.2f - 0.6f;
        for (uint32_t r = 0; r < N; ++r) {
            float acc = 0;
            for (uint32_t b = 0; b < BPR; ++b) {
                Q4KBlock& blk = w[r * BPR + b];
                blk.d = 0x3800;    // 0.5
                blk.dmin = 0x3400; // 0.25
                std::memset(blk.scales, 0, 12);
                for (int j = 0; j < 4; ++j) { blk.scales[j] = 2; blk.scales[j + 4] = 1; }
                for (int j = 4; j < 8; ++j) blk.scales[j + 4] = 0x12;
                uint8_t sc[8], m[8];
                for (int j = 0; j < 8; ++j) {
                    if (j < 4) { sc[j] = blk.scales[j] & 0x3F; m[j] = blk.scales[j + 4] & 0x3F; }
                    else {
                        sc[j] = uint8_t((blk.scales[j + 4] & 0x0F) | ((blk.scales[j - 4] >> 6) << 4));
                        m[j] = uint8_t((blk.scales[j + 4] >> 4) | ((blk.scales[j] >> 6) << 4));
                    }
                }
                for (int p = 0; p < 4; ++p) {
                    int lo = 2 * p, hi = 2 * p + 1;
                    float wlo = float(sc[lo]) * 0.5f, whi = float(sc[hi]) * 0.5f;
                    float blo = float(m[lo]) * 0.25f, bhi = float(m[hi]) * 0.25f;
                    for (int l = 0; l < 32; ++l) {
                        uint8_t nlo = uint8_t((p * 8 + l) % 15);
                        uint8_t nhi = uint8_t((p * 4 + l * 2) % 15);
                        blk.qs[p * 32 + l] = uint8_t((nlo & 0x0F) | ((nhi & 0x0F) << 4));
                        acc += (wlo * float(nlo) - blo) * x[lo * 32 + l];
                        acc += (whi * float(nhi) - bhi) * x[hi * 32 + l];
                    }
                }
            }
            ref[r] = acc;
        }
        MlFusionCompiler c;
        std::string err;
        CudaModule mod = CudaModule::load(c.emit_ptx_fused_gemv_q4_k(), &err);
        REQUIRE(mod.valid());
        CudaBuffer dw = CudaBuffer::alloc(N * BPR * sizeof(Q4KBlock)), dx = CudaBuffer::alloc(K * 4), dy = CudaBuffer::alloc(N * 4);
        dw.upload(w.data(), N * BPR * sizeof(Q4KBlock)); dx.upload(x.data(), K * 4);
        void* pw = dw.device_ptr(); void* px = dx.device_ptr(); void* py = dy.device_ptr();
        std::vector<void*> args = { &pw, &px, &py, &N, &K };
        REQUIRE(gpu_launch(mod, "fused_gemv_q4_k_kernel", N, 256, args, &err));
        std::vector<float> got(N);
        REQUIRE(dy.download(got.data(), N * 4));
        for (uint32_t r = 0; r < N; ++r) CHECK(near(got[r], ref[r], 1e-3f));
    }
}

TEST_CASE("GPU - fused residual LayerNorm fast kernel executes") {
    if (!gpu_ready()) return;
    MlFusionCompiler c;
    std::string err;
    CudaModule mod = CudaModule::load(c.emit_ptx_fused_residual_layernorm(), &err);
    REQUIRE(mod.valid());

    uint32_t B = 2, D = 37;
    float eps = 1e-5f;
    std::vector<float> x(B * D), res(B * D), gamma(D), beta(D), y(B * D, 0.0f), ref(B * D);
    for (uint32_t i = 0; i < D; ++i) { gamma[i] = 1.0f + float(i % 3) * 0.1f; beta[i] = 0.05f * float(i % 4); }
    for (uint32_t r = 0; r < B; ++r) {
        for (uint32_t i = 0; i < D; ++i) {
            x[r * D + i] = float((r + i) % 9) * 0.2f - 0.8f;
            res[r * D + i] = float((r * 2 + i) % 5) * 0.1f;
        }
        float mean = 0;
        for (uint32_t i = 0; i < D; ++i) mean += x[r * D + i] + res[r * D + i];
        mean /= float(D);
        float var = 0;
        for (uint32_t i = 0; i < D; ++i) { float d = (x[r * D + i] + res[r * D + i]) - mean; var += d * d; }
        var /= float(D);
        float rstd = 1.0f / std::sqrt(var + eps);
        for (uint32_t i = 0; i < D; ++i)
            ref[r * D + i] = ((x[r * D + i] + res[r * D + i]) - mean) * rstd * gamma[i] + beta[i];
    }
    CudaBuffer dx = CudaBuffer::alloc(B * D * 4), dres = CudaBuffer::alloc(B * D * 4);
    CudaBuffer dg = CudaBuffer::alloc(D * 4), db = CudaBuffer::alloc(D * 4), dy = CudaBuffer::alloc(B * D * 4);
    dx.upload(x.data(), B * D * 4); dres.upload(res.data(), B * D * 4);
    dg.upload(gamma.data(), D * 4); db.upload(beta.data(), D * 4);
    void* px = dx.device_ptr(); void* pres = dres.device_ptr(); void* pg = dg.device_ptr();
    void* pb = db.device_ptr(); void* py = dy.device_ptr();
    float feps = eps;
    std::vector<void*> args = { &px, &pres, &pg, &pb, &py, &B, &D, &feps };
    REQUIRE(gpu_launch(mod, "fused_residual_layernorm_kernel", B, 128, args, &err));
    std::vector<float> got(B * D);
    REQUIRE(dy.download(got.data(), B * D * 4));
    for (size_t i = 0; i < got.size(); ++i) CHECK(near(got[i], ref[i], 2e-3f));
}

TEST_CASE("GPU - MIR integer ops (unsigned/64-bit/select/indexed) execute") {
    if (!gpu_ready()) return;
    std::string err;

    // --- unsigned compare + unsigned divide ---
    {
        Module mod("mir_uint");
        Function* fn = mod.create_function("uint_ops", Type::void_type(),
                                           {Type::i32(), Type::i32(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        b.add_block_param(e, Type::i32());
        b.add_block_param(e, Type::i32());
        b.add_block_param(e, Type::ptr());
        Value* lt = b.build_ult(e->param(0), e->param(1));
        Value* q = b.build_udiv(e->param(0), e->param(1));
        b.build_store(Type::i32(), e->param(2), 0, lt);
        b.build_store(Type::i32(), e->param(2), 4, q);
        b.build_ret_void();

        std::string ptx = PtxTarget::emit_function(*fn);
        if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
        CudaModule m = CudaModule::load(ptx, &err);
        REQUIRE(m.valid());

        uint32_t a = 0xFFFFFFF0u, c = 4u; // unsigned: a > c, but signed a < c
        CudaBuffer dout = CudaBuffer::alloc(8);
        void* po = dout.device_ptr();
        uint32_t a32 = a, c32 = c;
        std::vector<void*> args = { &a32, &c32, &po };
        REQUIRE(gpu_launch(m, "uint_ops", 1, 1, args, &err));
        uint32_t got[2] = {0, 0};
        REQUIRE(dout.download(got, 8));
        CHECK(got[0] == 0u);          // a < c is false
        CHECK(got[1] == a / c);       // unsigned division
    }

    // --- 64-bit bitwise + shifts ---
    {
        Module mod("mir_bits");
        // (a, mask, shift, out): the mask is independent of the shift count so
        // every result is non-trivial and the 64-bit width is actually exercised.
        Function* fn = mod.create_function("bits_ops", Type::void_type(),
                                           {Type::i64(), Type::i64(), Type::i64(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        b.add_block_param(e, Type::i64());
        b.add_block_param(e, Type::i64());
        b.add_block_param(e, Type::i64());
        b.add_block_param(e, Type::ptr());
        Value* an = b.build_and(e->param(0), e->param(1));
        Value* sh = b.build_shl(an, e->param(2));
        Value* lr = b.build_lshr(an, e->param(2));
        Value* ar = b.build_ashr(e->param(0), e->param(2));
        b.build_store(Type::i64(), e->param(3), 0, an);
        b.build_store(Type::i64(), e->param(3), 8, sh);
        b.build_store(Type::i64(), e->param(3), 16, lr);
        b.build_store(Type::i64(), e->param(3), 24, ar);
        b.build_ret_void();

        std::string ptx = PtxTarget::emit_function(*fn);
        if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
        CudaModule m = CudaModule::load(ptx, &err);
        REQUIRE(m.valid());

        uint64_t a = 0xF0F0F0F0F0F0F0F0ull, mask = 0x00FFFFFFFFFFFF00ull, s = 8;
        CudaBuffer dout = CudaBuffer::alloc(32);
        void* po = dout.device_ptr();
        std::vector<void*> args = { &a, &mask, &s, &po };
        REQUIRE(gpu_launch(m, "bits_ops", 1, 1, args, &err));
        uint64_t got[4] = {0, 0, 0, 0};
        REQUIRE(dout.download(got, 32));
        uint64_t and_ref = a & mask;                       // 0x00F0F0F0F0F0F000
        CHECK(got[0] == and_ref);
        CHECK(got[1] == (and_ref << s));                   // 0xF0F0F0F0F0F00000
        CHECK(got[2] == (and_ref >> s));                   // 0x0000F0F0F0F0F0F0
        CHECK(got[3] == (uint64_t)((int64_t)a >> s));      // 0xFFF0F0F0F0F0F0F0
    }

    // --- select with a non-comparison condition + indexed memory ---
    {
        Module mod("mir_sel_idx");
        Function* fn = mod.create_function("sel_idx", Type::void_type(),
                                           {Type::ptr(), Type::i64(), Type::i32(), Type::i32(), Type::i32(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        b.add_block_param(e, Type::ptr());
        b.add_block_param(e, Type::i64());
        b.add_block_param(e, Type::i32());
        b.add_block_param(e, Type::i32());
        b.add_block_param(e, Type::i32());
        b.add_block_param(e, Type::ptr());
        Value* v = b.build_load_indexed(Type::i32(), e->param(0), e->param(1), 4, 0);
        Value* sel = b.build_select(e->param(2), e->param(3), e->param(4));
        b.build_store_indexed(Type::i32(), e->param(5), e->param(1), 4, 0, sel);
        (void)v;
        b.build_ret_void();

        std::string ptx = PtxTarget::emit_function(*fn);
        if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
        CudaModule m = CudaModule::load(ptx, &err);
        REQUIRE(m.valid());

        std::vector<int32_t> base = {10, 20, 30, 40, 50, 60};
        int64_t idx = 3;
        CudaBuffer din = CudaBuffer::alloc(base.size() * 4), dout = CudaBuffer::alloc(base.size() * 4);
        din.upload(base.data(), base.size() * 4);
        void* pin = din.device_ptr(); void* po = dout.device_ptr();
        int32_t cond = 5, tval = 111, fval = 222;
        std::vector<void*> args = { &pin, &idx, &cond, &tval, &fval, &po };
        REQUIRE(gpu_launch(m, "sel_idx", 1, 1, args, &err));
        std::vector<int32_t> got(base.size(), 0);
        REQUIRE(dout.download(got.data(), base.size() * 4));
        CHECK(got[idx] == tval); // cond != 0 -> true value
    }
}

TEST_CASE("GPU - public C ABI executes a kernel") {
    if (!brass_gpu_available()) {
        report_skip("brass_gpu_available() == 0: C ABI test not executed on device");
        return;
    }
    char name[256] = {0};
    if (brass_gpu_device_count() > 0) {
        CHECK(brass_gpu_device_name(0, name, sizeof(name)) == BRASS_OK);
        CHECK(std::strlen(name) > 0);
    }

    MlFusionCompiler c;
    std::string ptx = c.emit_ptx_q8_dot();
    BrassGpuModule mod = nullptr;
    REQUIRE(brass_gpu_module_load(ptx.c_str(), &mod) == BRASS_OK);
    REQUIRE(mod != nullptr);
    CHECK(brass_gpu_module_has_function(mod, "fused_q8_dot") == 1);

    const int n = 64;
    std::vector<int8_t> a(n), b(n);
    int32_t ref = 0;
    for (int i = 0; i < n; ++i) {
        a[i] = int8_t((i * 7) % 256 - 128);
        b[i] = int8_t((i * 11) % 256 - 128);
        ref += int32_t(a[i]) * int32_t(b[i]);
    }
    float sa = 0.05f, sb = 0.02f;
    float ref_out = float(ref) * sa * sb;

    BrassGpuBuffer da = brass_gpu_buffer_alloc(n), db = brass_gpu_buffer_alloc(n);
    BrassGpuBuffer dout = brass_gpu_buffer_alloc(4);
    REQUIRE(da && db && dout);
    REQUIRE(brass_gpu_buffer_upload(da, a.data(), n, 0) == BRASS_OK);
    REQUIRE(brass_gpu_buffer_upload(db, b.data(), n, 0) == BRASS_OK);

    void* pa = brass_gpu_buffer_device_ptr(da);
    void* pb = brass_gpu_buffer_device_ptr(db);
    void* po = brass_gpu_buffer_device_ptr(dout);
    uint64_t nn = n;
    void* args[] = { &pa, &pb, &sa, &sb, &po, &nn };
    REQUIRE(brass_gpu_module_launch(mod, "fused_q8_dot", 1, 1, 1, 1, 1, 1, args, 0) == BRASS_OK);
    REQUIRE(brass_gpu_synchronize() == BRASS_OK);
    float got = 0;
    REQUIRE(brass_gpu_buffer_download(dout, &got, 4, 0) == BRASS_OK);
    CHECK(near(got, ref_out, 1e-4f));

    brass_gpu_buffer_destroy(da);
    brass_gpu_buffer_destroy(db);
    brass_gpu_buffer_destroy(dout);
    brass_gpu_module_destroy(mod);
}

} // namespace
