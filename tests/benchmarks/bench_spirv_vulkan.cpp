// brass_spirv_bench: SpirvTarget kernels vs the same kernels hand-written in
// GLSL (tests/benchmarks/spirv/*.comp, compiled by glslc at build time),
// launched through the same brass::gpu::VulkanModule with the same
// push-constant ABI, reporting effective bandwidth. It exists to catch
// pathological codegen (scalar where GLSL vectorizes, lost alignment, extra
// address arithmetic in the loop); the numbers are recorded in
// docs/spirv_backend_design.md ("Measured").
//
//   brass_spirv_bench [--reps N]
//
// SwiGLU over 64M floats (reads gate + up, writes out: 768 MiB per run) and
// residual RMSNorm over 4096 x 4096 (reads x, res; writes x, y: 256 MiB per
// run, counting the pass-2 re-read of x as a cache hit). Each kernel's
// output is checked against the other's before timing. Also reports the
// subgroup size the driver actually used per requested size.

#include <brass/codegen/ml_fusion.hpp>
#include <brass/gpu/vulkan_driver.hpp>
#include <brass/target/spirv_target.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef BRASS_SPIRV_BENCH_SHADER_DIR
#define BRASS_SPIRV_BENCH_SHADER_DIR ""
#endif

using namespace brass;
using brass::gpu::VulkanArg;
using brass::gpu::VulkanBuffer;
using brass::gpu::VulkanDispatch;
using brass::gpu::VulkanModule;
using brass::gpu::VulkanModuleOptions;
using codegen::GpuKernel;

namespace {

bool read_spv(const std::string& name, std::vector<uint32_t>& words) {
    std::string dir = BRASS_SPIRV_BENCH_SHADER_DIR;
    if (dir.empty()) return false;
    std::ifstream f(dir + "/" + name + ".spv", std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    f.seekg(0);
    words.resize(static_cast<size_t>(n) / 4);
    return static_cast<bool>(f.read(reinterpret_cast<char*>(words.data()), n));
}

// A glslc module with the push-constant layout of a brass kernel (same
// parameter list), so one launch path drives both.
bool glsl_kernel(const std::string& name, const target::SpirvKernel& like, target::SpirvKernel& out) {
    out = like;
    out.entry = "main";
    out.capabilities = {"Shader", "PhysicalStorageBufferAddresses", "Int64", "GroupNonUniform"};
    out.extensions.clear();
    return read_spv(name, out.words);
}

VulkanModule load_or_die(const target::SpirvKernel& k, uint32_t subgroup = 32) {
    std::string err;
    VulkanModuleOptions mo;
    mo.subgroup_size = subgroup;
    VulkanModule m = VulkanModule::load(k, &err, mo);
    if (!m.valid()) {
        std::fprintf(stderr, "load %s: %s\n", k.entry.c_str(), err.c_str());
        std::exit(1);
    }
    return m;
}

// Median per-dispatch milliseconds over `trials` command buffers of `reps` dispatches.
double time_ms(const VulkanModule& m, uint32_t grid, uint32_t block, const std::vector<VulkanArg>& args, uint32_t reps) {
    std::vector<double> t;
    for (int trial = 0; trial < 5; ++trial) {
        double ms = 0;
        VulkanDispatch d;
        d.grid[0] = grid;
        d.block[0] = block;
        d.repeat = reps;
        d.gpu_ms = &ms;
        std::string err;
        if (!m.launch(d, args, &err)) {
            std::fprintf(stderr, "launch: %s\n", err.c_str());
            std::exit(1);
        }
        t.push_back(ms / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

std::vector<float> fetch(const VulkanBuffer& b, size_t n) {
    std::vector<float> v(n);
    if (!b.download(v.data(), n * 4)) std::exit(1);
    return v;
}

float max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0;
    for (size_t i = 0; i < a.size(); ++i) w = std::max(w, std::fabs(a[i] - b[i]) / (1.0f + std::fabs(b[i])));
    return w;
}

void report(const char* what, uint32_t block, double ms, double bytes) {
    std::printf("  %-28s block %4u  %8.3f ms  %7.1f GB/s\n", what, block, ms, bytes / (ms * 1e-3) / 1e9);
}

} // namespace

int main(int argc, char** argv) {
    uint32_t reps = 20;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = static_cast<uint32_t>(std::atoi(argv[++i]));
    if (!gpu::vulkan_available()) {
        std::printf("[SKIP] no Vulkan device: %s\n", gpu::vulkan_last_error().c_str());
        return 0;
    }
    const gpu::VulkanDeviceCaps& c = *gpu::vulkan_device_caps();
    std::printf("device: %s (%s %s), subgroup %u (%u..%u, requirable in compute: %s)\n", c.info.name.c_str(),
                c.driver_name.c_str(), c.driver_info.c_str(), c.subgroup_size, c.min_subgroup_size, c.max_subgroup_size,
                c.can_require_subgroup_size_in_compute ? "yes" : "no");
    codegen::MlFusionCompiler mc;

    // Which wave size a pipeline really gets.
    {
        target::SpirvKernel probe;
        probe.params = {{Type::ptr(), 0, 8}};
        probe.push_constant_bytes = 8;
        probe.local_size[0] = 64;
        if (glsl_kernel("subgroup_size", probe, probe)) {
            VulkanBuffer out = VulkanBuffer::alloc(4);
            for (uint32_t want : {0u, 32u, 64u}) {
                VulkanModule m = load_or_die(probe, want);
                for (uint32_t block : {64u, 256u}) {
                    std::string err;
                    m.launch_1d(1, block, {out}, &err);
                    uint32_t sg = 0;
                    out.download(&sg, 4);
                    std::printf("  subgroup size requested %-6s block %3u -> gl_SubgroupSize %u\n",
                                want ? std::to_string(want).c_str() : "driver", block, sg);
                }
            }
        } else {
            std::printf("  (glslc shaders not built: GLSL comparisons skipped)\n");
        }
    }

    // ---- SwiGLU, 64M floats --------------------------------------------
    {
        const uint32_t n = 64u << 20;
        std::vector<float> host(n);
        for (uint32_t i = 0; i < n; ++i) host[i] = static_cast<float>(i % 1999) * 0.004f - 4.0f;
        VulkanBuffer g = VulkanBuffer::alloc(size_t{n} * 4), u = VulkanBuffer::alloc(size_t{n} * 4);
        VulkanBuffer o1 = VulkanBuffer::alloc(size_t{n} * 4), o2 = VulkanBuffer::alloc(size_t{n} * 4);
        if (!g.upload(host.data(), size_t{n} * 4) || !u.upload(host.data(), size_t{n} * 4)) return 1;
        target::SpirvKernel bk = mc.compile_spirv(GpuKernel::swiglu);
        target::SpirvKernel gk;
        bool have_glsl = glsl_kernel("swiglu", bk, gk);
        VulkanModule bm = load_or_die(bk);
        std::printf("\nSwiGLU, %u floats (768 MiB moved per run)\n", n);
        const double bytes = 3.0 * n * 4;
        for (uint32_t block : {128u, 256u, 512u}) {
            uint32_t grid = (n / 4 + block - 1) / block;
            report("brass SPIR-V", block, time_ms(bm, grid, block, {g, u, o1, n}, reps), bytes);
            if (have_glsl) {
                VulkanModule gm = load_or_die(gk);
                report("GLSL (glslc -O)", block, time_ms(gm, grid, block, {g, u, o2, n}, reps), bytes);
            }
        }
        if (have_glsl) std::printf("  outputs agree to %.2g\n", static_cast<double>(max_rel(fetch(o1, n), fetch(o2, n))));
    }

    // ---- residual RMSNorm, 4096 x 4096 ----------------------------------
    {
        const uint32_t rows = 4096, d = 4096;
        const size_t n = size_t{rows} * d;
        std::vector<float> hx(n), hr(n), hg(d);
        for (size_t i = 0; i < n; ++i) {
            hx[i] = static_cast<float>(i % 7) * 0.25f - 0.75f;
            hr[i] = static_cast<float>(i % 5) * 0.01f;
        }
        for (uint32_t i = 0; i < d; ++i) hg[i] = 1.0f + static_cast<float>(i % 3) * 0.1f;
        VulkanBuffer x1 = VulkanBuffer::alloc(n * 4), x2 = VulkanBuffer::alloc(n * 4), r = VulkanBuffer::alloc(n * 4);
        VulkanBuffer gm_ = VulkanBuffer::alloc(d * 4), y1 = VulkanBuffer::alloc(n * 4), y2 = VulkanBuffer::alloc(n * 4);
        if (!x1.upload(hx.data(), n * 4) || !x2.upload(hx.data(), n * 4) || !r.upload(hr.data(), n * 4) ||
            !gm_.upload(hg.data(), d * 4))
            return 1;
        target::SpirvKernel bk = mc.compile_spirv(GpuKernel::residual_rms_norm);
        target::SpirvKernel gk;
        bool have_glsl = glsl_kernel("rms_norm", bk, gk);
        VulkanModule bm = load_or_die(bk);
        const float eps = 1e-5f;
        // One run each for the correctness cross-check (x is updated in place).
        bm.launch_1d(rows, 256, {x1, r, gm_, y1, rows, d, eps});
        if (have_glsl) {
            VulkanModule gmod = load_or_die(gk);
            gmod.launch_1d(rows, 256, {x2, r, gm_, y2, rows, d, eps});
            std::printf("\nresidual RMSNorm %ux%u: outputs agree to %.2g\n", rows, d,
                        static_cast<double>(max_rel(fetch(y1, n), fetch(y2, n))));
        } else {
            std::printf("\nresidual RMSNorm %ux%u\n", rows, d);
        }
        const double bytes = 4.0 * static_cast<double>(n) * 4;
        for (uint32_t block : {128u, 256u, 512u, 1024u}) {
            report("brass SPIR-V", block, time_ms(bm, rows, block, {x1, r, gm_, y1, rows, d, eps}, reps), bytes);
            if (c.can_require_subgroup_size_in_compute && c.max_subgroup_size >= 64) {
                VulkanModule b64 = load_or_die(bk, 64);
                report("brass SPIR-V, subgroup 64", block, time_ms(b64, rows, block, {x1, r, gm_, y1, rows, d, eps}, reps), bytes);
            }
            if (have_glsl) {
                VulkanModule gmod = load_or_die(gk);
                report("GLSL (glslc -O)", block, time_ms(gmod, rows, block, {x2, r, gm_, y2, rows, d, eps}, reps), bytes);
            }
        }
    }
    return 0;
}
