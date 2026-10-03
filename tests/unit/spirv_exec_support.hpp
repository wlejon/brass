#pragma once

// Helpers for the on-device SPIR-V tests (test_spirv_execution*.cpp,
// test_spirv_kernels.cpp): Vulkan availability with a visible [SKIP] line,
// compile (+ spirv-val) + load + launch, buffer upload/download, and the
// one-thread-per-element map kernel.

#include "spirv_test_support.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/gpu/vulkan_driver.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace spvexec {

using brass::gpu::VulkanArg;
using brass::gpu::VulkanBuffer;
using brass::gpu::VulkanDispatch;
using brass::gpu::VulkanModule;
using brass::gpu::VulkanModuleOptions;

// True when a Vulkan device is open; prints the device once, and a [SKIP]
// line on every call without one.
inline bool vk_ready() {
    static const bool ok = [] {
        bool a = brass::gpu::vulkan_available();
        if (a) {
            const brass::gpu::VulkanDeviceCaps* c = brass::gpu::vulkan_device_caps();
            std::printf("  [vulkan] %s (%s, driver %s), subgroup %u (%u..%u)\n", c->info.name.c_str(),
                        c->info.type.c_str(), c->driver_name.c_str(), c->subgroup_size, c->min_subgroup_size,
                        c->max_subgroup_size);
        }
        return a;
    }();
    if (!ok) spvtest::report_skip(("no Vulkan device (" + brass::gpu::vulkan_last_error() + "): not executed").c_str());
    return ok;
}

inline const brass::gpu::VulkanDeviceCaps& caps() { return *brass::gpu::vulkan_device_caps(); }

template <typename T>
VulkanBuffer upload(const std::vector<T>& host) {
    VulkanBuffer buf = VulkanBuffer::alloc(host.size() * sizeof(T));
    if (!buf.valid()) std::cerr << "alloc: " << brass::gpu::vulkan_last_error() << "\n";
    REQUIRE(buf.valid());
    REQUIRE(buf.upload(host.data(), host.size() * sizeof(T)));
    return buf;
}

// A zero-filled buffer of `n` elements of T.
template <typename T>
VulkanBuffer zeros(size_t n) {
    VulkanBuffer buf = VulkanBuffer::alloc(n * sizeof(T));
    REQUIRE(buf.valid());
    REQUIRE(buf.zero());
    return buf;
}

template <typename T>
std::vector<T> download(const VulkanBuffer& buf, size_t n) {
    std::vector<T> host(n);
    REQUIRE(buf.download(host.data(), n * sizeof(T)));
    return host;
}

// compile_checked (ISel + verify + spirv-val) then load on the device.
inline VulkanModule load(const brass::Function& fn, const VulkanModuleOptions& mo = {},
                         const brass::target::SpirvOptions& so = {}) {
    brass::target::SpirvKernel k = spvtest::compile_checked(fn, so);
    std::string err;
    VulkanModule m = VulkanModule::load(k, &err, mo);
    if (!m.valid()) std::cerr << "VulkanModule::load: " << err << "\n";
    REQUIRE(m.valid());
    return m;
}

inline VulkanModule load_kernel(const brass::target::SpirvKernel& k, const VulkanModuleOptions& mo = {}) {
    std::string err;
    VulkanModule m = VulkanModule::load(k, &err, mo);
    if (!m.valid()) std::cerr << "VulkanModule::load: " << err << "\n";
    REQUIRE(m.valid());
    return m;
}

inline void launch(const VulkanModule& m, uint32_t grid, uint32_t block, std::vector<VulkanArg> args) {
    VulkanDispatch d;
    d.grid[0] = grid;
    d.block[0] = block;
    std::string err;
    bool ok = m.launch(d, args, &err);
    if (!ok) std::cerr << "launch: " << err << "\n";
    REQUIRE(ok);
}

inline uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline uint64_t f64_bits(double f) { uint64_t u; std::memcpy(&u, &f, 8); return u; }

inline bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * (1.0 + std::fabs(b)); }

// MIR type for a host element type.
template <typename T> brass::Type mir_type();
template <> inline brass::Type mir_type<float>()    { return brass::Type::f32(); }
template <> inline brass::Type mir_type<double>()   { return brass::Type::f64(); }
template <> inline brass::Type mir_type<int32_t>()  { return brass::Type::i32(); }
template <> inline brass::Type mir_type<uint32_t>() { return brass::Type::i32(); }
template <> inline brass::Type mir_type<int64_t>()  { return brass::Type::i64(); }
template <> inline brass::Type mir_type<uint64_t>() { return brass::Type::i64(); }

using MapFn = std::function<brass::Value*(brass::codegen::KernelBuilder&, brass::Value*, brass::Value*)>;

// kernel(a, b, out, i32 n): i = global_tid_x; if (i < n) out[i] = f(a[i], b[i]),
// run over the inputs on device; returns out.
template <typename A, typename B, typename R>
std::vector<R> run_map(const MapFn& f, const std::vector<A>& a, const std::vector<B>& b) {
    using namespace brass;
    REQUIRE(a.size() == b.size());
    Module mod("map");
    Function* fn = mod.create_function("map_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    codegen::KernelBuilder kb(mod, fn);
    Builder& bd = kb.builder();
    BasicBlock* entry = bd.append_block("entry");
    Value* pa = bd.add_block_param(entry, Type::ptr());
    Value* pb = bd.add_block_param(entry, Type::ptr());
    Value* po = bd.add_block_param(entry, Type::ptr());
    Value* n = bd.add_block_param(entry, Type::i32());
    BasicBlock* body = bd.append_block("body");
    BasicBlock* done = bd.append_block("done");
    bd.position_at_end(entry);
    Value* i = kb.global_tid_x();
    bd.build_br_if(bd.build_slt(i, n), body, done);
    bd.position_at_end(body);
    Type ta = mir_type<A>(), tb = mir_type<B>(), tr = mir_type<R>();
    Value* x = bd.build_load_indexed(ta, pa, i, static_cast<uint8_t>(ta.size_in_bytes()));
    Value* y = bd.build_load_indexed(tb, pb, i, static_cast<uint8_t>(tb.size_in_bytes()));
    bd.build_store_indexed(tr, po, i, static_cast<uint8_t>(tr.size_in_bytes()), f(kb, x, y));
    bd.build_br(done);
    bd.position_at_end(done);
    bd.build_ret_void();

    VulkanModule m = load(*fn);
    VulkanBuffer da = upload(a), db = upload(b);
    VulkanBuffer dout = zeros<R>(a.size());
    uint32_t block = 64;
    uint32_t grid = static_cast<uint32_t>((a.size() + block - 1) / block);
    launch(m, grid, block, {da, db, dout, static_cast<int32_t>(a.size())});
    return download<R>(dout, a.size());
}

} // namespace spvexec
