#pragma once

// Shared helpers for the PTX backend tests (test_ptx_intrinsics.cpp,
// test_ptx_vector.cpp, test_ptx_cleanup.cpp, test_gpu_kernels*.cpp): ptxas /
// device availability with visible [SKIP] lines, lower+verify of a MIR
// function, upload/launch/download, and a small harness for kernels that map
// input buffers to an output buffer.

#include "test_framework.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/gpu/cuda_driver.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>
#include <brass/target/ptx_target.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace ptxtest {

using brass::gpu::CudaBuffer;
using brass::gpu::CudaModule;

inline void report_skip(const char* what) {
    std::cout << "  [SKIP] " << what << "\n" << std::flush;
}

inline const char* null_device() {
#if defined(_WIN32)
    return "NUL";
#else
    return "/dev/null";
#endif
}

inline std::string quiet(const std::string& cmd) {
    return cmd + " >" + null_device() + " 2>&1";
}

inline std::filesystem::path scratch_path(const char* name) {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = std::filesystem::current_path();
    return dir / name;
}

inline bool ptxas_available() {
    static const bool ok = (std::system(quiet("ptxas --version").c_str()) == 0);
    if (!ok) report_skip("ptxas not on PATH: assembly not validated");
    return ok;
}

inline bool ptxas_assembles(const std::string& ptx, const char* arch = "sm_70") {
    std::filesystem::path in = scratch_path("brass_ptx_check.ptx");
    std::filesystem::path out = scratch_path("brass_ptx_check.cubin");
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

inline bool gpu_ready() {
    static const bool ok = brass::gpu::cuda_available();
    if (!ok) report_skip(("CUDA unavailable (" + brass::gpu::cuda_last_error() + "): test not executed on device").c_str());
    return ok;
}

// Lower with PtxISel only (no cleanup), verify (dumping diagnostics on
// failure) and return the IR.
inline brass::ptx::Function lower_ok(const brass::Function& fn) {
    brass::ptx::PtxISel isel;
    brass::ptx::Function out = isel.lower(fn);
    auto diags = brass::ptx::verify(out);
    if (!diags.empty()) std::cerr << brass::ptx::format_diagnostics(diags) << brass::ptx::print_body(out);
    REQUIRE(diags.empty());
    return out;
}

// Defined in test_ptx_intrinsics.cpp: a kernel (i32, f32, f64, i64, ptr)
// that calls the intrinsic `name` per its signature table and stores any
// result through the pointer. Throws for a name the table does not know.
brass::Function* build_intrinsic_kernel(brass::Module& mod, const std::string& name);

// emit_function + ptxas check (when available). Returns the PTX text.
inline std::string emit_checked(const brass::Function& fn) {
    std::string ptx = brass::target::PtxTarget::emit_function(fn);
    if (ptxas_available()) {
        bool ok = ptxas_assembles(ptx);
        if (!ok) std::cerr << "ptxas rejected:\n" << ptx;
        CHECK(ok);
    }
    return ptx;
}

inline bool near(float a, float b, float rel) {
    return std::fabs(a - b) <= rel * (1.0f + std::fabs(b));
}

// A device buffer initialised from a host vector.
template <typename T>
CudaBuffer upload(const std::vector<T>& host) {
    CudaBuffer buf = CudaBuffer::alloc(host.size() * sizeof(T));
    REQUIRE(buf.valid());
    REQUIRE(buf.upload(host.data(), host.size() * sizeof(T)));
    return buf;
}

template <typename T>
std::vector<T> download(const CudaBuffer& buf, size_t n) {
    std::vector<T> host(n);
    REQUIRE(buf.download(host.data(), n * sizeof(T)));
    return host;
}

// Load a module and launch `entry` (grid x block) with pointer/scalar args.
inline void launch(const std::string& ptx, const char* entry, uint32_t grid, uint32_t block, std::vector<void*> args) {
    std::string err;
    CudaModule mod = CudaModule::load(ptx, &err);
    if (!mod.valid()) std::cerr << "driver JIT: " << err << "\n" << ptx;
    REQUIRE(mod.valid());
    bool ok = mod.launch_1d(entry, grid, block, args.data(), 0, &err);
    if (!ok) std::cerr << "launch: " << err << "\n";
    REQUIRE(ok);
}

// MIR type for a host element type.
template <typename T> brass::Type mir_type();
template <> inline brass::Type mir_type<float>()    { return brass::Type::f32(); }
template <> inline brass::Type mir_type<double>()   { return brass::Type::f64(); }
template <> inline brass::Type mir_type<int32_t>()  { return brass::Type::i32(); }
template <> inline brass::Type mir_type<uint32_t>() { return brass::Type::i32(); }
template <> inline brass::Type mir_type<int64_t>()  { return brass::Type::i64(); }
template <> inline brass::Type mir_type<uint64_t>() { return brass::Type::i64(); }

// Builds  kernel(a, b, out, n): i = global_tid_x; if (i < n) out[i] = f(a[i], b[i])
// with one element per thread, runs it over the inputs on device and returns
// the outputs. `f` receives the KernelBuilder and the two loaded elements.
struct MapKernel {
    brass::Module mod{"map"};
    brass::Function* fn = nullptr;

    MapKernel(brass::Type ta, brass::Type tb, brass::Type tr,
              const std::function<brass::Value*(brass::codegen::KernelBuilder&, brass::Value*, brass::Value*)>& f) {
        using namespace brass;
        fn = mod.create_function("map_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
        codegen::KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::ptr());
        Value* bb = b.add_block_param(entry, Type::ptr());
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        BasicBlock* body = b.append_block("body");
        BasicBlock* done = b.append_block("done");

        b.position_at_end(entry);
        Value* i = kb.global_tid_x();
        b.build_br_if(b.build_slt(i, n), body, done);

        b.position_at_end(body);
        Value* x = b.build_load_indexed(ta, a, i, static_cast<uint8_t>(ta.size_in_bytes()));
        Value* y = b.build_load_indexed(tb, bb, i, static_cast<uint8_t>(tb.size_in_bytes()));
        Value* r = f(kb, x, y);
        b.build_store_indexed(tr, out, i, static_cast<uint8_t>(tr.size_in_bytes()), r);
        b.build_br(done);

        b.position_at_end(done);
        b.build_ret_void();
    }
};

template <typename A, typename B, typename R>
std::vector<R> run_map(const std::function<brass::Value*(brass::codegen::KernelBuilder&, brass::Value*, brass::Value*)>& f,
                       const std::vector<A>& a, const std::vector<B>& b) {
    REQUIRE(a.size() == b.size());
    MapKernel k(mir_type<A>(), mir_type<B>(), mir_type<R>(), f);
    std::string ptx = emit_checked(*k.fn);
    if (!gpu_ready()) return {};

    CudaBuffer da = upload(a);
    CudaBuffer db = upload(b);
    CudaBuffer dout = CudaBuffer::alloc(a.size() * sizeof(R));
    REQUIRE(dout.valid() && dout.zero());
    void* pa = da.device_ptr();
    void* pb = db.device_ptr();
    void* po = dout.device_ptr();
    int32_t n = static_cast<int32_t>(a.size());
    uint32_t block = 128;
    uint32_t grid = static_cast<uint32_t>((a.size() + block - 1) / block);
    launch(ptx, "map_kernel", grid, block, {&pa, &pb, &po, &n});
    return download<R>(dout, a.size());
}

// Host f16 <-> f32 (round-to-nearest-even, handles subnormals/inf/nan).
inline float f16_to_f32_host(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | ((127u - 15u - static_cast<uint32_t>(e)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_f16_host(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) { // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign); // underflow -> 0
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half; // may carry into exponent, which is correct
    return static_cast<uint16_t>(sign | half);
}

} // namespace ptxtest
