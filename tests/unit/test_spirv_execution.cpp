// SPIR-V kernels executed on a Vulkan device (brass::gpu::VulkanModule),
// checked against values computed on the host: the device and the runtime
// (features enabled, push-constant packing of every parameter type), scalar
// ALU over i32/i64/f32/f64 including the edge semantics of
// docs/spirv_backend_design.md (MIN / -1 wraps, MIN % -1 is 0, unordered
// float `ne`, NMin/NMax with NaN, RoundEven), comparisons, conversions, and
// the control-flow shapes the structurizer produces. Memory, vectors,
// shared memory, subgroups, atomics and the intrinsics are in
// test_spirv_execution_mem.cpp. Without a Vulkan device every test prints a
// visible [SKIP] and passes.

#include "spirv_exec_support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace brass;
using namespace spvexec;
using codegen::KernelBuilder;

namespace {

constexpr int32_t kMin32 = std::numeric_limits<int32_t>::min();
constexpr int64_t kMin64 = std::numeric_limits<int64_t>::min();
const float kNaN = std::numeric_limits<float>::quiet_NaN();
const double kNaN64 = std::numeric_limits<double>::quiet_NaN();

template <typename T>
using HostOp = std::function<T(T, T)>;

template <typename T>
struct BinOp {
    const char* name;
    MapFn mir;
    HostOp<T> host;
};

template <typename T>
void check_binops(const char* type, const std::vector<BinOp<T>>& ops, const std::vector<T>& a, const std::vector<T>& b,
                  double tol = 0.0) {
    for (const BinOp<T>& op : ops) {
        std::vector<T> got = run_map<T, T, T>(op.mir, a, b);
        for (size_t i = 0; i < a.size(); ++i) {
            T want = op.host(a[i], b[i]);
            if constexpr (std::is_floating_point_v<T>) {
                // Without SignedZeroInfNanPreserve, Vulkan leaves NaN results
                // undefined (RADV's f64 division by NaN gives 0); only NMin /
                // NMax define them.
                bool nan_in = std::isnan(a[i]) || std::isnan(b[i]);
                if (nan_in && std::string(op.name) != "fmin" && std::string(op.name) != "fmax") continue;
            }
            bool ok;
            if constexpr (std::is_floating_point_v<T>) {
                ok = (std::isnan(want) && std::isnan(got[i])) || near(got[i], want, tol);
            } else {
                ok = got[i] == want;
            }
            if (!ok) {
                std::cerr << type << " " << op.name << "(" << +a[i] << ", " << +b[i] << "): device " << +got[i]
                          << " host " << +want << "\n";
            }
            CHECK(ok);
        }
    }
}

} // namespace

TEST_CASE("SPIR-V exec - the Vulkan device opens with the features the target needs") {
    if (!vk_ready()) return;
    const brass::gpu::VulkanDeviceCaps& c = caps();
    std::printf("    driver %s; int64 %d f64 %d 8bit %d 16bit %d shuffle %d f32 atomic add %d/%d i64 atomics %d "
                "clock %d sg-control %d (require in compute %d, full %d); push %u B, shared %u B, ts %.3g ns\n",
                c.driver_info.c_str(), c.shader_int64, c.shader_float64, c.storage_buffer_8bit, c.storage_buffer_16bit,
                c.subgroup_shuffle, c.buffer_float32_atomic_add, c.shared_float32_atomic_add, c.buffer_int64_atomics,
                c.subgroup_clock, c.subgroup_size_control, c.can_require_subgroup_size_in_compute,
                c.compute_full_subgroups, c.max_push_constant_bytes, c.max_shared_bytes, c.timestamp_period_ns);
    CHECK(c.buffer_device_address);
    CHECK(c.shader_int64);
    CHECK(c.max_push_constant_bytes >= 128u);
    CHECK(!brass::gpu::vulkan_devices().empty());

    VulkanBuffer buf = VulkanBuffer::alloc(10);
    REQUIRE(buf.valid());
    CHECK_EQ(buf.size(), 12u);
    CHECK(buf.device_address() != 0);
    CHECK(buf.device_address() % 16 == 0);
    std::vector<uint8_t> bytes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    REQUIRE(buf.upload(bytes.data(), 12));
    std::vector<uint8_t> mid(5);
    REQUIRE(buf.download(mid.data(), 5, 3));
    CHECK(mid == std::vector<uint8_t>({4, 5, 6, 7, 8}));
    CHECK(!buf.upload(bytes.data(), 12, 4)); // past the end
}

TEST_CASE("SPIR-V exec - a kernel's capabilities are checked against the device features before launch") {
    // No device needed: missing_for on a hand-made caps record.
    target::SpirvKernel k;
    k.entry = "needs";
    k.capabilities = {"Shader", "PhysicalStorageBufferAddresses", "Int64", "Float64", "AtomicFloat32AddEXT",
                      "ShaderClockKHR", "Float16"};
    k.push_constant_bytes = 200;
    k.shared_bytes = 1024;
    brass::gpu::VulkanDeviceCaps c;
    c.info.name = "TestGPU";
    c.buffer_device_address = c.shader_int64 = true;
    c.subgroup_clock = true;
    c.max_push_constant_bytes = 128;
    c.max_shared_bytes = 65536;
    std::string m = c.missing_for(k);
    CHECK(spvtest::has(m, "capability Float64 needs shaderFloat64, which TestGPU lacks"));
    CHECK(spvtest::has(m, "AtomicFloat32AddEXT needs shaderBufferFloat32AtomicAdd"));
    CHECK(spvtest::has(m, "capability Float16 is not one the Vulkan runtime knows"));
    CHECK(spvtest::has(m, "push constants take 200 bytes; the device allows 128"));
    CHECK(!spvtest::has(m, "Int64 needs"));
    CHECK(!spvtest::has(m, "ShaderClockKHR"));
    CHECK(!spvtest::has(m, "shared memory"));
    c.shader_float64 = c.buffer_float32_atomic_add = true;
    k.capabilities.pop_back();
    k.push_constant_bytes = 64;
    CHECK(c.missing_for(k).empty());

    if (!vk_ready()) return;
    // On the device: an unknown capability is refused at load with the reason.
    spvtest::Kernel kk({Type::ptr()});
    kk.b.build_ret_void();
    target::SpirvKernel sk = spvtest::compile_checked(*kk.fn);
    sk.capabilities.push_back("Float16");
    std::string err;
    VulkanModule mod = VulkanModule::load(sk, &err);
    CHECK(!mod.valid());
    CHECK(spvtest::has(err, "cannot run on"));
    CHECK(spvtest::has(err, "Float16"));
}

TEST_CASE("SPIR-V exec - every parameter type reaches the kernel through the push constants") {
    if (!vk_ready()) return;
    // k(i32 a, ptr out, f64 d, i32 b, i64 l, f32 x, ptr out2): the layout has
    // padding after a, b and x; each value is stored back for the host.
    spvtest::Kernel k({Type::i32(), Type::ptr(), Type::f64(), Type::i32(), Type::i64(), Type::f32(), Type::ptr()});
    Builder& b = k.b;
    b.build_store(Type::i32(), k.p(1), 0, k.p(0));
    b.build_store(Type::f64(), k.p(1), 8, k.p(2));
    b.build_store(Type::i32(), k.p(1), 16, k.p(3));
    b.build_store(Type::i64(), k.p(1), 24, k.p(4));
    b.build_store(Type::f32(), k.p(6), 0, k.p(5));
    b.build_ret_void();
    target::SpirvKernel sk = spvtest::compile_checked(*k.fn);
    REQUIRE_EQ(sk.params.size(), 7u);
    CHECK_EQ(sk.params[2].offset, 16u); // f64 after ptr at 8
    VulkanModule m = load_kernel(sk);
    VulkanBuffer out = zeros<uint64_t>(4), out2 = zeros<float>(1);
    launch(m, 1, 1, {int32_t{-7}, out, 2.5, uint32_t{0xDEADBEEFu}, int64_t{-1234567890123LL}, 0.75f, out2});
    std::vector<uint64_t> got = download<uint64_t>(out, 4);
    CHECK_EQ(static_cast<int32_t>(got[0]), -7);
    double d;
    std::memcpy(&d, &got[1], 8);
    CHECK_EQ(d, 2.5);
    CHECK_EQ(static_cast<uint32_t>(got[2]), 0xDEADBEEFu);
    CHECK_EQ(static_cast<int64_t>(got[3]), -1234567890123LL);
    CHECK_EQ(download<float>(out2, 1)[0], 0.75f);

    // Wrong argument count and kind are diagnosed, not launched.
    std::string err;
    VulkanDispatch dsp;
    dsp.block[0] = 1;
    CHECK(!m.launch(dsp, {int32_t{1}}, &err));
    CHECK(spvtest::has(err, "takes 7 arguments"));
    CHECK(!m.launch(dsp, {0.5f, out, 2.5, uint32_t{1}, int64_t{1}, 0.75f, out2}, &err));
    CHECK(spvtest::has(err, "argument 0"));
}

TEST_CASE("SPIR-V exec - a fixed LocalSize kernel and workgroup sizes from the spec constants") {
    if (!vk_ready()) return;
    // out[global_tid] = ntid * 1000 + tid
    auto build = [](spvtest::Kernel& k) {
        KernelBuilder kb(k.b);
        Value* g = kb.global_tid_x();
        kb.store_i32_indexed(k.p(0), g, k.b.build_add(k.b.build_mul(kb.ntid_x(), kb.const_i32(1000)), kb.tid_x()));
        k.b.build_ret_void();
    };
    for (uint32_t block : {1u, 32u, 64u, 96u, 256u, 1024u}) {
        spvtest::Kernel k({Type::ptr()});
        build(k);
        VulkanModule m = load(*k.fn);
        VulkanBuffer out = zeros<int32_t>(block * 3);
        launch(m, 3, block, {out});
        std::vector<int32_t> got = download<int32_t>(out, block * 3);
        for (uint32_t i = 0; i < block * 3; ++i) CHECK_EQ(got[i], static_cast<int32_t>(block * 1000 + i % block));
    }
    spvtest::Kernel k({Type::ptr()});
    build(k);
    target::SpirvOptions so;
    so.local_size_spec_constants = false;
    so.local_size_x = 64;
    VulkanModule m = load(*k.fn, {}, so);
    VulkanBuffer out = zeros<int32_t>(128);
    launch(m, 2, 64, {out});
    CHECK_EQ(download<int32_t>(out, 128)[127], 64 * 1000 + 63);
    std::string err;
    CHECK(!m.launch_1d(1, 32, {out}, &err)); // the fixed size must match
    CHECK(spvtest::has(err, "fixed LocalSize"));

    // SPIR-V 1.6 (Vulkan 1.3) modules run too when the device is 1.3.
    if (caps().info.api_version >= ((1u << 22) | (3u << 12))) {
        spvtest::Kernel k16({Type::ptr()});
        build(k16);
        target::SpirvOptions o16;
        o16.spirv_version = 0x00010600;
        VulkanModule m16 = load(*k16.fn, {}, o16);
        VulkanBuffer out16 = zeros<int32_t>(64);
        launch(m16, 1, 64, {out16});
        CHECK_EQ(download<int32_t>(out16, 64)[63], 64 * 1000 + 63);
    }
}

TEST_CASE("SPIR-V exec - 3-D grids and blocks: tid/ctaid/ntid/nctaid in every dimension") {
    if (!vk_ready()) return;
    // One record of 12 i32 per thread, at its flat global index.
    spvtest::Kernel k({Type::ptr()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* vals[12] = {kb.tid_x(), kb.tid_y(), kb.tid_z(), kb.ctaid_x(), kb.ctaid_y(), kb.ctaid_z(),
                       kb.ntid_x(), kb.ntid_y(), kb.ntid_z(), kb.nctaid_x(), kb.nctaid_y(), kb.nctaid_z()};
    auto lin = [&](Value* x, Value* y, Value* z, Value* nx, Value* ny) {
        return b.build_add(x, b.build_mul(nx, b.build_add(y, b.build_mul(ny, z))));
    };
    Value* block_lin = lin(vals[3], vals[4], vals[5], vals[9], vals[10]);
    Value* thread_lin = lin(vals[0], vals[1], vals[2], vals[6], vals[7]);
    Value* per_block = b.build_mul(vals[6], b.build_mul(vals[7], vals[8]));
    Value* flat = b.build_add(b.build_mul(block_lin, per_block), thread_lin);
    Value* rec = b.build_add(k.p(0), b.build_mul(b.build_zext_i64(flat), b.build_iconst_i64(48)));
    for (int i = 0; i < 12; ++i) b.build_store(Type::i32(), rec, i * 4, vals[i]);
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    VulkanDispatch d;
    uint32_t grid[3] = {3, 2, 2}, block[3] = {4, 2, 3};
    std::copy(grid, grid + 3, d.grid);
    std::copy(block, block + 3, d.block);
    uint32_t total = 3 * 2 * 2 * 4 * 2 * 3;
    VulkanBuffer out = zeros<int32_t>(total * 12);
    std::string err;
    REQUIRE(m.launch(d, {out}, &err));
    std::vector<int32_t> got = download<int32_t>(out, total * 12);
    uint32_t idx = 0;
    for (uint32_t bz = 0; bz < 2; ++bz)
        for (uint32_t by = 0; by < 2; ++by)
            for (uint32_t bx = 0; bx < 3; ++bx)
                for (uint32_t tz = 0; tz < 3; ++tz)
                    for (uint32_t ty = 0; ty < 2; ++ty)
                        for (uint32_t tx = 0; tx < 4; ++tx, ++idx) {
                            const int32_t* r = &got[idx * 12];
                            int32_t want[12] = {int32_t(tx), int32_t(ty), int32_t(tz), int32_t(bx), int32_t(by),
                                                int32_t(bz), 4, 2, 3, 3, 2, 2};
                            bool ok = std::equal(want, want + 12, r);
                            if (!ok) std::cerr << "thread record " << idx << " wrong\n";
                            CHECK(ok);
                        }
}

TEST_CASE("SPIR-V exec - i32 ALU with the MIR edge semantics") {
    if (!vk_ready()) return;
    std::vector<int32_t> a = {7, -7, 7, -7, kMin32, kMin32, 0, 123456789, -1, 2147483647, 5, -100};
    std::vector<int32_t> b = {3, 3, -3, -3, -1, 1, 5, 1000, -1, 2, 31, 7};
    auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    auto s = [](uint32_t v) { return static_cast<int32_t>(v); };
    using KB = KernelBuilder;
    std::vector<BinOp<int32_t>> ops = {
        {"add", [](KB& k, Value* x, Value* y) { return k.builder().build_add(x, y); }, [&](int32_t x, int32_t y) { return s(u(x) + u(y)); }},
        {"sub", [](KB& k, Value* x, Value* y) { return k.builder().build_sub(x, y); }, [&](int32_t x, int32_t y) { return s(u(x) - u(y)); }},
        {"mul", [](KB& k, Value* x, Value* y) { return k.builder().build_mul(x, y); }, [&](int32_t x, int32_t y) { return s(u(x) * u(y)); }},
        {"sdiv", [](KB& k, Value* x, Value* y) { return k.builder().build_sdiv(x, y); },
         [](int32_t x, int32_t y) { return (x == kMin32 && y == -1) ? kMin32 : x / y; }},
        {"smod", [](KB& k, Value* x, Value* y) { return k.builder().build_smod(x, y); },
         [](int32_t x, int32_t y) { return (x == kMin32 && y == -1) ? 0 : x % y; }},
        {"udiv", [](KB& k, Value* x, Value* y) { return k.builder().build_udiv(x, y); }, [&](int32_t x, int32_t y) { return s(u(x) / u(y)); }},
        {"umod", [](KB& k, Value* x, Value* y) { return k.builder().build_umod(x, y); }, [&](int32_t x, int32_t y) { return s(u(x) % u(y)); }},
        {"neg", [](KB& k, Value* x, Value*) { return k.builder().build_neg(x); }, [&](int32_t x, int32_t) { return s(0u - u(x)); }},
        {"and", [](KB& k, Value* x, Value* y) { return k.builder().build_and(x, y); }, [](int32_t x, int32_t y) { return x & y; }},
        {"or", [](KB& k, Value* x, Value* y) { return k.builder().build_or(x, y); }, [](int32_t x, int32_t y) { return x | y; }},
        {"xor", [](KB& k, Value* x, Value* y) { return k.builder().build_xor(x, y); }, [](int32_t x, int32_t y) { return x ^ y; }},
        {"not", [](KB& k, Value* x, Value*) { return k.builder().build_not(x); }, [](int32_t x, int32_t) { return ~x; }},
        // shift amounts masked to 0..31 (the backend leaves >= width undefined)
        {"shl", [](KB& k, Value* x, Value* y) { auto& b = k.builder(); return b.build_shl(x, b.build_and(y, b.build_iconst_i32(31))); },
         [&](int32_t x, int32_t y) { return s(u(x) << (y & 31)); }},
        {"lshr", [](KB& k, Value* x, Value* y) { auto& b = k.builder(); return b.build_lshr(x, b.build_and(y, b.build_iconst_i32(31))); },
         [&](int32_t x, int32_t y) { return s(u(x) >> (y & 31)); }},
        {"ashr", [](KB& k, Value* x, Value* y) { auto& b = k.builder(); return b.build_ashr(x, b.build_and(y, b.build_iconst_i32(31))); },
         [](int32_t x, int32_t y) { return x >> (y & 31); }},
        {"select", [](KB& k, Value* x, Value* y) { auto& b = k.builder(); return b.build_select(b.build_slt(x, y), x, y); },
         [](int32_t x, int32_t y) { return std::min(x, y); }},
        {"sdiv const -1", [](KB& k, Value* x, Value*) { auto& b = k.builder(); return b.build_sdiv(x, b.build_iconst_i32(-1)); },
         [&](int32_t x, int32_t) { return s(0u - u(x)); }},
        {"sdiv const 3", [](KB& k, Value* x, Value*) { auto& b = k.builder(); return b.build_sdiv(x, b.build_iconst_i32(3)); },
         [](int32_t x, int32_t) { return x / 3; }},
        {"smod const -1", [](KB& k, Value* x, Value*) { auto& b = k.builder(); return b.build_smod(x, b.build_iconst_i32(-1)); },
         [](int32_t, int32_t) { return 0; }},
    };
    check_binops<int32_t>("i32", ops, a, b);
}

TEST_CASE("SPIR-V exec - i64 ALU with the MIR edge semantics") {
    if (!vk_ready()) return;
    std::vector<int64_t> a = {7, -7, kMin64, kMin64, 0x123456789ABCDEFLL, -1, 1LL << 40, -(1LL << 50) + 3, 9, 77};
    std::vector<int64_t> b = {3, -3, -1, 7, 1000003, -1, 63, 40, 1LL << 33, -5};
    auto u = [](int64_t v) { return static_cast<uint64_t>(v); };
    auto s = [](uint64_t v) { return static_cast<int64_t>(v); };
    using KB = KernelBuilder;
    auto c63 = [](KB& k) { return k.builder().build_iconst_i64(63); };
    std::vector<BinOp<int64_t>> ops = {
        {"add", [](KB& k, Value* x, Value* y) { return k.builder().build_add(x, y); }, [&](int64_t x, int64_t y) { return s(u(x) + u(y)); }},
        {"sub", [](KB& k, Value* x, Value* y) { return k.builder().build_sub(x, y); }, [&](int64_t x, int64_t y) { return s(u(x) - u(y)); }},
        {"mul", [](KB& k, Value* x, Value* y) { return k.builder().build_mul(x, y); }, [&](int64_t x, int64_t y) { return s(u(x) * u(y)); }},
        {"sdiv", [](KB& k, Value* x, Value* y) { return k.builder().build_sdiv(x, y); },
         [](int64_t x, int64_t y) { return (x == kMin64 && y == -1) ? kMin64 : x / y; }},
        {"smod", [](KB& k, Value* x, Value* y) { return k.builder().build_smod(x, y); },
         [](int64_t x, int64_t y) { return (x == kMin64 && y == -1) ? 0 : x % y; }},
        {"udiv", [](KB& k, Value* x, Value* y) { return k.builder().build_udiv(x, y); }, [&](int64_t x, int64_t y) { return s(u(x) / u(y)); }},
        {"umod", [](KB& k, Value* x, Value* y) { return k.builder().build_umod(x, y); }, [&](int64_t x, int64_t y) { return s(u(x) % u(y)); }},
        {"neg", [](KB& k, Value* x, Value*) { return k.builder().build_neg(x); }, [&](int64_t x, int64_t) { return s(0u - u(x)); }},
        {"and", [](KB& k, Value* x, Value* y) { return k.builder().build_and(x, y); }, [](int64_t x, int64_t y) { return x & y; }},
        {"or", [](KB& k, Value* x, Value* y) { return k.builder().build_or(x, y); }, [](int64_t x, int64_t y) { return x | y; }},
        {"xor", [](KB& k, Value* x, Value* y) { return k.builder().build_xor(x, y); }, [](int64_t x, int64_t y) { return x ^ y; }},
        {"not", [](KB& k, Value* x, Value*) { return k.builder().build_not(x); }, [](int64_t x, int64_t) { return ~x; }},
        {"shl", [c63](KB& k, Value* x, Value* y) { return k.builder().build_shl(x, k.builder().build_and(y, c63(k))); },
         [&](int64_t x, int64_t y) { return s(u(x) << (y & 63)); }},
        {"lshr", [c63](KB& k, Value* x, Value* y) { return k.builder().build_lshr(x, k.builder().build_and(y, c63(k))); },
         [&](int64_t x, int64_t y) { return s(u(x) >> (y & 63)); }},
        {"ashr", [c63](KB& k, Value* x, Value* y) { return k.builder().build_ashr(x, k.builder().build_and(y, c63(k))); },
         [](int64_t x, int64_t y) { return x >> (y & 63); }},
        {"select ult", [](KB& k, Value* x, Value* y) { auto& b = k.builder(); return b.build_select(b.build_ult(x, y), x, y); },
         [&](int64_t x, int64_t y) { return u(x) < u(y) ? x : y; }},
    };
    check_binops<int64_t>("i64", ops, a, b);
}

TEST_CASE("SPIR-V exec - f32 and f64 ALU, NMin/NMax with NaN, RoundEven") {
    if (!vk_ready()) return;
    using KB = KernelBuilder;
    auto float_ops = [](auto zero) {
        using T = decltype(zero);
        return std::vector<BinOp<T>>{
            {"add", [](KB& k, Value* x, Value* y) { return k.builder().build_add(x, y); }, [](T x, T y) { return x + y; }},
            {"sub", [](KB& k, Value* x, Value* y) { return k.builder().build_sub(x, y); }, [](T x, T y) { return x - y; }},
            {"mul", [](KB& k, Value* x, Value* y) { return k.builder().build_mul(x, y); }, [](T x, T y) { return x * y; }},
            {"div", [](KB& k, Value* x, Value* y) { return k.builder().build_sdiv(x, y); }, [](T x, T y) { return x / y; }},
            {"rem", [](KB& k, Value* x, Value* y) { return k.builder().build_smod(x, y); }, [](T x, T y) { return std::fmod(x, y); }},
            {"neg", [](KB& k, Value* x, Value*) { return k.builder().build_neg(x); }, [](T x, T) { return -x; }},
            {"fma", [](KB& k, Value* x, Value* y) { return k.builder().build_fma(x, y, x); }, [](T x, T y) { return std::fma(x, y, x); }},
            {"sqrt|x|", [](KB& k, Value* x, Value*) { auto& b = k.builder(); return b.build_sqrt(b.build_fabs(x)); },
             [](T x, T) { return std::sqrt(std::fabs(x)); }},
            {"floor", [](KB& k, Value* x, Value*) { return k.builder().build_floor(x); }, [](T x, T) { return std::floor(x); }},
            {"ceil", [](KB& k, Value* x, Value*) { return k.builder().build_ceil(x); }, [](T x, T) { return std::ceil(x); }},
            {"round (even)", [](KB& k, Value* x, Value*) { return k.builder().build_round(x); }, [](T x, T) { return std::nearbyint(x); }},
            {"fabs", [](KB& k, Value* x, Value*) { return k.builder().build_fabs(x); }, [](T x, T) { return std::fabs(x); }},
            {"fmin", [](KB& k, Value* x, Value* y) { return k.builder().build_fmin(x, y); }, [](T x, T y) { return std::fmin(x, y); }},
            {"fmax", [](KB& k, Value* x, Value* y) { return k.builder().build_fmax(x, y); }, [](T x, T y) { return std::fmax(x, y); }},
        };
    };
    // Inputs avoid y == 0 for div/rem, and quotients large enough for rem's
    // x - y * trunc(x / y) (Vulkan's OpFRem precision) to lose bits; NaN
    // lanes are checked for fmin/fmax only.
    std::vector<float> a = {1.5f, -2.25f, 2.5f, 3.5f, -2.5f, 1e3f, 0.1f, kNaN, 4.0f, -0.0f, 7.75f, 100.0f};
    std::vector<float> b = {0.5f, 4.0f, -1.0f, 2.0f, 3.0f, 3.0f, 0.3f, 2.0f, kNaN, 1.0f, -2.5f, 7.0f};
    check_binops<float>("f32", float_ops(0.0f), a, b, 2e-6);
    std::vector<double> ad = {1.5, -2.25, 2.5, 3.5, -2.5, 1e3, 0.1, kNaN64, 4.0, -0.0, 7.75, 100.0};
    std::vector<double> bd = {0.5, 4.0, -1.0, 2.0, 3.0, 3.0, 0.3, 2.0, kNaN64, 1.0, -2.5, 7.0};
    check_binops<double>("f64", float_ops(0.0), ad, bd, 1e-14);
}

TEST_CASE("SPIR-V exec - comparisons as values and conditions, float ne unordered") {
    if (!vk_ready()) return;
    using Cmp = Value* (Builder::*)(Value*, Value*);
    const std::pair<const char*, Cmp> cmps[] = {
        {"eq", &Builder::build_eq}, {"ne", &Builder::build_ne}, {"slt", &Builder::build_slt}, {"ult", &Builder::build_ult},
        {"sle", &Builder::build_sle}, {"ule", &Builder::build_ule}, {"sgt", &Builder::build_sgt}, {"ugt", &Builder::build_ugt},
        {"sge", &Builder::build_sge}, {"uge", &Builder::build_uge}};
    // Each comparison as an i32 value (materialized) and as a select condition.
    auto cmp_map = [](Cmp c) {
        return [c](KernelBuilder& k, Value* x, Value* y) {
            Builder& b = k.builder();
            Value* as_value = (b.*c)(x, y);
            Value* as_cond = b.build_select((b.*c)(x, y), b.build_iconst_i32(10), b.build_iconst_i32(20));
            return b.build_add(as_value, as_cond);
        };
    };
    std::vector<int32_t> a = {1, -1, 5, kMin32, 0, 3};
    std::vector<int32_t> b = {1, 1, -5, 0, kMin32, 4};
    std::vector<int64_t> al = {1, -1, 5, kMin64, 0, 1LL << 40};
    std::vector<int64_t> bl = {1, 1, -5, 0, kMin64, (1LL << 40) + 1};
    std::vector<float> af = {1.0f, -1.0f, kNaN, 2.0f, kNaN, 0.0f};
    std::vector<float> bf = {1.0f, 1.0f, 1.0f, kNaN, kNaN, -0.0f};
    for (auto [name, c] : cmps) {
        auto host_i = [&](auto x, auto y) -> int32_t {
            using S = decltype(x);
            using U = std::make_unsigned_t<S>;
            U ux = static_cast<U>(x), uy = static_cast<U>(y);
            std::string n = name;
            bool r = n == "eq" ? x == y : n == "ne" ? x != y : n == "slt" ? x < y : n == "ult" ? ux < uy
                   : n == "sle" ? x <= y : n == "ule" ? ux <= uy : n == "sgt" ? x > y : n == "ugt" ? ux > uy
                   : n == "sge" ? x >= y : ux >= uy;
            return r ? 11 : 20;
        };
        std::vector<int32_t> g32 = run_map<int32_t, int32_t, int32_t>(cmp_map(c), a, b);
        std::vector<int32_t> g64 = run_map<int64_t, int64_t, int32_t>(cmp_map(c), al, bl);
        for (size_t i = 0; i < a.size(); ++i) {
            if (g32[i] != host_i(a[i], b[i])) std::cerr << "i32 " << name << " lane " << i << "\n";
            CHECK_EQ(g32[i], host_i(a[i], b[i]));
            if (g64[i] != host_i(al[i], bl[i])) std::cerr << "i64 " << name << " lane " << i << "\n";
            CHECK_EQ(g64[i], host_i(al[i], bl[i]));
        }
        std::vector<int32_t> gf = run_map<float, float, int32_t>(cmp_map(c), af, bf);
        for (size_t i = 0; i < af.size(); ++i) {
            float x = af[i], y = bf[i];
            std::string n = name;
            // Float comparisons ignore signedness; all ordered except ne.
            bool r = n == "eq" ? x == y : n == "ne" ? !(x == y) : (n == "slt" || n == "ult") ? x < y
                   : (n == "sle" || n == "ule") ? x <= y : (n == "sgt" || n == "ugt") ? x > y : x >= y;
            if (gf[i] != (r ? 11 : 20)) std::cerr << "f32 " << name << " lane " << i << "\n";
            CHECK_EQ(gf[i], r ? 11 : 20);
        }
    }
}

TEST_CASE("SPIR-V exec - conversions and bitcasts") {
    if (!vk_ready()) return;
    std::vector<int64_t> l = {0, -1, 5, -123456789012LL, 1LL << 40, 0x7FFFFFFF00000001LL};
    std::vector<double> d = {0.0, -1.75, 2.5, -123456.875, 1e9, 3.0e-300};
    std::vector<int64_t> got = run_map<int64_t, double, int64_t>([](KernelBuilder& k, Value* x, Value* y) {
        Builder& b = k.builder();
        // Pack several conversions into one i64 per lane via a checksum-free
        // layout is awkward; instead select one by the lane parity of x.
        Value* t32 = b.build_trunc_i32(x);
        Value* sx = b.build_sext_i64(t32);
        Value* zx = b.build_zext_i64(t32);
        Value* f = b.build_sitofp_f64_i64(x);
        Value* back = b.build_fptosi_i64(b.build_add(f, y));
        Value* r = b.build_xor(b.build_mul(sx, b.build_iconst_i64(3)), b.build_shl(zx, b.build_iconst_i64(1)));
        return b.build_add(r, back);
    }, l, d);
    for (size_t i = 0; i < l.size(); ++i) {
        int32_t t = static_cast<int32_t>(l[i]);
        int64_t sx = t;
        int64_t zx = static_cast<int64_t>(static_cast<uint32_t>(t));
        int64_t back = static_cast<int64_t>(static_cast<double>(l[i]) + d[i]);
        int64_t want = static_cast<int64_t>((static_cast<uint64_t>(sx) * 3u) ^ (static_cast<uint64_t>(zx) << 1)) + back;
        CHECK_EQ(got[i], want);
    }

    std::vector<float> f = {0.5f, -1.5f, 1e6f, 3.25f, -1e-3f, 65504.0f};
    std::vector<int32_t> iv = {1, -1, 1 << 24, 16777217, -100000, 7};
    std::vector<double> gd = run_map<float, int32_t, double>([](KernelBuilder& k, Value* x, Value* i) {
        Builder& b = k.builder();
        Value* e = b.build_fpext_f64_f32(x);                       // exact
        Value* fi = b.build_sitofp_f64_i32(i);
        Value* f32i = b.build_fpext_f64_f32(b.build_sitofp_f32_i32(i));  // rounds 16777217
        Value* tr = b.build_fpext_f64_f32(b.build_fptrunc_f32_f64(b.build_mul(e, b.build_fconst_f64(1.0 / 3.0))));
        Value* fs = b.build_sitofp_f64_i32(b.build_fptosi_i32_f32(x));   // truncation toward zero
        return b.build_add(b.build_add(b.build_add(e, fi), b.build_mul(f32i, b.build_fconst_f64(1024.0))),
                           b.build_add(tr, b.build_mul(fs, b.build_fconst_f64(1e-6))));
    }, f, iv);
    for (size_t i = 0; i < f.size(); ++i) {
        double e = f[i];
        double want = (e + iv[i] + static_cast<double>(static_cast<float>(iv[i])) * 1024.0) +
                      (static_cast<double>(static_cast<float>(e * (1.0 / 3.0))) + static_cast<int32_t>(f[i]) * 1e-6);
        if (!near(gd[i], want, 1e-15)) std::fprintf(stderr, "lane %zu device %.17g host %.17g\n", i, gd[i], want);
        CHECK(near(gd[i], want, 1e-15));
    }

    std::vector<double> bits_in = {1.0, -2.0, kNaN64, 1e-310};
    std::vector<int64_t> dummy(bits_in.size(), 0);
    std::vector<int64_t> gb = run_map<double, int64_t, int64_t>([](KernelBuilder& k, Value* x, Value*) {
        Builder& b = k.builder();
        Value* asi = b.build_bitcast_i64_f64(x);
        Value* asf = b.build_bitcast_f64_i64(b.build_xor(asi, b.build_iconst_i64(kMin64))); // flip the sign
        return b.build_bitcast_i64_f64(asf);
    }, bits_in, dummy);
    for (size_t i = 0; i < bits_in.size(); ++i)
        CHECK_EQ(static_cast<uint64_t>(gb[i]), f64_bits(bits_in[i]) ^ 0x8000000000000000ull);
}
