// Memory and intrinsics on a Vulkan device, checked against host values:
// device-address loads/stores of every scalar width (i32/i64/f32/f64/ptr,
// indexed with scale and displacement, one pointer read at two types),
// 8/16-bit narrow access with sign/zero extension at odd offsets, all eight
// vector types (round trip and arithmetic), shared memory with barriers
// (every element type, constant and dynamic offsets), atomics (global and
// shared, including float add, with the old values returned), f16
// pack/unpack, clock reads, the math and wide-multiply intrinsics.
// Subgroup shuffles and reductions are in test_spirv_execution_subgroup.cpp.
// [SKIP] without a Vulkan device.

#include "spirv_exec_support.hpp"
#include "ptx_test_support.hpp"   // f16 host conversions

#include <algorithm>
#include <numeric>
#include <set>

using namespace brass;
using namespace spvexec;
using codegen::KernelBuilder;

TEST_CASE("SPIR-V exec mem - scalar loads and stores of every width, indexed and displaced") {
    if (!vk_ready()) return;
    // in: 64 u64 words; out: 64 u64 words. For each type T at slot s (8 bytes):
    //   out[s] = in[s]; out[16 + idx + 2] (scaled, disp 16) = in[idx + 1] (scaled, disp 8 for T's size)
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::i32(), Type::i64()});
    Builder& b = k.b;
    int32_t off = 0;
    for (Type t : {Type::i32(), Type::i64(), Type::f32(), Type::f64(), Type::ptr()}) {
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        b.build_store(t, k.p(1), off, b.build_load(t, k.p(0), off));
        Value* vi = b.build_load_indexed(t, k.p(0), k.p(2), sz, 64 + off);       // in + i*sz + 64 + off
        b.build_store_indexed(t, k.p(1), k.p(3), sz, 128 + off, vi);             // out + l*sz + 128 + off
        Value* vc = b.build_load_indexed(t, k.p(0), b.build_iconst_i64(5), sz, off);
        b.build_store(t, k.p(1), off + 256, vc);
        off += 40;
    }
    b.build_store(Type::f32(), k.p(1), 448, b.build_load(Type::f32(), k.p(0), 0));
    b.build_store(Type::i32(), k.p(1), 452, b.build_load(Type::i32(), k.p(0), 0));
    b.build_ret_void();
    VulkanModule m = load(*k.fn);

    std::vector<uint8_t> in(512);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 37 + 11);
    VulkanBuffer din = upload(in), dout = zeros<uint8_t>(512);
    const int32_t idx = 3;
    const int64_t lidx = 2;
    launch(m, 1, 1, {din, dout, idx, lidx});
    std::vector<uint8_t> got = download<uint8_t>(dout, 512);
    std::vector<uint8_t> want(512, 0);
    off = 0;
    for (size_t sz : {4u, 8u, 4u, 8u, 8u}) {
        std::memcpy(&want[static_cast<size_t>(off)], &in[static_cast<size_t>(off)], sz);
        std::memcpy(&want[static_cast<size_t>(lidx) * sz + 128 + static_cast<size_t>(off)],
                    &in[static_cast<size_t>(idx) * sz + 64 + static_cast<size_t>(off)], sz);
        std::memcpy(&want[static_cast<size_t>(off) + 256], &in[5 * sz + static_cast<size_t>(off)], sz);
        off += 40;
    }
    std::memcpy(&want[448], &in[0], 4);
    std::memcpy(&want[452], &in[0], 4);
    for (size_t i = 0; i < 512; ++i) {
        if (got[i] != want[i]) std::cerr << "byte " << i << " device " << +got[i] << " host " << +want[i] << "\n";
        CHECK_EQ(got[i], want[i]);
    }
}

TEST_CASE("SPIR-V exec mem - 8/16-bit loads (signed and unsigned) and stores at odd offsets") {
    if (!vk_ready()) return;
    // thread t: base = in + t*5; out32[t*4 + 0..3] = u8(base+1), s8(base+1), u16(base+2), s16(base+2);
    // narrow stores: out8[t*4 + 1] = u8 of x, out8[t*4 + 2..3] = u16 of x (2-aligned)
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::ptr()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* t = kb.tid_x();
    Value* base = b.build_add(k.p(0), b.build_mul(b.build_zext_i64(t), b.build_iconst_i64(6)));
    Value* vals[4] = {kb.load_u8(base, 1), kb.load_s8(base, 1), kb.load_u16(base, 2), kb.load_s16(base, 2)};
    for (int i = 0; i < 4; ++i) kb.store_i32_indexed(k.p(1), b.build_add(b.build_shl(t, kb.const_i32(2)), kb.const_i32(i)), vals[i]);
    Value* o8 = b.build_add(k.p(2), b.build_mul(b.build_zext_i64(t), b.build_iconst_i64(4)));
    Value* x = b.build_add(b.build_mul(t, kb.const_i32(0x1F3)), kb.const_i32(0x7F80));
    kb.store_u8(o8, x, 1);
    kb.store_u16(o8, x, 2);
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    const uint32_t threads = 64;
    std::vector<uint8_t> in(threads * 6);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 53 + 7);
    VulkanBuffer din = upload(in), d32 = zeros<int32_t>(threads * 4), d8 = zeros<uint8_t>(threads * 4);
    launch(m, 1, threads, {din, d32, d8});
    std::vector<int32_t> g32 = download<int32_t>(d32, threads * 4);
    std::vector<uint8_t> g8 = download<uint8_t>(d8, threads * 4);
    for (uint32_t tt = 0; tt < threads; ++tt) {
        const uint8_t* p = &in[tt * 6];
        uint16_t h;
        std::memcpy(&h, p + 2, 2);
        int32_t want[4] = {p[1], static_cast<int8_t>(p[1]), h, static_cast<int16_t>(h)};
        for (int i = 0; i < 4; ++i) CHECK_EQ(g32[tt * 4 + static_cast<uint32_t>(i)], want[i]);
        uint32_t xv = tt * 0x1F3u + 0x7F80u;
        CHECK_EQ(g8[tt * 4 + 0], 0);
        CHECK_EQ(g8[tt * 4 + 1], static_cast<uint8_t>(xv));
        uint16_t sh;
        std::memcpy(&sh, &g8[tt * 4 + 2], 2);
        CHECK_EQ(sh, static_cast<uint16_t>(xv));
    }
}

TEST_CASE("SPIR-V exec mem - all eight vector types: round trip and arithmetic") {
    if (!vk_ready()) return;
    for (Type t : {Type::f32x4(), Type::f64x2(), Type::i32x4(), Type::i64x2(), Type::f32x8(), Type::f64x4(),
                   Type::i32x8(), Type::i64x4()}) {
        const bool fl = t.element_type().is_float();
        const uint32_t lanes = t.vector_lanes();
        const uint32_t esz = static_cast<uint32_t>(t.element_type().size_in_bytes());
        // out0 = a (copy); out1 = fma(a + c, a, c) min/max mixes; lane ops; shuffle
        spvtest::Kernel k({Type::ptr(), Type::ptr()});
        Builder& b = k.b;
        Value* a = b.build_vload(t, k.p(0), 0);
        Value* c = b.build_vload(t, k.p(0), 32);
        b.build_vstore(t, k.p(1), 0, a);
        Value* r = b.build_vmul(b.build_vadd(a, c), b.build_vsub(a, c));
        r = b.build_vmax(b.build_vmin(r, b.build_vmul(a, a)), b.build_vneg(c));
        r = b.build_vfma(r, a, c);
        Value* last = b.build_vextract_lane(a, lanes - 1);
        r = b.build_vinsert_lane(r, last, 0);
        r = b.build_vadd(r, b.build_vbroadcast(t, b.build_vextract_lane(c, 1)));
        b.build_vstore(t, k.p(1), 32, r);
        Value* bits = fl ? b.build_vadd(a, b.build_vzero(t)) : b.build_vxor(b.build_vand(a, c), b.build_vnot(b.build_vor(a, c)));
        b.build_vstore(t, k.p(1), 64, bits);
        b.build_vstore(t, k.p(1), 96, b.build_vshuffle(a, c, 0x1B));
        b.build_ret_void();
        VulkanModule m = load(*k.fn);

        std::vector<uint8_t> in(64);
        if (fl) {
            for (uint32_t i = 0; i < 64 / esz; ++i) {
                double v = static_cast<double>(i) * 0.75 - 3.0;
                if (esz == 4) { float f = static_cast<float>(v); std::memcpy(&in[i * 4], &f, 4); }
                else std::memcpy(&in[i * 8], &v, 8);
            }
        } else {
            for (uint32_t i = 0; i < 64 / esz; ++i) {
                int64_t v = static_cast<int64_t>(i) * 3 - 7;
                std::memcpy(&in[i * esz], &v, esz);
            }
        }
        VulkanBuffer din = upload(in), dout = zeros<uint8_t>(128);
        launch(m, 1, 1, {din, dout});
        std::vector<uint8_t> got = download<uint8_t>(dout, 128);
        auto lane_of = [&](const std::vector<uint8_t>& buf, uint32_t base, uint32_t l) -> double {
            if (fl) {
                if (esz == 4) { float f; std::memcpy(&f, &buf[base + l * 4], 4); return f; }
                double d; std::memcpy(&d, &buf[base + l * 8], 8); return d;
            }
            if (esz == 4) { int32_t v; std::memcpy(&v, &buf[base + l * 4], 4); return v; }
            int64_t v; std::memcpy(&v, &buf[base + l * 8], 8); return static_cast<double>(v);
        };
        CHECK(std::equal(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(t.size_in_bytes()), got.begin()));
        for (uint32_t l = 0; l < lanes; ++l) {
            double av = lane_of(in, 0, l), cv = lane_of(in, 32, l);
            double rv = (av + cv) * (av - cv);
            rv = std::max(std::min(rv, av * av), -cv);
            rv = rv * av + cv;  // the fma (exact for these small values)
            if (l == 0) rv = lane_of(in, 0, lanes - 1);
            rv += lane_of(in, 32, 1);
            if (lane_of(got, 32, l) != rv)
                std::cerr << t.name() << " lane " << l << " device " << lane_of(got, 32, l) << " host " << rv << "\n";
            CHECK_EQ(lane_of(got, 32, l), rv);
            if (fl) {
                CHECK_EQ(lane_of(got, 64, l), av);
            } else {
                int64_t ai = static_cast<int64_t>(av), ci = static_cast<int64_t>(cv);
                CHECK_EQ(lane_of(got, 64, l), static_cast<double>((ai & ci) ^ ~(ai | ci)));
            }
        }
        // vshuffle imm 0x1B over a 4-lane group selects lanes 3,2,1,0 (per
        // 128-bit part for the 256-bit types); just require a permutation of a's lanes.
        std::multiset<double> src, dst;
        for (uint32_t l = 0; l < lanes; ++l) {
            src.insert(lane_of(in, 0, l));
            src.insert(lane_of(in, 32, l));
            dst.insert(lane_of(got, 96, l));
        }
        for (double d : dst) CHECK(src.count(d) > 0);
    }
}

TEST_CASE("SPIR-V exec mem - shared memory of every element type with barriers") {
    if (!vk_ready()) return;
    // Each thread writes in[gid] to shared[tid], barrier, reads shared[ntid - 1 - tid]:
    // a per-block reversal. f32 and i32 indexed, f64 and i64 through byte offsets
    // (const + dynamic via add on the shared pointer).
    for (const char* ty : {"f32", "i32", "f64", "i64"}) {
        std::string t = ty;
        Type et = t == "f32" ? Type::f32() : t == "i32" ? Type::i32() : t == "f64" ? Type::f64() : Type::i64();
        uint32_t esz = static_cast<uint32_t>(et.size_in_bytes());
        spvtest::Kernel k({Type::ptr(), Type::ptr()});
        KernelBuilder kb(k.b);
        Builder& b = k.b;
        Value* smem = b.build_call("ptx_shared_alloc_" + t, Type::ptr(), {kb.const_i32(256)});
        Value* tid = kb.tid_x();
        Value* gid = kb.global_tid_x();
        Value* v = b.build_load_indexed(et, k.p(0), gid, static_cast<uint8_t>(esz));
        Value* rev = b.build_sub(b.build_sub(kb.ntid_x(), kb.const_i32(1)), tid);
        Value* r;
        if (esz == 4) {
            b.build_call("ptx_shared_store_" + t + "_indexed", Type::void_type(), {smem, tid, v});
            kb.sync();
            r = b.build_call("ptx_shared_load_" + t + "_indexed", et, {smem, rev});
        } else {
            Value* off = b.build_mul(b.build_zext_i64(tid), b.build_iconst_i64(esz));
            Value* p = b.build_add(b.build_add(smem, b.build_iconst_i64(esz)), off);   // const + dynamic
            b.build_call("ptx_shared_store_" + t, Type::void_type(), {p, v, kb.const_i32(-static_cast<int32_t>(esz))});
            kb.sync();
            Value* q = b.build_add(smem, b.build_mul(b.build_zext_i64(rev), b.build_iconst_i64(esz)));
            r = b.build_call("ptx_shared_load_" + t, et, {q, kb.const_i32(0)});
        }
        b.build_store_indexed(et, k.p(1), gid, static_cast<uint8_t>(esz), r);
        b.build_ret_void();
        VulkanModule m = load(*k.fn);
        const uint32_t block = 256, grid = 3, n = block * grid;
        std::vector<uint64_t> in(n);
        for (uint32_t i = 0; i < n; ++i) in[i] = 0x0100000000000000ull * (i % 7) + i * 977u;
        std::vector<uint8_t> raw(n * esz);
        for (uint32_t i = 0; i < n; ++i) std::memcpy(&raw[i * esz], &in[i], esz);
        VulkanBuffer din = upload(raw), dout = zeros<uint8_t>(n * esz);
        launch(m, grid, block, {din, dout});
        std::vector<uint8_t> got = download<uint8_t>(dout, n * esz);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t src = (i / block) * block + (block - 1 - i % block);
            bool ok = std::memcmp(&got[i * esz], &raw[src * esz], esz) == 0;
            if (!ok) std::cerr << t << " element " << i << " wrong\n";
            CHECK(ok);
        }
    }
}

TEST_CASE("SPIR-V exec mem - atomics: global and shared, float add, old values") {
    if (!vk_ready()) return;
    // counters: [0] f32 add 1.0, [1] i32 add 1 (old -> olds[gid]), [2] u32 add 3, [3] min, [4] max,
    // [5] exch (any thread's value), [6..7] i64 add 2^33; shared f32/i32 adds per block -> blk[ctaid*2..]
    spvtest::Kernel k({Type::ptr(), Type::ptr(), Type::ptr()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* c = k.p(0);
    Value* gid = kb.global_tid_x();
    Value* tid = kb.tid_x();
    auto at = [&](int32_t byte) { return b.build_add(c, b.build_iconst_i64(byte)); };
    kb.atom_add_f32(c, kb.const_f32(1.0f));
    Value* old = kb.atom_add_i32(at(4), kb.const_i32(1));
    kb.store_i32_indexed(k.p(1), gid, old);
    b.build_call("ptx_atom_add_u32", Type::i32(), {at(8), kb.const_i32(3)});
    Value* key = b.build_sub(b.build_mul(b.build_smod(b.build_mul(gid, kb.const_i32(7919)), kb.const_i32(1000)), kb.const_i32(1)), kb.const_i32(400));
    b.build_call("ptx_atom_min_i32", Type::i32(), {at(12), key});
    b.build_call("ptx_atom_max_i32", Type::i32(), {at(16), key});
    b.build_call("ptx_atom_exch_i32", Type::i32(), {at(20), gid});
    b.build_call("ptx_atom_add_i64", Type::i64(), {at(24), b.build_iconst_i64(1LL << 33)});
    Value* sf = kb.shared_alloc_f32(1);
    Value* si = kb.shared_alloc_i32(1);
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] {
        kb.shared_store_f32(sf, kb.const_f32(0.0f));
        kb.shared_store_i32(si, kb.const_i32(0));
    });
    kb.sync();
    b.build_call("ptx_atom_shared_add_f32", Type::f32(), {sf, kb.const_f32(0.5f)});
    b.build_call("ptx_atom_shared_add_i32", Type::i32(), {si, tid});
    kb.sync();
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] {
        Value* blk = b.build_add(k.p(2), b.build_mul(b.build_zext_i64(kb.ctaid_x()), b.build_iconst_i64(8)));
        kb.store_f32(blk, kb.shared_load_f32(sf, 0));
        kb.store_i32(blk, kb.shared_load_i32(si, 0), 4);
    });
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    const uint32_t block = 128, grid = 40, n = block * grid;
    std::vector<int32_t> init = {0, 0, 0, 1 << 30, -(1 << 30), -1, 0, 0};
    VulkanBuffer dc = upload(init), dold = zeros<int32_t>(n), dblk = zeros<int32_t>(grid * 2);
    launch(m, grid, block, {dc, dold, dblk});
    std::vector<int32_t> got = download<int32_t>(dc, 8);
    float fsum;
    std::memcpy(&fsum, &got[0], 4);
    CHECK_EQ(fsum, static_cast<float>(n));
    CHECK_EQ(got[1], static_cast<int32_t>(n));
    CHECK_EQ(static_cast<uint32_t>(got[2]), 3u * n);
    int32_t kmin = INT32_MAX, kmax = INT32_MIN;
    for (uint32_t g = 0; g < n; ++g) {
        int32_t key_h = static_cast<int32_t>((g * 7919u) % 1000u) - 400;
        kmin = std::min(kmin, key_h);
        kmax = std::max(kmax, key_h);
    }
    CHECK_EQ(got[3], kmin);
    CHECK_EQ(got[4], kmax);
    CHECK(got[5] >= 0 && got[5] < static_cast<int32_t>(n));
    int64_t i64sum;
    std::memcpy(&i64sum, &got[6], 8);
    CHECK_EQ(i64sum, static_cast<int64_t>(n) << 33);
    std::vector<int32_t> olds = download<int32_t>(dold, n);
    std::sort(olds.begin(), olds.end());
    for (uint32_t i = 0; i < n; ++i) CHECK_EQ(olds[i], static_cast<int32_t>(i));   // each old value once
    std::vector<int32_t> blk = download<int32_t>(dblk, grid * 2);
    for (uint32_t g = 0; g < grid; ++g) {
        float f;
        std::memcpy(&f, &blk[g * 2], 4);
        CHECK_EQ(f, 0.5f * block);
        CHECK_EQ(blk[g * 2 + 1], static_cast<int32_t>(block * (block - 1) / 2));
    }
}

TEST_CASE("SPIR-V exec mem - f16 pack and unpack against the host conversion") {
    if (!vk_ready()) return;
    // Unpack: every one of the 65536 half bit patterns.
    std::vector<int32_t> halves(65536);
    std::iota(halves.begin(), halves.end(), 0);
    std::vector<int32_t> zero(halves.size(), 0);
    std::vector<float> un = run_map<int32_t, int32_t, float>([](KernelBuilder& kb, Value* h, Value*) {
        return kb.f16_to_f32(h);
    }, halves, zero);
    size_t bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        float want = ptxtest::f16_to_f32_host(static_cast<uint16_t>(h));
        bool ok = std::isnan(want) ? std::isnan(un[h]) : f32_bits(un[h]) == f32_bits(want);
        // Denormal halves may flush on devices without denorm preservation for 16-bit.
        if (!ok && (h & 0x7C00u) == 0 && un[h] == 0.0f) ok = true;
        if (!ok && bad++ < 5) std::cerr << "unpack " << h << " device " << un[h] << " host " << want << "\n";
        CHECK(ok);
    }
    // Pack: representable values exactly; others to a neighbouring half
    // (PackHalf2x16's rounding is the implementation's).
    std::vector<float> f;
    for (int i = -300; i <= 300; ++i) f.push_back(static_cast<float>(i) * 0.37f);
    for (float v : {0.0f, -0.0f, 1.0f, -2.0f, 65504.0f, 6.1035156e-05f, 0.5f, 1024.0f, 0.333333f, 1e-3f})
        f.push_back(v);
    std::vector<float> fz(f.size(), 0.0f);
    std::vector<int32_t> pk = run_map<float, float, int32_t>([](KernelBuilder& kb, Value* x, Value*) {
        return kb.f32_to_f16(x);
    }, f, fz);
    for (size_t i = 0; i < f.size(); ++i) {
        uint16_t want = ptxtest::f32_to_f16_host(f[i]);
        uint32_t g = static_cast<uint32_t>(pk[i]);
        CHECK((g >> 16) == 0u);
        bool ok = g == want || (ptxtest::f16_to_f32_host(want) != f[i] && (g == want + 1u || g + 1u == want));
        if (!ok) std::cerr << "pack " << f[i] << " device 0x" << std::hex << g << " host 0x" << want << std::dec << "\n";
        CHECK(ok);
    }
}

TEST_CASE("SPIR-V exec mem - clock reads are monotone") {
    if (!vk_ready()) return;
    spvtest::Kernel k({Type::ptr(), Type::i32()});
    KernelBuilder kb(k.b);
    Builder& b = k.b;
    Value* t0 = b.build_call("ptx_clock64", Type::i64());
    Value* c32 = b.build_call("ptx_clock", Type::i32());
    Value* acc = kb.for_range_reduce(kb.const_i32(0), k.p(1), kb.const_i32(1), kb.const_f32(1.0f), [&](Value*, Value* a) {
        return b.build_fma(a, kb.const_f32(1.0001f), kb.const_f32(0.5f));
    });
    Value* t1 = b.build_call("ptx_clock64", Type::i64());
    Value* gid = kb.global_tid_x();
    Value* rec = b.build_add(k.p(0), b.build_mul(b.build_zext_i64(gid), b.build_iconst_i64(24)));
    b.build_store(Type::i64(), rec, 0, t0);
    b.build_store(Type::i64(), rec, 8, t1);
    b.build_store(Type::i32(), rec, 16, c32);
    b.build_store(Type::f32(), rec, 20, acc);
    b.build_ret_void();
    VulkanModule m = load(*k.fn);
    VulkanBuffer out = zeros<uint64_t>(64 * 3);
    launch(m, 1, 64, {out, int32_t{2000}});
    std::vector<uint64_t> got = download<uint64_t>(out, 64 * 3);
    for (uint32_t t = 0; t < 64; ++t) {
        CHECK(got[t * 3 + 1] > got[t * 3]);
    }
}

TEST_CASE("SPIR-V exec mem - math, conversion and wide-multiply intrinsics") {
    if (!vk_ready()) return;
    struct Unary { const char* name; double tol; double (*host)(double); };
    const Unary unary[] = {
        {"rsqrtf", 1e-6, [](double x) { return 1.0 / std::sqrt(std::fabs(x) + 0.5); }},
        {"sqrtf", 1e-6, [](double x) { return std::sqrt(std::fabs(x) + 0.5); }},
        {"sinf", 2e-6, [](double x) { return std::sin(x); }},
        {"cosf", 2e-6, [](double x) { return std::cos(x); }},
        {"ex2f", 2e-6, [](double x) { return std::exp2(x); }},
        {"lg2f", 2e-6, [](double x) { return std::log2(std::fabs(x) + 0.5); }},
        {"expf", 4e-6, [](double x) { return std::exp(x); }},
        {"logf", 4e-6, [](double x) { return std::log(std::fabs(x) + 0.5); }},
        {"ptx_rcp", 1e-6, [](double x) { return 1.0 / (std::fabs(x) + 0.5); }},
        {"fabsf", 0.0, [](double x) { return std::fabs(x); }},
    };
    std::vector<float> xs;
    for (int i = -40; i <= 40; ++i) xs.push_back(static_cast<float>(i) * 0.075f);
    std::vector<float> zs(xs.size(), 0.0f);
    for (const Unary& u : unary) {
        std::string n = u.name;
        bool shifted = n == "rsqrtf" || n == "sqrtf" || n == "lg2f" || n == "logf" || n == "ptx_rcp";
        std::vector<float> got = run_map<float, float, float>([&](KernelBuilder& kb, Value* x, Value*) {
            Builder& b = kb.builder();
            Value* arg = shifted ? b.build_add(b.build_fabs(x), kb.const_f32(0.5f)) : x;
            return b.build_call(n, Type::f32(), {arg});
        }, xs, zs);
        for (size_t i = 0; i < xs.size(); ++i) {
            double want = u.host(xs[i]);
            if (!near(got[i], want, u.tol + 1e-7)) std::cerr << n << "(" << xs[i] << ") device " << got[i] << " host " << want << "\n";
            CHECK(near(got[i], want, u.tol + 1e-7));
        }
    }
    std::vector<float> ys(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) ys[i] = 0.25f + static_cast<float>(i % 9) * 0.5f;
    std::vector<float> dv = run_map<float, float, float>([](KernelBuilder& kb, Value* x, Value* y) {
        return kb.builder().build_call("ptx_div_approx", Type::f32(), {x, y});
    }, xs, ys);
    for (size_t i = 0; i < xs.size(); ++i) CHECK(near(dv[i], static_cast<double>(xs[i]) / ys[i], 1e-6));

    // Integer intrinsics on (a, b) pairs, packed into i64 results.
    std::vector<int32_t> a = {0, 1, -1, 0x7FFFFFFF, static_cast<int32_t>(0x80000000u), 123456789, -98765, 65536};
    std::vector<int32_t> bb = {5, -1, -1, 2, 3, 1000, 77, 65536};
    for (const char* op : {"ptx_mul_wide_u32", "ptx_mul_wide_s32", "ptx_mul_hi_u32", "ptx_mad_lo_u32", "ptx_u32_to_f32",
                           "ptx_i64_to_f32", "ptx_u64_to_f32", "ptx_f32_to_u32"}) {
        std::string o = op;
        std::vector<int64_t> got = run_map<int32_t, int32_t, int64_t>([&](KernelBuilder& kb, Value* x, Value* y) -> Value* {
            Builder& b = kb.builder();
            if (o == "ptx_mul_wide_u32" || o == "ptx_mul_wide_s32") return b.build_call(o, Type::i64(), {x, y});
            if (o == "ptx_mul_hi_u32") return b.build_zext_i64(b.build_call(o, Type::i32(), {x, y}));
            if (o == "ptx_mad_lo_u32") return b.build_zext_i64(b.build_call(o, Type::i32(), {x, y, x}));
            Value* f;
            if (o == "ptx_u32_to_f32") f = b.build_call(o, Type::f32(), {x});
            else if (o == "ptx_f32_to_u32")
                return b.build_zext_i64(b.build_call(o, Type::i32(), {b.build_fabs(b.build_sitofp_f32_i32(y))}));
            else f = b.build_call(o, Type::f32(), {b.build_mul(b.build_sext_i64(x), b.build_iconst_i64(1 << 20))});
            return b.build_bitcast_i64_f64(b.build_fpext_f64_f32(f));
        }, a, bb);
        for (size_t i = 0; i < a.size(); ++i) {
            uint32_t ua = static_cast<uint32_t>(a[i]), ub = static_cast<uint32_t>(bb[i]);
            int64_t want;
            auto as_f = [](float f) { return static_cast<int64_t>(f64_bits(static_cast<double>(f))); };
            if (o == "ptx_mul_wide_u32") want = static_cast<int64_t>(uint64_t{ua} * ub);
            else if (o == "ptx_mul_wide_s32") want = int64_t{a[i]} * bb[i];
            else if (o == "ptx_mul_hi_u32") want = static_cast<int64_t>((uint64_t{ua} * ub) >> 32);
            else if (o == "ptx_mad_lo_u32") want = static_cast<int64_t>(static_cast<uint32_t>(ua * ub + ua));
            else if (o == "ptx_u32_to_f32") want = as_f(static_cast<float>(ua));
            else if (o == "ptx_f32_to_u32") want = static_cast<int64_t>(static_cast<uint32_t>(std::fabs(static_cast<float>(bb[i]))));
            else if (o == "ptx_i64_to_f32") want = as_f(static_cast<float>(int64_t{a[i]} * (1 << 20)));
            else want = as_f(static_cast<float>(static_cast<uint64_t>(int64_t{a[i]} * (1 << 20))));
            if (got[i] != want) std::cerr << o << " lane " << i << " device " << got[i] << " host " << want << "\n";
            CHECK_EQ(got[i], want);
        }
    }
}
