// Subgroup operations on a Vulkan device: the PTX shuffle modes (down, up,
// bfly, idx; f32 and i32; constant and register deltas) with PTX semantics on
// 32-lane segments, and the KernelBuilder warp / block reductions, each run
// with the pipeline's subgroup size required to be 32, 64 (when the device
// can require it) and left to the driver -- the kernels are written for a
// 32-thread warp and must be exact for every size. [SKIP] without a device.

#include "spirv_exec_support.hpp"

#include <algorithm>
#include <numeric>

using namespace brass;
using namespace spvexec;
using codegen::KernelBuilder;

namespace {

// Subgroup sizes to run with: 32 and 64 when requirable, and 0 (driver's choice).
std::vector<uint32_t> subgroup_sizes() {
    std::vector<uint32_t> s;
    const brass::gpu::VulkanDeviceCaps& c = caps();
    for (uint32_t want : {32u, 64u})
        if (c.can_require_subgroup_size_in_compute && want >= c.min_subgroup_size && want <= c.max_subgroup_size)
            s.push_back(want);
    s.push_back(0);
    return s;
}

enum class Mode { down, up, bfly, idx };

// PTX shfl.sync on the 32-thread warp containing `t` (full mask, nvcc clamps).
int32_t host_shfl(const std::vector<int32_t>& v, uint32_t t, Mode mode, uint32_t d) {
    uint32_t lane = t & 31u, base = t & ~31u;
    uint32_t src = t;
    switch (mode) {
        case Mode::down: src = lane + d <= 31 ? t + d : t; break;
        case Mode::up: src = lane >= d ? t - d : t; break;
        case Mode::bfly: src = base | (lane ^ (d & 31u)); break;
        case Mode::idx: src = base | (d & 31u); break;
    }
    return v[src];
}

} // namespace

TEST_CASE("SPIR-V exec subgroup - shuffles keep PTX semantics on 32-lane segments") {
    if (!vk_ready()) return;
    const uint32_t block = 128;
    std::vector<int32_t> in(block);
    for (uint32_t i = 0; i < block; ++i) in[i] = static_cast<int32_t>(i * 1000 + 7);
    struct Case { const char* name; Mode mode; bool f32; };
    const Case cases[] = {
        {"down", Mode::down, true}, {"up", Mode::up, true}, {"bfly", Mode::bfly, true}, {"idx", Mode::idx, true},
        {"down", Mode::down, false}, {"up", Mode::up, false}, {"bfly", Mode::bfly, false}, {"idx", Mode::idx, false},
    };
    for (uint32_t sg : subgroup_sizes()) {
        for (const Case& c : cases) {
            // out[t * 3 + 0] = shfl(v, const 1); [1] = shfl(v, const 16 / 5 for idx); [2] = shfl(v, register t % 7 + 1)
            spvtest::Kernel k({Type::ptr(), Type::ptr()});
            KernelBuilder kb(k.b);
            Builder& b = k.b;
            Value* t = kb.tid_x();
            Value* v = kb.load_i32_indexed(k.p(0), t);
            Type vt = c.f32 ? Type::f32() : Type::i32();
            Value* val = c.f32 ? b.build_sitofp_f32_i32(v) : v;
            std::string name = std::string("ptx_shfl_") + c.name + (c.f32 ? "_f32" : "_i32");
            Value* rdelta = b.build_add(b.build_smod(t, kb.const_i32(7)), kb.const_i32(1));
            uint32_t d2 = c.mode == Mode::idx ? 5u : 16u;
            Value* r[3] = {b.build_call(name, vt, {val, kb.const_i32(1)}),
                           b.build_call(name, vt, {val, kb.const_i32(static_cast<int32_t>(d2))}),
                           b.build_call(name, vt, {val, rdelta})};
            for (int i = 0; i < 3; ++i) {
                Value* iv = c.f32 ? b.build_fptosi_i32_f32(r[i]) : r[i];
                kb.store_i32_indexed(k.p(1), b.build_add(b.build_mul(t, kb.const_i32(3)), kb.const_i32(i)), iv);
            }
            b.build_ret_void();
            VulkanModuleOptions mo;
            mo.subgroup_size = sg;
            VulkanModule m = load(*k.fn, mo);
            CHECK_EQ(m.subgroup_size(), sg);
            VulkanBuffer din = upload(in), dout = zeros<int32_t>(block * 3);
            launch(m, 1, block, {din, dout});
            std::vector<int32_t> got = download<int32_t>(dout, block * 3);
            size_t bad = 0;
            for (uint32_t tt = 0; tt < block; ++tt) {
                int32_t want[3] = {host_shfl(in, tt, c.mode, 1), host_shfl(in, tt, c.mode, d2),
                                   host_shfl(in, tt, c.mode, tt % 7 + 1)};
                for (uint32_t i = 0; i < 3; ++i) {
                    if (got[tt * 3 + i] != want[i] && bad++ < 4)
                        std::cerr << name << " sg " << sg << " thread " << tt << " #" << i << " device " << got[tt * 3 + i]
                                  << " host " << want[i] << "\n";
                    CHECK_EQ(got[tt * 3 + i], want[i]);
                }
            }
        }
    }
}

TEST_CASE("SPIR-V exec subgroup - warp and block reductions at every block and subgroup size") {
    if (!vk_ready()) return;
    // block_reduce_sum over in[row * n + i] for i < n strided by ntid; warp max of
    // the per-thread partial per 32-thread warp. out[row] = sum, wmax[row*32 + warp] = max.
    Module mod("red");
    Function* f = mod.create_function("block_sum", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* in = b.add_block_param(e, Type::ptr());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* wmax = b.add_block_param(e, Type::ptr());
    Value* n = b.add_block_param(e, Type::i32());
    Value* scratch = kb.shared_alloc_f32(32);
    Value* tid = kb.tid_x();
    Value* row = kb.ctaid_x();
    Value* rowp = b.build_add(in, b.build_mul(b.build_zext_i64(b.build_mul(row, n)), b.build_iconst_i64(4)));
    Value* v = kb.for_range_reduce(tid, n, kb.ntid_x(), kb.const_f32(0.0f), [&](Value* i, Value* acc) {
        return b.build_add(acc, kb.load_f32_indexed(rowp, i));
    });
    Value* wm = kb.warp_reduce_max_f32(v);
    Value* total = kb.block_reduce_sum_f32(v, scratch);
    kb.if_then(b.build_eq(kb.lane_id(), kb.const_i32(0)), [&] {
        kb.store_f32_indexed(wmax, b.build_add(b.build_mul(row, kb.const_i32(32)), kb.warp_id()), wm);
    });
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] { kb.store_f32_indexed(out, row, total); });
    b.build_ret_void();
    target::SpirvKernel sk = spvtest::compile_checked(*f);

    const uint32_t rows = 3, n_elems = 3001;
    std::vector<float> data(rows * n_elems);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>((i * 7) % 13) - 4.0f;   // exact sums
    for (uint32_t sg : subgroup_sizes()) {
        VulkanModuleOptions mo;
        mo.subgroup_size = sg;
        VulkanModule m = load_kernel(sk, mo);
        for (uint32_t block : {32u, 64u, 96u, 128u, 256u, 512u, 1024u}) {
            VulkanBuffer din = upload(data), dout = zeros<float>(rows), dw = zeros<float>(rows * 32);
            launch(m, rows, block, {din, dout, dw, static_cast<int32_t>(n_elems)});
            std::vector<float> got = download<float>(dout, rows);
            std::vector<float> gw = download<float>(dw, rows * 32);
            for (uint32_t r = 0; r < rows; ++r) {
                std::vector<float> partial(block, 0.0f);
                for (uint32_t i = 0; i < n_elems; ++i) partial[i % block] += data[r * n_elems + i];
                float sum = std::accumulate(partial.begin(), partial.end(), 0.0f);
                if (got[r] != sum) std::cerr << "sg " << sg << " block " << block << " row " << r << " device " << got[r] << " host " << sum << "\n";
                CHECK_EQ(got[r], sum);
                for (uint32_t w = 0; w < block / 32; ++w) {
                    float mx = *std::max_element(partial.begin() + w * 32, partial.begin() + w * 32 + 32);
                    if (gw[r * 32 + w] != mx) std::cerr << "sg " << sg << " block " << block << " warp " << w << " max wrong\n";
                    CHECK_EQ(gw[r * 32 + w], mx);
                }
            }
        }
    }
}
