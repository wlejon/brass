// Stage 4 PTX intrinsics: every entry of the PtxISel intrinsic table has a
// signature row below (the coverage test fails for unknown names), lowers,
// verifies and assembles with ptxas; the on-device cases run each family
// against a host reference (visible [SKIP] without ptxas / CUDA).
//
// Vector opcodes and the KernelBuilder reduction/control-flow helpers are in
// test_ptx_vector.cpp.

#include "ptx_test_support.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>

using namespace brass;
using brass::codegen::KernelBuilder;
using brass::ptx::PtxISel;
using namespace ptxtest;

namespace {

// ---------------------------------------------------------------------------
// Signature table: how to call each intrinsic in a kernel with parameters
// (i32 i, f32 x, f64 d, i64 l, ptr p). Result type per kind.
// ---------------------------------------------------------------------------

enum class Sig {
    void_i32,        // () -> i32
    void_i64,        // () -> i64
    f32_f32,         // (f32) -> f32
    f32f32_f32,      // (f32, f32) -> f32
    f64_f64,         // (f64) -> f64          (fabs/fmin/fmax pick the type from the argument)
    i32_f32,         // (i32) -> f32
    i64_f32,         // (i64) -> f32
    f32_i32,         // (f32) -> i32
    f32_f64,         // (f32) -> f64
    f64_f32,         // (f64) -> f32
    shfl_f32,        // (f32, i32) -> f32
    shfl_i32,        // (i32, i32) -> i32
    shfl_sync_f32,   // (i32 mask, f32, i32) -> f32
    bar,             // () -> void
    bar_id,          // (i32) -> void
    bar_id_count,    // (i32, i32) -> void
    atom_f32,        // (ptr, f32) -> f32
    atom_i32,        // (ptr, i32) -> i32
    atom_i64,        // (ptr, i64) -> i64
    atom_shared_f32, // (shared ptr, f32) -> f32
    atom_shared_i32, // (shared ptr, i32) -> i32
    i32i32_i64,      // (i32, i32) -> i64
    i32i32_i32,      // (i32, i32) -> i32
    i32x3_i32,       // (i32, i32, i32) -> i32
    shared_alloc,    // (const i32) -> ptr
    shared_ld_f32, shared_ld_i32, shared_ld_f64, shared_ld_i64,     // (ptr[, off]) -> T
    shared_ldx_f32, shared_ldx_i32, shared_ldx_f64, shared_ldx_i64, // (ptr, index) -> T
    shared_st_f32, shared_st_i32, shared_st_f64, shared_st_i64,     // (ptr, T[, off]) -> void
    shared_stx_f32, shared_stx_i32, shared_stx_f64, shared_stx_i64, // (ptr, index, T) -> void
    load_narrow,     // (ptr[, off]) -> i32
    store_narrow,    // (ptr, i32[, off]) -> void
};

const std::map<std::string, Sig>& signatures() {
    static const std::map<std::string, Sig> kSigs = {
        {"ptx_tid_x", Sig::void_i32}, {"ptx_tid_y", Sig::void_i32}, {"ptx_tid_z", Sig::void_i32},
        {"ptx_ctaid_x", Sig::void_i32}, {"ptx_ctaid_y", Sig::void_i32}, {"ptx_ctaid_z", Sig::void_i32},
        {"ptx_ntid_x", Sig::void_i32}, {"ptx_ntid_y", Sig::void_i32}, {"ptx_ntid_z", Sig::void_i32},
        {"ptx_nctaid_x", Sig::void_i32}, {"ptx_nctaid_y", Sig::void_i32}, {"ptx_nctaid_z", Sig::void_i32},
        {"ptx_laneid", Sig::void_i32}, {"ptx_lane_id", Sig::void_i32}, {"ptx_warpid", Sig::void_i32},
        {"ptx_warp_id", Sig::void_i32}, {"ptx_nwarpid", Sig::void_i32}, {"ptx_smid", Sig::void_i32},
        {"ptx_nsmid", Sig::void_i32}, {"ptx_clock", Sig::void_i32}, {"ptx_global_tid_x", Sig::void_i32},
        {"ptx_global_id_x", Sig::void_i32}, {"ptx_clock64", Sig::void_i64}, {"ptx_globaltimer", Sig::void_i64},

        {"rsqrtf", Sig::f32_f32}, {"rsqrt", Sig::f32_f32}, {"ptx_rsqrt", Sig::f32_f32},
        {"sqrtf", Sig::f32_f32}, {"sqrt", Sig::f32_f32}, {"ptx_sqrt", Sig::f32_f32},
        {"sinf", Sig::f32_f32}, {"sin", Sig::f32_f32}, {"ptx_sin", Sig::f32_f32},
        {"cosf", Sig::f32_f32}, {"cos", Sig::f32_f32}, {"ptx_cos", Sig::f32_f32},
        {"ex2f", Sig::f32_f32}, {"ex2", Sig::f32_f32}, {"ptx_ex2", Sig::f32_f32},
        {"lg2f", Sig::f32_f32}, {"ptx_lg2", Sig::f32_f32},
        {"ptx_rcp", Sig::f32_f32}, {"ptx_rcp_approx", Sig::f32_f32},
        {"expf", Sig::f32_f32}, {"exp", Sig::f32_f32}, {"ptx_exp", Sig::f32_f32},
        {"logf", Sig::f32_f32}, {"log", Sig::f32_f32}, {"ptx_log", Sig::f32_f32},
        {"ptx_sqrt_rn", Sig::f32_f32},
        {"fabsf", Sig::f32_f32}, {"fabs", Sig::f64_f64}, {"ptx_fabs", Sig::f32_f32},
        {"fminf", Sig::f32f32_f32}, {"fmin", Sig::f32f32_f32}, {"ptx_fmin", Sig::f32f32_f32},
        {"fmaxf", Sig::f32f32_f32}, {"fmax", Sig::f32f32_f32}, {"ptx_fmax", Sig::f32f32_f32},
        {"ptx_div_approx", Sig::f32f32_f32},

        {"i32_to_f32", Sig::i32_f32}, {"ptx_i32_to_f32", Sig::i32_f32}, {"ptx_u32_to_f32", Sig::i32_f32},
        {"ptx_u64_to_f32", Sig::i64_f32}, {"ptx_i64_to_f32", Sig::i64_f32},
        {"ptx_f32_to_i32", Sig::f32_i32}, {"ptx_f32_to_u32", Sig::f32_i32},
        {"ptx_f16_to_f32", Sig::i32_f32}, {"ptx_f32_to_f16", Sig::f32_i32},
        {"ptx_f32_to_f64", Sig::f32_f64}, {"ptx_f64_to_f32", Sig::f64_f32},

        {"bar.sync", Sig::bar}, {"ptx_sync", Sig::bar}, {"ptx_bar_sync", Sig::bar_id},
        {"ptx_bar_sync_count", Sig::bar_id_count},

        {"ptx_shfl_down_sync_f32", Sig::shfl_sync_f32}, {"shfl_down_sync_f32", Sig::shfl_sync_f32},
        {"ptx_shfl_down_f32", Sig::shfl_f32}, {"ptx_shfl_up_f32", Sig::shfl_f32},
        {"ptx_shfl_bfly_f32", Sig::shfl_f32}, {"ptx_shfl_xor_f32", Sig::shfl_f32}, {"ptx_shfl_idx_f32", Sig::shfl_f32},
        {"ptx_shfl_down_i32", Sig::shfl_i32}, {"ptx_shfl_up_i32", Sig::shfl_i32},
        {"ptx_shfl_bfly_i32", Sig::shfl_i32}, {"ptx_shfl_xor_i32", Sig::shfl_i32}, {"ptx_shfl_idx_i32", Sig::shfl_i32},

        {"ptx_atom_add_f32", Sig::atom_f32}, {"ptx_atom_add_i32", Sig::atom_i32}, {"ptx_atom_add_u32", Sig::atom_i32},
        {"ptx_atom_add_i64", Sig::atom_i64}, {"ptx_atom_min_i32", Sig::atom_i32}, {"ptx_atom_max_i32", Sig::atom_i32},
        {"ptx_atom_exch_i32", Sig::atom_i32},
        {"ptx_atom_shared_add_f32", Sig::atom_shared_f32}, {"ptx_atom_shared_add_i32", Sig::atom_shared_i32},

        {"ptx_mul_wide_u32", Sig::i32i32_i64}, {"ptx_mul_wide_s32", Sig::i32i32_i64},
        {"ptx_mul_hi_u32", Sig::i32i32_i32}, {"ptx_mad_lo_u32", Sig::i32x3_i32},

        {"ptx_shared_alloc_f32", Sig::shared_alloc}, {"ptx_shared_alloc_i32", Sig::shared_alloc},
        {"ptx_shared_alloc_f64", Sig::shared_alloc}, {"ptx_shared_alloc_i64", Sig::shared_alloc},
        {"ptx_shared_load_f32", Sig::shared_ld_f32}, {"ptx_shared_load_i32", Sig::shared_ld_i32},
        {"ptx_shared_load_f64", Sig::shared_ld_f64}, {"ptx_shared_load_i64", Sig::shared_ld_i64},
        {"ptx_shared_load_f32_indexed", Sig::shared_ldx_f32}, {"ptx_shared_load_i32_indexed", Sig::shared_ldx_i32},
        {"ptx_shared_load_f64_indexed", Sig::shared_ldx_f64}, {"ptx_shared_load_i64_indexed", Sig::shared_ldx_i64},
        {"ptx_shared_store_f32", Sig::shared_st_f32}, {"ptx_shared_store_i32", Sig::shared_st_i32},
        {"ptx_shared_store_f64", Sig::shared_st_f64}, {"ptx_shared_store_i64", Sig::shared_st_i64},
        {"ptx_shared_store_f32_indexed", Sig::shared_stx_f32}, {"ptx_shared_store_i32_indexed", Sig::shared_stx_i32},
        {"ptx_shared_store_f64_indexed", Sig::shared_stx_f64}, {"ptx_shared_store_i64_indexed", Sig::shared_stx_i64},

        {"ptx_load_u8", Sig::load_narrow}, {"ptx_load_s8", Sig::load_narrow},
        {"ptx_load_u16", Sig::load_narrow}, {"ptx_load_s16", Sig::load_narrow},
        {"ptx_store_u8", Sig::store_narrow}, {"ptx_store_u16", Sig::store_narrow},
    };
    return kSigs;
}

// Builds a kernel (i32, f32, f64, i64, ptr) that calls `name` per its
// signature and stores any result through the pointer.
Function* build_intrinsic_kernel(Module& mod, const std::string& name, Sig sig) {
    Function* f = mod.create_function("intrin", Type::void_type(),
                                      {Type::i32(), Type::f32(), Type::f64(), Type::i64(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* i = b.add_block_param(e, Type::i32());
    Value* x = b.add_block_param(e, Type::f32());
    Value* d = b.add_block_param(e, Type::f64());
    Value* l = b.add_block_param(e, Type::i64());
    Value* p = b.add_block_param(e, Type::ptr());
    Value* c4 = b.build_iconst_i32(4);
    Value* smem = b.build_call("ptx_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(64)});
    Value* smem_i = b.build_call("ptx_shared_alloc_i32", Type::ptr(), {b.build_iconst_i32(64)});

    Value* r = nullptr;
    switch (sig) {
        case Sig::void_i32: r = b.build_call(name, Type::i32()); break;
        case Sig::void_i64: r = b.build_call(name, Type::i64()); break;
        case Sig::f32_f32: r = b.build_call(name, Type::f32(), {x}); break;
        case Sig::f32f32_f32: r = b.build_call(name, Type::f32(), {x, x}); break;
        case Sig::f64_f64: r = b.build_call(name, Type::f64(), {d}); break;
        case Sig::i32_f32: r = b.build_call(name, Type::f32(), {i}); break;
        case Sig::i64_f32: r = b.build_call(name, Type::f32(), {l}); break;
        case Sig::f32_i32: r = b.build_call(name, Type::i32(), {x}); break;
        case Sig::f32_f64: r = b.build_call(name, Type::f64(), {x}); break;
        case Sig::f64_f32: r = b.build_call(name, Type::f32(), {d}); break;
        case Sig::shfl_f32: r = b.build_call(name, Type::f32(), {x, i}); break;          // register delta
        case Sig::shfl_i32: r = b.build_call(name, Type::i32(), {i, c4}); break;         // immediate delta
        case Sig::shfl_sync_f32: r = b.build_call(name, Type::f32(), {i, x, i}); break;
        case Sig::bar: b.build_call(name, Type::void_type()); break;
        case Sig::bar_id: b.build_call(name, Type::void_type(), {b.build_iconst_i32(1)}); break;
        case Sig::bar_id_count: b.build_call(name, Type::void_type(), {b.build_iconst_i32(1), i}); break;
        case Sig::atom_f32: r = b.build_call(name, Type::f32(), {p, x}); break;
        case Sig::atom_i32: r = b.build_call(name, Type::i32(), {p, i}); break;
        case Sig::atom_i64: r = b.build_call(name, Type::i64(), {p, l}); break;
        case Sig::atom_shared_f32: r = b.build_call(name, Type::f32(), {smem, x}); break;
        case Sig::atom_shared_i32: r = b.build_call(name, Type::i32(), {smem_i, i}); break;
        case Sig::i32i32_i64: r = b.build_call(name, Type::i64(), {i, i}); break;
        case Sig::i32i32_i32: r = b.build_call(name, Type::i32(), {i, i}); break;
        case Sig::i32x3_i32: r = b.build_call(name, Type::i32(), {i, i, i}); break;
        case Sig::shared_alloc: r = b.build_call(name, Type::ptr(), {b.build_iconst_i32(16)}); break;
        case Sig::shared_ld_f32: r = b.build_call(name, Type::f32(), {smem, c4}); break;
        case Sig::shared_ld_i32: r = b.build_call(name, Type::i32(), {smem, i}); break;   // register offset
        case Sig::shared_ld_f64: r = b.build_call(name, Type::f64(), {smem}); break;
        case Sig::shared_ld_i64: r = b.build_call(name, Type::i64(), {smem, l}); break;   // 64-bit register offset
        case Sig::shared_ldx_f32: r = b.build_call(name, Type::f32(), {smem, i}); break;
        case Sig::shared_ldx_i32: r = b.build_call(name, Type::i32(), {smem, c4}); break;
        case Sig::shared_ldx_f64: r = b.build_call(name, Type::f64(), {smem, l}); break;
        case Sig::shared_ldx_i64: r = b.build_call(name, Type::i64(), {smem, i}); break;
        case Sig::shared_st_f32: b.build_call(name, Type::void_type(), {smem, x, c4}); break;
        case Sig::shared_st_i32: b.build_call(name, Type::void_type(), {smem, i}); break;
        case Sig::shared_st_f64: b.build_call(name, Type::void_type(), {smem, d, i}); break;
        case Sig::shared_st_i64: b.build_call(name, Type::void_type(), {smem, l, l}); break;
        case Sig::shared_stx_f32: b.build_call(name, Type::void_type(), {smem, i, x}); break;
        case Sig::shared_stx_i32: b.build_call(name, Type::void_type(), {smem, c4, i}); break;
        case Sig::shared_stx_f64: b.build_call(name, Type::void_type(), {smem, l, d}); break;
        case Sig::shared_stx_i64: b.build_call(name, Type::void_type(), {smem, i, l}); break;
        case Sig::load_narrow: r = b.build_call(name, Type::i32(), {p, c4}); break;
        case Sig::store_narrow: b.build_call(name, Type::void_type(), {p, i, i}); break;
    }
    if (r) {
        if (r->type() == Type::ptr()) b.build_store(Type::i64(), p, 0, r);
        else b.build_store(r->type(), p, 0, r);
    }
    b.build_ret_void();
    return f;
}

std::string intrinsic_kernel_ptx(const std::string& name, Sig sig) {
    Module mod("intrin");
    return target::PtxTarget::emit_function(*build_intrinsic_kernel(mod, name, sig));
}

} // namespace

// Shared with test_ptx_cleanup.cpp (declared in ptx_test_support.hpp).
brass::Function* ptxtest::build_intrinsic_kernel(brass::Module& mod, const std::string& name) {
    auto it = signatures().find(name);
    if (it == signatures().end()) throw std::runtime_error("no intrinsic signature for " + name);
    return ::build_intrinsic_kernel(mod, name, it->second);
}

// ---------------------------------------------------------------------------
// Coverage: the table and the signatures agree; everything assembles.
// ---------------------------------------------------------------------------

TEST_CASE("PTX intrinsics - every table entry has a signature, lowers, verifies and assembles") {
    const auto& sigs = signatures();
    std::vector<std::string_view> names = PtxISel::intrinsic_names();
    for (const auto& [name, sig] : sigs) {
        bool present = PtxISel::is_intrinsic(name);
        if (!present) std::cerr << "intrinsic table is missing " << name << "\n";
        CHECK(present);
    }
    bool have_ptxas = ptxas_available();
    for (std::string_view name : names) {
        auto it = sigs.find(std::string(name));
        if (it == sigs.end()) {
            std::cerr << "test_ptx_intrinsics.cpp has no signature for intrinsic " << name << "\n";
            CHECK(false);
            continue;
        }
        std::string ptx;
        try {
            ptx = intrinsic_kernel_ptx(it->first, it->second); // lowers + verifies inside PtxTarget
        } catch (const std::exception& ex) {
            std::cerr << "intrinsic " << name << ": " << ex.what() << "\n";
            CHECK(false);
            continue;
        }
        CHECK(ptx.find("call ") == std::string::npos); // lowered inline, not a call
        if (have_ptxas) {
            bool ok = ptxas_assembles(ptx);
            if (!ok) std::cerr << "ptxas rejected intrinsic " << name << ":\n" << ptx;
            CHECK(ok);
        }
    }
}

TEST_CASE("PTX intrinsics - shared_alloc rejects a non-constant count, mixed types are diagnosed") {
    Module mod("bad");
    Function* f = mod.create_function("bad", Type::void_type(), {Type::i32(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* n = b.add_block_param(e, Type::i32());
    Value* p = b.add_block_param(e, Type::ptr());
    Value* s = b.build_call("ptx_shared_alloc_f32", Type::ptr(), {n});
    b.build_store(Type::i64(), p, 0, s);
    b.build_ret_void();
    bool threw = false;
    try { target::PtxTarget::emit_function(*f); }
    catch (const std::runtime_error& err) {
        threw = true;
        CHECK(std::string(err.what()).find("compile-time constant") != std::string::npos);
    }
    CHECK(threw);

    Module mod2("bad2");
    Function* g = mod2.create_function("bad2", Type::void_type(), {Type::ptr()});
    Builder b2(mod2);
    b2.set_function(g);
    BasicBlock* e2 = b2.append_block("entry");
    b2.position_at_end(e2);
    Value* p2 = b2.add_block_param(e2, Type::ptr());
    Value* s2 = b2.build_call("ptx_shared_alloc_f32", Type::ptr(), {b2.build_iconst_i32(8)});
    Value* v = b2.build_call("ptx_shared_load_f32", Type::i32(), {s2}); // i32 result for an f32 load
    b2.build_store(Type::i32(), p2, 0, v);
    b2.build_ret_void();
    threw = false;
    try { target::PtxTarget::emit_function(*g); }
    catch (const std::runtime_error& err) {
        threw = true;
        CHECK(std::string(err.what()).find("does not match") != std::string::npos);
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Special registers
// ---------------------------------------------------------------------------

TEST_CASE("PTX intrinsics - special registers on device (grid 2 x block 64)") {
    Module mod("sreg");
    Function* f = mod.create_function("sreg", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* times = b.add_block_param(e, Type::ptr());
    Value* g = kb.global_tid_x();
    Value* base = b.build_mul(g, kb.const_i32(8));
    Value* fields[8] = { kb.tid_x(), kb.ctaid_x(), kb.ntid_x(), kb.nctaid_x(), kb.lane_id(), kb.warp_id(),
                         kb.tid_y(), b.build_add(kb.ctaid_y(), b.build_add(kb.ntid_y(), kb.nctaid_z())) };
    for (int k = 0; k < 8; ++k) {
        kb.store_i32_indexed(out, b.build_add(base, kb.const_i32(k)), fields[k]);
    }
    Value* t0 = b.build_call("ptx_clock64", Type::i64());
    Value* t1 = b.build_call("ptx_globaltimer", Type::i64());
    kb.store_i64_indexed(times, b.build_mul(g, kb.const_i32(2)), t0);
    kb.store_i64_indexed(times, b.build_add(b.build_mul(g, kb.const_i32(2)), kb.const_i32(1)), t1);
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    if (!gpu_ready()) return;
    const uint32_t grid = 2, block = 64;
    CudaBuffer dout = CudaBuffer::alloc(grid * block * 8 * 4);
    CudaBuffer dtimes = CudaBuffer::alloc(grid * block * 2 * 8);
    REQUIRE(dout.valid() && dout.zero() && dtimes.valid() && dtimes.zero());
    void* po = dout.device_ptr();
    void* pt = dtimes.device_ptr();
    launch(ptx, "sreg", grid, block, {&po, &pt});
    auto got = download<int32_t>(dout, grid * block * 8);
    auto ts = download<int64_t>(dtimes, grid * block * 2);
    for (uint32_t gi = 0; gi < grid * block; ++gi) {
        const int32_t* r = &got[gi * 8];
        CHECK_EQ(r[0], int32_t(gi % block));
        CHECK_EQ(r[1], int32_t(gi / block));
        CHECK_EQ(r[2], int32_t(block));
        CHECK_EQ(r[3], int32_t(grid));
        CHECK_EQ(r[4], int32_t(gi % 32));
        CHECK_EQ(r[5], int32_t((gi % block) / 32));
        CHECK_EQ(r[6], 0);
        CHECK_EQ(r[7], 0 + 1 + 1);
        CHECK(ts[gi * 2] > 0);
        CHECK(ts[gi * 2 + 1] > 0);
    }
}

// ---------------------------------------------------------------------------
// Math and conversions (elementwise, against host references)
// ---------------------------------------------------------------------------

TEST_CASE("PTX intrinsics - approximate and exact f32 math on device") {
    std::vector<float> xs, ys;
    for (int i = 0; i < 256; ++i) {
        xs.push_back(0.05f + static_cast<float>(i) * 0.25f); // up to ~64: exp stays finite in f32
        ys.push_back(1.5f + static_cast<float>((i * 7) % 13));
    }
    struct Case { const char* what; std::function<Value*(KernelBuilder&, Value*, Value*)> build; std::function<float(float, float)> ref; float tol; };
    std::vector<Case> cases = {
        {"rcp_approx",  [](KernelBuilder& kb, Value* x, Value*)   { return kb.rcp_approx(x); },        [](float x, float)   { return 1.0f / x; }, 1e-5f},
        {"div_approx",  [](KernelBuilder& kb, Value* x, Value* y) { return kb.div_approx(x, y); },     [](float x, float y) { return x / y; }, 1e-5f},
        {"ex2",         [](KernelBuilder& kb, Value* x, Value*)   { return kb.ex2_approx(x); },        [](float x, float)   { return std::exp2(x); }, 2e-5f},
        {"lg2",         [](KernelBuilder& kb, Value* x, Value*)   { return kb.lg2_approx(x); },        [](float x, float)   { return std::log2(x); }, 2e-5f},
        {"exp_fast",    [](KernelBuilder& kb, Value* x, Value*)   { return kb.exp_fast(x); },          [](float x, float)   { return std::exp(x); }, 5e-5f},
        {"log_fast",    [](KernelBuilder& kb, Value* x, Value*)   { return kb.log_fast(x); },          [](float x, float)   { return std::log(x); }, 5e-5f},
        {"rsqrt",       [](KernelBuilder& kb, Value* x, Value*)   { return kb.rsqrt_approx(x); },      [](float x, float)   { return 1.0f / std::sqrt(x); }, 1e-5f},
        {"sqrt_approx", [](KernelBuilder& kb, Value* x, Value*)   { return kb.sqrt_approx(x); },       [](float x, float)   { return std::sqrt(x); }, 1e-5f},
        {"sqrt_rn",     [](KernelBuilder& kb, Value* x, Value*)   { return kb.builder().build_call("ptx_sqrt_rn", Type::f32(), {x}); }, [](float x, float) { return std::sqrt(x); }, 1e-6f},
        {"fabs",        [](KernelBuilder& kb, Value* x, Value* y) { return kb.fabs(kb.sub(x, y)); },   [](float x, float y) { return std::fabs(x - y); }, 0.0f},
        {"fmin",        [](KernelBuilder& kb, Value* x, Value* y) { return kb.fmin(x, y); },           [](float x, float y) { return std::min(x, y); }, 0.0f},
        {"fmax",        [](KernelBuilder& kb, Value* x, Value* y) { return kb.fmax(x, y); },           [](float x, float y) { return std::max(x, y); }, 0.0f},
        {"sin",         [](KernelBuilder& kb, Value* x, Value*)   { return kb.builder().build_call("ptx_sin", Type::f32(), {x}); }, [](float x, float) { return std::sin(x); }, 2e-3f},
        {"fma_rn",      [](KernelBuilder& kb, Value* x, Value* y) { return kb.fma(x, y, x); },         [](float x, float y) { return std::fma(x, y, x); }, 1e-6f},
    };
    for (const Case& c : cases) {
        auto got = run_map<float, float, float>(c.build, xs, ys);
        if (got.empty()) return; // no device
        for (size_t i = 0; i < xs.size(); ++i) {
            float want = c.ref(xs[i], ys[i]);
            if (!near(got[i], want, c.tol)) std::cerr << c.what << "(" << xs[i] << ", " << ys[i] << ") = " << got[i] << ", want " << want << "\n";
            CHECK(near(got[i], want, c.tol));
        }
    }
}

TEST_CASE("PTX intrinsics - integer/float conversions on device") {
    std::vector<int32_t> is;
    std::vector<float> fs;
    for (int i = 0; i < 64; ++i) {
        is.push_back(static_cast<int32_t>(0x80000000u + static_cast<uint32_t>(i) * 0x01234567u)); // top bit set: u32 != s32
        fs.push_back(-1000.75f + static_cast<float>(i) * 33.3f);
    }
    // i32 -> f32 (signed) and u32 -> f32 (unsigned) of the same bits.
    {
        auto got = run_map<int32_t, float, float>([](KernelBuilder& kb, Value* i, Value*) { return kb.i32_to_f32(i); }, is, fs);
        if (got.empty()) return;
        for (size_t i = 0; i < is.size(); ++i) CHECK_EQ(got[i], static_cast<float>(is[i]));
        auto gotu = run_map<int32_t, float, float>([](KernelBuilder& kb, Value* i, Value*) { return kb.u32_to_f32(i); }, is, fs);
        for (size_t i = 0; i < is.size(); ++i) CHECK_EQ(gotu[i], static_cast<float>(static_cast<uint32_t>(is[i])));
    }
    // f32 -> i32 / u32 truncate toward zero.
    {
        auto got = run_map<int32_t, float, int32_t>([](KernelBuilder& kb, Value*, Value* x) { return kb.f32_to_i32(x); }, is, fs);
        for (size_t i = 0; i < fs.size(); ++i) CHECK_EQ(got[i], static_cast<int32_t>(fs[i]));
        auto gotu = run_map<int32_t, float, uint32_t>([](KernelBuilder& kb, Value*, Value* x) { return kb.f32_to_u32(x); }, is, fs);
        for (size_t i = 0; i < fs.size(); ++i) {
            uint32_t want = fs[i] < 0.0f ? 0u : static_cast<uint32_t>(fs[i]); // PTX cvt saturates
            CHECK_EQ(gotu[i], want);
        }
    }
    // i64 / u64 -> f32
    {
        std::vector<int64_t> ls;
        for (int i = 0; i < 64; ++i) ls.push_back(static_cast<int64_t>(0x8000000000000000ull + static_cast<uint64_t>(i) * 0x0123456789ull));
        auto got = run_map<int64_t, float, float>([](KernelBuilder& kb, Value* l, Value*) { return kb.builder().build_call("ptx_i64_to_f32", Type::f32(), {l}); }, ls, fs);
        for (size_t i = 0; i < ls.size(); ++i) CHECK_EQ(got[i], static_cast<float>(ls[i]));
        auto gotu = run_map<int64_t, float, float>([](KernelBuilder& kb, Value* l, Value*) { return kb.builder().build_call("ptx_u64_to_f32", Type::f32(), {l}); }, ls, fs);
        for (size_t i = 0; i < ls.size(); ++i) CHECK_EQ(gotu[i], static_cast<float>(static_cast<uint64_t>(ls[i])));
    }
    // f32 <-> f64
    {
        std::vector<double> ds;
        for (int i = 0; i < 64; ++i) ds.push_back(1.0 / 3.0 + i * 1e-7);
        auto up = run_map<float, double, double>([](KernelBuilder& kb, Value* x, Value*) { return kb.builder().build_call("ptx_f32_to_f64", Type::f64(), {x}); }, fs, ds);
        for (size_t i = 0; i < fs.size(); ++i) CHECK_EQ(up[i], static_cast<double>(fs[i]));
        auto down = run_map<float, double, float>([](KernelBuilder& kb, Value*, Value* d) { return kb.builder().build_call("ptx_f64_to_f32", Type::f32(), {d}); }, fs, ds);
        for (size_t i = 0; i < ds.size(); ++i) CHECK_EQ(down[i], static_cast<float>(ds[i]));
    }
}

TEST_CASE("PTX intrinsics - f16 conversions against a host reference") {
    // f16 bit patterns covering normals, subnormals, zero, negative values.
    std::vector<int32_t> bits;
    std::vector<float> vals;
    for (int i = 0; i < 512; ++i) {
        uint16_t h = static_cast<uint16_t>((i * 97) & 0x7BFF); // avoid inf/nan exponents
        if (i % 2) h |= 0x8000;
        bits.push_back(h);
        vals.push_back(-77.5f + static_cast<float>(i) * 0.3125f + (i % 3 == 0 ? 1e-3f : 0.0f));
    }
    // Sanity of the host helpers themselves.
    CHECK_EQ(f16_to_f32_host(0x3C00), 1.0f);
    CHECK_EQ(f16_to_f32_host(0xC000), -2.0f);
    CHECK_EQ(f16_to_f32_host(0x0001), 5.960464477539063e-8f);
    CHECK_EQ(f32_to_f16_host(1.0f), uint16_t(0x3C00));
    CHECK_EQ(f32_to_f16_host(65504.0f), uint16_t(0x7BFF));
    CHECK_EQ(f32_to_f16_host(1e-8f), uint16_t(0x0000));

    auto got = run_map<int32_t, float, float>([](KernelBuilder& kb, Value* h, Value*) { return kb.f16_to_f32(h); }, bits, vals);
    if (got.empty()) return;
    for (size_t i = 0; i < bits.size(); ++i) {
        float want = f16_to_f32_host(static_cast<uint16_t>(bits[i]));
        if (got[i] != want) std::cerr << "f16_to_f32(0x" << std::hex << bits[i] << std::dec << ") = " << got[i] << ", want " << want << "\n";
        CHECK_EQ(got[i], want);
    }
    auto back = run_map<int32_t, float, int32_t>([](KernelBuilder& kb, Value*, Value* x) { return kb.f32_to_f16(x); }, bits, vals);
    for (size_t i = 0; i < vals.size(); ++i) {
        uint16_t want = f32_to_f16_host(vals[i]);
        uint16_t have = static_cast<uint16_t>(back[i] & 0xFFFF);
        if (have != want) std::cerr << "f32_to_f16(" << vals[i] << ") = 0x" << std::hex << have << ", want 0x" << want << std::dec << "\n";
        CHECK_EQ(have, want);
    }
}

TEST_CASE("PTX intrinsics - mul.wide / mul.hi / mad.lo on device") {
    std::vector<int32_t> as, bs;
    for (int i = 0; i < 64; ++i) {
        as.push_back(static_cast<int32_t>(0xC0000000u + static_cast<uint32_t>(i) * 0x0F0F0F0Fu));
        bs.push_back(static_cast<int32_t>(0x7000000Fu - static_cast<uint32_t>(i) * 0x01010101u));
    }
    auto wide_u = run_map<int32_t, int32_t, uint64_t>([](KernelBuilder& kb, Value* a, Value* b) { return kb.builder().build_call("ptx_mul_wide_u32", Type::i64(), {a, b}); }, as, bs);
    if (wide_u.empty()) return;
    auto wide_s = run_map<int32_t, int32_t, int64_t>([](KernelBuilder& kb, Value* a, Value* b) { return kb.builder().build_call("ptx_mul_wide_s32", Type::i64(), {a, b}); }, as, bs);
    auto hi = run_map<int32_t, int32_t, uint32_t>([](KernelBuilder& kb, Value* a, Value* b) { return kb.builder().build_call("ptx_mul_hi_u32", Type::i32(), {a, b}); }, as, bs);
    auto mad = run_map<int32_t, int32_t, uint32_t>([](KernelBuilder& kb, Value* a, Value* b) { return kb.builder().build_call("ptx_mad_lo_u32", Type::i32(), {a, b, a}); }, as, bs);
    for (size_t i = 0; i < as.size(); ++i) {
        uint64_t ua = static_cast<uint32_t>(as[i]), ub = static_cast<uint32_t>(bs[i]);
        CHECK_EQ(wide_u[i], ua * ub);
        CHECK_EQ(wide_s[i], static_cast<int64_t>(as[i]) * static_cast<int64_t>(bs[i]));
        CHECK_EQ(hi[i], static_cast<uint32_t>((ua * ub) >> 32));
        CHECK_EQ(mad[i], static_cast<uint32_t>(ua * ub + ua));
    }
}

// ---------------------------------------------------------------------------
// Narrow loads/stores and atomics
// ---------------------------------------------------------------------------

TEST_CASE("PTX intrinsics - u8/s8/u16/s16 loads extend, u8/u16 stores truncate") {
    // kernel(bytes, out, n): per thread i: out[4i..4i+3] = {u8[i], s8[i], u16[i], s16[i]};
    // then bytes2[i] = u8 store of (u8[i] + 1), bytes2 halves = u16 store of (s16[i] ^ 0x5555)
    Module mod("narrow");
    Function* f = mod.create_function("narrow", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* bytes = b.add_block_param(e, Type::ptr());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* bytes2 = b.add_block_param(e, Type::ptr());
    Value* n = b.add_block_param(e, Type::i32());
    BasicBlock* body = b.append_block("body");
    BasicBlock* done = b.append_block("done");
    b.position_at_end(e);
    Value* i = kb.global_tid_x();
    b.build_br_if(b.build_slt(i, n), body, done);
    b.position_at_end(body);
    Value* i64 = b.build_sext_i64(i);
    Value* p1 = b.build_add(bytes, i64);                                   // &bytes[i]
    Value* p2 = b.build_add(bytes, b.build_shl(i64, kb.const_i64(1)));    // &halves[i]
    Value* u8 = kb.load_u8(p1);
    Value* s8 = kb.load_s8(p1);
    Value* u16 = kb.load_u16(p2);
    Value* s16 = kb.load_s16(p2);
    Value* base = b.build_mul(i, kb.const_i32(4));
    kb.store_i32_indexed(out, base, u8);
    kb.store_i32_indexed(out, b.build_add(base, kb.const_i32(1)), s8);
    kb.store_i32_indexed(out, b.build_add(base, kb.const_i32(2)), u16);
    kb.store_i32_indexed(out, b.build_add(base, kb.const_i32(3)), s16);
    Value* q1 = b.build_add(bytes2, i64);
    Value* q2 = b.build_add(bytes2, b.build_shl(i64, kb.const_i64(1)));
    kb.store_u8(q1, b.build_add(u8, kb.const_i32(1)), 512);                          // bytes2[512 + i]
    kb.store_u16(q2, b.build_xor(s16, kb.const_i32(0x5555)), 0);                     // halves2[i]
    b.build_br(done);
    b.position_at_end(done);
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    CHECK(ptx.find("ld.global.u8") != std::string::npos);
    CHECK(ptx.find("ld.global.s8") != std::string::npos);
    CHECK(ptx.find("ld.global.u16") != std::string::npos);
    CHECK(ptx.find("ld.global.s16") != std::string::npos);
    CHECK(ptx.find("st.global.u8") != std::string::npos);
    CHECK(ptx.find("st.global.u16") != std::string::npos);
    if (!gpu_ready()) return;

    const int32_t n_host = 256;
    std::vector<uint8_t> host(1024);
    for (size_t k = 0; k < host.size(); ++k) host[k] = static_cast<uint8_t>((k * 131 + 7) & 0xFF);
    CudaBuffer dbytes = upload(host);
    CudaBuffer dout = CudaBuffer::alloc(n_host * 4 * 4);
    CudaBuffer dbytes2 = CudaBuffer::alloc(1024);
    REQUIRE(dout.valid() && dout.zero() && dbytes2.valid() && dbytes2.zero());
    void* pb = dbytes.device_ptr(); void* po = dout.device_ptr(); void* pb2 = dbytes2.device_ptr();
    int32_t n_arg = n_host;
    launch(ptx, "narrow", 2, 128, {&pb, &po, &pb2, &n_arg});
    auto got = download<int32_t>(dout, n_host * 4);
    auto got2 = download<uint8_t>(dbytes2, 1024);
    for (int32_t k = 0; k < n_host; ++k) {
        uint16_t h = static_cast<uint16_t>(host[2 * k] | (host[2 * k + 1] << 8));
        CHECK_EQ(got[k * 4 + 0], int32_t(host[k]));
        CHECK_EQ(got[k * 4 + 1], int32_t(static_cast<int8_t>(host[k])));
        CHECK_EQ(got[k * 4 + 2], int32_t(h));
        CHECK_EQ(got[k * 4 + 3], int32_t(static_cast<int16_t>(h)));
        CHECK_EQ(got2[512 + k], uint8_t(host[k] + 1));
        uint16_t want16 = static_cast<uint16_t>(static_cast<int16_t>(h) ^ 0x5555);
        CHECK_EQ(got2[2 * k], uint8_t(want16 & 0xFF));
        CHECK_EQ(got2[2 * k + 1], uint8_t(want16 >> 8));
    }
}

TEST_CASE("PTX intrinsics - atom.add on global f32/i32 and shared f32") {
    // kernel(out): 256 threads: atom_add_f32(out[0], 1.5); atom_add_i32(out_i[1], tid);
    // shared: each thread atom-adds 2.0 into smem[0]; sync; thread 0 stores smem[0] to out[2].
    Module mod("atom");
    Function* f = mod.create_function("atom", Type::void_type(), {Type::ptr()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* smem = kb.shared_alloc_f32(4);
    Value* tid = kb.tid_x();
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] { kb.shared_store_f32(smem, kb.const_f32(0.0f)); });
    kb.sync();
    kb.atom_add_f32(out, kb.const_f32(1.5f));
    kb.atom_add_i32(b.build_add(out, kb.const_i64(4)), tid);
    b.build_call("ptx_atom_shared_add_f32", Type::f32(), {smem, kb.const_f32(2.0f)});
    kb.sync();
    kb.if_then(b.build_eq(tid, kb.const_i32(0)), [&] { kb.store_f32(out, kb.shared_load_f32(smem), 8); });
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    CHECK(ptx.find("atom.global.add.f32") != std::string::npos);
    CHECK(ptx.find("atom.global.add.u32") != std::string::npos);
    CHECK(ptx.find("atom.shared.add.f32") != std::string::npos);
    if (!gpu_ready()) return;
    CudaBuffer dout = CudaBuffer::alloc(16);
    REQUIRE(dout.valid() && dout.zero());
    void* po = dout.device_ptr();
    launch(ptx, "atom", 1, 256, {&po});
    auto got = download<uint32_t>(dout, 4);
    float f0, f2;
    std::memcpy(&f0, &got[0], 4);
    std::memcpy(&f2, &got[2], 4);
    CHECK_EQ(f0, 256 * 1.5f);
    CHECK_EQ(got[1], uint32_t(255 * 256 / 2));
    CHECK_EQ(f2, 512.0f);
}
