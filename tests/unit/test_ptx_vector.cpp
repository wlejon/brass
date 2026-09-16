// Stage 4 PTX vector lowering and KernelBuilder GPU helpers:
//   - every vector MIR opcode on f32x4 / f64x2 / i32x4 / f32x8 lowers to
//     per-lane PTX, assembles and matches a host reference on device,
//   - warp/block reductions, shared-memory round trips, shfl variants,
//     bar.sync ids, if_then / for_range,
//   - the ml_fusion gemv_q8_0 / gemv_q4_k MIR builders (previously blocked
//     on vzero / f32x8) lower and verify.

#include "ptx_test_support.hpp"

#include <brass/codegen/ml_fusion.hpp>

#include <numeric>

using namespace brass;
using brass::codegen::KernelBuilder;
using namespace ptxtest;

namespace {

// kernel(a, b, out): one thread; loads vectors from a and b and stores the
// results of `build` (a list of vector values) consecutively into out.
struct VecKernel {
    Module mod{"vec"};
    Function* fn = nullptr;

    VecKernel(Type vt, const std::function<std::vector<Value*>(KernelBuilder&, Value*, Value*)>& build) {
        fn = mod.create_function("vec_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr()});
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        Value* a = b.add_block_param(e, Type::ptr());
        Value* bb = b.add_block_param(e, Type::ptr());
        Value* out = b.add_block_param(e, Type::ptr());
        Value* va = kb.vload(vt, a);
        Value* vb = kb.vload(vt, bb);
        std::vector<Value*> results = build(kb, va, vb);
        int32_t off = 0;
        for (Value* r : results) {
            if (r->type().is_vector()) {
                kb.vstore(r->type(), out, off, r);
                off += static_cast<int32_t>(r->type().size_in_bytes());
            } else {
                b.build_store(r->type(), out, off, r);
                off += 16; // scalars take a full slot so the host indexing stays simple
            }
        }
        b.build_ret_void();
    }
};

template <typename T>
std::vector<T> run_vec(Type vt, const std::vector<T>& a, const std::vector<T>& b, size_t out_elems,
                       const std::function<std::vector<Value*>(KernelBuilder&, Value*, Value*)>& build,
                       std::string* ptx_out = nullptr) {
    VecKernel k(vt, build);
    std::string ptx = emit_checked(*k.fn);
    if (ptx_out) *ptx_out = ptx;
    if (!gpu_ready()) return {};
    CudaBuffer da = upload(a);
    CudaBuffer db = upload(b);
    CudaBuffer dout = CudaBuffer::alloc(out_elems * sizeof(T) + 64);
    REQUIRE(dout.valid() && dout.zero());
    void* pa = da.device_ptr(); void* pb = db.device_ptr(); void* po = dout.device_ptr();
    launch(ptx, "vec_kernel", 1, 1, {&pa, &pb, &po});
    return download<T>(dout, out_elems);
}

} // namespace

// ---------------------------------------------------------------------------
// Vector opcodes
// ---------------------------------------------------------------------------

TEST_CASE("PTX vector - f32x4 arithmetic, lanes, shuffle and zero lower per lane and run") {
    std::vector<float> a = {1.5f, -2.0f, 4.0f, 9.0f};
    std::vector<float> b = {0.5f, 4.0f, -1.0f, 3.0f};
    std::string ptx;
    // 13 vector results (4 floats each) + 1 scalar slot (4 floats) = 56 floats
    auto got = run_vec<float>(Type::f32x4(), a, b, 56, [](KernelBuilder& kb, Value* va, Value* vb) {
        Builder& bd = kb.builder();
        Value* s = kb.const_f32(7.25f);
        return std::vector<Value*>{
            kb.vadd(va, vb), kb.vsub(va, vb), kb.vmul(va, vb), kb.vdiv(va, vb),
            kb.vfma(va, vb, va), bd.build_vneg(va), kb.vmin(va, vb), kb.vmax(va, vb),
            bd.build_vsqrt(kb.vmul(va, va)), kb.vzero(Type::f32x4()), kb.vbroadcast(Type::f32x4(), s),
            bd.build_vinsert_lane(va, s, 1), bd.build_vshuffle(va, vb, 0x4E), // dst = {a[2], a[3], b[0], b[1]}
            bd.build_vextract_lane(vb, 2),
        };
    }, &ptx);
    for (const char* needle : {"add.f32", "sub.f32", "mul.f32", "div.rn.f32", "fma.rn.f32", "neg.f32", "min.f32", "max.f32",
                               "sqrt.rn.f32", "ld.global.v4.f32", "st.global.v4.f32"}) {
        if (ptx.find(needle) == std::string::npos) std::cerr << "missing " << needle << "\n" << ptx;
        CHECK(ptx.find(needle) != std::string::npos);
    }
    if (got.empty()) return;
    auto lane = [&](size_t result, size_t l) { return got[result * 4 + l]; };
    for (size_t l = 0; l < 4; ++l) {
        CHECK_EQ(lane(0, l), a[l] + b[l]);
        CHECK_EQ(lane(1, l), a[l] - b[l]);
        CHECK_EQ(lane(2, l), a[l] * b[l]);
        CHECK_EQ(lane(3, l), a[l] / b[l]);
        CHECK_EQ(lane(4, l), std::fma(a[l], b[l], a[l]));
        CHECK_EQ(lane(5, l), -a[l]);
        CHECK_EQ(lane(6, l), std::min(a[l], b[l]));
        CHECK_EQ(lane(7, l), std::max(a[l], b[l]));
        CHECK_EQ(lane(8, l), std::fabs(a[l]));
        CHECK_EQ(lane(9, l), 0.0f);
        CHECK_EQ(lane(10, l), 7.25f);
        CHECK_EQ(lane(11, l), l == 1 ? 7.25f : a[l]);
    }
    CHECK_EQ(lane(12, 0), a[2]); CHECK_EQ(lane(12, 1), a[3]); CHECK_EQ(lane(12, 2), b[0]); CHECK_EQ(lane(12, 3), b[1]);
    CHECK_EQ(lane(13, 0), b[2]);
}

TEST_CASE("PTX vector - f64x2 uses .v2 and f64 suffixes") {
    std::vector<double> a = {1.25, -3.0};
    std::vector<double> b = {0.5, 8.0};
    std::string ptx;
    auto got = run_vec<double>(Type::f64x2(), a, b, 10, [](KernelBuilder& kb, Value* va, Value* vb) {
        Builder& bd = kb.builder();
        return std::vector<Value*>{ kb.vadd(va, vb), kb.vfma(va, vb, vb), bd.build_vsqrt(kb.vmul(vb, vb)),
                                    bd.build_vshuffle(va, vb, 0x1), // dst = {a[1], b[0]}
                                    kb.vzero(Type::f64x2()) };
    }, &ptx);
    CHECK(ptx.find("ld.global.v2.f64") != std::string::npos);
    CHECK(ptx.find("fma.rn.f64") != std::string::npos);
    CHECK(ptx.find("sqrt.rn.f64") != std::string::npos);
    if (got.empty()) return;
    CHECK_EQ(got[0], a[0] + b[0]); CHECK_EQ(got[1], a[1] + b[1]);
    CHECK_EQ(got[2], std::fma(a[0], b[0], b[0])); CHECK_EQ(got[3], std::fma(a[1], b[1], b[1]));
    CHECK_EQ(got[4], std::fabs(b[0])); CHECK_EQ(got[5], std::fabs(b[1]));
    CHECK_EQ(got[6], a[1]); CHECK_EQ(got[7], b[0]);
    CHECK_EQ(got[8], 0.0); CHECK_EQ(got[9], 0.0);
}

TEST_CASE("PTX vector - i32x4 integer and bitwise lanes, v4.u32 loads") {
    std::vector<int32_t> a = {10, -20, 0x0F0F0F0F, 7};
    std::vector<int32_t> b = {3, 6, 0x00FF00FF, -2};
    std::string ptx;
    auto got = run_vec<int32_t>(Type::i32x4(), a, b, 40, [](KernelBuilder& kb, Value* va, Value* vb) {
        Builder& bd = kb.builder();
        return std::vector<Value*>{ kb.vadd(va, vb), kb.vsub(va, vb), kb.vmul(va, vb), kb.vdiv(va, vb),
                                    bd.build_vand(va, vb), bd.build_vor(va, vb), bd.build_vxor(va, vb), bd.build_vnot(va),
                                    kb.vfma(va, vb, va), kb.vmax(va, vb) };
    }, &ptx);
    for (const char* needle : {"ld.global.v4.u32", "add.s32", "mul.lo.s32", "div.s32", "and.b32", "or.b32", "xor.b32", "not.b32", "mad.lo.s32", "max.s32"}) {
        if (ptx.find(needle) == std::string::npos) std::cerr << "missing " << needle << "\n" << ptx;
        CHECK(ptx.find(needle) != std::string::npos);
    }
    if (got.empty()) return;
    for (size_t l = 0; l < 4; ++l) {
        CHECK_EQ(got[0 * 4 + l], a[l] + b[l]);
        CHECK_EQ(got[1 * 4 + l], a[l] - b[l]);
        CHECK_EQ(got[2 * 4 + l], a[l] * b[l]);
        CHECK_EQ(got[3 * 4 + l], a[l] / b[l]);
        CHECK_EQ(got[4 * 4 + l], a[l] & b[l]);
        CHECK_EQ(got[5 * 4 + l], a[l] | b[l]);
        CHECK_EQ(got[6 * 4 + l], a[l] ^ b[l]);
        CHECK_EQ(got[7 * 4 + l], ~a[l]);
        CHECK_EQ(got[8 * 4 + l], a[l] * b[l] + a[l]);
        CHECK_EQ(got[9 * 4 + l], std::max(a[l], b[l]));
    }
}

TEST_CASE("PTX vector - f32x8 is two v4 tuples 16 bytes apart") {
    std::vector<float> a(8), b(8);
    for (int i = 0; i < 8; ++i) { a[i] = 1.0f + static_cast<float>(i); b[i] = 0.5f * static_cast<float>(i); }
    std::string ptx;
    auto got = run_vec<float>(Type::f32x8(), a, b, 16, [](KernelBuilder& kb, Value* va, Value* vb) {
        return std::vector<Value*>{ kb.vfma(va, vb, kb.vzero(Type::f32x8())), kb.vsub(vb, va) };
    }, &ptx);
    CHECK(ptx.find("ld.global.v4.f32") != std::string::npos);
    CHECK(ptx.find("+ 16]") != std::string::npos);
    CHECK(ptx.find(".v8") == std::string::npos);
    if (got.empty()) return;
    for (size_t l = 0; l < 8; ++l) {
        CHECK_EQ(got[l], a[l] * b[l]);
        CHECK_EQ(got[8 + l], b[l] - a[l]);
    }
}

TEST_CASE("PTX vector - vector block arguments carry every lane through a loop") {
    // acc(f32x4) += a over 5 iterations via block params.
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f, 0.0f};
    Module mod("vloop");
    Function* fn = mod.create_function("vec_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    Builder& bd = kb.builder();
    BasicBlock* e = bd.append_block("entry");
    bd.position_at_end(e);
    Value* pa = bd.add_block_param(e, Type::ptr());
    bd.add_block_param(e, Type::ptr());
    Value* out = bd.add_block_param(e, Type::ptr());
    Value* va = kb.vload_f32x4(pa);
    BasicBlock* head = bd.append_block("head");
    Value* i = bd.add_block_param(head, Type::i32());
    Value* acc = bd.add_block_param(head, Type::f32x4());
    BasicBlock* body = bd.append_block("body");
    BasicBlock* done = bd.append_block("done");
    Value* facc = bd.add_block_param(done, Type::f32x4());
    bd.position_at_end(e);
    bd.build_br(head, {kb.const_i32(0), kb.vzero(Type::f32x4())});
    bd.position_at_end(head);
    bd.build_br_if(bd.build_slt(i, kb.const_i32(5)), body, {}, done, {acc});
    bd.position_at_end(body);
    bd.build_br(head, {kb.add(i, kb.const_i32(1)), kb.vadd(acc, va)});
    bd.position_at_end(done);
    kb.vstore_f32x4(out, facc);
    bd.build_ret_void();

    std::string ptx = emit_checked(*fn);
    if (!gpu_ready()) return;
    CudaBuffer da = upload(a), db = upload(b), dout = CudaBuffer::alloc(16);
    REQUIRE(dout.valid() && dout.zero());
    void* p0 = da.device_ptr(); void* p1 = db.device_ptr(); void* p2 = dout.device_ptr();
    launch(ptx, "vec_kernel", 1, 1, {&p0, &p1, &p2});
    auto got = download<float>(dout, 4);
    for (size_t l = 0; l < 4; ++l) CHECK_EQ(got[l], 5.0f * a[l]);
}

TEST_CASE("PTX vector - ml_fusion gemv_q8_0 / gemv_q4_k MIR builders lower and verify") {
    // These call CPU helpers (brass_dequant_*), which become plain PTX `call`s
    // to undeclared symbols, so ptxas is not expected to accept them; the
    // point is that vzero / f32x8 vload / vfma / vector block args all lower.
    codegen::MlFusionCompiler c;
    Module m8("q8");
    Function* q8 = c.build_gemv_q8_0(m8);
    ptx::Function l8 = lower_ok(*q8);
    Module m4("q4k");
    Function* q4 = c.build_gemv_q4_k(m4);
    ptx::Function l4 = lower_ok(*q4);
    std::string p8 = target::PtxTarget::emit_function(*q8);
    CHECK(p8.find("ld.global.v4.f32") != std::string::npos);
    CHECK(p8.find("fma.rn.f32") != std::string::npos);
    CHECK(p8.find("call ") != std::string::npos); // the CPU helper calls remain calls
}

// ---------------------------------------------------------------------------
// KernelBuilder helpers: shuffles, reductions, shared memory, control flow
// ---------------------------------------------------------------------------

TEST_CASE("PTX vector - shfl down/up/bfly/idx (f32 and i32) on two warps") {
    // out[tid*8 + k]: down1, up1, bfly1, idx3 as f32 of lane id; then i32 down4, bfly5, idx0, up (register delta 2)
    Module mod("shfl");
    Function* f = mod.create_function("shfl", Type::void_type(), {Type::ptr()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* tid = kb.tid_x();
    Value* v = kb.i32_to_f32(tid);
    Value* base = b.build_mul(tid, kb.const_i32(8));
    Value* reg_two = b.build_lshr(kb.ntid_x(), kb.const_i32(5)); // 64 >> 5 = 2: a register-valued delta
    Value* results[8] = {
        kb.shfl_down_f32(v, 1u), kb.shfl_up_f32(v, 1u), kb.shfl_bfly_f32(v, 1u), kb.shfl_idx_f32(v, 3u),
        kb.i32_to_f32(kb.shfl_down_i32(tid, 4u)), kb.i32_to_f32(kb.shfl_bfly_i32(tid, 5u)),
        kb.i32_to_f32(kb.shfl_idx_i32(tid, 0u)),
        b.build_call("ptx_shfl_up_f32", Type::f32(), {v, reg_two}),
    };
    for (int k = 0; k < 8; ++k) kb.store_f32_indexed(out, b.build_add(base, kb.const_i32(k)), results[k]);
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    CHECK(ptx.find("shfl.sync.up.b32") != std::string::npos);
    CHECK(ptx.find("shfl.sync.bfly.b32") != std::string::npos);
    CHECK(ptx.find("shfl.sync.idx.b32") != std::string::npos);
    if (!gpu_ready()) return;
    CudaBuffer dout = CudaBuffer::alloc(64 * 8 * 4);
    REQUIRE(dout.valid() && dout.zero());
    void* po = dout.device_ptr();
    launch(ptx, "shfl", 1, 64, {&po});
    auto got = download<float>(dout, 64 * 8);
    for (int t = 0; t < 64; ++t) {
        int lane = t % 32, wb = t - lane;
        const float* r = &got[t * 8];
        CHECK_EQ(r[0], float(lane + 1 < 32 ? t + 1 : t));
        CHECK_EQ(r[1], float(lane >= 1 ? t - 1 : t));
        CHECK_EQ(r[2], float(wb + (lane ^ 1)));
        CHECK_EQ(r[3], float(wb + 3));
        CHECK_EQ(r[4], float(lane + 4 < 32 ? t + 4 : t));
        CHECK_EQ(r[5], float(wb + (lane ^ 5)));
        CHECK_EQ(r[6], float(wb));
        CHECK_EQ(r[7], float(lane >= 2 ? t - 2 : t));
    }
}

namespace {

// kernel(in, out): x = in[tid]; total = block_reduce_sum(x); out[tid] = total;
// plus warp-level: out[ntid + tid] = warp_reduce_sum(x) (valid in lane 0)
std::string block_reduce_ptx() {
    static std::string cached;
    if (!cached.empty()) return cached;
    Module mod("bred");
    Function* f = mod.create_function("bred", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* in = b.add_block_param(e, Type::ptr());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* scratch = kb.shared_alloc_f32(32);
    Value* tid = kb.tid_x();
    Value* x = kb.load_f32_indexed(in, tid);
    Value* w = kb.warp_reduce_sum_f32(x);
    Value* total = kb.block_reduce_sum_f32(x, scratch);
    Value* total2 = kb.block_reduce_sum_f32(kb.mul(x, kb.const_f32(2.0f)), scratch); // scratch reuse
    kb.store_f32_indexed(out, tid, kb.add(total, total2));
    kb.store_f32_indexed(out, b.build_add(kb.ntid_x(), tid), w);
    b.build_ret_void();
    cached = emit_checked(*f);
    CHECK(cached.find("ld.shared.f32") != std::string::npos);
    CHECK(cached.find("st.shared.f32") != std::string::npos);
    CHECK(cached.find(".shared .align 16 .f32 smem_0[32]") != std::string::npos);
    return cached;
}

void check_block_reduce(uint32_t block, uint32_t active) {
    std::vector<float> in(block, 0.0f);
    for (uint32_t i = 0; i < active; ++i) in[i] = static_cast<float>(i % 17) + 0.25f;
    double sum = 0.0;
    for (float v : in) sum += v;
    CudaBuffer din = upload(in);
    CudaBuffer dout = CudaBuffer::alloc(block * 2 * 4);
    REQUIRE(dout.valid() && dout.zero());
    void* pi = din.device_ptr(); void* po = dout.device_ptr();
    launch(block_reduce_ptx(), "bred", 1, block, {&pi, &po});
    auto got = download<float>(dout, block * 2);
    for (uint32_t t = 0; t < block; ++t) {
        if (!near(got[t], static_cast<float>(3.0 * sum), 1e-5f)) std::cerr << "block " << block << " thread " << t << ": " << got[t] << " want " << 3.0 * sum << "\n";
        CHECK(near(got[t], static_cast<float>(3.0 * sum), 1e-5f));
    }
    for (uint32_t w = 0; w < block / 32; ++w) {
        double ws = 0.0;
        for (uint32_t l = 0; l < 32; ++l) ws += in[w * 32 + l];
        CHECK(near(got[block + w * 32], static_cast<float>(ws), 1e-5f));
    }
}

} // namespace

TEST_CASE("PTX vector - warp_reduce_sum / block_reduce_sum over 256 threads with 173 active, and 96 threads") {
    std::string ptx = block_reduce_ptx();
    CHECK(!ptx.empty());
    if (!gpu_ready()) return;
    check_block_reduce(256, 173);
    check_block_reduce(256, 256);
    check_block_reduce(96, 50);
    check_block_reduce(1024, 1000);
}

TEST_CASE("PTX vector - shared memory round trip with bar.sync id 1, f32 and i32 arrays in one kernel") {
    // smem_f[tid] = in[tid] * 2; smem_i[tid] = tid; bar.sync 1;
    // out[tid] = smem_f[127 - tid] + smem_i[127 - tid] (via byte-offset and indexed forms)
    Module mod("smem");
    Function* f = mod.create_function("smem", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* in = b.add_block_param(e, Type::ptr());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* smem_f = kb.shared_alloc_f32(128);
    Value* smem_i = kb.shared_alloc_i32(128);
    Value* tid = kb.tid_x();
    Value* x = kb.load_f32_indexed(in, tid);
    kb.shared_store_f32_indexed(smem_f, tid, kb.mul(x, kb.const_f32(2.0f)));
    kb.shared_store_i32_indexed(smem_i, tid, tid);
    kb.bar_sync(1);
    Value* mirror = kb.sub(kb.const_i32(127), tid);
    Value* fval = kb.shared_load_f32_indexed(smem_f, mirror);
    // i32 via the byte-offset form on a pointer advanced by arithmetic: (smem_i + mirror*4)
    Value* pi = b.build_add(smem_i, b.build_shl(b.build_sext_i64(mirror), kb.const_i64(2)));
    Value* ival = kb.shared_load_i32(pi, 0);
    // and the fixed-offset form: smem_f[1] via byte offset 4
    Value* one = kb.shared_load_f32(smem_f, 4);
    kb.store_f32_indexed(out, tid, kb.add(kb.add(fval, kb.i32_to_f32(ival)), one));
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    CHECK(ptx.find("bar.sync 1;") != std::string::npos);
    CHECK(ptx.find(".shared .align 16 .f32 smem_0[128]") != std::string::npos);
    CHECK(ptx.find(".shared .align 16 .u32 smem_1[128]") != std::string::npos);
    CHECK(ptx.find("mov.u64") != std::string::npos);
    if (!gpu_ready()) return;
    std::vector<float> host(128);
    for (int i = 0; i < 128; ++i) host[i] = 0.5f * static_cast<float>(i);
    CudaBuffer din = upload(host);
    CudaBuffer dout = CudaBuffer::alloc(128 * 4);
    REQUIRE(dout.valid() && dout.zero());
    void* p0 = din.device_ptr(); void* p1 = dout.device_ptr();
    launch(ptx, "smem", 1, 128, {&p0, &p1});
    auto got = download<float>(dout, 128);
    for (int t = 0; t < 128; ++t) CHECK_EQ(got[t], host[127 - t] * 2.0f + float(127 - t) + host[1] * 2.0f);
}

TEST_CASE("PTX vector - two kernels in one module each declare their own smem_0") {
    Module mod("two");
    for (const char* name : {"kernel_a", "kernel_b"}) {
        Function* f = mod.create_function(name, Type::void_type(), {Type::ptr()});
        KernelBuilder kb(mod, f);
        Builder& b = kb.builder();
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        Value* out = b.add_block_param(e, Type::ptr());
        Value* smem = kb.shared_alloc_f32(std::string(name) == "kernel_a" ? 16u : 48u);
        Value* tid = kb.tid_x();
        kb.shared_store_f32_indexed(smem, tid, kb.i32_to_f32(tid));
        kb.sync();
        kb.store_f32_indexed(out, tid, kb.shared_load_f32_indexed(smem, kb.sub(kb.const_i32(15), tid)));
        b.build_ret_void();
    }
    std::string ptx = target::PtxTarget::emit_module(mod);
    CHECK(ptx.find("smem_0[16]") != std::string::npos);
    CHECK(ptx.find("smem_0[48]") != std::string::npos);
    if (ptxas_available()) CHECK(ptxas_assembles(ptx));
    if (!gpu_ready()) return;
    std::string err;
    CudaModule m = CudaModule::load(ptx, &err);
    REQUIRE(m.valid());
    for (const char* name : {"kernel_a", "kernel_b"}) {
        CudaBuffer dout = CudaBuffer::alloc(16 * 4);
        REQUIRE(dout.valid() && dout.zero());
        void* po = dout.device_ptr();
        void* args[] = {&po};
        REQUIRE(m.launch_1d(name, 1, 16, args, 0, &err));
        auto got = download<float>(dout, 16);
        for (int t = 0; t < 16; ++t) CHECK_EQ(got[t], float(15 - t));
    }
}

TEST_CASE("PTX vector - if_then and for_range build correct control flow") {
    // single thread: for i in [0, n): out[i] = i * 3; if (i is odd) out[i] += 100
    Module mod("cf");
    Function* f = mod.create_function("cf", Type::void_type(), {Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* n = b.add_block_param(e, Type::i32());
    kb.for_range(kb.const_i32(0), n, kb.const_i32(1), [&](Value* i) {
        Value* v = kb.mul(i, kb.const_i32(3));
        kb.store_i32_indexed(out, i, v);
        Value* odd = b.build_and(i, kb.const_i32(1));
        kb.if_then(odd, [&] { kb.store_i32_indexed(out, i, kb.add(v, kb.const_i32(100))); });
    });
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    if (!gpu_ready()) return;
    CudaBuffer dout = CudaBuffer::alloc(10 * 4);
    REQUIRE(dout.valid() && dout.zero());
    void* po = dout.device_ptr();
    int32_t n_arg = 10;
    launch(ptx, "cf", 1, 1, {&po, &n_arg});
    auto got = download<int32_t>(dout, 10);
    for (int i = 0; i < 10; ++i) CHECK_EQ(got[i], i * 3 + (i % 2 ? 100 : 0));
}
