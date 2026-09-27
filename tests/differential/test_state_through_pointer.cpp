// Recurrences whose state lives in memory: each iteration loads the filter
// state through a pointer, computes, and stores it back through the same
// pointer, instead of carrying it in block parameters. broaudio's air-filter
// kernel was first written that way (per-voice state read and written
// through a section pointer every sample) and gave wrong results; these run
// the same shapes on every tier against a C++ reference.

#include "diff_harness.hpp"
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

using namespace brass;
using namespace brass::test;

namespace {

// What is being run, reported with a mismatch.
std::string g_tier;
int64_t g_n = 0;

void report_mismatch(bool same) {
    if (!same) std::cerr << "state-through-pointer mismatch on " << g_tier << " at n = " << g_n << "\n";
    CHECK(same);
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 1e-5f * std::max(1.0f, std::fabs(b));
}

std::vector<float> test_signal(size_t n, float amp) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = amp * std::sin(0.37f * static_cast<float>(i)) + (i == 0 ? 1.0f : 0.0f);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Scalar: a DF2T biquad over x in place, z1/z2 at st[0], st[1], coefficients
// b0 b1 b2 a1 a2 at c[0..4].
//   bq(x: ptr, st: ptr, c: ptr, n: i64)
std::unique_ptr<Module> build_biquad_state_in_memory() {
    auto mod = std::make_unique<Module>("bq_mem");
    Function* fn = mod->create_function("bq", Type::void_type(),
                                        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    Builder b(*mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* X = b.add_block_param(entry, Type::ptr());
    Value* ST = b.add_block_param(entry, Type::ptr());
    Value* C = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());
    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    Value* b0 = b.build_load(Type::f32(), C, 0);
    Value* b1 = b.build_load(Type::f32(), C, 4);
    Value* b2 = b.build_load(Type::f32(), C, 8);
    Value* a1 = b.build_load(Type::f32(), C, 12);
    Value* a2 = b.build_load(Type::f32(), C, 16);
    b.build_br(hdr, {b.build_iconst_i64(0)});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    b.build_br_if(b.build_slt(i, N), body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* v = b.build_load_indexed(Type::f32(), X, i, 4);
    Value* z1 = b.build_load(Type::f32(), ST, 0);
    Value* z2 = b.build_load(Type::f32(), ST, 4);
    Value* y = b.build_add(b.build_mul(b0, v), z1);
    Value* nz1 = b.build_add(b.build_sub(b.build_mul(b1, v), b.build_mul(a1, y)), z2);
    Value* nz2 = b.build_sub(b.build_mul(b2, v), b.build_mul(a2, y));
    b.build_store(Type::f32(), ST, 0, nz1);
    b.build_store(Type::f32(), ST, 4, nz2);
    b.build_store_indexed(Type::f32(), X, i, 4, y);
    b.build_br(hdr, {b.build_add(i, b.build_iconst_i64(1))});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return mod;
}

void biquad_reference(std::vector<float>& x, float st[2], const float c[5]) {
    for (float& v : x) {
        const float y = c[0] * v + st[0];
        const float nz1 = c[1] * v - c[3] * y + st[1];
        const float nz2 = c[2] * v - c[4] * y;
        st[0] = nz1;
        st[1] = nz2;
        v = y;
    }
}

// ---------------------------------------------------------------------------
// f32x8: broaudio's air sections with the lowpass output z and previous input
// xp kept in the section (loaded and stored every frame) while the pole and
// mix ramps are carried. Section k is at sec + k * kSectionBytes:
//   p[8] dp[8] m[8] dm[8] z[8] xp[8]
// Frames are 8 lanes at x + i * 32; the sections run one after another over
// the whole block.
//   air(x: ptr, sec: ptr, n: i32)
constexpr int kLanes = 8;
constexpr int kSections = 3;
constexpr int32_t kP = 0, kDp = 32, kM = 64, kDm = 96, kZ = 128, kXp = 160;
constexpr int32_t kSectionBytes = 192;

std::unique_ptr<Module> build_air_state_in_memory() {
    auto mod = std::make_unique<Module>("air_mem");
    mod->set_allow_fp_reassociation(false);
    Function* fn = mod->create_function("air", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    Builder b(*mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* X = b.add_block_param(entry, Type::ptr());
    Value* SEC = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i32());
    const Type v8 = Type::f32x8();
    Value* half = b.build_vbroadcast(v8, b.build_fconst_f32(0.5f));
    Value* one = b.build_vbroadcast(v8, b.build_fconst_f32(1.0f));
    Value* lane_bytes = b.build_iconst_i64(4 * kLanes);

    for (int k = 0; k < kSections; ++k) {
        const int32_t base = kSectionBytes * k;
        BasicBlock* hdr = b.create_block("hdr" + std::to_string(k));
        BasicBlock* body = b.create_block("body" + std::to_string(k));
        BasicBlock* done = b.create_block("done" + std::to_string(k));
        Value* dp = b.build_vload(v8, SEC, base + kDp);
        Value* dm = b.build_vload(v8, SEC, base + kDm);
        Value* p0 = b.build_vload(v8, SEC, base + kP);
        Value* m0 = b.build_vload(v8, SEC, base + kM);
        b.build_br(hdr, {b.build_iconst_i32(0), X, p0, m0});

        fn->append_block(hdr);
        b.position_at_end(hdr);
        Value* i = b.add_block_param(hdr, Type::i32());
        Value* ptr = b.add_block_param(hdr, Type::ptr());
        Value* pc = b.add_block_param(hdr, v8);
        Value* mc = b.add_block_param(hdr, v8);
        b.build_br_if(b.build_slt(i, N), body, {}, done, {});

        fn->append_block(body);
        b.position_at_end(body);
        Value* p = b.build_vadd(pc, dp);
        Value* m = b.build_vadd(mc, dm);
        Value* v = b.build_vload(v8, ptr, 0);
        Value* zs = b.build_vload(v8, SEC, base + kZ);
        Value* xp = b.build_vload(v8, SEC, base + kXp);
        // z = 0.5 (1 - p) (v + xp) + p z;  xp = v;  v += m (z - v)
        Value* z = b.build_vadd(b.build_vmul(b.build_vmul(half, b.build_vsub(one, p)), b.build_vadd(v, xp)),
                                b.build_vmul(p, zs));
        Value* out = b.build_vadd(v, b.build_vmul(m, b.build_vsub(z, v)));
        b.build_vstore(v8, SEC, base + kZ, z);
        b.build_vstore(v8, SEC, base + kXp, v);
        b.build_vstore(v8, ptr, 0, out);
        b.build_br(hdr, {b.build_add(i, b.build_iconst_i32(1)), b.build_add(ptr, lane_bytes), p, m});

        fn->append_block(done);
        b.position_at_end(done);
        b.build_vstore(v8, SEC, base + kP, pc);
        b.build_vstore(v8, SEC, base + kM, mc);
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return mod;
}

struct AirSection {
    float p[kLanes], dp[kLanes], m[kLanes], dm[kLanes], z[kLanes], xp[kLanes];
};
static_assert(sizeof(AirSection) == kSectionBytes);

void air_reference(std::vector<float>& x, AirSection* sec, int n) {
    for (int k = 0; k < kSections; ++k) {
        AirSection& s = sec[k];
        for (int i = 0; i < n; ++i) {
            for (int l = 0; l < kLanes; ++l) {
                s.p[l] = s.p[l] + s.dp[l];
                s.m[l] = s.m[l] + s.dm[l];
                const float v = x[static_cast<size_t>(i) * kLanes + l];
                const float z = (0.5f * (1.0f - s.p[l])) * (v + s.xp[l]) + s.p[l] * s.z[l];
                s.z[l] = z;
                s.xp[l] = v;
                x[static_cast<size_t>(i) * kLanes + l] = v + s.m[l] * (z - v);
            }
        }
    }
}

void init_sections(AirSection* sec) {
    for (int k = 0; k < kSections; ++k) {
        for (int l = 0; l < kLanes; ++l) {
            sec[k].p[l] = 0.2f + 0.05f * static_cast<float>(k) + 0.01f * static_cast<float>(l);
            sec[k].dp[l] = 0.0005f * static_cast<float>(l + 1);
            sec[k].m[l] = 0.3f + 0.02f * static_cast<float>(l);
            sec[k].dm[l] = -0.0003f * static_cast<float>(k + 1);
            sec[k].z[l] = 0.01f * static_cast<float>(l - k);
            sec[k].xp[l] = -0.02f * static_cast<float>(l);
        }
    }
}

// Each native form a kernel reaches: the plain JIT, tier 2 installed in a
// program, and the kernel JIT with the options broaudio compiles with, and
// with its defaults (FMA contraction and unrolling on).
// ---------------------------------------------------------------------------
// Scalar, per voice: the air stage of broaudio's chain kernel for a stereo
// voice, four sections per channel, with each (channel, section)'s z and xp
// read and written through the state pointer every frame and the section
// ramps (pole, mix) carried. More ramps than there are XMM registers, an i32
// frame counter and i32 interleaved indices, and the result added into a bus:
//   chain(x: ptr, st: ptr, bus: ptr, n: i32)
// st holds p[4] dp[4] m[4] dm[4], then z/xp pairs at 64 + (c * 4 + k) * 8.
constexpr int kChainSections = 4;
constexpr int32_t kChainState = 64;

std::unique_ptr<Module> build_chain_state_in_memory() {
    auto mod = std::make_unique<Module>("chain_mem");
    mod->set_allow_fp_reassociation(false);
    Function* fn = mod->create_function("chain", Type::void_type(),
                                        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    Builder b(*mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* X = b.add_block_param(entry, Type::ptr());
    Value* ST = b.add_block_param(entry, Type::ptr());
    Value* BUS = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i32());
    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    Value* half = b.build_fconst_f32(0.5f);
    Value* one = b.build_fconst_f32(1.0f);
    Value* dp[kChainSections];
    Value* dm[kChainSections];
    std::vector<Value*> init{b.build_iconst_i32(0)};
    for (int k = 0; k < kChainSections; ++k) {
        init.push_back(b.build_load(Type::f32(), ST, 4 * k));
        dp[k] = b.build_load(Type::f32(), ST, 16 + 4 * k);
        init.push_back(b.build_load(Type::f32(), ST, 32 + 4 * k));
        dm[k] = b.build_load(Type::f32(), ST, 48 + 4 * k);
    }
    b.build_br(hdr, init);

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i32());
    Value* pc[kChainSections];
    Value* mc[kChainSections];
    for (int k = 0; k < kChainSections; ++k) {
        pc[k] = b.add_block_param(hdr, Type::f32());
        mc[k] = b.add_block_param(hdr, Type::f32());
    }
    std::vector<Value*> finals;
    for (int k = 0; k < kChainSections; ++k) {
        finals.push_back(pc[k]);
        finals.push_back(mc[k]);
    }
    b.build_br_if(b.build_slt(i, N), body, {}, exit, finals);

    fn->append_block(body);
    b.position_at_end(body);
    Value* p[kChainSections];
    Value* m[kChainSections];
    for (int k = 0; k < kChainSections; ++k) {
        p[k] = b.build_add(pc[k], dp[k]);
        m[k] = b.build_add(mc[k], dm[k]);
    }
    Value* iL = b.build_add(i, i);
    Value* idx[2] = {iL, b.build_add(iL, b.build_iconst_i32(1))};
    for (int c = 0; c < 2; ++c) {
        Value* v = b.build_load_indexed(Type::f32(), X, idx[c], 4);
        for (int k = 0; k < kChainSections; ++k) {
            const int32_t at = kChainState + (c * kChainSections + k) * 8;
            Value* zs = b.build_load(Type::f32(), ST, at);
            Value* xp = b.build_load(Type::f32(), ST, at + 4);
            Value* z = b.build_add(b.build_mul(b.build_mul(half, b.build_sub(one, p[k])), b.build_add(v, xp)),
                                   b.build_mul(p[k], zs));
            b.build_store(Type::f32(), ST, at, z);
            b.build_store(Type::f32(), ST, at + 4, v);
            v = b.build_add(v, b.build_mul(m[k], b.build_sub(z, v)));
        }
        b.build_store_indexed(Type::f32(), BUS, idx[c], 4,
                              b.build_add(b.build_load_indexed(Type::f32(), BUS, idx[c], 4), v));
    }
    std::vector<Value*> next{b.build_add(i, b.build_iconst_i32(1))};
    for (int k = 0; k < kChainSections; ++k) {
        next.push_back(p[k]);
        next.push_back(m[k]);
    }
    b.build_br(hdr, next);

    fn->append_block(exit);
    b.position_at_end(exit);
    std::vector<Value*> out;
    for (int k = 0; k < 2 * kChainSections; ++k) out.push_back(b.add_block_param(exit, Type::f32()));
    for (int k = 0; k < kChainSections; ++k) {
        b.build_store(Type::f32(), ST, 4 * k, out[2 * k]);
        b.build_store(Type::f32(), ST, 32 + 4 * k, out[2 * k + 1]);
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return mod;
}

constexpr size_t kChainFloats = kChainState / 4 + 2 * kChainSections * 2;

void chain_reference(const std::vector<float>& x, float* st, std::vector<float>& bus, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < kChainSections; ++k) {
            st[k] = st[k] + st[4 + k];
            st[8 + k] = st[8 + k] + st[12 + k];
        }
        for (int c = 0; c < 2; ++c) {
            float v = x[static_cast<size_t>(2 * i + c)];
            for (int k = 0; k < kChainSections; ++k) {
                float* s = st + kChainState / 4 + (c * kChainSections + k) * 2;
                const float z = (0.5f * (1.0f - st[k])) * (v + s[1]) + st[k] * s[0];
                s[0] = z;
                s[1] = v;
                v = v + st[8 + k] * (z - v);
            }
            bus[static_cast<size_t>(2 * i + c)] = bus[static_cast<size_t>(2 * i + c)] + v;
        }
    }
}

void init_chain_state(float* st) {
    for (int k = 0; k < kChainSections; ++k) {
        st[k] = 0.15f + 0.1f * static_cast<float>(k);
        st[4 + k] = 0.0007f * static_cast<float>(k + 1);
        st[8 + k] = 0.25f + 0.05f * static_cast<float>(k);
        st[12 + k] = -0.0004f * static_cast<float>(k + 1);
    }
    for (size_t j = kChainState / 4; j < kChainFloats; ++j) st[j] = 0.003f * static_cast<float>(j) - 0.05f;
}

// Every kernel here takes at most four pointer / integer arguments and
// returns nothing, so each is called as one taking four integer registers.
using NativeFn = void (*)(uint64_t, uint64_t, uint64_t, uint64_t);

template <typename Run>
void for_each_native(Module& mod, const char* name, Run&& run) {
    for (DiffTier t : {DiffTier::Jit, DiffTier::Tier2}) {
        g_tier = diff_tier_name(t);
        run([&](const std::vector<RuntimeValue>& args) { run_on_tier(mod, name, args, t); });
    }
    for (bool broaudio_opts : {true, false}) {
        g_tier = broaudio_opts ? "kernel jit (broaudio options)" : "kernel jit (audio_realtime)";
        auto clone = clone_module(mod);
        codegen::KernelOptions opts = codegen::KernelOptions::audio_realtime();
        if (broaudio_opts) {
            opts.enable_fma = false;
            opts.enable_fp_reassociation = false;
            opts.enable_unroll = false;
        }
        codegen::KernelJit jit(opts);
        codegen::KernelFunction kfn = jit.compile(*clone, name);
        REQUIRE(kfn.is_valid());
        NativeFn f = kfn.as<NativeFn>();
        run([&](const std::vector<RuntimeValue>& args) {
            uint64_t raw[4] = {};
            for (size_t i = 0; i < args.size() && i < 4; ++i) raw[i] = args[i].raw_bits();
            f(raw[0], raw[1], raw[2], raw[3]);
        });
    }
}

} // namespace

TEST_CASE("State through a pointer - a biquad loading and storing z1/z2 each sample") {
    auto mod = build_biquad_state_in_memory();
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    const float c[5] = {0.067455f, 0.13491f, 0.067455f, -1.14298f, 0.41280f};
    for (size_t n : {size_t{1}, size_t{7}, size_t{64}, size_t{257}}) {
        g_n = static_cast<int64_t>(n);
        std::vector<float> want = test_signal(n, 0.5f);
        float want_st[2] = {0.03f, -0.01f};
        biquad_reference(want, want_st, c);

        auto check = [&](const std::function<void(const std::vector<RuntimeValue>&)>& call) {
            std::vector<float> x = test_signal(n, 0.5f);
            float st[2] = {0.03f, -0.01f};
            call({RuntimeValue::from_ptr(x.data()), RuntimeValue::from_ptr(st), RuntimeValue::from_ptr(c),
                  RuntimeValue::from_i64(static_cast<int64_t>(n))});
            bool same = close_enough(st[0], want_st[0]) && close_enough(st[1], want_st[1]);
            for (size_t i = 0; i < n; ++i) same = same && close_enough(x[i], want[i]);
            report_mismatch(same);
        };
        for (DiffTier t : {DiffTier::Interp, DiffTier::Fast, DiffTier::Baseline}) {
            g_tier = diff_tier_name(t);
            check([&](const std::vector<RuntimeValue>& args) { run_on_tier(*mod, "bq", args, t); });
        }
        for_each_native(*mod, "bq", check);
    }
}

TEST_CASE("State through a pointer - f32x8 air sections loading and storing z/xp each frame") {
    auto mod = build_air_state_in_memory();
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    for (int n : {1, 5, 64, 128}) {
        g_n = n;
        std::vector<float> want = test_signal(static_cast<size_t>(n) * kLanes, 0.8f);
        AirSection want_sec[kSections];
        init_sections(want_sec);
        air_reference(want, want_sec, n);

        auto check = [&](const std::function<void(const std::vector<RuntimeValue>&)>& call) {
            std::vector<float> x = test_signal(static_cast<size_t>(n) * kLanes, 0.8f);
            AirSection sec[kSections];
            init_sections(sec);
            call({RuntimeValue::from_ptr(x.data()), RuntimeValue::from_ptr(sec), RuntimeValue::from_i32(n)});
            bool same = true;
            for (size_t i = 0; i < x.size(); ++i) same = same && close_enough(x[i], want[i]);
            const float* got = &sec[0].p[0];
            const float* ref = &want_sec[0].p[0];
            for (size_t i = 0; i < sizeof(sec) / sizeof(float); ++i) same = same && close_enough(got[i], ref[i]);
            report_mismatch(same);
        };
        for_each_native(*mod, "air", check);
    }
}

TEST_CASE("State through a pointer - a stereo air chain reading and writing its sections each frame") {
    auto mod = build_chain_state_in_memory();
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    for (int n : {1, 3, 64, 255}) {
        g_n = n;
        const std::vector<float> x = test_signal(static_cast<size_t>(2 * n), 0.7f);
        std::vector<float> want_bus(static_cast<size_t>(2 * n), 0.25f);
        float want_st[kChainFloats];
        init_chain_state(want_st);
        chain_reference(x, want_st, want_bus, n);

        auto check = [&](const std::function<void(const std::vector<RuntimeValue>&)>& call) {
            std::vector<float> bus(static_cast<size_t>(2 * n), 0.25f);
            float st[kChainFloats];
            init_chain_state(st);
            call({RuntimeValue::from_ptr(x.data()), RuntimeValue::from_ptr(st), RuntimeValue::from_ptr(bus.data()),
                  RuntimeValue::from_i32(n)});
            bool same = true;
            for (size_t i = 0; i < bus.size(); ++i) same = same && close_enough(bus[i], want_bus[i]);
            for (size_t i = 0; i < kChainFloats; ++i) same = same && close_enough(st[i], want_st[i]);
            report_mismatch(same);
        };
        for (DiffTier t : {DiffTier::Interp, DiffTier::Fast, DiffTier::Baseline}) {
            g_tier = diff_tier_name(t);
            check([&](const std::vector<RuntimeValue>& args) { run_on_tier(*mod, "chain", args, t); });
        }
        for_each_native(*mod, "chain", check);
    }
}
