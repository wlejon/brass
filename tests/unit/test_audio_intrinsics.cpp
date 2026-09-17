#include "test_framework.hpp"
#include <brass/codegen/audio_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/target.hpp>

#include <vector>
#include <cmath>
#include <chrono>

using namespace brass;
using namespace brass::codegen;

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))

TEST_CASE("Audio JIT - Scoped FTZ/DAZ mode prevents subnormal traps") {
    // Check toggle and state restoration
    bool prev = ScopedFtzDaz::is_ftz_daz_enabled();
    {
        ScopedFtzDaz guard;
        CHECK(ScopedFtzDaz::is_ftz_daz_enabled());

        // In FTZ mode, a tiny subnormal float multiplied by another should flush to zero
        volatile float small = 1e-30f;
        volatile float smaller = small * 1e-15f; // would be subnormal in normal IEEE-754
        CHECK(smaller == 0.0f);
    }
    CHECK(ScopedFtzDaz::is_ftz_daz_enabled() == prev);
}

TEST_CASE("Audio JIT - Fast pitch and exp2 approximation accuracy") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_exp2_mod");
    Function* fn = mod.create_function("exp2_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64()
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* n_val = ab.builder().add_block_param(entry, Type::i64());

    BasicBlock* hdr = ab.builder().create_block("loop_hdr");
    BasicBlock* body = ab.builder().create_block("loop_body");
    BasicBlock* exit = ab.builder().create_block("loop_exit");

    ab.position_at_end(entry);
    ab.builder().build_br(hdr, {ab.const_i64(0)});

    fn->append_block(hdr);
    ab.position_at_end(hdr);
    Value* iv = ab.builder().add_block_param(hdr, Type::i64());
    Value* cond = ab.builder().build_slt(iv, n_val);
    ab.builder().build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    ab.position_at_end(body);

    Value* val = ab.load_f32_indexed(in_p, iv, 4, 0);
    Value* res = ab.exp2_fast(val);
    ab.store_f32_indexed(out_p, iv, res, 4, 0);

    Value* next_iv = ab.builder().build_add(iv, ab.const_i64(1));
    ab.builder().build_br(hdr, {next_iv});

    fn->append_block(exit);
    ab.position_at_end(exit);
    ab.builder().build_ret_void();

    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, float*, int64_t)>();

    // Test a wide range of semitones / octaves: -10 to +10 octaves
    constexpr int N = 201;
    std::vector<float> input(N);
    std::vector<float> output(N);
    for (int i = 0; i < N; ++i) {
        input[i] = -10.0f + static_cast<float>(i) * 0.1f;
    }

    fn_ptr(input.data(), output.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::exp2(input[i]);
        float actual = output[i];
        float rel_diff = std::abs(actual - expected) / expected;
        if (i < 10) {
            std::cout << "DEBUG exp2: in=" << input[i] << " act=" << actual << " exp=" << expected << " rel=" << rel_diff << "\n";
        }
        CHECK(rel_diff < 5e-4f);
    }
}

TEST_CASE("Audio JIT - Oscillator sin and cos normalized waveform accuracy") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_sincos_mod");
    Function* fn = mod.create_function("sincos_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_phase = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_sin = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_cos = ab.builder().add_block_param(entry, Type::ptr());
    Value* n_val = ab.builder().add_block_param(entry, Type::i64());

    BasicBlock* hdr = ab.builder().create_block("hdr");
    BasicBlock* body = ab.builder().create_block("body");
    BasicBlock* exit = ab.builder().create_block("exit");

    ab.position_at_end(entry);
    ab.builder().build_br(hdr, {ab.const_i64(0)});

    fn->append_block(hdr);
    ab.position_at_end(hdr);
    Value* iv = ab.builder().add_block_param(hdr, Type::i64());
    Value* cond = ab.builder().build_slt(iv, n_val);
    ab.builder().build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    ab.position_at_end(body);

    Value* ph = ab.load_f32_indexed(in_phase, iv, 4, 0);
    Value* s = ab.sin_norm(ph);
    Value* c = ab.cos_norm(ph);
    ab.store_f32_indexed(out_sin, iv, s, 4, 0);
    ab.store_f32_indexed(out_cos, iv, c, 4, 0);

    Value* next_iv = ab.builder().build_add(iv, ab.const_i64(1));
    ab.builder().build_br(hdr, {next_iv});

    fn->append_block(exit);
    ab.position_at_end(exit);
    ab.builder().build_ret_void();

    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, float*, float*, int64_t)>();

    constexpr int N = 128;
    std::vector<float> phases(N);
    std::vector<float> sins(N);
    std::vector<float> coss(N);
    for (int i = 0; i < N; ++i) {
        phases[i] = static_cast<float>(i) / static_cast<float>(N);
    }

    fn_ptr(phases.data(), sins.data(), coss.data(), N);

    for (int i = 0; i < N; ++i) {
        double expected_s = std::sin(2.0 * M_PI * phases[i]);
        double expected_c = std::cos(2.0 * M_PI * phases[i]);
        CHECK_NEAR(sins[i], static_cast<float>(expected_s), 1e-5f);
        CHECK_NEAR(coss[i], static_cast<float>(expected_c), 1e-5f);
    }
}

TEST_CASE("Audio JIT - Soft clip tanh saturation matches reference") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_softclip_mod");
    Function* fn = mod.create_function("softclip_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64()
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* n_val = ab.builder().add_block_param(entry, Type::i64());

    BasicBlock* hdr = ab.builder().create_block("hdr");
    BasicBlock* body = ab.builder().create_block("body");
    BasicBlock* exit = ab.builder().create_block("exit");

    ab.position_at_end(entry);
    ab.builder().build_br(hdr, {ab.const_i64(0)});

    fn->append_block(hdr);
    ab.position_at_end(hdr);
    Value* iv = ab.builder().add_block_param(hdr, Type::i64());
    Value* cond = ab.builder().build_slt(iv, n_val);
    ab.builder().build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    ab.position_at_end(body);

    Value* val = ab.load_f32_indexed(in_p, iv, 4, 0);
    Value* res = ab.soft_clip(val);
    ab.store_f32_indexed(out_p, iv, res, 4, 0);

    Value* next_iv = ab.builder().build_add(iv, ab.const_i64(1));
    ab.builder().build_br(hdr, {next_iv});

    fn->append_block(exit);
    ab.position_at_end(exit);
    ab.builder().build_ret_void();

    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, float*, int64_t)>();

    constexpr int N = 100;
    std::vector<float> input(N);
    std::vector<float> output(N);
    for (int i = 0; i < N; ++i) {
        input[i] = -5.0f + 0.1f * i;
    }

    fn_ptr(input.data(), output.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::tanh(input[i]);
        CHECK_NEAR(output[i], expected, 5e-4f);
    }
}

TEST_CASE("Audio JIT - Biquad DF2T IIR filter matches C++ reference") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_biquad_mod");
    Function* fn = mod.create_function("biquad_kernel", Type::void_type(), {
        Type::ptr(), // in_p
        Type::ptr(), // out_p
        Type::ptr(), // coeffs: [b0, b1, b2, a1, a2]
        Type::ptr(), // state: [z1, z2]
        Type::i64()  // n
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* coeffs_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* state_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* n_val = ab.builder().add_block_param(entry, Type::i64());

    ab.position_at_end(entry);

    // Load filter coefficients once outside loop
    BiquadCoeffs c;
    c.b0 = ab.load_f32(coeffs_p, 0);
    c.b1 = ab.load_f32(coeffs_p, 4);
    c.b2 = ab.load_f32(coeffs_p, 8);
    c.a1 = ab.load_f32(coeffs_p, 12);
    c.a2 = ab.load_f32(coeffs_p, 16);

    // Load initial state
    Value* init_z1 = ab.load_f32(state_p, 0);
    Value* init_z2 = ab.load_f32(state_p, 4);

    BasicBlock* hdr = ab.builder().create_block("hdr");
    BasicBlock* body = ab.builder().create_block("body");
    BasicBlock* exit = ab.builder().create_block("exit");

    ab.builder().build_br(hdr, {ab.const_i64(0), init_z1, init_z2});

    fn->append_block(hdr);
    ab.position_at_end(hdr);
    Value* iv = ab.builder().add_block_param(hdr, Type::i64());
    Value* cur_z1 = ab.builder().add_block_param(hdr, Type::f32());
    Value* cur_z2 = ab.builder().add_block_param(hdr, Type::f32());

    Value* cond = ab.builder().build_slt(iv, n_val);
    ab.builder().build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    ab.position_at_end(body);

    Value* x = ab.load_f32_indexed(in_p, iv, 4, 0);
    BiquadState s;
    s.z1 = cur_z1;
    s.z2 = cur_z2;

    Value* y = ab.biquad_df2t(c, s, x);
    ab.store_f32_indexed(out_p, iv, y, 4, 0);

    Value* next_iv = ab.builder().build_add(iv, ab.const_i64(1));
    ab.builder().build_br(hdr, {next_iv, s.z1, s.z2});

    fn->append_block(exit);
    ab.position_at_end(exit);
    // Write back final state
    ab.store_f32(state_p, cur_z1, 0);
    ab.store_f32(state_p, cur_z2, 4);
    ab.builder().build_ret_void();

    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, float*, const float*, float*, int64_t)>();

    // Test Lowpass biquad: standard normalized coeffs
    float b0 = 0.067455f, b1 = 0.13491f, b2 = 0.067455f;
    float a1 = -1.14298f, a2 = 0.41280f;
    float coeffs[5] = {b0, b1, b2, a1, a2};
    float jit_state[2] = {0.0f, 0.0f};
    float ref_z1 = 0.0f, ref_z2 = 0.0f;

    constexpr int N = 256;
    std::vector<float> in(N);
    std::vector<float> jit_out(N);
    std::vector<float> ref_out(N);

    // Unit impulse followed by a sine burst
    in[0] = 1.0f;
    for (int i = 1; i < N; ++i) {
        in[i] = 0.5f * std::sin(0.1f * i);
    }

    // Reference C++ implementation
    for (int i = 0; i < N; ++i) {
        float x_samp = in[i];
        float y = b0 * x_samp + ref_z1;
        ref_z1 = b1 * x_samp - a1 * y + ref_z2;
        ref_z2 = b2 * x_samp - a2 * y;
        ref_out[i] = y;
    }

    fn_ptr(in.data(), jit_out.data(), coeffs, jit_state, N);

    for (int i = 0; i < N; ++i) {
        CHECK_NEAR(jit_out[i], ref_out[i], 1e-6f);
    }
    CHECK_NEAR(jit_state[0], ref_z1, 1e-6f);
    CHECK_NEAR(jit_state[1], ref_z2, 1e-6f);
}

TEST_CASE("Audio JIT - Stereo interleaved SIMD shuffling and deinterleaving") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_stereo_shuffle_mod");
    Function* fn = mod.create_function("stereo_roundtrip_kernel", Type::void_type(), {
        Type::ptr(), // in_stereo: 8 floats = 4 stereo frames
        Type::ptr()  // out_stereo: 8 floats = 4 stereo frames
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_p = ab.builder().add_block_param(entry, Type::ptr());

    ab.position_at_end(entry);

    // Load two f32x4 stereo vectors: s0=[L0,R0,L1,R1], s1=[L2,R2,L3,R3]
    Value* s0 = ab.vload_f32x4(in_p, 0);
    Value* s1 = ab.vload_f32x4(in_p, 16);

    // Deinterleave to [L0,L1,L2,L3] and [R0,R1,R2,R3]
    Value* left = nullptr;
    Value* right = nullptr;
    ab.deinterleave_stereo_f32x4(s0, s1, left, right);

    // Interleave back
    Value* out_s0 = nullptr;
    Value* out_s1 = nullptr;
    ab.interleave_stereo_f32x4(left, right, out_s0, out_s1);

    // Store back
    ab.vstore_f32x4(out_p, out_s0, 0);
    ab.vstore_f32x4(out_p, out_s1, 16);
    ab.builder().build_ret_void();

    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, float*)>();

    alignas(16) float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(16) float output[8] = {0.0f};

    fn_ptr(input, output);

    for (int i = 0; i < 8; ++i) {
        CHECK_NEAR(output[i], input[i], 1e-6f);
    }
}

TEST_CASE("Audio JIT - Real-time compile preset compile speed") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] on non-x86 host\n"; return; }

    Module mod("audio_speed_mod");
    Function* fn = mod.create_function("simple_gain_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::f32(), Type::i64()
    });

    AudioKernelBuilder ab(mod, fn);
    BasicBlock* entry = ab.builder().append_block("entry");
    Value* in_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* out_p = ab.builder().add_block_param(entry, Type::ptr());
    Value* gain = ab.builder().add_block_param(entry, Type::f32());
    Value* n_val = ab.builder().add_block_param(entry, Type::i64());

    BasicBlock* hdr = ab.builder().create_block("hdr");
    BasicBlock* body = ab.builder().create_block("body");
    BasicBlock* exit = ab.builder().create_block("exit");

    ab.position_at_end(entry);
    ab.builder().build_br(hdr, {ab.const_i64(0)});

    fn->append_block(hdr);
    ab.position_at_end(hdr);
    Value* iv = ab.builder().add_block_param(hdr, Type::i64());
    Value* cond = ab.builder().build_slt(iv, n_val);
    ab.builder().build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    ab.position_at_end(body);
    Value* x = ab.load_f32_indexed(in_p, iv, 4, 0);
    Value* y = ab.builder().build_mul(x, gain);
    ab.store_f32_indexed(out_p, iv, y, 4, 0);
    Value* next_iv = ab.builder().build_add(iv, ab.const_i64(1));
    ab.builder().build_br(hdr, {next_iv});

    fn->append_block(exit);
    ab.position_at_end(exit);
    ab.builder().build_ret_void();

    auto t0 = std::chrono::high_resolution_clock::now();
    KernelJit jit(KernelOptions::audio_realtime());
    KernelFunction kfn = jit.compile(*fn);
    auto t1 = std::chrono::high_resolution_clock::now();

    CHECK(kfn.is_valid());
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Verify compile latency is fast (< 10 ms even on debug/unoptimized runner)
    CHECK(elapsed_ms < 10.0);
}
