#include "test_framework.hpp"
#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/target.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

using namespace brass;
using namespace brass::codegen;

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))

TEST_CASE("ML Fusion - FusedResidualRmsNorm") {
    MlFusionCompiler compiler;
    KernelFunction kfn = compiler.compile_residual_rms_norm();
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<FusedResidualRmsNormFn>();
    CHECK(fn_ptr != nullptr);

    constexpr uint64_t n = 64;
    constexpr float eps = 1e-5f;

    std::vector<float> x(n);
    std::vector<float> residual(n);
    std::vector<float> weight(n);
    std::vector<float> out(n, 0.0f);

    std::vector<float> ref_residual(n);
    std::vector<float> ref_out(n);

    // Populate data
    float sum_sq = 0.0f;
    for (uint64_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.25f - 0.5f;
        residual[i] = static_cast<float>(i % 5) * 0.3f;
        weight[i] = 1.0f + static_cast<float>(i % 3) * 0.1f;

        ref_residual[i] = x[i] + residual[i];
        sum_sq += ref_residual[i] * ref_residual[i];
    }

    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(n)) + eps);
    for (uint64_t i = 0; i < n; ++i) {
        ref_out[i] = ref_residual[i] * inv_rms * weight[i];
    }

    run_residual_rms_norm(fn_ptr, x.data(), residual.data(), weight.data(), out.data(), n, eps);

    for (uint64_t i = 0; i < n; ++i) {
        CHECK_NEAR(residual[i], ref_residual[i], 1e-5f);
        CHECK_NEAR(out[i], ref_out[i], 1e-2f);
    }

    // Verify PTX generation
    std::string ptx = compiler.emit_ptx_residual_rms_norm();
    CHECK(ptx.find("rsqrt.approx.f32") != std::string::npos);
    CHECK(ptx.find("fused_residual_rms_norm") != std::string::npos);
}

TEST_CASE("ML Fusion - FusedSwiGLU") {
    MlFusionCompiler compiler;
    KernelFunction kfn = compiler.compile_swiglu();
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<FusedSwiGLUFn>();
    CHECK(fn_ptr != nullptr);

    constexpr uint64_t n = 64;
    std::vector<float> gate(n);
    std::vector<float> up(n);
    std::vector<float> out(n, 0.0f);
    std::vector<float> ref_out(n);

    for (uint64_t i = 0; i < n; ++i) {
        gate[i] = (static_cast<float>(i) - 32.0f) * 0.1f;
        up[i] = static_cast<float>(i % 8) * 0.5f - 1.0f;

        float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        ref_out[i] = silu * up[i];
    }

    fn_ptr(gate.data(), up.data(), out.data(), n);

    for (uint64_t i = 0; i < n; ++i) {
        CHECK_NEAR(out[i], ref_out[i], 1e-4f);
    }

    // Verify PTX generation
    std::string ptx = compiler.emit_ptx_swiglu();
    CHECK(ptx.find("ex2.approx.f32") != std::string::npos);
    CHECK(ptx.find("0f3FB8AA3B") != std::string::npos);
    CHECK(ptx.find("fused_swiglu") != std::string::npos);
}

TEST_CASE("ML Fusion - FusedAdaLNModulate") {
    MlFusionCompiler compiler;

    // 1. Ungated
    {
        KernelFunction kfn = compiler.compile_adaln_modulate(false);
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<FusedAdaLNModulateFn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t n = 64;
        std::vector<float> x(n);
        std::vector<float> scale(n);
        std::vector<float> shift(n);
        std::vector<float> out(n, 0.0f);
        std::vector<float> ref_out(n);

        for (uint64_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>(i) * 0.1f;
            scale[i] = 0.05f * static_cast<float>(i % 4);
            shift[i] = -0.1f * static_cast<float>(i % 3);
            ref_out[i] = x[i] * (1.0f + scale[i]) + shift[i];
        }

        fn_ptr(x.data(), scale.data(), shift.data(), out.data(), n);

        for (uint64_t i = 0; i < n; ++i) {
            CHECK_NEAR(out[i], ref_out[i], 1e-4f);
        }

        std::string ptx = compiler.emit_ptx_adaln_modulate(false);
        CHECK(ptx.find("fma.rn.f32") != std::string::npos);
        CHECK(ptx.find("fused_adaln_modulate") != std::string::npos);
    }

    // 2. Gated
    {
        KernelFunction kfn = compiler.compile_adaln_modulate(true);
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<FusedAdaLNModulateGatedFn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t n = 64;
        std::vector<float> x(n);
        std::vector<float> scale(n);
        std::vector<float> shift(n);
        std::vector<float> gate(n);
        std::vector<float> out(n, 0.0f);
        std::vector<float> ref_out(n);

        for (uint64_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>(i) * 0.1f;
            scale[i] = 0.05f * static_cast<float>(i % 4);
            shift[i] = -0.1f * static_cast<float>(i % 3);
            gate[i] = 0.5f + static_cast<float>(i % 5) * 0.1f;
            ref_out[i] = (x[i] * (1.0f + scale[i]) + shift[i]) * gate[i];
        }

        fn_ptr(x.data(), scale.data(), shift.data(), gate.data(), out.data(), n);

        for (uint64_t i = 0; i < n; ++i) {
            CHECK_NEAR(out[i], ref_out[i], 1e-4f);
        }

        std::string ptx = compiler.emit_ptx_adaln_modulate(true);
        CHECK(ptx.find("fma.rn.f32") != std::string::npos);
        CHECK(ptx.find("fused_adaln_modulate_gated") != std::string::npos);
    }
}

TEST_CASE("ML Fusion - QuantizedQ8DotProduct") {
    MlFusionCompiler compiler;

    // 1. Flat Q8 Dot Product
    {
        KernelFunction kfn = compiler.compile_q8_dot();
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<QuantizedQ8DotProductFn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t n = 64;
        std::vector<int8_t> a(n);
        std::vector<int8_t> b(n);
        float scale_a = 0.05f;
        float scale_b = 0.02f;
        float out = 0.0f;

        int32_t ref_acc = 0;
        for (uint64_t i = 0; i < n; ++i) {
            a[i] = static_cast<int8_t>((i * 7) % 256 - 128);
            b[i] = static_cast<int8_t>((i * 11) % 256 - 128);
            ref_acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        }
        float ref_dot = static_cast<float>(ref_acc) * scale_a * scale_b;

        fn_ptr(a.data(), b.data(), scale_a, scale_b, &out, n);
        CHECK_NEAR(out, ref_dot, 1e-3f);

        std::string ptx = compiler.emit_ptx_q8_dot();
        CHECK(ptx.find("fused_q8_dot") != std::string::npos);
        CHECK(ptx.find("cvt.rn.f32.s32") != std::string::npos);
    }

    // 2. Block Q8_0 Dot Product
    {
        KernelFunction kfn = compiler.compile_block_q8_dot();
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<BlockQ8DotProductFn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t num_blocks = 4;
        std::vector<BlockQ8_0> blocks_a(num_blocks);
        std::vector<BlockQ8_0> blocks_b(num_blocks);
        float out = 0.0f;

        float ref_total_dot = 0.0f;
        for (uint64_t blk = 0; blk < num_blocks; ++blk) {
            blocks_a[blk].d = 0.01f * static_cast<float>(blk + 1);
            blocks_b[blk].d = 0.02f * static_cast<float>(blk + 2);

            int32_t blk_acc = 0;
            for (size_t j = 0; j < 32; ++j) {
                blocks_a[blk].qs[j] = static_cast<int8_t>((blk * 32 + j * 5) % 256 - 128);
                blocks_b[blk].qs[j] = static_cast<int8_t>((blk * 32 + j * 9) % 256 - 128);
                blk_acc += static_cast<int32_t>(blocks_a[blk].qs[j]) * static_cast<int32_t>(blocks_b[blk].qs[j]);
            }
            ref_total_dot += static_cast<float>(blk_acc) * blocks_a[blk].d * blocks_b[blk].d;
        }

        fn_ptr(blocks_a.data(), blocks_b.data(), &out, num_blocks);
        CHECK_NEAR(out, ref_total_dot, 1e-2f);

        std::string ptx = compiler.emit_ptx_block_q8_dot();
        CHECK(ptx.find("fused_block_q8_dot") != std::string::npos);
        CHECK(ptx.find("cvt.rn.f32.s32") != std::string::npos);
    }
}

TEST_CASE("Fused GEMV PTX Generation") {
    MlFusionCompiler compiler;

    // 1. Fused GEMV SwiGLU PTX
    {
        std::string ptx = compiler.emit_ptx_fused_gemv_swiglu();
        CHECK(ptx.find("fused_gemv_swiglu_kernel") != std::string::npos);
        CHECK(ptx.find("ld.global.v4.f32") != std::string::npos);
        CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
        CHECK(ptx.find("ex2.approx.f32") != std::string::npos);
    }

    // 2. Fused GEMV Residual PTX
    {
        std::string ptx = compiler.emit_ptx_fused_gemv_residual();
        CHECK(ptx.find("fused_gemv_residual_kernel") != std::string::npos);
        CHECK(ptx.find("ld.global.v4.f32") != std::string::npos);
        CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
        CHECK(ptx.find("ld.global.f32") != std::string::npos);
    }

    // 3. Fused Q8_0 GEMV PTX
    {
        std::string ptx = compiler.emit_ptx_fused_gemv_q8_0();
        CHECK(ptx.find("fused_gemv_q8_0_kernel") != std::string::npos);
        CHECK(ptx.find("ld.global.u16") != std::string::npos);
        CHECK(ptx.find("cvt.f32.f16") != std::string::npos);
        CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
    }

    // 4. Fused Q4_K GEMV PTX
    {
        std::string ptx = compiler.emit_ptx_fused_gemv_q4_k();
        CHECK(ptx.find("fused_gemv_q4_k_kernel") != std::string::npos);
        CHECK(ptx.find("ld.global.v4.u32") != std::string::npos);
        CHECK(ptx.find("cvt.f32.f16") != std::string::npos);
        CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
    }

    // 5. Fused Residual LayerNorm PTX
    {
        std::string ptx = compiler.emit_ptx_fused_residual_layernorm();
        CHECK(ptx.find("fused_residual_layernorm_kernel") != std::string::npos);
        CHECK(ptx.find("ld.global.v4.f32") != std::string::npos);
        CHECK(ptx.find("st.global.v4.f32") != std::string::npos);
        CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
        CHECK(ptx.find("rsqrt.approx.f32") != std::string::npos);
    }
}

TEST_CASE("ML Fusion - FusedGemvQ8_0 and FusedGemvQ4_K CPU JIT") {
    MlFusionCompiler compiler;

    // 1. Q8_0 CPU GEMV JIT
    {
        KernelFunction kfn = compiler.compile_gemv_q8_0();
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<MlFusionCompiler::GemvQ8_0Fn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t n = 4;
        constexpr uint64_t k = 128;
        constexpr uint64_t bpr = k / 32;

        struct Q8Block {
            uint16_t d;
            int8_t   qs[32];
        };
        static_assert(sizeof(Q8Block) == 34, "Q8Block must be 34 bytes");

        std::vector<Q8Block> w(n * bpr);
        std::vector<float> x(k);
        std::vector<float> y(n, 0.0f);
        std::vector<float> ref_y(n, 0.0f);

        for (uint64_t i = 0; i < k; ++i) {
            x[i] = static_cast<float>(i % 11) * 0.1f - 0.5f;
        }

        for (uint64_t r = 0; r < n; ++r) {
            float acc = 0.0f;
            for (uint64_t b = 0; b < bpr; ++b) {
                Q8Block& blk = w[r * bpr + b];
                blk.d = 0x3800; // 0.5f in FP16
                float d_val = 0.5f;
                for (size_t j = 0; j < 32; ++j) {
                    blk.qs[j] = static_cast<int8_t>((r * 17 + b * 7 + j * 3) % 25 - 12);
                    float w_elem = d_val * static_cast<float>(blk.qs[j]);
                    acc += w_elem * x[b * 32 + j];
                }
            }
            ref_y[r] = acc;
        }

        fn_ptr(w.data(), x.data(), y.data(), n, k);

        for (uint64_t r = 0; r < n; ++r) {
            CHECK_NEAR(y[r], ref_y[r], 1e-4f);
        }
    }

    // 2. Q4_K CPU GEMV JIT
    {
        KernelFunction kfn = compiler.compile_gemv_q4_k();
        CHECK(kfn.is_valid());
        auto fn_ptr = kfn.as<MlFusionCompiler::GemvQ4_KFn>();
        CHECK(fn_ptr != nullptr);

        constexpr uint64_t n = 2;
        constexpr uint64_t k = 256;
        constexpr uint64_t bpr = k / 256;

        struct Q4KBlock {
            uint16_t d;
            uint16_t dmin;
            uint8_t  scales[12];
            uint8_t  qs[128];
        };
        static_assert(sizeof(Q4KBlock) == 144, "Q4KBlock must be 144 bytes");

        std::vector<Q4KBlock> w(n * bpr);
        std::vector<float> x(k);
        std::vector<float> y(n, 0.0f);
        std::vector<float> ref_y(n, 0.0f);

        for (uint64_t i = 0; i < k; ++i) {
            x[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
        }

        for (uint64_t r = 0; r < n; ++r) {
            float acc = 0.0f;
            for (uint64_t b = 0; b < bpr; ++b) {
                Q4KBlock& blk = w[r * bpr + b];
                blk.d = 0x3800;    // 0.5f in FP16
                blk.dmin = 0x3400; // 0.25f in FP16
                float d_val = 0.5f;
                float dmin_val = 0.25f;

                std::memset(blk.scales, 0, 12);
                for (int j = 0; j < 4; ++j) {
                    blk.scales[j] = 2;     // sc = 2
                    blk.scales[j + 4] = 1; // m = 1
                }
                for (int j = 4; j < 8; ++j) {
                    blk.scales[j + 4] = 0x12; // sc_lo = 2, m_lo = 1
                }

                uint8_t sc[8], m[8];
                for (int j = 0; j < 8; ++j) {
                    if (j < 4) {
                        sc[j] = blk.scales[j] & 0x3F;
                        m[j]  = blk.scales[j + 4] & 0x3F;
                    } else {
                        sc[j] = static_cast<uint8_t>((blk.scales[j + 4] & 0x0F) | ((blk.scales[j - 4] >> 6) << 4));
                        m[j]  = static_cast<uint8_t>((blk.scales[j + 4] >> 4)   | ((blk.scales[j - 0] >> 6) << 4));
                    }
                }

                for (int p = 0; p < 4; ++p) {
                    const int is_lo = 2 * p;
                    const int is_hi = 2 * p + 1;
                    const float w_lo = static_cast<float>(sc[is_lo]) * d_val;
                    const float w_hi = static_cast<float>(sc[is_hi]) * d_val;
                    const float b_lo = static_cast<float>(m [is_lo]) * dmin_val;
                    const float b_hi = static_cast<float>(m [is_hi]) * dmin_val;

                    for (int l = 0; l < 32; ++l) {
                        uint8_t n_lo = static_cast<uint8_t>((p * 8 + l) % 15);
                        uint8_t n_hi = static_cast<uint8_t>((p * 4 + l * 2) % 15);
                        blk.qs[p * 32 + l] = static_cast<uint8_t>((n_lo & 0x0F) | ((n_hi & 0x0F) << 4));

                        float w0 = w_lo * static_cast<float>(n_lo) - b_lo;
                        float w1 = w_hi * static_cast<float>(n_hi) - b_hi;
                        acc += w0 * x[is_lo * 32 + l];
                        acc += w1 * x[is_hi * 32 + l];
                    }
                }
            }
            ref_y[r] = acc;
        }

        fn_ptr(w.data(), x.data(), y.data(), n, k);

        for (uint64_t r = 0; r < n; ++r) {
            CHECK_NEAR(y[r], ref_y[r], 1e-4f);
        }
    }
}

TEST_CASE("ML Fusion - FusedSwiGLU Accuracy Across Wide Range & KernelJit::prepare_module") {
    MlFusionCompiler compiler;
    KernelFunction kfn = compiler.compile_swiglu();
    REQUIRE(kfn.is_valid());
    auto fn_ptr = kfn.as<FusedSwiGLUFn>();
    REQUIRE(fn_ptr != nullptr);

    // 1. Dense sweep across [-20, 20]: verify tight ULP accuracy for up = 1.0
    std::vector<float> gate;
    std::vector<float> up;
    for (int k = -400; k <= 400; ++k) {
        gate.push_back(static_cast<float>(k) * 0.05f);
        up.push_back(1.0f);
    }
    // Append extreme tails where old (1 + u)^16 catastrophically diverged
    const float tails[] = {-100.0f, -85.0f, -50.0f, -30.0f, 30.0f, 50.0f, 85.0f, 100.0f};
    for (float t : tails) {
        gate.push_back(t);
        up.push_back(1.0f);
    }

    std::vector<float> out(gate.size(), 0.0f);
    fn_ptr(gate.data(), up.data(), out.data(), gate.size());

    auto ulp_dist = [](float a, float b) -> uint32_t {
        if (a == b) return 0;
        int32_t ia = 0, ib = 0;
        std::memcpy(&ia, &a, 4);
        std::memcpy(&ib, &b, 4);
        if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
        if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
        return static_cast<uint32_t>(std::abs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib)));
    };

    for (size_t i = 0; i < gate.size(); ++i) {
        float g = gate[i];
        float ref = g / (1.0f + std::exp(-g));
        CHECK(std::isfinite(out[i]));
        if (std::abs(g) <= 20.0f && std::abs(ref) > 1e-6f) {
            CHECK(ulp_dist(out[i], ref) <= 6u);
            CHECK(std::abs(out[i] - ref) <= std::max(1e-6f, 2e-6f * std::abs(ref)));
        } else {
            CHECK(std::abs(out[i] - ref) <= std::max(1e-6f, 1e-5f * std::abs(ref)));
        }
    }

    // 2. Verify KernelJit::prepare_module optimizes and validates a module cleanly
    Module mod("prep_mod");
    Builder b(mod);
    Function* fn = mod.create_function("prep_add", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    Value* z = b.build_add(x, b.build_iconst_i32(0));
    b.build_ret(z);
    fn->rebuild_cfg_predecessors();

    KernelOptions kopts;
    kopts.opt_level = 2;
    KernelJit::prepare_module(mod, kopts);
    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("ML Fusion - FusedSwiGLU NEON Vectorization, Remainder Lengths & Exp Accuracy") {
    // 1. Verify KernelJit::prepare_module vectorizes build_swiglu cleanly with
    // hoisted broadcasts in _vec_preheader and 4-wide vector ops in _vec_body.
    {
        Module mod("swiglu_vec_check");
        MlFusionCompiler fusion;
        Function* fn = fusion.build_swiglu(mod, "fused_swiglu");
        KernelOptions kopts;
        kopts.opt_level = 2;
        kopts.enable_vectorize = true;
        kopts.enable_fp_reassociation = true;
        KernelJit::prepare_module(mod, kopts);

        bool saw_vec_preheader = false;
        bool saw_vec_body = false;
        bool saw_vload = false;
        bool saw_vround = false;
        bool saw_vfma = false;
        bool saw_vconvert = false;
        bool saw_vshl = false;
        bool saw_vbitcast = false;
        bool saw_vstore = false;
        bool saw_vbroadcast_in_preheader = false;
        bool saw_vbroadcast_in_vec_body = false;

        for (const BasicBlock* bb : fn->blocks()) {
            const std::string_view bname = bb->name();
            const bool in_pre = bname.find("_vec_preheader") != std::string_view::npos;
            const bool in_body = bname.find("_vec_body") != std::string_view::npos;
            if (in_pre) saw_vec_preheader = true;
            if (in_body) saw_vec_body = true;
            for (const Instruction* inst : *bb) {
                if (in_pre && inst->opcode() == Opcode::vbroadcast) saw_vbroadcast_in_preheader = true;
                if (in_body) {
                    if (inst->opcode() == Opcode::vbroadcast) saw_vbroadcast_in_vec_body = true;
                    if (inst->opcode() == Opcode::vload) saw_vload = true;
                    if (inst->opcode() == Opcode::vround) saw_vround = true;
                    if (inst->opcode() == Opcode::vfma) saw_vfma = true;
                    if (inst->opcode() == Opcode::vconvert) saw_vconvert = true;
                    if (inst->opcode() == Opcode::vshl) saw_vshl = true;
                    if (inst->opcode() == Opcode::vbitcast) saw_vbitcast = true;
                    if (inst->opcode() == Opcode::vstore) saw_vstore = true;
                }
            }
        }
        CHECK(saw_vec_preheader);
        CHECK(saw_vec_body);
        CHECK(saw_vbroadcast_in_preheader);
        CHECK(!saw_vbroadcast_in_vec_body);
        CHECK(saw_vload);
        CHECK(saw_vround);
        CHECK(saw_vfma);
        CHECK(saw_vconvert);
        CHECK(saw_vshl);
        CHECK(saw_vbitcast);
        CHECK(saw_vstore);
    }

    // 2. Compare Scalar JIT (enable_vectorize=false) vs Vectorized JIT (enable_vectorize=true)
    KernelOptions scalar_opts;
    scalar_opts.opt_level = 2;
    scalar_opts.enable_vectorize = false;
    scalar_opts.enable_slp = false;
    MlFusionCompiler scalar_comp(KernelJit(scalar_opts, Target::host()));
    KernelFunction scalar_kfn = scalar_comp.compile_swiglu();
    REQUIRE(scalar_kfn.is_valid());
    auto scalar_fn = scalar_kfn.as<FusedSwiGLUFn>();

    KernelOptions vec_opts;
    vec_opts.opt_level = 2;
    vec_opts.enable_vectorize = true;
    MlFusionCompiler vec_comp(KernelJit(vec_opts, Target::host()));
    KernelFunction vec_kfn = vec_comp.compile_swiglu();
    REQUIRE(vec_kfn.is_valid());
    auto vec_fn = vec_kfn.as<FusedSwiGLUFn>();

    auto ulp_dist = [](float a, float b) -> uint32_t {
        if (a == b) return 0;
        int32_t ia = 0, ib = 0;
        std::memcpy(&ia, &a, 4);
        std::memcpy(&ib, &b, 4);
        if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
        if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
        return static_cast<uint32_t>(std::abs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib)));
    };

    // Non-multiple-of-4 lengths with guard sentinels before/after output slice
    for (uint64_t len : {uint64_t{0}, uint64_t{1}, uint64_t{3}, uint64_t{4}, uint64_t{5},
                         uint64_t{15}, uint64_t{16}, uint64_t{17}, uint64_t{1025}}) {
        std::vector<float> gate(len + 8), up(len + 8);
        std::vector<float> out_scalar(len + 8, -999.0f), out_vec(len + 8, -999.0f);
        for (uint64_t i = 0; i < len + 8; ++i) {
            gate[i] = -12.0f + 0.025f * static_cast<float>(i % 960);
            up[i] = 0.5f + 0.1f * static_cast<float>(i % 7);
        }
        scalar_fn(gate.data(), up.data(), out_scalar.data() + 4, len);
        vec_fn(gate.data(), up.data(), out_vec.data() + 4, len);

        for (uint64_t i = 0; i < 4; ++i) {
            CHECK_EQ(out_vec[i], -999.0f);
            CHECK_EQ(out_vec[4 + len + i], -999.0f);
        }
        for (uint64_t i = 0; i < len; ++i) {
            float ref = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
            CHECK(std::isfinite(out_vec[4 + i]));
            CHECK(ulp_dist(out_vec[4 + i], out_scalar[4 + i]) <= 2u);
            if (std::abs(ref) > 1e-6f) {
                CHECK(std::abs(out_vec[4 + i] - ref) <= std::max(1e-6f, 2e-6f * std::abs(ref)));
            }
        }
    }

    // Dense accuracy sweep of vectorizable Cody-Waite expf(x) across [-87.3, 88.0]
    {
        Module mod("cody_waite_exp_vec");
        Function* fn = mod.create_function("exp_loop", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i64()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* px = b.add_block_param(entry, Type::ptr());
        Value* po = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i64());
        BasicBlock* hdr = b.create_block("hdr");
        BasicBlock* body = b.create_block("body");
        BasicBlock* exit_bb = b.create_block("exit");
        b.position_at_end(entry);
        b.build_br(hdr, {b.build_iconst_i64(0)});
        fn->append_block(hdr);
        Value* iv = b.add_block_param(hdr, Type::i64());
        b.position_at_end(hdr);
        b.build_br_if(b.build_slt(iv, n), body, exit_bb);
        fn->append_block(body);
        b.position_at_end(body);
        Value* xi = b.build_load_indexed(Type::f32(), px, iv, 4, 0);
        Value* x = b.build_fmin_f32(b.build_fmax_f32(xi, b.build_fconst_f32(-87.3f)), b.build_fconst_f32(88.0f));
        Value* n_fp = b.build_round_f32(b.build_mul(x, b.build_fconst_f32(1.4426950408889634f)));
        Value* r1 = b.build_fma_f32(n_fp, b.build_fconst_f32(-0.693145751953125f), x);
        Value* r = b.build_fma_f32(n_fp, b.build_fconst_f32(-1.428606765330187e-6f), r1);
        Value* p = b.build_fma_f32(r, b.build_fconst_f32(1.0f / 720.0f), b.build_fconst_f32(1.0f / 120.0f));
        p = b.build_fma_f32(r, p, b.build_fconst_f32(1.0f / 24.0f));
        p = b.build_fma_f32(r, p, b.build_fconst_f32(1.0f / 6.0f));
        p = b.build_fma_f32(r, p, b.build_fconst_f32(0.5f));
        p = b.build_fma_f32(r, p, b.build_fconst_f32(1.0f));
        p = b.build_fma_f32(r, p, b.build_fconst_f32(1.0f));
        Value* k = b.build_fptosi_i32_f32(n_fp);
        Value* scale = b.build_bitcast_f32_i32(b.build_shl(b.build_add(k, b.build_iconst_i32(127)),
                                                           b.build_iconst_i32(23)));
        b.build_store_indexed(Type::f32(), po, iv, 4, 0, b.build_mul(scale, p));
        b.build_br(hdr, {b.build_add(iv, b.build_iconst_i64(1))});
        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        KernelOptions exp_opts;
        exp_opts.opt_level = 2;
        exp_opts.enable_vectorize = true;
        KernelJit exp_jit(exp_opts, Target::host());
        KernelFunction exp_kfn = exp_jit.compile(*fn);
        REQUIRE(exp_kfn.is_valid());
        auto exp_fn = exp_kfn.as<void(*)(const float*, float*, int64_t)>();

        std::vector<float> xs;
        for (int s = -873; s <= 880; ++s) {
            xs.push_back(static_cast<float>(s) * 0.1f);
        }
        std::vector<float> ys(xs.size(), 0.0f);
        exp_fn(xs.data(), ys.data(), static_cast<int64_t>(xs.size()));

        uint32_t max_ulp = 0;
        for (size_t idx = 0; idx < xs.size(); ++idx) {
            float ref = std::exp(xs[idx]);
            uint32_t u = ulp_dist(ys[idx], ref);
            if (u > max_ulp) max_ulp = u;
        }
        CHECK(max_ulp < 4u);
    }
}



