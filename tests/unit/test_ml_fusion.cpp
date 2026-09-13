#include "test_framework.hpp"
#include <brass/codegen/ml_fusion.hpp>

#include <vector>
#include <cmath>
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
        CHECK_NEAR(out[i], ref_out[i], 1e-4f);
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


