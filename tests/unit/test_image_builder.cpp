#include "test_framework.hpp"
#include <brass/codegen/image_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/target.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace brass;
using namespace brass::codegen;

namespace {

// ── C++ Scalar References ───────────────────────────────────────────────────

void ref_fused_preproc(
    const uint8_t* src, float* dst_planar,
    int32_t src_w, int32_t src_h,
    int32_t dst_w, int32_t dst_h,
    int32_t y_start, int32_t y_end,
    const float* mean, const float* inv_std,
    int channels = 3
) {
    const float xs = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float ys = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const int plane_size = dst_w * dst_h;

    for (int y = y_start; y < y_end; ++y) {
        float fy = (static_cast<float>(y) + 0.5f) * ys - 0.5f;
        float floor_y = std::floor(fy);
        int iy0 = static_cast<int>(floor_y);
        int iy1 = iy0 + 1;
        float ty_raw = fy - floor_y;

        int y0 = std::max(0, std::min(iy0, src_h - 1));
        int y1 = std::max(0, std::min(iy1, src_h - 1));
        float ty = std::max(0.0f, std::min(ty_raw, 1.0f));

        const int y0_stride = y0 * src_w;
        const int y1_stride = y1 * src_w;
        const int dst_row_offset = y * dst_w;

        for (int x = 0; x < dst_w; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * xs - 0.5f;
            float floor_x = std::floor(fx);
            int ix0 = static_cast<int>(floor_x);
            int ix1 = ix0 + 1;
            float tx_raw = fx - floor_x;

            int x0 = std::max(0, std::min(ix0, src_w - 1));
            int x1 = std::max(0, std::min(ix1, src_w - 1));
            float tx = std::max(0.0f, std::min(tx_raw, 1.0f));

            const uint8_t* p00 = src + (y0_stride + x0) * channels;
            const uint8_t* p10 = src + (y0_stride + x1) * channels;
            const uint8_t* p01 = src + (y1_stride + x0) * channels;
            const uint8_t* p11 = src + (y1_stride + x1) * channels;

            const float omtx = 1.0f - tx;
            const float omty = 1.0f - ty;

            for (int c = 0; c < channels; ++c) {
                float c00_f = static_cast<float>(p00[c]);
                float c10_f = static_cast<float>(p10[c]);
                float c01_f = static_cast<float>(p01[c]);
                float c11_f = static_cast<float>(p11[c]);

                float top = c00_f * omtx + c10_f * tx;
                float bot = c01_f * omtx + c11_f * tx;
                float interp = top * omty + bot * ty;

                float val_01 = interp * (1.0f / 255.0f);
                float norm = (val_01 - mean[c]) * inv_std[c];

                dst_planar[c * plane_size + dst_row_offset + x] = norm;
            }
        }
    }
}

void ref_resize_rgba8(
    const uint8_t* src, uint8_t* dst,
    int32_t src_w, int32_t src_h,
    int32_t dst_w, int32_t dst_h,
    int32_t y_start, int32_t y_end,
    bool premultiply_alpha
) {
    const float xs = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float ys = static_cast<float>(src_h) / static_cast<float>(dst_h);

    auto premul = [](uint8_t c, uint8_t a) -> uint8_t {
        return static_cast<uint8_t>((static_cast<int>(c) * a + 127) / 255);
    };
    auto unpremul = [](uint8_t c, uint8_t a) -> uint8_t {
        if (a == 0) return 0;
        int val = (static_cast<int>(c) * 255 + a / 2) / a;
        return static_cast<uint8_t>(std::min(255, val));
    };

    for (int y = y_start; y < y_end; ++y) {
        float fy = (static_cast<float>(y) + 0.5f) * ys - 0.5f;
        float floor_y = std::floor(fy);
        int iy0 = static_cast<int>(floor_y);
        int iy1 = iy0 + 1;
        float ty_raw = fy - floor_y;

        int y0 = std::max(0, std::min(iy0, src_h - 1));
        int y1 = std::max(0, std::min(iy1, src_h - 1));
        float ty = std::max(0.0f, std::min(ty_raw, 1.0f));

        const int dst_row_offset = y * dst_w;

        for (int x = 0; x < dst_w; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * xs - 0.5f;
            float floor_x = std::floor(fx);
            int ix0 = static_cast<int>(floor_x);
            int ix1 = ix0 + 1;
            float tx_raw = fx - floor_x;

            int x0 = std::max(0, std::min(ix0, src_w - 1));
            int x1 = std::max(0, std::min(ix1, src_w - 1));
            float tx = std::max(0.0f, std::min(tx_raw, 1.0f));

            auto get_pixel = [&](int px, int py, float out[4]) {
                const uint8_t* p = src + (py * src_w + px) * 4;
                uint8_t r = p[0], g = p[1], b = p[2], a = p[3];
                if (premultiply_alpha) {
                    r = premul(r, a);
                    g = premul(g, a);
                    b = premul(b, a);
                }
                out[0] = static_cast<float>(r);
                out[1] = static_cast<float>(g);
                out[2] = static_cast<float>(b);
                out[3] = static_cast<float>(a);
            };

            float p00[4], p10[4], p01[4], p11[4];
            get_pixel(x0, y0, p00);
            get_pixel(x1, y0, p10);
            get_pixel(x0, y1, p01);
            get_pixel(x1, y1, p11);

            const float omtx = 1.0f - tx;
            const float omty = 1.0f - ty;
            uint8_t interp_rgba[4];

            for (int c = 0; c < 4; ++c) {
                float top = p00[c] * omtx + p10[c] * tx;
                float bot = p01[c] * omtx + p11[c] * tx;
                float val = top * omty + bot * ty;
                float rounded = val + 0.5f;
                if (rounded < 0.0f) rounded = 0.0f;
                if (rounded > 255.0f) rounded = 255.0f;
                interp_rgba[c] = static_cast<uint8_t>(rounded);
            }

            uint8_t out_r = interp_rgba[0];
            uint8_t out_g = interp_rgba[1];
            uint8_t out_b = interp_rgba[2];
            uint8_t out_a = interp_rgba[3];

            if (premultiply_alpha) {
                out_r = unpremul(out_r, out_a);
                out_g = unpremul(out_g, out_a);
                out_b = unpremul(out_b, out_a);
            }

            uint8_t* dp = dst + (dst_row_offset + x) * 4;
            dp[0] = out_r;
            dp[1] = out_g;
            dp[2] = out_b;
            dp[3] = out_a;
        }
    }
}

} // namespace

// ── Math & Pixel Helper Tests ───────────────────────────────────────────────

TEST_CASE("ImageBuilder - Bilinear Interpolation Helper") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("test_bilinear_mod");
    ImageBuilder ib(mod);

    Function* fn = mod.create_function("test_bilinear", Type::f32(), {
        Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()
    });
    ib.builder().set_function(fn);
    BasicBlock* entry = ib.builder().append_block("entry");
    ib.builder().position_at_end(entry);

    Value* c00 = ib.builder().add_block_param(entry, Type::f32());
    Value* c10 = ib.builder().add_block_param(entry, Type::f32());
    Value* c01 = ib.builder().add_block_param(entry, Type::f32());
    Value* c11 = ib.builder().add_block_param(entry, Type::f32());
    Value* tx = ib.builder().add_block_param(entry, Type::f32());
    Value* ty = ib.builder().add_block_param(entry, Type::f32());

    Value* res = ib.bilinear_interp_f32(c00, c10, c01, c11, tx, ty);
    ib.builder().build_ret(res);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto interp_fn = kfn.as<float(*)(float, float, float, float, float, float)>();

    // Test corners
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 0.0f, 0.0f) - 10.0f) <= 1e-5f);
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 1.0f, 0.0f) - 20.0f) <= 1e-5f);
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 0.0f, 1.0f) - 30.0f) <= 1e-5f);
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 1.0f, 1.0f) - 40.0f) <= 1e-5f);

    // Test center
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 0.5f, 0.5f) - 25.0f) <= 1e-5f);

    // Test intermediate point
    float expected = (10.0f * 0.75f + 20.0f * 0.25f) * 0.5f + (30.0f * 0.75f + 40.0f * 0.25f) * 0.5f;
    CHECK(std::fabs(interp_fn(10.0f, 20.0f, 30.0f, 40.0f, 0.25f, 0.5f) - expected) <= 1e-5f);
}

TEST_CASE("ImageBuilder - Alpha Premultiply & Unpremultiply") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("test_alpha_mod");
    ImageBuilder ib(mod);

    // Create function that premultiplies then unpremultiplies
    Function* fn = mod.create_function("test_alpha_roundtrip", Type::i32(), {
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()
    });
    ib.builder().set_function(fn);
    BasicBlock* entry = ib.builder().append_block("entry");
    ib.builder().position_at_end(entry);

    Value* r = ib.builder().add_block_param(entry, Type::i32());
    Value* g = ib.builder().add_block_param(entry, Type::i32());
    Value* b = ib.builder().add_block_param(entry, Type::i32());
    Value* a = ib.builder().add_block_param(entry, Type::i32());
    Value* op_mode = ib.builder().add_block_param(entry, Type::i32()); // 0 = premul, 1 = unpremul, 2 = roundtrip

    Value *pr, *pg, *pb;
    ib.premultiply_alpha_u8(r, g, b, a, pr, pg, pb);

    Value *ur, *ug, *ub;
    ib.unpremultiply_alpha_u8(r, g, b, a, ur, ug, ub);

    Value *rr, *rg, *rb;
    ib.unpremultiply_alpha_u8(pr, pg, pb, a, rr, rg, rb);

    Value* res_pre = ib.builder().build_or(pr, ib.builder().build_shl(pg, ib.const_i32(8)));
    res_pre = ib.builder().build_or(res_pre, ib.builder().build_shl(pb, ib.const_i32(16)));

    Value* res_unpre = ib.builder().build_or(ur, ib.builder().build_shl(ug, ib.const_i32(8)));
    res_unpre = ib.builder().build_or(res_unpre, ib.builder().build_shl(ub, ib.const_i32(16)));

    Value* res_rt = ib.builder().build_or(rr, ib.builder().build_shl(rg, ib.const_i32(8)));
    res_rt = ib.builder().build_or(res_rt, ib.builder().build_shl(rb, ib.const_i32(16)));

    Value* is_mode0 = ib.builder().build_eq(op_mode, ib.const_i32(0));
    Value* is_mode1 = ib.builder().build_eq(op_mode, ib.const_i32(1));
    Value* out = ib.builder().build_select(is_mode0, res_pre,
                 ib.builder().build_select(is_mode1, res_unpre, res_rt));
    ib.builder().build_ret(out);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto alpha_fn = kfn.as<int32_t(*)(int32_t, int32_t, int32_t, int32_t, int32_t)>();

    // 1. Opaque alpha (a = 255): premultiply leaves RGB unchanged
    int32_t packed_pre = alpha_fn(200, 150, 100, 255, 0);
    CHECK_EQ(packed_pre & 0xFF, 200);
    CHECK_EQ((packed_pre >> 8) & 0xFF, 150);
    CHECK_EQ((packed_pre >> 16) & 0xFF, 100);

    // 2. Fully transparent (a = 0): premultiply yields 0, unpremultiply yields 0 (no divide-by-zero)
    packed_pre = alpha_fn(200, 150, 100, 0, 0);
    CHECK_EQ(packed_pre, 0);
    int32_t packed_unpre = alpha_fn(200, 150, 100, 0, 1);
    CHECK_EQ(packed_unpre, 0);

    // 3. Semi-transparent (a = 128): verify premultiply values (c * 128 + 127) / 255
    packed_pre = alpha_fn(200, 100, 50, 128, 0);
    CHECK_EQ(packed_pre & 0xFF, (200 * 128 + 127) / 255);
    CHECK_EQ((packed_pre >> 8) & 0xFF, (100 * 128 + 127) / 255);
    CHECK_EQ((packed_pre >> 16) & 0xFF, (50 * 128 + 127) / 255);

    // 4. Roundtrip: opaque channels roundtrip perfectly
    int32_t packed_rt = alpha_fn(210, 140, 70, 255, 2);
    CHECK_EQ(packed_rt & 0xFF, 210);
    CHECK_EQ((packed_rt >> 8) & 0xFF, 140);
    CHECK_EQ((packed_rt >> 16) & 0xFF, 70);
}

TEST_CASE("ImageBuilder - Uint8 <-> Float32 Conversion Helpers") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("test_conv_mod");
    ImageBuilder ib(mod);

    Function* fn = mod.create_function("test_conv", Type::i32(), {Type::i32()});
    ib.builder().set_function(fn);
    BasicBlock* entry = ib.builder().append_block("entry");
    ib.builder().position_at_end(entry);

    Value* val = ib.builder().add_block_param(entry, Type::i32());
    Value* f_val = ib.u8_to_f32(val);
    Value* u8_back = ib.f32_to_u8(f_val);
    ib.builder().build_ret(u8_back);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto conv_fn = kfn.as<int32_t(*)(int32_t)>();

    for (int i = 0; i <= 255; ++i) {
        CHECK_EQ(conv_fn(i), i);
    }
}

// ── Fused Preprocessing Kernel Tests ────────────────────────────────────────

TEST_CASE("ImageBuilder - JIT Fused Preprocessing Kernel (Bilinear Downscaling)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 64;
    const int32_t src_h = 48;
    const int32_t dst_w = 32;
    const int32_t dst_h = 24;
    const int channels = 3;

    // Create synthetic patterned RGB image [H, W, C]
    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * channels));
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int idx = (y * src_w + x) * channels;
            src[static_cast<size_t>(idx + 0)] = static_cast<uint8_t>((x * 255) / (src_w - 1));
            src[static_cast<size_t>(idx + 1)] = static_cast<uint8_t>((y * 255) / (src_h - 1));
            src[static_cast<size_t>(idx + 2)] = static_cast<uint8_t>(((x + y) * 255) / (src_w + src_h - 2));
        }
    }

    // Standard ImageNet normalization coefficients
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};
    const float inv_std[3] = {1.0f / std_dev[0], 1.0f / std_dev[1], 1.0f / std_dev[2]};

    // Allocate planar destination [C, H, W]
    const size_t planar_size = static_cast<size_t>(channels * dst_w * dst_h);
    std::vector<float> dst_jit(planar_size, 0.0f);
    std::vector<float> dst_ref(planar_size, 0.0f);

    // Compute scalar reference
    ref_fused_preproc(src.data(), dst_ref.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, mean, inv_std, channels);

    // Build and JIT compile kernel
    Module mod("fused_preproc_downscale_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_fused_preproc_function("image_preproc_fused", channels);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto preproc_fn = kfn.as<ImagePreprocFusedFn>();

    // Run JIT kernel
    preproc_fn(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, mean, inv_std);

    // Verify planar float outputs against C++ scalar reference math ((val / 255.0f - mean) / std)
    int mismatches = 0;
    for (size_t i = 0; i < planar_size; ++i) {
        float diff = std::fabs(dst_jit[i] - dst_ref[i]);
        if (diff > 1e-4f) {
            mismatches++;
            if (mismatches <= 5) {
                std::cout << "  Mismatch at " << i << ": JIT=" << dst_jit[i]
                          << " Ref=" << dst_ref[i] << " Diff=" << diff << "\n";
            }
        }
        CHECK(diff <= 1e-4f);
    }
    CHECK_EQ(mismatches, 0);
}

TEST_CASE("ImageBuilder - JIT Fused Preprocessing Kernel (Bilinear Upscaling)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 16;
    const int32_t src_h = 12;
    const int32_t dst_w = 32;
    const int32_t dst_h = 24;
    const int channels = 3;

    // Create synthetic patterned RGB image
    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * channels));
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int idx = (y * src_w + x) * channels;
            src[static_cast<size_t>(idx + 0)] = static_cast<uint8_t>((x * 17) % 256);
            src[static_cast<size_t>(idx + 1)] = static_cast<uint8_t>((y * 23) % 256);
            src[static_cast<size_t>(idx + 2)] = static_cast<uint8_t>((x * y * 7) % 256);
        }
    }

    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float inv_std[3] = {2.0f, 2.0f, 2.0f};

    const size_t planar_size = static_cast<size_t>(channels * dst_w * dst_h);
    std::vector<float> dst_jit(planar_size, 0.0f);
    std::vector<float> dst_ref(planar_size, 0.0f);

    ref_fused_preproc(src.data(), dst_ref.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, mean, inv_std, channels);

    Module mod("fused_preproc_upscale_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_fused_preproc_function("image_preproc_fused_up", channels);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto preproc_fn = kfn.as<ImagePreprocFusedFn>();

    preproc_fn(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, mean, inv_std);

    for (size_t i = 0; i < planar_size; ++i) {
        CHECK(std::fabs(dst_jit[i] - dst_ref[i]) <= 1e-4f);
    }
}

TEST_CASE("ImageBuilder - Fused Preprocessing Scanline Slicing") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 40;
    const int32_t src_h = 30;
    const int32_t dst_w = 20;
    const int32_t dst_h = 16;
    const int channels = 3;

    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * channels));
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 13) % 256);

    const float mean[3] = {0.4f, 0.5f, 0.6f};
    const float inv_std[3] = {1.5f, 1.2f, 1.0f};

    const size_t planar_size = static_cast<size_t>(channels * dst_w * dst_h);
    std::vector<float> dst_full(planar_size, 0.0f);
    std::vector<float> dst_sliced(planar_size, 0.0f);

    Module mod("fused_preproc_sliced_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_fused_preproc_function("image_preproc_sliced", channels);

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto preproc_fn = kfn.as<ImagePreprocFusedFn>();

    // 1. Run full range in one pass [0, dst_h)
    preproc_fn(src.data(), dst_full.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, mean, inv_std);

    // 2. Run in 2 slices: [0, 8) and [8, 16)
    preproc_fn(src.data(), dst_sliced.data(), src_w, src_h, dst_w, dst_h, 0, 8, mean, inv_std);
    preproc_fn(src.data(), dst_sliced.data(), src_w, src_h, dst_w, dst_h, 8, dst_h, mean, inv_std);

    for (size_t i = 0; i < planar_size; ++i) {
        CHECK(std::fabs(dst_sliced[i] - dst_full[i]) <= 1e-6f);
    }
}

// ── RGBA8 Resize Kernel Tests ───────────────────────────────────────────────

TEST_CASE("ImageBuilder - JIT RGBA8 Resize Kernel (Downscaling Straight Alpha)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 48;
    const int32_t src_h = 36;
    const int32_t dst_w = 24;
    const int32_t dst_h = 18;

    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * 4));
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int idx = (y * src_w + x) * 4;
            src[static_cast<size_t>(idx + 0)] = static_cast<uint8_t>((x * 255) / (src_w - 1));
            src[static_cast<size_t>(idx + 1)] = static_cast<uint8_t>((y * 255) / (src_h - 1));
            src[static_cast<size_t>(idx + 2)] = static_cast<uint8_t>(255 - src[static_cast<size_t>(idx + 0)]);
            src[static_cast<size_t>(idx + 3)] = static_cast<uint8_t>(255);
        }
    }

    std::vector<uint8_t> dst_jit(static_cast<size_t>(dst_w * dst_h * 4), 0);
    std::vector<uint8_t> dst_ref(static_cast<size_t>(dst_w * dst_h * 4), 0);

    ref_resize_rgba8(src.data(), dst_ref.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, false);

    Module mod("resize_rgba8_down_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_resize_rgba8_function("image_resize_rgba8");

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto resize_fn = kfn.as<ImageResizeRgba8Fn>();

    resize_fn(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, false);

    int max_diff = 0;
    for (size_t i = 0; i < dst_jit.size(); ++i) {
        int diff = std::abs(static_cast<int>(dst_jit[i]) - static_cast<int>(dst_ref[i]));
        if (diff > max_diff) max_diff = diff;
        CHECK(diff <= 1); // Allow at most 1 unit rounding difference
    }
    CHECK(max_diff <= 1);
}

TEST_CASE("ImageBuilder - JIT RGBA8 Resize Kernel (Downscaling with Premultiplied Alpha)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 32;
    const int32_t src_h = 32;
    const int32_t dst_w = 16;
    const int32_t dst_h = 16;

    // Create an image with varying alpha (transparent borders and semi-transparent center)
    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * 4));
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int idx = (y * src_w + x) * 4;
            src[static_cast<size_t>(idx + 0)] = 240;
            src[static_cast<size_t>(idx + 1)] = 60;
            src[static_cast<size_t>(idx + 2)] = 120;
            // Circular alpha mask
            float dx = (static_cast<float>(x) - 15.5f) / 16.0f;
            float dy = (static_cast<float>(y) - 15.5f) / 16.0f;
            float dist = std::sqrt(dx * dx + dy * dy);
            float a = std::max(0.0f, std::min(1.0f, (1.0f - dist) * 2.0f));
            src[static_cast<size_t>(idx + 3)] = static_cast<uint8_t>(a * 255.0f);
        }
    }

    std::vector<uint8_t> dst_jit(static_cast<size_t>(dst_w * dst_h * 4), 0);
    std::vector<uint8_t> dst_ref(static_cast<size_t>(dst_w * dst_h * 4), 0);

    ref_resize_rgba8(src.data(), dst_ref.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, true);

    Module mod("resize_rgba8_premul_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_resize_rgba8_function("image_resize_rgba8_premul");

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto resize_fn = kfn.as<ImageResizeRgba8Fn>();

    resize_fn(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, true);

    int max_diff = 0;
    for (size_t i = 0; i < dst_jit.size(); ++i) {
        int diff = std::abs(static_cast<int>(dst_jit[i]) - static_cast<int>(dst_ref[i]));
        if (diff > max_diff) max_diff = diff;
        CHECK(diff <= 2); // Integer division / rounding difference tolerance
    }
    CHECK(max_diff <= 2);
}

TEST_CASE("ImageBuilder - JIT RGBA8 Resize Kernel (Upscaling)") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 12;
    const int32_t src_h = 10;
    const int32_t dst_w = 24;
    const int32_t dst_h = 20;

    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * 4));
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<uint8_t>((i * 37) % 256);
    }

    std::vector<uint8_t> dst_jit(static_cast<size_t>(dst_w * dst_h * 4), 0);
    std::vector<uint8_t> dst_ref(static_cast<size_t>(dst_w * dst_h * 4), 0);

    ref_resize_rgba8(src.data(), dst_ref.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, false);

    Module mod("resize_rgba8_up_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_resize_rgba8_function("image_resize_rgba8_up");

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto resize_fn = kfn.as<ImageResizeRgba8Fn>();

    resize_fn(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, false);

    int max_diff = 0;
    for (size_t i = 0; i < dst_jit.size(); ++i) {
        int diff = std::abs(static_cast<int>(dst_jit[i]) - static_cast<int>(dst_ref[i]));
        if (diff > max_diff) max_diff = diff;
        CHECK(diff <= 2);
    }
    CHECK(max_diff <= 2);
}

TEST_CASE("ImageBuilder - RGBA8 Resize Scanline Slicing") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    const int32_t src_w = 32;
    const int32_t src_h = 24;
    const int32_t dst_w = 16;
    const int32_t dst_h = 12;

    std::vector<uint8_t> src(static_cast<size_t>(src_w * src_h * 4));
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>((i * 19) % 256);

    std::vector<uint8_t> dst_full(static_cast<size_t>(dst_w * dst_h * 4), 0);
    std::vector<uint8_t> dst_sliced(static_cast<size_t>(dst_w * dst_h * 4), 0);

    Module mod("resize_rgba8_slice_mod");
    ImageBuilder ib(mod);
    Function* fn = ib.build_resize_rgba8_function("image_resize_rgba8_slice");

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    auto resize_fn = kfn.as<ImageResizeRgba8Fn>();

    // 1. Full run
    resize_fn(src.data(), dst_full.data(), src_w, src_h, dst_w, dst_h, 0, dst_h, false);

    // 2. Sliced run: [0, 6) then [6, 12)
    resize_fn(src.data(), dst_sliced.data(), src_w, src_h, dst_w, dst_h, 0, 6, false);
    resize_fn(src.data(), dst_sliced.data(), src_w, src_h, dst_w, dst_h, 6, dst_h, false);

    for (size_t i = 0; i < dst_full.size(); ++i) {
        CHECK_EQ(dst_sliced[i], dst_full[i]);
    }
}
