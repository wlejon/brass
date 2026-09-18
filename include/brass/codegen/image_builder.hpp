#pragma once

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>
#include <string_view>

namespace brass::codegen {

// Typedef for JIT-compiled fused image preprocessing kernel:
// Bilinear resize + RGB uint8 to float32 + normalization + HWC-to-CHW planar transposition.
typedef void (*ImagePreprocFusedFn)(
    const uint8_t* src,
    float* dst_planar,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    int32_t y_start,
    int32_t y_end,
    const float* mean,
    const float* inv_std
);

// Typedef for JIT-compiled RGBA8 resize kernel:
// Bilinear resize + optional alpha premultiply/unpremultiply in a single fused pass.
typedef void (*ImageResizeRgba8Fn)(
    const uint8_t* src,
    uint8_t* dst,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    int32_t y_start,
    int32_t y_end,
    bool premultiply_alpha
);

// ─── ImageBuilder ────────────────────────────────────────────────────────────
//
// Specialized MIR builder for 2D image transformation, streaming preprocessing,
// resizing, alpha handling, and planar transposition kernels.
class ImageBuilder : public KernelBuilder {
public:
    ImageBuilder(Module& mod, Function* fn = nullptr)
        : KernelBuilder(mod, fn) {}
    explicit ImageBuilder(Builder& b) noexcept
        : KernelBuilder(b) {}

    // ── Math & Pixel Helpers ────────────────────────────────────────────────
    // Bilinear interpolation between four scalar float values:
    // c00 = (x0, y0), c10 = (x1, y0), c01 = (x0, y1), c11 = (x1, y1)
    // tx = fractional x [0, 1], ty = fractional y [0, 1]
    Value* bilinear_interp_f32(Value* c00, Value* c10, Value* c01, Value* c11, Value* tx, Value* ty);

    // Premultiplies alpha in uint8 [0, 255]:
    // out_c = (c * a + 127) / 255
    void premultiply_alpha_u8(Value* r, Value* g, Value* b, Value* a,
                              Value*& out_r, Value*& out_g, Value*& out_b);

    // Unpremultiplies alpha in uint8 [0, 255]:
    // out_c = a == 0 ? 0 : min(255, (c * 255 + a / 2) / a)
    void unpremultiply_alpha_u8(Value* r, Value* g, Value* b, Value* a,
                                Value*& out_r, Value*& out_g, Value*& out_b);

    // Converts uint8 [0, 255] to float32:
    // f = static_cast<float>(u8_val)
    // if scale and bias are provided: return f * scale + bias
    Value* u8_to_f32(Value* u8_val, Value* scale = nullptr, Value* bias = nullptr);

    // Converts float32 to uint8 [0, 255]:
    // clamps to [0, 255] and rounds to nearest integer
    Value* f32_to_u8(Value* f32_val);

    // Range clamp helpers
    Value* clamp_f32(Value* val, Value* min_val, Value* max_val);
    Value* clamp_i32(Value* val, Value* min_val, Value* max_val);
    Value* floor_f32(Value* x);

    // Byte load helper with dynamic byte offset
    Value* load_u8(Value* ptr, Value* byte_offset);
    using KernelBuilder::load_u8;

    // ── Loop and Kernel Emitters ────────────────────────────────────────────
    // Emits a 2D scanline loop iterating y in [y_start, y_end) and x in [0, dst_w):
    // - Computes sampling coordinate in source (src_x, src_y)
    // - Samples 4 surrounding pixels (bilinear)
    // - Interpolates channels
    // - Converts uint8 [0, 255] -> float32
    // - Normalizes: (channel - mean[c]) * inv_std[c]
    // - Stores directly to planar destination dst_planar[c * (dst_w * dst_h) + y * dst_w + x]
    void emit_fused_preproc_kernel(
        Function* fn,
        Value* src,
        Value* dst_planar,
        Value* src_w,
        Value* src_h,
        Value* dst_w,
        Value* dst_h,
        Value* y_start,
        Value* y_end,
        Value* mean,
        Value* inv_std,
        int channels = 3
    );

    // Fuses bilinear resize + optional premultiply / unpremultiply in a single pass
    // without intermediate memory allocations, storing to dst[(y * dst_w + x) * 4].
    void emit_resize_rgba8_kernel(
        Function* fn,
        Value* src,
        Value* dst,
        Value* src_w,
        Value* src_h,
        Value* dst_w,
        Value* dst_h,
        Value* y_start,
        Value* y_end,
        Value* premultiply_alpha
    );

    // ── High-Level Function Builders ────────────────────────────────────────
    Function* build_fused_preproc_function(
        std::string_view name = "image_preproc_fused",
        int channels = 3
    );

    Function* build_resize_rgba8_function(
        std::string_view name = "image_resize_rgba8"
    );
};

} // namespace brass::codegen
