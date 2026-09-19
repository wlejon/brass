#include <brass/codegen/image_builder.hpp>

#include <vector>

namespace brass::codegen {

// ── Math & Pixel Helpers ────────────────────────────────────────────────────

Value* ImageBuilder::bilinear_interp_f32(
    Value* c00, Value* c10, Value* c01, Value* c11, Value* tx, Value* ty
) {
    Value* one_f = const_f32(1.0f);
    Value* omtx = sub(one_f, tx);
    Value* omty = sub(one_f, ty);

    // top = c00 * (1 - tx) + c10 * tx
    Value* top = add(mul(c00, omtx), mul(c10, tx));
    // bot = c01 * (1 - tx) + c11 * tx
    Value* bot = add(mul(c01, omtx), mul(c11, tx));
    // res = top * (1 - ty) + bot * ty
    Value* res = add(mul(top, omty), mul(bot, ty));
    return res;
}

void ImageBuilder::premultiply_alpha_u8(
    Value* r, Value* g, Value* b, Value* a,
    Value*& out_r, Value*& out_g, Value*& out_b
) {
    Value* c127 = const_i32(127);
    Value* c255 = const_i32(255);

    auto premul = [&](Value* ch) -> Value* {
        Value* prod = builder().build_mul(ch, a);
        Value* sum = builder().build_add(prod, c127);
        return builder().build_sdiv(sum, c255);
    };

    out_r = premul(r);
    out_g = premul(g);
    out_b = premul(b);
}

void ImageBuilder::unpremultiply_alpha_u8(
    Value* r, Value* g, Value* b, Value* a,
    Value*& out_r, Value*& out_g, Value*& out_b
) {
    Value* zero_i = const_i32(0);
    Value* c255 = const_i32(255);
    Value* c2 = const_i32(2);

    Value* is_zero = builder().build_eq(a, zero_i);
    Value* safe_a = builder().build_select(is_zero, const_i32(1), a);
    Value* half_a = builder().build_sdiv(safe_a, c2);

    auto unpremul = [&](Value* ch) -> Value* {
        Value* prod = builder().build_mul(ch, c255);
        Value* sum = builder().build_add(prod, half_a);
        Value* div_val = builder().build_sdiv(sum, safe_a);
        Value* is_overflow = builder().build_sgt(div_val, c255);
        Value* clamped = builder().build_select(is_overflow, c255, div_val);
        return builder().build_select(is_zero, zero_i, clamped);
    };

    out_r = unpremul(r);
    out_g = unpremul(g);
    out_b = unpremul(b);
}

Value* ImageBuilder::u8_to_f32(Value* u8_val, Value* scale, Value* bias) {
    Value* clean_u8 = u8_val;
    if (u8_val->type().is_i64()) {
        clean_u8 = builder().build_trunc_i32(u8_val);
    }
    Value* f_val = builder().build_sitofp_f32_i32(clean_u8);
    if (scale && bias) {
        return fma(f_val, scale, bias);
    } else if (scale) {
        return mul(f_val, scale);
    } else if (bias) {
        return add(f_val, bias);
    }
    return f_val;
}

Value* ImageBuilder::clamp_f32(Value* val, Value* min_val, Value* max_val) {
    return builder().build_fmin_f32(builder().build_fmax_f32(val, min_val), max_val);
}

Value* ImageBuilder::clamp_i32(Value* val, Value* min_val, Value* max_val) {
    Value* is_less = builder().build_slt(val, min_val);
    Value* t = builder().build_select(is_less, min_val, val);
    Value* is_greater = builder().build_sgt(t, max_val);
    return builder().build_select(is_greater, max_val, t);
}

Value* ImageBuilder::floor_f32(Value* x) {
    return builder().build_floor_f32(x);
}

Value* ImageBuilder::f32_to_u8(Value* f32_val) {
    Value* half = const_f32(0.5f);
    Value* zero_f = const_f32(0.0f);
    Value* max_f = const_f32(255.0f);
    Value* rounded = add(f32_val, half);
    Value* clamped = clamp_f32(rounded, zero_f, max_f);
    return builder().build_fptosi_i32_f32(clamped);
}

Value* ImageBuilder::load_u8(Value* ptr, Value* byte_offset) {
    Value* off_i64 = byte_offset->type().is_i64() ? byte_offset : builder().build_zext_i64(byte_offset);
    Value* addr = builder().build_add(ptr, off_i64);
    if (builder().current_module() && !builder().current_module()->has_external_symbol("ptx_load_u8")) {
        builder().current_module()->add_external_symbol("ptx_load_u8");
    }
    return KernelBuilder::load_u8(addr, 0);
}

// ── Kernel Emitters ─────────────────────────────────────────────────────────

void ImageBuilder::emit_fused_preproc_kernel(
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
    int channels
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    if (builder().current_module()) {
        if (!builder().current_module()->has_external_symbol("ptx_load_u8")) {
            builder().current_module()->add_external_symbol("ptx_load_u8");
        }
    }

    Value* zero_i = const_i32(0);
    Value* one_i = const_i32(1);
    Value* zero_f = const_f32(0.0f);
    Value* one_f = const_f32(1.0f);
    Value* half_f = const_f32(0.5f);
    Value* inv_255 = const_f32(1.0f / 255.0f);

    Value* src_w_f = u8_to_f32(src_w);
    Value* src_h_f = u8_to_f32(src_h);
    Value* dst_w_f = u8_to_f32(dst_w);
    Value* dst_h_f = u8_to_f32(dst_h);

    Value* xs = div(src_w_f, dst_w_f);
    Value* ys = div(src_h_f, dst_h_f);

    Value* sw_minus_1 = builder().build_sub(src_w, one_i);
    Value* sh_minus_1 = builder().build_sub(src_h, one_i);
    Value* plane_size = builder().build_mul(dst_w, dst_h);
    Value* channels_val = const_i32(channels);

    // Preload mean and inv_std values for each channel
    std::vector<Value*> mean_vals(channels);
    std::vector<Value*> inv_std_vals(channels);
    for (int c = 0; c < channels; ++c) {
        mean_vals[c] = load_f32_indexed(mean, const_i64(c), 4, 0);
        inv_std_vals[c] = load_f32_indexed(inv_std, const_i64(c), 4, 0);
    }

    for_range(y_start, y_end, one_i, [&](Value* y) {
        Value* y_f = u8_to_f32(y);
        Value* fy = sub(mul(add(y_f, half_f), ys), half_f);
        Value* floor_y = floor_f32(fy);
        Value* iy0_raw = builder().build_fptosi_i32_f32(floor_y);
        Value* iy1_raw = builder().build_add(iy0_raw, one_i);
        Value* ty_raw = sub(fy, floor_y);

        Value* y0 = clamp_i32(iy0_raw, zero_i, sh_minus_1);
        Value* y1 = clamp_i32(iy1_raw, zero_i, sh_minus_1);
        Value* ty = clamp_f32(ty_raw, zero_f, one_f);

        Value* y0_stride = builder().build_mul(y0, src_w);
        Value* y1_stride = builder().build_mul(y1, src_w);
        Value* dst_row_offset = builder().build_mul(y, dst_w);

        for_range(zero_i, dst_w, one_i, [&](Value* x) {
            Value* x_f = u8_to_f32(x);
            Value* fx = sub(mul(add(x_f, half_f), xs), half_f);
            Value* floor_x = floor_f32(fx);
            Value* ix0_raw = builder().build_fptosi_i32_f32(floor_x);
            Value* ix1_raw = builder().build_add(ix0_raw, one_i);
            Value* tx_raw = sub(fx, floor_x);

            Value* x0 = clamp_i32(ix0_raw, zero_i, sw_minus_1);
            Value* x1 = clamp_i32(ix1_raw, zero_i, sw_minus_1);
            Value* tx = clamp_f32(tx_raw, zero_f, one_f);

            // Pixel indices in source
            Value* p00_pix = builder().build_add(y0_stride, x0);
            Value* p10_pix = builder().build_add(y0_stride, x1);
            Value* p01_pix = builder().build_add(y1_stride, x0);
            Value* p11_pix = builder().build_add(y1_stride, x1);

            Value* p00_byte_base = builder().build_mul(p00_pix, channels_val);
            Value* p10_byte_base = builder().build_mul(p10_pix, channels_val);
            Value* p01_byte_base = builder().build_mul(p01_pix, channels_val);
            Value* p11_byte_base = builder().build_mul(p11_pix, channels_val);

            Value* dst_pixel_idx = builder().build_add(dst_row_offset, x);

            for (int c = 0; c < channels; ++c) {
                Value* c_offset = const_i32(c);
                Value* off00 = builder().build_add(p00_byte_base, c_offset);
                Value* off10 = builder().build_add(p10_byte_base, c_offset);
                Value* off01 = builder().build_add(p01_byte_base, c_offset);
                Value* off11 = builder().build_add(p11_byte_base, c_offset);

                Value* u00 = load_u8(src, off00);
                Value* u10 = load_u8(src, off10);
                Value* u01 = load_u8(src, off01);
                Value* u11 = load_u8(src, off11);

                Value* f00 = u8_to_f32(u00);
                Value* f10 = u8_to_f32(u10);
                Value* f01 = u8_to_f32(u01);
                Value* f11 = u8_to_f32(u11);

                Value* interp = bilinear_interp_f32(f00, f10, f01, f11, tx, ty);
                Value* val_01 = mul(interp, inv_255);

                Value* diff = sub(val_01, mean_vals[c]);
                Value* norm = mul(diff, inv_std_vals[c]);

                Value* plane_offset = builder().build_mul(const_i32(c), plane_size);
                Value* dst_linear = builder().build_add(plane_offset, dst_pixel_idx);
                store_f32_indexed(dst_planar, builder().build_zext_i64(dst_linear), norm, 4, 0);
            }
        });
    });
}

void ImageBuilder::emit_resize_rgba8_kernel(
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
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    Value* zero_i = const_i32(0);
    Value* one_i = const_i32(1);
    Value* zero_f = const_f32(0.0f);
    Value* one_f = const_f32(1.0f);
    Value* half_f = const_f32(0.5f);
    Value* mask_ff = const_i32(0xFF);
    Value* shift_8 = const_i32(8);
    Value* shift_16 = const_i32(16);
    Value* shift_24 = const_i32(24);

    Value* src_w_f = u8_to_f32(src_w);
    Value* src_h_f = u8_to_f32(src_h);
    Value* dst_w_f = u8_to_f32(dst_w);
    Value* dst_h_f = u8_to_f32(dst_h);

    Value* xs = div(src_w_f, dst_w_f);
    Value* ys = div(src_h_f, dst_h_f);

    Value* sw_minus_1 = builder().build_sub(src_w, one_i);
    Value* sh_minus_1 = builder().build_sub(src_h, one_i);

    Value* is_premul = builder().build_ne(premultiply_alpha, zero_i);

    for_range(y_start, y_end, one_i, [&](Value* y) {
        Value* y_f = u8_to_f32(y);
        Value* fy = sub(mul(add(y_f, half_f), ys), half_f);
        Value* floor_y = floor_f32(fy);
        Value* iy0_raw = builder().build_fptosi_i32_f32(floor_y);
        Value* iy1_raw = builder().build_add(iy0_raw, one_i);
        Value* ty_raw = sub(fy, floor_y);

        Value* y0 = clamp_i32(iy0_raw, zero_i, sh_minus_1);
        Value* y1 = clamp_i32(iy1_raw, zero_i, sh_minus_1);
        Value* ty = clamp_f32(ty_raw, zero_f, one_f);

        Value* y0_stride = builder().build_mul(y0, src_w);
        Value* y1_stride = builder().build_mul(y1, src_w);
        Value* dst_row_offset = builder().build_mul(y, dst_w);

        for_range(zero_i, dst_w, one_i, [&](Value* x) {
            Value* x_f = u8_to_f32(x);
            Value* fx = sub(mul(add(x_f, half_f), xs), half_f);
            Value* floor_x = floor_f32(fx);
            Value* ix0_raw = builder().build_fptosi_i32_f32(floor_x);
            Value* ix1_raw = builder().build_add(ix0_raw, one_i);
            Value* tx_raw = sub(fx, floor_x);

            Value* x0 = clamp_i32(ix0_raw, zero_i, sw_minus_1);
            Value* x1 = clamp_i32(ix1_raw, zero_i, sw_minus_1);
            Value* tx = clamp_f32(tx_raw, zero_f, one_f);

            Value* p00_idx = builder().build_add(y0_stride, x0);
            Value* p10_idx = builder().build_add(y0_stride, x1);
            Value* p01_idx = builder().build_add(y1_stride, x0);
            Value* p11_idx = builder().build_add(y1_stride, x1);

            // Load 4 corner pixels as 32-bit words (each pixel is 4 bytes RGBA)
            Value* pix00 = load_i32_indexed(src, builder().build_zext_i64(p00_idx), 4, 0);
            Value* pix10 = load_i32_indexed(src, builder().build_zext_i64(p10_idx), 4, 0);
            Value* pix01 = load_i32_indexed(src, builder().build_zext_i64(p01_idx), 4, 0);
            Value* pix11 = load_i32_indexed(src, builder().build_zext_i64(p11_idx), 4, 0);

            auto unpack_and_premul = [&](Value* pix, Value*& r_f, Value*& g_f, Value*& b_f, Value*& a_f) {
                Value* r = builder().build_and(pix, mask_ff);
                Value* g = builder().build_and(builder().build_lshr(pix, shift_8), mask_ff);
                Value* b = builder().build_and(builder().build_lshr(pix, shift_16), mask_ff);
                Value* a = builder().build_and(builder().build_lshr(pix, shift_24), mask_ff);

                Value *pr, *pg, *pb;
                premultiply_alpha_u8(r, g, b, a, pr, pg, pb);

                Value* r_eff = builder().build_select(is_premul, pr, r);
                Value* g_eff = builder().build_select(is_premul, pg, g);
                Value* b_eff = builder().build_select(is_premul, pb, b);

                r_f = u8_to_f32(r_eff);
                g_f = u8_to_f32(g_eff);
                b_f = u8_to_f32(b_eff);
                a_f = u8_to_f32(a);
            };

            Value *r00, *g00, *b00, *a00;
            Value *r10, *g10, *b10, *a10;
            Value *r01, *g01, *b01, *a01;
            Value *r11, *g11, *b11, *a11;

            unpack_and_premul(pix00, r00, g00, b00, a00);
            unpack_and_premul(pix10, r10, g10, b10, a10);
            unpack_and_premul(pix01, r01, g01, b01, a01);
            unpack_and_premul(pix11, r11, g11, b11, a11);

            Value* interp_r_f = bilinear_interp_f32(r00, r10, r01, r11, tx, ty);
            Value* interp_g_f = bilinear_interp_f32(g00, g10, g01, g11, tx, ty);
            Value* interp_b_f = bilinear_interp_f32(b00, b10, b01, b11, tx, ty);
            Value* interp_a_f = bilinear_interp_f32(a00, a10, a01, a11, tx, ty);

            Value* interp_r = f32_to_u8(interp_r_f);
            Value* interp_g = f32_to_u8(interp_g_f);
            Value* interp_b = f32_to_u8(interp_b_f);
            Value* interp_a = f32_to_u8(interp_a_f);

            Value *unprem_r, *unprem_g, *unprem_b;
            unpremultiply_alpha_u8(interp_r, interp_g, interp_b, interp_a, unprem_r, unprem_g, unprem_b);

            Value* out_r = builder().build_select(is_premul, unprem_r, interp_r);
            Value* out_g = builder().build_select(is_premul, unprem_g, interp_g);
            Value* out_b = builder().build_select(is_premul, unprem_b, interp_b);
            Value* out_a = interp_a;

            // Pack into 32-bit RGBA integer
            Value* packed = builder().build_or(out_r, builder().build_shl(out_g, shift_8));
            packed = builder().build_or(packed, builder().build_shl(out_b, shift_16));
            packed = builder().build_or(packed, builder().build_shl(out_a, shift_24));

            Value* dst_pixel_idx = builder().build_add(dst_row_offset, x);
            store_i32_indexed(dst, builder().build_zext_i64(dst_pixel_idx), packed, 4, 0);
        });
    });
}

// ── Function Builders ───────────────────────────────────────────────────────

Function* ImageBuilder::build_fused_preproc_function(std::string_view name, int channels) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::void_type(), {
        Type::ptr(), // src: const uint8_t*
        Type::ptr(), // dst_planar: float*
        Type::i32(), // src_w: int32_t
        Type::i32(), // src_h: int32_t
        Type::i32(), // dst_w: int32_t
        Type::i32(), // dst_h: int32_t
        Type::i32(), // y_start: int32_t
        Type::i32(), // y_end: int32_t
        Type::ptr(), // mean: const float*
        Type::ptr()  // inv_std: const float*
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* src = builder().add_block_param(entry, Type::ptr());
    Value* dst_planar = builder().add_block_param(entry, Type::ptr());
    Value* src_w = builder().add_block_param(entry, Type::i32());
    Value* src_h = builder().add_block_param(entry, Type::i32());
    Value* dst_w = builder().add_block_param(entry, Type::i32());
    Value* dst_h = builder().add_block_param(entry, Type::i32());
    Value* y_start = builder().add_block_param(entry, Type::i32());
    Value* y_end = builder().add_block_param(entry, Type::i32());
    Value* mean = builder().add_block_param(entry, Type::ptr());
    Value* inv_std = builder().add_block_param(entry, Type::ptr());

    emit_fused_preproc_kernel(fn, src, dst_planar, src_w, src_h, dst_w, dst_h, y_start, y_end, mean, inv_std, channels);
    builder().build_ret_void();
    return fn;
}

Function* ImageBuilder::build_resize_rgba8_function(std::string_view name) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::void_type(), {
        Type::ptr(), // src: const uint8_t*
        Type::ptr(), // dst: uint8_t*
        Type::i32(), // src_w: int32_t
        Type::i32(), // src_h: int32_t
        Type::i32(), // dst_w: int32_t
        Type::i32(), // dst_h: int32_t
        Type::i32(), // y_start: int32_t
        Type::i32(), // y_end: int32_t
        Type::i32()  // premultiply_alpha: bool (as i32)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* src = builder().add_block_param(entry, Type::ptr());
    Value* dst = builder().add_block_param(entry, Type::ptr());
    Value* src_w = builder().add_block_param(entry, Type::i32());
    Value* src_h = builder().add_block_param(entry, Type::i32());
    Value* dst_w = builder().add_block_param(entry, Type::i32());
    Value* dst_h = builder().add_block_param(entry, Type::i32());
    Value* y_start = builder().add_block_param(entry, Type::i32());
    Value* y_end = builder().add_block_param(entry, Type::i32());
    Value* premultiply_alpha = builder().add_block_param(entry, Type::i32());

    emit_resize_rgba8_kernel(fn, src, dst, src_w, src_h, dst_w, dst_h, y_start, y_end, premultiply_alpha);
    builder().build_ret_void();
    return fn;
}

} // namespace brass::codegen
