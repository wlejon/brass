#include <brass/codegen/sdf_builder.hpp>

namespace brass::codegen {

// ── Scalar Math Helpers ─────────────────────────────────────────────────────

Value* SdfKernelBuilder::abs_f32(Value* v) {
    Value* zero = const_f32(0.0f);
    Value* is_neg = builder().build_slt(v, zero);
    Value* neg_v = builder().build_neg(v);
    return builder().build_select(is_neg, neg_v, v);
}

Value* SdfKernelBuilder::min_f32(Value* a, Value* b) {
    Value* is_less = builder().build_slt(a, b);
    return builder().build_select(is_less, a, b);
}

Value* SdfKernelBuilder::max_f32(Value* a, Value* b) {
    Value* is_greater = builder().build_sgt(a, b);
    return builder().build_select(is_greater, a, b);
}

Value* SdfKernelBuilder::clamp_f32(Value* x, Value* min_val, Value* max_val) {
    Value* t = max_f32(x, min_val);
    return min_f32(t, max_val);
}

Value* SdfKernelBuilder::sqrt_f32(Value* x) {
    if (builder().current_module() && !builder().current_module()->has_external_symbol("sqrtf")) {
        builder().current_module()->add_external_symbol("sqrtf");
    }
    return builder().build_call("sqrtf", Type::f32(), {x});
}

Value* SdfKernelBuilder::length2(Value* x, Value* y) {
    Value* x2 = mul(x, x);
    Value* sum = fma(y, y, x2);
    return sqrt_f32(sum);
}

Value* SdfKernelBuilder::length3(Value* x, Value* y, Value* z) {
    Value* x2 = mul(x, x);
    Value* xy2 = fma(y, y, x2);
    Value* sum = fma(z, z, xy2);
    return sqrt_f32(sum);
}

Value* SdfKernelBuilder::lerp_f32(Value* a, Value* b, Value* t) {
    Value* diff = sub(b, a);
    return add(a, mul(diff, t));
}

Value* SdfKernelBuilder::to_f32(Value* val) {
    if (val->type() == Type::f32()) return val;
    if (builder().current_module() && !builder().current_module()->has_external_symbol("i32_to_f32")) {
        builder().current_module()->add_external_symbol("i32_to_f32");
    }
    Value* i32_val = val;
    if (val->type().is_i64()) {
        i32_val = builder().build_trunc_i32(val);
    }
    return builder().build_call("i32_to_f32", Type::f32(), {i32_val});
}

Value* SdfKernelBuilder::floor_f32(Value* x) {
    if (builder().current_module() && !builder().current_module()->has_external_symbol("floorf")) {
        builder().current_module()->add_external_symbol("floorf");
    }
    return builder().build_call("floorf", Type::f32(), {x});
}

// ── Primitives ──────────────────────────────────────────────────────────────

Value* SdfKernelBuilder::sdf_sphere(Value* px, Value* py, Value* pz, Value* radius) {
    Value* len = length3(px, py, pz);
    return sub(len, radius);
}

Value* SdfKernelBuilder::sdf_box(Value* px, Value* py, Value* pz, Value* bx, Value* by, Value* bz) {
    Value* qx = sub(abs_f32(px), bx);
    Value* qy = sub(abs_f32(py), by);
    Value* qz = sub(abs_f32(pz), bz);
    Value* zero = const_f32(0.0f);
    Value* mx = max_f32(qx, zero);
    Value* my = max_f32(qy, zero);
    Value* mz = max_f32(qz, zero);
    Value* outside = length3(mx, my, mz);
    Value* max_q = max_f32(qx, max_f32(qy, qz));
    Value* inside = min_f32(max_q, zero);
    return add(outside, inside);
}

Value* SdfKernelBuilder::sdf_rounded_box(Value* px, Value* py, Value* pz,
                                        Value* bx, Value* by, Value* bz,
                                        Value* radius) {
    Value* qx = add(sub(abs_f32(px), bx), radius);
    Value* qy = add(sub(abs_f32(py), by), radius);
    Value* qz = add(sub(abs_f32(pz), bz), radius);
    Value* zero = const_f32(0.0f);
    Value* mx = max_f32(qx, zero);
    Value* my = max_f32(qy, zero);
    Value* mz = max_f32(qz, zero);
    Value* outside = length3(mx, my, mz);
    Value* max_q = max_f32(qx, max_f32(qy, qz));
    Value* inside = min_f32(max_q, zero);
    Value* d = add(outside, inside);
    return sub(d, radius);
}

Value* SdfKernelBuilder::sdf_cylinder(Value* px, Value* py, Value* pz, Value* radius, Value* half_height) {
    Value* d_x = sub(length2(px, pz), radius);
    Value* d_y = sub(abs_f32(py), half_height);
    Value* zero = const_f32(0.0f);
    Value* mx = max_f32(d_x, zero);
    Value* my = max_f32(d_y, zero);
    Value* outside = length2(mx, my);
    Value* max_d = max_f32(d_x, d_y);
    Value* inside = min_f32(max_d, zero);
    return add(outside, inside);
}

Value* SdfKernelBuilder::sdf_capsule(Value* px, Value* py, Value* pz,
                                    Value* ax, Value* ay, Value* az,
                                    Value* bx, Value* by, Value* bz,
                                    Value* radius) {
    Value* pax = sub(px, ax);
    Value* pay = sub(py, ay);
    Value* paz = sub(pz, az);
    Value* bax = sub(bx, ax);
    Value* bay = sub(by, ay);
    Value* baz = sub(bz, az);

    Value* dot_pa_ba = fma(paz, baz, fma(pay, bay, mul(pax, bax)));
    Value* dot_ba_ba = fma(baz, baz, fma(bay, bay, mul(bax, bax)));

    Value* h = clamp_f32(div(dot_pa_ba, dot_ba_ba), const_f32(0.0f), const_f32(1.0f));

    Value* vx = sub(pax, mul(bax, h));
    Value* vy = sub(pay, mul(bay, h));
    Value* vz = sub(paz, mul(baz, h));

    return sub(length3(vx, vy, vz), radius);
}

Value* SdfKernelBuilder::sdf_torus(Value* px, Value* py, Value* pz, Value* major_r, Value* minor_r) {
    Value* qx = sub(length2(px, pz), major_r);
    Value* qy = py;
    return sub(length2(qx, qy), minor_r);
}

Value* SdfKernelBuilder::sdf_plane(Value* px, Value* py, Value* pz, Value* nx, Value* ny, Value* nz, Value* d) {
    Value* dot = fma(pz, nz, fma(py, ny, mul(px, nx)));
    return add(dot, d);
}

// ── CSG Combinators ─────────────────────────────────────────────────────────

Value* SdfKernelBuilder::op_union(Value* d1, Value* d2) {
    return min_f32(d1, d2);
}

Value* SdfKernelBuilder::op_intersection(Value* d1, Value* d2) {
    return max_f32(d1, d2);
}

Value* SdfKernelBuilder::op_subtraction(Value* d1, Value* d2) {
    return max_f32(d1, builder().build_neg(d2));
}

Value* SdfKernelBuilder::op_smooth_union(Value* d1, Value* d2, Value* k) {
    Value* half = const_f32(0.5f);
    Value* one = const_f32(1.0f);
    Value* zero = const_f32(0.0f);
    Value* diff = sub(d2, d1);
    Value* term = div(mul(half, diff), k);
    Value* h = clamp_f32(add(half, term), zero, one);
    Value* one_minus_h = sub(one, h);
    Value* mixed = add(mul(d2, one_minus_h), mul(d1, h));
    Value* smooth = mul(k, mul(h, one_minus_h));
    return sub(mixed, smooth);
}

Value* SdfKernelBuilder::op_smooth_intersection(Value* d1, Value* d2, Value* k) {
    Value* half = const_f32(0.5f);
    Value* one = const_f32(1.0f);
    Value* zero = const_f32(0.0f);
    Value* diff = sub(d2, d1);
    Value* term = div(mul(half, diff), k);
    Value* h = clamp_f32(sub(half, term), zero, one);
    Value* one_minus_h = sub(one, h);
    Value* mixed = add(mul(d2, one_minus_h), mul(d1, h));
    Value* smooth = mul(k, mul(h, one_minus_h));
    return add(mixed, smooth);
}

Value* SdfKernelBuilder::op_smooth_subtraction(Value* d1, Value* d2, Value* k) {
    return op_smooth_intersection(d1, builder().build_neg(d2), k);
}

// ── Transforms ──────────────────────────────────────────────────────────────

void SdfKernelBuilder::translate(Value* px, Value* py, Value* pz,
                                Value* ox, Value* oy, Value* oz,
                                Value*& out_x, Value*& out_y, Value*& out_z) {
    out_x = sub(px, ox);
    out_y = sub(py, oy);
    out_z = sub(pz, oz);
}

void SdfKernelBuilder::rotate_y(Value* px, Value* py, Value* pz, Value* angle_rad,
                               Value*& out_x, Value*& out_y, Value*& out_z) {
    if (builder().current_module()) {
        if (!builder().current_module()->has_external_symbol("cosf")) {
            builder().current_module()->add_external_symbol("cosf");
        }
        if (!builder().current_module()->has_external_symbol("sinf")) {
            builder().current_module()->add_external_symbol("sinf");
        }
    }
    Value* c = builder().build_call("cosf", Type::f32(), {angle_rad});
    Value* s = builder().build_call("sinf", Type::f32(), {angle_rad});
    // Inverse rotation on query point: R(-angle_rad)
    out_x = sub(mul(px, c), mul(pz, s));
    out_y = py;
    out_z = add(mul(px, s), mul(pz, c));
}

void SdfKernelBuilder::scale_uniform(Value* px, Value* py, Value* pz, Value* s,
                                    Value*& out_x, Value*& out_y, Value*& out_z) {
    out_x = div(px, s);
    out_y = div(py, s);
    out_z = div(pz, s);
}

Value* SdfKernelBuilder::scale_uniform_dist(Value* dist, Value* s) {
    return mul(dist, s);
}

// ── Procedural Noise ────────────────────────────────────────────────────────

Value* SdfKernelBuilder::noise_3d(Value* px, Value* py, Value* pz) {
    Value* flr_x = floor_f32(px);
    Value* flr_y = floor_f32(py);
    Value* flr_z = floor_f32(pz);

    if (builder().current_module() && !builder().current_module()->has_external_symbol("f32_to_i32")) {
        builder().current_module()->add_external_symbol("f32_to_i32");
    }
    Value* i0 = builder().build_call("f32_to_i32", Type::i32(), {flr_x});
    Value* j0 = builder().build_call("f32_to_i32", Type::i32(), {flr_y});
    Value* k0 = builder().build_call("f32_to_i32", Type::i32(), {flr_z});

    Value* one_i = const_i32(1);
    Value* i1 = builder().build_add(i0, one_i);
    Value* j1 = builder().build_add(j0, one_i);
    Value* k1 = builder().build_add(k0, one_i);

    Value* fx = sub(px, flr_x);
    Value* fy = sub(py, flr_y);
    Value* fz = sub(pz, flr_z);

    // Cubic Hermite smoothstep: u = fx * fx * (3.0 - 2.0 * fx)
    Value* c_3 = const_f32(3.0f);
    Value* c_2 = const_f32(2.0f);
    Value* u = mul(mul(fx, fx), sub(c_3, mul(fx, c_2)));
    Value* v = mul(mul(fy, fy), sub(c_3, mul(fy, c_2)));
    Value* w = mul(mul(fz, fz), sub(c_3, mul(fz, c_2)));

    auto hash_corner = [&](Value* ix, Value* iy, Value* iz) -> Value* {
        Value* h = builder().build_xor(ix, builder().build_mul(iy, const_i32(374761393)));
        h = builder().build_xor(h, builder().build_mul(iz, const_i32(668265263)));
        Value* shift13 = builder().build_lshr(h, const_i32(13));
        h = builder().build_mul(builder().build_xor(h, shift13), const_i32(1274126177));
        Value* shift16 = builder().build_lshr(h, const_i32(16));
        h = builder().build_xor(h, shift16);

        // Mask to positive 15-bit integer [0, 32767]
        Value* masked = builder().build_and(h, const_i32(0x7FFF));
        if (builder().current_module() && !builder().current_module()->has_external_symbol("i32_to_f32")) {
            builder().current_module()->add_external_symbol("i32_to_f32");
        }
        Value* h_f = builder().build_call("i32_to_f32", Type::f32(), {masked});

        // Map [0, 32767] to [-1.0, 1.0]: h_f * (2.0 / 32767.0) - 1.0
        Value* scale = const_f32(2.0f / 32767.0f);
        Value* one_f = const_f32(1.0f);
        return sub(mul(h_f, scale), one_f);
    };

    Value* h000 = hash_corner(i0, j0, k0);
    Value* h100 = hash_corner(i1, j0, k0);
    Value* h010 = hash_corner(i0, j1, k0);
    Value* h110 = hash_corner(i1, j1, k0);
    Value* h001 = hash_corner(i0, j0, k1);
    Value* h101 = hash_corner(i1, j0, k1);
    Value* h011 = hash_corner(i0, j1, k1);
    Value* h111 = hash_corner(i1, j1, k1);

    Value* x00 = lerp_f32(h000, h100, u);
    Value* x10 = lerp_f32(h010, h110, u);
    Value* x01 = lerp_f32(h001, h101, u);
    Value* x11 = lerp_f32(h011, h111, u);

    Value* y0 = lerp_f32(x00, x10, v);
    Value* y1 = lerp_f32(x01, x11, v);

    return lerp_f32(y0, y1, w);
}

// ── Grid Loop Emitter ───────────────────────────────────────────────────────

void SdfKernelBuilder::emit_grid_eval_loop(
    Value* out_field,
    Value* dim_x, Value* dim_y, Value* dim_z,
    const std::function<Value*(Value* x, Value* y, Value* z)>& eval_fn
) {
    Type idx_type = dim_x->type();
    Value* zero = idx_type.is_i64() ? const_i64(0) : const_i32(0);
    Value* step = idx_type.is_i64() ? const_i64(1) : const_i32(1);

    for_range(zero, dim_z, step, [&](Value* z) {
        for_range(zero, dim_y, step, [&](Value* y) {
            for_range(zero, dim_x, step, [&](Value* x) {
                Value* dist = eval_fn(x, y, z);
                Value* z_dy = mul(z, dim_y);
                Value* z_dy_y = add(z_dy, y);
                Value* idx_x = mul(z_dy_y, dim_x);
                Value* linear_idx = add(idx_x, x);
                store_f32_indexed(out_field, linear_idx, dist, 4, 0);
            });
        });
    });
}

void SdfKernelBuilder::emit_grid_eval_loop(
    Value* out_field,
    Value* dim_x, Value* dim_y, Value* dim_z,
    Value* origin_x, Value* origin_y, Value* origin_z,
    Value* step_x, Value* step_y, Value* step_z,
    const std::function<Value*(Value* px, Value* py, Value* pz)>& eval_fn
) {
    Type idx_type = dim_x->type();
    Value* zero = idx_type.is_i64() ? const_i64(0) : const_i32(0);
    Value* step = idx_type.is_i64() ? const_i64(1) : const_i32(1);

    for_range(zero, dim_z, step, [&](Value* z) {
        for_range(zero, dim_y, step, [&](Value* y) {
            for_range(zero, dim_x, step, [&](Value* x) {
                Value* fx = to_f32(x);
                Value* fy = to_f32(y);
                Value* fz = to_f32(z);
                Value* px = fma(fx, step_x, origin_x);
                Value* py = fma(fy, step_y, origin_y);
                Value* pz = fma(fz, step_z, origin_z);

                Value* dist = eval_fn(px, py, pz);
                Value* z_dy = mul(z, dim_y);
                Value* z_dy_y = add(z_dy, y);
                Value* idx_x = mul(z_dy_y, dim_x);
                Value* linear_idx = add(idx_x, x);
                store_f32_indexed(out_field, linear_idx, dist, 4, 0);
            });
        });
    });
}

} // namespace brass::codegen
