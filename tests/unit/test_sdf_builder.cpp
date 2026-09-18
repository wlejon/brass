#include "test_framework.hpp"
#include <brass/codegen/sdf_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/target.hpp>

#include <vector>
#include <cmath>
#include <algorithm>

using namespace brass;
using namespace brass::codegen;

#ifndef CHECK_NEAR
#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))
#endif

namespace {

float ref_sphere(float px, float py, float pz, float radius) {
    return std::sqrt(px * px + py * py + pz * pz) - radius;
}

float ref_box(float px, float py, float pz, float bx, float by, float bz) {
    float qx = std::abs(px) - bx;
    float qy = std::abs(py) - by;
    float qz = std::abs(pz) - bz;
    float mx = std::max(qx, 0.0f);
    float my = std::max(qy, 0.0f);
    float mz = std::max(qz, 0.0f);
    float outside = std::sqrt(mx * mx + my * my + mz * mz);
    float inside = std::min(std::max(qx, std::max(qy, qz)), 0.0f);
    return outside + inside;
}

float ref_cylinder(float px, float py, float pz, float radius, float half_height) {
    float dx = std::sqrt(px * px + pz * pz) - radius;
    float dy = std::abs(py) - half_height;
    float mx = std::max(dx, 0.0f);
    float my = std::max(dy, 0.0f);
    float outside = std::sqrt(mx * mx + my * my);
    float inside = std::min(std::max(dx, dy), 0.0f);
    return outside + inside;
}

float ref_torus(float px, float py, float pz, float major_r, float minor_r) {
    float qx = std::sqrt(px * px + pz * pz) - major_r;
    float qy = py;
    return std::sqrt(qx * qx + qy * qy) - minor_r;
}

float ref_union(float d1, float d2) {
    return std::min(d1, d2);
}

float ref_smooth_union(float d1, float d2, float k) {
    float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return (d2 * (1.0f - h) + d1 * h) - k * h * (1.0f - h);
}

float ref_rounded_box(float px, float py, float pz, float bx, float by, float bz, float r) {
    float qx = std::abs(px) - bx + r;
    float qy = std::abs(py) - by + r;
    float qz = std::abs(pz) - bz + r;
    float mx = std::max(qx, 0.0f);
    float my = std::max(qy, 0.0f);
    float mz = std::max(qz, 0.0f);
    float outside = std::sqrt(mx * mx + my * my + mz * mz);
    float inside = std::min(std::max(qx, std::max(qy, qz)), 0.0f);
    return outside + inside - r;
}

float ref_capsule(float px, float py, float pz,
                  float ax, float ay, float az,
                  float bx, float by, float bz,
                  float r) {
    float pax = px - ax, pay = py - ay, paz = pz - az;
    float bax = bx - ax, bay = by - ay, baz = bz - az;
    float dot_pa_ba = pax * bax + pay * bay + paz * baz;
    float dot_ba_ba = bax * bax + bay * bay + baz * baz;
    float h = std::clamp(dot_pa_ba / dot_ba_ba, 0.0f, 1.0f);
    float vx = pax - bax * h;
    float vy = pay - bay * h;
    float vz = paz - baz * h;
    return std::sqrt(vx * vx + vy * vy + vz * vz) - r;
}

float ref_plane(float px, float py, float pz, float nx, float ny, float nz, float d) {
    return px * nx + py * ny + pz * nz + d;
}

float ref_intersection(float d1, float d2) {
    return std::max(d1, d2);
}

float ref_subtraction(float d1, float d2) {
    return std::max(d1, -d2);
}

float ref_smooth_intersection(float d1, float d2, float k) {
    float h = std::clamp(0.5f - 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return (d2 * (1.0f - h) + d1 * h) + k * h * (1.0f - h);
}

float ref_smooth_subtraction(float d1, float d2, float k) {
    return ref_smooth_intersection(d1, -d2, k);
}

void ref_rotate_y(float px, float py, float pz, float angle, float& ox, float& oy, float& oz) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    ox = px * c - pz * s;
    oy = py;
    oz = px * s + pz * c;
}

} // namespace

TEST_CASE("SDF JIT - Sphere Distance Field Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_sphere_mod");
    Function* fn = mod.create_function("eval_sphere", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* radius = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* d = kb.sdf_sphere(x, y, z, radius);
        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t, float)>();

    std::vector<float> test_x = {0.0f, 1.0f, 0.0f, 3.0f, -2.0f, 0.5f, -0.5f, 10.0f};
    std::vector<float> test_y = {0.0f, 0.0f, 2.0f, 4.0f, 2.0f, -0.5f, 0.5f, -10.0f};
    std::vector<float> test_z = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.5f, -0.5f, 5.0f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);
    float r = 1.5f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n, r);

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_sphere(test_x[static_cast<size_t>(i)],
                                    test_y[static_cast<size_t>(i)],
                                    test_z[static_cast<size_t>(i)], r);
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Exact Box Distance Field Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_box_mod");
    Function* fn = mod.create_function("eval_box", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* bx = kb.builder().add_block_param(entry, Type::f32());
    Value* by = kb.builder().add_block_param(entry, Type::f32());
    Value* bz = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* d = kb.sdf_box(x, y, z, bx, by, bz);
        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t, float, float, float)>();

    std::vector<float> test_x = {0.0f, 1.0f, 2.0f, -2.5f, 0.5f, 3.0f, -1.0f};
    std::vector<float> test_y = {0.0f, 0.5f, 0.0f, 1.5f, -0.2f, 2.0f, -1.0f};
    std::vector<float> test_z = {0.0f, 0.2f, -1.0f, 0.0f, 0.8f, -4.0f, 0.5f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);
    float box_x = 1.0f, box_y = 1.0f, box_z = 1.0f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n, box_x, box_y, box_z);

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_box(test_x[static_cast<size_t>(i)],
                                 test_y[static_cast<size_t>(i)],
                                 test_z[static_cast<size_t>(i)],
                                 box_x, box_y, box_z);
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Capped Cylinder Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_cylinder_mod");
    Function* fn = mod.create_function("eval_cylinder", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* radius = kb.builder().add_block_param(entry, Type::f32());
    Value* half_h = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* d = kb.sdf_cylinder(x, y, z, radius, half_h);
        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t, float, float)>();

    std::vector<float> test_x = {0.0f, 1.0f, 0.0f, 2.0f, 1.5f, 0.5f};
    std::vector<float> test_y = {0.0f, 0.0f, 3.0f, 1.0f, -2.5f, 0.5f};
    std::vector<float> test_z = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, -0.5f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);
    float r = 1.0f, h = 2.0f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n, r, h);

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_cylinder(test_x[static_cast<size_t>(i)],
                                     test_y[static_cast<size_t>(i)],
                                     test_z[static_cast<size_t>(i)], r, h);
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Torus Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_torus_mod");
    Function* fn = mod.create_function("eval_torus", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* maj_r = kb.builder().add_block_param(entry, Type::f32());
    Value* min_r = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* d = kb.sdf_torus(x, y, z, maj_r, min_r);
        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t, float, float)>();

    std::vector<float> test_x = {2.0f, 2.5f, 0.0f, 1.0f, -2.0f};
    std::vector<float> test_y = {0.0f, 0.5f, 1.0f, 0.0f, -0.3f};
    std::vector<float> test_z = {0.0f, 0.0f, 0.0f, 2.0f, 1.0f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);
    float major_r = 2.0f, minor_r = 0.5f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n, major_r, minor_r);

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_torus(test_x[static_cast<size_t>(i)],
                                   test_y[static_cast<size_t>(i)],
                                   test_z[static_cast<size_t>(i)],
                                   major_r, minor_r);
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - CSG Union and Smooth Union") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_csg_mod");
    Function* fn = mod.create_function("eval_csg", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* d1_p = kb.builder().add_block_param(entry, Type::ptr());
    Value* d2_p = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_u = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_su = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* k = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* d1 = kb.load_f32_indexed(d1_p, i, 4, 0);
        Value* d2 = kb.load_f32_indexed(d2_p, i, 4, 0);
        Value* u = kb.op_union(d1, d2);
        Value* su = kb.op_smooth_union(d1, d2, k);
        kb.store_f32_indexed(out_u, i, u, 4, 0);
        kb.store_f32_indexed(out_su, i, su, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, float*, float*, int64_t, float)>();

    std::vector<float> d1_vals = {1.0f, -0.5f, 0.2f, 2.0f, -1.0f, 0.05f};
    std::vector<float> d2_vals = {0.8f, 0.1f, 0.3f, -0.5f, -1.2f, -0.05f};
    int64_t n = static_cast<int64_t>(d1_vals.size());
    std::vector<float> res_u(n, 0.0f);
    std::vector<float> res_su(n, 0.0f);
    float k_val = 0.4f;

    fn_ptr(d1_vals.data(), d2_vals.data(), res_u.data(), res_su.data(), n, k_val);

    for (int64_t i = 0; i < n; ++i) {
        float expected_u = ref_union(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)]);
        float expected_su = ref_smooth_union(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)], k_val);
        CHECK_NEAR(res_u[static_cast<size_t>(i)], expected_u, 1e-5f);
        CHECK_NEAR(res_su[static_cast<size_t>(i)], expected_su, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Translation and Uniform Scale") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_xform_mod");
    Function* fn = mod.create_function("eval_xform", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* ox = kb.builder().add_block_param(entry, Type::f32());
    Value* oy = kb.builder().add_block_param(entry, Type::f32());
    Value* oz = kb.builder().add_block_param(entry, Type::f32());
    Value* scale = kb.builder().add_block_param(entry, Type::f32());
    Value* radius = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);

        Value* tx = nullptr;
        Value* ty = nullptr;
        Value* tz = nullptr;
        kb.translate(x, y, z, ox, oy, oz, tx, ty, tz);

        Value* sx = nullptr;
        Value* sy = nullptr;
        Value* sz = nullptr;
        kb.scale_uniform(tx, ty, tz, scale, sx, sy, sz);

        Value* unscaled_d = kb.sdf_sphere(sx, sy, sz, radius);
        Value* d = kb.scale_uniform_dist(unscaled_d, scale);

        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t,
                                float, float, float, float, float)>();

    std::vector<float> test_x = {2.0f, 3.0f, 0.0f, -1.0f};
    std::vector<float> test_y = {3.0f, 1.0f, 0.0f, -2.0f};
    std::vector<float> test_z = {4.0f, 0.0f, 0.0f, 1.0f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);

    float host_ox = 2.0f, host_oy = 3.0f, host_oz = 4.0f;
    float host_s = 2.0f;
    float host_r = 1.0f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n,
           host_ox, host_oy, host_oz, host_s, host_r);

    for (int64_t i = 0; i < n; ++i) {
        float tx = (test_x[static_cast<size_t>(i)] - host_ox) / host_s;
        float ty = (test_y[static_cast<size_t>(i)] - host_oy) / host_s;
        float tz = (test_z[static_cast<size_t>(i)] - host_oz) / host_s;
        float expected = (std::sqrt(tx * tx + ty * ty + tz * tz) - host_r) * host_s;
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - 3D Noise Evaluation Range and Continuity") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_noise_mod");
    Function* fn = mod.create_function("eval_noise", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_n = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* n_val = kb.noise_3d(x, y, z);
        kb.store_f32_indexed(out_n, i, n_val, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t)>();

    constexpr int N = 50;
    std::vector<float> test_x(N);
    std::vector<float> test_y(N);
    std::vector<float> test_z(N);
    std::vector<float> out_noise(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        test_x[static_cast<size_t>(i)] = -2.5f + static_cast<float>(i) * 0.1f;
        test_y[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i) * 0.05f;
        test_z[static_cast<size_t>(i)] = static_cast<float>(i) * -0.08f;
    }

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out_noise.data(), N);

    float min_val = 100.0f;
    float max_val = -100.0f;
    for (int i = 0; i < N; ++i) {
        float val = out_noise[static_cast<size_t>(i)];
        if (val < -1.0f || val > 1.0f) {
            std::cout << "DEBUG noise[" << i << "]: " << val << "\n";
        }
        CHECK(val >= -1.0f);
        CHECK(val <= 1.0f);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    // Verify non-trivial range
    CHECK(max_val > min_val);
}

TEST_CASE("SDF JIT - Grid Evaluation Loop against Reference Volume") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_grid_mod");
    Function* fn = mod.create_function("eval_grid", Type::void_type(), {
        Type::ptr(), Type::i64(), Type::i64(), Type::i64(),
        Type::f32(), Type::f32(), Type::f32(),
        Type::f32(), Type::f32(), Type::f32(),
        Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* out_field = kb.builder().add_block_param(entry, Type::ptr());
    Value* dim_x = kb.builder().add_block_param(entry, Type::i64());
    Value* dim_y = kb.builder().add_block_param(entry, Type::i64());
    Value* dim_z = kb.builder().add_block_param(entry, Type::i64());
    Value* ox = kb.builder().add_block_param(entry, Type::f32());
    Value* oy = kb.builder().add_block_param(entry, Type::f32());
    Value* oz = kb.builder().add_block_param(entry, Type::f32());
    Value* sx = kb.builder().add_block_param(entry, Type::f32());
    Value* sy = kb.builder().add_block_param(entry, Type::f32());
    Value* sz = kb.builder().add_block_param(entry, Type::f32());
    Value* radius = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.emit_grid_eval_loop(out_field, dim_x, dim_y, dim_z, ox, oy, oz, sx, sy, sz,
                           [&](Value* px, Value* py, Value* pz) {
        return kb.sdf_sphere(px, py, pz, radius);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(float*, int64_t, int64_t, int64_t,
                                float, float, float,
                                float, float, float,
                                float)>();

    const int64_t dx = 8;
    const int64_t dy = 8;
    const int64_t dz = 8;
    const size_t total_voxels = static_cast<size_t>(dx * dy * dz);
    std::vector<float> field(total_voxels, 0.0f);

    float origin_x = -1.0f, origin_y = -1.0f, origin_z = -1.0f;
    float step_x = 0.25f, step_y = 0.25f, step_z = 0.25f;
    float sphere_r = 0.75f;

    fn_ptr(field.data(), dx, dy, dz,
           origin_x, origin_y, origin_z,
           step_x, step_y, step_z,
           sphere_r);

    for (int64_t z = 0; z < dz; ++z) {
        for (int64_t y = 0; y < dy; ++y) {
            for (int64_t x = 0; x < dx; ++x) {
                float px = origin_x + static_cast<float>(x) * step_x;
                float py = origin_y + static_cast<float>(y) * step_y;
                float pz = origin_z + static_cast<float>(z) * step_z;
                float expected = ref_sphere(px, py, pz, sphere_r);
                size_t idx = static_cast<size_t>((z * dy + y) * dx + x);
                CHECK_NEAR(field[idx], expected, 1e-5f);
            }
        }
    }
}

TEST_CASE("SDF JIT - Rounded Box Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_rbox_mod");
    Function* fn = mod.create_function("eval_rbox", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32(), Type::f32(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_d = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* bx = kb.builder().add_block_param(entry, Type::f32());
    Value* by = kb.builder().add_block_param(entry, Type::f32());
    Value* bz = kb.builder().add_block_param(entry, Type::f32());
    Value* rad = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);
        Value* d = kb.sdf_rounded_box(x, y, z, bx, by, bz, rad);
        kb.store_f32_indexed(out_d, i, d, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t,
                                float, float, float, float)>();

    std::vector<float> test_x = {0.0f, 1.0f, 1.2f, -1.5f, 0.5f};
    std::vector<float> test_y = {0.0f, 0.5f, 1.2f, 0.0f, -0.8f};
    std::vector<float> test_z = {0.0f, 0.0f, 1.2f, 0.8f, 0.2f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> out(n, 0.0f);
    float box_x = 1.0f, box_y = 1.0f, box_z = 1.0f, r = 0.2f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(), out.data(), n, box_x, box_y, box_z, r);

    for (int64_t i = 0; i < n; ++i) {
        float expected = ref_rounded_box(test_x[static_cast<size_t>(i)],
                                         test_y[static_cast<size_t>(i)],
                                         test_z[static_cast<size_t>(i)],
                                         box_x, box_y, box_z, r);
        CHECK_NEAR(out[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Capsule and Plane Evaluation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_cap_plane_mod");
    Function* fn = mod.create_function("eval_cap_plane", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(),
        Type::f32(), Type::f32(), Type::f32(),
        Type::f32(), Type::f32(), Type::f32(),
        Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* xs = kb.builder().add_block_param(entry, Type::ptr());
    Value* ys = kb.builder().add_block_param(entry, Type::ptr());
    Value* zs = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_cap = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_plane = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* ax = kb.builder().add_block_param(entry, Type::f32());
    Value* ay = kb.builder().add_block_param(entry, Type::f32());
    Value* az = kb.builder().add_block_param(entry, Type::f32());
    Value* bx = kb.builder().add_block_param(entry, Type::f32());
    Value* by = kb.builder().add_block_param(entry, Type::f32());
    Value* bz = kb.builder().add_block_param(entry, Type::f32());
    Value* r = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(xs, i, 4, 0);
        Value* y = kb.load_f32_indexed(ys, i, 4, 0);
        Value* z = kb.load_f32_indexed(zs, i, 4, 0);

        Value* d_cap = kb.sdf_capsule(x, y, z, ax, ay, az, bx, by, bz, r);
        // Plane at Y = 0 with upward normal (0, 1, 0) and d = 0
        Value* d_plane = kb.sdf_plane(x, y, z, kb.const_f32(0.0f), kb.const_f32(1.0f), kb.const_f32(0.0f), kb.const_f32(0.0f));

        kb.store_f32_indexed(out_cap, i, d_cap, 4, 0);
        kb.store_f32_indexed(out_plane, i, d_plane, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, float*, int64_t,
                                float, float, float, float, float, float, float)>();

    std::vector<float> test_x = {0.0f, 1.0f, 0.0f, 2.0f};
    std::vector<float> test_y = {0.0f, 1.0f, 3.0f, -1.0f};
    std::vector<float> test_z = {0.0f, 0.0f, 1.0f, 0.0f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> res_cap(n, 0.0f);
    std::vector<float> res_plane(n, 0.0f);

    float ax_val = 0.0f, ay_val = -1.0f, az_val = 0.0f;
    float bx_val = 0.0f, by_val = 1.0f, bz_val = 0.0f;
    float r_val = 0.5f;

    fn_ptr(test_x.data(), test_y.data(), test_z.data(),
           res_cap.data(), res_plane.data(), n,
           ax_val, ay_val, az_val, bx_val, by_val, bz_val, r_val);

    for (int64_t i = 0; i < n; ++i) {
        float exp_cap = ref_capsule(test_x[static_cast<size_t>(i)],
                                   test_y[static_cast<size_t>(i)],
                                   test_z[static_cast<size_t>(i)],
                                   ax_val, ay_val, az_val, bx_val, by_val, bz_val, r_val);
        float exp_plane = ref_plane(test_x[static_cast<size_t>(i)],
                                    test_y[static_cast<size_t>(i)],
                                    test_z[static_cast<size_t>(i)],
                                    0.0f, 1.0f, 0.0f, 0.0f);
        CHECK_NEAR(res_cap[static_cast<size_t>(i)], exp_cap, 1e-5f);
        CHECK_NEAR(res_plane[static_cast<size_t>(i)], exp_plane, 1e-5f);
    }
}

TEST_CASE("SDF JIT - CSG Subtraction, Intersection and Smooth Variants") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_csg_ops_mod");
    Function* fn = mod.create_function("eval_csg_ops", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* d1_p = kb.builder().add_block_param(entry, Type::ptr());
    Value* d2_p = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_sub = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_inter = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_ssub = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_sinter = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* k = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* d1 = kb.load_f32_indexed(d1_p, i, 4, 0);
        Value* d2 = kb.load_f32_indexed(d2_p, i, 4, 0);

        Value* sub_v = kb.op_subtraction(d1, d2);
        Value* inter_v = kb.op_intersection(d1, d2);
        Value* ssub_v = kb.op_smooth_subtraction(d1, d2, k);
        Value* sinter_v = kb.op_smooth_intersection(d1, d2, k);

        kb.store_f32_indexed(out_sub, i, sub_v, 4, 0);
        kb.store_f32_indexed(out_inter, i, inter_v, 4, 0);
        kb.store_f32_indexed(out_ssub, i, ssub_v, 4, 0);
        kb.store_f32_indexed(out_sinter, i, sinter_v, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, float*, float*, float*, float*, int64_t, float)>();

    std::vector<float> d1_vals = {0.5f, -0.2f, 1.0f, -0.8f};
    std::vector<float> d2_vals = {0.3f, -0.1f, -0.5f, 0.4f};
    int64_t n = static_cast<int64_t>(d1_vals.size());
    std::vector<float> r_sub(n), r_inter(n), r_ssub(n), r_sinter(n);
    float k_val = 0.25f;

    fn_ptr(d1_vals.data(), d2_vals.data(),
           r_sub.data(), r_inter.data(), r_ssub.data(), r_sinter.data(),
           n, k_val);

    for (int64_t i = 0; i < n; ++i) {
        float exp_sub = ref_subtraction(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)]);
        float exp_inter = ref_intersection(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)]);
        float exp_ssub = ref_smooth_subtraction(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)], k_val);
        float exp_sinter = ref_smooth_intersection(d1_vals[static_cast<size_t>(i)], d2_vals[static_cast<size_t>(i)], k_val);

        CHECK_NEAR(r_sub[static_cast<size_t>(i)], exp_sub, 1e-5f);
        CHECK_NEAR(r_inter[static_cast<size_t>(i)], exp_inter, 1e-5f);
        CHECK_NEAR(r_ssub[static_cast<size_t>(i)], exp_ssub, 1e-5f);
        CHECK_NEAR(r_sinter[static_cast<size_t>(i)], exp_sinter, 1e-5f);
    }
}

TEST_CASE("SDF JIT - Rotate Y Transformation") {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] non-supported host arch\n";
        return;
    }

    Module mod("sdf_rot_mod");
    Function* fn = mod.create_function("eval_rot", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i64(), Type::f32()
    });

    SdfKernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    Value* in_x = kb.builder().add_block_param(entry, Type::ptr());
    Value* in_y = kb.builder().add_block_param(entry, Type::ptr());
    Value* in_z = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_x = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_y = kb.builder().add_block_param(entry, Type::ptr());
    Value* out_z = kb.builder().add_block_param(entry, Type::ptr());
    Value* count = kb.builder().add_block_param(entry, Type::i64());
    Value* angle = kb.builder().add_block_param(entry, Type::f32());

    kb.position_at_end(entry);
    kb.for_range(kb.const_i64(0), count, kb.const_i64(1), [&](Value* i) {
        Value* x = kb.load_f32_indexed(in_x, i, 4, 0);
        Value* y = kb.load_f32_indexed(in_y, i, 4, 0);
        Value* z = kb.load_f32_indexed(in_z, i, 4, 0);

        Value* rx = nullptr;
        Value* ry = nullptr;
        Value* rz = nullptr;
        kb.rotate_y(x, y, z, angle, rx, ry, rz);

        kb.store_f32_indexed(out_x, i, rx, 4, 0);
        kb.store_f32_indexed(out_y, i, ry, 4, 0);
        kb.store_f32_indexed(out_z, i, rz, 4, 0);
    });
    kb.builder().build_ret_void();

    KernelJit jit;
    KernelFunction kfn = jit.compile(*fn);
    CHECK(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*,
                                float*, float*, float*,
                                int64_t, float)>();

    std::vector<float> test_x = {1.0f, 0.0f, -1.0f, 2.0f};
    std::vector<float> test_y = {0.0f, 1.5f, -0.5f, 3.0f};
    std::vector<float> test_z = {0.0f, 1.0f, 2.0f, -1.0f};
    int64_t n = static_cast<int64_t>(test_x.size());
    std::vector<float> ox(n), oy(n), oz(n);
    float angle_rad = 0.78539816339f; // pi / 4

    fn_ptr(test_x.data(), test_y.data(), test_z.data(),
           ox.data(), oy.data(), oz.data(),
           n, angle_rad);

    for (int64_t i = 0; i < n; ++i) {
        float exp_x, exp_y, exp_z;
        ref_rotate_y(test_x[static_cast<size_t>(i)],
                     test_y[static_cast<size_t>(i)],
                     test_z[static_cast<size_t>(i)],
                     angle_rad, exp_x, exp_y, exp_z);
        CHECK_NEAR(ox[static_cast<size_t>(i)], exp_x, 1e-5f);
        CHECK_NEAR(oy[static_cast<size_t>(i)], exp_y, 1e-5f);
        CHECK_NEAR(oz[static_cast<size_t>(i)], exp_z, 1e-5f);
    }
}

