#pragma once

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>

#include <cstdint>
#include <functional>

namespace brass::codegen {

// ─── SdfKernelBuilder ────────────────────────────────────────────────────────
//
// Specialized MIR builder for Signed Distance Field (SDF) evaluation, procedural
// implicit geometry generation, CSG combinations, transformations, 3D lattice
// noise, and 3D dense grid volume evaluation kernels.
class SdfKernelBuilder : public KernelBuilder {
public:
    SdfKernelBuilder(Module& mod, Function* fn = nullptr)
        : KernelBuilder(mod, fn) {}
    explicit SdfKernelBuilder(Builder& b) noexcept
        : KernelBuilder(b) {}

    // ── Scalar Math Helpers ─────────────────────────────────────────────────
    // Branchless absolute value: select(slt(v, 0), neg(v), v)
    Value* abs_f32(Value* v);

    // Branchless minimum: select(slt(a, b), a, b)
    Value* min_f32(Value* a, Value* b);

    // Branchless maximum: select(sgt(a, b), a, b)
    Value* max_f32(Value* a, Value* b);

    // Clamp x to [min_val, max_val]
    Value* clamp_f32(Value* x, Value* min_val, Value* max_val);

    // Square root via external "sqrtf" symbol
    Value* sqrt_f32(Value* x);

    // 2D Euclidean length: sqrt(x^2 + y^2)
    Value* length2(Value* x, Value* y);

    // 3D Euclidean length: sqrt(x^2 + y^2 + z^2)
    Value* length3(Value* x, Value* y, Value* z);

    // Linear interpolation: a + (b - a) * t
    Value* lerp_f32(Value* a, Value* b, Value* t);

    // Converts integer (i32/i64) or float to f32
    Value* to_f32(Value* val);

    // Floor function for f32 returning float
    Value* floor_f32(Value* x);

    // ── Primitives ──────────────────────────────────────────────────────────
    // Sphere centered at origin with given radius
    Value* sdf_sphere(Value* px, Value* py, Value* pz, Value* radius);

    // Exact box SDF centered at origin with half-extents (bx, by, bz)
    Value* sdf_box(Value* px, Value* py, Value* pz, Value* bx, Value* by, Value* bz);

    // Exact rounded box SDF with half-extents (bx, by, bz) and rounding radius
    Value* sdf_rounded_box(Value* px, Value* py, Value* pz, Value* bx, Value* by, Value* bz, Value* radius);

    // Capped cylinder along Y axis centered at origin with given radius and half_height
    Value* sdf_cylinder(Value* px, Value* py, Value* pz, Value* radius, Value* half_height);

    // Capsule segment from point a(ax,ay,az) to point b(bx,by,bz) with radius
    Value* sdf_capsule(Value* px, Value* py, Value* pz,
                       Value* ax, Value* ay, Value* az,
                       Value* bx, Value* by, Value* bz,
                       Value* radius);

    // Torus in XZ plane centered at origin with major_r (ring) and minor_r (tube)
    Value* sdf_torus(Value* px, Value* py, Value* pz, Value* major_r, Value* minor_r);

    // Infinite plane with unit normal (nx, ny, nz) and signed offset d
    Value* sdf_plane(Value* px, Value* py, Value* pz, Value* nx, Value* ny, Value* nz, Value* d);

    // ── CSG Combinators ─────────────────────────────────────────────────────
    // Union: min(d1, d2)
    Value* op_union(Value* d1, Value* d2);

    // Intersection: max(d1, d2)
    Value* op_intersection(Value* d1, Value* d2);

    // Subtraction: max(d1, -d2)
    Value* op_subtraction(Value* d1, Value* d2);

    // Polynomial smooth union (smooth min) with blend radius k
    Value* op_smooth_union(Value* d1, Value* d2, Value* k);

    // Polynomial smooth intersection (smooth max) with blend radius k
    Value* op_smooth_intersection(Value* d1, Value* d2, Value* k);

    // Polynomial smooth subtraction with blend radius k
    Value* op_smooth_subtraction(Value* d1, Value* d2, Value* k);

    // ── Transforms ──────────────────────────────────────────────────────────
    // Point translation by offset (ox, oy, oz): p' = p - offset
    void translate(Value* px, Value* py, Value* pz,
                   Value* ox, Value* oy, Value* oz,
                   Value*& out_x, Value*& out_y, Value*& out_z);

    // Rotation around Y axis by angle_rad (radians): inverse rotation on query point
    void rotate_y(Value* px, Value* py, Value* pz, Value* angle_rad,
                  Value*& out_x, Value*& out_y, Value*& out_z);

    // Uniform scale: query coordinates are divided by s
    void scale_uniform(Value* px, Value* py, Value* pz, Value* s,
                       Value*& out_x, Value*& out_y, Value*& out_z);

    // Scale distance value back: dist * s
    Value* scale_uniform_dist(Value* dist, Value* s);

    // ── Procedural Noise ────────────────────────────────────────────────────
    // 3D hash/gradient lattice noise returning float in [-1, 1]
    Value* noise_3d(Value* px, Value* py, Value* pz);

    // ── Grid Loop Emitter ───────────────────────────────────────────────────
    // Helper that emits 3D loop over (dim_z, dim_y, dim_x) bounds,
    // calls eval_fn(x, y, z) to obtain distance, and stores to
    // out_field[(z * dim_y + y) * dim_x + x].
    void emit_grid_eval_loop(
        Value* out_field,
        Value* dim_x, Value* dim_y, Value* dim_z,
        const std::function<Value*(Value* x, Value* y, Value* z)>& eval_fn
    );

    // Helper that evaluates grid with linear coordinate mapping from grid indices to
    // world space coordinates: p = origin + index * step.
    void emit_grid_eval_loop(
        Value* out_field,
        Value* dim_x, Value* dim_y, Value* dim_z,
        Value* origin_x, Value* origin_y, Value* origin_z,
        Value* step_x, Value* step_y, Value* step_z,
        const std::function<Value*(Value* px, Value* py, Value* pz)>& eval_fn
    );
};

} // namespace brass::codegen
