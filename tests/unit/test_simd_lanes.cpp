// The lane compares, selects, shifts, conversions, bitcasts, roundings and
// vany: each op over memory operands, the interpreter's lanes checked against
// expected values and the x64 JIT's against the interpreter's, byte for byte,
// at both widths.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <functional>
#include <iostream>
#include <limits>

using namespace brass;

namespace {

using Body = std::function<Value*(Builder&, Value* a, Value* b, Value* c)>;

struct Buffers {
    alignas(32) uint8_t a[32] = {};
    alignas(32) uint8_t b[32] = {};
    alignas(32) uint8_t c[32] = {};
};

// fn(a, b, c, out): loads a, b, c as `in` vectors, stores body's result to
// out (a vector of the result's type, or an i32 for vany), returns 0.
void build_lane_fn(Module& mod, std::string_view name, Type in_a, Type in_b, Type in_c, const Body& body) {
    Function* fn = mod.create_function(name, Type::i32(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pc = b.add_block_param(entry, Type::ptr());
    Value* po = b.add_block_param(entry, Type::ptr());
    Value* va = b.build_vload(in_a, pa, 0);
    Value* vb = b.build_vload(in_b, pb, 0);
    Value* vc = b.build_vload(in_c, pc, 0);
    Value* r = body(b, va, vb, vc);
    if (r->type().is_vector()) b.build_vstore(r->type(), po, 0, r);
    else b.build_store(Type::i32(), po, 0, r);
    b.build_ret(b.build_iconst_i32(0));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

// Runs `name` on the interpreter and (on x64) the JIT; returns the
// interpreter's 32 output bytes after checking the JIT wrote the same.
std::array<uint8_t, 32> run_both(const Module& mod, std::string_view name, const Buffers& in, size_t out_bytes) {
    std::array<uint8_t, 32> interp_out{};
    alignas(32) uint8_t o1[32] = {};
    Interpreter interp;
    interp.run(mod, name, {RuntimeValue::from_ptr(in.a), RuntimeValue::from_ptr(in.b),
                           RuntimeValue::from_ptr(in.c), RuntimeValue::from_ptr(o1)});
    std::memcpy(interp_out.data(), o1, 32);
#if defined(__x86_64__) || defined(_M_X64)
    alignas(32) uint8_t o2[32] = {};
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    jit.invoke(name, {RuntimeValue::from_ptr(in.a), RuntimeValue::from_ptr(in.b),
                      RuntimeValue::from_ptr(in.c), RuntimeValue::from_ptr(o2)});
    const bool same = std::memcmp(o1, o2, out_bytes) == 0;
    if (!same) {
        std::cerr << name << ": JIT differs from the interpreter\n";
        print_module(mod, std::cerr);
    }
    CHECK(same);
#else
    (void)out_bytes;
#endif
    return interp_out;
}

template <typename T, size_t N>
void fill(uint8_t* dst, const T (&v)[N]) { std::memcpy(dst, v, sizeof(T) * N); }

template <typename T>
T lane(const std::array<uint8_t, 32>& o, size_t i) { T v; std::memcpy(&v, o.data() + i * sizeof(T), sizeof(T)); return v; }

const Opcode kCmps[] = {Opcode::vcmp_eq, Opcode::vcmp_ne, Opcode::vcmp_lt,
                        Opcode::vcmp_le, Opcode::vcmp_gt, Opcode::vcmp_ge};

} // namespace

TEST_CASE("SIMD lanes - compares at both widths, every lane type") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double dnan = std::numeric_limits<double>::quiet_NaN();
    Buffers f32in, f64in, i32in, i64in;
    fill(f32in.a, {1.f, 2.f, nan, -4.f, 5.f, 6.f, 7.f, -0.f});
    fill(f32in.b, {1.f, 3.f, 1.f, -5.f, 5.f, 2.f, nan, 0.f});
    fill(f64in.a, {1.0, dnan, -3.0, 4.0});
    fill(f64in.b, {1.0, 2.0, -2.0, 3.0});
    fill(i32in.a, {1, -2, 3, INT32_MIN, 5, 6, 7, 8});
    fill(i32in.b, {1, 2, -3, INT32_MAX, 5, 9, 7, 0});
    fill(i64in.a, {int64_t{1}, int64_t{-2}, INT64_MIN, int64_t{4}});
    fill(i64in.b, {int64_t{1}, int64_t{2}, INT64_MAX, int64_t{-4}});

    struct Case { Type t; const Buffers* in; };
    const Case cases[] = {
        {Type::f32x8(), &f32in}, {Type::f32x4(), &f32in}, {Type::f64x4(), &f64in}, {Type::f64x2(), &f64in},
        {Type::i32x8(), &i32in}, {Type::i32x4(), &i32in}, {Type::i64x4(), &i64in}, {Type::i64x2(), &i64in},
    };
    for (const Case& c : cases) {
        for (Opcode op : kCmps) {
            Module mod("lanes_cmp");
            std::string name = std::string(opcode_name(op)) + "_" + std::string(c.t.name());
            build_lane_fn(mod, name, c.t, c.t, c.t,
                          [&](Builder& b, Value* x, Value* y, Value*) { return b.build_vcmp(op, x, y); });
            auto o = run_both(mod, name, *c.in, c.t.size_in_bytes());
            // Spot checks: lane 0 is equal everywhere, lane 1 is a < b
            // except f64 (a NaN).
            const bool l0 = op == Opcode::vcmp_eq || op == Opcode::vcmp_le || op == Opcode::vcmp_ge;
            if (c.t.element_type().size_in_bytes() == 4) CHECK_EQ(lane<int32_t>(o, 0), l0 ? -1 : 0);
            else CHECK_EQ(lane<int64_t>(o, 0), l0 ? int64_t{-1} : int64_t{0});
            if (c.t.element_type() == Type::f64()) {
                CHECK_EQ(lane<int64_t>(o, 1), op == Opcode::vcmp_ne ? int64_t{-1} : int64_t{0});
            } else if (c.t.element_type().size_in_bytes() == 4) {
                const bool lt = op == Opcode::vcmp_ne || op == Opcode::vcmp_lt || op == Opcode::vcmp_le;
                CHECK_EQ(lane<int32_t>(o, 1), lt ? -1 : 0);
            }
        }
    }
}

TEST_CASE("SIMD lanes - select, shifts, bitcast, any") {
    Buffers sel;
    fill(sel.a, {-1, 0, -1, 0, 0, -1, 0, -1});
    fill(sel.b, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    fill(sel.c, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f, -7.f, -8.f});
    for (Type t : {Type::f32x8(), Type::f32x4(), Type::i32x8(), Type::f64x4(), Type::i64x4(), Type::f64x2()}) {
        Module mod("lanes_select");
        std::string name = "select_" + std::string(t.name());
        build_lane_fn(mod, name, t.vector_mask_type(), t, t,
                      [](Builder& b, Value* m, Value* x, Value* y) { return b.build_vselect(m, x, y); });
        auto o = run_both(mod, name, sel, t.size_in_bytes());
        if (t == Type::f32x8()) {
            CHECK_EQ(lane<float>(o, 0), 1.f);
            CHECK_EQ(lane<float>(o, 1), -2.f);
            CHECK_EQ(lane<float>(o, 7), 8.f);
        }
    }

    Buffers sh32, sh64;
    fill(sh32.a, {1, -8, INT32_MIN, 0x7fffffff, -1, 12345, -12345, 3});
    fill(sh32.b, {0, 1, 31, 32, 33, 4, 4, 100});
    fill(sh64.a, {int64_t{-8}, int64_t{-8}, INT64_MIN, int64_t{0x123456789}});
    fill(sh64.b, {int64_t{1}, int64_t{70}, int64_t{63}, int64_t{4}});
    for (Opcode op : {Opcode::vshl, Opcode::vlshr, Opcode::vashr}) {
        for (Type t : {Type::i32x8(), Type::i32x4(), Type::i64x4(), Type::i64x2()}) {
            Module mod("lanes_shift");
            std::string name = std::string(opcode_name(op)) + "_" + std::string(t.name());
            build_lane_fn(mod, name, t, t, t, [&](Builder& b, Value* x, Value* s, Value*) {
                return op == Opcode::vshl ? b.build_vshl(x, s) : op == Opcode::vlshr ? b.build_vlshr(x, s)
                                                                                    : b.build_vashr(x, s);
            });
            const bool w = t.element_type() == Type::i64();
            auto o = run_both(mod, name, w ? sh64 : sh32, t.size_in_bytes());
            if (op == Opcode::vashr && t == Type::i64x4()) {
                CHECK_EQ(lane<int64_t>(o, 0), int64_t{-4});
                CHECK_EQ(lane<int64_t>(o, 1), int64_t{-1});
                CHECK_EQ(lane<int64_t>(o, 2), int64_t{-1});
            }
            if (op == Opcode::vlshr && t == Type::i32x8()) {
                CHECK_EQ(lane<int32_t>(o, 3), 0);
                CHECK_EQ(lane<int32_t>(o, 1), 0x7ffffffc);
            }
        }
    }

    {
        Module mod("lanes_bitcast");
        build_lane_fn(mod, "bitcast", Type::f32x8(), Type::f32x8(), Type::f32x8(),
                      [](Builder& b, Value* x, Value*, Value*) { return b.build_vbitcast(Type::i32x8(), x); });
        auto o = run_both(mod, "bitcast", sel, 32);
        CHECK_EQ(lane<int32_t>(o, 0), -1);
    }

    Buffers none, some;
    fill(some.a, {0, 0, 0, 0, 0, 0, 0, INT32_MIN});
    for (Type t : {Type::i32x8(), Type::i32x4(), Type::i64x4(), Type::i64x2()}) {
        Module mod("lanes_any");
        std::string name = "any_" + std::string(t.name());
        build_lane_fn(mod, name, t, t, t, [](Builder& b, Value* m, Value*, Value*) { return b.build_vany(m); });
        CHECK_EQ(lane<int32_t>(run_both(mod, name, none, 4), 0), 0);
        // The set lane is the last of the 256-bit vectors only.
        CHECK_EQ(lane<int32_t>(run_both(mod, name, some, 4), 0), t.is_v256() ? 1 : 0);
    }
}

TEST_CASE("SIMD lanes - conversions and roundings") {
    Buffers fl, db, in32, in64;
    fill(fl.a, {1.5f, -1.5f, 2.5f, -2.5f, 3e9f, -3e9f, 0.49999997f, -0.f});
    fill(db.a, {1.5, -1.5, 2.5, -3.7e9});
    fill(in32.a, {1, -1, INT32_MAX, INT32_MIN, 7, -7, 1 << 24 | 1, 0});
    fill(in64.a, {int64_t{0x100000005}, int64_t{-1}, INT64_MIN, int64_t{0x7fffffff}});

    struct Conv { Type from, to; const Buffers* in; };
    const Conv convs[] = {
        {Type::i32x8(), Type::f32x8(), &in32}, {Type::i32x4(), Type::f32x4(), &in32},
        {Type::f32x8(), Type::i32x8(), &fl},   {Type::f32x4(), Type::i32x4(), &fl},
        {Type::i32x4(), Type::f64x4(), &in32}, {Type::f64x4(), Type::i32x4(), &db},
        {Type::f32x4(), Type::f64x4(), &fl},   {Type::f64x4(), Type::f32x4(), &db},
        {Type::i32x4(), Type::i64x4(), &in32}, {Type::i64x4(), Type::i32x4(), &in64},
    };
    for (const Conv& c : convs) {
        Module mod("lanes_convert");
        std::string name = "convert_" + std::string(c.from.name()) + "_" + std::string(c.to.name());
        build_lane_fn(mod, name, c.from, c.from, c.from,
                      [&](Builder& b, Value* x, Value*, Value*) { return b.build_vconvert(c.to, x); });
        auto o = run_both(mod, name, *c.in, c.to.size_in_bytes());
        if (c.from == Type::f32x8() && c.to == Type::i32x8()) {
            CHECK_EQ(lane<int32_t>(o, 0), 1);
            CHECK_EQ(lane<int32_t>(o, 1), -1);
            CHECK_EQ(lane<int32_t>(o, 4), INT32_MIN);   // out of range
        }
        if (c.from == Type::f64x4() && c.to == Type::i32x4()) {
            CHECK_EQ(lane<int32_t>(o, 1), -1);
            CHECK_EQ(lane<int32_t>(o, 3), INT32_MIN);
        }
        if (c.from == Type::i64x4()) {
            CHECK_EQ(lane<int32_t>(o, 0), 5);
            CHECK_EQ(lane<int32_t>(o, 1), -1);
            CHECK_EQ(lane<int32_t>(o, 2), 0);
            CHECK_EQ(lane<int32_t>(o, 3), 0x7fffffff);
        }
        if (c.from == Type::i32x4() && c.to == Type::i64x4()) CHECK_EQ(lane<int64_t>(o, 3), int64_t{INT32_MIN});
    }

    const Opcode rounds[] = {Opcode::vfloor, Opcode::vceil, Opcode::vtrunc, Opcode::vround};
    for (Opcode op : rounds) {
        for (Type t : {Type::f32x8(), Type::f32x4(), Type::f64x4(), Type::f64x2()}) {
            Module mod("lanes_round");
            std::string name = std::string(opcode_name(op)) + "_" + std::string(t.name());
            build_lane_fn(mod, name, t, t, t, [&](Builder& b, Value* x, Value*, Value*) {
                switch (op) {
                    case Opcode::vfloor: return b.build_vfloor(x);
                    case Opcode::vceil:  return b.build_vceil(x);
                    case Opcode::vtrunc: return b.build_vtrunc(x);
                    default:             return b.build_vround(x);
                }
            });
            auto o = run_both(mod, name, t.element_type() == Type::f32() ? fl : db, t.size_in_bytes());
            if (t == Type::f32x8()) {
                const float want0 = op == Opcode::vceil ? 2.f : op == Opcode::vround ? 2.f : 1.f;
                const float want2 = op == Opcode::vceil ? 3.f : 2.f;   // 2.5 rounds to even
                CHECK_EQ(lane<float>(o, 0), want0);
                CHECK_EQ(lane<float>(o, 2), want2);
            }
        }
    }
}
