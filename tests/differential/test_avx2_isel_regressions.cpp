#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/target.hpp>
#include <iostream>
#include <string>

// x64 256-bit vector instruction selection regressions:
//  - vbroadcast to i32x8 / i64x4 took its scalar straight from a GPR into
//    VPBROADCASTD/Q, whose source is an xmm: the lanes held whatever xmm had
//    the GPR's number.
//  - vneg, vnot and vsqrt on 256-bit values used 16-byte SSE instructions.
//  - a 256-bit op with no 256-bit instruction (integer vdiv) fell through
//    to the 128-bit path with no opcode: the result was silently operand 0.
//    It is now a compile error. i64x4 vmul, which AVX2 lacks too, goes
//    through 64-bit imul lane by lane, as i64x2's does.
// They run on AArch64 too, where a 256-bit value is a pair of V registers;
// there an f64 lane-0 insert was a scalar fmov that cleared lane 1.

using namespace brass;
using namespace brass::test;

namespace {

// fn(v: vec) = vbroadcast(k) + vbroadcast(v[2]) + v, with k a constant of
// the lane type: both scalars live in a register of the scalar's class (a
// GPR for the integer lanes) when the broadcast is selected.
void build_broadcast_add(Module& mod, std::string_view name, Type vec, Type scalar) {
    Function* fn = mod.create_function(name, vec, {vec});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* v = b.add_block_param(entry, vec);
    Value* k = nullptr;
    switch (scalar.kind()) {
        case TypeKind::I32: k = b.build_iconst_i32(1700); break;
        case TypeKind::I64: k = b.build_iconst_i64(-123456789012LL); break;
        case TypeKind::F32: k = b.build_fconst_f32(2.5f); break;
        default:            k = b.build_fconst_f64(-0.75); break;
    }
    Value* lane = b.build_vextract_lane(v, 2);
    Value* sum = b.build_vadd(b.build_vbroadcast(vec, k), b.build_vbroadcast(vec, lane));
    b.build_ret(b.build_vadd(sum, v));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

// fn(v: vec) = op(v)
void build_unary(Module& mod, std::string_view name, Type vec, Opcode op) {
    Function* fn = mod.create_function(name, vec, {vec});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* v = b.add_block_param(entry, vec);
    Value* r = nullptr;
    switch (op) {
        case Opcode::vneg: r = b.build_vneg(v); break;
        case Opcode::vnot: r = b.build_vnot(v); break;
        case Opcode::vsqrt: r = b.build_vsqrt(v); break;
        default: break;
    }
    REQUIRE(r != nullptr);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

// fn(v) swaps the first and last lanes and copies lane 1 into the first lane
// of the high half: extracts and inserts on both 128-bit halves.
void build_lane_shuffle(Module& mod, std::string_view name, Type vec) {
    Function* fn = mod.create_function(name, vec, {vec});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* v = b.add_block_param(entry, vec);
    const uint32_t n = vec.vector_lanes();
    Value* first = b.build_vextract_lane(v, 0);
    Value* last = b.build_vextract_lane(v, n - 1);
    Value* one = b.build_vextract_lane(v, 1);
    Value* r = b.build_vinsert_lane(v, last, 0);
    r = b.build_vinsert_lane(r, first, n - 1);
    r = b.build_vinsert_lane(r, one, n / 2);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

bool skip_no_v256_backend() {
    if (!Target::host().is_x64() && !Target::host().is_aarch64()) {
        std::cout << "  [SKIP] no 256-bit vector backend on this host\n";
        return true;
    }
    return false;
}

} // namespace

TEST_CASE("AVX2 ISel regression - integer vbroadcast goes through an xmm") {
    if (skip_no_v256_backend()) return;
    Module mod("avx2_bcast");
    build_broadcast_add(mod, "i32x8_bcast", Type::i32x8(), Type::i32());
    build_broadcast_add(mod, "i64x4_bcast", Type::i64x4(), Type::i64());
    build_broadcast_add(mod, "f32x8_bcast", Type::f32x8(), Type::f32());
    build_broadcast_add(mod, "f64x4_bcast", Type::f64x4(), Type::f64());

    assert_diff(mod, "i32x8_bcast", {RuntimeValue::from_i32x8(1, 2, 30, 4, 5, 6, 7, 8)});
    assert_diff(mod, "i64x4_bcast", {RuntimeValue::from_i64x4(1, 2, 3000000000000LL, 4)});
    assert_diff(mod, "f32x8_bcast", {RuntimeValue::from_f32x8(1, 2, 3.25f, 4, 5, 6, 7, 8)});
    assert_diff(mod, "f64x4_bcast", {RuntimeValue::from_f64x4(1, 2, 3.5, 4)});
}

TEST_CASE("AVX2 ISel regression - lane extract / insert reach the high 128 bits") {
    if (skip_no_v256_backend()) return;
    Module mod("avx2_lanes");
    build_lane_shuffle(mod, "i32x8_lanes", Type::i32x8());
    build_lane_shuffle(mod, "i64x4_lanes", Type::i64x4());
    build_lane_shuffle(mod, "f32x8_lanes", Type::f32x8());
    build_lane_shuffle(mod, "f64x4_lanes", Type::f64x4());
    build_lane_shuffle(mod, "i32x4_lanes", Type::i32x4());
    build_lane_shuffle(mod, "f64x2_lanes", Type::f64x2());

    assert_diff(mod, "i32x8_lanes", {RuntimeValue::from_i32x8(10, 11, 12, 13, 14, 15, 16, 17)});
    assert_diff(mod, "i64x4_lanes", {RuntimeValue::from_i64x4(10, 11, 12, 13)});
    assert_diff(mod, "f32x8_lanes", {RuntimeValue::from_f32x8(1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f)});
    assert_diff(mod, "f64x4_lanes", {RuntimeValue::from_f64x4(1.5, 2.5, 3.5, 4.5)});
    assert_diff(mod, "i32x4_lanes", {RuntimeValue::from_i32x4(10, 11, 12, 13)});
    assert_diff(mod, "f64x2_lanes", {RuntimeValue::from_f64x2(1.5, 2.5)});

    // f64 lane 0 insert keeps lane 1 (movsd merges; the lowering must say so).
    for (Type t : {Type::f64x2(), Type::f64x4()}) {
        const std::string name = "ins0_" + brass::to_string(t);
        Function* fn = mod.create_function(name, t, {t});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* v = b.add_block_param(entry, t);
        b.build_ret(b.build_vinsert_lane(v, b.build_fconst_f64(9.0), 0));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }
    assert_diff(mod, "ins0_f64x2", {RuntimeValue::from_f64x2(1.5, 2.5)});
    assert_diff(mod, "ins0_f64x4", {RuntimeValue::from_f64x4(1.5, 2.5, 3.5, 4.5)});
}

TEST_CASE("AVX2 ISel regression - vneg / vnot / vsqrt act on all 256 bits") {
    if (skip_no_v256_backend()) return;
    Module mod("avx2_unary");
    build_unary(mod, "f32x8_neg", Type::f32x8(), Opcode::vneg);
    build_unary(mod, "f64x4_neg", Type::f64x4(), Opcode::vneg);
    build_unary(mod, "i32x8_neg", Type::i32x8(), Opcode::vneg);
    build_unary(mod, "i64x4_neg", Type::i64x4(), Opcode::vneg);
    build_unary(mod, "i32x8_not", Type::i32x8(), Opcode::vnot);
    build_unary(mod, "i64x4_not", Type::i64x4(), Opcode::vnot);
    build_unary(mod, "f32x8_sqrt", Type::f32x8(), Opcode::vsqrt);
    build_unary(mod, "f64x4_sqrt", Type::f64x4(), Opcode::vsqrt);

    // Distinct values in the upper 128 bits, where the SSE forms did nothing.
    const RuntimeValue f32v = RuntimeValue::from_f32x8(1.0f, -2.0f, 4.0f, 9.0f, 16.0f, -25.0f, 36.0f, 0.25f);
    const RuntimeValue f64v = RuntimeValue::from_f64x4(4.0, -9.0, 16.0, 2.25);
    const RuntimeValue i32v = RuntimeValue::from_i32x8(1, -2, 3, -4, 5, -6, 7, 0x7FFFFFFF);
    const RuntimeValue i64v = RuntimeValue::from_i64x4(1, -2, 300000000000LL, -400000000000LL);

    assert_diff(mod, "f32x8_neg", {f32v});
    assert_diff(mod, "f64x4_neg", {f64v});
    assert_diff(mod, "i32x8_neg", {i32v});
    assert_diff(mod, "i64x4_neg", {i64v});
    assert_diff(mod, "i32x8_not", {i32v});
    assert_diff(mod, "i64x4_not", {i64v});
    const RuntimeValue f32pos = RuntimeValue::from_f32x8(1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 0.25f);
    const RuntimeValue f64pos = RuntimeValue::from_f64x4(4.0, 9.0, 16.0, 2.25);
    assert_diff(mod, "f32x8_sqrt", {f32pos});
    assert_diff(mod, "f64x4_sqrt", {f64pos});
}

TEST_CASE("AVX2 ISel regression - i64x4 vmul multiplies all four lanes") {
    if (skip_no_v256_backend()) return;
    Module mod("avx2_i64x4_mul");
    Function* fn = mod.create_function("i64x4_mul", Type::i64x4(), {Type::i64x4(), Type::i64x4()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, Type::i64x4());
    Value* p1 = b.add_block_param(entry, Type::i64x4());
    b.build_ret(b.build_vmul(p0, p1));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    // Distinct lanes in the high half, products past 32 bits, and a wrap.
    assert_diff(mod, "i64x4_mul", {RuntimeValue::from_i64x4(3, -7, 3000000000LL, 0x4000000000000001LL),
                                   RuntimeValue::from_i64x4(5, 11, -4000000LL, 6)});
}

// fn(c, a, b) = select(c != 0, a, b), on a scalar condition: what an if/else
// diamond on a uniform branch folds to. It went through a GPR with movq and
// cmov, which kept the low eight bytes of the chosen vector and zeroed the
// rest.
TEST_CASE("AVX2 ISel regression - a vector select on a scalar condition keeps every lane") {
    if (skip_no_v256_backend()) return;
    Module mod("avx2_scalar_select");
    for (Type t : {Type::f32x8(), Type::f64x4(), Type::i32x8(), Type::i64x4(), Type::f32x4(), Type::i32x4()}) {
        const std::string name = "sel_" + brass::to_string(t);
        // (A 256-bit function is invoked with 256-bit arguments only: its
        // condition comes in lane 0 of one.)
        const Type ct = t.is_v256() ? Type::i32x8() : Type::i32();
        Function* fn = mod.create_function(name, t, {t, t, ct});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, t);
        Value* y = b.add_block_param(entry, t);
        Value* c = b.add_block_param(entry, ct);
        if (t.is_v256()) c = b.build_vextract_lane(c, 0);
        b.build_ret(b.build_select(b.build_ne(c, b.build_iconst_i32(0)), x, y));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }
    const RuntimeValue f32a = RuntimeValue::from_f32x8(1, 2, 3, 4, 5, 6, 7, 8);
    const RuntimeValue f32b = RuntimeValue::from_f32x8(-1, -2, -3, -4, -5, -6, -7, -8);
    const RuntimeValue f64a = RuntimeValue::from_f64x4(1.5, 2.5, 3.5, 4.5);
    const RuntimeValue f64b = RuntimeValue::from_f64x4(-1.5, -2.5, -3.5, -4.5);
    const RuntimeValue i32a = RuntimeValue::from_i32x8(10, 11, 12, 13, 14, 15, 16, 17);
    const RuntimeValue i32b = RuntimeValue::from_i32x8(-10, -11, -12, -13, -14, -15, -16, -17);
    const RuntimeValue i64a = RuntimeValue::from_i64x4(10, 11, 3000000000000LL, 13);
    const RuntimeValue i64b = RuntimeValue::from_i64x4(-10, -11, -12, -4000000000000LL);
    const RuntimeValue f4a = RuntimeValue::from_f32x4(1, 2, 3, 4);
    const RuntimeValue f4b = RuntimeValue::from_f32x4(-1, -2, -3, -4);
    const RuntimeValue i4a = RuntimeValue::from_i32x4(1, 2, 3, 4);
    const RuntimeValue i4b = RuntimeValue::from_i32x4(-1, -2, -3, -4);
    for (int32_t c : {0, 1, -7}) {
        const RuntimeValue cv = RuntimeValue::from_i32(c);
        const RuntimeValue cw = RuntimeValue::from_i32x8(c, 0, 0, 0, 0, 0, 0, 0);
        assert_diff(mod, "sel_f32x8", {f32a, f32b, cw});
        assert_diff(mod, "sel_f64x4", {f64a, f64b, cw});
        assert_diff(mod, "sel_i32x8", {i32a, i32b, cw});
        assert_diff(mod, "sel_i64x4", {i64a, i64b, cw});
        assert_diff(mod, "sel_f32x4", {f4a, f4b, cv});
        assert_diff(mod, "sel_i32x4", {i4a, i4b, cv});
    }
}

TEST_CASE("AVX2 ISel regression - a 256-bit op with no instruction is a compile error") {
    if (skip_no_v256_backend()) return;
    struct Case { const char* name; Type type; Opcode op; };
    const Case cases[] = {
        {"i32x8_div", Type::i32x8(), Opcode::vdiv},
        {"i64x4_div", Type::i64x4(), Opcode::vdiv},
        {"i64x2_div", Type::i64x2(), Opcode::vdiv},
    };
    for (const Case& c : cases) {
        Module mod(std::string("avx2_gap_") + c.name);
        Function* fn = mod.create_function(c.name, c.type, {c.type, c.type});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* p0 = b.add_block_param(entry, c.type);
        Value* p1 = b.add_block_param(entry, c.type);
        b.build_ret(c.op == Opcode::vmul ? b.build_vmul(p0, p1) : b.build_vdiv(p0, p1));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        bool rejected = false;
        try {
            (void)object::compile_module_to_object(mod, Target::x64_linux());
        } catch (const codegen::UnsupportedOperation& e) {
            rejected = true;
            CHECK_EQ(e.stage(), std::string("x64 isel (vector)"));
        }
        if (!rejected) std::cout << "  not rejected: " << c.name << "\n";
        CHECK(rejected);
    }
}
