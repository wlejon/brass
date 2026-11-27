// Narrow 128-bit integer SIMD types (i8x16 and i16x8):
// - MIR parse/print roundtrip & verifier (including vshuffle rejection)
// - 4-tier CPU differential (Interpreter, FastInterpreter, BaselineJIT, OptJIT)
// - 2-tier extended lane differential (vcmp_*, vselect, vshl/vlshr/vashr, vbitcast, vany)
// - AArch64 NEON ISel + SMT equivalence (verify_arm64_isel_equivalence)
// - GPU lowering: PTX (+ ptxas), SPIR-V (+ spirv-val), MSL rejection

#include "test_framework.hpp"
#include "ptx_test_support.hpp"
#include "spirv_test_support.hpp"

#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/target/target.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::verify;

namespace {

std::unique_ptr<Module> parse_and_verify(const std::string& src) {
    DiagnosticReporter pdiag;
    auto mod = parse_module(src, &pdiag);
    if (!mod) std::cerr << "parse failed:\n" << pdiag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n" << src << "\n";
    REQUIRE(ok);
    return mod;
}

template <typename T>
std::array<T, 16> make_bytes(std::initializer_list<int> vals) {
    std::array<T, 16> out{};
    size_t i = 0;
    for (int v : vals) {
        if (i < out.size()) out[i++] = static_cast<T>(v);
    }
    return out;
}

template <typename T>
std::array<T, 8> make_words(std::initializer_list<int> vals) {
    std::array<T, 8> out{};
    size_t i = 0;
    for (int v : vals) {
        if (i < out.size()) out[i++] = static_cast<T>(v);
    }
    return out;
}

void check_arm64_equiv(Function& fn, const char* label) {
    DiagnosticReporter vdiag;
    Module* m = fn.parent();
    REQUIRE(m != nullptr);
    REQUIRE(verify_module(*m, &vdiag));
    aarch64::AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    auto lir = isel.lower(fn);
    REQUIRE(lir != nullptr);
    EquivOptions opts;
    opts.max_sat_conflicts = 50000;
    auto res = verify_arm64_isel_equivalence(fn, *lir, opts);
    if (res.status != EquivStatus::ProvenEquivalent) {
        std::cerr << label << " arm64 equiv failed: " << res.diagnostic << "\n";
    }
    CHECK(res.status == EquivStatus::ProvenEquivalent);
}

} // namespace

TEST_CASE("Narrow int SIMD - MIR parse/print roundtrip and verifier") {
    const std::string src = R"(
module @narrow_int_simd_roundtrip
func @test_i8x16(%p: ptr, %s: i8) -> i8x16 {
entry:
  %a = vload.i8x16 %p, 0
  %b = vbroadcast.i8x16 %s
  %sum = vadd.i8x16 %a, %b
  %sub = vsub.i8x16 %sum, %b
  %prod = vmul.i8x16 %sub, %a
  %mn = vmin.i8x16 %prod, %b
  %mx = vmax.i8x16 %mn, %a
  %neg = vneg.i8x16 %mx
  %ext = vextract_lane %neg, 5
  %ins = vinsert_lane %neg, %ext, 7
  %cmp = vcmp_lt %ins, %a
  %any = vany %cmp
  %z = vzero.i8x16
  %sel = vselect %cmp, %ins, %z
  %shl = vshl %sel, %b
  %lshr = vlshr %shl, %b
  %ashr = vashr %lshr, %b
  %b32 = vbitcast.i32x4 %ashr
  %back = vbitcast.i8x16 %b32
  vstore.i8x16 %p, 16, %back
  ret %back
}
func @test_i16x8(%p: ptr, %s: i16) -> i16x8 {
entry:
  %a = vload.i16x8 %p, 0
  %b = vbroadcast.i16x8 %s
  %sum = vadd.i16x8 %a, %b
  %mn = vmin.i16x8 %sum, %b
  %neg = vneg.i16x8 %mn
  %ext = vextract_lane %neg, 3
  %ins = vinsert_lane %neg, %ext, 4
  %cmp = vcmp_gt %ins, %a
  %sel = vselect %cmp, %ins, %a
  %sh = vashr %sel, %b
  vstore.i16x8 %p, 16, %sh
  ret %sh
}
)";
    auto mod1 = parse_and_verify(src);
    std::string printed = to_string(*mod1);
    auto mod2 = parse_and_verify(printed);
    REQUIRE(mod2->get_function("test_i8x16") != nullptr);
    REQUIRE(mod2->get_function("test_i16x8") != nullptr);

    // Negative verifier checks: vshuffle rejects i8x16 and i16x8.
    for (const std::string& bad_ty : {std::string("i8x16"), std::string("i16x8")}) {
        const std::string bad_src = "module @bad\nfunc @bad_shuf(%a: " + bad_ty + ", %b: " + bad_ty +
                                    ") -> " + bad_ty + " {\nentry:\n  %s = vshuffle %a, %b, 0xE4\n  ret %s\n}\n";
        DiagnosticReporter pdiag;
        auto bad_mod = parse_module(bad_src, &pdiag);
        REQUIRE(bad_mod != nullptr);
        DiagnosticReporter vdiag;
        CHECK(!verify_module(*bad_mod, &vdiag));
    }
}

TEST_CASE("Narrow int SIMD - 4-tier differential for i8x16 and i16x8 core ops") {
    const std::string src = R"(
module @narrow_int_4tier
func @core_i8x16(%pa: ptr, %pb: ptr, %pout: ptr, %sv: i32) -> i32 {
entry:
  %a = vload.i8x16 %pa, 0
  %b = vload.i8x16 %pb, 0
  %st = trunc.i8 %sv
  %bc = vbroadcast.i8x16 %st
  %v0 = vadd.i8x16 %a, %b
  %v1 = vsub.i8x16 %v0, %bc
  %v2 = vmul.i8x16 %v1, %a
  %v3 = vmin.i8x16 %v2, %b
  %v4 = vmax.i8x16 %v3, %a
  %v5 = vneg.i8x16 %v4
  %v6 = vand.i8x16 %v5, %b
  %v7 = vor.i8x16 %v6, %a
  %v8 = vxor.i8x16 %v7, %bc
  %v9 = vnot.i8x16 %v8
  %lane = vextract_lane %v9, 13
  %z = vzero.i8x16
  %ins = vinsert_lane %v9, %lane, 2
  %fin = vadd.i8x16 %ins, %z
  vstore.i8x16 %pout, 0, %fin
  %lz = vextract_lane %fin, 0
  %l64 = zext.i64 %lz
  %res = trunc.i32 %l64
  ret %res
}

func @core_i16x8(%pa: ptr, %pb: ptr, %pout: ptr, %sv: i32) -> i32 {
entry:
  %a = vload.i16x8 %pa, 0
  %b = vload.i16x8 %pb, 0
  %ap = alloca 4, 4
  store.i32 %ap, 0, %sv
  %st = load.i16 %ap, 0
  %bc = vbroadcast.i16x8 %st
  %v0 = vadd.i16x8 %a, %b
  %v1 = vsub.i16x8 %v0, %bc
  %v2 = vmul.i16x8 %v1, %a
  %v3 = vmin.i16x8 %v2, %b
  %v4 = vmax.i16x8 %v3, %a
  %v5 = vneg.i16x8 %v4
  %v6 = vand.i16x8 %v5, %b
  %v7 = vor.i16x8 %v6, %a
  %v8 = vxor.i16x8 %v7, %bc
  %v9 = vnot.i16x8 %v8
  %lane = vextract_lane %v9, 6
  %z = vzero.i16x8
  %ins = vinsert_lane %v9, %lane, 1
  %fin = vadd.i16x8 %ins, %z
  vstore.i16x8 %pout, 0, %fin
  %w0 = load.i32 %pout, 0
  %m16 = iconst.i32 65535
  %res = and.i32 %w0, %m16
  ret %res
}
)";
    const std::string div_src = R"(
module @narrow_int_vdiv
func @vdiv_i8x16(%pa: ptr, %pb: ptr, %pout: ptr) -> void {
entry:
  %a = vload.i8x16 %pa, 0
  %b = vload.i8x16 %pb, 0
  %q = vdiv.i8x16 %a, %b
  vstore.i8x16 %pout, 0, %q
  ret
}

func @vdiv_i16x8(%pa: ptr, %pb: ptr, %pout: ptr) -> void {
entry:
  %a = vload.i16x8 %pa, 0
  %b = vload.i16x8 %pb, 0
  %q = vdiv.i16x8 %a, %b
  vstore.i16x8 %pout, 0, %q
  ret
}
)";
    auto mod = parse_and_verify(src);
    auto div_mod = parse_and_verify(div_src);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    BaselineJitCompiler baseline;
    auto b8 = baseline.compile(*mod->get_function("core_i8x16"));
    auto b16 = baseline.compile(*mod->get_function("core_i16x8"));
    auto bd8 = baseline.compile(*div_mod->get_function("vdiv_i8x16"));
    auto bd16 = baseline.compile(*div_mod->get_function("vdiv_i16x8"));
    REQUIRE(b8.is_valid());
    REQUIRE(b16.is_valid());
    REQUIRE(bd8.is_valid());
    REQUIRE(bd16.is_valid());

    using CoreFn = int32_t (*)(const void*, const void*, void*, int32_t);
    using DivFn = void (*)(const void*, const void*, void*);
    auto j8 = reinterpret_cast<CoreFn>(jit.get_symbol_address("core_i8x16"));
    auto j16 = reinterpret_cast<CoreFn>(jit.get_symbol_address("core_i16x8"));

    Interpreter interp;
    interp.set_module(mod.get());
    FastInterpreter fast;

    // 16 bytes with wraparound (-128, 127), negatives, zeros
    const auto a8 = make_bytes<int8_t>({-128, -1, 0, 1, 100, -50, 7, -12, 42, -99, 127, -128, 3, -4, 55, -80});
    const auto b8v = make_bytes<int8_t>({-1, 2, -128, 127, 3, -4, -9, 12, -17, 100, 1, -1, 8, -3, 10, 77});
    std::array<int8_t, 16> out8_i{}, out8_fi{}, out8_b{}, out8_j{};
    const int32_t sv8 = -29;

    int32_t r8_i = interp.run(*mod->get_function("core_i8x16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a8.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b8v.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out8_i.data())),
        RuntimeValue::from_i32(sv8)}).as_i32();
    int32_t r8_fi = fast.run(*mod->get_function("core_i8x16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a8.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b8v.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out8_fi.data())),
        RuntimeValue::from_i32(sv8)}).as_i32();
    int32_t r8_b = b8.get_function_ptr<CoreFn>()(a8.data(), b8v.data(), out8_b.data(), sv8);
    int32_t r8_j = j8(a8.data(), b8v.data(), out8_j.data(), sv8);

    CHECK_EQ(r8_fi, r8_i);
    CHECK_EQ(r8_b, r8_i);
    CHECK_EQ(r8_j, r8_i);
    for (size_t i = 0; i < 16; ++i) {
        CHECK_EQ(out8_fi[i], out8_i[i]);
        CHECK_EQ(out8_b[i], out8_i[i]);
        CHECK_EQ(out8_j[i], out8_i[i]);
    }

    // 8 halfwords with wraparound (-32768, 32767)
    const auto a16 = make_words<int16_t>({-32768, -1, 0, 1000, -12345, 32767, -200, 4096});
    const auto b16v = make_words<int16_t>({-1, 40, -32768, -3, 777, -2, 500, -1024});
    std::array<int16_t, 8> out16_i{}, out16_fi{}, out16_b{}, out16_j{};
    const int32_t sv16 = 12345;

    int32_t r16_i = interp.run(*mod->get_function("core_i16x8"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b16v.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out16_i.data())),
        RuntimeValue::from_i32(sv16)}).as_i32();
    int32_t r16_fi = fast.run(*mod->get_function("core_i16x8"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b16v.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out16_fi.data())),
        RuntimeValue::from_i32(sv16)}).as_i32();
    int32_t r16_b = b16.get_function_ptr<CoreFn>()(a16.data(), b16v.data(), out16_b.data(), sv16);
    int32_t r16_j = j16(a16.data(), b16v.data(), out16_j.data(), sv16);

    CHECK_EQ(r16_fi, r16_i);
    CHECK_EQ(r16_b, r16_i);
    CHECK_EQ(r16_j, r16_i);
    for (size_t i = 0; i < 8; ++i) {
        CHECK_EQ(out16_fi[i], out16_i[i]);
        CHECK_EQ(out16_b[i], out16_i[i]);
        CHECK_EQ(out16_j[i], out16_i[i]);
    }

    // Integer vdiv edge cases: divisor 0 -> 0, MIN / -1 -> MIN
    const auto d8a = make_bytes<int8_t>({-128, 42, -99, 0, 127, -128, -7, 100, -1, 50, -50, 12, -12, 9, -8, 127});
    const auto d8b = make_bytes<int8_t>({-1, 0, 0, 0, -1, 2, 3, -4, -1, 7, 10, -5, 4, -3, 0, 127});
    std::array<int8_t, 16> dv8_i{}, dv8_fi{}, dv8_b{};
    Interpreter div_interp;
    div_interp.set_module(div_mod.get());
    div_interp.run(*div_mod->get_function("vdiv_i8x16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d8a.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d8b.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(dv8_i.data()))});
    fast.run(*div_mod->get_function("vdiv_i8x16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d8a.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d8b.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(dv8_fi.data()))});
    bd8.get_function_ptr<DivFn>()(d8a.data(), d8b.data(), dv8_b.data());
    CHECK_EQ(dv8_i[0], static_cast<int8_t>(-128));
    CHECK_EQ(dv8_i[1], static_cast<int8_t>(0));
    for (size_t i = 0; i < 16; ++i) {
        CHECK_EQ(dv8_fi[i], dv8_i[i]);
        CHECK_EQ(dv8_b[i], dv8_i[i]);
    }

    const auto d16a = make_words<int16_t>({-32768, 1234, 0, -500, 32767, -32768, -99, 700});
    const auto d16b = make_words<int16_t>({-1, 0, 0, 7, -1, 2, -11, -3});
    std::array<int16_t, 8> dv16_i{}, dv16_fi{}, dv16_b{};
    div_interp.run(*div_mod->get_function("vdiv_i16x8"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d16a.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d16b.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(dv16_i.data()))});
    fast.run(*div_mod->get_function("vdiv_i16x8"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d16a.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(d16b.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(dv16_fi.data()))});
    bd16.get_function_ptr<DivFn>()(d16a.data(), d16b.data(), dv16_b.data());
    CHECK_EQ(dv16_i[0], static_cast<int16_t>(-32768));
    CHECK_EQ(dv16_i[1], static_cast<int16_t>(0));
    for (size_t i = 0; i < 8; ++i) {
        CHECK_EQ(dv16_fi[i], dv16_i[i]);
        CHECK_EQ(dv16_b[i], dv16_i[i]);
    }
}

TEST_CASE("Narrow int SIMD - 2-tier differential for compare, select, shift, bitcast, vany") {
    const std::string src = R"(
module @narrow_int_lanes
func @lanes_i8x16(%pa: ptr, %pb: ptr, %ps: ptr, %pout: ptr) -> i32 {
entry:
  %a = vload.i8x16 %pa, 0
  %b = vload.i8x16 %pb, 0
  %sh = vload.i8x16 %ps, 0
  %ceq = vcmp_eq %a, %b
  %cne = vcmp_ne %a, %b
  %clt = vcmp_lt %a, %b
  %cle = vcmp_le %a, %b
  %cgt = vcmp_gt %a, %b
  %cge = vcmp_ge %a, %b
  %s1 = vselect %ceq, %a, %b
  %s2 = vselect %clt, %s1, %a
  %s3 = vselect %cgt, %s2, %b
  %s4 = vselect %cle, %s3, %a
  %s5 = vselect %cge, %s4, %b
  %s6 = vselect %cne, %s5, %a
  %c2 = iconst.i32 2
  %c2b = trunc.i8 %c2
  %bc2 = vbroadcast.i8x16 %c2b
  %sl_i = vshl %s6, %bc2
  %lr_i = vlshr %s6, %bc2
  %ar_i = vashr %s6, %bc2
  %sl_v = vshl %sl_i, %sh
  %lr_v = vlshr %lr_i, %sh
  %ar_v = vashr %ar_i, %sh
  %mix = vadd.i8x16 %sl_v, %lr_v
  %fin = vxor.i8x16 %mix, %ar_v
  %as32 = vbitcast.i32x4 %fin
  %as16 = vbitcast.i16x8 %as32
  %back = vbitcast.i8x16 %as16
  vstore.i8x16 %pout, 0, %back
  %has_lt = vany %clt
  %has_neg = vany %back
  %r = add.i32 %has_lt, %has_neg
  ret %r
}

func @lanes_i16x8(%pa: ptr, %pb: ptr, %ps: ptr, %pout: ptr) -> i32 {
entry:
  %a = vload.i16x8 %pa, 0
  %b = vload.i16x8 %pb, 0
  %sh = vload.i16x8 %ps, 0
  %ceq = vcmp_eq %a, %b
  %cne = vcmp_ne %a, %b
  %clt = vcmp_lt %a, %b
  %cle = vcmp_le %a, %b
  %cgt = vcmp_gt %a, %b
  %cge = vcmp_ge %a, %b
  %s1 = vselect %ceq, %a, %b
  %s2 = vselect %clt, %s1, %a
  %s3 = vselect %cgt, %s2, %b
  %s4 = vselect %cle, %s3, %a
  %s5 = vselect %cge, %s4, %b
  %s6 = vselect %cne, %s5, %a
  %c5 = iconst.i32 5
  %ap = alloca 4, 4
  store.i32 %ap, 0, %c5
  %c5h = load.i16 %ap, 0
  %bc5 = vbroadcast.i16x8 %c5h
  %sl_i = vshl %s6, %bc5
  %lr_i = vlshr %s6, %bc5
  %ar_i = vashr %s6, %bc5
  %sl_v = vshl %sl_i, %sh
  %lr_v = vlshr %lr_i, %sh
  %ar_v = vashr %ar_i, %sh
  %mix = vadd.i16x8 %sl_v, %lr_v
  %fin = vxor.i16x8 %mix, %ar_v
  %as32 = vbitcast.i32x4 %fin
  %back = vbitcast.i16x8 %as32
  vstore.i16x8 %pout, 0, %back
  %has_lt = vany %clt
  %has_neg = vany %back
  %r = add.i32 %has_lt, %has_neg
  ret %r
}

func @vany_pos_edge_i8(%pa: ptr) -> i32 {
entry:
  %a = vload.i8x16 %pa, 0
  %r = vany %a
  ret %r
}

func @vany_pos_edge_i16(%pa: ptr) -> i32 {
entry:
  %a = vload.i16x8 %pa, 0
  %r = vany %a
  ret %r
}
)";
    auto mod = parse_and_verify(src);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));

    using LanesFn = int32_t (*)(const void*, const void*, const void*, void*);
    using AnyFn = int32_t (*)(const void*);
    auto j8 = reinterpret_cast<LanesFn>(jit.get_symbol_address("lanes_i8x16"));
    auto j16 = reinterpret_cast<LanesFn>(jit.get_symbol_address("lanes_i16x8"));
    auto jany8 = reinterpret_cast<AnyFn>(jit.get_symbol_address("vany_pos_edge_i8"));
    auto jany16 = reinterpret_cast<AnyFn>(jit.get_symbol_address("vany_pos_edge_i16"));

    Interpreter interp;
    interp.set_module(mod.get());

    const auto a8 = make_bytes<int8_t>({-128, -5, 0, 5, 127, -12, 42, -99, 100, -1, 7, -8, 12, -13, 64, -64});
    const auto b8 = make_bytes<int8_t>({-128, 3, -1, 5, -128, 12, -42, 99, 100, 1, 7, -7, 11, -13, -1, 64});
    const auto s8 = make_bytes<uint8_t>({0, 1, 7, 8, 255, 3, 4, 9, 0, 2, 6, 8, 128, 5, 15, 7});
    std::array<int8_t, 16> out8_i{}, out8_j{};
    int32_t r8_i = interp.run(*mod->get_function("lanes_i8x16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a8.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b8.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(s8.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out8_i.data()))}).as_i32();
    int32_t r8_j = j8(a8.data(), b8.data(), s8.data(), out8_j.data());
    CHECK_EQ(r8_j, r8_i);
    for (size_t i = 0; i < 16; ++i) CHECK_EQ(out8_j[i], out8_i[i]);

    const auto a16 = make_words<int16_t>({-32768, -5, 0, 32767, -1234, 555, -2, 1000});
    const auto b16 = make_words<int16_t>({-32768, 3, -1, -32768, -1234, 554, 1, 1000});
    const auto s16 = make_words<uint16_t>({0, 1, 15, 16, 255, 7, 65535, 3});
    std::array<int16_t, 8> out16_i{}, out16_j{};
    int32_t r16_i = interp.run(*mod->get_function("lanes_i16x8"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(s16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out16_i.data()))}).as_i32();
    int32_t r16_j = j16(a16.data(), b16.data(), s16.data(), out16_j.data());
    CHECK_EQ(r16_j, r16_i);
    for (size_t i = 0; i < 8; ++i) CHECK_EQ(out16_j[i], out16_i[i]);

    // Check vany on all-non-negative vs single high-bit lane at lane 0, 7, 15
    std::array<uint8_t, 16> p8{};
    p8.fill(0x7F);
    CHECK_EQ(jany8(p8.data()), 0);
    for (size_t idx : {size_t{0}, size_t{7}, size_t{15}}) {
        std::array<uint8_t, 16> m8{};
        m8[idx] = 0x80;
        CHECK_EQ(jany8(m8.data()), 1);
    }
    std::array<uint16_t, 8> p16{};
    p16.fill(0x7FFF);
    CHECK_EQ(jany16(p16.data()), 0);
    for (size_t idx : {size_t{0}, size_t{3}, size_t{7}}) {
        std::array<uint16_t, 8> m16{};
        m16[idx] = 0x8000;
        CHECK_EQ(jany16(m16.data()), 1);
    }
}

TEST_CASE("Narrow int SIMD - AArch64 NEON ISel + SMT equivalence") {
    Module mod("neon_narrow_int_equiv");
    Builder b(mod);

    // 1. i8x16 core ops + insert/extract/broadcast/bitwise
    {
        Function* fn = mod.create_function("neon_i8x16_alu", Type::i8x16(), {Type::i8x16(), Type::i8x16(), Type::i8()});
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        Value* va = b.add_block_param(e, Type::i8x16());
        Value* vb = b.add_block_param(e, Type::i8x16());
        Value* sc = b.add_block_param(e, Type::i8());
        b.position_at_end(e);
        Value* bc = b.build_vbroadcast(Type::i8x16(), sc);
        Value* s = b.build_vsub(b.build_vadd(va, vb), bc);
        Value* m = b.build_vmul(s, va);
        Value* mm = b.build_vmax(b.build_vmin(m, vb), va);
        Value* neg = b.build_vneg(mm);
        Value* bw = b.build_vnot(b.build_vxor(b.build_vor(b.build_vand(neg, vb), va), bc));
        Value* ext = b.build_vextract_lane(bw, 11);
        Value* ins = b.build_vinsert_lane(bw, ext, 0);
        b.build_ret(ins);
        fn->rebuild_cfg_predecessors();
        check_arm64_equiv(*fn, "neon_i8x16_alu");
    }

    // 2. i8x16 cmp + select + shifts + bitcast + vany
    {
        Function* fn = mod.create_function("neon_i8x16_lanes", Type::i32(), {Type::i8x16(), Type::i8x16(), Type::i8x16()});
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        Value* va = b.add_block_param(e, Type::i8x16());
        Value* vb = b.add_block_param(e, Type::i8x16());
        Value* vs = b.add_block_param(e, Type::i8x16());
        b.position_at_end(e);
        Value* c0 = b.build_vcmp(Opcode::vcmp_lt, va, vb);
        Value* c1 = b.build_vcmp(Opcode::vcmp_eq, va, vb);
        Value* sel = b.build_vselect(c0, va, vb);
        Value* imm_sh = b.build_vbroadcast(Type::i8x16(), b.build_trunc_i8(b.build_iconst_i32(3)));
        Value* sh1 = b.build_vshl(sel, imm_sh);
        Value* sh2 = b.build_vlshr(sh1, vs);
        Value* sh3 = b.build_vashr(sh2, vs);
        Value* back = b.build_vbitcast(Type::i8x16(), b.build_vbitcast(Type::i32x4(), sh3));
        Value* a0 = b.build_vany(back);
        Value* a1 = b.build_vany(c1);
        b.build_ret(b.build_add(a0, a1));
        fn->rebuild_cfg_predecessors();
        check_arm64_equiv(*fn, "neon_i8x16_lanes");
    }

    // 3. i16x8 core ops + cmp + select + shifts + bitcast + vany
    {
        Function* fn = mod.create_function("neon_i16x8_all", Type::i32(), {Type::i16x8(), Type::i16x8(), Type::i16x8(), Type::i16()});
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        Value* va = b.add_block_param(e, Type::i16x8());
        Value* vb = b.add_block_param(e, Type::i16x8());
        Value* vs = b.add_block_param(e, Type::i16x8());
        Value* sc = b.add_block_param(e, Type::i16());
        b.position_at_end(e);
        Value* bc = b.build_vbroadcast(Type::i16x8(), sc);
        Value* s = b.build_vsub(b.build_vadd(va, vb), bc);
        Value* mm = b.build_vmax(b.build_vmin(b.build_vmul(s, va), vb), b.build_vneg(vb));
        Value* ext = b.build_vextract_lane(mm, 5);
        Value* ins = b.build_vinsert_lane(mm, ext, 2);
        Value* c0 = b.build_vcmp(Opcode::vcmp_gt, ins, va);
        Value* sel = b.build_vselect(c0, ins, vb);
        Value* sh = b.build_vashr(b.build_vlshr(b.build_vshl(sel, bc), vs), vs);
        Value* back = b.build_vbitcast(Type::i16x8(), b.build_vbitcast(Type::i32x4(), sh));
        b.build_ret(b.build_vany(back));
        fn->rebuild_cfg_predecessors();
        check_arm64_equiv(*fn, "neon_i16x8_all");
    }

    // 4. i16x8 shifts + vany directly verified via LIR & MIR equivalence
    {
        auto m16 = parse_and_verify(R"(
module @neon_i16x8_var_sh
func @sh16_var(%v: i16x8, %s: i16x8) -> i16x8 {
entry:
  %a = vshl %v, %s
  %b = vlshr %a, %s
  %c = vashr %b, %s
  ret %c
}
)");
        check_arm64_equiv(*m16->get_function("sh16_var"), "neon_i16x8_var_sh");
    }
}

TEST_CASE("Narrow int SIMD - PTX, SPIR-V, and MSL rejection") {
    const std::string ptx_src = R"(
module @narrow_ptx
func @kern_i8x16(%pa: ptr, %pb: ptr, %po: ptr, %sv: i32) -> void {
entry:
  %a = vload.i8x16 %pa, 0
  %b = vload.i8x16 %pb, 0
  %st = trunc.i8 %sv
  %bc = vbroadcast.i8x16 %st
  %v0 = vadd.i8x16 %a, %b
  %v1 = vsub.i8x16 %v0, %bc
  %v2 = vmul.i8x16 %v1, %a
  %v3 = vmin.i8x16 %v2, %b
  %v4 = vmax.i8x16 %v3, %a
  %v5 = vneg.i8x16 %v4
  %v6 = vand.i8x16 %v5, %b
  %v7 = vor.i8x16 %v6, %a
  %v8 = vxor.i8x16 %v7, %bc
  %v9 = vnot.i8x16 %v8
  %ext = vextract_lane %v9, 9
  %ins = vinsert_lane %v9, %ext, 3
  vstore.i8x16 %po, 0, %ins
  ret
}
func @kern_i16x8(%pa: ptr, %pb: ptr, %po: ptr) -> void {
entry:
  %a = vload.i16x8 %pa, 0
  %b = vload.i16x8 %pb, 0
  %st = vextract_lane %a, 2
  %bc = vbroadcast.i16x8 %st
  %v0 = vadd.i16x8 %a, %b
  %v1 = vsub.i16x8 %v0, %bc
  %v2 = vmul.i16x8 %v1, %a
  %v3 = vmin.i16x8 %v2, %b
  %v4 = vmax.i16x8 %v3, %a
  %v5 = vneg.i16x8 %v4
  %v6 = vand.i16x8 %v5, %b
  %v7 = vor.i16x8 %v6, %a
  %v8 = vxor.i16x8 %v7, %bc
  %v9 = vnot.i16x8 %v8
  %ext = vextract_lane %v9, 4
  %ins = vinsert_lane %v9, %ext, 1
  vstore.i16x8 %po, 0, %ins
  ret
}
)";
    auto ptx_mod = parse_and_verify(ptx_src);
    target::PtxOptions popts;
    popts.sm_arch = "sm_80";
    for (const char* name : {"kern_i8x16", "kern_i16x8"}) {
        Function* fn = ptx_mod->get_function(name);
        REQUIRE(fn != nullptr);
        std::string ptx = target::PtxTarget::emit_function(*fn, popts);
        CHECK(!ptx.empty());
        if (ptxtest::ptxas_available()) {
            CHECK(ptxtest::ptxas_assembles(ptx, "sm_80"));
        }
        // MSL must cleanly reject i8x16 / i16x8 rather than emitting invalid MSL.
        bool msl_threw = false;
        try {
            (void)target::MslTarget::compile(*fn);
        } catch (const std::exception&) {
            msl_threw = true;
        }
        CHECK(msl_threw);
    }

    const std::string spv_src = R"(
module @narrow_spv
func @spv_i8x16(%pa: ptr, %pb: ptr, %po: ptr) -> void {
entry:
  %a = vload.i8x16 %pa, 0
  %b = vload.i8x16 %pb, 0
  %z = vzero.i8x16
  %v0 = vadd.i8x16 %a, %b
  %v1 = vsub.i8x16 %v0, %z
  %v2 = vmul.i8x16 %v1, %a
  %v3 = vmin.i8x16 %v2, %b
  %v4 = vmax.i8x16 %v3, %a
  %v5 = vneg.i8x16 %v4
  %v6 = vand.i8x16 %v5, %b
  %v7 = vor.i8x16 %v6, %a
  %v8 = vxor.i8x16 %v7, %b
  %v9 = vnot.i8x16 %v8
  %ext = vextract_lane %v9, 10
  %ins = vinsert_lane %v9, %ext, 5
  vstore.i8x16 %po, 0, %ins
  ret
}
func @spv_i16x8(%pa: ptr, %pb: ptr, %po: ptr) -> void {
entry:
  %a = vload.i16x8 %pa, 0
  %b = vload.i16x8 %pb, 0
  %z = vzero.i16x8
  %v0 = vadd.i16x8 %a, %b
  %v1 = vsub.i16x8 %v0, %z
  %v2 = vmul.i16x8 %v1, %a
  %v3 = vmin.i16x8 %v2, %b
  %v4 = vmax.i16x8 %v3, %a
  %v5 = vneg.i16x8 %v4
  %v6 = vand.i16x8 %v5, %b
  %v7 = vor.i16x8 %v6, %a
  %v8 = vxor.i16x8 %v7, %b
  %v9 = vnot.i16x8 %v8
  %ext = vextract_lane %v9, 6
  %ins = vinsert_lane %v9, %ext, 2
  vstore.i16x8 %po, 0, %ins
  ret
}
)";
    auto spv_mod = parse_and_verify(spv_src);
    target::SpirvOptions sopts;
    for (const char* name : {"spv_i8x16", "spv_i16x8"}) {
        Function* fn = spv_mod->get_function(name);
        REQUIRE(fn != nullptr);
        std::vector<uint32_t> spv_words = target::SpirvTarget::emit_function(*fn, sopts);
        CHECK(!spv_words.empty());
        spvtest::require_valid(spv_words);
    }
}
