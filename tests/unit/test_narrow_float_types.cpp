// First-class f16 (IEEE 754 binary16) and bf16 (bfloat16) scalar MIR types
// and conversion opcodes: conversion bit-exactness, MIR parse/print/verify,
// SCCP constant folding, 4-tier differential execution (Interpreter,
// FastInterpreter, Baseline JIT, Optimizing JIT), GPU target lowering
// (PTX, SPIR-V, MSL) and SMT GPU equivalence verification.

#include "test_framework.hpp"
#include "ptx_test_support.hpp"
#include "spirv_test_support.hpp"

#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/half_float.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_equiv.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::verify;

namespace {

uint32_t f32_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

float bits_f32(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::unique_ptr<Module> parse_and_verify(const std::string& src) {
    DiagnosticReporter pdiag;
    auto mod = parse_module(src, &pdiag);
    if (!mod) std::cerr << "parse failed:\n" << pdiag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n" << src << "\n";
    REQUIRE(ok);
    return mod;
}

std::unique_ptr<Module> build_half_kernel(std::string_view name, bool use_bf16) {
    auto mod = std::make_unique<Module>("half_gpu_mod");
    Type hty = use_bf16 ? Type::bf16() : Type::f16();
    Function* fn = mod->create_function(name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(*mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");

    Value* in_p = b.add_block_param(entry, Type::ptr());
    Value* out_p = b.add_block_param(entry, Type::ptr());
    Value* n_p = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* gid = kb.global_tid_x();
    kb.if_then(b.build_slt(gid, n_p), [&] {
        Value* h = b.build_load_indexed(hty, in_p, gid, 2);
        Value* f = use_bf16 ? b.build_fpext_f32_bf16(h) : b.build_fpext_f32_f16(h);
        Value* two = b.build_fconst_f32(2.0f);
        Value* one = b.build_fconst_f32(1.0f);
        Value* scaled = b.build_add(b.build_mul(f, two), one);
        Value* h_out = use_bf16 ? b.build_fptrunc_bf16_f32(scaled) : b.build_fptrunc_f16_f32(scaled);
        b.build_store_indexed(hty, out_p, gid, 2, h_out);
    });
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Narrow floats - IEEE-754 f16 and bf16 conversion helpers edge cases") {
    CHECK_EQ(f32_to_f16_bits(0.0f), uint16_t{0x0000});
    CHECK_EQ(f32_to_f16_bits(-0.0f), uint16_t{0x8000});
    CHECK_EQ(f32_to_f16_bits(1.0f), uint16_t{0x3C00});
    CHECK_EQ(f32_to_f16_bits(-2.0f), uint16_t{0xC000});
    CHECK_EQ(f32_to_f16_bits(65504.0f), uint16_t{0x7BFF});
    CHECK_EQ(f32_to_f16_bits(65536.0f), uint16_t{0x7C00});
    CHECK_EQ(f32_to_f16_bits(-65536.0f), uint16_t{0xFC00});
    CHECK_EQ(f32_to_f16_bits(1.0f + 0.5f * (1.0f / 1024.0f)), uint16_t{0x3C00});
    CHECK_EQ(f32_to_f16_bits(1.0f + 1.5f * (1.0f / 1024.0f)), uint16_t{0x3C02});
    CHECK_EQ(f32_bits(f16_bits_to_f32(0x0001)), f32_bits(std::ldexp(1.0f, -24)));
    CHECK_EQ(f32_to_f16_bits(std::ldexp(1.0f, -24)), uint16_t{0x0001});
    uint16_t f16_nan = f32_to_f16_bits(bits_f32(0x7F800001u));
    CHECK((f16_nan & 0x7C00u) == 0x7C00u);
    CHECK((f16_nan & 0x0200u) != 0);

    CHECK_EQ(f32_to_bf16_bits(0.0f), uint16_t{0x0000});
    CHECK_EQ(f32_to_bf16_bits(-0.0f), uint16_t{0x8000});
    CHECK_EQ(f32_to_bf16_bits(1.0f), uint16_t{0x3F80});
    CHECK_EQ(f32_to_bf16_bits(-2.0f), uint16_t{0xC000});
    CHECK_EQ(f32_to_bf16_bits(bits_f32(0x3F808000u)), uint16_t{0x3F80});
    CHECK_EQ(f32_to_bf16_bits(bits_f32(0x3F818000u)), uint16_t{0x3F82});
    CHECK_EQ(f32_to_bf16_bits(bits_f32(0x7F800001u)), uint16_t{0x7FC0});
    CHECK_EQ(f32_bits(bf16_bits_to_f32(0x3F81u)), uint32_t{0x3F810000u});
}

TEST_CASE("Narrow floats - MIR print/parse roundtrip, verifier and SCCP") {
    const std::string src = R"(
module @half_rt
func @f(%a: f16, %b: bf16) -> f32 {
entry:
  %fa = fpext.f32.f16 %a
  %fb = fpext.f32.bf16 %b
  %sum = add.f32 %fa, %fb
  %ha = fptrunc.f16.f32 %sum
  %hb = fptrunc.bf16.f32 %sum
  %fa2 = fpext.f32.f16 %ha
  %fb2 = fpext.f32.bf16 %hb
  %out = add.f32 %fa2, %fb2
  ret %out
}
)";
    auto mod = parse_and_verify(src);
    std::string printed = to_string(*mod);
    auto mod2 = parse_and_verify(printed);
    CHECK(mod2->get_function("f") != nullptr);

    const std::string bad_arith = R"(
module @bad1
func @f(%a: f16, %b: f16) -> f16 {
entry:
  %s = add.f16 %a, %b
  ret %s
}
)";
    auto bad_mod = parse_module(bad_arith);
    REQUIRE(bad_mod != nullptr);
    DiagnosticReporter bad_diag;
    CHECK(!verify_module(*bad_mod, &bad_diag));

    Module sccp_mod("sccp_half");
    Function* fn = sccp_mod.create_function("fold", Type::f32(), {});
    Builder b(*fn);
    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);
    Value* c1 = b.build_fconst_f32(1.5f);
    Value* h = b.build_fptrunc_f16_f32(c1);
    Value* fh = b.build_fpext_f32_f16(h);
    Value* bf = b.build_fptrunc_bf16_f32(fh);
    Value* fbf = b.build_fpext_f32_bf16(bf);
    b.build_ret(fbf);

    CHECK(sccp_function(*fn));
    const Instruction* ret = bb->terminator();
    REQUIRE(ret != nullptr && ret->operand_count() == 1);
    const Instruction* def = ret->operand(0)->defining_instruction();
    REQUIRE(def != nullptr);
    CHECK(def->opcode() == Opcode::fconst_f64);
    CHECK(static_cast<float>(def->imm_f64()) == 1.5f);
}

TEST_CASE("Narrow floats - 4-tier differential scalar f16 and bf16 conversions") {
    const std::string src = R"(
module @half_scalar
func @f16_roundtrip(%x: f16) -> f16 {
entry:
  %f = fpext.f32.f16 %x
  %one = fconst.f32 1.25
  %s = add.f32 %f, %one
  %r = fptrunc.f16.f32 %s
  ret %r
}
func @bf16_roundtrip(%x: bf16) -> bf16 {
entry:
  %f = fpext.f32.bf16 %x
  %one = fconst.f32 1.25
  %s = add.f32 %f, %one
  %r = fptrunc.bf16.f32 %s
  ret %r
}
)";
    auto mod = parse_and_verify(src);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    BaselineJitCompiler baseline;
    auto b_f16 = baseline.compile(*mod->get_function("f16_roundtrip"));
    auto b_bf16 = baseline.compile(*mod->get_function("bf16_roundtrip"));
    REQUIRE(b_f16.is_valid());
    REQUIRE(b_bf16.is_valid());

    using HalfFn = uint16_t (*)(uint16_t);
    auto j_f16 = reinterpret_cast<HalfFn>(jit.get_symbol_address("f16_roundtrip"));
    auto j_bf16 = reinterpret_cast<HalfFn>(jit.get_symbol_address("bf16_roundtrip"));
    REQUIRE(j_f16 != nullptr);
    REQUIRE(j_bf16 != nullptr);
    auto bf_f16 = b_f16.get_function_ptr<HalfFn>();
    auto bf_bf16 = b_bf16.get_function_ptr<HalfFn>();

    const std::vector<float> vals = {
        0.0f, -0.0f, 1.0f, -2.5f, 0.125f, 100.0f, -0.0001f, 65000.0f,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
    };
    Interpreter interp;
    interp.set_module(mod.get());
    FastInterpreter fast;

    for (float v : vals) {
        uint16_t hbits = f32_to_f16_bits(v);
        uint16_t hwant = f32_to_f16_bits(f16_bits_to_f32(hbits) + 1.25f);
        RuntimeValue rv_h = RuntimeValue::from_f16_bits(hbits);
        CHECK_EQ(interp.run(*mod->get_function("f16_roundtrip"), {rv_h}).as_f16_bits(), hwant);
        CHECK_EQ(fast.run(*mod->get_function("f16_roundtrip"), {rv_h}).as_f16_bits(), hwant);
        CHECK_EQ(bf_f16(hbits), hwant);
        CHECK_EQ(j_f16(hbits), hwant);

        uint16_t bbits = f32_to_bf16_bits(v);
        uint16_t bwant = f32_to_bf16_bits(bf16_bits_to_f32(bbits) + 1.25f);
        RuntimeValue rv_bf = RuntimeValue::from_bf16_bits(bbits);
        CHECK_EQ(interp.run(*mod->get_function("bf16_roundtrip"), {rv_bf}).as_bf16_bits(), bwant);
        CHECK_EQ(fast.run(*mod->get_function("bf16_roundtrip"), {rv_bf}).as_bf16_bits(), bwant);
        CHECK_EQ(bf_bf16(bbits), bwant);
        CHECK_EQ(j_bf16(bbits), bwant);
    }

    uint16_t bf_snan = 0x7F81u;
    RuntimeValue rv_snan = RuntimeValue::from_bf16_bits(bf_snan);
    uint16_t bf_qwant = f32_to_bf16_bits(bf16_bits_to_f32(bf_snan) + 1.25f);
    CHECK((bf_qwant & 0x7FC0u) == 0x7FC0u);
    CHECK_EQ(interp.run(*mod->get_function("bf16_roundtrip"), {rv_snan}).as_bf16_bits(), bf_qwant);
    CHECK_EQ(fast.run(*mod->get_function("bf16_roundtrip"), {rv_snan}).as_bf16_bits(), bf_qwant);
    CHECK_EQ(bf_bf16(bf_snan), bf_qwant);
    CHECK_EQ(j_bf16(bf_snan), bf_qwant);
}

TEST_CASE("Narrow floats - 4-tier differential load/store buffer loops") {
    const std::string src = R"(
module @half_mem
func @map_f16(%in: ptr, %out: ptr, %n: i32) -> void {
entry:
  %z = iconst.i32 0
  br loop(%z)
loop(%i: i32):
  %cond = slt.i32 %i, %n
  br_if %cond, body, done
body:
  %h = load_indexed.f16 %in, %i, 2, 0
  %f = fpext.f32.f16 %h
  %two = fconst.f32 2.0
  %scaled = mul.f32 %f, %two
  %ho = fptrunc.f16.f32 %scaled
  store_indexed.f16 %out, %i, 2, 0, %ho
  %one = iconst.i32 1
  %ni = add.i32 %i, %one
  br loop(%ni)
done:
  ret
}
func @map_bf16(%in: ptr, %out: ptr, %n: i32) -> void {
entry:
  %z = iconst.i32 0
  br loop(%z)
loop(%i: i32):
  %cond = slt.i32 %i, %n
  br_if %cond, body, done
body:
  %b = load_indexed.bf16 %in, %i, 2, 0
  %f = fpext.f32.bf16 %b
  %two = fconst.f32 2.0
  %scaled = mul.f32 %f, %two
  %bo = fptrunc.bf16.f32 %scaled
  store_indexed.bf16 %out, %i, 2, 0, %bo
  %one = iconst.i32 1
  %ni = add.i32 %i, %one
  br loop(%ni)
done:
  ret
}
)";
    auto mod = parse_and_verify(src);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    BaselineJitCompiler baseline;
    auto b_f16 = baseline.compile(*mod->get_function("map_f16"));
    auto b_bf16 = baseline.compile(*mod->get_function("map_bf16"));
    REQUIRE(b_f16.is_valid());
    REQUIRE(b_bf16.is_valid());

    using MapFn = void (*)(const uint16_t*, uint16_t*, int32_t);
    auto j_f16 = reinterpret_cast<MapFn>(jit.get_symbol_address("map_f16"));
    auto j_bf16 = reinterpret_cast<MapFn>(jit.get_symbol_address("map_bf16"));
    auto bf_f16 = b_f16.get_function_ptr<MapFn>();
    auto bf_bf16 = b_bf16.get_function_ptr<MapFn>();

    const std::vector<float> inputs = {0.0f, -0.0f, 1.0f, -3.5f, 0.0625f, 42.25f, 1024.0f, 32768.0f};
    const int32_t n = static_cast<int32_t>(inputs.size());
    std::vector<uint16_t> in_f16(n), in_bf16(n), want_f16(n), want_bf16(n);
    for (int32_t i = 0; i < n; ++i) {
        in_f16[i] = f32_to_f16_bits(inputs[i]);
        want_f16[i] = f32_to_f16_bits(f16_bits_to_f32(in_f16[i]) * 2.0f);
        in_bf16[i] = f32_to_bf16_bits(inputs[i]);
        want_bf16[i] = f32_to_bf16_bits(bf16_bits_to_f32(in_bf16[i]) * 2.0f);
    }

    Interpreter interp;
    interp.set_module(mod.get());
    FastInterpreter fast;

    std::vector<uint16_t> out_i(n, 0xBEEF), out_fi(n, 0xBEEF), out_b(n, 0xBEEF), out_j(n, 0xBEEF);
    interp.run(*mod->get_function("map_f16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_f16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_i.data())),
        RuntimeValue::from_i32(n)});
    fast.run(*mod->get_function("map_f16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_f16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_fi.data())),
        RuntimeValue::from_i32(n)});
    bf_f16(in_f16.data(), out_b.data(), n);
    j_f16(in_f16.data(), out_j.data(), n);
    for (int32_t i = 0; i < n; ++i) {
        CHECK_EQ(out_i[i], want_f16[i]);
        CHECK_EQ(out_fi[i], want_f16[i]);
        CHECK_EQ(out_b[i], want_f16[i]);
        CHECK_EQ(out_j[i], want_f16[i]);
    }

    std::fill(out_i.begin(), out_i.end(), uint16_t{0xBEEF});
    std::fill(out_fi.begin(), out_fi.end(), uint16_t{0xBEEF});
    std::fill(out_b.begin(), out_b.end(), uint16_t{0xBEEF});
    std::fill(out_j.begin(), out_j.end(), uint16_t{0xBEEF});
    interp.run(*mod->get_function("map_bf16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_bf16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_i.data())),
        RuntimeValue::from_i32(n)});
    fast.run(*mod->get_function("map_bf16"), {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_bf16.data())),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_fi.data())),
        RuntimeValue::from_i32(n)});
    bf_bf16(in_bf16.data(), out_b.data(), n);
    j_bf16(in_bf16.data(), out_j.data(), n);
    for (int32_t i = 0; i < n; ++i) {
        CHECK_EQ(out_i[i], want_bf16[i]);
        CHECK_EQ(out_fi[i], want_bf16[i]);
        CHECK_EQ(out_b[i], want_bf16[i]);
        CHECK_EQ(out_j[i], want_bf16[i]);
    }
}

TEST_CASE("Narrow floats - PTX, SPIR-V, MSL lowering and SMT GPU equivalence") {
    for (bool use_bf16 : {false, true}) {
        auto mod = build_half_kernel(use_bf16 ? "k_bf16" : "k_f16", use_bf16);
        Function* fn = mod->get_function(use_bf16 ? "k_bf16" : "k_f16");
        DiagnosticReporter vdiag;
        bool verified = verify_module(*mod, &vdiag);
        if (!verified) std::cerr << "verify_module failed:\n" << vdiag.format_all() << "\n";
        REQUIRE(verified);

        target::PtxOptions sm70;
        sm70.sm_arch = "sm_70";
        std::string ptx70 = target::PtxTarget::emit_function(*fn, sm70);
        target::PtxOptions sm80;
        sm80.sm_arch = "sm_80";
        std::string ptx80 = target::PtxTarget::emit_function(*fn, sm80);
        if (use_bf16) {
            CHECK(ptx80.find("cvt.rn.bf16.f32") != std::string::npos);
        } else {
            CHECK(ptx70.find("cvt.f32.f16") != std::string::npos);
            CHECK(ptx70.find("cvt.rn.f16.f32") != std::string::npos);
        }
        if (ptxtest::ptxas_available()) {
            CHECK(ptxtest::ptxas_assembles(ptx70, "sm_70"));
            CHECK(ptxtest::ptxas_assembles(ptx80, "sm_80"));
        }

        target::SpirvOptions sopts;
        std::vector<uint32_t> spv_words = target::SpirvTarget::emit_function(*fn, sopts);
        CHECK(!spv_words.empty());
        spvtest::require_valid(spv_words);

        spirv::Module spv_mod(sopts.spirv_version);
        {
            spirv::SpirvISel sisel(spv_mod, sopts);
            sisel.lower(*fn);
        }

        target::MslKernel msl = target::MslTarget::compile(*fn);
        CHECK(msl.source.find("uint16_t") != std::string::npos);
        if (!use_bf16) CHECK(msl.source.find("half") != std::string::npos);

        EquivOptions eq_opts;
        std::string err;
        auto r_ptx = verify_ptx_equivalence(*fn, sm80, &err, eq_opts);
        if (r_ptx.status != EquivStatus::ProvenEquivalent) std::cerr << "MIR vs PTX err: " << err << "\n";
        CHECK(r_ptx.status == EquivStatus::ProvenEquivalent);

        err.clear();
        auto r_spv = verify_spirv_equivalence(*fn, sopts, &err, eq_opts);
        if (r_spv.status != EquivStatus::ProvenEquivalent) std::cerr << "MIR vs SPIR-V err: " << err << "\n";
        CHECK(r_spv.status == EquivStatus::ProvenEquivalent);

        ptx::PtxISel ptx_isel;
        ptx::Function ptx_fn = ptx_isel.lower(*fn);
        err.clear();
        auto r_cross = verify_spirv_ptx_equivalence(*fn, spv_mod, ptx_fn, &err, eq_opts);
        if (r_cross.status != EquivStatus::ProvenEquivalent) std::cerr << "SPIR-V vs PTX err: " << err << "\n";
        CHECK(r_cross.status == EquivStatus::ProvenEquivalent);
    }
}
