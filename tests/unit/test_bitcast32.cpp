// The 32-bit bitcast pair (bitcast.i32 <f32> / bitcast.f32 <i32>) on the CPU
// side: text round trip and the short spellings, verifier typing, the SCCP
// fold, the MIR interpreter, the bytecode VM, the baseline tier and the
// optimizing tier through KernelJit. The GPU backends are covered in
// test_ptx_isel.cpp, test_spirv_isel.cpp and test_spirv_execution.cpp.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace brass;

namespace {

uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float float_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

const std::vector<float> kFloats = {0.0f, -0.0f, 1.0f, -2.5f, 3.14159265f, 1e-40f, 65504.0f,
                                    std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::quiet_NaN()};
const std::vector<uint32_t> kWords = {0u, 1u, 0x3f800000u, 0xbfc00000u, 0x7f800000u, 0x80000000u,
                                      0x7fffffffu, 0xffffffffu, 0x00012345u};

std::unique_ptr<Module> parse_verified(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

// f32 -> i32 -> (flip the sign bit) -> f32 -> i32: the bits of -x, which
// every tier must produce bit-exactly, NaN and denormals included.
const char* kFlipSrc =
    "module @m\n"
    "func @flip(%0: f32) -> i32 {\n"
    "entry:\n"
    "  %1 = bitcast.i32.f32 %0\n"
    "  %2 = iconst.i32 -2147483648\n"
    "  %3 = xor.i32 %1, %2\n"
    "  %4 = bitcast.f32.i32 %3\n"
    "  %5 = bitcast.i32.f32 %4\n"
    "  ret %5\n"
    "}\n"
    "func @widen(%0: i32) -> f32 {\n"
    "entry:\n"
    "  %1 = bitcast.f32.i32 %0\n"
    "  ret %1\n"
    "}\n";

} // namespace

TEST_CASE("Bitcast32 - prints, parses and round-trips, and the short spellings parse") {
    auto mod = parse_verified(kFlipSrc);
    std::string printed = to_string(*mod);
    CHECK(printed.find("bitcast.i32.f32 %0") != std::string::npos);
    CHECK(printed.find("bitcast.f32.i32 %3") != std::string::npos);
    auto again = parse_verified(printed);
    CHECK_EQ(to_string(*again), printed);

    auto short_forms = parse_verified(
        "module @m\nfunc @s(%0: f32) -> f32 {\nentry:\n  %1 = bitcast.i32 %0\n  %2 = bitcast.f32 %1\n  ret %2\n}\n");
    const Function* s = short_forms->get_function("s");
    REQUIRE(s != nullptr);
    const BasicBlock* bb = s->blocks().front();
    auto it = bb->begin();
    CHECK((*it)->opcode() == Opcode::bitcast_i32_f32);
    CHECK((*it)->type() == Type::i32());
    ++it;
    CHECK((*it)->opcode() == Opcode::bitcast_f32_i32);
    CHECK((*it)->type() == Type::f32());
}

TEST_CASE("Bitcast32 - the verifier wants the same-width operand") {
    const char* bad[] = {
        "module @m\nfunc @b(%0: i32) -> i32 {\nentry:\n  %1 = bitcast.i32.f32 %0\n  ret %1\n}\n",
        "module @m\nfunc @b(%0: f64) -> i32 {\nentry:\n  %1 = bitcast.i32.f32 %0\n  ret %1\n}\n",
        "module @m\nfunc @b(%0: f32) -> f32 {\nentry:\n  %1 = bitcast.f32.i32 %0\n  ret %1\n}\n",
        "module @m\nfunc @b(%0: i64) -> f32 {\nentry:\n  %1 = bitcast.f32.i32 %0\n  ret %1\n}\n",
    };
    for (const char* src : bad) {
        DiagnosticReporter diag;
        auto mod = parse_module(src, &diag);
        REQUIRE(mod != nullptr);
        DiagnosticReporter vdiag;
        CHECK(!verify_module(*mod, &vdiag));
        CHECK(vdiag.format_all().find("operand") != std::string::npos);
    }
}

TEST_CASE("Bitcast32 - SCCP folds constant bitcasts and leaves NaN patterns to run") {
    Module mod("fold");
    Builder b(mod);
    Function* fn = mod.create_function("fold", Type::i32(), {});
    b.set_function(fn);
    b.append_block("entry");
    Value* f = b.build_bitcast_f32_i32(b.build_iconst_i32(0x40490fdb)); // pi
    Value* r = b.build_bitcast_i32_f32(b.build_mul(f, b.build_fconst_f32(2.0f)));
    b.build_ret(r);
    Function* nan = mod.create_function("nan", Type::f32(), {});
    b.set_function(nan);
    b.append_block("entry");
    b.build_ret(b.build_bitcast_f32_i32(b.build_iconst_i32(0x7f800001))); // signalling NaN
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    CHECK(sccp_function(*fn));
    size_t left = 0;
    for (const Instruction* inst : *fn->blocks().front())
        if (inst->opcode() == Opcode::bitcast_i32_f32 || inst->opcode() == Opcode::bitcast_f32_i32) ++left;
    CHECK_EQ(left, size_t(0));
    Interpreter interp;
    CHECK_EQ(static_cast<uint32_t>(interp.run(*fn, {}).as_i32()), bits_of(float_of(0x40490fdbu) * 2.0f));

    sccp_function(*nan);
    size_t kept = 0;
    for (const Instruction* inst : *nan->blocks().front())
        if (inst->opcode() == Opcode::bitcast_f32_i32) ++kept;
    CHECK_EQ(kept, size_t(1));
    CHECK_EQ(static_cast<uint32_t>(interp.run(*nan, {}).raw_bits()), 0x7f800001u);
}

TEST_CASE("Bitcast32 - interpreter and bytecode VM move the bits unchanged") {
    auto mod = parse_verified(kFlipSrc);
    Interpreter interp;
    FastInterpreter vm;
    vm.set_module(mod.get());
    const Function& flip = *mod->get_function("flip");
    const Function& widen = *mod->get_function("widen");
    for (float x : kFloats) {
        const uint32_t want = bits_of(x) ^ 0x80000000u;
        CHECK_EQ(static_cast<uint32_t>(interp.run(*mod, "flip", {RuntimeValue::from_f32(x)}).as_i32()), want);
        CHECK_EQ(static_cast<uint32_t>(vm.run(flip, {RuntimeValue::from_f32(x)}).as_i32()), want);
    }
    for (uint32_t w : kWords) {
        RuntimeValue a = interp.run(*mod, "widen", {RuntimeValue::from_i32(static_cast<int32_t>(w))});
        RuntimeValue v = vm.run(widen, {RuntimeValue::from_i32(static_cast<int32_t>(w))});
        CHECK(a.is_f32());
        CHECK_EQ(static_cast<uint32_t>(a.raw_bits()), w);
        CHECK_EQ(static_cast<uint32_t>(v.raw_bits() & 0xffffffffu), w);
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

TEST_CASE("Bitcast32 - baseline tier (movd / fmov through stack slots)") {
    auto mod = parse_verified(kFlipSrc);
    codegen::BaselineJitCompiler compiler;
    auto flip = compiler.compile(*mod->get_function("flip"));
    auto widen = compiler.compile(*mod->get_function("widen"));
    for (float x : kFloats) {
        RuntimeValue r = flip.invoke({RuntimeValue::from_f32(x)});
        CHECK_EQ(static_cast<uint32_t>(r.as_i32()), bits_of(x) ^ 0x80000000u);
    }
    for (uint32_t w : kWords) {
        RuntimeValue r = widen.invoke({RuntimeValue::from_i32(static_cast<int32_t>(w))});
        CHECK_EQ(static_cast<uint32_t>(r.raw_bits() & 0xffffffffu), w);
    }
}

TEST_CASE("Bitcast32 - optimizing tier through KernelJit, scalar and in a loop") {
    Module mod("k");
    Builder b(mod);
    // flip(f32 x) -> i32, as kFlipSrc, then -x * x through the GPR detour.
    Function* flip = mod.create_function("flip", Type::i32(), {Type::f32()});
    b.set_function(flip);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, Type::f32());
    Value* neg = b.build_bitcast_f32_i32(b.build_xor(b.build_bitcast_i32_f32(x), b.build_iconst_i32(INT32_MIN)));
    b.build_ret(b.build_bitcast_i32_f32(b.build_mul(neg, x)));

    // widen(const i32* in, f32* out, i64 n): out[i] = bf16 in[i] as f32.
    Function* widen = mod.create_function("widen", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i64()});
    b.set_function(widen);
    BasicBlock* we = b.append_block("entry");
    Value* in = b.add_block_param(we, Type::ptr());
    Value* out = b.add_block_param(we, Type::ptr());
    Value* n = b.add_block_param(we, Type::i64());
    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* done = b.create_block("done");
    b.position_at_end(we);
    b.build_br(hdr, {b.build_iconst_i64(0)});
    widen->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    b.build_br_if(b.build_slt(i, n), body, {}, done, {});
    widen->append_block(body);
    b.position_at_end(body);
    Value* h = b.build_load_indexed(Type::i32(), in, i, 4);
    Value* f = b.build_bitcast_f32_i32(b.build_shl(h, b.build_iconst_i32(16)));
    b.build_store_indexed(Type::f32(), out, i, 4, f);
    b.build_br(hdr, {b.build_add(i, b.build_iconst_i64(1))});
    widen->append_block(done);
    b.position_at_end(done);
    b.build_ret_void();
    widen->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::KernelJit jit;
    codegen::KernelFunction kflip = jit.compile(mod, "flip");
    REQUIRE(kflip.is_valid());
    auto flip_fn = kflip.as<uint32_t (*)(float)>();
    for (float v : kFloats) {
        if (v != v) continue; // NaN * NaN: the payload is the FPU's choice
        CHECK_EQ(flip_fn(v), bits_of(-v * v));
    }

    codegen::KernelFunction kwiden = jit.compile(mod, "widen");
    REQUIRE(kwiden.is_valid());
    auto widen_fn = kwiden.as<void (*)(const int32_t*, float*, int64_t)>();
    std::vector<int32_t> hs;
    for (int k = 0; k < 37; ++k) hs.push_back((k * 0x0491) & 0xffff);
    std::vector<float> got(hs.size(), 0.0f);
    widen_fn(hs.data(), got.data(), static_cast<int64_t>(hs.size()));
    for (size_t k = 0; k < hs.size(); ++k) CHECK_EQ(bits_of(got[k]), static_cast<uint32_t>(hs[k]) << 16);
}

#endif
