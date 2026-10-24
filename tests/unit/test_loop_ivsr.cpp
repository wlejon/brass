// Induction-variable strength reduction (strength_reduce_induction_variables):
// it moves a loop's exit compare onto the scaled variable only when that
// cannot overflow, steps a decrementing variable down, and leaves GC-heap
// bases alone; a variable stride becomes a pointer variable, and the
// unroller's hoisted trip test holds at the edges of the counter's range.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <iostream>
#include <limits>
#include <string>

using namespace brass;

namespace {

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

int64_t run(const Module& mod, int64_t arg) {
    Interpreter interp;
    return interp.run(mod, "f", {RuntimeValue::from_i64(arg)}).as_i64();
}

// The header compare of the loop headed by block `hdr`.
const Instruction* header_compare(const Function& fn) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() != "hdr") continue;
        const Instruction* term = bb->terminator();
        REQUIRE(term && term->opcode() == Opcode::br_if);
        return term->operand(0)->defining_instruction();
    }
    REQUIRE(false);
    return nullptr;
}

const Value* header_param(const Function& fn, size_t i) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() == "hdr") return bb->param(i);
    }
    REQUIRE(false);
    return nullptr;
}

void check_ivsr_keeps_answer(std::string_view text, int64_t arg, bool expect_change) {
    auto before = parse(text);
    auto mod = parse(text);
    Function& fn = *mod->get_function("f");
    CHECK_EQ(strength_reduce_induction_variables(fn), expect_change);
    DiagnosticReporter diag;
    const bool ok = verify_module(*mod, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n" << to_string(*mod) << "\n";
    REQUIRE(ok);
    CHECK_EQ(run(*mod, arg), run(*before, arg));
}

// i runs 0..4 and leaves through the early exit, but the header compares it
// with 2^62: scaled by 8 that limit wraps to 0, so `8*i < 8*limit` would end
// the loop at once.
constexpr std::string_view kHugeLimit = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 64, 8
  %zero = iconst.i64 0
  %big = iconst.i64 4611686018427387904
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %big
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %i, 8, %i
  %v = load_indexed.i64 %buf, %i, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  %stop = sgt.i64 %i1, %n
  br_if %stop, exit(%acc2), hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";

constexpr std::string_view kSmallLimit = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 64, 8
  %zero = iconst.i64 0
  %lim = iconst.i64 6
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %lim
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %i, 8, %n
  %v = load_indexed.i64 %buf, %i, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";

// j counts down from 15 while i counts the trips; j only indexes, so it is
// replaced by a scaled variable, which must count down with it.
constexpr std::string_view kCountDown = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 128, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %top = iconst.i64 15
  br hdr(%zero, %top, %zero)

hdr(%i: i64, %j: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit(%acc)

body:
  store_indexed.i64 %buf, %j, 8, %n
  %v = load_indexed.i64 %buf, %j, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  %j1 = sub.i64 %j, %one
  br hdr(%i1, %j1, %acc2)

exit(%r: i64):
  ret %r
}
)";

} // namespace

TEST_CASE("IVSR - an exit compare whose scaled limit would overflow stays unscaled") {
    // i stays live for the compare, so its accesses keep addressing it
    // scaled by 8 and no second, scaled variable is made.
    check_ivsr_keeps_answer(kHugeLimit, 4, false);
    auto mod = parse(kHugeLimit);
    Function& fn = *mod->get_function("f");
    strength_reduce_induction_variables(fn);
    CHECK(header_compare(fn)->operand(0) == header_param(fn, 0));
}

TEST_CASE("IVSR - an exit compare with small constant bounds moves to the scaled variable") {
    check_ivsr_keeps_answer(kSmallLimit, 7, true);
    auto mod = parse(kSmallLimit);
    Function& fn = *mod->get_function("f");
    strength_reduce_induction_variables(fn);
    CHECK(header_compare(fn)->operand(0) != header_param(fn, 0));
}

TEST_CASE("IVSR - a decrementing induction variable scales downwards") {
    for (int64_t n : {0, 1, 5, 12}) check_ivsr_keeps_answer(kCountDown, n, n >= 0);
}

TEST_CASE("IVSR - GC-heap bases and i32 variables are left alone") {
    auto mod = parse(R"(
func @f(%buf: gcref, %n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%zero, %zero)

hdr(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit(%acc)

body:
  %v = load_indexed.i64 %buf, %i, 8, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}

func @g(%buf: ptr, %n: i32) -> i64 {
bb0:
  %zero = iconst.i32 0
  %z64 = iconst.i64 0
  %one = iconst.i32 1
  br hdr(%zero, %z64)

hdr(%i: i32, %acc: i64):
  %c = slt.i32 %i, %n
  br_if %c, body, exit(%acc)

body:
  %w = sext.i64 %i
  %v = load_indexed.i64 %buf, %w, 8
  %acc2 = add.i64 %acc, %v
  %i1 = add.i32 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)");
    CHECK(!strength_reduce_induction_variables(*mod->get_function("f")));
    CHECK(!strength_reduce_induction_variables(*mod->get_function("g")));
    CHECK(verify_module(*mod));
}

namespace {

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

// A column walk: kS steps by a stride only known at run time (1..4), so its
// accesses become loads and stores through a pointer variable, which the
// unroller then copies. The trip count is n & 15.
constexpr std::string_view kVariableStride = R"(
func @f(%n: i64) -> i64 {
bb0:
  %buf = alloca 1024, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %three = iconst.i64 3
  %fifteen = iconst.i64 15
  %four = iconst.i64 4
  %lim4 = iconst.i64 128
  %lim = and.i64 %n, %fifteen
  %hi = lshr.i64 %n, %four
  %sm = and.i64 %hi, %three
  %s = add.i64 %sm, %one
  br fill(%zero)

fill(%f: i64):
  %fc = slt.i64 %f, %lim4
  br_if %fc, fill_body, pre

fill_body:
  %fv = mul.i64 %f, %f
  %fv2 = add.i64 %fv, %n
  store_indexed.i64 %buf, %f, 8, %fv2
  %f2 = add.i64 %f, %one
  br fill(%f2)

pre:
  br hdr(%zero, %zero, %zero)

hdr(%k: i64, %ks: i64, %acc: i64):
  %c = slt.i64 %k, %lim
  br_if %c, body, exit(%acc)

body:
  %v = load_indexed.i64 %buf, %ks, 8, 8
  %acc2 = add.i64 %acc, %v
  %ks2 = add.i64 %ks, %s
  %k2 = add.i64 %k, %one
  br hdr(%k2, %ks2, %acc2)

exit(%r: i64):
  ret %r
}
)";

// Trip tests at the edges of the counter's range: i runs from `start` up to
// `limit` (signed, then unsigned); the unrolled loop's hoisted, clamped
// limit must neither run a copy past the bound nor wrap.
std::string edge_loop(const char* cmp, int64_t start, int64_t limit) {
    return std::string(R"(
func @f(%n: i64) -> i64 {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  %seven = iconst.i64 7
  %start = iconst.i64 )") + std::to_string(start) + R"(
  %limit = iconst.i64 )" + std::to_string(limit) + R"(
  %lim = add.i64 %limit, %n
  br hdr(%start, %zero)

hdr(%i: i64, %acc: i64):
  %c = )" + cmp + R"(.i64 %i, %lim
  br_if %c, body, exit(%acc)

body:
  %b = and.i64 %i, %seven
  %acc2 = add.i64 %acc, %b
  %i1 = add.i64 %i, %one
  br hdr(%i1, %acc2)

exit(%r: i64):
  ret %r
}
)";
}

} // namespace

TEST_CASE("IVSR - a variable stride walks a pointer variable the unroller copies") {
    for (int64_t n : {0, 1, 3, 4, 5, 15, 16 + 7, 32 + 9, 48 + 15}) {
        auto before = parse(kVariableStride);
        auto mod = parse(kVariableStride);
        Function& fn = *mod->get_function("f");
        CHECK(optimize_function_loops(fn));
        DiagnosticReporter diag;
        const bool ok = verify_module(*mod, &diag);
        if (!ok) std::cerr << diag.format_all() << "\n" << to_string(*mod) << "\n";
        REQUIRE(ok);
        CHECK_EQ(count_op(fn, Opcode::load_indexed), size_t{0});
        CHECK(count_op(fn, Opcode::load) >= 5u);  // four unrolled copies and the remainder
        CHECK_EQ(run(*mod, n), run(*before, n));
    }
}

TEST_CASE("Loop unroll - the trip test holds at the edges of the counter's range") {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    struct Case {
        const char* cmp;
        int64_t start;
        int64_t limit;
    };
    const Case cases[] = {
        {"slt", kMax - 10, kMax},  // the last copies sit right below MAX
        {"slt", kMin, kMin + 2},   // limit below MIN + 3: only the remainder runs
        {"slt", kMin, kMin},       // no trips
        {"slt", -5, 6},
        {"ult", -11, -1},          // unsigned, right below 2^64 - 1
        {"ult", 0, 2},             // limit below 3
        {"ult", 0, 0},
    };
    for (const Case& c : cases) {
        for (int64_t n : {0, 1, 2, 3}) {
            const std::string text = edge_loop(c.cmp, c.start, c.limit);
            auto before = parse(text);
            auto mod = parse(text);
            optimize_function_loops(*mod->get_function("f"));
            REQUIRE(verify_module(*mod));
            CHECK_EQ(count_op(*mod->get_function("f"), Opcode::select), size_t{1});  // the clamp
            // (n wraps the first and fifth limits to below the start: no trips.)
            CHECK_EQ(run(*mod, n), run(*before, n));
        }
    }
}

TEST_CASE("F64 Demote & IVSR - Counted bitwise accumulator loop demotes, folds ToInt32 guard, and unrolls 4x") {
    constexpr std::string_view kScalarLoop = R"(
func @scalar(%n: f64) -> f64 {
b0:
  %zero = fconst.f64 0
  br b1(%zero, %zero)

b1(%s: f64, %i: f64):
  %c = slt.f64 %i, %n
  br_if %c, b2, b_exit

b2:
  %three = fconst.f64 3
  %p = mul.f64 %i, %three
  %p_bits = bitcast.i64.f64 %p
  %mask = iconst.i64 9223372036854775807
  %p_abs = and.i64 %p_bits, %mask
  %lim54 = iconst.i64 4850380504492539904
  %fast = ult.i64 %p_abs, %lim54
  br_if %fast, b_fast, b_slow

b_fast:
  %p_i64 = fptosi.i64 %p
  %p_i32 = trunc.i32 %p_i64
  br b_join(%p_i32)

b_slow:
  %zero32 = iconst.i32 0
  br b_join(%zero32)

b_join(%lhs: i32):
  %i_i64 = fptosi.i64 %i
  %i_i32 = trunc.i32 %i_i64
  %one32 = iconst.i32 1
  %shr = lshr.i32 %i_i32, %one32
  %x = xor.i32 %lhs, %shr
  %m255 = iconst.i32 255
  %masked = and.i32 %x, %m255
  %df = sitofp.f64.i32 %masked
  %s2 = add.f64 %s, %df
  br b_latch(%s2, %i)

b_latch(%s_l: f64, %i_l: f64):
  %one_f = fconst.f64 1
  %i2 = add.f64 %i_l, %one_f
  br b1(%s_l, %i2)

b_exit:
  ret %s
}
)";
    auto mod = parse(kScalarLoop);
    Function& fn = *mod->get_function("scalar");
    LoopOptOptions opt;
    CHECK(optimize_function_loops(fn, opt));
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    // All f64 arithmetic inside the loop is demoted to integer, the ToInt32
    // slow path is folded away, and the loop is 4x unrolled.
    CHECK_EQ(count_op(fn, Opcode::fptosi_i64), size_t{0});
    CHECK(count_op(fn, Opcode::xor_) >= 4u);

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto scalar_fn = jit.get_function_ptr<double(*)(double)>("scalar");
    REQUIRE(scalar_fn != nullptr);
    auto ref_scalar = [](int64_t n) -> double {
        int64_t s = 0;
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t lhs = static_cast<uint32_t>(static_cast<int32_t>(i * 3));
            const uint32_t rhs = static_cast<uint32_t>(i) >> 1;
            s += static_cast<int64_t>((lhs ^ rhs) & 0xffu);
        }
        return static_cast<double>(s);
    };
    CHECK_EQ(scalar_fn(1000.0), ref_scalar(1000));
    CHECK_EQ(scalar_fn(100000.0), ref_scalar(100000));
}

TEST_CASE("IVSR - 2D stencil blur loop decomposes (y+dy)*w+(x+dx) into constant displacements and scaled IV") {
    constexpr std::string_view kBlurLoop = R"(
func @blur(%src: ptr, %dst: ptr, %w: f64, %h: f64) -> f64 {
b0:
  %one = fconst.f64 1
  br b_y(%one)

b_y(%y: f64):
  %hm1 = sub.f64 %h, %one
  %cy = slt.f64 %y, %hm1
  br_if %cy, b_y_body, b_exit

b_y_body:
  br b_x(%one)

b_x(%x: f64):
  %wm1 = sub.f64 %w, %one
  %cx = slt.f64 %x, %wm1
  br_if %cx, b_x_body, b_y_latch

b_x_body:
  %ym1 = sub.f64 %y, %one
  %yp1 = add.f64 %y, %one
  %xm1 = sub.f64 %x, %one
  %xp1 = add.f64 %x, %one
  %rt = mul.f64 %ym1, %w
  %rm = mul.f64 %y, %w
  %rb = mul.f64 %yp1, %w
  %i00 = fptosi.i64 %rt
  %x0 = fptosi.i64 %xm1
  %x1 = fptosi.i64 %x
  %x2 = fptosi.i64 %xp1
  %r1 = fptosi.i64 %rm
  %r2 = fptosi.i64 %rb
  %idx00 = add.i64 %i00, %x0
  %idx01 = add.i64 %i00, %x1
  %idx02 = add.i64 %i00, %x2
  %idx10 = add.i64 %r1, %x0
  %idx11 = add.i64 %r1, %x1
  %idx12 = add.i64 %r1, %x2
  %idx20 = add.i64 %r2, %x0
  %idx21 = add.i64 %r2, %x1
  %idx22 = add.i64 %r2, %x2
  %v00 = load_indexed.f32 %src, %idx00, 4
  %v01 = load_indexed.f32 %src, %idx01, 4
  %v02 = load_indexed.f32 %src, %idx02, 4
  %v10 = load_indexed.f32 %src, %idx10, 4
  %v11 = load_indexed.f32 %src, %idx11, 4
  %v12 = load_indexed.f32 %src, %idx12, 4
  %v20 = load_indexed.f32 %src, %idx20, 4
  %v21 = load_indexed.f32 %src, %idx21, 4
  %v22 = load_indexed.f32 %src, %idx22, 4
  %s0 = add.f32 %v00, %v01
  %s1 = add.f32 %s0, %v02
  %s2 = add.f32 %s1, %v10
  %s3 = add.f32 %s2, %v11
  %s4 = add.f32 %s3, %v12
  %s5 = add.f32 %s4, %v20
  %s6 = add.f32 %s5, %v21
  %s7 = add.f32 %s6, %v22
  store_indexed.f32 %dst, %idx11, 4, %s7
  %x_next = add.f64 %x, %one
  br b_x(%x_next)

b_y_latch:
  %y_next = add.f64 %y, %one
  br b_y(%y_next)

b_exit:
  %z = iconst.i64 0
  %out0 = load_indexed.f32 %dst, %z, 4, 20
  %retv = fpext.f64.f32 %out0
  ret %retv
}
)";
    auto mod = parse(kBlurLoop);
    Function& fn = *mod->get_function("blur");
    CHECK(optimize_function_loops(fn));
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    // All 6 in-loop fptosi_i64 in the stencil are demoted away (only %w conversion in preheader remains).
    CHECK(count_op(fn, Opcode::fptosi_i64) <= 1u);

    std::vector<float> src(16, 2.0f), dst(16, 0.0f);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto blur_fn = jit.get_function_ptr<double(*)(float*, float*, double, double)>("blur");
    REQUIRE(blur_fn != nullptr);
    CHECK_EQ(blur_fn(src.data(), dst.data(), 4.0, 4.0), 18.0);
}

TEST_CASE("IVSR - 2D matrix multiply inner loop strength-reduces b[k*n+j] to pointer IV") {
    constexpr std::string_view kMatLoop = R"(
func @mat(%a: ptr, %b: ptr, %c: ptr, %n: f64) -> f64 {
b0:
  %zero = fconst.f64 0
  %one = fconst.f64 1
  br b_i(%zero)

b_i(%i: f64):
  %ci = slt.f64 %i, %n
  br_if %ci, b_i_body, b_exit

b_i_body:
  %in = mul.f64 %i, %n
  br b_j(%zero)

b_j(%j: f64):
  %cj = slt.f64 %j, %n
  br_if %cj, b_j_body, b_i_latch

b_j_body:
  br b_k(%zero, %zero)

b_k(%s: f64, %k: f64):
  %ck = slt.f64 %k, %n
  br_if %ck, b_k_body, b_j_latch(%s)

b_k_body:
  %ik_f = add.f64 %in, %k
  %ik = fptosi.i64 %ik_f
  %va = load_indexed.f64 %a, %ik, 8
  %kn = mul.f64 %k, %n
  %kj_f = add.f64 %kn, %j
  %kj = fptosi.i64 %kj_f
  %vb = load_indexed.f64 %b, %kj, 8
  %prod = mul.f64 %va, %vb
  %s2 = add.f64 %s, %prod
  %k2 = add.f64 %k, %one
  br b_k(%s2, %k2)

b_j_latch(%s_out: f64):
  %ij_f = add.f64 %in, %j
  %ij = fptosi.i64 %ij_f
  store_indexed.f64 %c, %ij, 8, %s_out
  %j2 = add.f64 %j, %one
  br b_j(%j2)

b_i_latch:
  %i2 = add.f64 %i, %one
  br b_i(%i2)

b_exit:
  %z = iconst.i64 0
  %r = load_indexed.f64 %c, %z, 8
  ret %r
}
)";
    auto mod = parse(kMatLoop);
    Function& fn = *mod->get_function("mat");
    CHECK(optimize_function_loops(fn));
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    CHECK(count_op(fn, Opcode::fptosi_i64) <= 1u);
    // b[k*n+j] becomes a direct `load` through a stepped pointer IV (4x unrolled + remainder).
    CHECK(count_op(fn, Opcode::load) >= 5u);

    std::vector<double> a(16, 3.0), b(16, 2.0), c(16, 0.0);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto mat_fn = jit.get_function_ptr<double(*)(double*, double*, double*, double)>("mat");
    REQUIRE(mat_fn != nullptr);
    CHECK_EQ(mat_fn(a.data(), b.data(), c.data(), 4.0), 24.0);
}
