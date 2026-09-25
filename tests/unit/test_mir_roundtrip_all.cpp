// Printer/parser round trip over every opcode and every attribute the
// printer writes: the corpus below is in canonical printed form, so
// print(parse(corpus)) must reproduce it byte for byte (nothing the parser
// reads is dropped, nothing the printer writes is unreadable), and the test
// walks the whole Opcode enum to make sure the corpus leaves none out.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <iostream>
#include <set>
#include <string>

using namespace brass;

namespace {

constexpr std::string_view kCorpus = R"(module @all_opcodes

extern @rt_alloc allocator
extern @rt_pure pure
extern @rt_arrays array_new array_get
extern @rt_set array_set
extern @rt_frem frem
extern @rt_plain

func @scalars(%0: i32, %1: i64, %2: f64, %3: f32) -> i64 {
bb0:
  %4 = iconst.i32 -7
  %5 = iconst.i64 9
  %6 = fconst.f64 1.5
  %7 = fconst.f32 2.25
  %8 = patchable_const.i32 @site32, 3
  %9 = patchable_const.i64 @site64, -4
  %10 = sext.i64 %0
  %11 = zext.i64 %0
  %12 = trunc.i32 %1
  %13 = trunc.i8 %1
  %14 = fptosi.i32 %2
  %15 = fptosi.i64 %2
  %16 = fptosi.i32.f32 %3
  %17 = fptosi.i64.f32 %3
  %18 = sitofp.f64.i32 %0
  %19 = sitofp.f64.i64 %1
  %20 = sitofp.f32.i32 %0
  %21 = sitofp.f32.i64 %1
  %22 = fptrunc.f32.f64 %2
  %23 = fpext.f64.f32 %3
  %24 = bitcast.i64.f64 %2
  %25 = bitcast.f64.i64 %1
  %26 = add.i64 %1, %5
  %27 = sub.i64 %1, %5
  %28 = mul.i64 %1, %5
  %29 = sdiv.i64 %1, %5
  %30 = udiv.i64 %1, %5
  %31 = smod.i64 %1, %5
  %32 = umod.i64 %1, %5
  %33 = shl.i64 %1, %5
  %34 = lshr.i64 %1, %5
  %35 = ashr.i64 %1, %5
  %36 = and.i64 %1, %5
  %37 = or.i64 %1, %5
  %38 = xor.i64 %1, %5
  %39 = neg.i64 %1
  %40 = clz.i64 %1
  %41 = ctz.i64 %1
  %42 = popcnt.i64 %1
  %43 = not.i64 %1
  %44 = eq.i64 %1, %5
  %45 = ne.i64 %1, %5
  %46 = slt.i64 %1, %5
  %47 = ult.i64 %1, %5
  %48 = sle.i64 %1, %5
  %49 = ule.i64 %1, %5
  %50 = sgt.i32 %0, %4
  %51 = ugt.i32 %0, %4
  %52 = sge.i32 %0, %4
  %53 = uge.i32 %0, %4
  %54 = sadd_overflow.i64 %1, %5
  %55 = ssub_overflow.i64 %1, %5
  %56 = smul_overflow.i64 %1, %5
  %57 = uadd_overflow.i32 %0, %4
  %58 = usub_overflow.i32 %0, %4
  %59 = umul_overflow.i32 %0, %4
  %60 = select.i64 %44, %1, %5
  %61 = fma.f64 %2, %6, %2
  %62 = fma.f32 %3, %7, %3
  %63 = sqrt_f32 %3
  %64 = sqrt_f64 %2
  %65 = floor_f32 %3
  %66 = floor_f64 %2
  %67 = ceil_f32 %3
  %68 = ceil_f64 %2
  %69 = round_f32 %3
  %70 = round_f64 %2
  %71 = fabs_f32 %3
  %72 = fabs_f64 %2
  %73 = fmin_f32 %3, %7
  %74 = fmin_f64 %2, %6
  %75 = fmax_f32 %3, %7
  %76 = fmax_f64 %2, %6
  %77 = add.f64 %2, %6
  %78 = sdiv.f64 %2, %6
  ret %26
}

func @memory(%0: ptr, %1: i64, %2: gcref, %3: i32) -> i64 {
bb0:
  %4 = alloca 64, 8
  %5 = load.i64 %0
  %6 = load.i32 %0, 12
  store.i64 %4, %1
  store.f64 %4, 16, %1
  %7 = load_indexed.i64 %0, %1, 8
  %8 = load_indexed.i32 %0, %1, 4, 8
  store_indexed.i64 %0, %1, 8, %5
  store_indexed.i8 %0, %1, 1, 3, %6
  write_barrier %2, %2
  %9 = pinned_tls_read
  pinned_tls_write %9
  %10 = read_sp
  %11 = call.i64 @rt_pure(%1)
  call @rt_plain()
  %12 = func_addr @memory
  %13 = call_indirect.i64 %12(%0, %1, %2, %3)
  call_indirect %12()
  %14 = patchable_call.i64 @site_call, @rt_plain(%1, %5)
  patchable_call @site_void, @rt_plain()
  safepoint
  %15 = slt.i64 %1, %5
  guard %15, @deopt_here, [%1, %5]
  guard %15, @deopt_bare
  resume_point 3
  br_if %15, bb1(%1), bb2
  br bb2

bb1(%16: i64):
  switch.i64 %16, default: bb2, [0: bb3(%16), -5: bb2]

bb2:
  switch.i32 %3, default: bb3(%1), []

bb3(%17: i64):
  ret %17

bb4:
  unreachable

resume_table {
  entry 1 -> bb2
  entry 4 -> bb3
}
}

func @exceptions(%0: i64) -> i64 {
bb0:
  %1 = invoke.i64 @rt_plain(%0), bb1(%0), bb2
  invoke @rt_plain(), bb1(%0), bb2

bb1(%2: i64):
  ret %2

bb2:
  %3 = landing_pad
  %4 = landing_pad.ptr
  throw %3

bb3:
  resume %3

bb4:
  resume
}

func @vectors(%0: ptr, %1: f32, %2: i64) -> void {
bb0:
  %3: f32x4 = vload.f32x4 %0, 0
  %4: f32x4 = vload.f32x4 %0, 16
  %5: f32x4 = vadd %3, %4
  %6: f32x4 = vsub %3, %4
  %7: f32x4 = vmul %3, %4
  %8: f32x4 = vdiv %3, %4
  %9: f32x4 = vfma %3, %4, %5
  %10: f32x4 = vneg %3
  %11: f32x4 = vmin %3, %4
  %12: f32x4 = vmax %3, %4
  %13: f32x4 = vsqrt %3
  %14: i64x4 = vload.i64x4 %0, 32
  %15: i64x4 = vand %14, %14
  %16: i64x4 = vor %14, %14
  %17: i64x4 = vxor %14, %14
  %18: i64x4 = vnot %14
  %19: f32x4 = vbroadcast.f32x4 %1
  %20: f32 = vextract_lane %3, 2
  %21: f32x4 = vinsert_lane %3, %20, 1
  %22: f32x4 = vshuffle %3, %4, 0x1B
  %23: f64x2 = vzero.f64x2
  %24: i64x2 = vbroadcast.i64x2 %2
  %25: i32x8 = vzero.i32x8
  %26: f64x4 = vzero.f64x4
  %27: f32x8 = vzero.f32x8
  %28: i32x4 = vzero.i32x4
  vstore.f32x4 %0, 48, %22
  vstore.i64x4 %0, 0, %18
  ret
}

func @coroutines(%0: i64) -> i64 {
bb0:
  %1 = coro_create @coroutines(%0)
  %2 = coro_suspend.i64 %0, 2
  coro_suspend 0
  %3 = coro_resume.i64 %1, %0
  coro_resume %1
  coro_destroy %1
  ret %3
}
)";

std::unique_ptr<Module> parse_or_report(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod || diag.has_errors()) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    REQUIRE(!diag.has_errors());
    return mod;
}

// Prints the first line where `a` and `b` differ.
void report_first_difference(const std::string& a, const std::string& b) {
    size_t line = 1, i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        if (a[i] == '\n') ++line;
        ++i;
    }
    auto line_at = [](const std::string& s, size_t pos) {
        const size_t start = s.rfind('\n', pos == 0 ? 0 : pos - 1);
        const size_t from = start == std::string::npos ? 0 : start + 1;
        const size_t end = s.find('\n', pos);
        return s.substr(from, end == std::string::npos ? std::string::npos : end - from);
    };
    std::cerr << "first difference at line " << line << ":\n  expected: " << line_at(a, i)
              << "\n  printed:  " << line_at(b, i) << "\n";
}

} // namespace

TEST_CASE("Roundtrip - every opcode and printed attribute survives print(parse(text))") {
    const std::string corpus(kCorpus);
    auto mod = parse_or_report(corpus);
    const std::string printed = to_string(*mod);
    if (printed != corpus) report_first_difference(corpus, printed);
    CHECK(printed == corpus);

    auto again = parse_or_report(printed);
    CHECK(to_string(*again) == printed);
}

TEST_CASE("Roundtrip - the corpus covers the whole Opcode enum") {
    auto mod = parse_or_report(kCorpus);
    std::set<Opcode> seen;
    for (const Function* fn : mod->functions()) {
        for (const BasicBlock* bb : fn->blocks()) {
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) seen.insert(inst->opcode());
        }
    }
    const auto last = static_cast<uint16_t>(Opcode::coro_destroy);
    for (uint16_t i = 0; i <= last; ++i) {
        const auto op = static_cast<Opcode>(i);
        if (!seen.count(op)) std::cerr << "opcode missing from the round-trip corpus: " << opcode_name(op) << "\n";
        CHECK(seen.count(op) == 1u);
    }
}

TEST_CASE("Roundtrip - runtime symbol roles print and parse back") {
    auto mod = parse_or_report(kCorpus);
    CHECK(mod->has_symbol_role("rt_alloc", SymbolRole::Allocator));
    CHECK(mod->is_allocation_function("rt_alloc"));
    CHECK(mod->has_symbol_role("rt_pure", SymbolRole::Pure));
    CHECK(mod->has_symbol_role("rt_arrays", SymbolRole::ArrayNew));
    CHECK(mod->has_symbol_role("rt_arrays", SymbolRole::ArrayGet));
    CHECK(!mod->has_symbol_role("rt_arrays", SymbolRole::ArraySet));
    CHECK(mod->has_symbol_role("rt_set", SymbolRole::ArraySet));
    CHECK(mod->has_symbol_role("rt_frem", SymbolRole::FloatRem));
    CHECK(mod->symbol_roles("rt_plain").empty());
    for (SymbolRole role : kAllSymbolRoles) {
        auto parsed = parse_symbol_role(symbol_role_name(role));
        REQUIRE(parsed.has_value());
        CHECK(*parsed == role);
    }

    // A declaration a module clone copies keeps its roles.
    auto copy = clone_module(*mod);
    CHECK(copy->has_symbol_role("rt_frem", SymbolRole::FloatRem));

    DiagnosticReporter diag;
    auto bad = parse_module("extern @x allocatr\n", &diag);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("Unknown runtime symbol role 'allocatr'") != std::string::npos);
}
