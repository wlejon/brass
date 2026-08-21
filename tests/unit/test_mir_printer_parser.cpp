#include "test_framework.hpp"
#include <brass/brass.hpp>

using namespace brass;

TEST_CASE("Lexer basic tokens and source locations") {
    std::string_view src = "func @fib(%0: i32) -> i32 {\n"
                           "  // comment line\n"
                           "  %1 = add.i32 %0, 42\n"
                           "  ret %1\n"
                           "}\n";
    Lexer lexer(src, "test.mir");

    Token t1 = lexer.next_token();
    CHECK_EQ(t1.kind, TokenKind::Kw_func);
    CHECK_EQ(t1.location.line, 1u);

    Token t2 = lexer.next_token();
    CHECK_EQ(t2.kind, TokenKind::SymbolIdent);
    CHECK_EQ(t2.text, "@fib");

    Token t3 = lexer.next_token();
    CHECK_EQ(t3.kind, TokenKind::LParen);

    Token t4 = lexer.next_token();
    CHECK_EQ(t4.kind, TokenKind::ValueIdent);
    CHECK_EQ(t4.text, "%0");

    Token t5 = lexer.next_token();
    CHECK_EQ(t5.kind, TokenKind::Colon);

    Token t6 = lexer.next_token();
    CHECK_EQ(t6.kind, TokenKind::Kw_i32);

    Token t7 = lexer.next_token();
    CHECK_EQ(t7.kind, TokenKind::RParen);

    Token t8 = lexer.next_token();
    CHECK_EQ(t8.kind, TokenKind::Arrow);

    Token t9 = lexer.next_token();
    CHECK_EQ(t9.kind, TokenKind::Kw_i32);

    Token t10 = lexer.next_token();
    CHECK_EQ(t10.kind, TokenKind::LBrace);

    Token t11 = lexer.next_token();
    CHECK_EQ(t11.kind, TokenKind::ValueIdent);
    CHECK_EQ(t11.text, "%1");
    CHECK_EQ(t11.location.line, 3u);

    Token t12 = lexer.next_token();
    CHECK_EQ(t12.kind, TokenKind::Equal);

    Token t13 = lexer.next_token();
    CHECK_EQ(t13.kind, TokenKind::Ident);
    CHECK_EQ(t13.text, "add.i32");

    Token t14 = lexer.next_token();
    CHECK_EQ(t14.kind, TokenKind::ValueIdent);
    CHECK_EQ(t14.text, "%0");

    Token t15 = lexer.next_token();
    CHECK_EQ(t15.kind, TokenKind::Comma);

    Token t16 = lexer.next_token();
    CHECK_EQ(t16.kind, TokenKind::IntLiteral);
    CHECK_EQ(t16.int_val, 42);

    Token t17 = lexer.next_token();
    CHECK_EQ(t17.kind, TokenKind::Kw_ret);

    Token t18 = lexer.next_token();
    CHECK_EQ(t18.kind, TokenKind::ValueIdent);
    CHECK_EQ(t18.text, "%1");

    Token t19 = lexer.next_token();
    CHECK_EQ(t19.kind, TokenKind::RBrace);

    Token t20 = lexer.next_token();
    CHECK_EQ(t20.kind, TokenKind::Eof);
}

TEST_CASE("Lexer number literals hex and float") {
    std::string_view src = "0 42 -100 0x1f -0x2A 3.14 -0.5 1e-5 0x1.5p+3 nan inf -inf";
    Lexer lexer(src);

    Token t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::IntLiteral);
    CHECK_EQ(t.int_val, 0);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::IntLiteral);
    CHECK_EQ(t.int_val, 42);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::IntLiteral);
    CHECK_EQ(t.int_val, -100);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::IntLiteral);
    CHECK_EQ(t.int_val, 0x1f);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::IntLiteral);
    CHECK_EQ(t.int_val, -0x2A);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::abs(t.float_val - 3.14) < 1e-9);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::abs(t.float_val - (-0.5)) < 1e-9);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::abs(t.float_val - 1e-5) < 1e-12);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::abs(t.float_val - 10.5) < 1e-9);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::isnan(t.float_val));

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::isinf(t.float_val) && t.float_val > 0);

    t = lexer.next_token();
    CHECK_EQ(t.kind, TokenKind::FloatLiteral);
    CHECK(std::isinf(t.float_val) && t.float_val < 0);
}

static void test_roundtrip(const std::string& mir_text) {
    DiagnosticReporter diag1;
    auto mod1 = parse_module(mir_text, &diag1);
    if (!mod1 || diag1.has_errors()) {
        std::cerr << "Parse error:\n" << diag1.format_all() << "\n";
    }
    REQUIRE(mod1 != nullptr);
    REQUIRE(!diag1.has_errors());

    DiagnosticReporter ver_diag1;
    bool ok1 = verify_module(*mod1, &ver_diag1);
    if (!ok1) {
        std::cerr << "Verify error:\n" << ver_diag1.format_all() << "\n";
    }
    REQUIRE(ok1);

    std::string printed1 = to_string(*mod1);
    CHECK_EQ(printed1, mir_text);

    DiagnosticReporter diag2;
    auto mod2 = parse_module(printed1, &diag2);
    REQUIRE(mod2 != nullptr);
    REQUIRE(!diag2.has_errors());

    DiagnosticReporter ver_diag2;
    bool ok2 = verify_module(*mod2, &ver_diag2);
    REQUIRE(ok2);

    std::string printed2 = to_string(*mod2);
    CHECK_EQ(printed2, printed1);
}

TEST_CASE("Roundtrip Golden 1: Fibonacci and recursive calls") {
    std::string text = "module @fib_module\n\n"
                       "func @fibonacci(%0: i32) -> i32 {\n"
                       "bb0:\n"
                       "  %1 = iconst.i32 2\n"
                       "  %2 = slt.i32 %0, %1\n"
                       "  br_if %2, bb_base, bb_rec\n\n"
                       "bb_base:\n"
                       "  ret %0\n\n"
                       "bb_rec:\n"
                       "  %3 = iconst.i32 1\n"
                       "  %4 = sub.i32 %0, %3\n"
                       "  %5 = call.i32 @fibonacci(%4)\n"
                       "  %6 = iconst.i32 2\n"
                       "  %7 = sub.i32 %0, %6\n"
                       "  %8 = call.i32 @fibonacci(%7)\n"
                       "  %9 = add.i32 %5, %8\n"
                       "  ret %9\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 2: Sum 1 to N loop with block parameters") {
    std::string text = "module @loop_module\n\n"
                       "func @sum_to_n(%0: i32) -> i64 {\n"
                       "bb0:\n"
                       "  %1 = iconst.i32 1\n"
                       "  %2 = iconst.i64 0\n"
                       "  br loop_header(%1, %2)\n\n"
                       "loop_header(%3: i32, %4: i64):\n"
                       "  %5 = sle.i32 %3, %0\n"
                       "  br_if %5, loop_body, loop_exit(%4)\n\n"
                       "loop_body:\n"
                       "  %6 = sext.i64 %3\n"
                       "  %7 = add.i64 %4, %6\n"
                       "  %8 = iconst.i32 1\n"
                       "  %9 = add.i32 %3, %8\n"
                       "  br loop_header(%9, %7)\n\n"
                       "loop_exit(%10: i64):\n"
                       "  ret %10\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 3: Collatz sequence") {
    std::string text = "module @collatz_module\n\n"
                       "func @collatz_steps(%0: i64) -> i32 {\n"
                       "bb0:\n"
                       "  %1 = iconst.i32 0\n"
                       "  br loop_head(%0, %1)\n\n"
                       "loop_head(%2: i64, %3: i32):\n"
                       "  %4 = iconst.i64 1\n"
                       "  %5 = eq.i64 %2, %4\n"
                       "  br_if %5, done(%3), check_even\n\n"
                       "check_even:\n"
                       "  %6 = iconst.i64 1\n"
                       "  %7 = and.i64 %2, %6\n"
                       "  %8 = iconst.i64 0\n"
                       "  %9 = eq.i64 %7, %8\n"
                       "  br_if %9, step_even, step_odd\n\n"
                       "step_even:\n"
                       "  %10 = iconst.i64 1\n"
                       "  %11 = lshr.i64 %2, %10\n"
                       "  %12 = iconst.i32 1\n"
                       "  %13 = add.i32 %3, %12\n"
                       "  br loop_head(%11, %13)\n\n"
                       "step_odd:\n"
                       "  %14 = iconst.i64 3\n"
                       "  %15 = mul.i64 %2, %14\n"
                       "  %16 = iconst.i64 1\n"
                       "  %17 = add.i64 %15, %16\n"
                       "  %18 = iconst.i32 1\n"
                       "  %19 = add.i32 %3, %18\n"
                       "  br loop_head(%17, %19)\n\n"
                       "done(%20: i32):\n"
                       "  ret %20\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 4: Arithmetic, Bitwise, Conversions, and Float") {
    std::string text = "module @math_ops_module\n\n"
                       "func @test_math(%0: i32, %1: i64, %2: f64) -> f64 {\n"
                       "bb0:\n"
                       "  %3 = sext.i64 %0\n"
                       "  %4 = zext.i64 %0\n"
                       "  %5 = trunc.i32 %1\n"
                       "  %6 = fptosi.i32 %2\n"
                       "  %7 = fptosi.i64 %2\n"
                       "  %8 = sitofp.f64.i32 %0\n"
                       "  %9 = sitofp.f64.i64 %1\n"
                       "  %10 = bitcast.i64.f64 %2\n"
                       "  %11 = bitcast.f64.i64 %1\n"
                       "  %12 = add.i32 %0, %5\n"
                       "  %13 = sub.i32 %0, %5\n"
                       "  %14 = mul.i32 %0, %5\n"
                       "  %15 = sdiv.i32 %0, %5\n"
                       "  %16 = udiv.i32 %0, %5\n"
                       "  %17 = smod.i32 %0, %5\n"
                       "  %18 = umod.i32 %0, %5\n"
                       "  %19 = neg.i32 %0\n"
                       "  %20 = and.i32 %0, %5\n"
                       "  %21 = or.i32 %0, %5\n"
                       "  %22 = xor.i32 %0, %5\n"
                       "  %23 = shl.i32 %0, %5\n"
                       "  %24 = lshr.i32 %0, %5\n"
                       "  %25 = ashr.i32 %0, %5\n"
                       "  %26 = not.i32 %0\n"
                       "  %27 = clz.i32 %0\n"
                       "  %28 = ctz.i32 %0\n"
                       "  %29 = popcnt.i32 %0\n"
                       "  %30 = eq.i32 %0, %5\n"
                       "  %31 = ne.i32 %0, %5\n"
                       "  %32 = slt.i32 %0, %5\n"
                       "  %33 = ult.i32 %0, %5\n"
                       "  %34 = sle.i32 %0, %5\n"
                       "  %35 = ule.i32 %0, %5\n"
                       "  %36 = sgt.i32 %0, %5\n"
                       "  %37 = ugt.i32 %0, %5\n"
                       "  %38 = sge.i32 %0, %5\n"
                       "  %39 = uge.i32 %0, %5\n"
                       "  %40 = add.f64 %2, %8\n"
                       "  ret %40\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 5: Memory loads and stores (offset, indexed scale 1, 2, 4, 8)") {
    std::string text = "module @memory_module\n\n"
                       "func @test_memory(%0: ptr, %1: gcref, %2: i64) -> void {\n"
                       "bb0:\n"
                       "  %3 = load.i32 %0\n"
                       "  %4 = load.i32 %0, 16\n"
                       "  store.i32 %0, %3\n"
                       "  store.i32 %0, 24, %4\n"
                       "  %5 = load_indexed.i32 %1, %2, 1\n"
                       "  %6 = load_indexed.i32 %1, %2, 2\n"
                       "  %7 = load_indexed.i32 %1, %2, 4, 8\n"
                       "  %8 = load_indexed.i64 %1, %2, 8, 32\n"
                       "  store_indexed.i32 %1, %2, 1, %3\n"
                       "  store_indexed.i32 %1, %2, 2, %3\n"
                       "  store_indexed.i32 %1, %2, 4, 8, %7\n"
                       "  store_indexed.i64 %1, %2, 8, 32, %8\n"
                       "  ret\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 6: Speculation with guards, state maps, and resume tables") {
    std::string text = "module @spec_module\n\n"
                       "func @test_spec(%0: i32, %1: ptr) -> i32 {\n"
                       "bb0:\n"
                       "  %2 = iconst.i32 100\n"
                       "  %3 = slt.i32 %0, %2\n"
                       "  guard %3, @deopt_stub_0, [%0, %1]\n"
                       "  safepoint\n"
                       "  br bb_fast\n\n"
                       "bb_fast:\n"
                       "  %4 = iconst.i32 1\n"
                       "  %5 = add.i32 %0, %4\n"
                       "  ret %5\n\n"
                       "resume_table {\n"
                       "  entry 0 -> bb_fast\n"
                       "}\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 7: Patchable constants, patchable calls, and indirect calls") {
    std::string text = "module @patchable_module\n\n"
                       "extern @runtime_helper\n"
                       "extern @fallback_target\n\n"
                       "func @test_patchable(%0: ptr, %1: i32) -> i32 {\n"
                       "bb0:\n"
                       "  %2 = patchable_const.i32 @ic_site_1, 42\n"
                       "  %3 = patchable_call.i32 @ic_call_site_1, @runtime_helper(%2, %1)\n"
                       "  %4 = call_indirect.i32 %0(%3)\n"
                       "  ret %4\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Roundtrip Golden 8: Multi-function module with extern declarations") {
    std::string text = "module @multi_fn_module\n\n"
                       "extern @malloc\n"
                       "extern @free\n\n"
                       "func @alloc_buffer(%0: i64) -> ptr {\n"
                       "bb0:\n"
                       "  %1 = call.ptr @malloc(%0)\n"
                       "  ret %1\n"
                       "}\n\n"
                       "func @release_buffer(%0: ptr) -> void {\n"
                       "bb0:\n"
                       "  call @free(%0)\n"
                       "  ret\n"
                       "}\n";
    test_roundtrip(text);
}

TEST_CASE("Parser negative error cases with actionable diagnostics") {
    // 1. Bad top-level syntax
    {
        DiagnosticReporter diag;
        auto mod = parse_module("random_junk %0", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Unexpected top-level token") != std::string::npos);
    }

    // 2. Unclosed brace in function
    {
        DiagnosticReporter diag;
        auto mod = parse_module("func @bad() -> void {\nbb0:\n ret\n", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Expected '}'") != std::string::npos);
    }

    // 3. Undefined SSA value
    {
        DiagnosticReporter diag;
        auto mod = parse_module("func @bad(%0: i32) -> i32 {\nbb0:\n %1 = add.i32 %0, %nonexistent\n ret %1\n}\n", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Use of undefined value") != std::string::npos);
    }

    // 4. Missing colon after block label
    {
        DiagnosticReporter diag;
        auto mod = parse_module("func @bad() -> void {\nbb0\n ret\n}\n", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Expected ':'") != std::string::npos);
    }

    // 5. Unknown opcode
    {
        DiagnosticReporter diag;
        auto mod = parse_module("func @bad() -> void {\nbb0:\n not_a_real_opcode\n ret\n}\n", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Unrecognized opcode") != std::string::npos);
    }

    // 6. Missing symbol in patchable const
    {
        DiagnosticReporter diag;
        auto mod = parse_module("func @bad() -> i32 {\nbb0:\n %0 = patchable_const.i32 42\n ret %0\n}\n", &diag);
        CHECK(mod == nullptr);
        CHECK(diag.has_errors());
        CHECK(diag.format_all().find("Expected symbol identifier") != std::string::npos);
    }
}
