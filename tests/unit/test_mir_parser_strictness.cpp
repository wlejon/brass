// Textual MIR edge cases from bug sweep 6: integer literals that do not fit
// their field, switch case values outside the condition's type, the bare
// `sitofp.f64` on an i64, and uses that textually precede a dominating
// definition.

#include "test_framework.hpp"
#include <brass/brass.hpp>

#include <sstream>
#include <string>

using namespace brass;

namespace {

// Parses `text`; on failure returns null and leaves the messages in `errors`.
std::unique_ptr<Module> parse_text(std::string_view text, std::string& errors) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag, "strict.mir");
    errors = diag.format_all();
    if (diag.has_errors()) return nullptr;
    return mod;
}

bool parse_fails_with(std::string_view text, std::string_view needle) {
    std::string errors;
    auto mod = parse_text(text, errors);
    return !mod && errors.find(needle) != std::string::npos;
}

bool verifies(const Module& mod, std::string* errors = nullptr) {
    DiagnosticReporter diag;
    const bool ok = verify_module(mod, &diag) && !diag.has_errors();
    if (errors) *errors = diag.format_all();
    return ok;
}

std::string print(const Module& mod) {
    std::ostringstream os;
    print_module(mod, os);
    return os.str();
}

std::string iconst_fn(std::string_view type, std::string_view literal) {
    return "func @f() -> " + std::string(type) + " {\nbb0:\n  %0 = iconst." + std::string(type) + " " +
           std::string(literal) + "\n  ret %0\n}\n";
}

const Instruction* first_inst(const Module& mod, std::string_view fn_name) {
    const Function* fn = mod.get_function(fn_name);
    if (!fn || fn->blocks().empty()) return nullptr;
    return fn->blocks().front()->head();
}

} // namespace

TEST_CASE("MIR parser: iconst literals must fit their type") {
    const char* out_of_range[][2] = {
        {"i32", "2147483648"},
        {"i32", "-2147483649"},
        {"i32", "4294967295"},   // unsigned decimal spelling: the printer writes -1
        {"i32", "4294967296"},
        {"i32", "0x100000000"},
        {"i64", "9223372036854775808"},
        {"i64", "-9223372036854775809"},
        {"i64", "18446744073709551615"},
        {"i64", "99999999999999999999999999"},
        {"i64", "0x10000000000000000"},
    };
    for (const auto& c : out_of_range) {
        CHECK(parse_fails_with(iconst_fn(c[0], c[1]), "out of range for " + std::string(c[0])));
    }

    struct Ok { const char* type; const char* literal; int64_t value; };
    const Ok in_range[] = {
        {"i32", "2147483647", 2147483647},
        {"i32", "-2147483648", INT32_MIN},
        {"i32", "0xFFFFFFFF", -1},          // hex spells the 32-bit pattern
        {"i32", "0x80000000", INT32_MIN},
        {"i32", "-0x80000000", INT32_MIN},
        {"i64", "9223372036854775807", INT64_MAX},
        {"i64", "-9223372036854775808", INT64_MIN},
        {"i64", "0xFFFFFFFFFFFFFFFF", -1},
        {"i64", "0x8000000000000000", INT64_MIN},
    };
    for (const Ok& c : in_range) {
        std::string errors;
        auto mod = parse_text(iconst_fn(c.type, c.literal), errors);
        REQUIRE(mod != nullptr);
        const Instruction* inst = first_inst(*mod, "f");
        REQUIRE(inst != nullptr);
        if (std::string_view(c.type) == "i32") {
            CHECK_EQ(static_cast<int64_t>(inst->imm_i32()), c.value);
        } else {
            CHECK_EQ(inst->imm_i64(), c.value);
        }
        // What the printer writes parses back to the same text.
        const std::string printed = print(*mod);
        auto again = parse_text(printed, errors);
        REQUIRE(again != nullptr);
        CHECK_EQ(print(*again), printed);
    }
}

TEST_CASE("MIR parser: offsets, immediates and case values must fit their field") {
    CHECK(parse_fails_with("func @f(%0: ptr) -> i64 {\nbb0:\n  %1 = load.i64 %0, 99999999999\n  ret %1\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f(%0: ptr, %1: i64) -> void {\nbb0:\n  store.i64 %0, 2147483648, %1\n  ret\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f(%0: ptr, %1: i64) -> i64 {\nbb0:\n"
                           "  %2 = load_indexed.i64 %0, %1, 256\n  ret %2\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f() -> ptr {\nbb0:\n  %0 = alloca 4294967296, 8\n  ret %0\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f() -> ptr {\nbb0:\n  %0 = alloca -1, 8\n  ret %0\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f() -> f64 {\nbb0:\n  %0 = fconst.f64 99999999999999999999999\n  ret %0\n}\n",
                           "out of range"));
    // switch cases are checked against the condition's type.
    CHECK(parse_fails_with("func @f(%0: i32) -> i32 {\nbb0:\n  switch.i32 %0, default: bb1, [4294967297: bb1]\n"
                           "bb1:\n  ret %0\n}\n",
                           "out of range"));
    CHECK(parse_fails_with("func @f(%0: i64) -> i64 {\nbb0:\n"
                           "  switch.i64 %0, default: bb1, [99999999999999999999: bb1]\nbb1:\n  ret %0\n}\n",
                           "out of range"));
    // In range: the extreme negative offset and an i64 case beyond 32 bits.
    std::string errors;
    auto mod = parse_text("func @f(%0: ptr, %1: i64) -> i64 {\nbb0:\n  %2 = load.i64 %0, -2147483648\n"
                          "  switch.i64 %1, default: bb1, [4294967297: bb1, -9223372036854775808: bb1]\n"
                          "bb1:\n  ret %2\n}\n", errors);
    REQUIRE(mod != nullptr);
    CHECK(verifies(*mod));
}

TEST_CASE("MIR verifier: switch case values must fit the condition and be distinct") {
    auto build = [](Type cond_t, std::initializer_list<int64_t> values, std::string& errors) {
        Module mod("sw");
        Builder b(mod);
        Function* fn = mod.create_function("f", Type::i32(), {cond_t});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* dflt = b.append_block("dflt");
        BasicBlock* hit = b.append_block("hit");
        b.position_at_end(entry);
        Value* cond = b.add_block_param(entry, cond_t);
        std::vector<SwitchCase> cases;
        for (int64_t v : values) cases.push_back(SwitchCase(v, hit));
        b.build_switch(cond, dflt, Span<const SwitchCase>(cases.data(), cases.size()));
        b.position_at_end(dflt);
        b.build_ret(b.build_iconst_i32(0));
        b.position_at_end(hit);
        b.build_ret(b.build_iconst_i32(1));
        return verifies(mod, &errors);
    };
    std::string errors;
    CHECK(build(Type::i32(), {INT32_MIN, -1, 0, INT32_MAX}, errors));
    CHECK(build(Type::i64(), {INT64_MIN, int64_t{1} << 32, INT64_MAX}, errors));

    CHECK_FALSE(build(Type::i32(), {int64_t{1} << 32 | 1}, errors));
    CHECK(errors.find("does not fit the i32 condition") != std::string::npos);
    CHECK_FALSE(build(Type::i32(), {int64_t{INT32_MIN} - 1}, errors));
    CHECK_FALSE(build(Type::i32(), {0xFFFFFFFFll}, errors));
    CHECK_FALSE(build(Type::i8(), {128}, errors));
    CHECK_FALSE(build(Type::i32(), {1, 2, 1}, errors));
    CHECK(errors.find("duplicate case value 1") != std::string::npos);
}

TEST_CASE("MIR verifier: a parsed switch with duplicate cases is rejected") {
    std::string errors;
    auto mod = parse_text("func @f(%0: i64) -> i64 {\nbb0:\n  switch.i64 %0, default: bb1, [1: bb2, 1: bb1]\n"
                          "bb1:\n  ret %0\nbb2:\n  ret %0\n}\n", errors);
    REQUIRE(mod != nullptr);
    CHECK_FALSE(verifies(*mod, &errors));
    CHECK(errors.find("duplicate case value 1") != std::string::npos);
}

TEST_CASE("MIR parser: bare sitofp takes its source width from the operand") {
    std::string errors;
    auto mod = parse_text("func @f(%0: i64, %1: i32) -> f64 {\nbb0:\n"
                          "  %2 = sitofp.f64 %0\n  %3 = sitofp.f64 %1\n"
                          "  %4 = sitofp.f32 %0\n  %5 = sitofp.f32 %1\n"
                          "  %6 = fpext.f64.f32 %4\n  %7 = fpext.f64.f32 %5\n"
                          "  %8 = add.f64 %2, %3\n  %9 = add.f64 %8, %6\n  %10 = add.f64 %9, %7\n"
                          "  ret %10\n}\n", errors);
    REQUIRE(mod != nullptr);
    CHECK(verifies(*mod, &errors));
    const Instruction* inst = first_inst(*mod, "f");
    REQUIRE(inst != nullptr);
    CHECK(inst->opcode() == Opcode::sitofp_f64_i64);
    CHECK(inst->next()->opcode() == Opcode::sitofp_f64_i32);
    CHECK(inst->next()->next()->opcode() == Opcode::sitofp_f32_i64);
    CHECK(inst->next()->next()->next()->opcode() == Opcode::sitofp_f32_i32);

    Interpreter interp;
    interp.set_module(mod.get());
    RuntimeValue r = interp.run(*mod->get_function("f"),
                                {RuntimeValue::from_i64(int64_t{1} << 40), RuntimeValue::from_i32(-3)});
    CHECK_EQ(r.as_f64(), 2.0 * static_cast<double>(int64_t{1} << 40) - 6.0);

    // An explicit source suffix is kept as written (and the verifier
    // reports the mismatch).
    auto explicit_mod = parse_text("func @g(%0: i64) -> f64 {\nbb0:\n  %1 = sitofp.f64.i32 %0\n  ret %1\n}\n", errors);
    REQUIRE(explicit_mod != nullptr);
    CHECK_FALSE(verifies(*explicit_mod));
}

TEST_CASE("MIR parser: a use may precede its dominating definition in the text") {
    const char* text =
        "func @main() -> i64 {\n"
        "bb0:\n"
        "  %1 = iconst.i64 5\n"
        "  br bb2\n"
        "\n"
        "bb1:\n"
        "  %3 = add.i64 %2, %1\n"
        "  %4 = mul.i64 %3, %5\n"
        "  ret %4\n"
        "\n"
        "bb3(%6: i64):\n"
        "  %5 = add.i64 %6, %2\n"
        "  br bb1\n"
        "\n"
        "bb2:\n"
        "  %2 = iconst.i64 6\n"
        "  br bb3(%1)\n"
        "}\n";
    std::string errors;
    auto mod = parse_text(text, errors);
    REQUIRE(mod != nullptr);
    CHECK(verifies(*mod, &errors));

    // Block layout stays in text order.
    const Function* fn = mod->get_function("main");
    REQUIRE(fn != nullptr);
    REQUIRE_EQ(fn->blocks().size(), size_t{4});
    CHECK_EQ(fn->blocks()[1]->name(), std::string_view("bb1"));
    CHECK_EQ(fn->blocks()[2]->name(), std::string_view("bb3"));

    Interpreter interp;
    interp.set_module(mod.get());
    RuntimeValue r = interp.run(*mod->get_function("main"), {});
    CHECK_EQ(r.as_i64(), (6 + 5) * (5 + 6));

    // The printed form parses back to itself.
    const std::string printed = print(*mod);
    auto again = parse_text(printed, errors);
    REQUIRE(again != nullptr);
    CHECK_EQ(print(*again), printed);
}

TEST_CASE("MIR parser: forward references still reject undefined names, dominance is left to the verifier") {
    // A name nothing defines.
    CHECK(parse_fails_with("func @f() -> i64 {\nbb0:\n  br bb1\nbb1:\n  %1 = add.i64 %9, %9\n  ret %1\n}\n",
                           "Use of undefined value: '%9'"));
    // A use before its definition in the same block never resolves.
    CHECK(parse_fails_with("func @f() -> i64 {\nbb0:\n  %1 = add.i64 %2, %2\n  %2 = iconst.i64 1\n  ret %1\n}\n",
                           "Use of undefined value: '%2'"));
    // A definition in a block that does not dominate the use parses, and
    // the verifier rejects it.
    std::string errors;
    auto mod = parse_text("func @f(%0: i32) -> i64 {\nbb0:\n  br_if %0, bb1, bb2\n"
                          "bb1:\n  ret %5\n"
                          "bb2:\n  %5 = iconst.i64 7\n  ret %5\n}\n", errors);
    REQUIRE(mod != nullptr);
    CHECK_FALSE(verifies(*mod));
}
