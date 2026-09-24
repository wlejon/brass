// Per-program runtime state beyond dispatch and tiering: type feedback, OSR,
// the running program seen by the IL runtime's resolver, and the aarch64
// baseline's invocation hook. Each owned FunctionDispatchTable has its own;
// the default program keeps the process-wide instances.
#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

extern "C" void brass_tier1_record_invocation_fb(void* feedback);

using namespace brass;
using namespace brass::runtime;

namespace {

// func @<name>(%n: i64) -> i64: sum of 0..n-1 in a loop.
Function* build_sum_loop(Module& mod, const std::string& name) {
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* sum_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* sum_next = b.build_add(sum_val, i_val);
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, sum_next});

    b.position_at_end(b3);
    b.build_ret(sum_val);
    return fn;
}

// The bytes of `mov x0, #imm` as the aarch64 baseline emits it.
std::vector<uint8_t> mov_x0_bytes(uint64_t imm) {
    aarch64::CodeBuffer buf;
    aarch64::AArch64Encoder enc(buf);
    enc.mov(aarch64::GPR::X0, imm);
    return buf.bytes();
}

bool contains(const uint8_t* code, size_t size, const std::vector<uint8_t>& needle) {
    return std::search(code, code + size, needle.begin(), needle.end()) != code + size;
}

} // namespace

TEST_CASE("Program runtime state - type feedback is per program") {
    FunctionDispatchTable prog_a;
    FunctionDispatchTable prog_b;

    FeedbackRegistry& fa = prog_a.tiering().type_feedback();
    FeedbackRegistry& fb = prog_b.tiering().type_feedback();
    CHECK(&fa != &fb);
    CHECK(&fa != &FeedbackRegistry::instance());
    CHECK(&FunctionDispatchTable::instance().tiering().type_feedback() == &FeedbackRegistry::instance());

    const std::string name = "prs_tfv_fn";
    // Recorded through the program's own TieringFeedback.
    prog_a.tiering().get_feedback(name).type_feedback_vector()->record_call_target(7, 0x1000, "a_target");
    prog_b.tiering().get_feedback(name).type_feedback_vector()->record_call_target(7, 0x2000, "b_target");

    const TypeFeedbackVector* va = fa.find(name);
    const TypeFeedbackVector* vb = fb.find(name);
    REQUIRE(va != nullptr);
    REQUIRE(vb != nullptr);
    REQUIRE(va->find_slot(7) != nullptr);
    REQUIRE(vb->find_slot(7) != nullptr);
    CHECK_EQ(va->find_slot(7)->targets.size(), size_t{1});
    CHECK(va->find_slot(7)->targets[0].target_name == "a_target");
    CHECK(vb->find_slot(7)->targets[0].target_name == "b_target");
    CHECK(FeedbackRegistry::instance().find(name) == nullptr);

    // The default program still records into the global registry.
    TieringRegistry::instance().get_feedback(name).type_feedback_vector()->record_call_target(7, 0x3000, "d");
    CHECK(FeedbackRegistry::instance().find(name) != nullptr);
    CHECK_EQ(fa.find(name)->find_slot(7)->targets.size(), size_t{1});
    FeedbackRegistry::instance().clear();
}

TEST_CASE("Program runtime state - OSR coordinators are per program") {
    // The modules outlive the programs that route into them.
    Module mod_a("prs_osr_a");
    Module mod_b("prs_osr_b");
    FunctionDispatchTable prog_a;
    FunctionDispatchTable prog_b;
    CHECK(&prog_a.osr() != &prog_b.osr());
    CHECK(&prog_a.osr() != &OsrCoordinator::instance());
    CHECK(&FunctionDispatchTable::instance().osr() == &OsrCoordinator::instance());
    CHECK(&prog_a.osr().registry() == &prog_a.tiering());
    CHECK(&OsrCoordinator::instance().registry() == &TieringRegistry::instance());

    bool threw = false;
    try {
        OsrCoordinator bad(TieringRegistry::instance());
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);

    const uint64_t default_threshold = TieringRegistry::instance().default_config().backedge_osr_threshold;
    const uint64_t default_migrations = OsrCoordinator::instance().total_osr_migrations();
    prog_a.osr().set_enabled(true);
    prog_a.osr().set_threshold(50);
    CHECK_EQ(prog_a.tiering().default_config().backedge_osr_threshold, uint64_t{50});
    CHECK_EQ(TieringRegistry::instance().default_config().backedge_osr_threshold, default_threshold);
    CHECK(!OsrCoordinator::instance().is_enabled());

    // Both programs run a same-named loop; only A has OSR on.
    Function* fn_a = build_sum_loop(mod_a, "prs_sum");
    Function* fn_b = build_sum_loop(mod_b, "prs_sum");

    Interpreter interp_a;
    interp_a.set_dispatch_table(&prog_a);
    interp_a.set_module(&mod_a);
    Interpreter interp_b;
    interp_b.set_dispatch_table(&prog_b);
    interp_b.set_module(&mod_b);

    const int64_t n = 500;
    const int64_t expected = (n - 1) * n / 2;
    CHECK_EQ(interp_a.run(*fn_a, {RuntimeValue::from_i64(n)}).as_i64(), expected);
    CHECK_EQ(interp_b.run(*fn_b, {RuntimeValue::from_i64(n)}).as_i64(), expected);

    CHECK(prog_a.osr().total_osr_migrations() > 0);
    CHECK_EQ(prog_b.osr().total_osr_migrations(), uint64_t{0});
    CHECK_EQ(OsrCoordinator::instance().total_osr_migrations(), default_migrations);
    // A's backedges counted into A's registry, not the default program's.
    REQUIRE(prog_a.tiering().find_feedback("prs_sum") != nullptr);
    CHECK(prog_a.tiering().find_feedback("prs_sum")->backedge_count() >= 50);
    CHECK(TieringRegistry::instance().find_feedback("prs_sum") == nullptr);
}

namespace {
// A host runtime's by-name function lookup: the running program's handles
// (the default program's outside any ProgramScope).
void* resolve_in_running_program(const char* name) {
    auto* handle = current_program().find(name);
    return handle ? handle->native_entry() : nullptr;
}
}

TEST_CASE("Program runtime state - a host resolves functions in the running program") {
    Module mod("prs_il_mod"); // outlives the programs that route into it
    FunctionDispatchTable prog_a;
    FunctionDispatchTable prog_b;
    int marker_a = 0;
    int marker_b = 0;
    prog_a.get_or_create("prs_il_fn")->set_native_entry(&marker_a);
    prog_b.get_or_create("prs_il_fn")->set_native_entry(&marker_b);

    CHECK(&current_program() == &FunctionDispatchTable::instance());
    CHECK(resolve_in_running_program("prs_il_fn") == nullptr);
    {
        ProgramScope scope_a(prog_a);
        CHECK(&current_program() == &prog_a);
        CHECK(resolve_in_running_program("prs_il_fn") == &marker_a);
        {
            ProgramScope scope_b(prog_b);
            CHECK(resolve_in_running_program("prs_il_fn") == &marker_b);
        }
        CHECK(resolve_in_running_program("prs_il_fn") == &marker_a);
    }
    CHECK(&current_program() == &FunctionDispatchTable::instance());

    // An interpreter run makes its table the running program.
    Function* fn = mod.create_function("prs_il_main", Type::i64(), {});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    b.build_ret(b.build_call("prs_probe", Type::i64()));

    Interpreter interp;
    interp.set_dispatch_table(&prog_b);
    interp.set_module(&mod);
    void* seen = nullptr;
    interp.register_external_function("prs_probe", [&](Interpreter&, const std::vector<RuntimeValue>&) {
        seen = resolve_in_running_program("prs_il_fn");
        return RuntimeValue::from_i64(1);
    });
    CHECK_EQ(interp.run(*fn, {}).as_i64(), int64_t{1});
    CHECK(seen == &marker_b);
    CHECK(&current_program() == &FunctionDispatchTable::instance());
}

TEST_CASE("Program runtime state - aarch64 baseline bakes in its program's feedback") {
    FunctionDispatchTable prog_a;
    FunctionDispatchTable prog_b;
    Module mod("prs_a64");
    Function* fn = mod.create_function("prs_a64_fn", Type::i64(), {});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    b.build_ret(b.build_iconst_i64(3));

    auto compiled_a = aarch64::compile_baseline_aarch64(*fn, Target::aarch64_linux(), nullptr, &prog_a.tiering());
    auto compiled_d = aarch64::compile_baseline_aarch64(*fn, Target::aarch64_linux());
    REQUIRE(compiled_a.is_valid());
    REQUIRE(compiled_d.is_valid());

    const uint64_t fb_a = reinterpret_cast<uint64_t>(prog_a.tiering().find_feedback("prs_a64_fn"));
    const uint64_t fb_d = reinterpret_cast<uint64_t>(TieringRegistry::instance().find_feedback("prs_a64_fn"));
    REQUIRE(fb_a != 0);
    REQUIRE(fb_d != 0);
    CHECK(prog_b.tiering().find_feedback("prs_a64_fn") == nullptr);

    const auto* code_a = static_cast<const uint8_t*>(compiled_a.entry_point());
    const auto* code_d = static_cast<const uint8_t*>(compiled_d.entry_point());
    CHECK(contains(code_a, compiled_a.code_size(), mov_x0_bytes(fb_a)));
    CHECK(!contains(code_a, compiled_a.code_size(), mov_x0_bytes(fb_d)));
    CHECK(contains(code_d, compiled_d.code_size(), mov_x0_bytes(fb_d)));
    // The hook called is the feedback-pointer form.
    {
        aarch64::CodeBuffer buf;
        aarch64::AArch64Encoder enc(buf);
        enc.mov(aarch64::GPR::X16, reinterpret_cast<uint64_t>(&brass_tier1_record_invocation_fb));
        CHECK(contains(code_a, compiled_a.code_size(), buf.bytes()));
    }
}
