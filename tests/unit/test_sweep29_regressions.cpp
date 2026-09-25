// Regressions from bug sweep 29:
// - Every guard parsed from text MIR got resume id 0. Tier-2 deopt finds the
//   guard that failed by its resume id, so a failure at a function's second
//   guard ran the first guard's exit. The builder now gives each guard of a
//   function its own id (text may name one with `, id N`), and the verifier
//   rejects two different guards sharing an id.
// - While OSR code ran, one deopt handler per thread treated every guard
//   failure as the OSR'd function's. The OSR module is the whole module
//   compiled again, so its loop calls the module's own copies of its
//   callees; a callee's failure resumed the outer function (its exit stub's
//   result became the whole call's) or threw "No guard with resume id". The
//   handler now resumes the function whose code failed.
#include "test_framework.hpp"
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

std::vector<const Instruction*> guards_of(const Function& fn) {
    std::vector<const Instruction*> out;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst->opcode() == Opcode::guard) out.push_back(inst);
        }
    }
    return out;
}

// @h2(x): guard x < 100 exits to @s1 (3x), then guard x < 50 exits to @s3
// (5x). x = 70 fails only the second guard: 350.
const char* kTwoGuards = R"(module @s29_two
func @s1(%x: i64) -> i64 {
b0:
  %k = iconst.i64 3
  %r = mul.i64 %x, %k
  ret %r
}
func @s3(%x: i64) -> i64 {
b0:
  %k = iconst.i64 5
  %r = mul.i64 %x, %k
  ret %r
}
func @h2(%x: i64) -> i64 {
b0:
  %k1 = iconst.i64 100
  %ok1 = slt.i64 %x, %k1
  guard %ok1, @s1, [%x]
  %k2 = iconst.i64 50
  %ok2 = slt.i64 %x, %k2
  guard %ok2, @s3, [%x]
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
)";

// @f / @f3 loop n times calling @g(i); @g's guard fails for i >= 1000 and
// exits to @gstub (7i). @f also has a guard that never fails (stub 424242).
const char* kOsrCallee = R"(module @s29_osr
func @gstub(%x: i64) -> i64 {
b0:
  %s = iconst.i64 7
  %r = mul.i64 %x, %s
  ret %r
}
func @g(%x: i64) -> i64 {
b0:
  %lim = iconst.i64 1000
  %ok = slt.i64 %x, %lim
  guard %ok, @gstub, [%x]
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
func @fstub(%acc: i64, %i: i64) -> i64 {
b0:
  %r = iconst.i64 424242
  ret %r
}
func @f(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done(%acc)
body:
  %big = iconst.i64 1000000000
  %ok = slt.i64 %i, %big
  guard %ok, @fstub, [%acc, %i]
  %v = call.i64 @g(%i)
  %acc2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %in = add.i64 %i, %one
  br loop(%in, %acc2)
done(%r: i64):
  ret %r
}
func @f3(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, done(%acc)
body:
  %v = call.i64 @g(%i)
  %acc2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %in = add.i64 %i, %one
  br loop(%in, %acc2)
done(%r: i64):
  ret %r
}
)";

// sum_{i<1000} (i + 1) + sum_{1000<=i<2000} 7i
constexpr int64_t kOsrWant = 10997000;

int64_t run_osr_case(const char* fname, bool fast, uint64_t* osr_count) {
    auto mod = parse_ok(kOsrCallee);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(50);
    int64_t got = 0;
    if (fast) {
        FastInterpreter fi;
        fi.set_dispatch_table(&prog);
        fi.set_module(mod.get());
        // The program's OSR code compiles in the background: a first run
        // asks for it, the second enters it.
        CHECK_EQ(fi.run(*mod->get_function(fname), {RuntimeValue::from_i64(2000)}).as_i64(), kOsrWant);
        CompilePool::shared().wait_owner(&prog.osr());
        got = fi.run(*mod->get_function(fname), {RuntimeValue::from_i64(2000)}).as_i64();
    } else {
        Interpreter in;
        in.set_dispatch_table(&prog);
        in.set_module(mod.get());
        got = in.run(*mod->get_function(fname), {RuntimeValue::from_i64(2000)}).as_i64();
    }
    *osr_count = prog.osr().total_osr_migrations();
    return got;
}

} // namespace

TEST_CASE("Sweep29 - text MIR guards get distinct resume ids") {
    auto mod = parse_ok(kTwoGuards);
    auto gs = guards_of(*mod->get_function("h2"));
    REQUIRE_EQ(gs.size(), size_t(2));
    CHECK_EQ(gs[0]->resume_id(), 0u);
    CHECK_EQ(gs[1]->resume_id(), 1u);
}

TEST_CASE("Sweep29 - builder guards get distinct resume ids") {
    Module mod("s29_builder");
    Function* fn = mod.create_function("f", Type::i64(), {Type::i32()});
    Builder b(*fn);
    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);
    Value* c = b.add_param(Type::i32());
    Instruction* g0 = b.build_guard(c, "x");
    Instruction* g1 = b.build_guard(c, "y");
    g1->set_resume_id(7);
    Instruction* g2 = b.build_guard(c, "z");
    CHECK_EQ(g0->resume_id(), 0u);
    CHECK_EQ(g1->resume_id(), 7u);
    CHECK_EQ(g2->resume_id(), 8u);
}

TEST_CASE("Sweep29 - explicit guard ids parse and round-trip through the printer") {
    const char* src = R"(module @s29_ids
func @s1(%x: i64) -> i64 {
b0:
  ret %x
}
func @f(%x: i64) -> i64 {
b0:
  %k = iconst.i64 10
  %ok = slt.i64 %x, %k
  guard %ok, @s1, [%x], id 5
  guard %ok, @s1, [%x]
  guard %ok, @s1, id 2
  ret %x
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    auto gs = guards_of(*mod->get_function("f"));
    REQUIRE_EQ(gs.size(), size_t(3));
    CHECK_EQ(gs[0]->resume_id(), 5u);
    CHECK_EQ(gs[1]->resume_id(), 6u);
    CHECK_EQ(gs[2]->resume_id(), 2u);
    const std::string text = to_string(*mod);
    auto again = parse_module(text, &diag);
    REQUIRE(again != nullptr);
    auto gs2 = guards_of(*again->get_function("f"));
    REQUIRE_EQ(gs2.size(), size_t(3));
    for (size_t i = 0; i < 3; ++i) CHECK_EQ(gs2[i]->resume_id(), gs[i]->resume_id());
}

TEST_CASE("Sweep29 - the verifier rejects two different guards with one resume id") {
    const char* src = R"(module @s29_dup
func @s1(%x: i64) -> i64 {
b0:
  ret %x
}
func @s3(%x: i64) -> i64 {
b0:
  ret %x
}
func @f(%x: i64) -> i64 {
b0:
  %k = iconst.i64 10
  %ok = slt.i64 %x, %k
  guard %ok, @s1, [%x], id 3
  guard %ok, @s3, [%x], id 3
  ret %x
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    CHECK(!verify_module(*mod, &diag));
}

TEST_CASE("Sweep29 - tier-2 deopt at a function's second guard runs that guard's exit") {
    auto mod = parse_ok(kTwoGuards);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(TieringConfig{});
    prog.tiering().get_feedback("h2").set_deopt_threshold(1000000);
    FunctionHandle* h2 = prog.get_or_create("h2", mod->get_function("h2"));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h2, *mod, "h2");
    REQUIRE(res.success);
    Interpreter in;
    in.set_dispatch_table(&prog);
    in.set_module(mod.get());
    CHECK_EQ(in.run(*mod->get_function("h2"), {RuntimeValue::from_i64(10)}).as_i64(), 11);
    CHECK_EQ(in.run(*mod->get_function("h2"), {RuntimeValue::from_i64(70)}).as_i64(), 350);
    CHECK_EQ(in.run(*mod->get_function("h2"), {RuntimeValue::from_i64(120)}).as_i64(), 360);
}

TEST_CASE("Sweep29 - a callee's guard failing in OSR code resumes the callee (Interpreter)") {
    uint64_t osr = 0;
    CHECK_EQ(run_osr_case("f", false, &osr), kOsrWant);
    CHECK(osr > 0);
    CHECK_EQ(run_osr_case("f3", false, &osr), kOsrWant);
    CHECK(osr > 0);
}

TEST_CASE("Sweep29 - a callee's guard failing in OSR code resumes the callee (FastInterpreter)") {
    uint64_t osr = 0;
    CHECK_EQ(run_osr_case("f", true, &osr), kOsrWant);
    CHECK(osr > 0);
    CHECK_EQ(run_osr_case("f3", true, &osr), kOsrWant);
    CHECK(osr > 0);
}
