// A frame in OSR code whose code is invalidated leaves it. The OSR copy of a
// loop carries the bodies of what the loop calls; a guard failing in one of
// those finishes only the callee in Tier 0 and returns into the loop, so once
// the callee's speculation is wrong every iteration deoptimized: in bronze a
// 20,000-iteration loop took about 20,000 deopts of 15 us each in one frame.
// The code now polls a flag at each backedge to its header, set when the
// entry is invalidated, and hands the loop back to the interpreter there.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// @ol_slow(%x) = %x * 100
// @ol_callee(%x, %flip): %a = %x * 3; guard %x < %flip (resume 7, state
//   [%a, %x]); ret %a + 1; resume 7: ret %a + @ol_slow(%x)
// @ol_loop(%n, %flip): for i < n (with an i32 count %k beside it):
//   acc += @ol_callee(i, flip); ret acc + %k
void build_program(Module& mod) {
    Function* slow = mod.create_function("ol_slow", Type::i64(), {Type::i64()});
    {
        Builder b(*slow);
        b.position_at_end(b.append_block("entry"));
        Value* x = b.add_param(Type::i64());
        b.build_ret(b.build_mul(x, b.build_iconst_i64(100)));
    }
    Function* callee = mod.create_function("ol_callee", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(*callee);
        b.position_at_end(b.append_block("entry"));
        Value* x = b.add_param(Type::i64());
        Value* flip = b.add_param(Type::i64());
        Value* a = b.build_mul(x, b.build_iconst_i64(3));
        Instruction* g = b.build_guard(b.build_slt(x, flip), "ol_no_stub", {a, x});
        g->set_resume_id(7);
        b.build_ret(b.build_add(a, b.build_iconst_i64(1)));
        BasicBlock* resume = b.append_block("resume7");
        b.position_at_end(resume);
        Value* ra = b.add_param(Type::i64());
        Value* rx = b.add_param(Type::i64());
        b.build_ret(b.build_add(ra, b.build_call("ol_slow", Type::i64(), {rx})));
        callee->add_resume_point(7, resume);
    }
    Function* loop_fn = mod.create_function("ol_loop", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(*loop_fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop = b.append_block("loop");
        BasicBlock* done = b.append_block("done");
        b.position_at_end(entry);
        Value* n = b.add_param(Type::i64());
        Value* flip = b.add_param(Type::i64());
        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop, {zero, zero, b.build_iconst_i32(0)});

        b.position_at_end(loop);
        Value* i = b.add_block_param(loop, Type::i64());
        Value* acc = b.add_block_param(loop, Type::i64());
        Value* k = b.add_block_param(loop, Type::i32());
        Value* r = b.build_call("ol_callee", Type::i64(), {i, flip});
        Value* acc2 = b.build_add(acc, r);
        Value* k2 = b.build_add(k, b.build_iconst_i32(1));
        Value* i2 = b.build_add(i, b.build_iconst_i64(1));
        b.build_br_if(b.build_slt(i2, n), loop, {i2, acc2, k2}, done, {acc2, k2});

        b.position_at_end(done);
        Value* racc = b.add_block_param(done, Type::i64());
        Value* rk = b.add_block_param(done, Type::i32());
        b.build_ret(b.build_add(racc, b.build_sext_i64(rk)));
    }
}

int64_t expected(int64_t n, int64_t flip) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) acc += i < flip ? 3 * i + 1 : 103 * i;
    return acc + n;
}

void init_program(FunctionDispatchTable& prog) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(100);
}

} // namespace

TEST_CASE("OSR leave - a loop whose carried callee keeps failing leaves the invalidated code") {
    Module mod("osr_leave");
    build_program(mod);
    const Function& fn = *mod.get_function("ol_loop");
    FunctionDispatchTable prog;
    init_program(prog);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(&mod);

    // The first run asks for the loop's OSR code; the second enters it, and
    // from %flip on every call fails the guard in the callee's copy.
    constexpr int64_t kNever = INT64_MAX;
    auto run = [&](int64_t n, int64_t flip) {
        return interp.run(fn, {RuntimeValue::from_i64(n), RuntimeValue::from_i64(flip)}).as_i64();
    };
    CHECK_EQ(run(2000, kNever), expected(2000, kNever));
    CompilePool::shared().wait_owner(&prog.osr());
    REQUIRE_EQ(prog.pipeline().tier2_deopts(), 0u);
    const uint64_t migrations = prog.osr().total_osr_migrations();

    constexpr int64_t kN = 40000;
    constexpr int64_t kFlip = 20000;
    CHECK_EQ(run(kN, kFlip), expected(kN, kFlip));
    REQUIRE(prog.osr().total_osr_migrations() > migrations);

    // The failures invalidate the entry after the threshold's worth; the
    // frame then leaves at the next backedge and finishes interpreted.
    const uint64_t threshold = prog.tiering().default_config().deopt_threshold;
    CHECK(prog.pipeline().tier2_deopts() <= threshold);
    CHECK_EQ(prog.osr().total_osr_leaves(), 1u);

    // The invalid entry is never entered again, and the program still gives
    // the interpreter's answer both ways.
    const uint64_t deopts = prog.pipeline().tier2_deopts();
    CHECK_EQ(run(kN, kFlip), expected(kN, kFlip));
    CHECK_EQ(run(5000, kNever), expected(5000, kNever));
    CHECK_EQ(prog.pipeline().tier2_deopts(), deopts);
}

TEST_CASE("OSR leave - code that is never invalidated finishes the loop itself") {
    Module mod("osr_leave_valid");
    build_program(mod);
    const Function& fn = *mod.get_function("ol_loop");
    FunctionDispatchTable prog;
    init_program(prog);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(&mod);
    constexpr int64_t kNever = INT64_MAX;
    auto run = [&](int64_t n) {
        return interp.run(fn, {RuntimeValue::from_i64(n), RuntimeValue::from_i64(kNever)}).as_i64();
    };
    CHECK_EQ(run(2000), expected(2000, kNever));
    CompilePool::shared().wait_owner(&prog.osr());
    const uint64_t migrations = prog.osr().total_osr_migrations();
    CHECK_EQ(run(50000), expected(50000, kNever));
    CHECK(prog.osr().total_osr_migrations() > migrations);
    CHECK_EQ(prog.osr().total_osr_leaves(), 0u);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 0u);
}

#endif
