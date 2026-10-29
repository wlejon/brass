#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <string_view>

using namespace brass;
using namespace brass::test;

// for (i = 0; i < 1000; i += s) { acc += (double)(int64)i; if (++trips >= 50) break; }
// with s an int32 from the caller: its range is every int32, so only a loop
// versioned on 1 <= s <= 2^31 can carry i as an integer. Every other s runs
// the copy, as written: zero and negative steps reach the trip cap.
TEST_CASE("F64 Demote - Loop-Invariant Step Versioned On Its Sign, Counter Demoted, Copy Runs Other Steps") {
    Module mod("test_demote_step");
    Builder b(mod);
    Function* fn = mod.create_function("strided", Type::f64(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* s32 = b.add_block_param(b0, Type::i32());
    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");
    BasicBlock* b4 = b.create_block("b4");

    Value* step = b.build_sitofp_f64_i32(s32);
    Value* zero = b.build_fconst_f64(0.0);
    b.build_br(b1, {zero, zero, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* i = b.add_block_param(b1, Type::f64());
    Value* acc = b.add_block_param(b1, Type::f64());
    Value* trips = b.add_block_param(b1, Type::f64());
    b.build_br_if(b.build_slt(i, b.build_fconst_f64(1000.0)), b2, {}, b3, {acc});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* as_int = b.build_sitofp_f64_i64(b.build_fptosi_i64(i));
    Value* acc2 = b.build_add(acc, as_int);
    Value* trips2 = b.build_add(trips, b.build_fconst_f64(1.0));
    b.build_br_if(b.build_sge(trips2, b.build_fconst_f64(50.0)), b3, {acc2}, b4, {});

    fn->append_block(b4);
    b.position_at_end(b4);
    b.build_br(b1, {b.build_add(i, step), acc2, trips2});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* r = b.add_block_param(b3, Type::f64());
    b.build_ret(r);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    f64_demote_pass(*fn, {});
    REQUIRE(verify_function(*fn, &diag));
    CHECK_EQ(b1->param(0)->type(), Type::i64());
    bool has_copy = false;
    for (const BasicBlock* bb : fn->blocks()) {
        if (bb && std::string_view(bb->name()).find("_f64c") != std::string_view::npos) has_copy = true;
    }
    CHECK(has_copy);

    LoopOptOptions opt;
    optimize_function_loops(*fn, opt);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto strided = jit.get_function_ptr<double (*)(int32_t)>("strided");
    REQUIRE(strided != nullptr);
    CHECK_EQ(strided(3), 3675.0);        // 0, 3, ..., 147: the cap
    CHECK_EQ(strided(30), 16830.0);      // 0, 30, ..., 990: the limit
    CHECK_EQ(strided(1), 1225.0);        // 0 .. 49
    CHECK_EQ(strided(0), 0.0);           // the copy: 0 fifty times
    CHECK_EQ(strided(-5), -6125.0);      // the copy: 0, -5, ..., -245
    CHECK_EQ(strided(2147483647), 0.0);  // one trip, then past the limit
    CHECK_EQ(strided(-2147483647 - 1), -2147483648.0 * 1225.0);
}
