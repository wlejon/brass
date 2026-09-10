#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>

using namespace brass;

TEST_CASE("Coroutine Transform - Single Suspend Transformation") {
    Module mod("coro_single_suspend");
    Builder b(mod);

    Function* fn = mod.create_function("simple_coro", Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());
    Value* c1 = b.build_iconst_i64(10);
    Value* c2 = b.build_iconst_i64(20);
    Value* sum = b.build_add(c1, c2);
    Value* res = b.build_coro_suspend(sum, 1, Type::i64());
    Value* final_val = b.build_add(sum, res);
    b.build_ret(final_val);

    CoroTransformStats stats;
    CoroTransformOptions opts;
    opts.stats = &stats;
    CoroTransformPass pass(opts);

    bool changed = pass.run_on_function(*fn);
    CHECK(changed);
    CHECK(stats.coroutines_transformed == 1);
    CHECK(stats.suspend_points_transformed == 1);
    CHECK(stats.variables_spilled >= 1);

    // Verify entry block is the injected dispatch switch
    CHECK(fn->entry_block()->name() == "bb_coro_entry");
    CHECK(fn->entry_block()->tail() != nullptr);
    CHECK(fn->entry_block()->tail()->opcode() == Opcode::switch_);

    // Check resume points registered
    CHECK(fn->resume_points().size() == 1);
    CHECK(fn->resume_points()[0].first == 1);

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cerr << "VERIFY ERROR:\n" << diag.format_all() << "\nMODULE:\n" << to_string(mod) << "\n";
    }
    CHECK(ok);
}

TEST_CASE("Coroutine Transform - Multiple Suspend Points & Spilling") {
    Module mod("coro_multi_suspend");
    Builder b(mod);

    Function* fn = mod.create_function("multi_step", Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* v1 = b.build_iconst_i64(100);
    b.build_coro_suspend(v1, 1, Type::i64());

    Value* v2 = b.build_add(v1, b.build_iconst_i64(50));
    b.build_coro_suspend(v2, 2, Type::i64());

    Value* v3 = b.build_add(v2, b.build_iconst_i64(25));
    b.build_ret(v3);

    CoroTransformStats stats;
    CoroTransformOptions opts;
    opts.stats = &stats;
    CoroTransformPass pass(opts);

    bool changed = pass.run_on_function(*fn);
    CHECK(changed);
    CHECK(stats.coroutines_transformed == 1);
    CHECK(stats.suspend_points_transformed == 2);
    CHECK(fn->resume_points().size() == 2);

    DiagnosticReporter diag;
    bool ok2 = verify_module(mod, &diag);
    if (!ok2) {
        std::cerr << "VERIFY MULTI-SUSPEND ERROR:\n" << diag.format_all() << "\nMODULE:\n" << to_string(mod) << "\n";
    }
    CHECK(ok2);
}

TEST_CASE("Coroutine Transform - Module Pass Idempotency") {
    Module mod("coro_idempotent");
    Builder b(mod);

    Function* normal_fn = mod.create_function("normal_fn", Type::i64(), {});
    b.set_function(normal_fn);
    BasicBlock* b0 = b.append_block("entry");
    b.position_at_end(b0);
    b.build_ret(b.build_iconst_i64(42));

    Function* coro_fn = mod.create_function("coro_fn", Type::i64(), {Type::gcref()});
    b.set_function(coro_fn);
    BasicBlock* b1 = b.append_block("entry");
    b.position_at_end(b1);
    b.add_block_param(b1, Type::gcref());
    b.build_coro_suspend(b.build_iconst_i64(1), 1, Type::i64());
    b.build_ret(b.build_iconst_i64(2));

    CoroTransformPass pass;
    CHECK(pass.run_on_module(mod));
    // Second run should do nothing (already transformed)
    CHECK(!pass.run_on_module(mod));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}
