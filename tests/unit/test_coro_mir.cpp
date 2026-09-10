#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/runtime/coroutine.hpp>
#include <sstream>

using namespace brass;

TEST_CASE("Coroutine MIR - Opcode Predicates") {
    CHECK(is_coro_op(Opcode::coro_create));
    CHECK(is_coro_op(Opcode::coro_suspend));
    CHECK(is_coro_op(Opcode::coro_resume));
    CHECK(is_coro_op(Opcode::coro_destroy));
    CHECK(!is_coro_op(Opcode::add));

    CHECK(is_coro_suspend(Opcode::coro_suspend));
    CHECK(!is_coro_suspend(Opcode::coro_resume));

    CHECK(is_coro_resume(Opcode::coro_resume));
    CHECK(!is_coro_resume(Opcode::coro_suspend));

    CHECK(has_side_effects(Opcode::coro_create));
    CHECK(has_side_effects(Opcode::coro_suspend));
    CHECK(has_side_effects(Opcode::coro_resume));
    CHECK(has_side_effects(Opcode::coro_destroy));
}

TEST_CASE("Coroutine MIR - Builder & Verifier") {
    Module mod("coro_builder_mod");
    Builder b(mod);

    // coro_target: function with coro_suspend
    Function* target = mod.create_function("coro_target", Type::i64(), {Type::gcref()});
    b.set_function(target);
    BasicBlock* t_entry = b.append_block("entry");
    b.position_at_end(t_entry);
    b.add_block_param(t_entry, Type::gcref());
    Value* yield_val = b.build_iconst_i64(42);
    Value* res = b.build_coro_suspend(yield_val, 1, Type::i64());
    Value* final_res = b.build_add(yield_val, res);
    b.build_ret(final_res);

    // main: creates and resumes coro
    Function* main_fn = mod.create_function("main", Type::i64(), {});
    b.set_function(main_fn);
    BasicBlock* m_entry = b.append_block("entry");
    b.position_at_end(m_entry);
    Value* coro = b.build_coro_create("coro_target", {});
    Value* r1 = b.build_coro_resume(coro, b.build_iconst_i64(0), Type::i64());
    b.build_coro_destroy(coro);
    b.build_ret(r1);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("Coroutine MIR - Printer & Parser Roundtrip") {
    Module mod("coro_roundtrip_mod");
    Builder b(mod);

    Function* coro_fn = mod.create_function("my_coro", Type::i64(), {});
    b.set_function(coro_fn);
    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);
    Value* c10 = b.build_iconst_i64(10);
    Value* s1 = b.build_coro_suspend(c10, 1, Type::i64());
    b.build_ret(s1);

    Function* caller_fn = mod.create_function("caller", Type::i64(), {});
    b.set_function(caller_fn);
    BasicBlock* c_bb = b.append_block("entry");
    b.position_at_end(c_bb);
    Value* inst_coro = b.build_coro_create("my_coro", {});
    Value* r = b.build_coro_resume(inst_coro, b.build_iconst_i64(5), Type::i64());
    b.build_coro_destroy(inst_coro);
    b.build_ret(r);

    std::string printed = to_string(mod);

    CHECK(printed.find("coro_suspend") != std::string::npos);
    CHECK(printed.find("coro_create") != std::string::npos);
    CHECK(printed.find("coro_resume") != std::string::npos);
    CHECK(printed.find("coro_destroy") != std::string::npos);

    DiagnosticReporter diag;
    auto parsed_mod = parse_module(printed, &diag);
    CHECK(parsed_mod != nullptr);
    CHECK(!diag.has_errors());

    DiagnosticReporter v_diag;
    CHECK(verify_module(*parsed_mod, &v_diag));
}

TEST_CASE("Coroutine MIR - Reference Interpreter") {
    Module mod("coro_interp_mod");
    Builder b(mod);

    // Generator counting: yields 100, then yields 200, returns 300
    Function* gen_fn = mod.create_function("gen_test", Type::i64(), {Type::gcref()});
    b.set_function(gen_fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());
    b.build_coro_suspend(b.build_iconst_i64(100), 1, Type::i64());
    b.build_coro_suspend(b.build_iconst_i64(200), 2, Type::i64());
    b.build_ret(b.build_iconst_i64(300));

    CoroTransformPass pass;
    CHECK(pass.run_on_module(mod));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));

    Function* target = mod.get_function("gen_test");
    CHECK(target != nullptr);

    uintptr_t frame = brass_coro_create(nullptr, 16, 0);
    CHECK(frame != 0);

    Interpreter interp;

    // First resume: yields 100
    RuntimeValue v1 = interp.run(*target, {RuntimeValue::from_ptr(frame)});
    CHECK(v1.as_i64() == 100);
    CHECK(brass_coro_is_done(frame) == 0);

    // Second resume: yields 200
    RuntimeValue v2 = interp.run(*target, {RuntimeValue::from_ptr(frame)});
    CHECK(v2.as_i64() == 200);
    CHECK(brass_coro_is_done(frame) == 0);

    // Third resume: completes and returns 300
    RuntimeValue v3 = interp.run(*target, {RuntimeValue::from_ptr(frame)});
    CHECK(v3.as_i64() == 300);
    CHECK(brass_coro_is_done(frame) == 1);

    brass_coro_destroy(frame);
}
