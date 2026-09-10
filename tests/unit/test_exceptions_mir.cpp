#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <sstream>

using namespace brass;

TEST_CASE("Exceptions MIR - Opcode Predicates") {
    CHECK(is_terminator(Opcode::throw_));
    CHECK(is_terminator(Opcode::resume));
    CHECK(is_terminator(Opcode::invoke));
    CHECK(!is_terminator(Opcode::landing_pad));

    CHECK(is_branch(Opcode::invoke));
    CHECK(!is_branch(Opcode::throw_));
    CHECK(!is_branch(Opcode::landing_pad));

    CHECK(is_call(Opcode::invoke));
    CHECK(!is_call(Opcode::throw_));

    CHECK(has_side_effects(Opcode::throw_));
    CHECK(has_side_effects(Opcode::resume));
    CHECK(has_side_effects(Opcode::invoke));
    CHECK(has_side_effects(Opcode::landing_pad));
}

TEST_CASE("Exceptions MIR - Verifier Valid Constructs") {
    Module mod("valid_eh");
    Builder b(mod);

    // callee: throws %x
    Function* callee = mod.create_function("callee", Type::i32(), {Type::i32()});
    b.set_function(callee);
    BasicBlock* callee_entry = b.append_block("entry");
    Value* x = b.add_block_param(callee_entry, Type::i32());
    b.build_throw(x);

    // caller: invoke @callee(%arg), normal_bb, unwind_bb
    Function* caller = mod.create_function("caller", Type::i32(), {Type::i32()});
    b.set_function(caller);
    BasicBlock* caller_entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(caller_entry);
    Value* arg = b.add_block_param(caller_entry, Type::i32());
    Instruction* inv = b.build_invoke("callee", Type::i32(), {arg}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* exc_i32 = b.build_trunc_i32(exc);
    b.build_ret(exc_i32);

    caller->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(ok);
}

TEST_CASE("Exceptions MIR - Verifier Rejections") {
    // 1. landing_pad in entry block must be rejected
    {
        Module mod("invalid_entry_lpad");
        Builder b(mod);
        Function* fn = mod.create_function("fn", Type::void_type(), {});
        b.set_function(fn);

        b.append_block("entry");
        b.build_landing_pad(Type::i64());
        b.build_ret_void();

        DiagnosticReporter diag;
        CHECK(!verify_module(mod, &diag));
    }

    // 2. landing_pad not first non-phi in block must be rejected
    {
        Module mod("invalid_lpad_pos");
        Builder b(mod);
        Function* fn = mod.create_function("fn", Type::void_type(), {});
        b.set_function(fn);
        BasicBlock* entry_bb = b.append_block("entry");
        BasicBlock* catch_bb = b.append_block("catch_bb");
        b.position_at_end(entry_bb);
        b.build_br(catch_bb);

        b.position_at_end(catch_bb);
        b.build_iconst_i32(10);
        b.build_landing_pad(Type::i64());
        b.build_ret_void();

        (void)entry_bb;
        DiagnosticReporter diag;
        CHECK(!verify_module(mod, &diag));
    }

    // 3. throw not at terminator position must be rejected
    {
        Module mod("invalid_throw_pos");
        Builder b(mod);
        Function* fn = mod.create_function("fn", Type::void_type(), {});
        b.set_function(fn);
        BasicBlock* entry_bb = b.append_block("entry");
        Value* v = b.build_iconst_i32(42);
        b.build_throw(v);
        b.build_iconst_i32(10); // Dead inst after terminator

        (void)entry_bb;
        DiagnosticReporter diag;
        CHECK(!verify_module(mod, &diag));
    }
}

TEST_CASE("Exceptions MIR - Printer & Parser Roundtrip") {
    Module mod("roundtrip_eh");
    Builder b(mod);

    Function* callee = mod.create_function("callee", Type::i32(), {Type::i32()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    Value* c_x = b.add_block_param(c_entry, Type::i32());
    b.build_throw(c_x);

    Function* caller = mod.create_function("caller", Type::i32(), {Type::i32()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* arg = b.add_block_param(entry, Type::i32());
    Instruction* inv = b.build_invoke("callee", Type::i32(), {arg}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* exc_trunc = b.build_trunc_i32(exc);
    b.build_ret(exc_trunc);

    // Print to string
    std::ostringstream oss;
    print_module(mod, oss);
    std::string mir_text = oss.str();
    CHECK(!mir_text.empty());

    // Parse back
    auto parsed_mod = parse_module(mir_text);
    REQUIRE(parsed_mod != nullptr);

    DiagnosticReporter diag;
    CHECK(verify_module(*parsed_mod, &diag));

    Function* p_caller = parsed_mod->get_function("caller");
    REQUIRE(p_caller != nullptr);
    CHECK_EQ(p_caller->blocks().size(), 3u);
}

TEST_CASE("Exceptions MIR - Interpreter Basic Unwind & Catch") {
    Module mod("interp_eh");
    Builder b(mod);

    // callee: if %x > 5 throw %x, else return %x + 1
    Function* callee = mod.create_function("callee", Type::i32(), {Type::i32()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    BasicBlock* c_throw = b.append_block("c_throw");
    BasicBlock* c_ok = b.append_block("c_ok");

    b.position_at_end(c_entry);
    Value* x = b.add_block_param(c_entry, Type::i32());
    Value* five = b.build_iconst_i32(5);
    Value* cmp = b.build_sgt(x, five);
    b.build_br_if(cmp, c_throw, c_ok);

    b.position_at_end(c_throw);
    b.build_throw(x);

    b.position_at_end(c_ok);
    Value* one = b.build_iconst_i32(1);
    Value* res = b.build_add(x, one);
    b.build_ret(res);

    // caller: invoke @callee(%arg), normal_bb, unwind_bb
    Function* caller = mod.create_function("caller", Type::i32(), {Type::i32()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* arg = b.add_block_param(entry, Type::i32());
    Instruction* inv = b.build_invoke("callee", Type::i32(), {arg}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* exc = b.build_landing_pad(Type::i64());
    Value* exc_trunc = b.build_trunc_i32(exc);
    Value* hundred = b.build_iconst_i32(100);
    Value* caught_res = b.build_add(exc_trunc, hundred);
    b.build_ret(caught_res);

    Interpreter interp;

    // Normal path: arg = 3 -> returns 4
    RuntimeValue r1 = interp.run(*caller, {RuntimeValue::from_i32(3)});
    CHECK_EQ(r1.as_i32(), 4);

    // Exception path: arg = 8 -> throws 8 -> catches -> 8 + 100 = 108
    RuntimeValue r2 = interp.run(*caller, {RuntimeValue::from_i32(8)});
    CHECK_EQ(r2.as_i32(), 108);
}
