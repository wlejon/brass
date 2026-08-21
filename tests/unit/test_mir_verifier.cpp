#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>

using namespace brass;

TEST_CASE("Verifier rejects empty function") {
    Module mod("empty_fn_mod");
    mod.create_function("empty", Type::void_type());

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("has no basic blocks") != std::string::npos);
}

TEST_CASE("Verifier rejects entry block with predecessors") {
    Module mod("entry_pred_mod");
    Function* fn = mod.create_function("loop_entry", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.build_br(entry); // branch back to entry

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("must have no predecessors") != std::string::npos);
}

TEST_CASE("Verifier rejects entry block parameter mismatch") {
    Module mod("entry_param_mod");
    Function* fn = mod.create_function("mismatch_params", Type::void_type(), {Type::i32(), Type::f64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.add_block_param(entry, Type::i32());
    b.add_block_param(entry, Type::i64()); // Expected f64
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("does not match signature") != std::string::npos);
}

TEST_CASE("Verifier rejects block missing terminator") {
    Module mod("missing_term_mod");
    Function* fn = mod.create_function("no_term", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    b.build_iconst_i32(42);
    // No ret or branch

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("does not end with a valid terminator") != std::string::npos);
}

TEST_CASE("Verifier rejects terminator not at end of block") {
    Module mod("term_middle_mod");
    Function* fn = mod.create_function("term_mid", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    b.build_ret_void();
    b.build_iconst_i32(10); // Instruction after return!

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("not at the end of the block") != std::string::npos);
}

TEST_CASE("Verifier rejects branch argument count mismatch") {
    Module mod("branch_arg_count_mod");
    Function* fn = mod.create_function("bad_branch_count", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* target = b.append_block("target");
    b.add_block_param(target, Type::i32());
    b.add_block_param(target, Type::i64());
    b.position_at_end(target);
    b.build_ret_void();

    b.position_at_end(entry);
    Value* c1 = b.build_iconst_i32(1);
    b.build_br(target, {c1}); // target expects 2 arguments, passes 1

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("expects 2") != std::string::npos);
}

TEST_CASE("Verifier rejects branch argument type mismatch") {
    Module mod("branch_arg_type_mod");
    Function* fn = mod.create_function("bad_branch_type", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* target = b.append_block("target");
    b.add_block_param(target, Type::f64());
    b.position_at_end(target);
    b.build_ret_void();

    b.position_at_end(entry);
    Value* c1 = b.build_iconst_i32(1); // i32 passed to f64 parameter
    b.build_br(target, {c1});

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("does not match target parameter") != std::string::npos);
}

TEST_CASE("Verifier rejects br_if condition of non-i32 type") {
    Module mod("br_if_cond_mod");
    Function* fn = mod.create_function("bad_br_if", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    BasicBlock* bb1 = b.append_block("bb1");
    BasicBlock* bb2 = b.append_block("bb2");
    b.position_at_end(bb1); b.build_ret_void();
    b.position_at_end(bb2); b.build_ret_void();

    b.position_at_end(fn->entry_block());
    Value* bad_cond = b.build_iconst_i64(1); // i64 instead of i32
    b.build_br_if(bad_cond, bb1, bb2);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("br_if condition must be i32") != std::string::npos);
}

TEST_CASE("Verifier rejects SSA dominance violation inside same block") {
    Module mod("ssa_same_block_mod");
    Function* fn = mod.create_function("use_before_def", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");

    // Manually create an add instruction that uses a value created afterwards
    Value* future_val = b.create_value(Type::i32());

    // instruction using future_val before it is defined
    Instruction* add_inst = mod.arena().make<Instruction>(Opcode::add, Type::i32());
    add_inst->add_operand(future_val);
    add_inst->add_operand(future_val);
    Value* res = b.create_value(Type::i32());
    res->set_defining_instruction(add_inst);
    add_inst->set_result(res);
    b.insert(add_inst);

    // now define future_val afterwards
    Instruction* c_inst = mod.arena().make<Instruction>(Opcode::iconst_i32, Type::i32());
    c_inst->set_imm_i32(10);
    future_val->set_defining_instruction(c_inst);
    c_inst->set_result(future_val);
    b.insert(c_inst);

    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("used before or at its definition") != std::string::npos);
}

TEST_CASE("Verifier rejects SSA dominance violation across parallel branches") {
    Module mod("ssa_cross_branch_mod");
    Function* fn = mod.create_function("cross_branch_use", Type::i32());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    BasicBlock* then_bb = b.append_block("then_bb");
    BasicBlock* else_bb = b.append_block("else_bb");
    BasicBlock* merge_bb = b.append_block("merge_bb");

    // entry branches to then or else
    b.position_at_end(fn->entry_block());
    Value* cond = b.build_iconst_i32(1);
    b.build_br_if(cond, then_bb, else_bb);

    // then defines a value
    b.position_at_end(then_bb);
    Value* then_val = b.build_iconst_i32(100);
    b.build_br(merge_bb);

    // else branches to merge
    b.position_at_end(else_bb);
    b.build_br(merge_bb);

    // merge illegally uses then_val which does not dominate merge!
    b.position_at_end(merge_bb);
    Value* one = b.build_iconst_i32(1);
    Value* bad_sum = b.build_add(then_val, one);
    b.build_ret(bad_sum);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("does not dominate use") != std::string::npos);
}

TEST_CASE("Verifier rejects binary arithmetic type mismatch") {
    Module mod("bad_bin_types_mod");
    Function* fn = mod.create_function("bad_add", Type::i32());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    Value* a = b.build_iconst_i32(10);
    Value* c = b.build_iconst_i64(20);

    // add i32 and i64
    Instruction* bad_add = mod.arena().make<Instruction>(Opcode::add, Type::i32());
    bad_add->add_operand(a);
    bad_add->add_operand(c);
    Value* res = b.create_value(Type::i32());
    res->set_defining_instruction(bad_add);
    bad_add->set_result(res);
    b.insert(bad_add);

    b.build_ret(res);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("Operand types mismatch") != std::string::npos);
}

TEST_CASE("Verifier rejects return type mismatch") {
    Module mod("bad_ret_type_mod");
    Function* fn = mod.create_function("bad_ret", Type::i32());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    Value* f_val = b.build_fconst_f64(3.14);
    b.build_ret(f_val); // returns f64 when function returns i32

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("does not match function return type") != std::string::npos);
}

TEST_CASE("Verifier rejects invalid memory operations") {
    Module mod("bad_mem_mod");
    Function* fn = mod.create_function("bad_mem", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    b.append_block("entry");
    Value* non_ptr = b.build_iconst_i32(1234);

    // Load with non-pointer base
    b.build_load(Type::i32(), non_ptr, 0);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.has_errors());
    CHECK(diag.format_all().find("Load base must be ptr or gcref") != std::string::npos);
}

TEST_CASE("Verifier checks resume points validity") {
    Module mod("resume_points_mod");
    Function* fn = mod.create_function("resume_fn", Type::void_type());
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* resume_target = b.append_block("resume_target");

    b.position_at_end(entry);
    b.build_br(resume_target);

    b.position_at_end(resume_target);
    b.build_resume_point(10);
    b.build_ret_void();

    fn->add_resume_point(10, resume_target);
    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(ok);

    // Now set an invalid resume point with null block
    fn->add_resume_point(11, nullptr);
    DiagnosticReporter bad_diag;
    bool bad_ok = verify_module(mod, &bad_diag);
    CHECK(!bad_ok);
    CHECK(bad_diag.format_all().find("null target block") != std::string::npos);
}

TEST_CASE("Verifier rejects duplicate function names in module") {
    Module mod("dup_fn_mod");
    mod.create_function("foo", Type::void_type());
    mod.create_function("foo", Type::void_type()); // Duplicate name

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    CHECK(!ok);
    CHECK(diag.format_all().find("Duplicate function name") != std::string::npos);
}
