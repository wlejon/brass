#include "test_framework.hpp"
#include <brass/brass.hpp>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

TEST_CASE("Differential - Guard Fast Path & Slow Path Exit") {
    Module mod("diff_guard_mod");

    // Twin fallback function:
    // func @twin_fallback(%resume_id: i32, %buf: ptr) -> i64
    // resume_table {
    //   entry 0 -> bb_res_0
    // }
    // bb0:
    //   ret 0
    // bb_res_0:
    //   %x = load.i64 %buf, 0
    //   %y = load.i64 %buf, 8
    //   %res = add %x, %y
    //   ret %res
    Function* twin = mod.create_function("twin_fallback", Type::i64(), {Type::i32(), Type::ptr()});
    Builder tb(*twin);
    BasicBlock* tb0 = tb.append_block("bb0");
    BasicBlock* tb_res = tb.append_block("bb_res_0");
    twin->add_resume_point(0, tb_res);

    tb.position_at_end(tb0);
    tb.add_param(Type::i32());
    tb.add_param(Type::ptr());
    tb.build_ret(tb.build_iconst_i64(0));

    tb.position_at_end(tb_res);
    Value* x = tb.build_load(Type::i64(), twin->entry_block()->param(1), 0);
    Value* y = tb.build_load(Type::i64(), twin->entry_block()->param(1), 8);
    Value* sum = tb.build_add(x, y);
    tb.build_ret(sum);

    // Speculative function:
    // func @spec_fn(%a: i64, %b: i64, %type_tag: i32) -> i64
    //   guard %type_tag, @twin_fallback, [%a, %b]
    //   %mul = mul %a, %b
    //   ret %mul
    Function* spec = mod.create_function("spec_fn", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    Builder sb(*spec);
    BasicBlock* sb_entry = sb.append_block("entry");
    sb.position_at_end(sb_entry);
    Value* sa = sb.add_param(Type::i64());
    Value* sb_param = sb.add_param(Type::i64());
    Value* stype = sb.add_param(Type::i32());

    Instruction* g = sb.build_guard(stype, "twin_fallback", {sa, sb_param});
    g->set_resume_id(0);
    Value* smul = sb.build_mul(sa, sb_param);
    sb.build_ret(smul);

    // Run both on Interpreter and JIT
    Interpreter interp;
    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Case 1: type_tag = 1 (Fast path: a * b)
    for (int64_t a = 1; a <= 5; ++a) {
        for (int64_t b = 1; b <= 5; ++b) {
            std::vector<RuntimeValue> args = {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b), RuntimeValue::from_i32(1)};
            RuntimeValue interp_res = interp.run(*spec, args);
            RuntimeValue jit_res = engine.invoke("spec_fn", args);

            CHECK_EQ(interp_res.as_i64(), a * b);
            CHECK_EQ(jit_res.as_i64(), a * b);
            CHECK_EQ(interp_res.as_i64(), jit_res.as_i64());
        }
    }

    // Case 2: type_tag = 0 (Slow path: deopt into twin at resume_id=0 -> a + b)
    for (int64_t a = 10; a <= 15; ++a) {
        for (int64_t b = 20; b <= 25; ++b) {
            std::vector<RuntimeValue> args = {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b), RuntimeValue::from_i32(0)};

            // For interpreter, custom deopt handler or twin function resolution
            interp.set_module(&mod);
            interp.set_deopt_handler([&](Interpreter& i, const DeoptResult& d) -> RuntimeValue {
                const Function* t_fn = mod.get_function("twin_fallback");
                uint64_t buf[2] = {static_cast<uint64_t>(d.state_map[0].as_i64()), static_cast<uint64_t>(d.state_map[1].as_i64())};
                return i.resume(*t_fn, d.resume_id, {RuntimeValue::from_i32(static_cast<int32_t>(d.resume_id)), RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buf))});
            });

            RuntimeValue interp_res = interp.run(*spec, args);
            RuntimeValue jit_res = engine.invoke("spec_fn", args);

            CHECK_EQ(interp_res.as_i64(), a + b);
            CHECK_EQ(jit_res.as_i64(), a + b);
            CHECK_EQ(interp_res.as_i64(), jit_res.as_i64());
        }
    }
}

TEST_CASE("Differential - Interior Resume Point Execution") {
    Module mod("diff_resume_mod");

    // func @loop_twin(%resume_id: i32, %state_buf: ptr) -> i64
    // resume_table {
    //   entry 0 -> bb_res_0
    // }
    // entry:
    //   %init_i = iconst 0
    //   %init_acc = iconst 0
    //   br loop_header(%init_i, %init_acc)
    // bb_res_0:
    //   %res_i = load.i64 %state_buf, 0
    //   %res_acc = load.i64 %state_buf, 8
    //   br loop_header(%res_i, %res_acc)
    // loop_header(%i: i64, %acc: i64):
    //   %cond = slt %i, 10
    //   br_if %cond, loop_body, loop_exit
    // loop_body:
    //   %val = load.i64 %state_buf, 16
    //   %acc_new = add %acc, %val
    //   %i_new = add %i, 1
    //   br loop_header(%i_new, %acc_new)
    // loop_exit:
    //   ret %acc
    Function* fn = mod.create_function("loop_twin", Type::i64(), {Type::i32(), Type::ptr()});
    Builder b(*fn);

    BasicBlock* bb_entry = b.append_block("entry");
    BasicBlock* bb_res = b.append_block("bb_res_0");
    BasicBlock* bb_header = b.append_block("loop_header");
    BasicBlock* bb_body = b.append_block("loop_body");
    BasicBlock* bb_exit = b.append_block("loop_exit");

    fn->add_resume_point(0, bb_res);

    b.position_at_end(bb_entry);
    (void)b.add_param(Type::i32());
    Value* state_buf = b.add_param(Type::ptr());
    Value* init_i = b.build_iconst_i64(0);
    Value* init_acc = b.build_iconst_i64(0);
    b.build_br(bb_header, {init_i, init_acc});

    b.position_at_end(bb_res);
    Value* res_i = b.build_load(Type::i64(), state_buf, 0);
    Value* res_acc = b.build_load(Type::i64(), state_buf, 8);
    b.build_br(bb_header, {res_i, res_acc});

    b.position_at_end(bb_header);
    Value* i_param = b.add_block_param(bb_header, Type::i64());
    Value* acc_param = b.add_block_param(bb_header, Type::i64());
    Value* loop_cond = b.build_slt(i_param, b.build_iconst_i64(10));
    b.build_br_if(loop_cond, bb_body, bb_exit);

    b.position_at_end(bb_body);
    Value* val = b.build_load(Type::i64(), state_buf, 16);
    Value* acc_new = b.build_add(acc_param, val);
    Value* i_new = b.build_add(i_param, b.build_iconst_i64(1));
    b.build_br(bb_header, {i_new, acc_new});

    b.position_at_end(bb_exit);
    b.build_ret(acc_param);

    Interpreter interp;
    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    uint64_t full_state[3] = {0, 0, 7};
    uint64_t resume_state[3] = {5, 35, 7}; // Resuming at step 5 with accumulated sum 35

    // Standard invocation: full loop executes 10 iterations -> 70
    RuntimeValue standard_interp = interp.run(*fn, {RuntimeValue::from_i32(999), RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(full_state))});
    RuntimeValue standard_jit = engine.invoke("loop_twin", {RuntimeValue::from_i32(999), RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(full_state))});
    CHECK_EQ(standard_interp.as_i64(), 70);
    CHECK_EQ(standard_jit.as_i64(), 70);

    // Direct resume at entry 0: loop resumes from i=5..9 with acc=35 -> 70
    RuntimeValue resume_interp = interp.run(*fn, {RuntimeValue::from_i32(0), RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(resume_state))});
    RuntimeValue resume_jit = engine.resume("loop_twin", 0, {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(resume_state))});
    CHECK_EQ(resume_interp.as_i64(), 70);
    CHECK_EQ(resume_jit.as_i64(), 70);
}

TEST_CASE("Differential - Patchable Constants and Calls") {
    Module mod("diff_patch_mod");

    Function* s_add = mod.create_function("diff_stub_add", Type::i32(), {Type::i32()});
    Builder ba(*s_add);
    BasicBlock* bba = ba.append_block("entry");
    ba.position_at_end(bba);
    ba.build_ret(ba.build_add(ba.add_param(Type::i32()), ba.build_iconst_i32(100)));

    Function* s_sub = mod.create_function("diff_stub_sub", Type::i32(), {Type::i32()});
    Builder bs(*s_sub);
    BasicBlock* bbs = bs.append_block("entry");
    bs.position_at_end(bbs);
    bs.build_ret(bs.build_sub(bs.add_param(Type::i32()), bs.build_iconst_i32(50)));

    Function* caller = mod.create_function("diff_caller", Type::i32(), {Type::i32()});
    Builder bc(*caller);
    BasicBlock* bbc = bc.append_block("entry");
    bc.position_at_end(bbc);
    Value* x = bc.add_param(Type::i32());
    Value* bias = bc.build_patchable_const_i32("diff_bias", 5);
    Value* biased_x = bc.build_add(x, bias);
    Value* res = bc.build_patchable_call("diff_call", "diff_stub_add", Type::i32(), {biased_x});
    bc.build_ret(res);

    Interpreter interp;
    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Phase 1: Default configuration (bias = 5, call = diff_stub_add): (x + 5) + 100
    int32_t input = 20;
    RuntimeValue i1 = interp.run(*caller, {RuntimeValue::from_i32(input)});
    RuntimeValue j1 = engine.invoke("diff_caller", {RuntimeValue::from_i32(input)});
    CHECK_EQ(i1.as_i32(), 125);
    CHECK_EQ(j1.as_i32(), 125);

    // Phase 2: Patch constant bias = 30 in both
    interp.patch_const("diff_bias", 30);
    engine.patch_const32("diff_bias", 30);
    RuntimeValue i2 = interp.run(*caller, {RuntimeValue::from_i32(input)});
    RuntimeValue j2 = engine.invoke("diff_caller", {RuntimeValue::from_i32(input)});
    CHECK_EQ(i2.as_i32(), 150); // (20 + 30) + 100
    CHECK_EQ(j2.as_i32(), 150);

    // Phase 3: Patch call site = diff_stub_sub in both: (x + 30) - 50
    interp.patch_call("diff_call", "diff_stub_sub");
    engine.patch_call("diff_call", "diff_stub_sub");
    RuntimeValue i3 = interp.run(*caller, {RuntimeValue::from_i32(input)});
    RuntimeValue j3 = engine.invoke("diff_caller", {RuntimeValue::from_i32(input)});
    CHECK_EQ(i3.as_i32(), 0); // (20 + 30) - 50 = 0
    CHECK_EQ(j3.as_i32(), 0);
}
