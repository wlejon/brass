#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("Bytecode Compiler - Simple Arithmetic") {
    Module mod("arith_mod");
    Builder b(mod);

    // func @test_arith(%a: i32, %b: i32) -> i32
    Function* fn = mod.create_function("test_arith", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* a = b.add_block_param(bb, Type::i32());
    Value* b_param = b.add_block_param(bb, Type::i32());

    Value* sum = b.build_add(a, b_param);
    Value* diff = b.build_sub(sum, b_param);
    Value* prod = b.build_mul(diff, b_param);
    Value* quot = b.build_sdiv(prod, a);
    b.build_ret(quot);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);
    CHECK_EQ(bfn->name, "test_arith");
    CHECK_EQ(bfn->num_params, 2);
    CHECK_EQ(bfn->return_type, Type::i32());
    CHECK(bfn->num_registers >= 6);
    CHECK(bfn->code.size() >= 5);

    // Check that entry params got r0 and r1
    // The instructions emitted should decode to add_i32, sub_i32, mul_i32, sdiv_i32, ret
    bool has_add = false;
    bool has_sub = false;
    bool has_mul = false;
    bool has_sdiv = false;
    bool has_ret = false;

    for (uint32_t raw : bfn->code) {
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::add_i32) has_add = true;
        if (op == BytecodeOp::sub_i32) has_sub = true;
        if (op == BytecodeOp::mul_i32) has_mul = true;
        if (op == BytecodeOp::sdiv_i32) has_sdiv = true;
        if (op == BytecodeOp::ret) has_ret = true;
    }

    CHECK(has_add);
    CHECK(has_sub);
    CHECK(has_mul);
    CHECK(has_sdiv);
    CHECK(has_ret);
}

TEST_CASE("Bytecode Compiler - Constants and Constant Pool Indexing") {
    Module mod("const_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_consts", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");

    // Small constants: should fit in inline 16-bit immediate (mov_imm)
    Value* c_small = b.build_iconst_i32(42);
    Value* c_small_neg = b.build_iconst_i32(-100);

    // Large constants: cannot fit in 16 bits, must go into constant pool (load_const)
    Value* c_large = b.build_iconst_i64(0x123456789ABCDEF0ULL);
    Value* c_float = b.build_fconst_f64(3.141592653589793);
    Value* c_float_bits = b.build_bitcast_f64_i64(c_float);

    Value* sum1 = b.build_add(c_small, c_small_neg);
    Value* sum1_64 = b.build_sext_i64(sum1);
    Value* sum2 = b.build_add(sum1_64, c_large);
    Value* total = b.build_add(sum2, c_float_bits);
    b.build_ret(total);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);
    CHECK(bfn->constants.size() >= 2); // At least c_large and c_float

    bool has_mov_imm = false;
    bool has_load_const = false;

    for (uint32_t raw : bfn->code) {
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::mov_imm) has_mov_imm = true;
        if (op == BytecodeOp::load_const) has_load_const = true;
    }

    CHECK(has_mov_imm);
    CHECK(has_load_const);

    // Verify constant pool content
    bool found_large = false;
    for (uint64_t val : bfn->constants) {
        if (val == 0x123456789ABCDEF0ULL) {
            found_large = true;
            break;
        }
    }
    CHECK(found_large);
}

TEST_CASE("Bytecode Compiler - Conditionals and If-Else Diamond") {
    Module mod("cond_mod");
    Builder b(mod);

    // func @abs_val(%x: i32) -> i32
    Function* fn = mod.create_function("abs_val", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* then_bb = b.append_block("then");
    BasicBlock* else_bb = b.append_block("else");
    BasicBlock* merge_bb = b.append_block("merge");

    Value* x = b.add_block_param(entry, Type::i32());
    Value* zero = b.build_iconst_i32(0);
    Value* cond = b.build_slt(x, zero);
    b.build_br_if(cond, then_bb, {}, else_bb, {});

    // then: neg_x = neg(x); br merge(neg_x)
    b.position_at_end(then_bb);
    Value* neg_x = b.build_neg(x);
    b.build_br(merge_bb, {neg_x});

    // else: br merge(x)
    b.position_at_end(else_bb);
    b.build_br(merge_bb, {x});

    // merge(%res: i32): ret %res
    b.position_at_end(merge_bb);
    Value* res = b.add_block_param(merge_bb, Type::i32());
    b.build_ret(res);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);
    CHECK(bfn->code.size() >= 6);

    // Look for jump_if and jump instructions with backpatched relative offsets
    bool found_jump_if = false;
    bool found_jump = false;

    for (size_t pc = 0; pc < bfn->code.size(); ++pc) {
        uint32_t raw = bfn->code[pc];
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::jump_if) {
            found_jump_if = true;
            int16_t offset = decode_s16(raw);
            int32_t target_pc = static_cast<int32_t>(pc) + offset;
            CHECK(target_pc >= 0);
            CHECK(target_pc < static_cast<int32_t>(bfn->code.size()));
        }
        if (op == BytecodeOp::jump) {
            found_jump = true;
            int16_t offset = decode_s16(raw);
            int32_t target_pc = static_cast<int32_t>(pc) + offset;
            CHECK(target_pc >= 0);
            CHECK(target_pc < static_cast<int32_t>(bfn->code.size()));
        }
    }

    CHECK(found_jump_if);
    CHECK(found_jump);
}

TEST_CASE("Bytecode Compiler - Loops with Block Arguments (Accumulator Sum 1..N)") {
    Module mod("loop_mod");
    Builder b(mod);

    // func @sum_1_to_n(%n: i32) -> i32
    Function* fn = mod.create_function("sum_1_to_n", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("loop_header");
    BasicBlock* body = b.append_block("loop_body");
    BasicBlock* exit = b.append_block("loop_exit");

    Value* n = b.add_block_param(entry, Type::i32());
    Value* zero = b.build_iconst_i32(0);
    Value* one = b.build_iconst_i32(1);
    b.build_br(header, {one, zero});

    // header(i, acc)
    b.position_at_end(header);
    Value* i_val = b.add_block_param(header, Type::i32());
    Value* acc_val = b.add_block_param(header, Type::i32());
    Value* cond = b.build_sle(i_val, n);
    b.build_br_if(cond, body, {}, exit, {});

    // body: acc = acc + i; i = i + 1; br header(i, acc)
    b.position_at_end(body);
    Value* next_acc = b.build_add(acc_val, i_val);
    Value* next_i = b.build_add(i_val, one);
    b.build_br(header, {next_i, next_acc});

    // exit: ret acc
    b.position_at_end(exit);
    b.build_ret(acc_val);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);

    // Look for backward jump (loop backedge) with negative offset
    bool found_backedge = false;
    for (size_t pc = 0; pc < bfn->code.size(); ++pc) {
        uint32_t raw = bfn->code[pc];
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::jump) {
            int16_t offset = decode_s16(raw);
            if (offset < 0) {
                found_backedge = true;
                int32_t target_pc = static_cast<int32_t>(pc) + offset;
                CHECK(target_pc >= 0);
            }
        }
    }

    CHECK(found_backedge);
}

TEST_CASE("Bytecode Compiler - Mutual Register Moves & Cyclic Swaps") {
    Module mod("cycle_mod");
    Builder b(mod);

    // Test a block that swaps two values in a loop: loop_bb(x, y) -> br loop_bb(y, x)
    Function* fn = mod.create_function("swap_loop", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop = b.append_block("loop");
    BasicBlock* exit = b.append_block("exit");

    Value* a = b.add_block_param(entry, Type::i32());
    Value* b_param = b.add_block_param(entry, Type::i32());
    b.build_br(loop, {a, b_param});

    // loop(%x: i32, %y: i32)
    b.position_at_end(loop);
    Value* x = b.add_block_param(loop, Type::i32());
    Value* y = b.add_block_param(loop, Type::i32());
    Value* cond = b.build_eq(x, y);
    // When cond == 0, branch to loop passing {y, x} - a DIRECT cyclic swap!
    b.build_br_if(cond, exit, {}, loop, {y, x});

    b.position_at_end(exit);
    b.build_ret(x);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);

    // Check that scratch register was allocated and moves were emitted
    CHECK(bfn->num_registers > 4); // x, y, cond, scratch_reg, scratch_reg2

    // There should be multiple mov instructions to break the swap cycle:
    // mov scratch, y
    // mov y, x
    // mov x, scratch
    int mov_count = 0;
    for (uint32_t raw : bfn->code) {
        if (decode_op(raw) == BytecodeOp::mov) {
            mov_count++;
        }
    }

    CHECK(mov_count >= 3);
}

TEST_CASE("Bytecode Compiler - Recursive Fibonacci") {
    Module mod("fib_mod");
    Builder b(mod);

    // func @fib(%n: i32) -> i32
    Function* fn = mod.create_function("fib", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_bb = b.append_block("base");
    BasicBlock* rec_bb = b.append_block("rec");

    Value* n = b.add_block_param(entry, Type::i32());
    Value* two = b.build_iconst_i32(2);
    Value* cond = b.build_slt(n, two);
    b.build_br_if(cond, base_bb, {}, rec_bb, {});

    b.position_at_end(base_bb);
    b.build_ret(n);

    b.position_at_end(rec_bb);
    Value* one = b.build_iconst_i32(1);
    Value* n1 = b.build_sub(n, one);
    Value* fib1 = b.build_call("fib", Type::i32(), {n1});
    Value* n2 = b.build_sub(n, two);
    Value* fib2 = b.build_call("fib", Type::i32(), {n2});
    Value* sum = b.build_add(fib1, fib2);
    b.build_ret(sum);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);
    CHECK_EQ(bfn->call_sites.size(), 2);
    CHECK_EQ(bfn->call_sites[0].callee, "fib");
    CHECK_EQ(bfn->call_sites[1].callee, "fib");
    CHECK_EQ(bfn->call_sites[0].arg_regs.size(), 1);
    CHECK_EQ(bfn->call_sites[1].arg_regs.size(), 1);

    int call_count = 0;
    for (uint32_t raw : bfn->code) {
        if (decode_op(raw) == BytecodeOp::call) {
            call_count++;
        }
    }
    CHECK_EQ(call_count, 2);
}

TEST_CASE("Bytecode Compiler - Memory Operations (alloca, load, store)") {
    Module mod("mem_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_mem", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* val = b.add_block_param(bb, Type::i64());
    Value* ptr = b.build_alloca(8, 8);
    b.build_store(Type::i64(), ptr, val);
    Value* loaded = b.build_load(Type::i64(), ptr);
    b.build_ret(loaded);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);

    bool has_alloca = false;
    bool has_store = false;
    bool has_load = false;

    for (uint32_t raw : bfn->code) {
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::alloca_) has_alloca = true;
        if (op == BytecodeOp::store64) has_store = true;
        if (op == BytecodeOp::load64) has_load = true;
    }

    CHECK(has_alloca);
    CHECK(has_store);
    CHECK(has_load);
}

TEST_CASE("Bytecode Compiler - Bitwise, CLZ, CTZ, Popcnt & Conversions") {
    Module mod("bits_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_bits_conv", Type::i64(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* x = b.add_block_param(bb, Type::i32());
    Value* c1 = b.build_clz(x);
    Value* c2 = b.build_ctz(x);
    Value* p = b.build_popcnt(x);
    Value* s = b.build_add(c1, c2);
    Value* total = b.build_add(s, p);
    Value* total64 = b.build_zext_i64(total);
    b.build_ret(total64);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);

    bool has_clz = false;
    bool has_ctz = false;
    bool has_popcnt = false;
    bool has_zext = false;

    for (uint32_t raw : bfn->code) {
        BytecodeOp op = decode_op(raw);
        if (op == BytecodeOp::clz_i32) has_clz = true;
        if (op == BytecodeOp::ctz_i32) has_ctz = true;
        if (op == BytecodeOp::popcnt_i32) has_popcnt = true;
        if (op == BytecodeOp::zext64) has_zext = true;
    }

    CHECK(has_clz);
    CHECK(has_ctz);
    CHECK(has_popcnt);
    CHECK(has_zext);
}

TEST_CASE("Bytecode Compiler - Module Compilation and Symbol Resolution") {
    Module mod("multi_mod");
    Builder b(mod);

    Function* f1 = mod.create_function("func_one", Type::i32(), {Type::i32()});
    b.set_function(f1);
    BasicBlock* bb1 = b.append_block("entry");
    Value* p1 = b.add_block_param(bb1, Type::i32());
    Value* two = b.build_iconst_i32(2);
    Value* prod = b.build_mul(p1, two);
    b.build_ret(prod);

    Function* f2 = mod.create_function("func_two", Type::i32(), {Type::i32()});
    b.set_function(f2);
    BasicBlock* bb2 = b.append_block("entry");
    Value* p2 = b.add_block_param(bb2, Type::i32());
    Value* call_res = b.build_call("func_one", Type::i32(), {p2});
    b.build_ret(call_res);

    BytecodeCompiler compiler;
    auto bmod = compiler.compile(mod);

    REQUIRE(bmod != nullptr);
    CHECK_EQ(bmod->name(), "multi_mod");
    CHECK_EQ(bmod->function_count(), 2);
    CHECK(bmod->has_symbol("func_one"));
    CHECK(bmod->has_symbol("func_two"));

    BytecodeFunction* fn1 = bmod->get_function("func_one");
    REQUIRE(fn1 != nullptr);
    CHECK_EQ(fn1->name, "func_one");

    BytecodeFunction* fn2 = bmod->get_function("func_two");
    REQUIRE(fn2 != nullptr);
    CHECK_EQ(fn2->name, "func_two");
}

TEST_CASE("Bytecode Compiler - Disassembler Output") {
    Module mod("disasm_mod");
    Builder b(mod);

    Function* fn = mod.create_function("add_test", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* a = b.add_block_param(bb, Type::i32());
    Value* b_param = b.add_block_param(bb, Type::i32());
    Value* sum = b.build_add(a, b_param);
    b.build_ret(sum);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    std::string text = disassemble(*bfn);
    CHECK(text.find("function @add_test") != std::string::npos);
    CHECK(text.find(".registers") != std::string::npos);
    CHECK(text.find(".params") != std::string::npos);
    CHECK(text.find("add_i32") != std::string::npos);
    CHECK(text.find("ret") != std::string::npos);
}
