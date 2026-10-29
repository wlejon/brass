#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>

using namespace brass;

namespace {

// The opcodes of a function in order, skipping the data words of two-word
// instructions.
std::vector<BytecodeOp> ops_of(const BytecodeFunction& bfn) {
    std::vector<BytecodeOp> ops;
    for (size_t pc = 0; pc < bfn.code.size(); pc += bytecode_inst_words(decode_op(bfn.code[pc]))) {
        ops.push_back(decode_op(bfn.code[pc]));
    }
    return ops;
}

bool has_op(const BytecodeFunction& bfn, BytecodeOp op) {
    for (BytecodeOp o : ops_of(bfn)) {
        if (o == op) return true;
    }
    return false;
}

size_t count_op(const BytecodeFunction& bfn, BytecodeOp op) {
    size_t n = 0;
    for (BytecodeOp o : ops_of(bfn)) n += o == op ? 1 : 0;
    return n;
}

// Runs `fn` on the reference interpreter and the FastInterpreter.
void check_same_i64(Module& mod, Function& fn, const std::vector<RuntimeValue>& args, int64_t expected) {
    Interpreter oracle;
    oracle.set_module(&mod);
    RuntimeValue want = oracle.run(fn, args);
    CHECK_EQ(want.as_i64(), expected);

    FastInterpreter fast;
    fast.set_module(&mod);
    RuntimeValue got = fast.run(fn, args);
    CHECK_EQ(got.as_i64(), expected);
}

} // namespace

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
    CHECK(bfn->num_registers >= 4);
    CHECK(bfn->code.size() >= 5);

    CHECK(has_op(*bfn, BytecodeOp::add_i32));
    CHECK(has_op(*bfn, BytecodeOp::sub_i32));
    CHECK(has_op(*bfn, BytecodeOp::mul_i32));
    CHECK(has_op(*bfn, BytecodeOp::sdiv_i32));
    CHECK(has_op(*bfn, BytecodeOp::ret));
}

TEST_CASE("Bytecode Compiler - Constants and Constant Pool Indexing") {
    Module mod("const_mod");
    Builder b(mod);

    Function* fn = mod.create_function("test_consts", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");

    // i32 constants are an inline 32-bit immediate (zero-extended).
    Value* c_small = b.build_iconst_i32(42);
    Value* c_small_neg = b.build_iconst_i32(-100);

    // A 64-bit constant outside int32 and float constants use the pool.
    Value* c_large = b.build_iconst_i64(0x123456789ABCDEF0LL);
    Value* c_float = b.build_fconst_f64(3.141592653589793);
    Value* c_float_bits = b.build_bitcast_i64_f64(c_float);
    // A small i64 constant is a sign-extended immediate.
    Value* c_small64 = b.build_iconst_i64(-7);

    Value* sum1 = b.build_add(c_small, c_small_neg);
    Value* sum1_64 = b.build_sext_i64(sum1);
    Value* sum2 = b.build_add(sum1_64, c_large);
    Value* sum3 = b.build_add(sum2, c_small64);
    Value* total = b.build_add(sum3, c_float_bits);
    b.build_ret(total);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);

    REQUIRE(bfn != nullptr);
    CHECK(bfn->constants.size() >= 2); // At least c_large and c_float

    CHECK(has_op(*bfn, BytecodeOp::iconst32));
    CHECK(has_op(*bfn, BytecodeOp::mov_imm));
    CHECK(has_op(*bfn, BytecodeOp::load_const));

    bool found_large = false;
    for (uint64_t val : bfn->constants) {
        if (val == 0x123456789ABCDEF0ULL) found_large = true;
    }
    CHECK(found_large);

    double pi = 3.141592653589793;
    int64_t pi_bits = 0;
    std::memcpy(&pi_bits, &pi, sizeof(pi));
    const int64_t expected = static_cast<int64_t>(static_cast<uint64_t>(-58) + 0x123456789ABCDEF0ULL +
                                                  static_cast<uint64_t>(-7) + static_cast<uint64_t>(pi_bits));
    check_same_i64(mod, *fn, {}, expected);
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
    b.position_at_end(entry);

    Value* x = b.add_block_param(entry, Type::i32());
    Value* zero = b.build_iconst_i32(0);
    Value* cond = b.build_slt(x, zero);
    b.build_br_if(cond, then_bb, {}, else_bb, {});

    b.position_at_end(then_bb);
    Value* neg_x = b.build_neg(x);
    b.build_br(merge_bb, {neg_x});

    b.position_at_end(else_bb);
    b.build_br(merge_bb, {x});

    b.position_at_end(merge_bb);
    Value* res = b.add_block_param(merge_bb, Type::i32());
    b.build_ret(res);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);

    // The slt feeding only the br_if becomes a fused compare-branch; every
    // branch target lands inside the function.
    bool found_cond = false;
    bool found_jump = false;
    for (size_t pc = 0; pc < bfn->code.size(); pc += bytecode_inst_words(decode_op(bfn->code[pc]))) {
        BytecodeOp op = decode_op(bfn->code[pc]);
        if (!is_bytecode_jump(op)) continue;
        if (is_bytecode_cond_branch(op)) found_cond = true;
        if (op == BytecodeOp::jump) found_jump = true;
        int64_t target = bytecode_branch_target(bfn->code[pc], pc);
        CHECK(target >= 0);
        CHECK(target < static_cast<int64_t>(bfn->code.size()));
    }
    CHECK(found_cond);
    CHECK(found_jump);
    CHECK(has_op(*bfn, BytecodeOp::br_slt_i32) || has_op(*bfn, BytecodeOp::br_sle_i32));
    CHECK_FALSE(has_op(*bfn, BytecodeOp::slt_i32));

    FastInterpreter fast;
    fast.set_module(&mod);
    CHECK_EQ(fast.run(*fn, {RuntimeValue::from_i32(-5)}).as_i32(), 5);
    CHECK_EQ(fast.run(*fn, {RuntimeValue::from_i32(9)}).as_i32(), 9);
}

TEST_CASE("Bytecode Compiler - Loops with Block Arguments (Accumulator Sum 1..N)") {
    Module mod("loop_mod");
    Builder b(mod);

    Function* fn = mod.create_function("sum_1_to_n", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("loop_header");
    BasicBlock* body = b.append_block("loop_body");
    BasicBlock* exit = b.append_block("loop_exit");
    b.position_at_end(entry);

    Value* n = b.add_block_param(entry, Type::i32());
    Value* zero = b.build_iconst_i32(0);
    Value* one = b.build_iconst_i32(1);
    b.build_br(header, {one, zero});

    b.position_at_end(header);
    Value* i_val = b.add_block_param(header, Type::i32());
    Value* acc_val = b.add_block_param(header, Type::i32());
    Value* cond = b.build_sle(i_val, n);
    b.build_br_if(cond, body, {}, exit, {});

    b.position_at_end(body);
    Value* next_acc = b.build_add(acc_val, i_val);
    Value* next_i = b.build_add(i_val, one);
    b.build_br(header, {next_i, next_acc});

    b.position_at_end(exit);
    b.build_ret(acc_val);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);

    bool found_backedge = false;
    for (size_t pc = 0; pc < bfn->code.size(); pc += bytecode_inst_words(decode_op(bfn->code[pc]))) {
        BytecodeWord raw = bfn->code[pc];
        if (decode_op(raw) == BytecodeOp::jump && decode_imm32(raw) < 0) {
            found_backedge = true;
            CHECK(bytecode_branch_target(raw, pc) >= 0);
        }
    }
    CHECK(found_backedge);

    FastInterpreter fast;
    fast.set_module(&mod);
    CHECK_EQ(fast.run(*fn, {RuntimeValue::from_i32(100)}).as_i32(), 5050);
}

TEST_CASE("Bytecode Compiler - Mutual Register Moves & Cyclic Swaps") {
    Module mod("cycle_mod");
    Builder b(mod);

    // loop(x, y) -> br loop(y, x): a cyclic parallel move.
    Function* fn = mod.create_function("swap_loop", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop = b.append_block("loop");
    BasicBlock* exit = b.append_block("exit");
    b.position_at_end(entry);

    Value* a = b.add_block_param(entry, Type::i32());
    Value* b_param = b.add_block_param(entry, Type::i32());
    b.build_br(loop, {a, b_param});

    b.position_at_end(loop);
    Value* x = b.add_block_param(loop, Type::i32());
    Value* y = b.add_block_param(loop, Type::i32());
    Value* cond = b.build_eq(x, y);
    b.build_br_if(cond, exit, {}, loop, {y, x});

    b.position_at_end(exit);
    b.build_ret(x);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);

    // The swap goes through the scratch register: three moves.
    CHECK(bfn->num_registers > 4);
    CHECK(count_op(*bfn, BytecodeOp::mov) >= 3);

    FastInterpreter fast;
    fast.set_module(&mod);
    CHECK_EQ(fast.run(*fn, {RuntimeValue::from_i32(4), RuntimeValue::from_i32(4)}).as_i32(), 4);
}

TEST_CASE("Bytecode Compiler - Recursive Fibonacci") {
    Module mod("fib_mod");
    Builder b(mod);

    Function* fn = mod.create_function("fib", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_bb = b.append_block("base");
    BasicBlock* rec_bb = b.append_block("rec");
    b.position_at_end(entry);

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
    CHECK_EQ(count_op(*bfn, BytecodeOp::call), 2);

    FastInterpreter fast;
    fast.set_module(&mod);
    CHECK_EQ(fast.run(*fn, {RuntimeValue::from_i32(15)}).as_i32(), 610);
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

    CHECK(has_op(*bfn, BytecodeOp::alloca_));
    CHECK(has_op(*bfn, BytecodeOp::store64));
    CHECK(has_op(*bfn, BytecodeOp::load64));

    check_same_i64(mod, *fn, {RuntimeValue::from_i64(-123456789012LL)}, -123456789012LL);
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

    CHECK(has_op(*bfn, BytecodeOp::clz_i32));
    CHECK(has_op(*bfn, BytecodeOp::ctz_i32));
    CHECK(has_op(*bfn, BytecodeOp::popcnt_i32));
    CHECK(has_op(*bfn, BytecodeOp::zext64));

    // 0x00F0: clz 24, ctz 4, popcnt 4.
    check_same_i64(mod, *fn, {RuntimeValue::from_i32(0xF0)}, 32);
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

    FastInterpreter fast;
    fast.set_bytecode_module(bmod.get());
    CHECK_EQ(fast.run("func_two", {RuntimeValue::from_i32(21)}).as_i32(), 42);
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

// ---------------------------------------------------------------------------
// Register allocation on large functions
// ---------------------------------------------------------------------------

TEST_CASE("Bytecode Compiler - More than 256 simultaneously live values") {
    Module mod("wide_mod");
    Builder b(mod);

    // v_k = x * (k + 1) for k < 600, all live until they are summed.
    constexpr int kValues = 600;
    Function* fn = mod.create_function("wide", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* x = b.add_block_param(bb, Type::i64());
    std::vector<Value*> vals;
    for (int k = 0; k < kValues; ++k) {
        vals.push_back(b.build_mul(x, b.build_iconst_i64(k + 1)));
    }
    Value* acc = vals[0];
    for (int k = 1; k < kValues; ++k) acc = b.build_add(acc, vals[static_cast<size_t>(k)]);
    b.build_ret(acc);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);
    CHECK(bfn->num_registers > 256);
    CHECK(bfn->num_registers < bfn->num_ssa_values);

    const int64_t sum_k = static_cast<int64_t>(kValues) * (kValues + 1) / 2;
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(3)}, 3 * sum_k);
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(-11)}, -11 * sum_k);
}

TEST_CASE("Bytecode Compiler - More than 65535 SSA values reuse a few registers") {
    Module mod("long_mod");
    Builder b(mod);

    // A 70000-long dependency chain: each value dies at its only use.
    constexpr int kChain = 70000;
    Function* fn = mod.create_function("chain", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* v = b.add_block_param(bb, Type::i64());
    Value* one = b.build_iconst_i64(1);
    for (int k = 0; k < kChain; ++k) v = b.build_add(v, one);
    b.build_ret(v);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);
    CHECK(bfn->num_ssa_values > 65535u);
    CHECK(bfn->num_registers <= 8u);

    check_same_i64(mod, *fn, {RuntimeValue::from_i64(5)}, 5 + kChain);
}

TEST_CASE("Bytecode Compiler - Exceeding the register file is a hard error") {
    Module mod("too_wide_mod");
    Builder b(mod);

    // 66000 values live at once cannot fit 16-bit register numbers.
    constexpr int kValues = 66000;
    Function* fn = mod.create_function("too_wide", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");
    std::vector<Value*> vals;
    for (int k = 0; k < kValues; ++k) vals.push_back(b.build_iconst_i64(k));
    Value* acc = vals[0];
    for (int k = 1; k < kValues; ++k) acc = b.build_add(acc, vals[static_cast<size_t>(k)]);
    b.build_ret(acc);

    BytecodeCompiler compiler;
    bool threw = false;
    try {
        (void)compiler.compile(*fn);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Bytecode Compiler - Register reuse across loops, calls and types") {
    Module mod("reuse_mod");
    Builder b(mod);

    // helper(a) = a * 3 + 1
    Function* helper = mod.create_function("helper", Type::i64(), {Type::i64()});
    b.set_function(helper);
    BasicBlock* hb = b.append_block("entry");
    Value* ha = b.add_block_param(hb, Type::i64());
    b.build_ret(b.build_add(b.build_mul(ha, b.build_iconst_i64(3)), b.build_iconst_i64(1)));

    // f(n): acc += helper(i*i); facc += i * 0.5; returns acc + (i64)facc
    Function* fn = mod.create_function("mixed", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop = b.append_block("loop");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");
    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i64());
    b.build_br(loop, {b.build_iconst_i64(0), b.build_iconst_i64(0), b.build_fconst_f64(0.0)});

    b.position_at_end(loop);
    Value* i = b.add_block_param(loop, Type::i64());
    Value* acc = b.add_block_param(loop, Type::i64());
    Value* facc = b.add_block_param(loop, Type::f64());
    b.build_br_if(b.build_slt(i, n), body, {}, exit, {});

    b.position_at_end(body);
    Value* sq = b.build_mul(i, i);
    Value* h = b.build_call("helper", Type::i64(), {sq});
    Value* d = b.build_sitofp_f64_i64(i);
    Value* fnext = b.build_add(facc, b.build_mul(d, b.build_fconst_f64(0.5)));
    Value* acc2 = b.build_add(acc, h);
    Value* i2 = b.build_add(i, b.build_iconst_i64(1));
    b.build_br(loop, {i2, acc2, fnext});

    b.position_at_end(exit);
    b.build_ret(b.build_add(acc, b.build_fptosi_i64(facc)));

    for (int64_t count : {0, 1, 7, 100}) {
        int64_t want_acc = 0;
        double want_f = 0.0;
        for (int64_t k = 0; k < count; ++k) {
            want_acc += k * k * 3 + 1;
            want_f += static_cast<double>(k) * 0.5;
        }
        check_same_i64(mod, *fn, {RuntimeValue::from_i64(count)}, want_acc + static_cast<int64_t>(want_f));
    }
}

TEST_CASE("Bytecode Compiler - A safepoint roots the values live at it, not those whose interval spans it") {
    Module mod("safepoint_live_mod");
    Builder b(mod);

    Function* helper = mod.create_function("helper", Type::i64(), {Type::i64()});
    b.set_function(helper);
    BasicBlock* hb = b.append_block("entry");
    Value* ha = b.add_block_param(hb, Type::i64());
    b.build_ret(b.build_add(ha, b.build_iconst_i64(1)));

    // `slow` and `tail` are laid out around `fast`, so p's interval spans the
    // call in `fast`, a path on which p is never written. Its register there
    // holds whatever it held before: rooted, a collection would trace a dead
    // word.
    Function* fn = mod.create_function("split", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* slow = b.append_block("slow");
    BasicBlock* fast = b.append_block("fast");
    BasicBlock* tail = b.append_block("tail");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_br_if(b.build_slt(x, b.build_iconst_i64(0)), slow, {b.build_mul(x, b.build_iconst_i64(7))}, fast, {});

    b.position_at_end(slow);
    Value* p = b.add_block_param(slow, Type::i64());
    b.build_br(tail);

    b.position_at_end(fast);
    b.build_ret(b.build_call("helper", Type::i64(), {x}));

    b.position_at_end(tail);
    Value* h = b.build_call("helper", Type::i64(), {x});
    b.build_ret(b.build_add(p, h));

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);
    REQUIRE_EQ(bfn->gc_safepoints.size(), size_t{2});
    const BcReg p_reg = bfn->ssa_to_reg[p->id()];
    const BcReg x_reg = bfn->ssa_to_reg[x->id()];
    auto live_at = [&](size_t k, BcReg r) {
        const auto& sp = bfn->gc_safepoints[k];
        for (uint32_t i = 0; i < sp.count; ++i) {
            if (bfn->gc_live_regs[sp.first + i] == r) return true;
        }
        return false;
    };
    // fast's call, then tail's: x is an operand of both, p live only past
    // the second.
    CHECK(live_at(0, x_reg));
    CHECK(!live_at(0, p_reg));
    CHECK(live_at(1, x_reg));
    CHECK(live_at(1, p_reg));

    check_same_i64(mod, *fn, {RuntimeValue::from_i64(5)}, 6);
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(-2)}, -15);
}

TEST_CASE("Bytecode Compiler - Switch edges carry block arguments") {
    Module mod("switch_args_mod");
    Builder b(mod);

    Function* fn = mod.create_function("pick", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* join = b.append_block("join");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* ten = b.build_iconst_i64(10);
    Value* twenty = b.build_iconst_i64(20);
    Value* minus = b.build_iconst_i64(-1);
    std::vector<SwitchCase> cases;
    cases.emplace_back(1, join, std::vector<Value*>{ten, x});
    cases.emplace_back(2, join, std::vector<Value*>{twenty, ten});
    std::vector<Value*> dflt = {minus, twenty};
    b.build_switch(x, join, Span<Value* const>(dflt.data(), dflt.size()),
                   Span<const SwitchCase>(cases.data(), cases.size()));

    b.position_at_end(join);
    Value* p = b.add_block_param(join, Type::i64());
    Value* q = b.add_block_param(join, Type::i64());
    b.build_ret(b.build_add(b.build_mul(p, b.build_iconst_i64(100)), q));

    check_same_i64(mod, *fn, {RuntimeValue::from_i64(1)}, 1001);
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(2)}, 2010);
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(3)}, -80);
}

TEST_CASE("Bytecode Compiler - Fused Float Compare-Branches (f64 and f32, including NaN)") {
    Module mod("float_br_mod");
    Builder b(mod);

    const Opcode ops[] = {Opcode::eq, Opcode::ne, Opcode::slt, Opcode::sle, Opcode::sgt, Opcode::sge};
    const char* names[] = {"eq", "ne", "slt", "sle", "sgt", "sge"};

    auto build_cmp = [&](Builder& bld, Opcode op, Value* lhs, Value* rhs) -> Value* {
        switch (op) {
            case Opcode::eq: return bld.build_eq(lhs, rhs);
            case Opcode::ne: return bld.build_ne(lhs, rhs);
            case Opcode::slt: return bld.build_slt(lhs, rhs);
            case Opcode::sle: return bld.build_sle(lhs, rhs);
            case Opcode::sgt: return bld.build_sgt(lhs, rhs);
            case Opcode::sge: return bld.build_sge(lhs, rhs);
            default: return nullptr;
        }
    };

    // Build both normal (false_bb is fallthrough) and inverted (true_bb is fallthrough) layouts
    // for every float comparison opcode on f64 and f32.
    std::vector<Function*> f64_fns;
    std::vector<Function*> f32_fns;
    for (size_t i = 0; i < 6; ++i) {
        for (int layout_invert = 0; layout_invert < 2; ++layout_invert) {
            // f64
            {
                std::string fname = std::string("f64_") + names[i] + (layout_invert ? "_inv" : "_dir");
                Function* fn = mod.create_function(fname, Type::i64(), {Type::f64(), Type::f64()});
                b.set_function(fn);
                BasicBlock* entry = b.append_block("entry");
                BasicBlock* first_bb = b.append_block(layout_invert ? "true_bb" : "false_bb");
                BasicBlock* second_bb = b.append_block(layout_invert ? "false_bb" : "true_bb");
                BasicBlock* true_bb = layout_invert ? first_bb : second_bb;
                BasicBlock* false_bb = layout_invert ? second_bb : first_bb;

                b.position_at_end(entry);
                Value* x = b.add_block_param(entry, Type::f64());
                Value* y = b.add_block_param(entry, Type::f64());
                Value* cmp = build_cmp(b, ops[i], x, y);
                b.build_br_if(cmp, true_bb, {}, false_bb, {});

                b.position_at_end(true_bb);
                b.build_ret(b.build_iconst_i64(111));

                b.position_at_end(false_bb);
                b.build_ret(b.build_iconst_i64(222));

                f64_fns.push_back(fn);
            }
            // f32
            {
                std::string fname = std::string("f32_") + names[i] + (layout_invert ? "_inv" : "_dir");
                Function* fn = mod.create_function(fname, Type::i64(), {Type::f32(), Type::f32()});
                b.set_function(fn);
                BasicBlock* entry = b.append_block("entry");
                BasicBlock* first_bb = b.append_block(layout_invert ? "true_bb" : "false_bb");
                BasicBlock* second_bb = b.append_block(layout_invert ? "false_bb" : "true_bb");
                BasicBlock* true_bb = layout_invert ? first_bb : second_bb;
                BasicBlock* false_bb = layout_invert ? second_bb : first_bb;

                b.position_at_end(entry);
                Value* x = b.add_block_param(entry, Type::f32());
                Value* y = b.add_block_param(entry, Type::f32());
                Value* cmp = build_cmp(b, ops[i], x, y);
                b.build_br_if(cmp, true_bb, {}, false_bb, {});

                b.position_at_end(true_bb);
                b.build_ret(b.build_iconst_i64(111));

                b.position_at_end(false_bb);
                b.build_ret(b.build_iconst_i64(222));

                f32_fns.push_back(fn);
            }
        }
    }

    BytecodeCompiler compiler;
    for (Function* fn : f64_fns) {
        auto bfn = compiler.compile(*fn);
        REQUIRE(bfn != nullptr);
        CHECK_FALSE(has_op(*bfn, BytecodeOp::jump_if));
        CHECK_FALSE(has_op(*bfn, BytecodeOp::jump_if_not));
        bool found_fused = false;
        for (BytecodeOp op : ops_of(*bfn)) {
            if (is_bytecode_cond_branch(op)) found_fused = true;
        }
        CHECK(found_fused);
    }
    for (Function* fn : f32_fns) {
        auto bfn = compiler.compile(*fn);
        REQUIRE(bfn != nullptr);
        CHECK_FALSE(has_op(*bfn, BytecodeOp::jump_if));
        CHECK_FALSE(has_op(*bfn, BytecodeOp::jump_if_not));
        bool found_fused = false;
        for (BytecodeOp op : ops_of(*bfn)) {
            if (is_bytecode_cond_branch(op)) found_fused = true;
        }
        CHECK(found_fused);
    }

    Interpreter oracle;
    oracle.set_module(&mod);
    FastInterpreter fast;
    fast.set_module(&mod);

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double d_vals[] = {qnan, -qnan, -inf, -2.5, -0.0, +0.0, 1.25, 2.5, inf};
    for (Function* fn : f64_fns) {
        for (double vx : d_vals) {
            for (double vy : d_vals) {
                std::vector<RuntimeValue> args = {RuntimeValue::from_f64(vx), RuntimeValue::from_f64(vy)};
                CHECK_EQ(fast.run(*fn, args).as_i64(), oracle.run(*fn, args).as_i64());
            }
        }
    }

    const float fqnan = std::numeric_limits<float>::quiet_NaN();
    const float finf = std::numeric_limits<float>::infinity();
    const float f_vals[] = {fqnan, -fqnan, -finf, -2.5f, -0.0f, +0.0f, 1.25f, 2.5f, finf};
    for (Function* fn : f32_fns) {
        for (float vx : f_vals) {
            for (float vy : f_vals) {
                std::vector<RuntimeValue> args = {RuntimeValue::from_f32(vx), RuntimeValue::from_f32(vy)};
                CHECK_EQ(fast.run(*fn, args).as_i64(), oracle.run(*fn, args).as_i64());
            }
        }
    }
}

TEST_CASE("Bytecode Compiler - Fused Indexed Load and Store Superinstructions") {
    Module mod("indexed_mem_mod");
    Builder b(mod);

    Function* fn = mod.create_function("indexed_rw", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* idx = b.add_block_param(bb, Type::i64());

    Value* buf64 = b.build_alloca(64, 8);
    Value* buf32 = b.build_alloca(32, 4);
    Value* buf16 = b.build_alloca(16, 2);
    Value* buf8 = b.build_alloca(8, 1);

    b.build_store_indexed(Type::i64(), buf64, idx, 8, 0, b.build_iconst_i64(0x1122334455667788LL));
    b.build_store_indexed(Type::i32(), buf32, idx, 4, 0, b.build_iconst_i32(0x12345678));
    b.build_store_indexed(Type::i16(), buf16, idx, 2, 0, b.build_iconst_i32(0xABCD));
    b.build_store_indexed(Type::i8(), buf8, idx, 1, 0, b.build_trunc_i8(b.build_iconst_i32(0x7F)));

    Value* v64 = b.build_load_indexed(Type::i64(), buf64, idx, 8, 0);
    Value* v32 = b.build_zext_i64(b.build_load_indexed(Type::i32(), buf32, idx, 4, 0));
    Value* v16 = b.build_zext_i64(b.build_load_indexed(Type::i16(), buf16, idx, 2, 0));
    Value* v8 = b.build_zext_i64(b.build_load_indexed(Type::i8(), buf8, idx, 1, 0));

    Value* sum = b.build_add(b.build_add(v64, v32), b.build_add(v16, v8));
    b.build_ret(sum);

    BytecodeCompiler compiler;
    auto bfn = compiler.compile(*fn);
    REQUIRE(bfn != nullptr);

    CHECK(has_op(*bfn, BytecodeOp::store64_indexed));
    CHECK(has_op(*bfn, BytecodeOp::store32_indexed));
    CHECK(has_op(*bfn, BytecodeOp::store16_indexed));
    CHECK(has_op(*bfn, BytecodeOp::store8_indexed));
    CHECK(has_op(*bfn, BytecodeOp::load64_indexed));
    CHECK(has_op(*bfn, BytecodeOp::load32_indexed));
    CHECK(has_op(*bfn, BytecodeOp::load16_indexed));
    CHECK(has_op(*bfn, BytecodeOp::load8_indexed));
    CHECK_FALSE(has_op(*bfn, BytecodeOp::index_addr));

    std::string dis = disassemble(*bfn);
    CHECK(dis.find("load64_indexed") != std::string::npos);
    CHECK(dis.find("store64_indexed") != std::string::npos);

    const int64_t expected = 0x1122334455667788LL + 0x12345678LL + 0xABCDLL + 0x7FLL;
    check_same_i64(mod, *fn, {RuntimeValue::from_i64(3)}, expected);
}

TEST_CASE("Bytecode Compiler - Parallel Moves with Multi-Cycles and Fan-Out") {
    Module mod("parallel_moves_mod");
    Builder b(mod);

    // 7 block parameters: a 2-cycle (0<->1), a 3-cycle (2->3->4->2), and fan-out (5<-0, 6<-2).
    Function* fn = mod.create_function("perm_loop", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop = b.append_block("loop");
    BasicBlock* exit = b.append_block("exit");

    b.position_at_end(entry);
    Value* steps = b.add_block_param(entry, Type::i64());
    b.build_br(loop, {
        steps,
        b.build_iconst_i64(10),
        b.build_iconst_i64(20),
        b.build_iconst_i64(30),
        b.build_iconst_i64(40),
        b.build_iconst_i64(50),
        b.build_iconst_i64(60),
        b.build_iconst_i64(70)
    });

    b.position_at_end(loop);
    Value* k = b.add_block_param(loop, Type::i64());
    Value* p0 = b.add_block_param(loop, Type::i64());
    Value* p1 = b.add_block_param(loop, Type::i64());
    Value* p2 = b.add_block_param(loop, Type::i64());
    Value* p3 = b.add_block_param(loop, Type::i64());
    Value* p4 = b.add_block_param(loop, Type::i64());
    Value* p5 = b.add_block_param(loop, Type::i64());
    Value* p6 = b.add_block_param(loop, Type::i64());

    Value* done = b.build_sle(k, b.build_iconst_i64(0));
    Value* k_next = b.build_sub(k, b.build_iconst_i64(1));
    b.build_br_if(done, exit, {}, loop, {k_next, p1, p0, p4, p2, p3, p0, p2});

    b.position_at_end(exit);
    Value* res = b.build_add(
        b.build_add(b.build_mul(p0, b.build_iconst_i64(1)), b.build_mul(p1, b.build_iconst_i64(10))),
        b.build_add(
            b.build_add(b.build_mul(p2, b.build_iconst_i64(100)), b.build_mul(p3, b.build_iconst_i64(1000))),
            b.build_add(b.build_mul(p4, b.build_iconst_i64(10000)),
                        b.build_add(b.build_mul(p5, b.build_iconst_i64(100000)),
                                    b.build_mul(p6, b.build_iconst_i64(1000000))))));
    b.build_ret(res);

    Interpreter oracle;
    oracle.set_module(&mod);
    FastInterpreter fast;
    fast.set_module(&mod);
    for (int64_t s = 0; s <= 6; ++s) {
        std::vector<RuntimeValue> args = {RuntimeValue::from_i64(s)};
        CHECK_EQ(fast.run(*fn, args).as_i64(), oracle.run(*fn, args).as_i64());
    }
}

