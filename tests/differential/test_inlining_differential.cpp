#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Inlining - Deep Linear Call Chain") {
    // f4(x) = (x ^ 7) + 3
    // f3(x) = f4(x - 5) * 2
    // f2(x) = f3(x + 10) + 1
    // f1(x) = f2(x * 3) - 4
    // f0(x) = f1(x) + 100

    Module mod_ref("linear_chain_ref");
    Module mod_inline("linear_chain_inline");

    auto build_chain = [](Module& m) {
        // f4
        Function* f4 = m.create_function("f4", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(f4);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c7 = b.build_iconst_i64(7);
            Value* xor_v = b.build_xor(x, c7);
            Value* c3 = b.build_iconst_i64(3);
            b.build_ret(b.build_add(xor_v, c3));
            f4->rebuild_cfg_predecessors();
        }
        // f3
        Function* f3 = m.create_function("f3", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(f3);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c5 = b.build_iconst_i64(5);
            Value* sub = b.build_sub(x, c5);
            Value* call_res = b.build_call("f4", Type::i64(), {sub});
            Value* c2 = b.build_iconst_i64(2);
            b.build_ret(b.build_mul(call_res, c2));
            f3->rebuild_cfg_predecessors();
        }
        // f2
        Function* f2 = m.create_function("f2", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(f2);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c10 = b.build_iconst_i64(10);
            Value* add = b.build_add(x, c10);
            Value* call_res = b.build_call("f3", Type::i64(), {add});
            Value* c1 = b.build_iconst_i64(1);
            b.build_ret(b.build_add(call_res, c1));
            f2->rebuild_cfg_predecessors();
        }
        // f1
        Function* f1 = m.create_function("f1", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(f1);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c3 = b.build_iconst_i64(3);
            Value* mul = b.build_mul(x, c3);
            Value* call_res = b.build_call("f2", Type::i64(), {mul});
            Value* c4 = b.build_iconst_i64(4);
            b.build_ret(b.build_sub(call_res, c4));
            f1->rebuild_cfg_predecessors();
        }
        // f0
        Function* f0 = m.create_function("f0", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(f0);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* call_res = b.build_call("f1", Type::i64(), {x});
            Value* c100 = b.build_iconst_i64(100);
            b.build_ret(b.build_add(call_res, c100));
            f0->rebuild_cfg_predecessors();
        }
    };

    build_chain(mod_ref);
    build_chain(mod_inline);

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_inline));

    // Optimize mod_inline via IPO
    bool inlined = optimize_module_ipo(mod_inline);
    CHECK(inlined);
    REQUIRE(verify_module(mod_inline));

    Interpreter interp;
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod_inline));

    for (int64_t v = -15; v <= 25; ++v) {
        RuntimeValue ref_res = interp.run(mod_ref, "f0", {RuntimeValue::from_i64(v)});
        RuntimeValue jit_res = jit.invoke("f0", {RuntimeValue::from_i64(v)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }
}

TEST_CASE("Differential Inlining - Diamond Call Graph") {
    // Diamond structure: root calls left_fn and right_fn; both call leaf.

    Module mod_ref("diamond_ref");
    Module mod_inline("diamond_inline");

    auto build_diamond = [](Module& m) {
        // leaf: (y) -> (y * 3) + 7
        Function* leaf = m.create_function("leaf", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(leaf);
            BasicBlock* bb = b.append_block("entry");
            Value* y = b.add_block_param(bb, Type::i64());
            Value* c3 = b.build_iconst_i64(3);
            Value* mul = b.build_mul(y, c3);
            Value* c7 = b.build_iconst_i64(7);
            b.build_ret(b.build_add(mul, c7));
            leaf->rebuild_cfg_predecessors();
        }

        // left: (x) -> leaf(x + 2) * 2
        Function* left = m.create_function("left_fn", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(left);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c2 = b.build_iconst_i64(2);
            Value* add = b.build_add(x, c2);
            Value* l_res = b.build_call("leaf", Type::i64(), {add});
            b.build_ret(b.build_mul(l_res, c2));
            left->rebuild_cfg_predecessors();
        }

        // right: (x) -> leaf(x - 4) ^ 15
        Function* right = m.create_function("right_fn", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(right);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c4 = b.build_iconst_i64(4);
            Value* sub = b.build_sub(x, c4);
            Value* r_res = b.build_call("leaf", Type::i64(), {sub});
            Value* c15 = b.build_iconst_i64(15);
            b.build_ret(b.build_xor(r_res, c15));
            right->rebuild_cfg_predecessors();
        }

        // root: (x) -> left(x) + right(x)
        Function* root = m.create_function("root", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(root);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* l = b.build_call("left_fn", Type::i64(), {x});
            Value* r = b.build_call("right_fn", Type::i64(), {x});
            b.build_ret(b.build_add(l, r));
            root->rebuild_cfg_predecessors();
        }
    };

    build_diamond(mod_ref);
    build_diamond(mod_inline);

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_inline));

    bool inlined = optimize_module_ipo(mod_inline);
    CHECK(inlined);
    REQUIRE(verify_module(mod_inline));

    Interpreter interp;
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod_inline));

    for (int64_t v : {-50, -10, 0, 1, 5, 12, 33, 100}) {
        RuntimeValue ref_res = interp.run(mod_ref, "root", {RuntimeValue::from_i64(v)});
        RuntimeValue jit_res = jit.invoke("root", {RuntimeValue::from_i64(v)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }
}

TEST_CASE("Differential Inlining - Loop with Inlined Polynomial Kernel") {
    // poly2(x, a, b, c) = a*x^2 + b*x + c
    // Loop sums poly2(i, 2, 3, 5) for i in [0, count)

    Module mod_ref("loop_poly_ref");
    Module mod_inline("loop_poly_inline");

    auto build_loop_poly = [](Module& m) {
        Function* poly = m.create_function("poly2", Type::i64(), {Type::i64(), Type::i64(), Type::i64(), Type::i64()});
        {
            Builder b(m);
            b.set_function(poly);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, Type::i64());
            Value* a = b.add_block_param(entry, Type::i64());
            Value* b_val = b.add_block_param(entry, Type::i64());
            Value* c = b.add_block_param(entry, Type::i64());

            Value* x2 = b.build_mul(x, x);
            Value* term2 = b.build_mul(a, x2);
            Value* term1 = b.build_mul(b_val, x);
            Value* sum12 = b.build_add(term2, term1);
            Value* res = b.build_add(sum12, c);
            b.build_ret(res);
            poly->rebuild_cfg_predecessors();
        }

        Function* loop_fn = m.create_function("sum_poly", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(loop_fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* loop_hdr = b.create_block("loop_hdr");
            BasicBlock* loop_body = b.create_block("loop_body");
            BasicBlock* exit_bb = b.create_block("exit_bb");

            loop_fn->append_block(loop_hdr);
            loop_fn->append_block(loop_body);
            loop_fn->append_block(exit_bb);

            Value* count = b.add_block_param(entry, Type::i64());
            Value* zero = b.build_iconst_i64(0);
            b.build_br(loop_hdr, {zero, zero});

            b.position_at_end(loop_hdr);
            Value* i = b.add_block_param(loop_hdr, Type::i64());
            Value* acc = b.add_block_param(loop_hdr, Type::i64());
            Value* cond = b.build_slt(i, count);
            b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

            b.position_at_end(loop_body);
            Value* c2 = b.build_iconst_i64(2);
            Value* c3 = b.build_iconst_i64(3);
            Value* c5 = b.build_iconst_i64(5);
            Value* term = b.build_call("poly2", Type::i64(), {i, c2, c3, c5});
            Value* next_acc = b.build_add(acc, term);
            Value* c1 = b.build_iconst_i64(1);
            Value* next_i = b.build_add(i, c1);
            b.build_br(loop_hdr, {next_i, next_acc});

            b.position_at_end(exit_bb);
            Value* final_res = b.add_block_param(exit_bb, Type::i64());
            b.build_ret(final_res);
            loop_fn->rebuild_cfg_predecessors();
        }
    };

    build_loop_poly(mod_ref);
    build_loop_poly(mod_inline);

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_inline));

    bool inlined = optimize_module_ipo(mod_inline);
    CHECK(inlined);
    REQUIRE(verify_module(mod_inline));

    Interpreter interp;
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod_inline));

    for (int64_t n : {0, 1, 3, 7, 12, 20}) {
        RuntimeValue ref_res = interp.run(mod_ref, "sum_poly", {RuntimeValue::from_i64(n)});
        RuntimeValue jit_res = jit.invoke("sum_poly", {RuntimeValue::from_i64(n)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }
}
