#include "test_framework.hpp"
#include "../differential/diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <climits>
#include <cstdint>
#include <sstream>

using namespace brass;
using namespace brass::test;

TEST_CASE("IL Expressibility - Switch i32 basic and fallback") {
    Module mod("switch_mod_i32");
    Function* fn = mod.create_function("eval_switch_i32", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_zero = b.create_block("bb_zero");
    BasicBlock* bb_one = b.create_block("bb_one");
    BasicBlock* bb_neg = b.create_block("bb_neg");
    BasicBlock* bb_def = b.create_block("bb_def");

    fn->append_block(bb_zero);
    fn->append_block(bb_one);
    fn->append_block(bb_neg);
    fn->append_block(bb_def);

    Value* val = b.add_block_param(entry, Type::i32());

    b.build_switch(val, bb_def, {
        SwitchCase(0, bb_zero),
        SwitchCase(1, bb_one),
        SwitchCase(-5, bb_neg)
    });

    b.position_at_end(bb_zero);
    b.build_ret(b.build_iconst_i32(100));

    b.position_at_end(bb_one);
    b.build_ret(b.build_iconst_i32(200));

    b.position_at_end(bb_neg);
    b.build_ret(b.build_iconst_i32(300));

    b.position_at_end(bb_def);
    b.build_ret(b.build_iconst_i32(999));

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    assert_diff(mod, "eval_switch_i32", {RuntimeValue::from_i32(0)});   // 100
    assert_diff(mod, "eval_switch_i32", {RuntimeValue::from_i32(1)});   // 200
    assert_diff(mod, "eval_switch_i32", {RuntimeValue::from_i32(-5)});  // 300
    assert_diff(mod, "eval_switch_i32", {RuntimeValue::from_i32(42)});  // 999
    assert_diff(mod, "eval_switch_i32", {RuntimeValue::from_i32(-1)});  // 999
}

TEST_CASE("IL Expressibility - Switch i64 with large values and block arguments") {
    Module mod("switch_mod_i64");
    Function* fn = mod.create_function("eval_switch_i64", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_small = b.create_block("bb_small");
    BasicBlock* bb_large = b.create_block("bb_large");
    BasicBlock* bb_def = b.create_block("bb_def");

    fn->append_block(bb_small);
    fn->append_block(bb_large);
    fn->append_block(bb_def);

    Value* val = b.add_block_param(entry, Type::i64());
    Value* extra = b.add_block_param(entry, Type::i64());

    Value* p_small = b.add_block_param(bb_small, Type::i64());
    Value* p_large = b.add_block_param(bb_large, Type::i64());
    Value* p_def = b.add_block_param(bb_def, Type::i64());

    int64_t large_case = 0x123456789ABCDEF0LL;

    b.build_switch(val, bb_def, {extra}, {
        SwitchCase(10, bb_small, {extra}),
        SwitchCase(large_case, bb_large, {extra})
    });

    b.position_at_end(bb_small);
    b.build_ret(b.build_add(p_small, b.build_iconst_i64(1)));

    b.position_at_end(bb_large);
    b.build_ret(b.build_add(p_large, b.build_iconst_i64(2)));

    b.position_at_end(bb_def);
    b.build_ret(b.build_add(p_def, b.build_iconst_i64(99)));

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    assert_diff(mod, "eval_switch_i64", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(50)});
    assert_diff(mod, "eval_switch_i64", {RuntimeValue::from_i64(large_case), RuntimeValue::from_i64(50)});
    assert_diff(mod, "eval_switch_i64", {RuntimeValue::from_i64(0), RuntimeValue::from_i64(50)});
}

TEST_CASE("IL Expressibility - Switch Textual MIR Round-trip") {
    std::string_view mir_src =
        "func @dispatch_test(%v: i32) -> i32 {\n"
        "entry(%v: i32):\n"
        "  switch.i32 %v, default: bb_def(%v), [1: bb1(%v), 2: bb2(%v)]\n"
        "bb1(%p1: i32):\n"
        "  %c10 = iconst.i32 10\n"
        "  ret %c10\n"
        "bb2(%p2: i32):\n"
        "  %c20 = iconst.i32 20\n"
        "  ret %c20\n"
        "bb_def(%pd: i32):\n"
        "  %c0 = iconst.i32 0\n"
        "  ret %c0\n"
        "}\n";

    DiagnosticReporter diag;
    auto parsed_mod = parse_module(mir_src, &diag);
    if (!parsed_mod) {
        std::cerr << "Parser error: " << diag.format_all() << "\n";
    }
    REQUIRE(parsed_mod != nullptr);
    REQUIRE(verify_module(*parsed_mod, &diag));

    assert_diff(*parsed_mod, "dispatch_test", {RuntimeValue::from_i32(1)});
    assert_diff(*parsed_mod, "dispatch_test", {RuntimeValue::from_i32(2)});
    assert_diff(*parsed_mod, "dispatch_test", {RuntimeValue::from_i32(99)});
}

TEST_CASE("IL Expressibility - Float Int Conversions and Bitcasts") {
    Module mod("conv_mod");
    Builder b(mod);

    // 1. sitofp_f64_i32
    {
        Function* fn = mod.create_function("test_sitofp_i32", Type::f64(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i32());
        Value* f = b.build_sitofp_f64_i32(x);
        b.build_ret(f);
    }
    // 2. sitofp_f64_i64
    {
        Function* fn = mod.create_function("test_sitofp_i64", Type::f64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* f = b.build_sitofp_f64_i64(x);
        b.build_ret(f);
    }
    // 3. fptosi_i32
    {
        Function* fn = mod.create_function("test_fptosi_i32", Type::i32(), {Type::f64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::f64());
        Value* i = b.build_fptosi_i32(x);
        b.build_ret(i);
    }
    // 4. fptosi_i64
    {
        Function* fn = mod.create_function("test_fptosi_i64", Type::i64(), {Type::f64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::f64());
        Value* i = b.build_fptosi_i64(x);
        b.build_ret(i);
    }
    // 5. bitcast roundtrip
    {
        Function* fn = mod.create_function("test_bitcast_roundtrip", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* f = b.build_bitcast_f64_i64(x);
        Value* out = b.build_bitcast_i64_f64(f);
        b.build_ret(out);
    }
    // 6. trunc, zext, sext
    {
        Function* fn = mod.create_function("test_ext_trunc", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* tr = b.build_trunc_i32(x);
        Value* sx = b.build_sext_i64(tr);
        b.build_ret(sx);
    }

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    assert_diff(mod, "test_sitofp_i32", {RuntimeValue::from_i32(0)});
    assert_diff(mod, "test_sitofp_i32", {RuntimeValue::from_i32(-42)});
    assert_diff(mod, "test_sitofp_i32", {RuntimeValue::from_i32(INT32_MAX)});
    assert_diff(mod, "test_sitofp_i32", {RuntimeValue::from_i32(INT32_MIN)});

    assert_diff(mod, "test_sitofp_i64", {RuntimeValue::from_i64(0)});
    assert_diff(mod, "test_sitofp_i64", {RuntimeValue::from_i64(-12345678901234LL)});
    assert_diff(mod, "test_sitofp_i64", {RuntimeValue::from_i64(1LL << 50)});

    assert_diff(mod, "test_fptosi_i32", {RuntimeValue::from_f64(0.0)});
    assert_diff(mod, "test_fptosi_i32", {RuntimeValue::from_f64(42.8)});
    assert_diff(mod, "test_fptosi_i32", {RuntimeValue::from_f64(-42.8)});

    assert_diff(mod, "test_fptosi_i64", {RuntimeValue::from_f64(1e12 + 0.5)});
    assert_diff(mod, "test_fptosi_i64", {RuntimeValue::from_f64(-1e12 - 0.5)});

    // Bitcasts: quiet NaN, signaling NaN, -0.0, subnormals
    assert_diff(mod, "test_bitcast_roundtrip", {RuntimeValue::from_i64(0x7FF8000000000001ULL)}); // NaN
    assert_diff(mod, "test_bitcast_roundtrip", {RuntimeValue::from_i64(0x8000000000000000ULL)}); // -0.0
    assert_diff(mod, "test_bitcast_roundtrip", {RuntimeValue::from_i64(0x0000000000000001ULL)}); // subnormal

    assert_diff(mod, "test_ext_trunc", {RuntimeValue::from_i64(0x0000000080000000ULL)}); // Sign bit in 32-bit sets high 32 bits
    assert_diff(mod, "test_ext_trunc", {RuntimeValue::from_i64(0x000000007FFFFFFFULL)});
}

TEST_CASE("IL Expressibility - Multi-Argument Function Calls (Stack Arguments & Alignment)") {
    Module mod("multi_arg_mod");
    Builder b(mod);

    // Callee with 8 i64 arguments
    Function* callee8 = mod.create_function("callee8", Type::i64(), {
        Type::i64(), Type::i64(), Type::i64(), Type::i64(),
        Type::i64(), Type::i64(), Type::i64(), Type::i64()
    });
    {
        b.set_function(callee8);
        BasicBlock* bb = b.append_block("entry");
        std::vector<Value*> args;
        for (int i = 0; i < 8; ++i) {
            args.push_back(b.add_block_param(bb, Type::i64()));
        }
        Value* sum = args[0];
        for (size_t i = 1; i < args.size(); ++i) {
            sum = b.build_add(sum, args[i]);
        }
        b.build_ret(sum);
    }

    // Caller passing 8 arguments
    Function* caller8 = mod.create_function("caller8", Type::i64(), {Type::i64()});
    {
        b.set_function(caller8);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* r = b.build_call("callee8", Type::i64(), {
            x,
            b.build_iconst_i64(10),
            b.build_iconst_i64(20),
            b.build_iconst_i64(30),
            b.build_iconst_i64(40),
            b.build_iconst_i64(50),
            b.build_iconst_i64(60),
            b.build_iconst_i64(70)
        });
        b.build_ret(r);
    }

    // Callee with 10 mixed i64 and f64 arguments
    Function* callee10 = mod.create_function("callee10", Type::f64(), {
        Type::i64(), Type::f64(), Type::i64(), Type::f64(),
        Type::i64(), Type::f64(), Type::i64(), Type::f64(),
        Type::i64(), Type::f64()
    });
    {
        b.set_function(callee10);
        BasicBlock* bb = b.append_block("entry");
        std::vector<Value*> params;
        for (int i = 0; i < 10; ++i) {
            params.push_back(b.add_block_param(bb, (i % 2 == 0) ? Type::i64() : Type::f64()));
        }
        Value* total = params[1];
        for (int i = 0; i < 10; ++i) {
            if (i == 1) continue;
            Value* v = (i % 2 == 0) ? b.build_sitofp_f64_i64(params[i]) : params[i];
            total = b.build_add(total, v);
        }
        b.build_ret(total);
    }

    // Caller passing 10 arguments
    Function* caller10 = mod.create_function("caller10", Type::f64(), {Type::f64()});
    {
        b.set_function(caller10);
        BasicBlock* bb = b.append_block("entry");
        Value* f = b.add_block_param(bb, Type::f64());
        Value* res = b.build_call("callee10", Type::f64(), {
            b.build_iconst_i64(1), f,
            b.build_iconst_i64(2), b.build_fconst_f64(2.5),
            b.build_iconst_i64(3), b.build_fconst_f64(3.5),
            b.build_iconst_i64(4), b.build_fconst_f64(4.5),
            b.build_iconst_i64(5), b.build_fconst_f64(5.5)
        });
        b.build_ret(res);
    }

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    assert_diff(mod, "caller8", {RuntimeValue::from_i64(1)});
    assert_diff(mod, "caller8", {RuntimeValue::from_i64(100)});
    assert_diff(mod, "caller10", {RuntimeValue::from_f64(1.5)});
}

TEST_CASE("IL Expressibility - Overflow-Checked Arithmetic Primitives") {
    Module mod("overflow_mod");
    Builder b(mod);

    // 1. sadd_overflow i32
    {
        Function* fn = mod.create_function("test_sadd_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_sadd_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 2. ssub_overflow i32
    {
        Function* fn = mod.create_function("test_ssub_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_ssub_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 3. smul_overflow i32
    {
        Function* fn = mod.create_function("test_smul_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_smul_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 4. uadd_overflow i32
    {
        Function* fn = mod.create_function("test_uadd_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_uadd_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 5. usub_overflow i32
    {
        Function* fn = mod.create_function("test_usub_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_usub_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 6. umul_overflow i32
    {
        Function* fn = mod.create_function("test_umul_ovf_i32", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i32());
        Value* rhs = b.add_block_param(bb, Type::i32());
        Value* ovf = b.build_umul_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 7. sadd_overflow i64
    {
        Function* fn = mod.create_function("test_sadd_ovf_i64", Type::i32(), {Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i64());
        Value* rhs = b.add_block_param(bb, Type::i64());
        Value* ovf = b.build_sadd_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 8. smul_overflow i64
    {
        Function* fn = mod.create_function("test_smul_ovf_i64", Type::i32(), {Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i64());
        Value* rhs = b.add_block_param(bb, Type::i64());
        Value* ovf = b.build_smul_overflow(lhs, rhs);
        b.build_ret(ovf);
    }
    // 9. umul_overflow i64
    {
        Function* fn = mod.create_function("test_umul_ovf_i64", Type::i32(), {Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* lhs = b.add_block_param(bb, Type::i64());
        Value* rhs = b.add_block_param(bb, Type::i64());
        Value* ovf = b.build_umul_overflow(lhs, rhs);
        b.build_ret(ovf);
    }

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // sadd i32
    assert_diff(mod, "test_sadd_ovf_i32", {RuntimeValue::from_i32(10), RuntimeValue::from_i32(20)}); // 0
    assert_diff(mod, "test_sadd_ovf_i32", {RuntimeValue::from_i32(INT32_MAX), RuntimeValue::from_i32(1)}); // 1
    assert_diff(mod, "test_sadd_ovf_i32", {RuntimeValue::from_i32(INT32_MIN), RuntimeValue::from_i32(-1)}); // 1

    // ssub i32
    assert_diff(mod, "test_ssub_ovf_i32", {RuntimeValue::from_i32(20), RuntimeValue::from_i32(10)}); // 0
    assert_diff(mod, "test_ssub_ovf_i32", {RuntimeValue::from_i32(INT32_MIN), RuntimeValue::from_i32(1)}); // 1
    assert_diff(mod, "test_ssub_ovf_i32", {RuntimeValue::from_i32(INT32_MAX), RuntimeValue::from_i32(-1)}); // 1

    // smul i32
    assert_diff(mod, "test_smul_ovf_i32", {RuntimeValue::from_i32(100), RuntimeValue::from_i32(200)}); // 0
    assert_diff(mod, "test_smul_ovf_i32", {RuntimeValue::from_i32(100000), RuntimeValue::from_i32(100000)}); // 1

    // uadd i32
    assert_diff(mod, "test_uadd_ovf_i32", {RuntimeValue::from_i32(10), RuntimeValue::from_i32(20)}); // 0
    assert_diff(mod, "test_uadd_ovf_i32", {RuntimeValue::from_i32(static_cast<int32_t>(0xFFFFFFFFU)), RuntimeValue::from_i32(1)}); // 1

    // usub i32
    assert_diff(mod, "test_usub_ovf_i32", {RuntimeValue::from_i32(20), RuntimeValue::from_i32(10)}); // 0
    assert_diff(mod, "test_usub_ovf_i32", {RuntimeValue::from_i32(10), RuntimeValue::from_i32(20)}); // 1

    // umul i32
    assert_diff(mod, "test_umul_ovf_i32", {RuntimeValue::from_i32(100), RuntimeValue::from_i32(200)}); // 0
    assert_diff(mod, "test_umul_ovf_i32", {RuntimeValue::from_i32(0), RuntimeValue::from_i32(static_cast<int32_t>(0xFFFFFFFFU))}); // 0
    assert_diff(mod, "test_umul_ovf_i32", {RuntimeValue::from_i32(static_cast<int32_t>(0x80000000U)), RuntimeValue::from_i32(2)}); // 1

    // sadd i64
    assert_diff(mod, "test_sadd_ovf_i64", {RuntimeValue::from_i64(10), RuntimeValue::from_i64(20)}); // 0
    assert_diff(mod, "test_sadd_ovf_i64", {RuntimeValue::from_i64(INT64_MAX), RuntimeValue::from_i64(1)}); // 1

    // smul i64
    assert_diff(mod, "test_smul_ovf_i64", {RuntimeValue::from_i64(1000), RuntimeValue::from_i64(2000)}); // 0
    assert_diff(mod, "test_smul_ovf_i64", {RuntimeValue::from_i64(1LL << 35), RuntimeValue::from_i64(1LL << 35)}); // 1

    // umul i64
    assert_diff(mod, "test_umul_ovf_i64", {RuntimeValue::from_i64(1000), RuntimeValue::from_i64(2000)}); // 0
    assert_diff(mod, "test_umul_ovf_i64", {RuntimeValue::from_i64(0), RuntimeValue::from_i64(static_cast<int64_t>(UINT64_MAX))}); // 0
    assert_diff(mod, "test_umul_ovf_i64", {RuntimeValue::from_i64(1ULL << 35), RuntimeValue::from_i64(1ULL << 35)}); // 1
}
