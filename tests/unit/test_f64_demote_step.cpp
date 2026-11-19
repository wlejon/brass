#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <limits>
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
        if (!bb) continue;
        if (std::string_view(bb->name()).find("_f64c") != std::string_view::npos) {
            has_copy = true;
            CHECK(bb->is_cold());
        } else {
            CHECK_FALSE(bb->is_cold());
        }
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

TEST_CASE("F64 Demote - Entry-Param Value Versioning Marks Fallback _f64v Blocks Cold") {
    Module mod("test_demote_entry_cold");
    Builder b(mod);
    Function* fn = mod.create_function("scaled_sum", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());
    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");

    Value* zero = b.build_fconst_f64(0.0);
    b.build_br(b1, {zero, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* acc = b.add_block_param(b1, Type::f64());
    Value* i = b.add_block_param(b1, Type::f64());
    b.build_br_if(b.build_slt(i, n), b2, {}, b3, {acc});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* scaled = b.build_mul(i, b.build_fconst_f64(3.0));
    Value* as_int = b.build_sitofp_f64_i64(b.build_fptosi_i64(scaled));
    b.build_br(b1, {b.build_add(acc, as_int), b.build_add(i, b.build_fconst_f64(1.0))});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* r = b.add_block_param(b3, Type::f64());
    b.build_ret(r);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    CHECK(f64_demote_pass(*fn, {}));
    REQUIRE(verify_function(*fn, &diag));
    size_t cold_f64v = 0;
    for (const BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        if (std::string_view(bb->name()).find("_f64v") != std::string_view::npos) {
            ++cold_f64v;
            CHECK(bb->is_cold());
        } else {
            CHECK_FALSE(bb->is_cold());
        }
    }
    CHECK(cold_f64v > size_t{0});
}

TEST_CASE("F64 Demote - Canonicalizes i32 IV vs f64 Invariant Loop Bounds Across All Relations and Edge Values") {
    for (Opcode cmp : {Opcode::slt, Opcode::sle, Opcode::sgt, Opcode::sge}) {
        Module mod("test_i32_f64_bound");
        Builder b(mod);
        // count_trips(start: i32, limit: f64, step: i32) -> i32
        Function* fn = mod.create_function("count_trips", Type::i32(), {Type::i32(), Type::f64(), Type::i32()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* start = b.add_block_param(entry, Type::i32());
        Value* limit = b.add_block_param(entry, Type::f64());
        Value* step = b.add_block_param(entry, Type::i32());
        BasicBlock* hdr = b.create_block("hdr");
        BasicBlock* body = b.create_block("body");
        BasicBlock* done = b.create_block("done");
        b.build_br(hdr, {start, b.build_iconst_i32(0)});

        fn->append_block(hdr);
        b.position_at_end(hdr);
        Value* iv = b.add_block_param(hdr, Type::i32());
        Value* trips = b.add_block_param(hdr, Type::i32());
        Value* iv_f = b.build_sitofp_f64_i32(iv);
        Value* cond = nullptr;
        switch (cmp) {
            case Opcode::slt: cond = b.build_slt(iv_f, limit); break;
            case Opcode::sle: cond = b.build_sle(iv_f, limit); break;
            case Opcode::sgt: cond = b.build_sgt(iv_f, limit); break;
            case Opcode::sge: cond = b.build_sge(iv_f, limit); break;
            default: break;
        }
        b.build_br_if(cond, body, {}, done, {trips});

        fn->append_block(body);
        b.position_at_end(body);
        Value* trips2 = b.build_add(trips, b.build_iconst_i32(1));
        Value* iv2 = b.build_add(iv, step);
        // Safety cap so +Inf bounds never spin infinitely when testing.
        b.build_br_if(b.build_sge(trips2, b.build_iconst_i32(25)), done, {trips2}, hdr, {iv2, trips2});

        fn->append_block(done);
        Value* ret_v = b.add_block_param(done, Type::i32());
        b.position_at_end(done);
        b.build_ret(ret_v);

        fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_function(*fn, &diag));

        CHECK(f64_demote_pass(*fn, {}));
        REQUIRE(verify_function(*fn, &diag));
        size_t sitofp_in_hdr = 0;
        for (const Instruction* inst : *hdr) {
            if (inst && inst->opcode() == Opcode::sitofp_f64_i32) ++sitofp_in_hdr;
        }
        CHECK_EQ(sitofp_in_hdr, size_t{0});

        LoopOptOptions opt;
        optimize_function_loops(*fn, opt);
        REQUIRE(verify_function(*fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));
        auto f = jit.get_function_ptr<int32_t (*)(int32_t, double, int32_t)>("count_trips");
        REQUIRE(f != nullptr);

        auto ref = [&](int32_t s, double lim, int32_t st) -> int32_t {
            int64_t i = s;
            int32_t t = 0;
            while (true) {
                const double d = static_cast<double>(static_cast<int32_t>(i));
                const bool ok = cmp == Opcode::slt ? (d < lim)
                              : cmp == Opcode::sle ? (d <= lim)
                              : cmp == Opcode::sgt ? (d > lim)
                                                   : (d >= lim);
                if (!ok) return t;
                ++t;
                i = static_cast<int32_t>(i + st);
                if (t >= 25) return t;
            }
        };

        const int32_t st = (cmp == Opcode::slt || cmp == Opcode::sle) ? 1 : -1;
        for (double lim : {5.0, 5.5, 4.999, -0.5, -5.5, 0.0, -0.0,
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           2147483647.0, 2147483647.5, -2147483648.0, -2147483648.5}) {
            CHECK_EQ(f(0, lim, st), ref(0, lim, st));
            CHECK_EQ(f(2, lim, st), ref(2, lim, st));
            CHECK_EQ(f(2147483640, lim, 1), ref(2147483640, lim, 1));
            CHECK_EQ(f(-2147483642, lim, -1), ref(-2147483642, lim, -1));
        }
    }
}
