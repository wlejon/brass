#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/write_barrier_elim.hpp>

#include <iostream>
#include <memory>
#include <string>

using namespace brass;

namespace {

size_t count_write_barriers(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::write_barrier) {
                count++;
            }
        }
    }
    return count;
}

} // namespace

TEST_CASE("WBE - Non-pointer value elimination") {
    Module mod("test_wbe_non_ptr");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());

    Value* int_val = b.build_iconst_i64(42);
    // Write barrier on integer value
    b.build_write_barrier(obj, int_val);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 0ULL);
    CHECK_EQ(wbe.stats().eliminated_non_pointer, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Young provenance allocation elimination") {
    Module mod("test_wbe_young");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::gcref(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* child = b.add_block_param(entry, Type::gcref());

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask1 = b.build_iconst_i64(1);
    Value* tag1 = b.build_iconst_i32(1);

    // obj is freshly allocated in nursery
    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask1, tag1});

    // Write barrier storing child into young obj
    b.build_write_barrier(obj, child);
    b.build_ret(obj);

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 0ULL);
    CHECK_EQ(wbe.stats().eliminated_young_provenance, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Redundant intra-block write barrier elimination") {
    Module mod("test_wbe_redundant");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref(), Type::gcref(), Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());
    Value* val1 = b.add_block_param(entry, Type::gcref());
    Value* val2 = b.add_block_param(entry, Type::gcref());

    // The card is marked only when the stored value is young, so a barrier
    // for a different value is not made redundant by the first one; a
    // repeat of the same object/value pair is.
    b.build_write_barrier(obj, val1);
    b.build_write_barrier(obj, val2);
    b.build_write_barrier(obj, val1);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 3ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 2ULL);
    CHECK_EQ(wbe.stats().eliminated_redundant, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 2U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Preserving required cross-generation barrier") {
    Module mod("test_wbe_preserve");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref(), Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());
    Value* child = b.add_block_param(entry, Type::gcref());

    b.build_write_barrier(obj, child);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(!changed);
    CHECK_EQ(count_write_barriers(*fn), 1ULL);
    CHECK_EQ(wbe.stats().remaining_barriers, 1U);
    CHECK_EQ(wbe.stats().total_eliminated(), 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

// The facts flow across blocks: they hold at a block when they hold at the
// end of every predecessor, and any GC point (a call that may collect)
// clears them on its path.
namespace {

std::unique_ptr<Module> parse_wbe(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    REQUIRE(verify_module(*mod, &vdiag));
    return mod;
}

// `@f`: a barrier in entry (FIRST), a diamond whose left arm holds LEFT, and
// a barrier at the join.
std::string diamond(const char* first, const char* left) {
    std::string s = R"(module @wbe_cfg
extern @wbe_ext

func @f(%o: gcref, %v: gcref, %c: i64) -> i64 {
entry:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %k = iconst.i32 2
  %y = call.gcref @brass_gc_alloc(%sz, %z, %k)
)";
    s += first;
    s += R"(
  %one = iconst.i64 1
  %two = iconst.i64 2
  %go = eq.i64 %c, %z
  br_if %go, a, b
a:
)";
    s += left;
    s += R"(
  br j(%one)
b:
  br j(%two)
j(%p: i64):
  write_barrier %o, %v
  write_barrier %y, %v
  write_barrier %o, %p
  ret %z
}
)";
    return s;
}

} // namespace

TEST_CASE("WBE - Facts cross blocks when every path keeps them") {
    auto mod = parse_wbe(diamond("  write_barrier %o, %v", "  %w = add.i64 %c, %c").c_str());
    Function* fn = mod->get_function("f");
    WriteBarrierElimination wbe;
    CHECK(wbe.run_on_function(*fn));
    // (o, v) is barriered on every path; y is young on every path; p is an
    // integer on both incoming edges.
    CHECK_EQ(wbe.stats().eliminated_redundant, 1U);
    CHECK_EQ(wbe.stats().eliminated_young_provenance, 1U);
    CHECK_EQ(wbe.stats().eliminated_non_pointer, 1U);
    CHECK_EQ(count_write_barriers(*fn), 1ULL);
    DiagnosticReporter diag;
    CHECK(verify_module(*mod, &diag));
}

TEST_CASE("WBE - A GC point on one path keeps the join's barriers") {
    auto mod = parse_wbe(diamond("  write_barrier %o, %v", "  %w = call.i64 @wbe_ext(%c)").c_str());
    Function* fn = mod->get_function("f");
    WriteBarrierElimination wbe;
    wbe.run_on_function(*fn);
    // A collection may promote y and must see the card of (o, v) marked
    // again; p is still an integer.
    CHECK_EQ(wbe.stats().eliminated_redundant, 0U);
    CHECK_EQ(wbe.stats().eliminated_young_provenance, 0U);
    CHECK_EQ(wbe.stats().eliminated_non_pointer, 1U);
    CHECK_EQ(count_write_barriers(*fn), 3ULL);
    DiagnosticReporter diag;
    CHECK(verify_module(*mod, &diag));
}
