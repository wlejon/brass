#include "test_framework.hpp"
#include <brass/debug/source_loc.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/builder.hpp>
#include <brass/codegen/lir.hpp>

TEST_CASE("SourceLoc - DebugLoc basic properties and equality") {
    brass::DebugLoc default_loc;
    CHECK(!default_loc.is_valid());
    CHECK_EQ(default_loc.file_id, 0u);
    CHECK_EQ(default_loc.line, 0u);
    CHECK_EQ(default_loc.column, 0u);
    CHECK_EQ(default_loc.inlined_at_id, 0u);

    brass::DebugLoc loc1(1, 42, 10, 0);
    CHECK(loc1.is_valid());
    CHECK_EQ(loc1.file_id, 1u);
    CHECK_EQ(loc1.line, 42u);
    CHECK_EQ(loc1.column, 10u);
    CHECK_EQ(loc1.inlined_at_id, 0u);

    brass::DebugLoc loc2(1, 42, 10, 0);
    CHECK(loc1 == loc2);

    brass::DebugLoc loc3(1, 42, 11, 0);
    CHECK(loc1 != loc3);

    brass::DebugLoc loc4(1, 42, 10, 2);
    CHECK(loc1 != loc4);
    CHECK_EQ(loc4.inlined_at_id, 2u);
}

TEST_CASE("SourceLoc - DebugContext file interning") {
    brass::DebugContext ctx;
    CHECK_EQ(ctx.file_count(), 0u);

    // Empty path returns 0
    CHECK_EQ(ctx.get_or_add_file(""), 0u);

    // Add first file
    uint32_t id1 = ctx.get_or_add_file("src/main.js");
    CHECK_EQ(id1, 1u);
    CHECK_EQ(ctx.file_count(), 1u);
    CHECK_EQ(ctx.get_file(id1), "src/main.js");
    CHECK_EQ(ctx.get_file_id("src/main.js"), 1u);

    // Re-adding returns same ID
    uint32_t id1_again = ctx.get_or_add_file("src/main.js");
    CHECK_EQ(id1_again, id1);
    CHECK_EQ(ctx.file_count(), 1u);

    // Add second file
    uint32_t id2 = ctx.get_or_add_file("src/util.js");
    CHECK_EQ(id2, 2u);
    CHECK_EQ(ctx.file_count(), 2u);
    CHECK_EQ(ctx.get_file(id2), "src/util.js");

    // Query non-existent
    CHECK_EQ(ctx.get_file_id("non_existent.js"), 0u);
    CHECK_EQ(ctx.get_file(0), "");
    CHECK_EQ(ctx.get_file(999), "");

    // Clear
    ctx.clear();
    CHECK_EQ(ctx.file_count(), 0u);
    CHECK_EQ(ctx.get_file(id1), "");
}

TEST_CASE("SourceLoc - InlinedScope tracking and hierarchy") {
    brass::DebugContext ctx;
    uint32_t f_main = ctx.get_or_add_file("main.js");
    uint32_t f_math = ctx.get_or_add_file("math.js");
    uint32_t f_core = ctx.get_or_add_file("core.js");

    // Call site in main.js calling add(): line 20, col 5
    brass::DebugLoc call_add(f_main, 20, 5);
    uint32_t scope_add = ctx.record_inlined_scope("add", call_add, 0);
    CHECK_EQ(scope_add, 1u);
    CHECK_EQ(ctx.inlined_scope_count(), 1u);

    const brass::InlinedScope* s1 = ctx.get_inlined_scope(scope_add);
    REQUIRE(s1 != nullptr);
    CHECK_EQ(s1->callee_name, "add");
    CHECK_EQ(s1->callsite_loc, call_add);
    CHECK_EQ(s1->parent_inlined_at_id, 0u);

    // Call site inside add() calling abs(): math.js: line 10, col 3
    brass::DebugLoc call_abs(f_math, 10, 3);
    uint32_t scope_abs = ctx.record_inlined_scope("abs", call_abs, scope_add);
    CHECK_EQ(scope_abs, 2u);
    CHECK_EQ(ctx.inlined_scope_count(), 2u);

    const brass::InlinedScope* s2 = ctx.get_inlined_scope(scope_abs);
    REQUIRE(s2 != nullptr);
    CHECK_EQ(s2->callee_name, "abs");
    CHECK_EQ(s2->callsite_loc, call_abs);
    CHECK_EQ(s2->parent_inlined_at_id, scope_add);

    // Call site inside abs() calling clamp(): core.js: line 5, col 1
    brass::DebugLoc call_clamp(f_core, 5, 1);
    uint32_t scope_clamp = ctx.record_inlined_scope("clamp", call_clamp, scope_abs);
    CHECK_EQ(scope_clamp, 3u);

    // Check traversal up the chain
    const brass::InlinedScope* cur = ctx.get_inlined_scope(scope_clamp);
    REQUIRE(cur != nullptr);
    CHECK_EQ(cur->callee_name, "clamp");

    cur = ctx.get_inlined_scope(cur->parent_inlined_at_id);
    REQUIRE(cur != nullptr);
    CHECK_EQ(cur->callee_name, "abs");

    cur = ctx.get_inlined_scope(cur->parent_inlined_at_id);
    REQUIRE(cur != nullptr);
    CHECK_EQ(cur->callee_name, "add");

    CHECK_EQ(cur->parent_inlined_at_id, 0u);
    CHECK(ctx.get_inlined_scope(0) == nullptr);
    CHECK(ctx.get_inlined_scope(100) == nullptr);
}

TEST_CASE("SourceLoc - InlinedScope wrapping") {
    brass::DebugContext ctx;
    uint32_t f1 = ctx.get_or_add_file("outer.js");
    uint32_t f2 = ctx.get_or_add_file("inner.js");

    // Callee has an existing inlined helper
    brass::DebugLoc helper_call(f2, 50, 4);
    uint32_t inner_scope = ctx.record_inlined_scope("sub_helper", helper_call, 0);

    // Outer caller inlines callee at outer.js:100:8
    brass::DebugLoc caller_call(f1, 100, 8);
    uint32_t outer_scope = ctx.record_inlined_scope("callee", caller_call, 0);

    // Wrap inner scope under outer scope
    uint32_t wrapped_scope = ctx.wrap_inlined_scope(inner_scope, outer_scope);
    CHECK_NE(wrapped_scope, inner_scope);

    const brass::InlinedScope* ws = ctx.get_inlined_scope(wrapped_scope);
    REQUIRE(ws != nullptr);
    CHECK_EQ(ws->callee_name, "sub_helper");
    CHECK_EQ(ws->callsite_loc, helper_call);
    CHECK_EQ(ws->parent_inlined_at_id, outer_scope);

    const brass::InlinedScope* ps = ctx.get_inlined_scope(ws->parent_inlined_at_id);
    REQUIRE(ps != nullptr);
    CHECK_EQ(ps->callee_name, "callee");
    CHECK_EQ(ps->callsite_loc, caller_call);
    CHECK_EQ(ps->parent_inlined_at_id, 0u);
}

TEST_CASE("SourceLoc - Instruction and Builder propagation") {
    brass::Module mod("test_mod");
    brass::Builder builder(mod);
    brass::Function* fn = mod.create_function("foo", brass::Type::i32());
    builder.set_function(fn);
    brass::BasicBlock* entry = builder.append_block("entry");
    builder.position_at_end(entry);

    uint32_t f_id = mod.debug_context().get_or_add_file("foo.js");
    brass::DebugLoc loc(f_id, 12, 4);
    builder.set_current_loc(loc);
    CHECK_EQ(builder.current_loc(), loc);

    brass::Value* v1 = builder.build_iconst_i32(100);
    REQUIRE(v1 != nullptr);
    brass::Instruction* inst = v1->defining_instruction();
    REQUIRE(inst != nullptr);
    CHECK(inst->loc().is_valid());
    CHECK_EQ(inst->loc().file_id, f_id);
    CHECK_EQ(inst->loc().line, 12u);
    CHECK_EQ(inst->loc().column, 4u);

    // Clear location
    builder.clear_current_loc();
    CHECK(!builder.current_loc().is_valid());

    brass::Instruction* ret = builder.build_ret(v1);
    REQUIRE(ret != nullptr);
    CHECK(!ret->loc().is_valid());

    // Explicit set_loc on ret
    ret->set_loc(f_id, 13, 2);
    CHECK(ret->loc().is_valid());
    CHECK_EQ(ret->loc().line, 13u);
}
