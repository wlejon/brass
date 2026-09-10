#include "test_framework.hpp"
#include <brass/debug/debug_section.hpp>
#include <brass/debug/symbolicator.hpp>

TEST_CASE("Symbolicator - Binary section serialization and deserialization roundtrip") {
    brass::DebugContext ctx;
    uint32_t f_main = ctx.get_or_add_file("app/main.js");
    uint32_t f_helper = ctx.get_or_add_file("app/helper.js");
    uint32_t f_math = ctx.get_or_add_file("app/math.js");

    brass::DebugLoc callsite_helper(f_main, 40, 10);
    uint32_t scope_helper = ctx.record_inlined_scope("computeHelper", callsite_helper, 0);

    brass::DebugLoc callsite_math(f_helper, 15, 2);
    uint32_t scope_math = ctx.record_inlined_scope("fastMul", callsite_math, scope_helper);

    std::vector<brass::FunctionDebugTable> tables;

    // Function 1: main
    brass::FunctionDebugTable fn_main("main", 120);
    fn_main.add_line_entry(0, brass::DebugLoc(f_main, 35, 1));
    fn_main.add_line_entry(24, brass::DebugLoc(f_math, 8, 4, scope_math));
    fn_main.add_line_entry(60, brass::DebugLoc(f_helper, 20, 5, scope_helper));
    fn_main.add_line_entry(95, brass::DebugLoc(f_main, 42, 2));
    tables.push_back(fn_main);

    // Function 2: helper
    brass::FunctionDebugTable fn_helper("standalone_helper", 48);
    fn_helper.add_line_entry(0, brass::DebugLoc(f_helper, 1, 1));
    fn_helper.add_line_entry(16, brass::DebugLoc(f_helper, 5, 8));
    tables.push_back(fn_helper);

    // Serialize
    std::vector<uint8_t> binary_data = brass::serialize_debug_section(ctx, tables);
    CHECK(!binary_data.empty());
    CHECK(binary_data.size() >= 40);

    // Verify magic and version
    uint32_t magic = static_cast<uint32_t>(binary_data[0]) |
                     (static_cast<uint32_t>(binary_data[1]) << 8) |
                     (static_cast<uint32_t>(binary_data[2]) << 16) |
                     (static_cast<uint32_t>(binary_data[3]) << 24);
    CHECK_EQ(magic, brass::kDebugSectionMagic);

    // Deserialize into fresh structures
    brass::DebugContext loaded_ctx;
    std::vector<brass::FunctionDebugTable> loaded_tables;
    bool ok = brass::deserialize_debug_section(binary_data.data(), binary_data.size(), loaded_ctx, loaded_tables);
    REQUIRE(ok);

    // Verify files
    CHECK_EQ(loaded_ctx.file_count(), 3u);
    CHECK_EQ(loaded_ctx.get_file(1), "app/main.js");
    CHECK_EQ(loaded_ctx.get_file(2), "app/helper.js");
    CHECK_EQ(loaded_ctx.get_file(3), "app/math.js");

    // Verify scopes
    CHECK_EQ(loaded_ctx.inlined_scope_count(), 2u);
    const auto* s1 = loaded_ctx.get_inlined_scope(1);
    REQUIRE(s1 != nullptr);
    CHECK_EQ(s1->callee_name, "computeHelper");
    CHECK_EQ(s1->callsite_loc, callsite_helper);
    CHECK_EQ(s1->parent_inlined_at_id, 0u);

    const auto* s2 = loaded_ctx.get_inlined_scope(2);
    REQUIRE(s2 != nullptr);
    CHECK_EQ(s2->callee_name, "fastMul");
    CHECK_EQ(s2->callsite_loc, callsite_math);
    CHECK_EQ(s2->parent_inlined_at_id, 1u);

    // Verify tables
    REQUIRE_EQ(loaded_tables.size(), 2u);
    CHECK_EQ(loaded_tables[0].function_name(), "main");
    CHECK_EQ(loaded_tables[0].code_size(), 120u);
    REQUIRE_EQ(loaded_tables[0].line_entries().size(), 4u);
    CHECK(loaded_tables[0].line_entries() == fn_main.line_entries());

    CHECK_EQ(loaded_tables[1].function_name(), "standalone_helper");
    CHECK_EQ(loaded_tables[1].code_size(), 48u);
    REQUIRE_EQ(loaded_tables[1].line_entries().size(), 2u);
    CHECK(loaded_tables[1].line_entries() == fn_helper.line_entries());
}

TEST_CASE("Symbolicator - Corrupt binary debug section handling") {
    brass::DebugContext ctx;
    std::vector<brass::FunctionDebugTable> tables;

    // Null or too small
    CHECK(!brass::deserialize_debug_section(nullptr, 0, ctx, tables));
    uint8_t tiny[10] = {0};
    CHECK(!brass::deserialize_debug_section(tiny, sizeof(tiny), ctx, tables));

    // Bad magic
    uint8_t bad_magic[40] = {0};
    CHECK(!brass::deserialize_debug_section(bad_magic, sizeof(bad_magic), ctx, tables));
}

TEST_CASE("Symbolicator - StackFrame to_string and StackTrace format") {
    brass::StackFrame f1;
    f1.function_name = "renderComponent";
    f1.file = "ui/view.js";
    f1.line = 55;
    f1.column = 12;
    f1.is_inlined = false;

    CHECK_EQ(f1.to_string(), "    at renderComponent (ui/view.js:55:12)");

    brass::StackFrame f2;
    f2.function_name = "formatText";
    f2.file = "ui/text.js";
    f2.line = 18;
    f2.column = 4;
    f2.is_inlined = true;

    CHECK_EQ(f2.to_string(), "    at formatText (ui/text.js:18:4)");

    brass::StackTrace trace;
    trace.frames.push_back(f2);
    trace.frames.push_back(f1);

    std::string expected =
        "    at formatText (ui/text.js:18:4)\n"
        "    at renderComponent (ui/view.js:55:12)\n";
    CHECK_EQ(trace.format(), expected);
}

TEST_CASE("Symbolicator - Inlined callsite expansion") {
    brass::DebugContext ctx;
    uint32_t f_main = ctx.get_or_add_file("main.js");
    uint32_t f_engine = ctx.get_or_add_file("engine.js");
    uint32_t f_kernel = ctx.get_or_add_file("kernel.js");

    // Scope 1: engineRun inlined in main.js at line 100, col 5
    uint32_t s_engine = ctx.record_inlined_scope("engineRun", brass::DebugLoc(f_main, 100, 5), 0);
    // Scope 2: kernelDispatch inlined in engine.js at line 45, col 8
    uint32_t s_kernel = ctx.record_inlined_scope("kernelDispatch", brass::DebugLoc(f_engine, 45, 8), s_engine);

    brass::FunctionDebugTable fn("main", 200);
    // Non-inlined entry
    fn.add_line_entry(0, brass::DebugLoc(f_main, 90, 1, 0));
    // 1-level inlined entry (inside engineRun)
    fn.add_line_entry(30, brass::DebugLoc(f_engine, 30, 2, s_engine));
    // 2-level inlined entry (inside kernelDispatch)
    fn.add_line_entry(80, brass::DebugLoc(f_kernel, 12, 16, s_kernel));

    brass::Symbolicator symbolicator(ctx, {fn});

    // 1. Symbolize offset 10 (inside non-inlined main)
    brass::StackTrace t1 = symbolicator.symbolize_offset("main", 10);
    REQUIRE_EQ(t1.frame_count(), 1u);
    CHECK_EQ(t1.frames[0].function_name, "main");
    CHECK_EQ(t1.frames[0].file, "main.js");
    CHECK_EQ(t1.frames[0].line, 90u);
    CHECK(!t1.frames[0].is_inlined);

    // 2. Symbolize offset 40 (inside 1-level inlined engineRun)
    brass::StackTrace t2 = symbolicator.symbolize_offset("main", 40);
    REQUIRE_EQ(t2.frame_count(), 2u);
    // Frame 0: leaf inlined callee
    CHECK_EQ(t2.frames[0].function_name, "engineRun");
    CHECK_EQ(t2.frames[0].file, "engine.js");
    CHECK_EQ(t2.frames[0].line, 30u);
    CHECK(t2.frames[0].is_inlined);
    // Frame 1: caller main
    CHECK_EQ(t2.frames[1].function_name, "main");
    CHECK_EQ(t2.frames[1].file, "main.js");
    CHECK_EQ(t2.frames[1].line, 100u);
    CHECK_EQ(t2.frames[1].column, 5u);
    CHECK(!t2.frames[1].is_inlined);

    // 3. Symbolize offset 90 (inside 2-level inlined kernelDispatch)
    brass::StackTrace t3 = symbolicator.symbolize_offset("main", 90);
    REQUIRE_EQ(t3.frame_count(), 3u);
    // Frame 0: innermost kernelDispatch
    CHECK_EQ(t3.frames[0].function_name, "kernelDispatch");
    CHECK_EQ(t3.frames[0].file, "kernel.js");
    CHECK_EQ(t3.frames[0].line, 12u);
    CHECK_EQ(t3.frames[0].column, 16u);
    CHECK(t3.frames[0].is_inlined);
    // Frame 1: intermediate engineRun
    CHECK_EQ(t3.frames[1].function_name, "engineRun");
    CHECK_EQ(t3.frames[1].file, "engine.js");
    CHECK_EQ(t3.frames[1].line, 45u);
    CHECK_EQ(t3.frames[1].column, 8u);
    CHECK(t3.frames[1].is_inlined);
    // Frame 2: outermost main
    CHECK_EQ(t3.frames[2].function_name, "main");
    CHECK_EQ(t3.frames[2].file, "main.js");
    CHECK_EQ(t3.frames[2].line, 100u);
    CHECK_EQ(t3.frames[2].column, 5u);
    CHECK(!t3.frames[2].is_inlined);
}
