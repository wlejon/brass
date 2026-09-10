#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <memory>

TEST_CASE("DebugDifferential - Inlining, Optimization, Source Map and Binary Debug Section Roundtrip") {
    brass::Module mod("differential_debug");

    uint32_t f_main = mod.debug_context().get_or_add_file("app/main.js");
    uint32_t f_math = mod.debug_context().get_or_add_file("lib/math.js");
    uint32_t f_core = mod.debug_context().get_or_add_file("lib/core.js");

    // 1. Callee function: double_val(x) in core.js
    brass::Function* fn_core = mod.create_function("double_val", brass::Type::i32(), {brass::Type::i32()});
    {
        brass::Builder b(mod);
        b.set_function(fn_core);
        brass::BasicBlock* bb = b.append_block("entry");
        b.position_at_end(bb);
        brass::Value* arg0 = b.add_block_param(bb, brass::Type::i32());

        b.set_current_loc(f_core, 15, 4);
        brass::Value* two = b.build_iconst_i32(2);
        b.set_current_loc(f_core, 16, 8);
        brass::Value* res = b.build_mul(arg0, two);
        b.set_current_loc(f_core, 17, 2);
        b.build_ret(res);
        fn_core->rebuild_cfg_predecessors();
    }

    // 2. Intermediate function: compute_term(a, b) in math.js
    brass::Function* fn_math = mod.create_function("compute_term", brass::Type::i32(), {brass::Type::i32(), brass::Type::i32()});
    brass::Instruction* call_core_inst = nullptr;
    {
        brass::Builder b(mod);
        b.set_function(fn_math);
        brass::BasicBlock* bb = b.append_block("entry");
        b.position_at_end(bb);
        brass::Value* a = b.add_block_param(bb, brass::Type::i32());
        brass::Value* b_val = b.add_block_param(bb, brass::Type::i32());

        b.set_current_loc(f_math, 30, 6);
        brass::Value* sum = b.build_add(a, b_val);

        b.set_current_loc(f_math, 32, 10);
        brass::Value* call_res = b.build_call("double_val", brass::Type::i32(), {sum});
        call_core_inst = call_res->defining_instruction();

        b.set_current_loc(f_math, 33, 2);
        b.build_ret(call_res);
        fn_math->rebuild_cfg_predecessors();
    }

    // Inline double_val into compute_term
    REQUIRE(call_core_inst != nullptr);
    brass::InlineResult res1 = brass::inline_call_site(*fn_math, call_core_inst, *fn_core);
    CHECK(res1.success || res1.split_head != nullptr);

    // 3. Outermost function: run_pipeline(x) in main.js
    brass::Function* fn_main = mod.create_function("run_pipeline", brass::Type::i32(), {brass::Type::i32()});
    brass::Instruction* call_math_inst = nullptr;
    {
        brass::Builder b(mod);
        b.set_function(fn_main);
        brass::BasicBlock* bb = b.append_block("entry");
        b.position_at_end(bb);
        brass::Value* x = b.add_block_param(bb, brass::Type::i32());

        b.set_current_loc(f_main, 50, 2);
        brass::Value* ten = b.build_iconst_i32(10);

        b.set_current_loc(f_main, 55, 12);
        brass::Value* math_res = b.build_call("compute_term", brass::Type::i32(), {x, ten});
        call_math_inst = math_res->defining_instruction();

        b.set_current_loc(f_main, 60, 4);
        b.build_ret(math_res);
        fn_main->rebuild_cfg_predecessors();
    }

    // Inline compute_term into run_pipeline
    REQUIRE(call_math_inst != nullptr);
    brass::InlineResult res2 = brass::inline_call_site(*fn_main, call_math_inst, *fn_math);
    CHECK(res2.success || res2.split_head != nullptr);

    // Verify MIR module passes verification
    brass::DiagnosticReporter diag;
    bool verify_ok = brass::verify_module(mod, &diag);
    CHECK(verify_ok);

    // Run standard compiler optimizations (GVN, CFG simplify)
    brass::gvn_module(mod);
    brass::cfg_simplify_module(mod);
    CHECK(brass::verify_module(mod, &diag));

    // Compile to object file with LIR instruction selection and scheduling
    brass::Target target = brass::Target::host();
    brass::codegen::SchedOptions sched_opts;
    sched_opts.enable_pre_ra = true;
    sched_opts.enable_post_ra = true;

    brass::object::ObjectFile obj = brass::object::compile_module_to_object(mod, target, sched_opts);

    // Verify .brass_dbg custom section was generated
    const brass::object::Section* dbg_sec = obj.get_section(".brass_dbg");
    REQUIRE(dbg_sec != nullptr);
    CHECK(!dbg_sec->data.empty());
    CHECK(dbg_sec->data.size() >= 40);

    // Deserialize .brass_dbg section and verify full symbolication
    brass::DebugContext loaded_ctx;
    std::vector<brass::FunctionDebugTable> loaded_tables;
    bool des_ok = brass::deserialize_debug_section(
        dbg_sec->data.data(),
        dbg_sec->data.size(),
        loaded_ctx,
        loaded_tables
    );
    REQUIRE(des_ok);
    CHECK_EQ(loaded_ctx.file_count(), 3u);

    // Find run_pipeline table
    const brass::FunctionDebugTable* pipe_table = nullptr;
    for (const auto& tbl : loaded_tables) {
        if (tbl.function_name() == "run_pipeline") {
            pipe_table = &tbl;
            break;
        }
    }
    REQUIRE(pipe_table != nullptr);
    CHECK(!pipe_table->line_entries().empty());

    // Symbolicate offsets in run_pipeline
    brass::Symbolicator symbolicator(loaded_ctx, loaded_tables);

    // Verify that at least one offset in run_pipeline symbolizes to inlined core.js:16
    bool found_nested_inline = false;
    for (const auto& entry : pipe_table->line_entries()) {
        brass::StackTrace trace = symbolicator.symbolize_offset("run_pipeline", entry.code_offset);
        if (trace.frame_count() >= 3) {
            // Frame 0: double_val in core.js
            // Frame 1: compute_term in math.js
            // Frame 2: run_pipeline in main.js
            if (trace.frames[0].file == "lib/core.js" &&
                trace.frames[1].file == "lib/math.js" &&
                trace.frames[2].file == "app/main.js") {
                found_nested_inline = true;
                CHECK(trace.frames[0].is_inlined);
                CHECK(trace.frames[1].is_inlined);
                CHECK(!trace.frames[2].is_inlined);
            }
        }
    }
    CHECK(found_nested_inline);

    // Generate V3 JSON Source Map from the debug table
    brass::SourceMap sm = pipe_table->to_source_map(loaded_ctx, "run_pipeline.js");
    std::string json = sm.to_json();
    CHECK(!json.empty());

    std::string err;
    auto parsed_sm = brass::SourceMap::parse_json(json, &err);
    REQUIRE(parsed_sm != nullptr);
    CHECK(err.empty());

    // Check that source map and debug table agree on offset resolution
    for (const auto& entry : pipe_table->line_entries()) {
        brass::DebugLoc sm_loc = parsed_sm->resolve_offset(entry.code_offset);
        CHECK_EQ(sm_loc.file_id, entry.loc.file_id);
        CHECK_EQ(sm_loc.line, entry.loc.line);
        CHECK_EQ(sm_loc.column, entry.loc.column);
    }
}
