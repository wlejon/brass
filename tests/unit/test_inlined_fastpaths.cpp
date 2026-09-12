#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/il_translator/il_property.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace brass;
using namespace brass::il;
using namespace brass::runtime;

static constexpr uint64_t kUndefinedTag = 0xFFF6000000000000ULL;

TEST_CASE("InlinedFastpath - Direct MIR Fastpath Diamond Construction") {
    Module mod("test_mir_fastpath");
    Function* fn = mod.create_function("test_fn", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* obj = b.add_block_param(b0, Type::i64());
    Value* idx = b.add_block_param(b0, Type::i64());
    Value* val = b.add_block_param(b0, Type::i64());

    PropertyLoweringHelper helper(true, /*enable_inlined_fastpaths=*/true);
    CHECK(helper.enable_inlined_fastpaths());

    helper.lower_elem_set(b, obj, idx, val, 0);
    Value* res = helper.lower_elem_get(b, obj, idx);
    b.build_ret(res);

    bool found_load_indexed = false;
    bool found_store_indexed = false;
    bool found_elem_set_call = false;
    bool found_elem_get_call = false;

    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::load_indexed) {
                found_load_indexed = true;
                CHECK_EQ(static_cast<int>(inst->scale()), 8);
                CHECK_EQ(inst->offset(), 8);
            }
            if (inst->opcode() == Opcode::store_indexed) {
                found_store_indexed = true;
                CHECK_EQ(static_cast<int>(inst->scale()), 8);
                CHECK_EQ(inst->offset(), 8);
            }
            if (inst->opcode() == Opcode::call) {
                std::string callee(inst->symbol());
                if (callee == "bronze_elem_set") found_elem_set_call = true;
                if (callee == "bronze_elem_get") found_elem_get_call = true;
            }
        }
    }

    CHECK(found_load_indexed);
    CHECK(found_store_indexed);
    CHECK(found_elem_set_call);
    CHECK(found_elem_get_call);
}

TEST_CASE("InlinedFastpath - Array Element Read/Write Hit on Fast Path") {
    DynamicObject* arr = DynamicObject::create_array(nullptr, 8);
    REQUIRE(arr != nullptr);
    arr->set_length(8);
    uint64_t arr_raw = reinterpret_cast<uint64_t>(arr);

    Module mod("test_rw_fastpath");
    Function* fn = mod.create_function("rw_fn", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* obj = b.add_block_param(b0, Type::i64());
    Value* idx = b.add_block_param(b0, Type::i64());
    Value* val = b.add_block_param(b0, Type::i64());

    PropertyLoweringHelper helper(true, /*enable_inlined_fastpaths=*/true);
    helper.lower_elem_set(b, obj, idx, val, 0);
    Value* loaded = helper.lower_elem_get(b, obj, idx);
    b.build_ret(loaded);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t, int64_t)>("rw_fn");
    REQUIRE(fn_ptr != nullptr);

    // Test writing and reading multiple indices
    int64_t v0 = 4242;
    int64_t res0 = fn_ptr(static_cast<int64_t>(arr_raw), 0, v0);
    CHECK_EQ(res0, v0);
    CHECK_EQ(arr->get_element(0).raw(), static_cast<uint64_t>(v0));

    int64_t v3 = 987654;
    int64_t res3 = fn_ptr(static_cast<int64_t>(arr_raw), 3, v3);
    CHECK_EQ(res3, v3);
    CHECK_EQ(arr->get_element(3).raw(), static_cast<uint64_t>(v3));

    int64_t v7 = 11223344;
    int64_t res7 = fn_ptr(static_cast<int64_t>(arr_raw), 7, v7);
    CHECK_EQ(res7, v7);
    CHECK_EQ(arr->get_element(7).raw(), static_cast<uint64_t>(v7));

    // Cleanup
    DynamicObject::destroy_non_gc(arr);
}

TEST_CASE("InlinedFastpath - Array Element Out-of-bounds Fallback") {
    DynamicObject* arr = DynamicObject::create_array(nullptr, 2);
    REQUIRE(arr != nullptr);
    arr->set_length(2);
    uint64_t arr_raw = reinterpret_cast<uint64_t>(arr);

    Module mod("test_oob_fallback");
    Function* fn_set = mod.create_function("oob_set", Type::void_type(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b_set(mod);
    b_set.set_function(fn_set);
    BasicBlock* b_set_0 = b_set.append_block("b0");
    Value* s_obj = b_set.add_block_param(b_set_0, Type::i64());
    Value* s_idx = b_set.add_block_param(b_set_0, Type::i64());
    Value* s_val = b_set.add_block_param(b_set_0, Type::i64());
    PropertyLoweringHelper helper(true, true);
    helper.lower_elem_set(b_set, s_obj, s_idx, s_val, 0);
    b_set.build_ret_void();

    Function* fn_get = mod.create_function("oob_get", Type::i64(), {Type::i64(), Type::i64()});
    Builder b_get(mod);
    b_get.set_function(fn_get);
    BasicBlock* b_get_0 = b_get.append_block("b0");
    Value* g_obj = b_get.add_block_param(b_get_0, Type::i64());
    Value* g_idx = b_get.add_block_param(b_get_0, Type::i64());
    Value* g_res = helper.lower_elem_get(b_get, g_obj, g_idx);
    b_get.build_ret(g_res);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto set_ptr = jit.get_function_ptr<void(*)(int64_t, int64_t, int64_t)>("oob_set");
    auto get_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("oob_get");
    REQUIRE(set_ptr != nullptr);
    REQUIRE(get_ptr != nullptr);

    // Initial length is 2. Write at index 5 triggers fallback expansion
    set_ptr(static_cast<int64_t>(arr_raw), 5, 5555);
    CHECK(arr->length() >= 6);

    // Read back index 5
    int64_t val5 = get_ptr(static_cast<int64_t>(arr_raw), 5);
    CHECK_EQ(val5, 5555);

    // Read index 20 (out of bounds, never written)
    int64_t undef_val = get_ptr(static_cast<int64_t>(arr_raw), 20);
    CHECK_EQ(undef_val, static_cast<int64_t>(kUndefinedTag));

    DynamicObject::destroy_non_gc(arr);
}

TEST_CASE("InlinedFastpath - Negative Index and Null Object") {
    DynamicObject* arr = DynamicObject::create_array(nullptr, 4);
    REQUIRE(arr != nullptr);
    arr->set_length(4);
    uint64_t arr_raw = reinterpret_cast<uint64_t>(arr);

    Module mod("test_guards");
    Function* fn_get = mod.create_function("guard_get", Type::i64(), {Type::i64(), Type::i64()});
    Builder b_get(mod);
    b_get.set_function(fn_get);
    BasicBlock* b0 = b_get.append_block("b0");
    Value* g_obj = b_get.add_block_param(b0, Type::i64());
    Value* g_idx = b_get.add_block_param(b0, Type::i64());
    PropertyLoweringHelper helper(true, true);
    Value* res = helper.lower_elem_get(b_get, g_obj, g_idx);
    b_get.build_ret(res);

    Function* fn_set = mod.create_function("guard_set", Type::void_type(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b_set(mod);
    b_set.set_function(fn_set);
    BasicBlock* b_set_0 = b_set.append_block("b0");
    Value* s_obj = b_set.add_block_param(b_set_0, Type::i64());
    Value* s_idx = b_set.add_block_param(b_set_0, Type::i64());
    Value* s_val = b_set.add_block_param(b_set_0, Type::i64());
    helper.lower_elem_set(b_set, s_obj, s_idx, s_val, 0);
    b_set.build_ret_void();

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto get_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("guard_get");
    auto set_ptr = jit.get_function_ptr<void(*)(int64_t, int64_t, int64_t)>("guard_set");
    REQUIRE(get_ptr != nullptr);
    REQUIRE(set_ptr != nullptr);

    // Null object read -> undefined
    int64_t null_res = get_ptr(0, 0);
    CHECK_EQ(null_res, static_cast<int64_t>(kUndefinedTag));

    // Null object write -> no crash
    set_ptr(0, 0, 123);

    // Negative index read -> undefined
    int64_t neg_res = get_ptr(static_cast<int64_t>(arr_raw), -1);
    CHECK_EQ(neg_res, static_cast<int64_t>(kUndefinedTag));

    int64_t neg_res2 = get_ptr(static_cast<int64_t>(arr_raw), -100);
    CHECK_EQ(neg_res2, static_cast<int64_t>(kUndefinedTag));

    // Negative index write -> no crash, no modification
    set_ptr(static_cast<int64_t>(arr_raw), -1, 999);
    CHECK_EQ(get_ptr(static_cast<int64_t>(arr_raw), -1), static_cast<int64_t>(kUndefinedTag));

    DynamicObject::destroy_non_gc(arr);
}

TEST_CASE("InlinedFastpath - Property Slot Access") {
    DynamicObject* obj = DynamicObject::create(nullptr, nullptr);
    REQUIRE(obj != nullptr);
    uint64_t obj_raw = reinterpret_cast<uint64_t>(obj);

    Module mod("test_prop_slot");
    Function* fn = mod.create_function("slot_rw", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* obj_val = b.add_block_param(b0, Type::i64());
    Value* val0 = b.add_block_param(b0, Type::i64());
    Value* val1 = b.add_block_param(b0, Type::i64());

    PropertyLoweringHelper helper(true, true);
    helper.lower_prop_set_slot(b, obj_val, 0, val0);
    helper.lower_prop_set_slot(b, obj_val, 1, val1);

    Value* r0 = helper.lower_prop_get_slot(b, obj_val, 0);
    Value* r1 = helper.lower_prop_get_slot(b, obj_val, 1);
    Value* sum = b.build_add(r0, r1);
    b.build_ret(sum);

    // Verify offsets in MIR
    int store_count = 0;
    int load_count = 0;
    for (Instruction* inst : *b0) {
        if (inst->opcode() == Opcode::store) {
            if (store_count == 0) CHECK_EQ(inst->offset(), 24);
            if (store_count == 1) CHECK_EQ(inst->offset(), 32);
            store_count++;
        }
        if (inst->opcode() == Opcode::load) {
            if (load_count == 0) CHECK_EQ(inst->offset(), 24);
            if (load_count == 1) CHECK_EQ(inst->offset(), 32);
            load_count++;
        }
    }
    CHECK_EQ(store_count, 2);
    CHECK_EQ(load_count, 2);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(mod));

    auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t, int64_t)>("slot_rw");
    REQUIRE(fn_ptr != nullptr);

    int64_t total = fn_ptr(static_cast<int64_t>(obj_raw), 100, 250);
    CHECK_EQ(total, 350);
    CHECK_EQ(obj->inline_slots[0].raw(), 100ULL);
    CHECK_EQ(obj->inline_slots[1].raw(), 250ULL);

    DynamicObject::destroy_non_gc(obj);
}

TEST_CASE("InlinedFastpath - End-to-End Bronze IL Fastpaths Enabled vs Disabled") {
    const char* il_code = R"(
module test_fastpath_e2e.js

func computeArray() -> f64 {
  b0:
    %0: dynamic = create.array 4
    %1: f64 = const.f64 300
    %2: dynamic = box.f64 %1
    %3: i32 = const.i32 0
    elem.set %0, %3, %2
    %4: f64 = const.f64 400
    %5: dynamic = box.f64 %4
    %6: i32 = const.i32 1
    elem.set %0, %6, %5

    %7: dynamic = elem.get %0, %3
    %8: f64 = unbox.f64 %7
    %9: dynamic = elem.get %0, %6
    %10: f64 = unbox.f64 %9
    %11: f64 = add %8, %10
    ret %11
}
)";

    // 1. With inlined fastpaths ENABLED
    {
        TranslatorOptions opts;
        opts.enable_inlined_fastpaths = true;
        DiagnosticReporter diag;
        TranslationResult res = translate_bronze_il(il_code, opts, &diag);
        REQUIRE(res.success);
        REQUIRE(res.module != nullptr);

        bool has_load_indexed = false;
        bool has_store_indexed = false;
        for (Function* fn : res.module->functions()) {
            for (BasicBlock* bb : fn->blocks()) {
                for (Instruction* inst : *bb) {
                    if (inst->opcode() == Opcode::load_indexed) has_load_indexed = true;
                    if (inst->opcode() == Opcode::store_indexed) has_store_indexed = true;
                }
            }
        }
        CHECK(has_load_indexed);
        CHECK(has_store_indexed);

        codegen::JitExecutionEngine jit(Target::host());
        register_bronze_runtime_symbols(&jit);
        REQUIRE(jit.compile_and_load(*res.module));
        auto fn_ptr = jit.get_function_ptr<double(*)()>("computeArray");
        REQUIRE(fn_ptr != nullptr);
        double val = fn_ptr();
        CHECK_EQ(val, 700.0);
    }

    // 2. With inlined fastpaths DISABLED
    {
        TranslatorOptions opts;
        opts.enable_inlined_fastpaths = false;
        DiagnosticReporter diag;
        TranslationResult res = translate_bronze_il(il_code, opts, &diag);
        REQUIRE(res.success);
        REQUIRE(res.module != nullptr);

        bool has_load_indexed = false;
        bool has_store_indexed = false;
        bool has_elem_get_call = false;
        bool has_elem_set_call = false;
        for (Function* fn : res.module->functions()) {
            for (BasicBlock* bb : fn->blocks()) {
                for (Instruction* inst : *bb) {
                    if (inst->opcode() == Opcode::load_indexed) has_load_indexed = true;
                    if (inst->opcode() == Opcode::store_indexed) has_store_indexed = true;
                    if (inst->opcode() == Opcode::call) {
                        if (inst->symbol() == "bronze_elem_get") has_elem_get_call = true;
                        if (inst->symbol() == "bronze_elem_set") has_elem_set_call = true;
                    }
                }
            }
        }
        CHECK(!has_load_indexed);
        CHECK(!has_store_indexed);
        CHECK(has_elem_get_call);
        CHECK(has_elem_set_call);

        codegen::JitExecutionEngine jit(Target::host());
        register_bronze_runtime_symbols(&jit);
        REQUIRE(jit.compile_and_load(*res.module));
        auto fn_ptr = jit.get_function_ptr<double(*)()>("computeArray");
        REQUIRE(fn_ptr != nullptr);
        double val = fn_ptr();
        CHECK_EQ(val, 700.0);
    }
}
