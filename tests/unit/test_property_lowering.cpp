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

TEST_CASE("PropertyLowering - Direct MIR Helper Construction") {
    Module mod("test_mir_prop");
    Function* fn = mod.create_function("test_func", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* obj = b.add_block_param(b0, Type::i64());
    Value* val = b.add_block_param(b0, Type::i64());

    // Test with PIC enabled
    PropertyLoweringHelper pic_helper(true);
    pic_helper.lower_prop_set(b, obj, "fieldA", 10, val, 0, 0, 101);
    Value* get_res = pic_helper.lower_prop_get(b, obj, "fieldA", 10, 102);
    CHECK(get_res != nullptr);

    // Test element ops
    Value* idx = b.build_iconst_i64(3);
    pic_helper.lower_elem_set(b, obj, idx, val, 0);
    Value* elem_res = pic_helper.lower_elem_get(b, obj, idx);
    CHECK(elem_res != nullptr);

    // Test method def
    pic_helper.lower_method_def(b, obj, "compute", 11, val);

    b.build_ret(get_res);

    // Inspect instructions in b0
    bool found_ic_set = false;
    bool found_ic_get = false;
    bool found_elem_set = false;
    bool found_elem_get = false;
    bool found_method_def = false;

    for (Instruction* inst : *b0) {
        if (inst->opcode() == Opcode::call) {
            std::string callee(inst->symbol());
            if (callee == "brass_ic_set_prop") found_ic_set = true;
            if (callee == "brass_ic_get_prop") found_ic_get = true;
            if (callee == "bronze_elem_set") found_elem_set = true;
            if (callee == "bronze_elem_get") found_elem_get = true;
            if (callee == "bronze_method_def") found_method_def = true;
        }
    }

    CHECK(found_ic_set);
    CHECK(found_ic_get);
    CHECK(found_elem_set);
    CHECK(found_elem_get);
    CHECK(found_method_def);
}

TEST_CASE("PropertyLowering - Bronze IL Property and Element JIT Execution") {
    const char* il_src = R"(
module test_props.js

func runTest() -> f64 {
  b0:
    %0: dynamic = create.object
    %1: f64 = const.f64 42
    %2: dynamic = box.f64 %1
    prop.set %0, "x", %2
    %3: f64 = const.f64 58
    %4: dynamic = box.f64 %3
    prop.set %0, "y", %4

    %5: dynamic = prop.get %0, "x"
    %6: f64 = unbox.f64 %5
    %7: dynamic = prop.get %0, "y"
    %8: f64 = unbox.f64 %7
    %9: f64 = add %6, %8

    %10: dynamic = create.array 4
    %11: f64 = const.f64 100
    %12: dynamic = box.f64 %11
    %13: i32 = const.i32 0
    elem.set %10, %13, %12
    %14: dynamic = elem.get %10, %13
    %15: f64 = unbox.f64 %14

    %16: f64 = add %9, %15
    ret %16
}
)";

    // Run with PIC enabled
    {
        TranslatorOptions opts;
        opts.enable_pic = true;
        DiagnosticReporter diag;
        TranslationResult res = translate_bronze_il(il_src, opts, &diag);
        if (!res.success) {
            std::cerr << "DIAG ERROR:\n" << diag.format_all() << std::endl;
        }
        REQUIRE(res.success);
        REQUIRE(res.module != nullptr);

        codegen::JitExecutionEngine jit(Target::host());
        register_bronze_runtime_symbols(&jit);
        REQUIRE(jit.compile_and_load(*res.module));

        auto run_fn = jit.get_function_ptr<double(*)()>("runTest");
        REQUIRE(run_fn != nullptr);
        double val = run_fn();
        // 42 + 58 + 100 = 200
        CHECK_EQ(val, 200.0);
    }

    // Run without PIC (direct fast path)
    {
        TranslatorOptions opts;
        opts.enable_pic = false;
        DiagnosticReporter diag;
        TranslationResult res = translate_bronze_il(il_src, opts, &diag);
        REQUIRE(res.success);
        REQUIRE(res.module != nullptr);

        codegen::JitExecutionEngine jit(Target::host());
        register_bronze_runtime_symbols(&jit);
        REQUIRE(jit.compile_and_load(*res.module));

        auto run_fn = jit.get_function_ptr<double(*)()>("runTest");
        REQUIRE(run_fn != nullptr);
        double val = run_fn();
        CHECK_EQ(val, 200.0);
    }
}

TEST_CASE("PropertyLowering - Method Definition and Method Invocation") {
    const char* il_method = R"(
module test_method.js

func multiplier(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = prop.get %0, "factor"
    %3: f64 = unbox.f64 %2
    %4: f64 = unbox.f64 %1
    %5: f64 = mul %3, %4
    %6: dynamic = box.f64 %5
    ret %6
}

func testMethod() -> f64 {
  b0:
    %0: dynamic = create.object
    %1: f64 = const.f64 7
    %2: dynamic = box.f64 %1
    prop.set %0, "factor", %2

    %4: dynamic = create.func @multiplier, 1, %0
    method.def %0, "multiply", %4

    %5: dynamic = prop.get %0, "multiply"
    %6: f64 = const.f64 6
    %7: dynamic = box.f64 %6
    %8: dynamic = call.dynamic %5, %0, 1, %7
    %9: f64 = unbox.f64 %8
    ret %9
}
)";

    TranslatorOptions opts;
    opts.enable_pic = true;
    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_method, opts, &diag);
    if (!res.success) {
        std::cerr << "METHOD DIAG ERROR:\n" << diag.format_all() << std::endl;
    }
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto fn = jit.get_function_ptr<double(*)()>("testMethod");
    REQUIRE(fn != nullptr);
    double result = fn();
    // 7 * 6 = 42
    CHECK_EQ(result, 42.0);
}

TEST_CASE("PropertyLowering - DynamicObject Cheney Moving GC Survival") {
    HostGC host_gc(128 * 1024);
    set_active_host_gc(&host_gc);

    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // 1. Allocate root object
    DynamicObject* parent_obj = DynamicObject::create(&host_gc, root);
    REQUIRE(parent_obj != nullptr);
    uintptr_t parent_addr = reinterpret_cast<uintptr_t>(parent_obj);
    host_gc.register_root(&parent_addr);

    // 2. Set inline properties on parent
    parent_obj->set_property("alpha", HostValue::from_i32(111), registry, &host_gc);
    parent_obj->set_property("beta", HostValue::from_double(222.5), registry, &host_gc);

    // 3. Allocate child object and set on parent
    DynamicObject* child_obj = DynamicObject::create(&host_gc, root);
    REQUIRE(child_obj != nullptr);
    child_obj->set_property("gamma", HostValue::from_i32(333), registry, &host_gc);
    uintptr_t old_child_addr = reinterpret_cast<uintptr_t>(child_obj);

    parent_obj->set_property("child", HostValue::from_gcref(old_child_addr), registry, &host_gc);

    // 4. Allocate element array and set element
    DynamicObject* array_obj = DynamicObject::create_array(&host_gc, 4);
    REQUIRE(array_obj != nullptr);
    array_obj->set_element(0, HostValue::from_i32(999), &host_gc);
    array_obj->set_element(1, HostValue::from_gcref(old_child_addr), &host_gc);
    uintptr_t old_array_addr = reinterpret_cast<uintptr_t>(array_obj);

    parent_obj->set_property("items", HostValue::from_gcref(old_array_addr), registry, &host_gc);

    // 5. Trigger garbage collection
    host_gc.collect();

    // 6. Verify parent object was relocated
    auto* relocated_parent = reinterpret_cast<DynamicObject*>(parent_addr);
    CHECK(host_gc.is_address_in_active_space(parent_addr));
    CHECK_EQ(relocated_parent->get_property("alpha").as_i32(), 111);
    CHECK_EQ(relocated_parent->get_property("beta").as_double(), 222.5);

    // 7. Verify child object was relocated and is live
    HostValue child_val = relocated_parent->get_property("child");
    CHECK(child_val.is_gcref());
    uintptr_t new_child_addr = child_val.as_gcref();
    CHECK_NE(new_child_addr, old_child_addr);
    CHECK(host_gc.is_address_in_active_space(new_child_addr));

    auto* relocated_child = reinterpret_cast<DynamicObject*>(new_child_addr);
    CHECK_EQ(relocated_child->get_property("gamma").as_i32(), 333);

    // 8. Verify array object and elements were relocated
    HostValue items_val = relocated_parent->get_property("items");
    CHECK(items_val.is_gcref());
    uintptr_t new_array_addr = items_val.as_gcref();
    CHECK_NE(new_array_addr, old_array_addr);
    CHECK(host_gc.is_address_in_active_space(new_array_addr));

    auto* relocated_array = reinterpret_cast<DynamicObject*>(new_array_addr);
    CHECK_EQ(relocated_array->get_element(0).as_i32(), 999);
    CHECK_EQ(relocated_array->get_element(1).as_gcref(), new_child_addr);

    // Cleanup
    host_gc.unregister_root(&parent_addr);
    set_active_host_gc(nullptr);
}
