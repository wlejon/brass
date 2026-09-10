#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <random>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

using namespace brass;
using namespace brass::runtime;
using namespace brass::il;

// Reference property lookup for differential verification
static HostValue reference_get_prop(const DynamicObject* obj, const std::string& name) {
    if (!obj || !obj->shape) return HostValue::undefined_val();
    const PropertyDescriptor* desc = obj->shape->find_property(name);
    if (!desc) return HostValue::undefined_val();
    return obj->get_slot(desc->slot_index);
}

// Reference property store for differential verification
static void reference_set_prop(DynamicObject* obj, const std::string& name, HostValue val, ShapeRegistry& registry) {
    if (!obj) return;
    Shape* current = obj->shape ? obj->shape : registry.get_root_shape();
    const PropertyDescriptor* desc = current->find_property(name);
    if (desc) {
        obj->set_slot(desc->slot_index, val);
    } else {
        Shape* next = registry.transition_to(current, name);
        obj->shape = next;
        obj->set_slot(next->slot_count() - 1, val);
    }
}

TEST_CASE("Differential PIC - Monomorphic, Polymorphic, and Megamorphic IC vs Reference") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // Create 6 different shapes with property "val" at various slots
    // Shape 0: {val} (slot 0)
    Shape* s0 = registry.transition_to(root, "val");
    // Shape 1: {a, val} (slot 1)
    Shape* s1 = registry.transition_to(registry.transition_to(root, "a"), "val");
    // Shape 2: {b, c, val} (slot 2)
    Shape* s2 = registry.transition_to(registry.transition_to(registry.transition_to(root, "b"), "c"), "val");
    // Shape 3: {d, e, f, val} (slot 3)
    Shape* s3 = registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(root, "d"), "e"), "f"), "val");
    // Shape 4: {g, h, i, j, val} (slot 4)
    Shape* s4 = registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(root, "g"), "h"), "i"), "j"), "val");
    // Shape 5 (deep, triggers out of line): {k0..k8, val} (slot 9)
    Shape* s5 = root;
    for (int i = 0; i < 9; ++i) {
        s5 = registry.transition_to(s5, "k" + std::to_string(i));
    }
    s5 = registry.transition_to(s5, "val");

    std::vector<Shape*> shapes = {s0, s1, s2, s3, s4, s5};

    // Create dynamic objects for each shape
    std::vector<DynamicObject*> objects;
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto* obj = DynamicObject::create(nullptr, shapes[i]);
        uint32_t slot = shapes[i]->find_property("val")->slot_index;
        obj->set_slot(slot, HostValue::from_i32(static_cast<int32_t>(1000 + i * 10)));
        objects.push_back(obj);
    }

    // 1. Differential test for Load IC across 10,000 iterations
    InlineCache ic_load(501, "val", 0, /*is_load=*/true);
    std::mt19937 rng(42);

    // First: monomorphic phase (only obj 0)
    for (int iter = 0; iter < 100; ++iter) {
        HostValue ref = reference_get_prop(objects[0], "val");
        HostValue ic_val = ic_load.execute_get(objects[0]);
        CHECK_EQ(ic_val.raw(), ref.raw());
    }
    CHECK_EQ(ic_load.state(), ICState::Monomorphic);
    CHECK_EQ(ic_load.entry_count(), 1u);

    // Second: polymorphic phase (objs 0, 1, 2, 3)
    for (int iter = 0; iter < 400; ++iter) {
        size_t idx = static_cast<size_t>(rng() % 4);
        HostValue ref = reference_get_prop(objects[idx], "val");
        HostValue ic_val = ic_load.execute_get(objects[idx]);
        CHECK_EQ(ic_val.raw(), ref.raw());
    }
    CHECK_EQ(ic_load.state(), ICState::Polymorphic);
    CHECK_EQ(ic_load.entry_count(), 4u);

    // Third: megamorphic phase (introduce objs 4 and 5)
    for (int iter = 0; iter < 5000; ++iter) {
        size_t idx = static_cast<size_t>(rng() % shapes.size());
        HostValue ref = reference_get_prop(objects[idx], "val");
        HostValue ic_val = ic_load.execute_get(objects[idx]);
        CHECK_EQ(ic_val.raw(), ref.raw());
    }
    CHECK_EQ(ic_load.state(), ICState::Megamorphic);

    // 2. Differential test for Store IC across 5,000 iterations
    InlineCache ic_store(502, "val", 0, /*is_load=*/false);
    for (int iter = 0; iter < 5000; ++iter) {
        size_t idx = static_cast<size_t>(rng() % shapes.size());
        int32_t new_val = static_cast<int32_t>(rng() % 100000);

        // Store via reference
        reference_set_prop(objects[idx], "val", HostValue::from_i32(new_val), registry);
        HostValue expected = reference_get_prop(objects[idx], "val");
        CHECK_EQ(expected.as_i32(), new_val);

        // Overwrite and store via IC
        ic_store.execute_set(objects[idx], HostValue::from_i32(new_val + 1), registry, nullptr);
        HostValue actual = reference_get_prop(objects[idx], "val");

        CHECK_EQ(actual.as_i32(), new_val + 1);
    }

    // Free objects
    for (auto* obj : objects) {
        DynamicObject::destroy_non_gc(obj);
    }
}

TEST_CASE("Differential PIC - Bronze IL JIT Execution vs Reference") {
    const char* il_code = R"(
module pic_diff.js

func accessProperty(%0: dynamic) -> f64 {
  b0:
    %1: dynamic = prop.get %0, "count"
    %2: f64 = unbox.f64 %1
    ret %2
}
)";

    TranslatorOptions opts;
    opts.enable_pic = true;
    DiagnosticReporter diag;
    TranslationResult res = translate_bronze_il(il_code, opts, &diag);
    REQUIRE(res.success);
    REQUIRE(res.module != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    register_bronze_runtime_symbols(&jit);
    REQUIRE(jit.compile_and_load(*res.module));

    auto access_fn = jit.get_function_ptr<double(*)(uint64_t)>("accessProperty");
    REQUIRE(access_fn != nullptr);

    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // 3 distinct shapes all containing "count"
    Shape* shape1 = registry.transition_to(root, "count");
    Shape* shape2 = registry.transition_to(registry.transition_to(root, "dummy1"), "count");
    Shape* shape3 = registry.transition_to(registry.transition_to(registry.transition_to(root, "dummy1"), "dummy2"), "count");

    DynamicObject* o1 = DynamicObject::create(nullptr, shape1);
    o1->set_slot(0, HostValue::from_double(123.0));

    DynamicObject* o2 = DynamicObject::create(nullptr, shape2);
    o2->set_slot(1, HostValue::from_double(456.0));

    DynamicObject* o3 = DynamicObject::create(nullptr, shape3);
    o3->set_slot(2, HostValue::from_double(789.0));

    // Warm up IC site through JIT
    std::mt19937 rng(999);
    DynamicObject* arr[3] = {o1, o2, o3};
    double expected_vals[3] = {123.0, 456.0, 789.0};

    for (int i = 0; i < 3000; ++i) {
        int idx = rng() % 3;
        double result = access_fn(reinterpret_cast<uint64_t>(arr[idx]));
        CHECK_EQ(result, expected_vals[idx]);
    }

    DynamicObject::destroy_non_gc(o1);
    DynamicObject::destroy_non_gc(o2);
    DynamicObject::destroy_non_gc(o3);
}
