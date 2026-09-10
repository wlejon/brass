#include "test_framework.hpp"
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/patcher.hpp>

using namespace brass;
using namespace brass::runtime;

TEST_CASE("InlineCache - State lifecycle and polymorphic transitions") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // 5 distinct shapes all having property "x" at different or same slot offsets:
    // Shape A: {x}
    Shape* shapeA = registry.transition_to(root, "x");
    // Shape B: {a, x}
    Shape* shapeB = registry.transition_to(registry.transition_to(root, "a"), "x");
    // Shape C: {b, c, x}
    Shape* shapeC = registry.transition_to(registry.transition_to(registry.transition_to(root, "b"), "c"), "x");
    // Shape D: {d, e, f, x}
    Shape* shapeD = registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(root, "d"), "e"), "f"), "x");
    // Shape E: {g, h, i, j, x}
    Shape* shapeE = registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(registry.transition_to(root, "g"), "h"), "i"), "j"), "x");

    DynamicObject* objA = DynamicObject::create(nullptr, shapeA);
    objA->set_slot(0, HostValue::from_i32(100));

    DynamicObject* objB = DynamicObject::create(nullptr, shapeB);
    objB->set_slot(1, HostValue::from_i32(200));

    DynamicObject* objC = DynamicObject::create(nullptr, shapeC);
    objC->set_slot(2, HostValue::from_i32(300));

    DynamicObject* objD = DynamicObject::create(nullptr, shapeD);
    objD->set_slot(3, HostValue::from_i32(400));

    DynamicObject* objE = DynamicObject::create(nullptr, shapeE);
    objE->set_slot(4, HostValue::from_i32(500));

    InlineCache ic(42, "x", 0, /*is_load=*/true);

    // 1. Initial state: Uninitialized
    CHECK_EQ(ic.state(), ICState::Uninitialized);
    CHECK_EQ(ic.entry_count(), 0u);
    CHECK_EQ(ic.hit_count(), 0u);
    CHECK_EQ(ic.miss_count(), 0u);

    // 2. First access with objA -> Monomorphic
    HostValue resA = ic.execute_get(objA);
    CHECK_EQ(resA.as_i32(), 100);
    CHECK_EQ(ic.state(), ICState::Monomorphic);
    CHECK_EQ(ic.entry_count(), 1u);
    CHECK_EQ(ic.cached_shape(0), shapeA);
    CHECK_EQ(ic.cached_slot(0), 0u);
    CHECK_EQ(ic.hit_count(), 0u);
    CHECK_EQ(ic.miss_count(), 1u);

    // 3. Repeated access with objA -> Monomorphic HIT!
    HostValue resA2 = ic.execute_get(objA);
    CHECK_EQ(resA2.as_i32(), 100);
    CHECK_EQ(ic.state(), ICState::Monomorphic);
    CHECK_EQ(ic.hit_count(), 1u);
    CHECK_EQ(ic.miss_count(), 1u);

    // 4. Access with objB -> Transitions to Polymorphic (2 entries)
    HostValue resB = ic.execute_get(objB);
    CHECK_EQ(resB.as_i32(), 200);
    CHECK_EQ(ic.state(), ICState::Polymorphic);
    CHECK_EQ(ic.entry_count(), 2u);
    CHECK_EQ(ic.cached_shape(1), shapeB);
    CHECK_EQ(ic.cached_slot(1), 1u);
    CHECK_EQ(ic.hit_count(), 1u);
    CHECK_EQ(ic.miss_count(), 2u);

    // 5. Hits on both shapeA and shapeB in Polymorphic state
    CHECK_EQ(ic.execute_get(objA).as_i32(), 100);
    CHECK_EQ(ic.execute_get(objB).as_i32(), 200);
    CHECK_EQ(ic.hit_count(), 3u);
    CHECK_EQ(ic.miss_count(), 2u);

    // 6. Access with objC -> Polymorphic (3 entries)
    HostValue resC = ic.execute_get(objC);
    CHECK_EQ(resC.as_i32(), 300);
    CHECK_EQ(ic.state(), ICState::Polymorphic);
    CHECK_EQ(ic.entry_count(), 3u);

    // 7. Access with objD -> Polymorphic (4 entries, limit reached)
    HostValue resD = ic.execute_get(objD);
    CHECK_EQ(resD.as_i32(), 400);
    CHECK_EQ(ic.state(), ICState::Polymorphic);
    CHECK_EQ(ic.entry_count(), 4u);

    // 8. Access with 5th shape (objE) -> Transitions to Megamorphic!
    HostValue resE = ic.execute_get(objE);
    CHECK_EQ(resE.as_i32(), 500);
    CHECK_EQ(ic.state(), ICState::Megamorphic);

    // 9. Subsequent access in Megamorphic continues to resolve correctly via dictionary
    CHECK_EQ(ic.execute_get(objA).as_i32(), 100);
    CHECK_EQ(ic.execute_get(objE).as_i32(), 500);

    DynamicObject::destroy_non_gc(objA);
    DynamicObject::destroy_non_gc(objB);
    DynamicObject::destroy_non_gc(objC);
    DynamicObject::destroy_non_gc(objD);
    DynamicObject::destroy_non_gc(objE);
}

TEST_CASE("InlineCache - Invalidation and reset") {
    ShapeRegistry registry;
    Shape* shape = registry.transition_to(registry.get_root_shape(), "val");
    DynamicObject* obj = DynamicObject::create(nullptr, shape);
    obj->set_slot(0, HostValue::from_i32(42));

    InlineCache ic(10, "val", 0, true);
    ic.execute_get(obj);
    CHECK_EQ(ic.state(), ICState::Monomorphic);
    CHECK_EQ(ic.entry_count(), 1u);

    // Invalidate IC
    ic.invalidate();
    CHECK_EQ(ic.state(), ICState::Uninitialized);
    CHECK_EQ(ic.entry_count(), 0u);
    CHECK_EQ(ic.cached_shape(0), nullptr);

    // After reset, re-transitions cleanly to Monomorphic on next access
    HostValue res = ic.execute_get(obj);
    CHECK_EQ(res.as_i32(), 42);
    CHECK_EQ(ic.state(), ICState::Monomorphic);

    DynamicObject::destroy_non_gc(obj);
}

TEST_CASE("InlineCache - Store cache and property creation") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    DynamicObject* obj = DynamicObject::create(nullptr, root);
    InlineCache set_ic(20, "count", 0, /*is_load=*/false);

    CHECK_EQ(set_ic.state(), ICState::Uninitialized);

    // First store -> miss handler allocates slot and sets Monomorphic
    set_ic.execute_set(obj, HostValue::from_i32(99), registry);
    CHECK_EQ(set_ic.state(), ICState::Monomorphic);
    CHECK_EQ(set_ic.entry_count(), 1u);
    CHECK_EQ(obj->get_property("count").as_i32(), 99);

    // Second store with same shape -> Monomorphic HIT
    set_ic.execute_set(obj, HostValue::from_i32(100), registry);
    CHECK_EQ(set_ic.hit_count(), 1u);
    CHECK_EQ(obj->get_property("count").as_i32(), 100);

    DynamicObject::destroy_non_gc(obj);
}

TEST_CASE("InlineCache - ICRegistry statistics and registry query") {
    ICRegistry reg;
    InlineCache* ic1 = reg.get_or_create_ic(1, "propA", 0, true);
    InlineCache* ic2 = reg.get_or_create_ic(2, "propB", 0, false);

    REQUIRE(ic1 != nullptr);
    REQUIRE(ic2 != nullptr);
    CHECK_EQ(reg.size(), 2u);
    CHECK_EQ(reg.find_ic(1), ic1);
    CHECK_EQ(reg.find_ic(2), ic2);
    CHECK_EQ(reg.find_ic(3), nullptr);

    std::ostringstream ss;
    reg.dump_stats(ss);
    std::string report = ss.str();
    CHECK(report.find("Polymorphic Inline Cache (PIC) Statistics") != std::string::npos);
    CHECK(report.find("propA") != std::string::npos);
    CHECK(report.find("propB") != std::string::npos);
}

TEST_CASE("InlineCache - Patcher integration") {
    ShapeRegistry registry;
    Shape* shape = registry.transition_to(registry.get_root_shape(), "patched_prop");

    InlineCache ic(99, "patched_prop", 0, true);

    int64_t patch_target = 0;
    ic.set_patch_point(&patch_target);

    // Patch monomorphic IC should update the 64-bit constant at patch_target
    bool ok = patch_monomorphic_ic(ic, shape, 0);
    CHECK(ok);
    CHECK_EQ(reinterpret_cast<const Shape*>(patch_target), shape);
}
