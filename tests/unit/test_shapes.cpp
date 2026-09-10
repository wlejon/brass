#include "test_framework.hpp"
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <string>

using namespace brass;
using namespace brass::runtime;

TEST_CASE("Shapes - Root shape creation and properties") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    REQUIRE(root != nullptr);
    CHECK_EQ(root->parent_shape(), nullptr);
    CHECK_EQ(root->slot_count(), 0u);
    CHECK_EQ(root->property_count(), 0u);
    CHECK_EQ(root->find_property("x"), nullptr);
    CHECK(!root->find_slot("x").has_value());
    CHECK_EQ(registry.shape_count(), 1u);
}

TEST_CASE("Shapes - Sequential property transitions and slots") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    Shape* s_x = registry.transition_to(root, "x");
    REQUIRE(s_x != nullptr);
    CHECK_NE(s_x, root);
    CHECK_EQ(s_x->parent_shape(), root);
    CHECK_EQ(s_x->slot_count(), 1u);
    CHECK_EQ(s_x->property_count(), 1u);

    const PropertyDescriptor* p_x = s_x->find_property("x");
    REQUIRE(p_x != nullptr);
    CHECK_EQ(p_x->name, "x");
    CHECK_EQ(p_x->slot_index, 0u);
    CHECK_EQ(s_x->find_slot("x").value_or(999), 0u);

    Shape* s_xy = registry.transition_to(s_x, "y");
    REQUIRE(s_xy != nullptr);
    CHECK_NE(s_xy, s_x);
    CHECK_EQ(s_xy->parent_shape(), s_x);
    CHECK_EQ(s_xy->slot_count(), 2u);
    CHECK_EQ(s_xy->property_count(), 2u);

    CHECK_EQ(s_xy->find_slot("x").value_or(999), 0u);
    CHECK_EQ(s_xy->find_slot("y").value_or(999), 1u);

    Shape* s_xyz = registry.transition_to(s_xy, "z");
    REQUIRE(s_xyz != nullptr);
    CHECK_EQ(s_xyz->slot_count(), 3u);
    CHECK_EQ(s_xyz->property_count(), 3u);
    CHECK_EQ(s_xyz->find_slot("x").value_or(999), 0u);
    CHECK_EQ(s_xyz->find_slot("y").value_or(999), 1u);
    CHECK_EQ(s_xyz->find_slot("z").value_or(999), 2u);
}

TEST_CASE("Shapes - Canonical deduplication and transition path reuse") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    // First path: root -> x -> y
    Shape* s_x1 = registry.transition_to(root, "x");
    Shape* s_y1 = registry.transition_to(s_x1, "y");

    // Second path: root -> x -> y
    Shape* s_x2 = registry.transition_to(root, "x");
    Shape* s_y2 = registry.transition_to(s_x2, "y");

    // Duplicate property additions must yield the EXACT same canonical Shape pointers
    CHECK_EQ(s_x1, s_x2);
    CHECK_EQ(s_y1, s_y2);

    // Calling transition on a shape for an existing property returns the shape itself
    Shape* s_x_again = registry.transition_to(s_x1, "x");
    CHECK_EQ(s_x_again, s_x1);

    // Registry shape count: root (1) + s_x (1) + s_y (1) = 3 total shapes
    CHECK_EQ(registry.shape_count(), 3u);
}

TEST_CASE("Shapes - Branching transition tree") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    Shape* s_x = registry.transition_to(root, "x");
    Shape* s_xy = registry.transition_to(s_x, "y");
    Shape* s_xz = registry.transition_to(s_x, "z");

    CHECK_NE(s_xy, s_xz);
    CHECK_EQ(s_xy->parent_shape(), s_x);
    CHECK_EQ(s_xz->parent_shape(), s_x);

    CHECK(s_xy->find_property("y") != nullptr);
    CHECK(s_xy->find_property("z") == nullptr);

    CHECK(s_xz->find_property("z") != nullptr);
    CHECK(s_xz->find_property("y") == nullptr);
}

TEST_CASE("Shapes - Property attributes and flags") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    PropertyAttributes ro_attr = PropertyAttributes::ReadOnly;
    Shape* s_ro = registry.transition_to(root, "constProp", 0, ro_attr);
    const PropertyDescriptor* p_ro = s_ro->find_property("constProp");
    REQUIRE(p_ro != nullptr);
    CHECK(p_ro->is_readonly());
    CHECK(p_ro->is_enumerable());
    CHECK(p_ro->is_configurable());
    CHECK(!p_ro->is_method());

    PropertyAttributes method_attr = PropertyAttributes::Method | PropertyAttributes::DontDelete;
    Shape* s_method = registry.transition_to(root, "myMethod", 0, method_attr);
    const PropertyDescriptor* p_method = s_method->find_property("myMethod");
    REQUIRE(p_method != nullptr);
    CHECK(!p_method->is_readonly());
    CHECK(p_method->is_method());
    CHECK(!p_method->is_configurable());
}

TEST_CASE("Shapes - Symbol ID transitions and lookups") {
    ShapeRegistry registry;
    Shape* root = registry.get_root_shape();

    Shape* s1 = registry.transition_to(root, 101);
    Shape* s2 = registry.transition_to(s1, 102);

    CHECK_EQ(s1->find_slot(101).value_or(999), 0u);
    CHECK_EQ(s2->find_slot(101).value_or(999), 0u);
    CHECK_EQ(s2->find_slot(102).value_or(999), 1u);

    // Reuse of symbol transition path
    Shape* s1_dup = registry.transition_to(root, 101);
    CHECK_EQ(s1, s1_dup);
    Shape* s2_dup = registry.transition_to(s1_dup, 102);
    CHECK_EQ(s2, s2_dup);
}

TEST_CASE("Shapes - DynamicObject mutation updates shapes") {
    ShapeRegistry registry;
    DynamicObject* obj1 = DynamicObject::create(nullptr, registry.get_root_shape());
    DynamicObject* obj2 = DynamicObject::create(nullptr, registry.get_root_shape());

    REQUIRE(obj1 != nullptr);
    REQUIRE(obj2 != nullptr);
    CHECK_EQ(obj1->shape, registry.get_root_shape());

    // obj1: set x, y
    obj1->set_property("x", HostValue::from_f64(10.5), registry);
    obj1->set_property("y", HostValue::from_f64(20.5), registry);

    // obj2: set x, y in same order
    obj2->set_property("x", HostValue::from_f64(30.0), registry);
    obj2->set_property("y", HostValue::from_f64(40.0), registry);

    // Both objects must share identical Shape pointer
    CHECK_EQ(obj1->shape, obj2->shape);
    CHECK_EQ(obj1->get_property("x").as_f64(), 10.5);
    CHECK_EQ(obj1->get_property("y").as_f64(), 20.5);
    CHECK_EQ(obj2->get_property("x").as_f64(), 30.0);
    CHECK_EQ(obj2->get_property("y").as_f64(), 40.0);

    DynamicObject::destroy_non_gc(obj1);
    DynamicObject::destroy_non_gc(obj2);
}
