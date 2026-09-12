#include "test_framework.hpp"
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/shape.hpp>
#include <sstream>

using namespace brass;
using namespace brass::runtime;
using namespace brass::test;

TEST_CASE("TypeFeedback - Recording call targets and monomorphic, polymorphic, megamorphic transitions") {
    TypeFeedbackVector tfv("test_func");
    CHECK_EQ(tfv.function_name(), "test_func");
    CHECK_EQ(tfv.slot_count(), 0);

    // Initial query
    CHECK(tfv.find_slot(1) == nullptr);

    // 1. Monomorphic: single target recorded once
    tfv.record_call_target(1, 0x1000, "callee_A");
    auto* slot = tfv.find_slot(1);
    REQUIRE(slot != nullptr);
    CHECK_EQ(slot->site_id, 1);
    CHECK_EQ(slot->total_invocations, 1);
    CHECK(slot->is_monomorphic());
    CHECK(!slot->is_polymorphic());
    CHECK(!slot->is_megamorphic());

    const CallFeedback* mono = slot->get_monomorphic_target();
    REQUIRE(mono != nullptr);
    CHECK_EQ(mono->target_addr, 0x1000);
    CHECK_EQ(mono->target_name, "callee_A");
    CHECK_EQ(mono->count, 1);

    // Record the same target again: still monomorphic, total and target count increment
    tfv.record_call_target(1, 0x1000, "callee_A");
    CHECK_EQ(slot->total_invocations, 2);
    CHECK(slot->is_monomorphic());
    CHECK_EQ(mono->count, 2);

    // 2. Polymorphic: record a second target (Degree 2)
    tfv.record_call_target(1, 0x2000, "callee_B");
    CHECK_EQ(slot->total_invocations, 3);
    CHECK(!slot->is_monomorphic());
    CHECK(slot->is_polymorphic());
    CHECK(!slot->is_megamorphic());
    CHECK(slot->get_monomorphic_target() == nullptr);
    CHECK_EQ(slot->targets.size(), 2);
    CHECK_EQ(slot->targets[1].target_name, "callee_B");
    CHECK_EQ(slot->targets[1].count, 1);

    // Record targets 3 and 4: still polymorphic (degree 3 and 4)
    tfv.record_call_target(1, 0x3000, "callee_C");
    CHECK(slot->is_polymorphic());
    CHECK_EQ(slot->targets.size(), 3);

    tfv.record_call_target(1, 0x4000, "callee_D");
    CHECK(slot->is_polymorphic());
    CHECK_EQ(slot->targets.size(), 4);
    CHECK(!slot->is_megamorphic());

    // 3. Megamorphic: record a fifth target (>4)
    tfv.record_call_target(1, 0x5000, "callee_E");
    CHECK(!slot->is_monomorphic());
    CHECK(!slot->is_polymorphic());
    CHECK(slot->is_megamorphic());
    CHECK_EQ(slot->targets.size(), 5);
}

TEST_CASE("TypeFeedback - Property shape recording and transitions") {
    TypeFeedbackVector tfv("prop_test");

    ShapeRegistry shape_reg;
    Shape* root = shape_reg.get_root_shape();
    Shape* s1 = shape_reg.transition_to(root, "x");
    Shape* s2 = shape_reg.transition_to(s1, "y");

    // Record shape s1 at site_id 10
    tfv.record_property_shape(10, s1, 0);
    auto* slot = tfv.find_slot(10);
    REQUIRE(slot != nullptr);
    CHECK_EQ(slot->kind, FeedbackSlotKind::Property);
    CHECK_EQ(slot->site_id, 10);
    CHECK_EQ(slot->total_invocations, 1);
    CHECK_EQ(slot->slot_index, 0);
    CHECK(slot->is_property_monomorphic());
    CHECK(!slot->is_property_polymorphic());
    CHECK_EQ(slot->get_monomorphic_shape(), s1);

    // Record same shape again
    tfv.record_property_shape(10, s1, 0);
    CHECK_EQ(slot->total_invocations, 2);
    CHECK_EQ(slot->observed_shapes.size(), 1);

    // Record second shape s2
    tfv.record_property_shape(10, s2, 1);
    CHECK_EQ(slot->total_invocations, 3);
    CHECK_EQ(slot->slot_index, 1);
    CHECK(!slot->is_property_monomorphic());
    CHECK(slot->is_property_polymorphic());
    CHECK_EQ(slot->observed_shapes.size(), 2);
}

TEST_CASE("TypeFeedback - FeedbackRegistry singleton and C bridge functions") {
    auto& reg = FeedbackRegistry::instance();
    reg.clear();

    // Test C bridge functions
    brass_record_call_feedback("test_bridge_fn", 42, 0xABCD, "target_xyz");
    const auto* tfv = reg.find("test_bridge_fn");
    REQUIRE(tfv != nullptr);
    CHECK_EQ(tfv->function_name(), "test_bridge_fn");

    const auto* slot = tfv->find_slot(42);
    REQUIRE(slot != nullptr);
    CHECK(slot->is_monomorphic());
    CHECK_EQ(slot->get_monomorphic_target()->target_addr, 0xABCD);
    CHECK_EQ(slot->get_monomorphic_target()->target_name, "target_xyz");

    // Test property C bridge
    ShapeRegistry shape_reg;
    Shape* root = shape_reg.get_root_shape();
    brass_record_property_feedback("test_bridge_fn", 99, root, 4);
    const auto* prop_slot = tfv->find_slot(99);
    REQUIRE(prop_slot != nullptr);
    CHECK_EQ(prop_slot->kind, FeedbackSlotKind::Property);
    CHECK_EQ(prop_slot->slot_index, 4);
    CHECK(prop_slot->is_property_monomorphic());

    // Dump stats should not throw
    std::ostringstream ss;
    reg.dump_stats(ss);
    CHECK(!ss.str().empty());
}
