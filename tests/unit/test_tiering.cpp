#include "test_framework.hpp"
#include <brass/runtime/tiering.hpp>
#include <sstream>

using namespace brass::runtime;

TEST_CASE("Tiering - Feedback counters and threshold progression") {
    TieringFeedback fb("hot_fn");
    CHECK_EQ(fb.function_name(), "hot_fn");
    CHECK_EQ(fb.tier_level(), TierLevel::Tier0_Interpreter);
    CHECK_EQ(fb.invocation_count(), 0);
    CHECK_EQ(fb.backedge_count(), 0);
    CHECK_EQ(fb.deopt_count(), 0);
    CHECK(!fb.is_bailed_out());

    // Invocations
    for (uint64_t i = 0; i < 50; ++i) {
        fb.record_invocation();
    }
    CHECK_EQ(fb.invocation_count(), 50);
    CHECK(!fb.should_tier_up(100));

    for (uint64_t i = 0; i < 50; ++i) {
        fb.record_invocation();
    }
    CHECK_EQ(fb.invocation_count(), 100);
    CHECK(fb.should_tier_up(100));

    // Backedges (OSR trigger)
    for (uint64_t i = 0; i < 150; ++i) {
        fb.record_backedge();
    }
    CHECK_EQ(fb.backedge_count(), 150);
    CHECK(fb.should_trigger_osr(100));
}

TEST_CASE("Tiering - Deopt Ratchet and Bailout Behavior") {
    TieringFeedback fb("failing_guard_fn");
    fb.set_tier_level(TierLevel::Tier2_Optimized);
    CHECK_EQ(fb.tier_level(), TierLevel::Tier2_Optimized);

    // Record deopts below ratchet threshold
    for (uint64_t i = 0; i < 4; ++i) {
        fb.record_deoptimization();
    }
    CHECK_EQ(fb.deopt_count(), 4);
    CHECK_EQ(fb.tier_level(), TierLevel::Tier2_Optimized);

    // 5th deopt ratchets down from Tier2 to Tier1
    fb.record_deoptimization();
    CHECK_EQ(fb.deopt_count(), 5);
    CHECK_EQ(fb.tier_level(), TierLevel::Tier1_Baseline);

    // Continuing to deopt ratchets down to Tier0 then bails out
    for (uint64_t i = 0; i < 5; ++i) {
        fb.record_deoptimization();
    }
    CHECK(fb.is_bailed_out());
    CHECK(!fb.bailout_reason().empty());
}

TEST_CASE("Tiering - Explicit Bailout and Reset") {
    TieringFeedback fb("unstable_fn");
    fb.record_invocation();
    fb.record_backedge();
    fb.record_bailout("Guard failed continuously in loop");

    CHECK(fb.is_bailed_out());
    CHECK_EQ(fb.bailout_reason(), "Guard failed continuously in loop");
    CHECK(!fb.should_trigger_osr(1));
    CHECK(!fb.should_tier_up(1));

    fb.reset();
    CHECK(!fb.is_bailed_out());
    CHECK_EQ(fb.invocation_count(), 0);
    CHECK_EQ(fb.backedge_count(), 0);
    CHECK_EQ(fb.tier_level(), TierLevel::Tier0_Interpreter);
}

TEST_CASE("Tiering - Registry Registry Lookups and Statistics") {
    TieringRegistry& reg = TieringRegistry::instance();
    reg.clear();

    CHECK(!reg.has("test_fn"));
    TieringFeedback& fb = reg.get_or_create("test_fn");
    CHECK_EQ(fb.function_name(), "test_fn");
    CHECK(reg.has("test_fn"));

    fb.record_invocation();
    fb.record_backedge();

    std::ostringstream oss;
    reg.dump_stats(oss);
    std::string out = oss.str();
    CHECK(out.find("test_fn") != std::string::npos);

    reg.clear();
    CHECK(!reg.has("test_fn"));
}
