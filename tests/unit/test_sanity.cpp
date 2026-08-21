#include "test_framework.hpp"
#include <brass/brass.hpp>

TEST_CASE("Sanity check - test framework runs") {
    int one = 1;
    CHECK(one + one == 2);
    REQUIRE_EQ(2 * 3, 6);
    CHECK_NE(10, 20);
}

TEST_CASE("Brass version info") {
    CHECK_EQ(brass::version_major(), 0);
    CHECK_EQ(brass::version_minor(), 1);
    CHECK_EQ(brass::version_patch(), 0);
}

BRASS_TEST_MAIN()
