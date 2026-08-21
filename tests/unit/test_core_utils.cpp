#include "test_framework.hpp"
#include <brass/core/string_pool.hpp>
#include <brass/core/span.hpp>
#include <brass/core/bitset.hpp>
#include <brass/core/diagnostics.hpp>

using namespace brass;

TEST_CASE("StringPool interning and deduplication") {
    StringPool pool;

    std::string_view s1 = pool.intern("hello");
    std::string_view s2 = pool.intern("world");
    std::string_view s3 = pool.intern("hello");

    CHECK_EQ(s1, "hello");
    CHECK_EQ(s2, "world");
    CHECK_EQ(s1, s3);
    CHECK(s1.data() == s3.data()); // Pointer equality guaranteed for interned strings
    CHECK(s1.data() != s2.data());

    CHECK_EQ(pool.size(), size_t{2});
    CHECK(pool.contains("hello"));
    CHECK(pool.contains("world"));
    CHECK(!pool.contains("foo"));

    std::string_view empty = pool.intern("");
    CHECK_EQ(empty, "");
}

TEST_CASE("Span operations and slicing") {
    int arr[] = {10, 20, 30, 40, 50};
    Span<int> sp(arr);

    CHECK_EQ(sp.size(), size_t{5});
    CHECK_EQ(sp.size_bytes(), 5 * sizeof(int));
    CHECK_EQ(sp[0], 10);
    CHECK_EQ(sp.back(), 50);

    Span<int> sub = sp.subspan(1, 3);
    CHECK_EQ(sub.size(), size_t{3});
    CHECK_EQ(sub[0], 20);
    CHECK_EQ(sub[1], 30);
    CHECK_EQ(sub[2], 40);

    Span<int> first2 = sp.first(2);
    CHECK_EQ(first2.size(), size_t{2});
    CHECK_EQ(first2[0], 10);
    CHECK_EQ(first2[1], 20);

    Span<int> last2 = sp.last(2);
    CHECK_EQ(last2.size(), size_t{2});
    CHECK_EQ(last2[0], 40);
    CHECK_EQ(last2[1], 50);
}

TEST_CASE("BitSet dynamic operations") {
    BitSet bs(128);
    CHECK_EQ(bs.size(), size_t{128});
    CHECK(bs.none());
    CHECK_EQ(bs.count(), size_t{0});

    bs.set(5);
    bs.set(63);
    bs.set(64);
    bs.set(127);

    CHECK(bs.test(5));
    CHECK(bs.test(63));
    CHECK(bs.test(64));
    CHECK(bs.test(127));
    CHECK(!bs.test(0));
    CHECK(!bs.test(6));
    CHECK_EQ(bs.count(), size_t{4});

    CHECK_EQ(bs.find_first(), size_t{5});
    CHECK_EQ(bs.find_next(5), size_t{63});
    CHECK_EQ(bs.find_next(63), size_t{64});
    CHECK_EQ(bs.find_next(64), size_t{127});
    CHECK_EQ(bs.find_next(127), BitSet::npos);

    bs.reset(63);
    CHECK(!bs.test(63));
    CHECK_EQ(bs.count(), size_t{3});

    BitSet other(128);
    other.set(5);
    other.set(100);

    BitSet and_res = bs & other;
    CHECK_EQ(and_res.count(), size_t{1});
    CHECK(and_res.test(5));

    BitSet or_res = bs | other;
    CHECK_EQ(or_res.count(), size_t{4});
    CHECK(or_res.test(5));
    CHECK(or_res.test(64));
    CHECK(or_res.test(100));
    CHECK(or_res.test(127));

    BitSet diff = bs.difference(other);
    CHECK(!diff.test(5));
    CHECK(diff.test(64));
    CHECK(diff.test(127));
}

TEST_CASE("SmallBitSet fixed stack operations") {
    SmallBitSet<64> sbs;
    CHECK(sbs.none());
    CHECK_EQ(sbs.count(), size_t{0});

    sbs.set(0);
    sbs.set(31);
    sbs.set(63);
    CHECK_EQ(sbs.count(), size_t{3});
    CHECK(sbs[0]);
    CHECK(sbs[31]);
    CHECK(sbs[63]);
    CHECK(!sbs[1]);

    CHECK_EQ(sbs.find_first(), size_t{0});
    CHECK_EQ(sbs.find_next(0), size_t{31});
    CHECK_EQ(sbs.find_next(31), size_t{63});
    CHECK_EQ(sbs.find_next(63), SmallBitSet<64>::npos);

    sbs.set_all();
    CHECK(sbs.all());
    CHECK_EQ(sbs.count(), size_t{64});

    sbs.reset();
    CHECK(sbs.none());
}

TEST_CASE("DiagnosticReporter error collection and formatting") {
    DiagnosticReporter diag;
    CHECK(!diag.has_errors());
    CHECK(!diag.has_warnings());

    diag.warning(SourceLocation("test.mir", 10, 5), "Unused variable %x");
    diag.error(SourceLocation("test.mir", 20, 1), "Type mismatch in add instruction");

    CHECK(diag.has_errors());
    CHECK(diag.has_warnings());
    CHECK_EQ(diag.error_count(), size_t{1});
    CHECK_EQ(diag.warning_count(), size_t{1});

    std::string output = diag.format_all();
    CHECK(output.find("test.mir:10:5: warning: Unused variable %x") != std::string::npos);
    CHECK(output.find("test.mir:20:1: error: Type mismatch in add instruction") != std::string::npos);

    diag.clear();
    CHECK(!diag.has_errors());
    CHECK_EQ(diag.error_count(), size_t{0});
}
