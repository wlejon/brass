#include "test_framework.hpp"
#include <brass/core/arena.hpp>
#include <cstring>
#include <vector>

using namespace brass;

TEST_CASE("Arena basic allocation and alignment") {
    Arena arena(1024); // 1 KB chunk

    void* p1 = arena.allocate(10, 8);
    CHECK(p1 != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p1) % 8 == 0);

    void* p2 = arena.allocate(17, 16);
    CHECK(p2 != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p2) % 16 == 0);

    void* p3 = arena.allocate(5, 32);
    CHECK(p3 != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p3) % 32 == 0);

    void* p4 = arena.allocate(128, 64);
    CHECK(p4 != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p4) % 64 == 0);

    CHECK(arena.bytes_allocated() >= (10 + 17 + 5 + 128));
}

TEST_CASE("Arena make objects with constructors") {
    Arena arena(1024);

    struct Point {
        int x;
        int y;
        std::string name;
        Point(int x_, int y_, std::string n) : x(x_), y(y_), name(std::move(n)) {}
    };

    Point* pt = arena.make<Point>(42, 84, "origin");
    CHECK_EQ(pt->x, 42);
    CHECK_EQ(pt->y, 84);
    CHECK_EQ(pt->name, "origin");
}

TEST_CASE("Arena span allocation and copying") {
    Arena arena(1024);

    Span<int> span1 = arena.allocate_span<int>(5);
    CHECK_EQ(span1.size(), size_t{5});
    for (size_t i = 0; i < span1.size(); ++i) {
        span1[i] = static_cast<int>(i * 10);
    }

    std::vector<int> src = {100, 200, 300};
    Span<int> span2 = arena.copy_span(src);
    CHECK_EQ(span2.size(), size_t{3});
    CHECK_EQ(span2[0], 100);
    CHECK_EQ(span2[1], 200);
    CHECK_EQ(span2[2], 300);

    Span<int> span3 = arena.copy_span(span1);
    CHECK(span1 == span3);
}

TEST_CASE("Arena large allocation exceeding chunk size") {
    Arena arena(512);

    void* large = arena.allocate(4096, 64);
    CHECK(large != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(large) % 64 == 0);
    std::memset(large, 0xAB, 4096);

    void* small = arena.allocate(32, 8);
    CHECK(small != nullptr);
    std::memset(small, 0xCD, 32);

    CHECK(arena.bytes_capacity() >= 4096);
}

TEST_CASE("Arena scoped marker reset") {
    Arena arena(1024);

    arena.allocate(100, 8);
    Arena::Marker marker = arena.get_marker();
    size_t alloc_before = arena.bytes_allocated();

    void* temp1 = arena.allocate(200, 8);
    void* temp2 = arena.allocate(300, 8);
    CHECK(temp1 != nullptr);
    CHECK(temp2 != nullptr);
    CHECK(arena.bytes_allocated() > alloc_before);

    arena.reset_to_marker(marker);
    CHECK_EQ(arena.bytes_allocated(), alloc_before);

    void* new_alloc = arena.allocate(200, 8);
    CHECK(new_alloc != nullptr);
}

TEST_CASE("Arena ScopedArenaReset RAII helper") {
    Arena arena(1024);

    arena.allocate(50, 8);
    size_t initial_alloc = arena.bytes_allocated();

    {
        ScopedArenaReset scope(arena);
        arena.allocate(250, 8);
        arena.allocate(150, 8);
        CHECK(arena.bytes_allocated() >= initial_alloc + 400);
    }

    CHECK_EQ(arena.bytes_allocated(), initial_alloc);
}

TEST_CASE("Arena reset and clear") {
    Arena arena(1024);

    arena.allocate(100, 8);
    arena.allocate(200, 8);
    CHECK(arena.bytes_allocated() > 0);
    size_t cap = arena.bytes_capacity();

    arena.reset();
    CHECK_EQ(arena.bytes_allocated(), size_t{0});
    CHECK_EQ(arena.bytes_capacity(), cap); // Memory chunks retained

    arena.allocate(50, 8);
    CHECK_EQ(arena.bytes_allocated(), size_t{50});

    arena.clear();
    CHECK_EQ(arena.bytes_allocated(), size_t{0});
    CHECK_EQ(arena.bytes_capacity(), size_t{0});
    CHECK_EQ(arena.chunk_count(), size_t{0});
}
