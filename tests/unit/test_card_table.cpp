#include "test_framework.hpp"
#include <brass/gc/card_table.hpp>
#include <vector>

using namespace brass;

TEST_CASE("CardTable - Initial state is clean") {
    constexpr size_t HEAP_SIZE = 64 * 1024; // 64 KB = 128 cards
    std::vector<uint8_t> heap(HEAP_SIZE);
    uintptr_t heap_base = reinterpret_cast<uintptr_t>(heap.data());

    CardTable ct(heap_base, HEAP_SIZE);
    CHECK_EQ(ct.card_count(), 128ULL);
    CHECK(ct.is_all_clean());

    for (size_t i = 0; i < HEAP_SIZE; i += CardTable::CARD_SIZE) {
        CHECK(!ct.is_dirty_addr(heap_base + i));
    }
}

TEST_CASE("CardTable - Mark and clean individual cards") {
    constexpr size_t HEAP_SIZE = 32 * 1024; // 32 KB = 64 cards
    std::vector<uint8_t> heap(HEAP_SIZE);
    uintptr_t heap_base = reinterpret_cast<uintptr_t>(heap.data());

    CardTable ct(heap_base, HEAP_SIZE);

    uintptr_t target_addr = heap_base + 512 * 5 + 42; // card index 5
    CHECK_EQ(ct.card_index(target_addr), 5ULL);

    CHECK(!ct.is_dirty_addr(target_addr));
    ct.mark_card(target_addr);
    CHECK(ct.is_dirty_addr(target_addr));
    CHECK(!ct.is_all_clean());

    // Neighboring cards should still be clean
    CHECK(!ct.is_dirty_addr(heap_base + 512 * 4));
    CHECK(!ct.is_dirty_addr(heap_base + 512 * 6));

    // Clean this specific card
    ct.clean_card_addr(target_addr);
    CHECK(!ct.is_dirty_addr(target_addr));
    CHECK(ct.is_all_clean());
}

TEST_CASE("CardTable - Mark ranges and clean_all") {
    constexpr size_t HEAP_SIZE = 128 * 1024; // 256 cards
    std::vector<uint8_t> heap(HEAP_SIZE);
    uintptr_t heap_base = reinterpret_cast<uintptr_t>(heap.data());

    CardTable ct(heap_base, HEAP_SIZE);

    // Mark range spanning cards 10 to 14
    uintptr_t start_addr = heap_base + 10 * CardTable::CARD_SIZE + 100;
    size_t size = 4 * CardTable::CARD_SIZE; // crosses multiple cards
    ct.mark_range(start_addr, size);

    CHECK(!ct.is_dirty_index(9));
    CHECK(ct.is_dirty_index(10));
    CHECK(ct.is_dirty_index(11));
    CHECK(ct.is_dirty_index(12));
    CHECK(ct.is_dirty_index(13));
    CHECK(ct.is_dirty_index(14));
    CHECK(!ct.is_dirty_index(15));

    ct.clean_all();
    CHECK(ct.is_all_clean());
    for (size_t i = 10; i <= 14; ++i) {
        CHECK(!ct.is_dirty_index(i));
    }
}

TEST_CASE("CardTable - for_each_dirty_card callback") {
    constexpr size_t HEAP_SIZE = 64 * 1024; // 128 cards
    std::vector<uint8_t> heap(HEAP_SIZE);
    uintptr_t heap_base = reinterpret_cast<uintptr_t>(heap.data());

    CardTable ct(heap_base, HEAP_SIZE);

    ct.mark_card_index(3);
    ct.mark_card_index(70);
    ct.mark_card_index(127);

    std::vector<size_t> visited_indices;
    ct.for_each_dirty_card([&](size_t card_idx) {
        visited_indices.push_back(card_idx);
        uintptr_t card_start = ct.card_address(card_idx);
        uintptr_t card_end = card_start + CardTable::CARD_SIZE;
        CHECK_EQ(card_start, heap_base + card_idx * CardTable::CARD_SIZE);
        CHECK_EQ(card_end, card_start + CardTable::CARD_SIZE);
    });

    REQUIRE_EQ(visited_indices.size(), 3ULL);
    CHECK_EQ(visited_indices[0], 3ULL);
    CHECK_EQ(visited_indices[1], 70ULL);
    CHECK_EQ(visited_indices[2], 127ULL);
}

TEST_CASE("CardTable - Fast 64-bit word skipping") {
    // Heap size with multiple 64-card words (512 cards = 256 KB)
    constexpr size_t HEAP_SIZE = 256 * 1024; // 512 cards = 8 words of 64 bytes
    std::vector<uint8_t> heap(HEAP_SIZE);
    uintptr_t heap_base = reinterpret_cast<uintptr_t>(heap.data());

    CardTable ct(heap_base, HEAP_SIZE);

    // Dirty cards in word 0, word 4, word 7
    ct.mark_card_index(10);
    ct.mark_card_index(4 * 64 + 5);
    ct.mark_card_index(7 * 64 + 63);

    std::vector<size_t> visited;
    ct.for_each_dirty_card([&](size_t card_idx) {
        visited.push_back(card_idx);
    });

    REQUIRE_EQ(visited.size(), 3ULL);
    CHECK_EQ(visited[0], 10ULL);
    CHECK_EQ(visited[1], 261ULL);
    CHECK_EQ(visited[2], 511ULL);
}
