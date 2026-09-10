#include <brass/gc/card_table.hpp>

namespace brass {

CardTable::CardTable(uintptr_t heap_base, size_t heap_size) {
    if (heap_size > 0) {
        init(heap_base, heap_size);
    }
}

void CardTable::init(uintptr_t heap_base, size_t heap_size) {
    heap_base_ = heap_base;
    heap_size_ = heap_size;
    size_t num_cards = (heap_size + CARD_SIZE - 1) >> CARD_SHIFT;
    cards_.assign(num_cards, CARD_CLEAN);
}

} // namespace brass
