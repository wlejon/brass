#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <concepts>

namespace brass {

class CardTable {
public:
    static constexpr size_t CARD_SHIFT = 9; // 512 bytes per card
    static constexpr size_t CARD_SIZE = 1ULL << CARD_SHIFT; // 512 bytes
    static constexpr uint8_t CARD_DIRTY = 0x00;
    static constexpr uint8_t CARD_CLEAN = 0x01;

    explicit CardTable(uintptr_t heap_base = 0, size_t heap_size = 0);
    ~CardTable() = default;

    CardTable(const CardTable&) = default;
    CardTable& operator=(const CardTable&) = default;
    CardTable(CardTable&&) noexcept = default;
    CardTable& operator=(CardTable&&) noexcept = default;

    void init(uintptr_t heap_base, size_t heap_size);

    [[nodiscard]] size_t card_index(uintptr_t addr) const noexcept {
        return (addr - heap_base_) >> CARD_SHIFT;
    }

    [[nodiscard]] uintptr_t card_address(size_t index) const noexcept {
        return heap_base_ + (index << CARD_SHIFT);
    }

    [[nodiscard]] bool is_valid_address(uintptr_t addr) const noexcept {
        return addr >= heap_base_ && addr < (heap_base_ + heap_size_);
    }

    [[nodiscard]] size_t card_count() const noexcept {
        return cards_.size();
    }

    [[nodiscard]] uintptr_t heap_base() const noexcept {
        return heap_base_;
    }

    [[nodiscard]] size_t heap_size() const noexcept {
        return heap_size_;
    }

    [[nodiscard]] uint8_t* byte_map_base() noexcept {
        return cards_.data();
    }

    [[nodiscard]] const uint8_t* byte_map_base() const noexcept {
        return cards_.data();
    }

    void mark_card(uintptr_t addr) noexcept {
        if (is_valid_address(addr)) {
            size_t idx = card_index(addr);
            cards_[idx] = CARD_DIRTY;
        }
    }

    void mark_card_by_index(size_t index) noexcept {
        if (index < cards_.size()) {
            cards_[index] = CARD_DIRTY;
        }
    }

    void mark_card_index(size_t index) noexcept {
        mark_card_by_index(index);
    }

    void mark_range(uintptr_t start, size_t size) noexcept {
        if (!is_valid_address(start) || size == 0) return;
        size_t start_idx = card_index(start);
        uintptr_t end = start + size - 1;
        size_t end_idx = is_valid_address(end) ? card_index(end) : (cards_.size() - 1);
        for (size_t i = start_idx; i <= end_idx && i < cards_.size(); ++i) {
            cards_[i] = CARD_DIRTY;
        }
    }

    [[nodiscard]] bool is_dirty_index(size_t index) const noexcept {
        return is_dirty(index);
    }

    [[nodiscard]] bool is_dirty(size_t index) const noexcept {
        if (index < cards_.size()) {
            return cards_[index] == CARD_DIRTY;
        }
        return false;
    }

    [[nodiscard]] bool is_dirty_addr(uintptr_t addr) const noexcept {
        if (is_valid_address(addr)) {
            return is_dirty(card_index(addr));
        }
        return false;
    }

    void clean_card(size_t index) noexcept {
        if (index < cards_.size()) {
            cards_[index] = CARD_CLEAN;
        }
    }

    void clean_card_addr(uintptr_t addr) noexcept {
        if (is_valid_address(addr)) {
            clean_card(card_index(addr));
        }
    }

    void clean_all() noexcept {
        if (!cards_.empty()) {
            std::memset(cards_.data(), CARD_CLEAN, cards_.size());
        }
    }

    [[nodiscard]] bool is_all_clean() const noexcept {
        for (uint8_t c : cards_) {
            if (c != CARD_CLEAN) return false;
        }
        return true;
    }

    template <typename Fn>
    void for_each_dirty_card(Fn&& callback) const {
        for_each_dirty_card_in_range(0, cards_.size(), std::forward<Fn>(callback));
    }

    template <typename Fn>
    void for_each_dirty_card_in_range(size_t start_card, size_t end_card, Fn&& callback) const {
        if (start_card >= cards_.size()) return;
        if (end_card > cards_.size()) end_card = cards_.size();
        if (start_card >= end_card) return;

        size_t i = start_card;
        while (i < end_card && (reinterpret_cast<uintptr_t>(&cards_[i]) & 7) != 0) {
            if (cards_[i] == CARD_DIRTY) {
                callback(i);
            }
            ++i;
        }

        static constexpr uint64_t CLEAN_WORD = 0x0101010101010101ULL;
        while (i + 8 <= end_card) {
            uint64_t word = *reinterpret_cast<const uint64_t*>(&cards_[i]);
            if (word != CLEAN_WORD) {
                for (size_t b = 0; b < 8; ++b) {
                    if (cards_[i + b] == CARD_DIRTY) {
                        callback(i + b);
                    }
                }
            }
            i += 8;
        }

        while (i < end_card) {
            if (cards_[i] == CARD_DIRTY) {
                callback(i);
            }
            ++i;
        }
    }

private:
    uintptr_t heap_base_ = 0;
    size_t heap_size_ = 0;
    std::vector<uint8_t> cards_;
};

} // namespace brass
