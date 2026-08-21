#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <bit>
#include <string>
#include <algorithm>

namespace brass {

class BitSet {
public:
    static constexpr size_t npos = static_cast<size_t>(-1);
    static constexpr size_t BitsPerWord = 64;

    BitSet() noexcept : num_bits_(0) {}
    
    explicit BitSet(size_t num_bits) : num_bits_(num_bits) {
        words_.resize((num_bits + BitsPerWord - 1) / BitsPerWord, 0);
    }

    BitSet(size_t num_bits, bool init_val) : num_bits_(num_bits) {
        uint64_t fill = init_val ? ~uint64_t{0} : uint64_t{0};
        words_.resize((num_bits + BitsPerWord - 1) / BitsPerWord, fill);
        trim_unused_bits();
    }

    size_t size() const noexcept { return num_bits_; }
    bool empty() const noexcept { return num_bits_ == 0; }

    void resize(size_t num_bits) {
        resize(num_bits, false);
    }

    void resize(size_t num_bits, bool init_val) {
        size_t old_words = words_.size();
        size_t new_words = (num_bits + BitsPerWord - 1) / BitsPerWord;
        uint64_t fill = init_val ? ~uint64_t{0} : uint64_t{0};
        words_.resize(new_words, fill);
        if (init_val && num_bits > num_bits_ && old_words > 0 && (num_bits_ % BitsPerWord != 0)) {
            // Set newly exposed bits in the previously partial word
            size_t rem = num_bits_ % BitsPerWord;
            words_[old_words - 1] |= (~uint64_t{0} << rem);
        }
        num_bits_ = num_bits;
        trim_unused_bits();
    }

    void set(size_t bit_idx) {
        if (bit_idx < num_bits_) {
            words_[bit_idx / BitsPerWord] |= (uint64_t{1} << (bit_idx % BitsPerWord));
        }
    }

    void set(size_t bit_idx, bool val) {
        if (val) {
            set(bit_idx);
        } else {
            reset(bit_idx);
        }
    }

    void reset(size_t bit_idx) {
        if (bit_idx < num_bits_) {
            words_[bit_idx / BitsPerWord] &= ~(uint64_t{1} << (bit_idx % BitsPerWord));
        }
    }

    void set_all() {
        std::fill(words_.begin(), words_.end(), ~uint64_t{0});
        trim_unused_bits();
    }

    void reset() noexcept {
        std::fill(words_.begin(), words_.end(), uint64_t{0});
    }

    void clear() noexcept {
        words_.clear();
        num_bits_ = 0;
    }

    bool test(size_t bit_idx) const noexcept {
        if (bit_idx >= num_bits_) return false;
        return (words_[bit_idx / BitsPerWord] & (uint64_t{1} << (bit_idx % BitsPerWord))) != 0;
    }

    bool operator[](size_t bit_idx) const noexcept {
        return test(bit_idx);
    }

    size_t count() const noexcept {
        size_t total = 0;
        for (uint64_t w : words_) {
            total += static_cast<size_t>(std::popcount(w));
        }
        return total;
    }

    bool any() const noexcept {
        for (uint64_t w : words_) {
            if (w != 0) return true;
        }
        return false;
    }

    bool none() const noexcept {
        return !any();
    }

    bool all() const noexcept {
        if (num_bits_ == 0) return true;
        return count() == num_bits_;
    }

    size_t find_first() const noexcept {
        for (size_t i = 0; i < words_.size(); ++i) {
            if (words_[i] != 0) {
                size_t bit = static_cast<size_t>(std::countr_zero(words_[i]));
                size_t idx = i * BitsPerWord + bit;
                return (idx < num_bits_) ? idx : npos;
            }
        }
        return npos;
    }

    size_t find_next(size_t prev) const noexcept {
        if (prev + 1 >= num_bits_) {
            return npos;
        }
        size_t next_idx = prev + 1;
        size_t word_idx = next_idx / BitsPerWord;
        size_t bit_offset = next_idx % BitsPerWord;

        // Mask off bits below bit_offset in the first word
        uint64_t w = words_[word_idx] & (~uint64_t{0} << bit_offset);
        if (w != 0) {
            size_t bit = static_cast<size_t>(std::countr_zero(w));
            size_t idx = word_idx * BitsPerWord + bit;
            return (idx < num_bits_) ? idx : npos;
        }

        for (size_t i = word_idx + 1; i < words_.size(); ++i) {
            if (words_[i] != 0) {
                size_t bit = static_cast<size_t>(std::countr_zero(words_[i]));
                size_t idx = i * BitsPerWord + bit;
                return (idx < num_bits_) ? idx : npos;
            }
        }
        return npos;
    }

    BitSet& operator&=(const BitSet& other) {
        size_t common = std::min(words_.size(), other.words_.size());
        for (size_t i = 0; i < common; ++i) {
            words_[i] &= other.words_[i];
        }
        for (size_t i = common; i < words_.size(); ++i) {
            words_[i] = 0;
        }
        return *this;
    }

    BitSet& operator|=(const BitSet& other) {
        if (other.num_bits_ > num_bits_) {
            resize(other.num_bits_);
        }
        size_t common = std::min(words_.size(), other.words_.size());
        for (size_t i = 0; i < common; ++i) {
            words_[i] |= other.words_[i];
        }
        trim_unused_bits();
        return *this;
    }

    BitSet& operator^=(const BitSet& other) {
        if (other.num_bits_ > num_bits_) {
            resize(other.num_bits_);
        }
        size_t common = std::min(words_.size(), other.words_.size());
        for (size_t i = 0; i < common; ++i) {
            words_[i] ^= other.words_[i];
        }
        trim_unused_bits();
        return *this;
    }

    BitSet operator~() const {
        BitSet res = *this;
        for (size_t i = 0; i < res.words_.size(); ++i) {
            res.words_[i] = ~res.words_[i];
        }
        res.trim_unused_bits();
        return res;
    }

    // Set difference: this & ~other
    BitSet difference(const BitSet& other) const {
        BitSet res = *this;
        size_t common = std::min(res.words_.size(), other.words_.size());
        for (size_t i = 0; i < common; ++i) {
            res.words_[i] &= ~other.words_[i];
        }
        return res;
    }

    bool operator==(const BitSet& other) const noexcept {
        if (num_bits_ != other.num_bits_) return false;
        return words_ == other.words_;
    }

    bool operator!=(const BitSet& other) const noexcept {
        return !(*this == other);
    }

    friend BitSet operator&(BitSet a, const BitSet& b) {
        a &= b;
        return a;
    }

    friend BitSet operator|(BitSet a, const BitSet& b) {
        a |= b;
        return a;
    }

    friend BitSet operator^(BitSet a, const BitSet& b) {
        a ^= b;
        return a;
    }

private:
    void trim_unused_bits() noexcept {
        if (num_bits_ % BitsPerWord != 0 && !words_.empty()) {
            size_t rem = num_bits_ % BitsPerWord;
            uint64_t mask = (uint64_t{1} << rem) - 1;
            words_.back() &= mask;
        }
    }

    size_t num_bits_ = 0;
    std::vector<uint64_t> words_;
};

template <size_t N>
class SmallBitSet {
public:
    static constexpr size_t npos = static_cast<size_t>(-1);
    static constexpr size_t BitsPerWord = 64;
    static constexpr size_t NumWords = (N + BitsPerWord - 1) / BitsPerWord;

    constexpr SmallBitSet() noexcept : words_{} {}

    constexpr size_t size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return N == 0; }

    constexpr void set(size_t bit_idx) noexcept {
        if (bit_idx < N) {
            words_[bit_idx / BitsPerWord] |= (uint64_t{1} << (bit_idx % BitsPerWord));
        }
    }

    constexpr void set(size_t bit_idx, bool val) noexcept {
        if (val) {
            set(bit_idx);
        } else {
            reset(bit_idx);
        }
    }

    constexpr void reset(size_t bit_idx) noexcept {
        if (bit_idx < N) {
            words_[bit_idx / BitsPerWord] &= ~(uint64_t{1} << (bit_idx % BitsPerWord));
        }
    }

    constexpr void set_all() noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            words_[i] = ~uint64_t{0};
        }
        trim_unused_bits();
    }

    constexpr void reset() noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            words_[i] = 0;
        }
    }

    constexpr bool test(size_t bit_idx) const noexcept {
        if (bit_idx >= N) return false;
        return (words_[bit_idx / BitsPerWord] & (uint64_t{1} << (bit_idx % BitsPerWord))) != 0;
    }

    constexpr bool operator[](size_t bit_idx) const noexcept {
        return test(bit_idx);
    }

    constexpr size_t count() const noexcept {
        size_t total = 0;
        for (size_t i = 0; i < NumWords; ++i) {
            total += static_cast<size_t>(std::popcount(words_[i]));
        }
        return total;
    }

    constexpr bool any() const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            if (words_[i] != 0) return true;
        }
        return false;
    }

    constexpr bool none() const noexcept {
        return !any();
    }

    constexpr bool all() const noexcept {
        if constexpr (N == 0) return true;
        return count() == N;
    }

    constexpr size_t find_first() const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            if (words_[i] != 0) {
                size_t bit = static_cast<size_t>(std::countr_zero(words_[i]));
                size_t idx = i * BitsPerWord + bit;
                return (idx < N) ? idx : npos;
            }
        }
        return npos;
    }

    constexpr size_t find_next(size_t prev) const noexcept {
        if (prev + 1 >= N) {
            return npos;
        }
        size_t next_idx = prev + 1;
        size_t word_idx = next_idx / BitsPerWord;
        size_t bit_offset = next_idx % BitsPerWord;

        uint64_t w = words_[word_idx] & (~uint64_t{0} << bit_offset);
        if (w != 0) {
            size_t bit = static_cast<size_t>(std::countr_zero(w));
            size_t idx = word_idx * BitsPerWord + bit;
            return (idx < N) ? idx : npos;
        }

        for (size_t i = word_idx + 1; i < NumWords; ++i) {
            if (words_[i] != 0) {
                size_t bit = static_cast<size_t>(std::countr_zero(words_[i]));
                size_t idx = i * BitsPerWord + bit;
                return (idx < N) ? idx : npos;
            }
        }
        return npos;
    }

    constexpr SmallBitSet& operator&=(const SmallBitSet& other) noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            words_[i] &= other.words_[i];
        }
        return *this;
    }

    constexpr SmallBitSet& operator|=(const SmallBitSet& other) noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            words_[i] |= other.words_[i];
        }
        trim_unused_bits();
        return *this;
    }

    constexpr SmallBitSet& operator^=(const SmallBitSet& other) noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            words_[i] ^= other.words_[i];
        }
        trim_unused_bits();
        return *this;
    }

    constexpr SmallBitSet operator~() const noexcept {
        SmallBitSet res = *this;
        for (size_t i = 0; i < NumWords; ++i) {
            res.words_[i] = ~res.words_[i];
        }
        res.trim_unused_bits();
        return res;
    }

    constexpr bool operator==(const SmallBitSet& other) const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            if (words_[i] != other.words_[i]) return false;
        }
        return true;
    }

    constexpr bool operator!=(const SmallBitSet& other) const noexcept {
        return !(*this == other);
    }

private:
    constexpr void trim_unused_bits() noexcept {
        if constexpr (N % BitsPerWord != 0 && NumWords > 0) {
            size_t rem = N % BitsPerWord;
            uint64_t mask = (uint64_t{1} << rem) - 1;
            words_[NumWords - 1] &= mask;
        }
    }

    std::array<uint64_t, (N + BitsPerWord - 1) / BitsPerWord == 0 ? 1 : (N + BitsPerWord - 1) / BitsPerWord> words_{};
};

} // namespace brass
