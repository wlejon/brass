#pragma once

#include <cstddef>
#include <type_traits>
#include <iterator>
#include <stdexcept>

namespace brass {

template <typename T>
class Span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr Span() noexcept : data_(nullptr), size_(0) {}
    constexpr Span(T* data, size_type size) noexcept : data_(data), size_(size) {}
    constexpr Span(T* first, T* last) noexcept : data_(first), size_(static_cast<size_type>(last - first)) {}

    template <size_type N>
    constexpr Span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}

    template <typename Container,
              typename = std::enable_if_t<
                  !std::is_same_v<std::remove_cvref_t<Container>, Span> &&
                  std::is_convertible_v<decltype(std::declval<Container&>().data()), T*>>>
    constexpr Span(Container& c) noexcept : data_(c.data()), size_(c.size()) {}

    template <typename Container,
              typename = std::enable_if_t<
                  !std::is_same_v<std::remove_cvref_t<Container>, Span> &&
                  std::is_convertible_v<decltype(std::declval<const Container&>().data()), T*>>>
    constexpr Span(const Container& c) noexcept : data_(c.data()), size_(c.size()) {}

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    constexpr Span(const Span<U>& other) noexcept : data_(other.data()), size_(other.size()) {}

    constexpr pointer data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type size_bytes() const noexcept { return size_ * sizeof(T); }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr reference operator[](size_type idx) const noexcept { return data_[idx]; }
    constexpr reference front() const noexcept { return data_[0]; }
    constexpr reference back() const noexcept { return data_[size_ - 1]; }

    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }
    constexpr const_iterator cbegin() const noexcept { return data_; }
    constexpr const_iterator cend() const noexcept { return data_ + size_; }

    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    constexpr Span<T> subspan(size_type offset) const {
        if (offset > size_) {
            return Span<T>();
        }
        return Span<T>(data_ + offset, size_ - offset);
    }

    constexpr Span<T> subspan(size_type offset, size_type count) const {
        if (offset > size_) {
            return Span<T>();
        }
        size_type available = size_ - offset;
        size_type actual = (count < available) ? count : available;
        return Span<T>(data_ + offset, actual);
    }

    constexpr Span<T> first(size_type count) const {
        return subspan(0, count);
    }

    constexpr Span<T> last(size_type count) const {
        if (count >= size_) return *this;
        return subspan(size_ - count, count);
    }

    bool operator==(const Span<T>& other) const noexcept {
        if (size_ != other.size_) return false;
        for (size_type i = 0; i < size_; ++i) {
            if (!(data_[i] == other.data_[i])) return false;
        }
        return true;
    }

    bool operator!=(const Span<T>& other) const noexcept {
        return !(*this == other);
    }

private:
    T* data_ = nullptr;
    size_type size_ = 0;
};

template <typename T>
Span(T*, std::size_t) -> Span<T>;

template <typename T, std::size_t N>
Span(T (&)[N]) -> Span<T>;

template <typename Container>
Span(Container&) -> Span<typename Container::value_type>;

template <typename Container>
Span(const Container&) -> Span<const typename Container::value_type>;

} // namespace brass
