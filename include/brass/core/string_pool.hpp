#pragma once

#include <brass/core/arena.hpp>
#include <string_view>
#include <unordered_set>

namespace brass {

class StringPool {
public:
    StringPool();
    explicit StringPool(size_t chunk_size);
    ~StringPool() = default;

    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
    StringPool(StringPool&&) noexcept = default;
    StringPool& operator=(StringPool&&) noexcept = default;

    std::string_view intern(std::string_view str);
    std::string_view intern(const char* str);

    bool contains(std::string_view str) const;
    size_t size() const noexcept;
    void clear();

private:
    Arena arena_;
    std::unordered_set<std::string_view> pool_;
};

} // namespace brass
