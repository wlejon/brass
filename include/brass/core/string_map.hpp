#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace brass {

// Hash for string-keyed maps that are looked up by string_view: with
// std::equal_to<> it makes find() heterogeneous, so a lookup builds no key
// string.
struct StringViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

template <typename T>
using StringMap = std::unordered_map<std::string, T, StringViewHash, std::equal_to<>>;

} // namespace brass
