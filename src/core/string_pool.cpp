#include <brass/core/string_pool.hpp>
#include <cstring>

namespace brass {

StringPool::StringPool()
    : arena_(16 * 1024) {}

StringPool::StringPool(size_t chunk_size)
    : arena_(chunk_size) {}

std::string_view StringPool::intern(std::string_view str) {
    if (str.empty()) {
        return std::string_view("");
    }

    auto it = pool_.find(str);
    if (it != pool_.end()) {
        return *it;
    }

    // Allocate memory from arena including null-terminator for convenience
    char* mem = static_cast<char*>(arena_.allocate(str.size() + 1, 1));
    std::memcpy(mem, str.data(), str.size());
    mem[str.size()] = '\0';

    std::string_view interned_view(mem, str.size());
    pool_.insert(interned_view);
    return interned_view;
}

std::string_view StringPool::intern(const char* str) {
    if (str == nullptr) {
        return std::string_view("");
    }
    return intern(std::string_view(str));
}

bool StringPool::contains(std::string_view str) const {
    if (str.empty()) {
        return true;
    }
    return pool_.find(str) != pool_.end();
}

size_t StringPool::size() const noexcept {
    return pool_.size();
}

void StringPool::clear() {
    pool_.clear();
    arena_.clear();
}

} // namespace brass
