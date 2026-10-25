#include <brass/core/env.hpp>

#include <cstdlib>

namespace brass {

std::string env_value(const char* name) {
#if defined(_MSC_VER)
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return {};
    std::string v(owned);
    std::free(owned);
    return v;
#else
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
}

} // namespace brass
