#include <brass/mir/range_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/instruction.hpp>
#include <iostream>
#include <sstream>
#include <cmath>
#include <bit>
#include <algorithm>
#include <limits>

namespace brass {

namespace {

#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 int128_t;
#endif

inline bool add_overflows(int64_t a, int64_t b, int64_t& out) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) + b;
    if (res > INT64_MAX || res < INT64_MIN) return true;
    out = static_cast<int64_t>(res);
    return false;
#else
    if (b > 0 && a > INT64_MAX - b) return true;
    if (b < 0 && a < INT64_MIN - b) return true;
    out = a + b;
    return false;
#endif
}

inline bool sub_overflows(int64_t a, int64_t b, int64_t& out) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) - b;
    if (res > INT64_MAX || res < INT64_MIN) return true;
    out = static_cast<int64_t>(res);
    return false;
#else
    if (b < 0 && a > INT64_MAX + b) return true;
    if (b > 0 && a < INT64_MIN + b) return true;
    out = a - b;
    return false;
#endif
}

inline bool mul_overflows(int64_t a, int64_t b, int64_t& out) noexcept {
#if defined(__SIZEOF_INT128__)
    int128_t res = static_cast<int128_t>(a) * b;
    if (res > INT64_MAX || res < INT64_MIN) return true;
    out = static_cast<int64_t>(res);
    return false;
#else
    if (a == 0 || b == 0) { out = 0; return false; }
    if (a == -1 && b == INT64_MIN) return true;
    if (b == -1 && a == INT64_MIN) return true;
    if (a > 0 && b > 0 && a > INT64_MAX / b) return true;
    if (a > 0 && b < 0 && b < INT64_MIN / a) return true;
    if (a < 0 && b > 0 && a < INT64_MIN / b) return true;
    if (a < 0 && b < 0 && a < INT64_MAX / b) return true;
    out = a * b;
    return false;
#endif
}

inline int64_t safe_div(int64_t a, int64_t b) noexcept {
    if (b == 0) return 0;
    if (a == INT64_MIN && b == -1) return INT64_MAX;
    return a / b;
}

inline int64_t safe_mod(int64_t a, int64_t b) noexcept {
    if (b == 0) return 0;
    if (a == INT64_MIN && b == -1) return 0;
    return a % b;
}

} // namespace

std::string ValueRange::to_string() const {
    if (is_empty()) return "[empty]";
    if (is_constant()) return "[" + std::to_string(min_val) + "]";
    std::ostringstream ss;
    ss << "[";
    if (min_val == INT64_MIN) ss << "-inf";
    else ss << min_val;
    ss << ", ";
    if (max_val == INT64_MAX) ss << "+inf";
    else ss << max_val;
    ss << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const ValueRange& r) {
    os << r.to_string();
    return os;
}

ValueRange ValueRange::add(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    int64_t min_res = 0;
    int64_t max_res = 0;
    if (add_overflows(a.min_val, b.min_val, min_res) ||
        add_overflows(a.max_val, b.max_val, max_res)) {
        return full();
    }
    return range(min_res, max_res);
}

ValueRange ValueRange::sub(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    int64_t min_res = 0;
    int64_t max_res = 0;
    if (sub_overflows(a.min_val, b.max_val, min_res) ||
        sub_overflows(a.max_val, b.min_val, max_res)) {
        return full();
    }
    return range(min_res, max_res);
}

ValueRange ValueRange::mul(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    int64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    if (mul_overflows(a.min_val, b.min_val, p1) ||
        mul_overflows(a.min_val, b.max_val, p2) ||
        mul_overflows(a.max_val, b.min_val, p3) ||
        mul_overflows(a.max_val, b.max_val, p4)) {
        return full();
    }
    int64_t min_v = std::min({p1, p2, p3, p4});
    int64_t max_v = std::max({p1, p2, p3, p4});
    return range(min_v, max_v);
}

ValueRange ValueRange::and_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val & b.min_val);
    }
    if (a.is_non_negative() || b.is_non_negative()) {
        int64_t upper = INT64_MAX;
        if (a.is_non_negative() && b.is_non_negative()) {
            upper = std::min(a.max_val, b.max_val);
        } else if (a.is_non_negative()) {
            upper = a.max_val;
        } else {
            upper = b.max_val;
        }
        return range(0, upper);
    }
    return full();
}

ValueRange ValueRange::or_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val | b.min_val);
    }
    if (a.is_non_negative() && b.is_non_negative()) {
        int64_t lower = std::max(a.min_val, b.min_val);
        uint64_t m = static_cast<uint64_t>(a.max_val | b.max_val);
        if (m == 0) return constant(0);
        int clz = std::countl_zero(m);
        uint64_t bound = (clz == 0) ? UINT64_MAX : ((1ULL << (64 - clz)) - 1ULL);
        int64_t upper = (bound > static_cast<uint64_t>(INT64_MAX)) ? INT64_MAX : static_cast<int64_t>(bound);
        return range(lower, upper);
    }
    return full();
}

ValueRange ValueRange::xor_(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (a.is_constant() && b.is_constant()) {
        return constant(a.min_val ^ b.min_val);
    }
    if (a.is_non_negative() && b.is_non_negative()) {
        uint64_t m = static_cast<uint64_t>(a.max_val | b.max_val);
        if (m == 0) return constant(0);
        int clz = std::countl_zero(m);
        uint64_t bound = (clz == 0) ? UINT64_MAX : ((1ULL << (64 - clz)) - 1ULL);
        int64_t upper = (bound > static_cast<uint64_t>(INT64_MAX)) ? INT64_MAX : static_cast<int64_t>(bound);
        return range(0, upper);
    }
    return full();
}

ValueRange ValueRange::shl(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val < 63) {
        int shift = static_cast<int>(b.min_val);
        int64_t factor = 1LL << shift;
        int64_t p1 = 0, p2 = 0;
        if (mul_overflows(a.min_val, factor, p1) || mul_overflows(a.max_val, factor, p2)) {
            return full();
        }
        return range(std::min(p1, p2), std::max(p1, p2));
    }
    return full();
}

ValueRange ValueRange::lshr(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val <= 64) {
        int shift = static_cast<int>(b.min_val);
        if (shift >= 64) return constant(0);
        if (a.is_non_negative()) {
            return range(a.min_val >> shift, a.max_val >> shift);
        }
    }
    return full();
}

ValueRange ValueRange::ashr(const ValueRange& a, const ValueRange& b) noexcept {
    if (a.is_empty() || b.is_empty()) return empty();
    if (b.is_constant() && b.min_val >= 0 && b.min_val < 64) {
        int shift = static_cast<int>(b.min_val);
        return range(a.min_val >> shift, a.max_val >> shift);
    }
    return full();
}

ValueRange ValueRange::zext(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    if (a.is_non_negative() && a.max_val <= static_cast<int64_t>(UINT32_MAX)) {
        return range(a.min_val, a.max_val);
    }
    return range(0, static_cast<int64_t>(UINT32_MAX));
}

ValueRange ValueRange::sext(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    int64_t min_v = std::max(static_cast<int64_t>(INT32_MIN), a.min_val);
    int64_t max_v = std::min(static_cast<int64_t>(INT32_MAX), a.max_val);
    if (min_v > max_v) return range(INT32_MIN, INT32_MAX);
    return range(min_v, max_v);
}

ValueRange ValueRange::trunc(const ValueRange& a) noexcept {
    if (a.is_empty()) return empty();
    if (a.min_val >= INT32_MIN && a.max_val <= INT32_MAX) {
        return a;
    }
    return range(INT32_MIN, INT32_MAX);
}

ValueRange ValueRange::select(const ValueRange& cond, const ValueRange& then_r, const ValueRange& else_r) noexcept {
    if (cond.is_empty() || then_r.is_empty() || else_r.is_empty()) return empty();
    if (cond.is_constant()) {
        return (cond.min_val != 0) ? then_r : else_r;
    }
    ValueRange res = then_r;
    res.union_with(else_r);
    return res;
}

} // namespace brass
