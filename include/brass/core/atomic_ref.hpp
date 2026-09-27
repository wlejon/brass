#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace brass {

#if defined(__cpp_lib_atomic_ref)
template <typename T>
using AtomicRef = std::atomic_ref<T>;
#elif defined(__GNUC__) || defined(__clang__)
template <typename T>
class AtomicRef {
public:
    static_assert(std::is_trivially_copyable_v<T>, "AtomicRef requires trivially copyable type");

    explicit AtomicRef(T& obj) noexcept : ptr_(&obj) {}

    T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        switch (order) {
            case std::memory_order_relaxed: return __atomic_load_n(ptr_, __ATOMIC_RELAXED);
            case std::memory_order_acquire: return __atomic_load_n(ptr_, __ATOMIC_ACQUIRE);
            case std::memory_order_consume: return __atomic_load_n(ptr_, __ATOMIC_CONSUME);
            case std::memory_order_seq_cst: default: return __atomic_load_n(ptr_, __ATOMIC_SEQ_CST);
        }
    }

    void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
        switch (order) {
            case std::memory_order_relaxed: __atomic_store_n(ptr_, desired, __ATOMIC_RELAXED); break;
            case std::memory_order_release: __atomic_store_n(ptr_, desired, __ATOMIC_RELEASE); break;
            case std::memory_order_seq_cst: default: __atomic_store_n(ptr_, desired, __ATOMIC_SEQ_CST); break;
        }
    }

    T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        switch (order) {
            case std::memory_order_relaxed: return __atomic_fetch_add(ptr_, arg, __ATOMIC_RELAXED);
            case std::memory_order_release: return __atomic_fetch_add(ptr_, arg, __ATOMIC_RELEASE);
            case std::memory_order_acquire: return __atomic_fetch_add(ptr_, arg, __ATOMIC_ACQUIRE);
            case std::memory_order_acq_rel: return __atomic_fetch_add(ptr_, arg, __ATOMIC_ACQ_REL);
            case std::memory_order_seq_cst: default: return __atomic_fetch_add(ptr_, arg, __ATOMIC_SEQ_CST);
        }
    }

    T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        switch (order) {
            case std::memory_order_relaxed: return __atomic_fetch_or(ptr_, arg, __ATOMIC_RELAXED);
            case std::memory_order_release: return __atomic_fetch_or(ptr_, arg, __ATOMIC_RELEASE);
            case std::memory_order_acquire: return __atomic_fetch_or(ptr_, arg, __ATOMIC_ACQUIRE);
            case std::memory_order_acq_rel: return __atomic_fetch_or(ptr_, arg, __ATOMIC_ACQ_REL);
            case std::memory_order_seq_cst: default: return __atomic_fetch_or(ptr_, arg, __ATOMIC_SEQ_CST);
        }
    }

private:
    T* ptr_;
};
#else
template <typename T>
using AtomicRef = std::atomic_ref<T>;
#endif

} // namespace brass
