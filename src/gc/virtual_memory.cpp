// Address-space reservation for the heap (heap_internal.hpp): a heap reserves
// its whole range once and commits pieces as its spaces grow.

#include "heap_internal.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace brass::gc::detail {

void gc_fatal(const char* message) {
    std::fprintf(stderr, "brass: fatal: gc: %s\n", message);
    std::fflush(stderr);
    std::abort();
}

void* vm_reserve(size_t bytes, size_t alignment) {
    const size_t padded = bytes + alignment;
#if defined(_WIN32)
    void* raw = VirtualAlloc(nullptr, padded, MEM_RESERVE, PAGE_NOACCESS);
    if (!raw) throw std::bad_alloc();
    const uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(uintptr_t{alignment} - 1);
    // Windows cannot release part of a reservation: re-reserve exactly the
    // aligned range (retrying if another thread takes it in between).
    VirtualFree(raw, 0, MEM_RELEASE);
    for (int attempt = 0; attempt < 8; ++attempt) {
        void* exact = VirtualAlloc(reinterpret_cast<void*>(aligned), bytes, MEM_RESERVE, PAGE_NOACCESS);
        if (exact) return exact;
        raw = VirtualAlloc(nullptr, padded, MEM_RESERVE, PAGE_NOACCESS);
        if (!raw) break;
        const uintptr_t again = (reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(uintptr_t{alignment} - 1);
        VirtualFree(raw, 0, MEM_RELEASE);
        exact = VirtualAlloc(reinterpret_cast<void*>(again), bytes, MEM_RESERVE, PAGE_NOACCESS);
        if (exact) return exact;
    }
    throw std::bad_alloc();
#else
    void* raw = mmap(nullptr, padded, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) throw std::bad_alloc();
    const uintptr_t start = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (start + alignment - 1) & ~(uintptr_t{alignment} - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = start + padded;
    if (end > aligned + bytes) munmap(reinterpret_cast<void*>(aligned + bytes), end - (aligned + bytes));
    return reinterpret_cast<void*>(aligned);
#endif
}

void vm_commit(void* address, size_t bytes) {
    if (bytes == 0) return;
#if defined(_WIN32)
    if (!VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE)) throw std::bad_alloc();
#else
    if (mprotect(address, bytes, PROT_READ | PROT_WRITE) != 0) throw std::bad_alloc();
#endif
}

void vm_decommit(void* address, size_t bytes) {
    if (bytes == 0) return;
#if defined(_WIN32)
    VirtualFree(address, bytes, MEM_DECOMMIT);
#else
    // A fresh anonymous mapping over the range: its pages read as zero once
    // committed again on every POSIX system (MADV_DONTNEED does not promise
    // that on macOS).
    mmap(address, bytes, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#endif
}

void vm_release(void* address, size_t bytes) {
    if (!address) return;
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(address, 0, MEM_RELEASE);
#else
    munmap(address, bytes);
#endif
}

} // namespace brass::gc::detail
