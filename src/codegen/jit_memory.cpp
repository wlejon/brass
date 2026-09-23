// Executable memory for the JIT: W^X. A block is mapped read-write, the
// loader copies and relocates into it, and make_executable_read_only() then
// turns the code pages read-execute; nothing here maps a page writable and
// executable at once. (Apple Silicon is the exception the platform makes:
// code must live in a MAP_JIT mapping, whose write permission is switched
// per thread by pthread_jit_write_protect_np, so each thread still sees it as
// either writable or executable, never both.)
#include <brass/codegen/jit_exec.hpp>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <pthread.h>
#include <libkern/OSCacheControl.h>
#endif
// The unwinder's dynamic registration interface (libgcc, and libunwind on
// Apple platforms).
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);
#endif

namespace brass::codegen {

namespace {
std::vector<std::pair<uintptr_t, uintptr_t>>& jit_ranges() {
    static auto* ranges = new std::vector<std::pair<uintptr_t, uintptr_t>>();
    return *ranges;
}

std::mutex& jit_ranges_mutex() {
    static auto* m = new std::mutex();
    return *m;
}

void register_jit_memory_range(void* ptr, size_t size) {
    if (!ptr || size == 0) return;
    std::lock_guard<std::mutex> lock(jit_ranges_mutex());
    jit_ranges().push_back({reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(ptr) + size});
}

void unregister_jit_memory_range(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(jit_ranges_mutex());
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    auto& ranges = jit_ranges();
    ranges.erase(
        std::remove_if(ranges.begin(), ranges.end(),
                       [p](const auto& range) { return range.first == p; }),
        ranges.end());
}
} // namespace

size_t jit_system_page_size() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    long sz = sysconf(_SC_PAGESIZE);
    return (sz > 0) ? static_cast<size_t>(sz) : 4096;
#endif
}

bool is_jit_code_address(const void* addr) noexcept {
    if (!addr) return false;
    uintptr_t p = reinterpret_cast<uintptr_t>(addr);
    std::lock_guard<std::mutex> lock(jit_ranges_mutex());
    for (const auto& [start, end] : jit_ranges()) {
        if (p >= start && p < end) return true;
    }
    return false;
}

JitMemoryBlock::JitMemoryBlock(size_t size) {
    if (size == 0) return;
    size_t page_sz = jit_system_page_size();
    size_t page_aligned = (size + page_sz - 1) & ~(page_sz - 1);
#if defined(_WIN32)
    ptr_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, page_aligned, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#elif defined(__APPLE__) && defined(__aarch64__)
    ptr_ = static_cast<uint8_t*>(mmap(nullptr, page_aligned, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0));
    if (ptr_ == MAP_FAILED) {
        ptr_ = nullptr;
    } else {
        pthread_jit_write_protect_np(0);
    }
#else
    ptr_ = static_cast<uint8_t*>(mmap(nullptr, page_aligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (ptr_ == MAP_FAILED) ptr_ = nullptr;
#endif
    if (ptr_) size_ = page_aligned;
}

JitMemoryBlock::JitMemoryBlock(size_t code_size, size_t data_size) {
#if defined(__APPLE__) && defined(__aarch64__)
    // A MAP_JIT mapping is write-protected as a whole per thread once the
    // code is sealed, so data pages cannot share it; the loader puts them in
    // their own block beside this one.
    *this = JitMemoryBlock(code_size);
    (void)data_size;
#else
    size_t page_sz = jit_system_page_size();
    size_t code_pages = (code_size + page_sz - 1) & ~(page_sz - 1);
    size_t data_pages = (data_size + page_sz - 1) & ~(page_sz - 1);
    *this = JitMemoryBlock(code_pages + data_pages);
#endif
}

JitMemoryBlock::~JitMemoryBlock() {
    reset();
}

JitMemoryBlock::JitMemoryBlock(JitMemoryBlock&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_), unwind_table_(other.unwind_table_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.unwind_table_ = nullptr;
}

JitMemoryBlock& JitMemoryBlock::operator=(JitMemoryBlock&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        unwind_table_ = other.unwind_table_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.unwind_table_ = nullptr;
    }
    return *this;
}

bool JitMemoryBlock::register_unwind_info(size_t offset, uint32_t count) {
    if (!ptr_ || unwind_table_ || offset >= size_) return false;
    uint8_t* table = ptr_ + offset;
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__))
    if (count == 0 || offset + size_t{count} * sizeof(RUNTIME_FUNCTION) > size_) return false;
    if (!RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(table), count,
                             static_cast<DWORD64>(reinterpret_cast<uintptr_t>(ptr_)))) {
        return false;
    }
    unwind_table_ = table;
    return true;
#elif !defined(_WIN32)
    (void)count;
#if defined(__APPLE__)
    // Apple's libunwind takes one FDE per call: the first after the CIE.
    uint32_t cie_len = 0;
    std::memcpy(&cie_len, table, 4);
    table += 4 + cie_len;
#endif
    __register_frame(table);
    unwind_table_ = table;
    return true;
#else
    (void)table;
    (void)count;
    return false;
#endif
}

void JitMemoryBlock::unregister_unwind_info() noexcept {
    if (!unwind_table_) return;
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__))
    RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(unwind_table_));
#elif !defined(_WIN32)
    __deregister_frame(unwind_table_);
#endif
    unwind_table_ = nullptr;
}

void JitMemoryBlock::reset() {
    unregister_unwind_info();
    if (ptr_) {
        unregister_jit_memory_range(ptr_);
#if defined(_WIN32)
        VirtualFree(ptr_, 0, MEM_RELEASE);
#else
        munmap(ptr_, size_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }
}

bool JitMemoryBlock::make_executable_read_only(size_t code_size) {
    if (!ptr_) return false;
#if defined(__APPLE__) && defined(__aarch64__)
    (void)code_size;
    pthread_jit_write_protect_np(1);
    sys_dcache_flush(ptr_, size_);
    sys_icache_invalidate(ptr_, size_);
    register_jit_memory_range(ptr_, size_);
    return true;
#else
    size_t page_sz = jit_system_page_size();
    size_t protect_size = (code_size == 0) ? size_ : ((code_size + page_sz - 1) & ~(page_sz - 1));
    if (protect_size > size_) protect_size = size_;
#if defined(_WIN32)
    DWORD old_protect;
    if (!VirtualProtect(ptr_, protect_size, PAGE_EXECUTE_READ, &old_protect)) return false;
    FlushInstructionCache(GetCurrentProcess(), ptr_, protect_size);
#else
    if (mprotect(ptr_, protect_size, PROT_READ | PROT_EXEC) != 0) return false;
    __builtin___clear_cache(reinterpret_cast<char*>(ptr_), reinterpret_cast<char*>(ptr_ + protect_size));
#endif
    register_jit_memory_range(ptr_, protect_size);
    return true;
#endif
}

bool JitMemoryBlock::make_read_write() {
    if (!ptr_) return false;
    unregister_jit_memory_range(ptr_);
#if defined(_WIN32)
    DWORD old_protect;
    return VirtualProtect(ptr_, size_, PAGE_READWRITE, &old_protect) != 0;
#elif defined(__APPLE__) && defined(__aarch64__)
    pthread_jit_write_protect_np(0);
    return true;
#else
    return mprotect(ptr_, size_, PROT_READ | PROT_WRITE) == 0;
#endif
}

DataMemoryBlock::DataMemoryBlock(size_t size, void* address_hint) {
    if (size == 0) return;
    size_t page_sz = jit_system_page_size();
    size_t page_aligned = (size + page_sz - 1) & ~(page_sz - 1);
#if defined(_WIN32)
    ptr_ = static_cast<uint8_t*>(VirtualAlloc(address_hint, page_aligned, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!ptr_ && address_hint) {
        ptr_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, page_aligned, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    }
#else
    ptr_ = static_cast<uint8_t*>(mmap(address_hint, page_aligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (ptr_ == MAP_FAILED) {
        ptr_ = nullptr;
    }
#endif
    if (ptr_) size_ = page_aligned;
}

DataMemoryBlock::~DataMemoryBlock() {
    reset();
}

DataMemoryBlock::DataMemoryBlock(DataMemoryBlock&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DataMemoryBlock& DataMemoryBlock::operator=(DataMemoryBlock&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DataMemoryBlock::reset() {
    if (ptr_) {
#if defined(_WIN32)
        VirtualFree(ptr_, 0, MEM_RELEASE);
#else
        munmap(ptr_, size_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }
}

} // namespace brass::codegen
