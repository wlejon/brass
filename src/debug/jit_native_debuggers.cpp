// JIT code for native debuggers and profilers that take it in process:
//
//   GDB (and LLDB) JIT interface: an in-memory ELF symbol file per
//   registration, linked into __jit_debug_descriptor, announced through
//   __jit_debug_register_code, which the debugger breaks on. Unregistering
//   unlinks it the same way. (WinDbg / cdb have no such interface.)
//
//   VTune's JIT profiling API: VTune names its collector library in
//   INTEL_JIT_PROFILER64 (32); brass loads it and reports each function
//   as a method load (and its retirement as an unload), as the
//   jitprofiling static library would, without linking it.

#include "jit_sinks.hpp"
#include <atomic>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <cstdlib>
#endif

extern "C" {

jit_descriptor __jit_debug_descriptor = {1, BRASS_JIT_NOACTION, nullptr, nullptr};

// The debugger's breakpoint. It must survive as a distinct function the
// debugger can find, never folded or inlined away.
static volatile int brass_jit_debug_register_calls = 0;
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline, used))
#endif
void __jit_debug_register_code(void) {
    brass_jit_debug_register_calls = brass_jit_debug_register_calls + 1;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#endif
}

} // extern "C"

namespace brass::debug::detail {

namespace {

template <typename T>
void put(std::vector<uint8_t>& out, size_t at, T v) {
    std::memcpy(out.data() + at, &v, sizeof v);
}

uint16_t elf_machine16() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 183;
#else
    return 62;
#endif
}

struct GdbEntry {
    jit_code_entry entry{};
    std::vector<uint8_t> symfile;
};

} // namespace

std::vector<uint8_t> build_gdb_symfile(JitTier tier, const JitCodeRange* ranges, size_t count) {
    constexpr size_t kEhdr = 64, kShdr = 64, kSym = 24, kSections = 5;
    uintptr_t lo = UINTPTR_MAX, hi = 0;
    std::string strtab(1, '\0');
    std::vector<std::pair<uint32_t, const JitCodeRange*>> syms;
    for (size_t i = 0; i < count; ++i) {
        const JitCodeRange& r = ranges[i];
        if (!r.code || !r.size) continue;
        const auto a = reinterpret_cast<uintptr_t>(r.code);
        if (a < lo) lo = a;
        if (a + r.size > hi) hi = a + r.size;
        syms.emplace_back(static_cast<uint32_t>(strtab.size()), &r);
        strtab += tool_name(r.name, tier);
        strtab += '\0';
    }
    if (syms.empty()) return {};
    const std::string shstrtab = std::string("\0.text\0.symtab\0.strtab\0.shstrtab\0", 34);
    const uint32_t n_text = 1, n_symtab = 7, n_strtab = 15, n_shstrtab = 23;

    const size_t symtab_off = kEhdr;
    const size_t symtab_size = (syms.size() + 1) * kSym;
    const size_t strtab_off = symtab_off + symtab_size;
    const size_t shstrtab_off = strtab_off + strtab.size();
    const size_t shdr_off = (shstrtab_off + shstrtab.size() + 7) & ~size_t{7};
    std::vector<uint8_t> out(shdr_off + kSections * kShdr, 0);

    // ELF header: 64-bit, little-endian, an executable whose one section
    // is the JIT code, where it is.
    const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(out.data(), ident, sizeof ident);
    put<uint16_t>(out, 16, 2);                 // e_type ET_EXEC
    put<uint16_t>(out, 18, elf_machine16());   // e_machine
    put<uint32_t>(out, 20, 1);                 // e_version
    put<uint64_t>(out, 40, shdr_off);          // e_shoff
    put<uint16_t>(out, 52, kEhdr);             // e_ehsize
    put<uint16_t>(out, 54, 56);                // e_phentsize
    put<uint16_t>(out, 58, kShdr);             // e_shentsize
    put<uint16_t>(out, 60, kSections);         // e_shnum
    put<uint16_t>(out, 62, 4);                 // e_shstrndx

    // Symbols: the null one, then a global function per range.
    for (size_t i = 0; i < syms.size(); ++i) {
        const size_t at = symtab_off + (i + 1) * kSym;
        put<uint32_t>(out, at, syms[i].first);
        out[at + 4] = 0x12;                    // STB_GLOBAL | STT_FUNC
        put<uint16_t>(out, at + 6, 1);         // .text
        put<uint64_t>(out, at + 8, reinterpret_cast<uintptr_t>(syms[i].second->code));
        put<uint64_t>(out, at + 16, syms[i].second->size);
    }
    std::memcpy(out.data() + strtab_off, strtab.data(), strtab.size());
    std::memcpy(out.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

    auto shdr = [&](size_t idx, uint32_t name, uint32_t type, uint64_t flags, uint64_t addr, uint64_t off,
                    uint64_t size, uint32_t link, uint32_t info, uint64_t align, uint64_t entsize) {
        const size_t at = shdr_off + idx * kShdr;
        put<uint32_t>(out, at, name);
        put<uint32_t>(out, at + 4, type);
        put<uint64_t>(out, at + 8, flags);
        put<uint64_t>(out, at + 16, addr);
        put<uint64_t>(out, at + 24, off);
        put<uint64_t>(out, at + 32, size);
        put<uint32_t>(out, at + 40, link);
        put<uint32_t>(out, at + 44, info);
        put<uint64_t>(out, at + 48, align);
        put<uint64_t>(out, at + 56, entsize);
    };
    shdr(1, n_text, 8 /*NOBITS*/, 6 /*ALLOC|EXECINSTR*/, lo, 0, hi - lo, 0, 0, 16, 0);
    shdr(2, n_symtab, 2 /*SYMTAB*/, 0, 0, symtab_off, symtab_size, 3, 1, 8, kSym);
    shdr(3, n_strtab, 3 /*STRTAB*/, 0, 0, strtab_off, strtab.size(), 0, 0, 1, 0);
    shdr(4, n_shstrtab, 3 /*STRTAB*/, 0, 0, shstrtab_off, shstrtab.size(), 0, 0, 1, 0);
    return out;
}

void* gdb_jit_register(std::vector<uint8_t> symfile) {
    if (symfile.empty()) return nullptr;
    auto* e = new GdbEntry();
    e->symfile = std::move(symfile);
    e->entry.symfile_addr = reinterpret_cast<const char*>(e->symfile.data());
    e->entry.symfile_size = e->symfile.size();
    e->entry.prev_entry = nullptr;
    e->entry.next_entry = __jit_debug_descriptor.first_entry;
    if (e->entry.next_entry) e->entry.next_entry->prev_entry = &e->entry;
    __jit_debug_descriptor.first_entry = &e->entry;
    __jit_debug_descriptor.relevant_entry = &e->entry;
    __jit_debug_descriptor.action_flag = BRASS_JIT_REGISTER_FN;
    __jit_debug_register_code();
    return e;
}

void gdb_jit_unregister(void* entry) noexcept {
    auto* e = static_cast<GdbEntry*>(entry);
    if (!e) return;
    jit_code_entry* je = &e->entry;
    if (je->prev_entry) je->prev_entry->next_entry = je->next_entry;
    else __jit_debug_descriptor.first_entry = je->next_entry;
    if (je->next_entry) je->next_entry->prev_entry = je->prev_entry;
    __jit_debug_descriptor.relevant_entry = je;
    __jit_debug_descriptor.action_flag = BRASS_JIT_UNREGISTER_FN;
    __jit_debug_register_code();
    __jit_debug_descriptor.relevant_entry = nullptr;
    __jit_debug_descriptor.action_flag = BRASS_JIT_NOACTION;
    delete e;
}

// ---- VTune --------------------------------------------------------------

namespace {

// jitprofiling.h, the parts brass uses.
constexpr int kMethodLoadFinished = 13;   // iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED
constexpr int kMethodUnloadStart = 14;    // iJVM_EVENT_TYPE_METHOD_UNLOAD_START
struct IJitMethodLoad {
    unsigned int method_id;
    char* method_name;
    void* method_load_address;
    unsigned int method_size;
    unsigned int line_number_size;
    void* line_number_table;
    unsigned int class_id;
    char* class_file_name;
    char* source_file_name;
};

using NotifyEventFn = int (*)(int, void*);
using InitializeFn = int (*)();

struct VTune {
    NotifyEventFn notify = nullptr;
    unsigned int next_id = 1;
};

VTune* vtune() {
    static VTune* v = [] {
        auto* t = new VTune();
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
        const char* var = "INTEL_JIT_PROFILER64";
#else
        const char* var = "INTEL_JIT_PROFILER32";
#endif
#if defined(_WIN32)
        char path[1024];
        const DWORD n = GetEnvironmentVariableA(var, path, sizeof path);
        if (n == 0 || n >= sizeof path) return t;
        HMODULE lib = LoadLibraryA(path);
        if (!lib) return t;
        auto notify = reinterpret_cast<NotifyEventFn>(reinterpret_cast<void*>(GetProcAddress(lib, "NotifyEvent")));
        auto init = reinterpret_cast<InitializeFn>(reinterpret_cast<void*>(GetProcAddress(lib, "Initialize")));
#else
        const char* path = std::getenv(var);
        if (!path || !*path) return t;
        void* lib = dlopen(path, RTLD_LAZY);
        if (!lib) return t;
        auto notify = reinterpret_cast<NotifyEventFn>(dlsym(lib, "NotifyEvent"));
        auto init = reinterpret_cast<InitializeFn>(dlsym(lib, "Initialize"));
#endif
        // Initialize answers whether anything is collecting (0: nothing).
        if (notify && (!init || init() != 0)) t->notify = notify;
        return t;
    }();
    return v;
}

} // namespace

std::vector<uint32_t> vtune_register(JitTier tier, const JitCodeRange* ranges, size_t count) {
    std::vector<uint32_t> ids;
    VTune* v = vtune();
    if (!v->notify) return ids;
    for (size_t i = 0; i < count; ++i) {
        if (!ranges[i].code || !ranges[i].size) continue;
        std::string name = tool_name(ranges[i].name, tier);
        IJitMethodLoad m{};
        m.method_id = v->next_id++;
        m.method_name = name.data();
        m.method_load_address = const_cast<void*>(ranges[i].code);
        m.method_size = static_cast<unsigned int>(ranges[i].size);
        v->notify(kMethodLoadFinished, &m);
        ids.push_back(m.method_id);
    }
    return ids;
}

void vtune_unregister(const std::vector<uint32_t>& ids) noexcept {
    VTune* v = vtune();
    if (!v->notify) return;
    for (uint32_t id : ids) {
        unsigned int mid = id;
        v->notify(kMethodUnloadStart, &mid);
    }
}

} // namespace brass::debug::detail
