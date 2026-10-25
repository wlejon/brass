#include <brass/gc/fiber.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#define BRASS_FIBER_NOINLINE __declspec(noinline)
#else
#define BRASS_FIBER_NOINLINE __attribute__((noinline))
#endif

#if !defined(_WIN32)
extern "C" {
// fiber_switch.cpp. Saves the callee-saved registers on the running stack,
// stores its stack pointer through `save_sp`, and resumes the stack whose
// pointer is `load_sp` where its own switch left it.
void brass_fiber_swap(void** save_sp, void* load_sp);
// The first code a new stack runs: calls brass_fiber_main with the fiber,
// and marks the end of the stack for every unwinder.
void brass_fiber_start();
void brass_fiber_main(void* impl);
}
#endif

namespace brass::gc {

namespace {

[[noreturn]] void fiber_fatal(const char* what) {
    std::fprintf(stderr, "brass: fatal: %s\n", what);
    std::fflush(stderr);
    std::abort();
}

// One stack of a thread: a fiber's, or the thread's own. While another of
// the thread's stacks runs, this one is "switched out": it keeps the chains
// its frames made and where it switched away, and sits in the thread's list
// that collections walk.
struct StackRecord {
    StackChains chains;
    StackRecord* resumer = nullptr;  // the stack this one returns to when it suspends
    StackRecord* prev = nullptr;     // the thread's switched-out list
    StackRecord* next = nullptr;
    bool switched_out = false;
    bool active = false;             // running, or waiting on a fiber it resumed
    Fiber* fiber = nullptr;          // null: the thread's own stack
#if defined(_WIN32)
    void* handle = nullptr;          // the system fiber
    CONTEXT context;                 // captured where it switched away
    uintptr_t low = 0;               // its extent then (the TIB names only the running stack)
    uintptr_t high = 0;
#else
    void* saved_sp = nullptr;        // what brass_fiber_swap left on it
    uintptr_t top = 0;               // the stack's base (highest address); 0 for the thread's own
#endif
};

thread_local StackRecord t_own_stack;
thread_local StackRecord* t_running = nullptr;       // null: the thread's own stack
thread_local StackRecord* t_switched_out = nullptr;  // head of the list

StackRecord* running_record() noexcept { return t_running ? t_running : &t_own_stack; }

void link_switched_out(StackRecord* r) noexcept {
    r->prev = nullptr;
    r->next = t_switched_out;
    if (t_switched_out) t_switched_out->prev = r;
    t_switched_out = r;
    r->switched_out = true;
}

void unlink_switched_out(StackRecord* r) noexcept {
    if (!r->switched_out) return;
    if (r->prev) r->prev->next = r->next;
    else t_switched_out = r->next;
    if (r->next) r->next->prev = r->prev;
    r->prev = r->next = nullptr;
    r->switched_out = false;
}

// Where a switched-out stack's walk starts, as stack_walker.hpp takes it.
SwitchedStack position_of(StackRecord& r) noexcept {
    SwitchedStack s;
    s.entries = r.chains.entry_scopes;
#if defined(_WIN32)
    s.context = &r.context;
    s.low = r.low;
    s.high = r.high;
#if defined(_M_ARM64)
    s.fp = static_cast<uintptr_t>(r.context.Fp);
#endif
#elif defined(__x86_64__)
    // brass_fiber_swap's frame: mxcsr and x87 control word, r15, r14, r13,
    // r12, rbx, rbp, then the return address into its caller.
    const auto* w = static_cast<const uintptr_t*>(r.saved_sp);
    s.fp = w[6];
    s.ip = w[7];
    s.sp = reinterpret_cast<uintptr_t>(r.saved_sp) + 64;
#elif defined(__aarch64__)
    // x19..x28, then x29 (the frame pointer) and x30 (the return address),
    // the eight callee-saved vector halves and fpcr: 176 bytes.
    const auto* w = static_cast<const uintptr_t*>(r.saved_sp);
    s.fp = w[10];
    s.ip = w[11];
    s.sp = reinterpret_cast<uintptr_t>(r.saved_sp) + 176;
#else
#error "brass fibers: no stack switch for this architecture"
#endif
    return s;
}

// The machine switch from `from` (the running stack) to `to`. When `from`
// stays (it will be resumed again), what a collection needs to walk it is
// recorded first; a finished fiber's stack is never walked again.
BRASS_FIBER_NOINLINE void machine_switch(StackRecord* from, StackRecord* to, bool from_stays) {
#if defined(_WIN32)
    if (from_stays) {
        // The context of this frame, which stays as it is until the switch
        // back returns here; the walk unwinds from it to this frame's callers.
        RtlCaptureContext(&from->context);
        PNT_TIB tib = reinterpret_cast<PNT_TIB>(NtCurrentTeb());
        from->low = reinterpret_cast<uintptr_t>(tib->StackLimit);
        from->high = reinterpret_cast<uintptr_t>(tib->StackBase);
    }
    SwitchToFiber(to->handle);
#else
    (void)from_stays;
    brass_fiber_swap(&from->saved_sp, to->saved_sp);
#endif
}

// Every switch: the chains are exchanged, the switched-out list updated and
// the running stack named before the machine switch, so whatever runs on
// `to` first (a collection included) sees the thread as it now is.
void switch_stacks(StackRecord* from, StackRecord* to, bool from_stays) {
    from->chains = brass_exchange_stack_chains(to->chains);
    to->chains = StackChains{};
    if (from_stays) link_switched_out(from);
    unlink_switched_out(to);
    t_running = to == &t_own_stack ? nullptr : to;
    machine_switch(from, to, from_stays);
}

} // namespace

struct Fiber::Impl {
    StackRecord rec;
    Entry entry = nullptr;
    void* arg = nullptr;
    size_t stack_bytes_rounded = 0;
    bool started = false;
    bool finished = false;
#if !defined(_WIN32)
    void* mapping = nullptr;
    size_t mapping_bytes = 0;
#endif
};

namespace {

struct FiberPool {
    static constexpr size_t kMaxCached = 16;
    std::vector<std::unique_ptr<Fiber::Impl>> free_list;
    ~FiberPool() {
        for (auto& impl : free_list) {
            if (!impl) continue;
#if defined(_WIN32)
            if (impl->rec.handle) DeleteFiber(impl->rec.handle);
#else
            if (impl->mapping) munmap(impl->mapping, impl->mapping_bytes);
#endif
        }
    }
};

thread_local FiberPool t_fiber_pool;

// The fiber's outermost frame. The entry may not throw (an exception that
// leaves it ends the process here, which is the only frame that could catch
// it); when it returns the fiber is finished and switches back. If pooled and
// reused by a subsequent Fiber::create, it resumes at the loop back-edge.
void fiber_main(Fiber::Impl* impl) noexcept {
    for (;;) {
        impl->entry(impl->arg);
        impl->finished = true;
        StackRecord* self = &impl->rec;
        StackRecord* to = self->resumer;
        self->resumer = nullptr;
        self->active = false;
        switch_stacks(self, to, false);
        if (impl->finished) {
            fiber_fatal("a finished fiber was resumed");
        }
    }
}

size_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

#if defined(_WIN32)
void WINAPI fiber_proc(void* impl) { fiber_main(static_cast<Fiber::Impl*>(impl)); }
#endif

} // namespace

} // namespace brass::gc

#if !defined(_WIN32)
extern "C" void brass_fiber_main(void* impl) {
    brass::gc::fiber_main(static_cast<brass::gc::Fiber::Impl*>(impl));
}
#endif

namespace brass::gc {

Fiber::Fiber(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) { impl_->rec.fiber = this; }

std::unique_ptr<Fiber> Fiber::create(Entry entry, void* arg, size_t stack_bytes) {
    if (entry == nullptr) return nullptr;
    const size_t page = page_size();
    const size_t bytes = ((stack_bytes < 16 * 1024 ? 16 * 1024 : stack_bytes) + page - 1) / page * page;
    for (size_t i = t_fiber_pool.free_list.size(); i > 0; --i) {
        if (t_fiber_pool.free_list[i - 1]->stack_bytes_rounded == bytes) {
            std::unique_ptr<Impl> reused = std::move(t_fiber_pool.free_list[i - 1]);
            t_fiber_pool.free_list.erase(t_fiber_pool.free_list.begin() + static_cast<ptrdiff_t>(i - 1));
            reused->entry = entry;
            reused->arg = arg;
            reused->started = false;
            reused->finished = false;
            reused->rec.chains = StackChains{};
            reused->rec.resumer = nullptr;
            reused->rec.active = false;
            return std::unique_ptr<Fiber>(new Fiber(std::move(reused)));
        }
    }
    auto impl = std::make_unique<Impl>();
    impl->entry = entry;
    impl->arg = arg;
    impl->stack_bytes_rounded = bytes;
#if defined(_WIN32)
    // The system reserves `bytes`, commits as the stack grows and keeps a
    // guard page below what is committed.
    impl->rec.handle = CreateFiberEx(64 * 1024, bytes, FIBER_FLAG_FLOAT_SWITCH, &fiber_proc, impl.get());
    if (impl->rec.handle == nullptr) return nullptr;
#else
    const size_t total = bytes + page;  // the guard page below the stack
    void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return nullptr;
    if (mprotect(mem, page, PROT_NONE) != 0) {
        munmap(mem, total);
        return nullptr;
    }
    impl->mapping = mem;
    impl->mapping_bytes = total;
    // The frame brass_fiber_swap pops the first time it switches here: the
    // fiber in a callee-saved register, a null frame pointer, and a return
    // into brass_fiber_start, the stack pointer then 16-byte aligned.
    const uintptr_t top = (reinterpret_cast<uintptr_t>(mem) + total) & ~uintptr_t{15};
#if defined(__x86_64__)
    auto* w = reinterpret_cast<uintptr_t*>(top - 80);
    std::memset(w, 0, 80);
    const uint32_t mxcsr = 0x1F80;   // the default: all exceptions masked, round to nearest
    const uint16_t fpucw = 0x037F;
    std::memcpy(reinterpret_cast<char*>(w), &mxcsr, sizeof mxcsr);
    std::memcpy(reinterpret_cast<char*>(w) + 4, &fpucw, sizeof fpucw);
    w[4] = reinterpret_cast<uintptr_t>(impl.get());  // r12
    w[6] = 0;                                         // rbp
    w[7] = reinterpret_cast<uintptr_t>(&brass_fiber_start);
#elif defined(__aarch64__)
    auto* w = reinterpret_cast<uintptr_t*>(top - 176);
    std::memset(w, 0, 176);
    w[0] = reinterpret_cast<uintptr_t>(impl.get());  // x19
    w[10] = 0;                                        // x29
    w[11] = reinterpret_cast<uintptr_t>(&brass_fiber_start);
    uint64_t fpcr = 0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    w[20] = static_cast<uintptr_t>(fpcr);
#endif
    impl->rec.saved_sp = w;
    impl->rec.top = reinterpret_cast<uintptr_t>(mem) + total;
#endif
    return std::unique_ptr<Fiber>(new Fiber(std::move(impl)));
}

Fiber::~Fiber() {
    if (impl_->rec.active) fiber_fatal("a fiber was destroyed while running or waiting on a fiber it resumed");
    unlink_switched_out(&impl_->rec);
    if (impl_->finished && t_fiber_pool.free_list.size() < FiberPool::kMaxCached) {
        impl_->rec.fiber = nullptr;
        t_fiber_pool.free_list.push_back(std::move(impl_));
        return;
    }
#if defined(_WIN32)
    if (impl_->rec.handle) DeleteFiber(impl_->rec.handle);
#else
    if (impl_->mapping) munmap(impl_->mapping, impl_->mapping_bytes);
#endif
}

void Fiber::resume() {
    if (impl_->finished) fiber_fatal("a finished fiber was resumed");
    if (impl_->rec.active) fiber_fatal("a fiber was resumed while running or waiting on a fiber it resumed");
    StackRecord* from = running_record();
#if defined(_WIN32)
    if (from == &t_own_stack) {
        // The thread's own stack is a system fiber too, so it can be
        // switched back to; a thread a host already runs as fibers is
        // whichever of its fibers is resuming this one.
        if (!IsThreadAFiber()) {
            from->handle = ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
            if (from->handle == nullptr) fiber_fatal("the thread could not be converted to a fiber");
        } else {
            from->handle = GetCurrentFiber();
        }
    }
#endif
    from->active = true;
    impl_->rec.resumer = from;
    impl_->rec.active = true;
    impl_->started = true;
    switch_stacks(from, &impl_->rec, true);
}

void Fiber::suspend() {
    StackRecord* self = t_running;
    if (self == nullptr) fiber_fatal("Fiber::suspend called on a thread's own stack");
    StackRecord* to = self->resumer;
    self->resumer = nullptr;
    self->active = false;
    switch_stacks(self, to, true);
}

Fiber* Fiber::current() noexcept { return t_running ? t_running->fiber : nullptr; }

void Fiber::for_each_stack_span(uintptr_t sp, void (*fn)(void* ctx, const StackSpan& span), void* ctx) {
    StackSpan running;
    running.low = sp;
    if (t_running != nullptr) {
#if defined(_WIN32)
        running.high = reinterpret_cast<uintptr_t>(reinterpret_cast<PNT_TIB>(NtCurrentTeb())->StackBase);
#else
        running.high = t_running->top;
#endif
    }
    fn(ctx, running);
    for (StackRecord* r = t_switched_out; r != nullptr; r = r->next) {
        StackSpan s;
#if defined(_WIN32)
#if defined(_M_X64)
        s.low = static_cast<uintptr_t>(r->context.Rsp);
#else
        s.low = static_cast<uintptr_t>(r->context.Sp);
#endif
        s.high = r->fiber != nullptr ? r->high : 0;
        s.regs = &r->context;
        s.regs_end = &r->context + 1;
#else
        s.low = reinterpret_cast<uintptr_t>(r->saved_sp);
        s.high = r->top;
#endif
        fn(ctx, s);
    }
}

bool Fiber::started() const noexcept { return impl_->started; }
bool Fiber::finished() const noexcept { return impl_->finished; }

} // namespace brass::gc

namespace brass {

void brass_append_switched_stack_roots(std::vector<uintptr_t*>& roots) {
    if (gc::t_switched_out == nullptr) return;
    static const ModuleStackMap* const registry_only = [] {
        auto* m = new ModuleStackMap();
        m->set_indexed_by_code_registry(true);
        return m;
    }();
    const ModuleStackMap* installed = brass_get_active_stack_maps();
    const ModuleStackMap& maps = installed ? *installed : *registry_only;
    auto push = [](void** slot, void* user_data) {
        auto* vec = static_cast<std::vector<uintptr_t*>*>(user_data);
        if (slot != nullptr && *slot != nullptr) vec->push_back(reinterpret_cast<uintptr_t*>(slot));
    };
    for (gc::StackRecord* r = gc::t_switched_out; r != nullptr; r = r->next) {
        const SwitchedStack s = gc::position_of(*r);
        brass_stack_walk_switched(s, maps, push, &roots);
        brass_append_switched_native_frame_roots(r->chains, s, roots);
    }
}

} // namespace brass
