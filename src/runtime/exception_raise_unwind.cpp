// brass_seh_raise / brass_seh_raise_above off Windows: the native raise of a
// brass value to a landing pad of generated code: every throw on AArch64,
// whose callee-saved d8-d15 the JIT frame walker (brass_throw_impl) does not
// restore, and on x86-64 each throw the walker cannot place, e.g. because it
// is made again from a C++ frame (the Tier-0 bridge's host entry, a coroutine
// resume entry, a deopt handler's outcome) or passes a baseline frame on its
// way.
//
// There is no OS dispatcher to raise through, so the search is the one
// exception_win64.cpp makes with RtlVirtualUnwind, made with the process's
// DWARF unwinder instead (libunwind on Apple platforms, libgcc's
// _Unwind_Backtrace elsewhere): every frame the unwinder can step through,
// generated code included (its .eh_frame is registered at load), with each
// frame's callee-saved registers restored as they were at its call. The
// first frame with a scope for its call site in the JIT registry is entered
// at its pad through brass_jump_to_landing_pad, exactly as the frame walker
// enters it. Only frames below the innermost generated-code entry
// (GeneratedCodeEntryScope) count, as on Windows: the throw leaves as a C++
// exception past it. The frames jumped over are abandoned without running
// anything, so brass_seh_raise jumps only when no frame on the way is one
// outside the registry whose FDE names an LSDA (a C++ frame with destructors,
// or an AOT function with pads). Past such a frame it throws a C++
// BrassException instead, which the C++ unwinder carries through those
// frames and brass_sysv_personality lands at the first pad; with no
// registered pad at all it returns false, and the caller's C++ throw reaches
// an AOT function's pad the same way. brass_seh_raise_above runs on behalf of
// deoptimized frames and always jumps.

#include <brass/runtime/exception.hpp>
#include <brass/gc/native_frames.hpp>

#if !defined(_WIN32) && (defined(__x86_64__) || defined(__aarch64__))
#define BRASS_UNWIND_RAISE 1
#include <cstdint>
#include <cstring>
#if defined(__APPLE__)
#include <libunwind.h>
#else
#include <unwind.h>
#endif
#endif

namespace brass::runtime {

#if defined(BRASS_UNWIND_RAISE)

namespace {

// One frame as the unwinder restores it.
struct NativeFrame {
    uintptr_t pc = 0;       // return address into the frame (not a call site)
    uintptr_t sp = 0;       // its stack pointer at that call; 0 if unknown
    uintptr_t fp = 0;
    uintptr_t fn_start = 0; // its function's start per the unwind info; 0 if unknown
    bool has_lsda = false;  // its FDE names an LSDA: the frame has unwinding of its own
    SavedRegisters regs;
};

// DWARF register numbers (libunwind's unw_regnum_t uses the same ones).
#if defined(__aarch64__)
constexpr int kFpReg = 29;
#else
constexpr int kFpReg = 6; // rbp
#endif

template <typename Gpr, typename Fpr>
void restore_callee_saved(SavedRegisters& r, Gpr gpr, Fpr fpr) {
#if defined(__aarch64__)
    r.x19 = gpr(19);
    r.x20 = gpr(20);
    r.x21 = gpr(21);
    r.x22 = gpr(22);
    r.x23 = gpr(23);
    r.x24 = gpr(24);
    r.x25 = gpr(25);
    r.x26 = gpr(26);
    r.x27 = gpr(27);
    r.x28 = gpr(28);
    r.fp = gpr(29);
    r.d8 = fpr(72);  // v8..v15: only their low 64 bits are callee-saved
    r.d9 = fpr(73);
    r.d10 = fpr(74);
    r.d11 = fpr(75);
    r.d12 = fpr(76);
    r.d13 = fpr(77);
    r.d14 = fpr(78);
    r.d15 = fpr(79);
#else
    (void)fpr;
    r.rbx = gpr(3);
    r.r12 = gpr(12);
    r.r13 = gpr(13);
    r.r14 = gpr(14);
    r.r15 = gpr(15);
#endif
}

// Calls visit(frame) for each frame above the caller's, innermost first,
// until it returns true or the unwinder can go no further.
template <typename Visit>
void walk_native_frames(Visit&& visit) {
#if defined(__APPLE__)
    unw_context_t uc;
    unw_cursor_t cur;
    if (unw_getcontext(&uc) != 0 || unw_init_local(&cur, &uc) != 0) return;
    for (int depth = 0; depth < 100000 && unw_step(&cur) > 0; ++depth) {
        auto gpr = [&cur](int reg) -> uint64_t {
            unw_word_t v = 0;
            return unw_get_reg(&cur, reg, &v) == 0 ? static_cast<uint64_t>(v) : 0;
        };
        NativeFrame f;
        unw_word_t v = 0;
        if (unw_get_reg(&cur, UNW_REG_IP, &v) != 0 || v == 0) return;
        f.pc = static_cast<uintptr_t>(v);
        if (unw_get_reg(&cur, UNW_REG_SP, &v) == 0) f.sp = static_cast<uintptr_t>(v);
        f.fp = static_cast<uintptr_t>(gpr(kFpReg));
        unw_proc_info_t info;
        if (unw_get_proc_info(&cur, &info) == 0) {
            f.fn_start = static_cast<uintptr_t>(info.start_ip);
            f.has_lsda = info.lsda != 0;
        }
        restore_callee_saved(f.regs, gpr, [&cur](int reg) -> uint64_t {
            unw_fpreg_t d = 0;
            uint64_t bits = 0;
            if (unw_get_fpreg(&cur, reg, &d) == 0) std::memcpy(&bits, &d, sizeof(bits));
            return bits;
        });
        if (visit(static_cast<const NativeFrame&>(f))) return;
    }
#else
    struct State {
        Visit* visit;
        uintptr_t callee_cfa; // the previous (callee) frame's CFA: this frame's sp
        int depth;
    } state{&visit, 0, 0};
    _Unwind_Backtrace([](_Unwind_Context* ctx, void* arg) -> _Unwind_Reason_Code {
        auto* s = static_cast<State*>(arg);
        auto gpr = [ctx](int reg) -> uint64_t { return static_cast<uint64_t>(_Unwind_GetGR(ctx, reg)); };
        NativeFrame f;
        f.pc = static_cast<uintptr_t>(_Unwind_GetIP(ctx));
        if (f.pc == 0 || ++s->depth > 100000) return _URC_END_OF_STACK;
        f.sp = s->callee_cfa;
        s->callee_cfa = static_cast<uintptr_t>(_Unwind_GetCFA(ctx));
        f.fp = static_cast<uintptr_t>(gpr(kFpReg));
        f.fn_start = reinterpret_cast<uintptr_t>(_Unwind_FindEnclosingFunction(reinterpret_cast<void*>(f.pc)));
        f.has_lsda = _Unwind_GetLanguageSpecificData(ctx) != nullptr;
        // libgcc keeps the low 64 bits of v8-v15 (the callee-saved part) as
        // 8-byte registers, readable as general ones.
        restore_callee_saved(f.regs, gpr, gpr);
        return (*s->visit)(static_cast<const NativeFrame&>(f)) ? _URC_END_OF_STACK : _URC_NO_REASON;
    }, &state);
#endif
}

// The innermost GeneratedCodeEntryScope on this stack, or UINTPTR_MAX. A
// frame's sp is at or below it for every frame the entering C++ frame
// called (the scope is one of that frame's locals, and its callees' frames
// lie below its own sp); the entering frame's callers are above it.
uintptr_t entry_boundary() noexcept {
    volatile char here = 0;
    const uintptr_t sp = reinterpret_cast<uintptr_t>(&here);
    for (const GeneratedCodeEntryScope* s = brass_innermost_entry_scope(); s; s = s->outer()) {
        if (s->address() >= sp) return s->address();
    }
    return UINTPTR_MAX;
}

// Where the throw lands in `f`: its function's pad for the call it made, as
// the frame walker computes it (brass_throw_impl).
struct PadTarget {
    void* ip = nullptr;
    void* fp = nullptr;
    void* sp = nullptr;
    SavedRegisters regs;
};

bool landing_pad_in(const NativeFrame& f, PadTarget& out) {
    uintptr_t fn_start = 0;
    const FunctionExceptionTable* table = nullptr;
    const ExceptionScopeEntry* scope = get_global_exception_registry().find_scope_by_pc(f.pc, &fn_start, &table);
    if (!scope || !table) return false;
    out.ip = reinterpret_cast<void*>(fn_start + scope->landing_pad_offset);
    out.fp = reinterpret_cast<void*>(f.fp);
#if defined(__aarch64__)
    out.sp = reinterpret_cast<void*>(f.fp);
#else
    out.sp = reinterpret_cast<void*>(f.fp - table->frame_size());
#endif
    out.regs = f.regs;
    return true;
}

// Whether the raise must stop short of `f` and leave the throw to the C++
// unwinder: `f` is not registered generated code and its FDE names an LSDA,
// so it has destructors to run (a C++ frame) or pads the registry does not
// hold (an AOT object's), and brass_sysv_personality lands the throw there.
bool needs_cxx_unwind(const NativeFrame& f) {
    return f.has_lsda && get_global_exception_registry().find_function_by_pc(f.pc) == nullptr;
}

// Whether `f` is a frame of the function starting at `entry`, by its unwind
// info or by the JIT registry's range for it.
bool frame_of(const NativeFrame& f, const void* entry) {
    const auto e = reinterpret_cast<uintptr_t>(entry);
    if (f.fn_start == e) return true;
    uintptr_t fn_start = 0;
    (void)get_global_exception_registry().find_scope_by_pc(f.pc, &fn_start, nullptr);
    return fn_start == e;
}

} // namespace

bool brass_seh_raise(HostValue val) {
    const uintptr_t boundary = entry_boundary();
    PadTarget target;
    bool found = false;
    bool blocked = false;  // a frame on the way needs the C++ unwinder
    walk_native_frames([&](const NativeFrame& f) {
        if (f.sp != 0 && f.sp >= boundary) return true;
        found = landing_pad_in(f, target);
        if (!found && needs_cxx_unwind(f)) blocked = true;
        return found;
    });
    if (!found) return false;
    // The pad is there, past frames that must unwind: the C++ unwinder takes
    // the throw to it (brass_sysv_personality lands it).
    if (blocked) throw BrassException(val);
    brass_set_current_exception(val);
    brass_jump_to_landing_pad(target.ip, target.fp, target.sp, val, target.regs);
}

bool brass_seh_raise_above(HostValue val, const void* deopted_entry, uintptr_t stack_limit) {
    if (!deopted_entry) return false;
    const uintptr_t boundary = entry_boundary();
    if (boundary < stack_limit) stack_limit = boundary;
    // As on Windows: the frames up to the deoptimized one are C++ frames,
    // and its call into the deopt entry lies in no invoke scope (a pad there
    // belongs to the code the Tier-0 continuation ran: nothing is raised).
    PadTarget target;
    bool found = false;
    bool past_deopted = false;
    walk_native_frames([&](const NativeFrame& f) {
        if (!past_deopted) {
            if (!frame_of(f, deopted_entry)) return false;
            PadTarget own;
            if (landing_pad_in(f, own)) return true;
            past_deopted = true;
            return false;
        }
        if (f.sp != 0 && f.sp >= stack_limit) return true;
        found = landing_pad_in(f, target);
        return found;
    });
    if (!found) return false;
    brass_set_current_exception(val);
    brass_jump_to_landing_pad(target.ip, target.fp, target.sp, val, target.regs);
}

#elif !(defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__)))

// No unwinder walk here (and no Win64 SEH): the throw stays a C++ exception.
bool brass_seh_raise(HostValue) {
    return false;
}

bool brass_seh_raise_above(HostValue, const void*, uintptr_t) {
    return false;
}

#endif

} // namespace brass::runtime
