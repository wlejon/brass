// Win64 SEH support for brass exceptions in code the OS unwinder describes:
// AOT COFF objects (and brass-built DLLs) whose .pdata/.xdata name
// brass_seh_personality as the language handler.
//
// A throw that the JIT frame walker cannot place (its first frame is not
// registered JIT code) is raised as a Win64 SEH exception carrying the brass
// value. The OS dispatcher then calls brass_seh_personality for each frame
// with a handler; on a scope hit the personality unwinds to the landing pad
// with RtlUnwindEx, which restores the frame's nonvolatile registers and
// delivers the value in RAX (X0 on ARM64), exactly as the JIT walker's
// brass_jump_to_landing_pad does.
//
// The personality lands a C++ `throw BrassException(v)` the same way: a
// helper or host function that generated code called raises into the
// generated caller's pad by throwing one, and the C++ frames between unwind
// as they do for any C++ exception.

#include <brass/runtime/exception.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/gc/native_frames.hpp>

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__))
#define BRASS_WIN64_SEH 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <cstring>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace brass::runtime {

void emit_win64_seh_scope_table(object::Section& xdata_sec, const FunctionExceptionTable& table,
                                std::string_view fn_symbol) {
    // Win64 SEH scope table format:
    // DWORD count;
    // for each scope: DWORD begin, end, landing_pad
    // Each address is an image-relative RVA: the function-relative offset
    // is stored as the addend of an ADDR32NB relocation against the
    // function's symbol, so the linker (or the JIT's loader) makes it
    // comparable with DispatcherContext->ControlPc - ImageBase.
    if (fn_symbol.empty()) {
        throw std::runtime_error("Win64 SEH scope table: no function symbol for '" +
                                 std::string(table.function_name()) + "'");
    }
    const auto& scopes = table.scopes();
    xdata_sec.emit32(static_cast<uint32_t>(scopes.size()));

    auto emit_rva = [&](uint32_t fn_offset) {
        object::ObjectRelocation r;
        r.offset = xdata_sec.data.size();
        r.kind = object::RelocKind::Addr32NB;
        r.symbol_name = std::string(fn_symbol);
        r.addend = static_cast<int64_t>(fn_offset);
        xdata_sec.relocations.push_back(std::move(r));
        xdata_sec.emit32(fn_offset);
    };
    for (const auto& s : scopes) {
        emit_rva(s.begin_offset);
        emit_rva(s.end_offset);
        emit_rva(s.landing_pad_offset);
    }
}

namespace {

struct Win64SehScopeLayout {
    uint32_t begin_rva;
    uint32_t end_rva;
    uint32_t landing_pad_rva;
};

} // namespace

uint64_t brass_seh_find_landing_pad(uint64_t control_pc, uint64_t image_base, const void* handler_data) noexcept {
    if (control_pc == 0) return 0;
    // ControlPc is a return address for every frame a brass throw passes
    // through; the call itself is the byte before it (as find_scope does).
    uint64_t call_pc = control_pc - 1;

    if (handler_data) {
        const uint32_t* p = static_cast<const uint32_t*>(handler_data);
        uint32_t count = p[0];
        const auto* scopes = reinterpret_cast<const Win64SehScopeLayout*>(p + 1);
        if (call_pc < image_base) return 0;
        uint64_t rva = call_pc - image_base;
        for (uint32_t i = 0; i < count; ++i) {
            if (rva >= scopes[i].begin_rva && rva < scopes[i].end_rva) {
                return image_base + scopes[i].landing_pad_rva;
            }
        }
        return 0;
    }

    // No scope table in the unwind info: JIT code registered by address.
    uintptr_t fn_start = 0;
    const FunctionExceptionTable* fn_table = nullptr;
    const ExceptionScopeEntry* scope =
        get_global_exception_registry().find_scope_by_pc(static_cast<uintptr_t>(control_pc), &fn_start, &fn_table);
    if (scope) {
        return fn_start + scope->landing_pad_offset;
    }
    return 0;
}

#if defined(BRASS_WIN64_SEH)

#if defined(_MSC_VER)

namespace {

// MSVC's C++ exception records (the ABI clang-cl shares): the exception code,
// the magic numbers of ExceptionInformation[0], and the throw descriptors,
// whose pointers are image-relative on 64-bit targets
// (ExceptionInformation[3] is the image base).
constexpr DWORD kMsvcCxxExceptionCode = 0xE06D7363u;  // 'msc' | 0xE0000000

struct MsvcThrowInfo {
    uint32_t attributes;
    int32_t unwind;
    int32_t forward_compat;
    int32_t catchable_type_array;
};

struct MsvcCatchableType {
    uint32_t properties;
    int32_t type_descriptor;
    int32_t mdisp;  // the offset of this base in the thrown object
    int32_t pdisp;
    int32_t vdisp;
    int32_t size_or_offset;
    int32_t copy_function;
};

struct MsvcTypeDescriptor {
    const void* vftable;
    void* spare;
    char name[1];  // decorated, NUL-terminated: type_info::raw_name()
};

// The value of a C++ `throw BrassException(v)`, when `er` is one. The type is
// matched by its decorated name, not by the throw descriptor's address: every
// image that throws one (brass itself, a host, a front end's runtime DLL) has
// its own descriptors.
bool cxx_brass_exception_bits(const EXCEPTION_RECORD* er, uint64_t& bits) noexcept {
    if (er->ExceptionCode != kMsvcCxxExceptionCode || er->NumberParameters < 4) return false;
    const ULONG_PTR magic = er->ExceptionInformation[0];
    if (magic != 0x19930520 && magic != 0x19930521 && magic != 0x19930522) return false;
    const auto* object = reinterpret_cast<const char*>(er->ExceptionInformation[1]);
    const auto* throw_info = reinterpret_cast<const MsvcThrowInfo*>(er->ExceptionInformation[2]);
    const auto image = static_cast<uintptr_t>(er->ExceptionInformation[3]);
    if (!object || !throw_info || !image || throw_info->catchable_type_array == 0) return false;
    const auto* array = reinterpret_cast<const int32_t*>(image + static_cast<uint32_t>(throw_info->catchable_type_array));
    const char* wanted = typeid(BrassException).raw_name();
    for (int32_t i = 0; i < array[0]; ++i) {
        const auto* ct = reinterpret_cast<const MsvcCatchableType*>(image + static_cast<uint32_t>(array[1 + i]));
        const auto* td = reinterpret_cast<const MsvcTypeDescriptor*>(image + static_cast<uint32_t>(ct->type_descriptor));
        if (std::strcmp(td->name, wanted) != 0) continue;
        const auto* thrown = reinterpret_cast<const BrassException*>(object + ct->mdisp);
        bits = thrown->value().raw();
        return true;
    }
    return false;
}

} // namespace

#endif

extern "C" int brass_seh_personality(
    void* ExceptionRecord,
    void* EstablisherFrame,
    void* ContextRecord,
    void* DispatcherContext
) {
    (void)ContextRecord;
    auto* er = static_cast<EXCEPTION_RECORD*>(ExceptionRecord);
    auto* dc = static_cast<DISPATCHER_CONTEXT*>(DispatcherContext);
    if (!er || !dc) return ExceptionContinueSearch;
    if (er->ExceptionFlags & EXCEPTION_UNWIND) {
        return ExceptionContinueSearch;
    }

    // Brass pads catch brass values, raised either way: natively (a brass
    // SEH exception) or as a C++ BrassException thrown by compiled code the
    // generated code called. The C++ one unwinds the compiled frames it
    // passes as any C++ exception does: RtlUnwindEx below runs their
    // destructors. Everything else (other C++ exceptions, hardware faults)
    // passes through, and the unwind pass needs nothing from us.
    uint64_t bits = 0;
    if (er->ExceptionCode == BRASS_SEH_EXCEPTION_CODE && er->NumberParameters >= 1) {
        bits = static_cast<uint64_t>(er->ExceptionInformation[0]);
    }
#if defined(_MSC_VER)
    else if (!cxx_brass_exception_bits(er, bits)) {
        return ExceptionContinueSearch;
    }
#else
    else {
        return ExceptionContinueSearch;
    }
#endif

    uint64_t target = brass_seh_find_landing_pad(dc->ControlPc, dc->ImageBase, dc->HandlerData);
    if (!target) return ExceptionContinueSearch;

    // Unwind every frame above this one (running their termination
    // handlers) and resume at the pad in this frame with the thrown value in
    // the return register. RtlUnwindEx does not return.
    brass_set_current_exception(HostValue::from_raw(bits));
    RtlUnwindEx(EstablisherFrame,
                reinterpret_cast<PVOID>(target),
                er,
                reinterpret_cast<PVOID>(static_cast<uintptr_t>(bits)),
                dc->ContextRecord,
                dc->HistoryTable);
    __fastfail(FAST_FAIL_INVALID_ARG);
    return ExceptionContinueSearch; // unreachable
}

// Whether a frame's language handler is brass_seh_personality: directly, or
// (x64 JIT code beyond 4 GB of it) through the JIT image's jump thunk
// `jmp qword ptr [rip+0]; <address>` (jit_exec.cpp).
static bool is_brass_handler(PEXCEPTION_ROUTINE handler) noexcept {
    const void* personality = reinterpret_cast<const void*>(&brass_seh_personality);
    if (!handler) return false;
    if (reinterpret_cast<const void*>(handler) == personality) return true;
#if defined(_M_X64) || defined(__x86_64__)
    const auto* t = reinterpret_cast<const uint8_t*>(handler);
    if (t[0] == 0xFF && t[1] == 0x25 && t[2] == 0 && t[3] == 0 && t[4] == 0 && t[5] == 0) {
        uint64_t target = 0;
        std::memcpy(&target, t + 6, sizeof(target));
        return target == reinterpret_cast<uint64_t>(personality);
    }
#endif
    return false;
}

// The innermost point where generated code was entered from C++ (the
// innermost GeneratedCodeEntryScope on this stack, native_frames.hpp), or
// UINTPTR_MAX when there is none. A pad in a frame at or above it must never
// be raised to: the C++ frames between (Tier 0, host callbacks) are /EHs, so
// an SEH exception would pass their catch blocks and destructors by. A throw
// with no pad below it leaves as a C++ exception instead, which unwinds the
// native frames below it (.pdata) into the C++ frame that entered them.
static uintptr_t entry_boundary() noexcept {
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    volatile char here = 0;
    const uintptr_t sp = reinterpret_cast<uintptr_t>(&here);
    for (const GeneratedCodeEntryScope* s = brass_innermost_entry_scope(); s; s = s->outer()) {
        const uintptr_t a = s->address();
        if (a >= sp && a >= low && a < high) {
            // Callers stop at the first frame whose establisher is >= the
            // returned value. On x64 a frame's establisher is its own stack
            // pointer after the prolog, below the scope for every frame the
            // entering C++ frame called. On ARM64 it is the stack pointer at
            // the frame's entry, i.e. the caller's SP at the call: for the
            // generated frame the entering C++ frame called directly, that is
            // the C++ frame's SP, which may equal the scope's address when
            // the scope is its lowest local. Only frames whose establisher
            // lies strictly above the scope (the entering frame and its
            // callers) are past the boundary there.
#if defined(_M_ARM64) || defined(__aarch64__)
            return a + 1;
#else
            return a;
#endif
        }
    }
    return UINTPTR_MAX;
}

// Search pass run before raising: is there a brass landing pad up the stack
// below the innermost generated-code entry, as the OS unwinder sees it?
// Walks the same .pdata/.xdata the dispatcher will, so a hit here is a hit
// there (the dispatcher stops at the first pad, which is at or below it).
static bool seh_pad_exists() noexcept {
    const uintptr_t boundary = entry_boundary();
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    for (int depth = 0; depth < 100000; ++depth) {
#if defined(_M_ARM64) || defined(__aarch64__)
        DWORD64 pc = ctx.Pc;
#else
        DWORD64 pc = ctx.Rip;
#endif
        if (pc == 0) return false;
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(pc, &image_base, nullptr);
        if (!fe) return false; // leaf or undescribed code: the dispatcher stops too
        PVOID handler_data = nullptr;
        DWORD64 establisher = 0;
        PEXCEPTION_ROUTINE handler = RtlVirtualUnwind(UNW_FLAG_EHANDLER, image_base, pc, fe, &ctx,
                                                      &handler_data, &establisher, nullptr);
        if (establisher >= boundary) return false;
        if (is_brass_handler(handler) &&
            brass_seh_find_landing_pad(pc, image_base, handler_data) != 0) {
            return true;
        }
    }
    return false;
}

bool brass_seh_raise(HostValue val) {
    if (!seh_pad_exists()) return false;
    ULONG_PTR info[1] = { static_cast<ULONG_PTR>(val.raw()) };
    RaiseException(BRASS_SEH_EXCEPTION_CODE, EXCEPTION_NONCONTINUABLE, 1, info);
    __fastfail(FAST_FAIL_INVALID_ARG);
    return false; // unreachable
}

bool brass_seh_raise_above(HostValue val, const void* deopted_entry, uintptr_t stack_limit) {
    if (!deopted_entry) return false;
    // Never past the innermost generated-code entry (entry_boundary).
    const uintptr_t boundary = entry_boundary();
    if (boundary < stack_limit) stack_limit = boundary;
    // The walk the dispatcher will make: the frames up to the deoptimized
    // one are C++ frames (no brass pads) and the deoptimized frame's call
    // into the deopt entry lies in no invoke scope, so the first pad found
    // above it is the one the dispatcher lands in.
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    bool past_deopted = false;
    for (int depth = 0; depth < 100000; ++depth) {
#if defined(_M_ARM64) || defined(__aarch64__)
        DWORD64 pc = ctx.Pc;
#else
        DWORD64 pc = ctx.Rip;
#endif
        if (pc == 0) return false;
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fe = RtlLookupFunctionEntry(pc, &image_base, nullptr);
        if (!fe) return false;
        // The function's start: a guard exit's pc lies in a chained entry
        // (coff_unwind.cpp), which names the function's primary entry.
        DWORD64 fn_begin = image_base + fe->BeginAddress;
#if defined(_M_X64) || defined(__x86_64__)
        for (const RUNTIME_FUNCTION* e = fe;;) {
            const uint8_t* ui = reinterpret_cast<const uint8_t*>(image_base + e->UnwindData);
            if (((ui[0] >> 3) & UNW_FLAG_CHAININFO) == 0) {
                fn_begin = image_base + e->BeginAddress;
                break;
            }
            const size_t slots = (static_cast<size_t>(ui[2]) + 1) & ~size_t(1);
            e = reinterpret_cast<const RUNTIME_FUNCTION*>(ui + 4 + 2 * slots);
        }
#endif
        const bool is_deopted = !past_deopted &&
            fn_begin == static_cast<DWORD64>(reinterpret_cast<uintptr_t>(deopted_entry));
        PVOID handler_data = nullptr;
        DWORD64 establisher = 0;
        PEXCEPTION_ROUTINE handler = RtlVirtualUnwind(UNW_FLAG_EHANDLER, image_base, pc, fe, &ctx,
                                                      &handler_data, &establisher, nullptr);
        if (is_deopted) {
            // Its own pads belong to the code the Tier-0 continuation ran.
            if (is_brass_handler(handler) &&
                brass_seh_find_landing_pad(pc, image_base, handler_data) != 0) {
                return false;
            }
            past_deopted = true;
            continue;
        }
        if (!past_deopted) continue;
        if (establisher >= stack_limit) return false;
        if (is_brass_handler(handler) &&
            brass_seh_find_landing_pad(pc, image_base, handler_data) != 0) {
            ULONG_PTR info[1] = { static_cast<ULONG_PTR>(val.raw()) };
            RaiseException(BRASS_SEH_EXCEPTION_CODE, EXCEPTION_NONCONTINUABLE, 1, info);
            __fastfail(FAST_FAIL_INVALID_ARG);
        }
    }
    return false;
}

#else

// No OS unwinder calls this off Windows; it exists so objects referencing it
// link, and it never claims a frame. brass_seh_raise and
// brass_seh_raise_above are in exception_raise_unwind.cpp there.
extern "C" int brass_seh_personality(void*, void*, void*, void*) {
    return 1; // ExceptionContinueSearch
}

#endif

} // namespace brass::runtime
