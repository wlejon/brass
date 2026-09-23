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

#include <brass/runtime/exception.hpp>
#include <brass/object/object_writer.hpp>

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

#include <stdexcept>
#include <string>

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

    // Only brass exceptions land in brass pads; everything else (C++
    // exceptions from host callbacks, hardware faults) passes through, and
    // the unwind pass needs nothing from us.
    if (er->ExceptionCode != BRASS_SEH_EXCEPTION_CODE || er->NumberParameters < 1) {
        return ExceptionContinueSearch;
    }
    if (er->ExceptionFlags & EXCEPTION_UNWIND) {
        return ExceptionContinueSearch;
    }

    uint64_t target = brass_seh_find_landing_pad(dc->ControlPc, dc->ImageBase, dc->HandlerData);
    if (!target) return ExceptionContinueSearch;

    // Unwind every frame above this one (running their termination
    // handlers) and resume at the pad in this frame with the thrown value in
    // the return register. RtlUnwindEx does not return.
    RtlUnwindEx(EstablisherFrame,
                reinterpret_cast<PVOID>(target),
                er,
                reinterpret_cast<PVOID>(static_cast<uintptr_t>(er->ExceptionInformation[0])),
                dc->ContextRecord,
                dc->HistoryTable);
    __fastfail(FAST_FAIL_INVALID_ARG);
    return ExceptionContinueSearch; // unreachable
}

// Search pass run before raising: is there a brass landing pad anywhere up
// the stack, as the OS unwinder sees it? Walks the same .pdata/.xdata the
// dispatcher will, so a hit here is a hit there.
static bool seh_pad_exists() noexcept {
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    const void* personality = reinterpret_cast<const void*>(&brass_seh_personality);
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
                                                      &handler_data, &establisher, nullptr);        if (handler && reinterpret_cast<const void*>(handler) == personality &&
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

#else

// No OS unwinder calls this off Windows; it exists so objects referencing it
// link, and it never claims a frame.
extern "C" int brass_seh_personality(void*, void*, void*, void*) {
    return 1; // ExceptionContinueSearch
}

bool brass_seh_raise(HostValue) {
    return false;
}

#endif

} // namespace brass::runtime
