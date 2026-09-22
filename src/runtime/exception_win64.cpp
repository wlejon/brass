#include <brass/runtime/exception.hpp>
#include <brass/object/object_writer.hpp>

namespace brass::runtime {

void emit_win64_seh_scope_table(object::Section& xdata_sec, const FunctionExceptionTable& table) {
    // Win64 SEH scope table format:
    // DWORD count;
    // for each scope:
    //   DWORD begin_offset;
    //   DWORD end_offset;
    //   DWORD landing_pad_offset;
    const auto& scopes = table.scopes();
    xdata_sec.emit32(static_cast<uint32_t>(scopes.size()));

    for (const auto& s : scopes) {
        xdata_sec.emit32(s.begin_offset);
        xdata_sec.emit32(s.end_offset);
        xdata_sec.emit32(s.landing_pad_offset);
    }
}

struct Win64DispatcherContextLayout {
    uint64_t ControlPc;
    uint64_t ImageBase;
    void* FunctionEntry;
    uint64_t EstablisherFrame;
    uint64_t TargetIp;
    void* ContextRecord;
    void* LanguageHandler;
    void* HandlerData;
};

struct Win64SehScopeLayout {
    uint32_t begin_offset;
    uint32_t end_offset;
    uint32_t landing_pad_offset;
};

extern "C" int brass_seh_personality(
    void* ExceptionRecord,
    void* EstablisherFrame,
    void* ContextRecord,
    void* DispatcherContext
) {
    (void)ExceptionRecord;
    (void)EstablisherFrame;
    (void)ContextRecord;

    if (!DispatcherContext) {
        return 1; // ExceptionContinueSearch
    }

    auto* dc = reinterpret_cast<Win64DispatcherContextLayout*>(DispatcherContext);
    uint64_t pc = dc->ControlPc;

    // 1. If HandlerData points to the Win64 scope table emitted by emit_win64_seh_scope_table
    if (dc->HandlerData) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(dc->HandlerData);
        uint32_t count = p[0];
        const auto* scopes = reinterpret_cast<const Win64SehScopeLayout*>(p + 1);

        uint64_t offset = pc;
        if (dc->ImageBase > 0 && pc >= dc->ImageBase) {
            offset = pc - dc->ImageBase;
        }

        for (uint32_t i = 0; i < count; ++i) {
            if (offset >= scopes[i].begin_offset && offset < scopes[i].end_offset) {
                // Landing pad identified!
                uint64_t target = scopes[i].landing_pad_offset;
                if (dc->ImageBase > 0) {
                    target += dc->ImageBase;
                }
                dc->TargetIp = target;
                return 0; // ExceptionContinueExecution (target identified)
            }
        }
    }

    // 2. Fallback: check global exception table registry by PC
    uintptr_t fn_start = 0;
    const FunctionExceptionTable* fn_table = nullptr;
    const ExceptionScopeEntry* scope = get_global_exception_registry().find_scope_by_pc(pc, &fn_start, &fn_table);
    if (scope) {
        dc->TargetIp = fn_start + scope->landing_pad_offset;
        return 0; // ExceptionContinueExecution
    }

    // No matching landing pad found in this frame
    return 1; // ExceptionContinueSearch
}

} // namespace brass::runtime
