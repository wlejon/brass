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

#if defined(_WIN32)
extern "C" int brass_seh_personality(
    void* ExceptionRecord,
    void* EstablisherFrame,
    void* ContextRecord,
    void* DispatcherContext
) {
    (void)ExceptionRecord;
    (void)EstablisherFrame;
    (void)ContextRecord;
    (void)DispatcherContext;

    // Standard Win64 SEH personality routine disposition.
    // 1 = ExceptionContinueSearch
    return 1;
}
#endif

} // namespace brass::runtime
