// brass_sysv_personality: the personality routine the .eh_frame CIE of
// generated code names off Windows (ElfCfiBuilder, and the baseline tier's
// own CIE). It is the SysV counterpart of brass_seh_personality.
//
// Generated code's landing pads catch brass values. A helper or host
// function called from generated code raises into its caller's pads by
// throwing a C++ `BrassException(v)`; the Itanium unwinder carries it up
// through the C++ frames in between (running their destructors) and asks
// this routine at each generated frame whether the call site has a pad. The
// pad comes from the frame's LSDA (emit_sysv_lsda: gcc's layout with a udata4
// call-site table and no type table) when the FDE names one, and otherwise
// from the JIT registry, which holds every in-process function's table. The
// frame is entered at the pad with the value in the return register (RAX /
// X0) and its callee-saved registers restored by the unwinder, exactly as
// brass_jump_to_landing_pad enters one. Every other exception passes through:
// a generated frame has nothing of its own to clean up.

#include <brass/runtime/exception.hpp>

#if !defined(_WIN32) && (defined(__x86_64__) || defined(__aarch64__))
#include <cstdint>
#include <cstring>
#include <typeinfo>
#include <unwind.h>

extern "C" {
void* __cxa_begin_catch(void*) noexcept;
void __cxa_end_catch();
}

namespace brass::runtime {

namespace {

// Itanium C++ ABI exception classes: "GNUCC++\0" (libstdc++) and
// "CLNGC++\0" (libc++abi); a last byte of 1 marks a dependent exception
// (std::rethrow_exception), whose header points at the primary one's object.
constexpr uint64_t kGnuCxx = 0x474E5543432B2B00ull;
constexpr uint64_t kClangCxx = 0x434C4E47432B2B00ull;

// The Itanium C++ ABI (2.2.1) places the unwinder's _Unwind_Exception last
// in the __cxa_exception header, directly before the thrown object, and the
// header's first field (exceptionType; primaryException in a dependent
// header) ten pointer-sized words before it: exceptionDestructor,
// unexpectedHandler, terminateHandler, nextException, the two int counters,
// actionRecord, languageSpecificData, catchTemp and adjustedPtr lie between.
// libstdc++ and libc++abi agree on it.
constexpr size_t kHeaderFirstFieldOffset = 10 * sizeof(void*);

const void* header_first_field(const _Unwind_Exception* ue) {
    const auto* p = reinterpret_cast<const char*>(ue) - kHeaderFirstFieldOffset;
    const void* v = nullptr;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// The value a C++ BrassException carries, when `ue` is one.
bool cxx_brass_exception_bits(uint64_t cls, const _Unwind_Exception* ue, uint64_t& bits) {
    const uint64_t kind = cls & ~uint64_t{0xFF};
    const uint64_t last = cls & 0xFF;
    if ((kind != kGnuCxx && kind != kClangCxx) || last > 1) return false;
    const _Unwind_Exception* primary = ue;
    if (last == 1) {
        // A dependent header's first field is the primary thrown object,
        // which follows the primary header's _Unwind_Exception.
        const auto* object = static_cast<const _Unwind_Exception*>(header_first_field(ue));
        if (!object) return false;
        primary = object - 1;
    }
    const auto* type = static_cast<const std::type_info*>(header_first_field(primary));
    if (!type || *type != typeid(BrassException)) return false;
    const auto* thrown = reinterpret_cast<const BrassException*>(primary + 1);
    bits = thrown->value().raw();
    return true;
}

uint32_t read_u32(const uint8_t*& p) {
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}

uint64_t read_uleb(const uint8_t*& p) {
    uint64_t v = 0;
    unsigned shift = 0;
    uint8_t b = 0;
    do {
        b = *p++;
        v |= static_cast<uint64_t>(b & 0x7F) << shift;
        shift += 7;
    } while (b & 0x80);
    return v;
}

// The pad for the call at function offset `call_off`, from a brass LSDA;
// 0 when the call lies in no scope.
uintptr_t pad_from_lsda(const uint8_t* lsda, uintptr_t fn_start, uintptr_t call_off) {
    constexpr uint8_t kOmit = 0xFF, kUdata4 = 0x03;
    const uint8_t* p = lsda;
    if (*p++ != kOmit) return 0;        // LPStart: always the function start
    if (*p++ != kOmit) read_uleb(p);    // TType: unused by brass pads
    if (*p++ != kUdata4) return 0;
    const uint64_t table_len = read_uleb(p);
    const uint8_t* end = p + table_len;
    while (p < end) {
        const uint32_t start = read_u32(p);
        const uint32_t len = read_u32(p);
        const uint32_t lp = read_u32(p);
        read_uleb(p);                   // action
        if (call_off >= start && call_off < uintptr_t{start} + len) {
            return lp ? fn_start + lp : 0;
        }
    }
    return 0;
}

uintptr_t landing_pad_for(_Unwind_Context* ctx) {
    int before = 0;
    const uintptr_t ip = _Unwind_GetIPInfo(ctx, &before);
    if (ip == 0) return 0;
    const uintptr_t call_ip = before ? ip : ip - 1;
    if (const auto* lsda = static_cast<const uint8_t*>(_Unwind_GetLanguageSpecificData(ctx))) {
        const uintptr_t fn_start = _Unwind_GetRegionStart(ctx);
        return call_ip >= fn_start ? pad_from_lsda(lsda, fn_start, call_ip - fn_start) : 0;
    }
    uintptr_t fn_start = 0;
    const ExceptionScopeEntry* scope =
        get_global_exception_registry().find_scope_by_pc(call_ip + 1, &fn_start, nullptr);
    return scope ? fn_start + scope->landing_pad_offset : 0;
}

constexpr int kReturnReg = 0;   // DWARF register 0: rax on x86-64, x0 on AArch64

} // namespace

extern "C" int brass_default_sysv_personality(int version, int actions, uint64_t exception_class,
                                      void* exception_object, void* context) {
    if (version != 1) return _URC_FATAL_PHASE1_ERROR;
    auto* ue = static_cast<_Unwind_Exception*>(exception_object);
    auto* ctx = static_cast<_Unwind_Context*>(context);
    if (!ue || !ctx || (actions & _UA_FORCE_UNWIND)) return _URC_CONTINUE_UNWIND;

    uint64_t bits = 0;
    if (!cxx_brass_exception_bits(exception_class, ue, bits)) return _URC_CONTINUE_UNWIND;
    const uintptr_t pad = landing_pad_for(ctx);
    if (!pad) return _URC_CONTINUE_UNWIND;
    if (actions & _UA_SEARCH_PHASE) return _URC_HANDLER_FOUND;
    if (!(actions & _UA_HANDLER_FRAME)) return _URC_CONTINUE_UNWIND;

    // This frame catches it: the C++ exception is finished with here (the
    // pad holds its value), so it is caught and released as a C++ handler
    // would, which also keeps std::uncaught_exceptions() right.
    __cxa_begin_catch(ue);
    __cxa_end_catch();
    brass_set_current_exception(HostValue::from_raw(bits));
    _Unwind_SetGR(ctx, kReturnReg, static_cast<_Unwind_Word>(bits));
    _Unwind_SetIP(ctx, pad);
    return _URC_INSTALL_CONTEXT;
}

} // namespace brass::runtime

#else

namespace brass::runtime {

// No Itanium unwinder here: nothing names this personality, and it lands
// nothing (8 is _URC_CONTINUE_UNWIND).
extern "C" int brass_default_sysv_personality(int, int, uint64_t, void*, void*) {
    return 8;
}

} // namespace brass::runtime

#endif
