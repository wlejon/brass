// Native frame stepping off Windows (native_unwind.hpp). On Linux and the
// other ELF hosts, each step reads the frame's FDE from the unwinder
// (_Unwind_Find_FDE: the registered .eh_frame of generated code first, then
// every loaded image's) and runs its call-frame program up to the frame's pc,
// following the CFA, the frame-pointer register and the return address; the
// callee-saved registers the walk does not need are not tracked. On Apple
// platforms libunwind steps a cursor set to the frame, which reads compact
// unwind entries as well as DWARF.

#include <brass/gc/native_unwind.hpp>

#if defined(BRASS_NATIVE_UNWIND)

#include <cstring>
#include <dlfcn.h>
#include <unwind.h>
#if defined(__APPLE__)
#include <libunwind.h>
#endif

namespace brass {

namespace {

#if defined(__aarch64__)
constexpr unsigned kSpReg = 31;
constexpr unsigned kFpReg = 29;
#else
constexpr unsigned kSpReg = 7;   // rsp
constexpr unsigned kFpReg = 6;   // rbp
#endif

uintptr_t load_word(uintptr_t addr) noexcept {
    uintptr_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

// Code no unwind information describes keeps a frame record ([fp] the
// caller's frame pointer, [fp + 8] the return address), as generated code
// and brass's stubs do.
bool step_by_frame_record(NativeUnwindFrame& f) noexcept {
    if (f.fp == 0 || (f.fp % 8) != 0 || f.fp < f.sp) return false;
    const uintptr_t ra = load_word(f.fp + 8);
    if (ra == 0) return false;
    const uintptr_t next_fp = load_word(f.fp);
    f.sp = f.fp + 16;
    f.fp = next_fp;
    f.ip = ra;
    return true;
}

#if !defined(__APPLE__)

} // namespace

namespace detail {
const void* find_fde(uintptr_t pc, uintptr_t& function_start) noexcept;  // native_unwind_fde.cpp
} // namespace detail

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    bool need(size_t n) noexcept {
        if (!ok || static_cast<size_t>(end - p) < n) ok = false;
        return ok;
    }
    uint8_t u8() noexcept { return need(1) ? *p++ : 0; }
    uint64_t fixed(size_t n) noexcept {
        if (!need(n)) return 0;
        uint64_t v = 0;
        std::memcpy(&v, p, n);  // little-endian hosts only
        p += n;
        return v;
    }
    uint64_t uleb() noexcept {
        uint64_t v = 0;
        unsigned shift = 0;
        for (;;) {
            if (!need(1)) return 0;
            const uint8_t b = *p++;
            if (shift < 64) v |= static_cast<uint64_t>(b & 0x7F) << shift;
            shift += 7;
            if (!(b & 0x80)) return v;
        }
    }
    int64_t sleb() noexcept {
        int64_t v = 0;
        unsigned shift = 0;
        uint8_t b = 0;
        do {
            if (!need(1)) return 0;
            b = *p++;
            if (shift < 64) v |= static_cast<int64_t>(b & 0x7F) << shift;
            shift += 7;
        } while (b & 0x80);
        if (shift < 64 && (b & 0x40)) v |= -(int64_t{1} << shift);
        return v;
    }
    void skip(size_t n) noexcept {
        if (need(n)) p += n;
    }
};

// The size of a pointer in encoding `enc` (DW_EH_PE_*), 0 for a LEB128.
size_t encoded_size(uint8_t enc) noexcept {
    switch (enc & 0x0F) {
    case 0x00: return sizeof(void*);  // absptr
    case 0x02: case 0x0A: return 2;   // udata2 / sdata2
    case 0x03: case 0x0B: return 4;   // udata4 / sdata4
    case 0x04: case 0x0C: return 8;   // udata8 / sdata8
    default: return 0;                // uleb128 / sleb128
    }
}

void skip_encoded(Reader& r, uint8_t enc) noexcept {
    if (enc == 0xFF) return;  // omit
    const size_t n = encoded_size(enc);
    if (n) r.skip(n);
    else r.uleb();
}

// How the caller's value of one register is recovered.
struct Rule {
    enum Kind : uint8_t { Same, Undefined, Offset, ValOffset, Register, Unsupported };
    Kind kind = Same;
    int64_t value = 0;  // the CFA offset, or the register
};

struct Row {
    unsigned cfa_reg = kSpReg;
    int64_t cfa_offset = 0;
    bool cfa_ok = true;
    Rule fp;
    Rule ra;
    bool ra_signed = false;  // AArch64 pointer authentication
};

struct Cie {
    uint64_t code_align = 1;
    int64_t data_align = 1;
    unsigned ra_reg = 0;
    uint8_t fde_encoding = 0;
    bool has_z = false;
    const uint8_t* instructions = nullptr;
    const uint8_t* end = nullptr;
};

bool parse_cie(const uint8_t* cie, Cie& out) noexcept {
    uint32_t len = 0;
    std::memcpy(&len, cie, 4);
    if (len == 0 || len == 0xFFFFFFFFu) return false;
    Reader r{cie + 4, cie + 4 + len};
    if (r.fixed(4) != 0) return false;  // CIE id
    const uint8_t version = r.u8();
    const char* aug = reinterpret_cast<const char*>(r.p);
    const size_t aug_len = strnlen(aug, static_cast<size_t>(r.end - r.p));
    r.skip(aug_len + 1);
    if (aug[0] != '\0' && aug[0] != 'z') return false;  // pre-'z' augmentations
    out.code_align = r.uleb();
    out.data_align = r.sleb();
    out.ra_reg = version == 1 ? r.u8() : static_cast<unsigned>(r.uleb());
    if (aug[0] == 'z') {
        out.has_z = true;
        const uint64_t data_len = r.uleb();
        const uint8_t* data_end = r.p + data_len;
        for (size_t i = 1; i < aug_len && r.ok && r.p < data_end; ++i) {
            switch (aug[i]) {
            case 'R': out.fde_encoding = r.u8(); break;
            case 'L': r.u8(); break;
            case 'P': skip_encoded(r, r.u8()); break;
            default: break;  // 'S', 'B', 'G': no data
            }
        }
        r.p = data_end;
    }
    out.instructions = r.p;
    out.end = r.end;
    return r.ok && r.p <= r.end;
}

void set_rule(Row& row, const Cie& cie, uint64_t reg, Rule rule) noexcept {
    if (reg == kFpReg) row.fp = rule;
    if (reg == cie.ra_reg) row.ra = rule;
}

void restore_rule(Row& row, const Row& initial, const Cie& cie, uint64_t reg) noexcept {
    if (reg == kFpReg) row.fp = initial.fp;
    if (reg == cie.ra_reg) row.ra = initial.ra;
}

// Runs one call-frame program over `row` until the location passes
// `target`. False on an instruction it does not know, or a stack of
// remembered rows too deep.
bool run_cfa_program(const uint8_t* p, const uint8_t* end, const Cie& cie, uintptr_t& loc,
                     uintptr_t target, Row& row, const Row& initial) noexcept {
    constexpr int kMaxRemembered = 16;
    Row remembered[kMaxRemembered];
    int depth = 0;
    Reader r{p, end};
    auto advance = [&](uint64_t delta) {
        loc += static_cast<uintptr_t>(delta * cie.code_align);
        return loc <= target;
    };
    while (r.ok && r.p < r.end) {
        const uint8_t op = r.u8();
        switch (op & 0xC0) {
        case 0x40:  // advance_loc
            if (!advance(op & 0x3F)) return true;
            continue;
        case 0x80:  // offset
            set_rule(row, cie, op & 0x3F, {Rule::Offset, static_cast<int64_t>(r.uleb()) * cie.data_align});
            continue;
        case 0xC0:  // restore
            restore_rule(row, initial, cie, op & 0x3F);
            continue;
        default: break;
        }
        switch (op) {
        case 0x00: break;  // nop
        case 0x02: if (!advance(r.fixed(1))) return true; break;
        case 0x03: if (!advance(r.fixed(2))) return true; break;
        case 0x04: if (!advance(r.fixed(4))) return true; break;
        case 0x05: {  // offset_extended
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::Offset, static_cast<int64_t>(r.uleb()) * cie.data_align});
            break;
        }
        case 0x06: restore_rule(row, initial, cie, r.uleb()); break;           // restore_extended
        case 0x07: set_rule(row, cie, r.uleb(), {Rule::Undefined, 0}); break;  // undefined
        case 0x08: set_rule(row, cie, r.uleb(), {Rule::Same, 0}); break;       // same_value
        case 0x09: {  // register
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::Register, static_cast<int64_t>(r.uleb())});
            break;
        }
        case 0x0A:  // remember_state
            if (depth == kMaxRemembered) return false;
            remembered[depth++] = row;
            break;
        case 0x0B:  // restore_state
            if (depth == 0) return false;
            row = remembered[--depth];
            break;
        case 0x0C:  // def_cfa
            row.cfa_reg = static_cast<unsigned>(r.uleb());
            row.cfa_offset = static_cast<int64_t>(r.uleb());
            row.cfa_ok = true;
            break;
        case 0x0D: row.cfa_reg = static_cast<unsigned>(r.uleb()); break;            // def_cfa_register
        case 0x0E: row.cfa_offset = static_cast<int64_t>(r.uleb()); break;          // def_cfa_offset
        case 0x0F: row.cfa_ok = false; r.skip(r.uleb()); break;                      // def_cfa_expression
        case 0x10: case 0x16: {  // expression, val_expression
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::Unsupported, 0});
            r.skip(r.uleb());
            break;
        }
        case 0x11: {  // offset_extended_sf
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::Offset, r.sleb() * cie.data_align});
            break;
        }
        case 0x12:  // def_cfa_sf
            row.cfa_reg = static_cast<unsigned>(r.uleb());
            row.cfa_offset = r.sleb() * cie.data_align;
            row.cfa_ok = true;
            break;
        case 0x13: row.cfa_offset = r.sleb() * cie.data_align; break;  // def_cfa_offset_sf
        case 0x14: {  // val_offset
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::ValOffset, static_cast<int64_t>(r.uleb()) * cie.data_align});
            break;
        }
        case 0x15: {  // val_offset_sf
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::ValOffset, r.sleb() * cie.data_align});
            break;
        }
        case 0x2D: row.ra_signed = !row.ra_signed; break;  // AArch64 negate_ra_state
        case 0x2E: r.uleb(); break;                         // GNU_args_size
        case 0x2F: {  // GNU_negative_offset_extended
            const uint64_t reg = r.uleb();
            set_rule(row, cie, reg, {Rule::Offset, -static_cast<int64_t>(r.uleb()) * cie.data_align});
            break;
        }
        default: return false;  // set_loc and anything newer
        }
    }
    return r.ok;
}

bool register_value(const NativeUnwindFrame& f, int64_t reg, uintptr_t& out) noexcept {
    if (reg == kSpReg) out = f.sp;
    else if (reg == kFpReg) out = f.fp;
    else return false;
    return true;
}

bool apply_rule(const Rule& rule, const NativeUnwindFrame& f, uintptr_t cfa, uintptr_t same, uintptr_t& out) noexcept {
    switch (rule.kind) {
    case Rule::Same: out = same; return true;
    case Rule::Undefined: out = 0; return true;
    case Rule::Offset: out = load_word(cfa + static_cast<uintptr_t>(rule.value)); return true;
    case Rule::ValOffset: out = cfa + static_cast<uintptr_t>(rule.value); return true;
    case Rule::Register: return register_value(f, rule.value, out);
    default: return false;
    }
}

// The outcome of stepping by CFI: the frame was stepped, the stack ends
// here (the return address is undefined), or no CFI describes the frame.
enum class CfiStep { Stepped, End, NoCfi, Failed };

CfiStep step_by_cfi(NativeUnwindFrame& f, uintptr_t lookup) noexcept {
    uintptr_t fn_start = 0;
    const auto* fde = static_cast<const uint8_t*>(detail::find_fde(lookup, fn_start));
    if (!fde) return CfiStep::NoCfi;

    uint32_t len = 0;
    std::memcpy(&len, fde, 4);
    if (len == 0 || len == 0xFFFFFFFFu) return CfiStep::Failed;
    int32_t cie_delta = 0;
    std::memcpy(&cie_delta, fde + 4, 4);
    Cie cie;
    if (!parse_cie(fde + 4 - cie_delta, cie)) return CfiStep::Failed;

    Reader r{fde + 8, fde + 4 + len};
    const size_t ptr_size = encoded_size(cie.fde_encoding);
    if (ptr_size == 0) return CfiStep::Failed;
    r.skip(ptr_size);  // pc_begin: fn_start has it decoded
    r.skip(ptr_size);  // pc_range
    if (cie.has_z) r.skip(r.uleb());
    if (!r.ok) return CfiStep::Failed;

    // A register no rule names keeps its value ("same value"): for the
    // return address that is a leaf's link register, never a caller's frame.
    Row initial;
    uintptr_t loc = fn_start;
    if (!run_cfa_program(cie.instructions, cie.end, cie, loc, UINTPTR_MAX, initial, initial)) return CfiStep::Failed;
    Row row = initial;
    loc = fn_start;
    if (!run_cfa_program(r.p, r.end, cie, loc, lookup, row, initial)) return CfiStep::Failed;
    if (!row.cfa_ok) return CfiStep::Failed;

    uintptr_t base = 0;
    if (!register_value(f, row.cfa_reg, base)) return CfiStep::Failed;
    const uintptr_t cfa = base + static_cast<uintptr_t>(row.cfa_offset);
    if (cfa <= f.sp || (cfa % 8) != 0) return CfiStep::Failed;

    uintptr_t ra = 0, fp = 0;
    if (row.ra.kind == Rule::Undefined) return CfiStep::End;
    if (row.ra.kind == Rule::Same) return CfiStep::Failed;
    if (!apply_rule(row.ra, f, cfa, 0, ra) || !apply_rule(row.fp, f, cfa, f.fp, fp)) return CfiStep::Failed;
#if defined(__aarch64__)
    if (row.ra_signed) ra &= (uintptr_t{1} << 48) - 1;  // strip the pointer authentication code
#endif
    if (ra == 0) return CfiStep::End;
    f.ip = ra;
    f.sp = cfa;
    f.fp = fp;
    return CfiStep::Stepped;
}

#endif // !__APPLE__

} // namespace

bool brass_unwind_step(NativeUnwindFrame& f, bool ip_is_return_address) noexcept {
    if (f.ip == 0) return false;
    const uintptr_t lookup = ip_is_return_address ? f.ip - 1 : f.ip;
#if defined(__APPLE__)
    unw_context_t uc;
    unw_cursor_t cur;
    if (unw_getcontext(&uc) != 0 || unw_init_local(&cur, &uc) != 0) return false;
    // The IP last: setting it looks up the unwind information of the code
    // there, at the call itself (its row is the return address's).
    if (unw_set_reg(&cur, UNW_REG_SP, f.sp) != 0 || unw_set_reg(&cur, kFpReg, f.fp) != 0 ||
        unw_set_reg(&cur, UNW_REG_IP, lookup) != 0) {
        return false;
    }
    unw_proc_info_t info;
    if (unw_get_proc_info(&cur, &info) != 0 || info.start_ip == 0) {
        NativeUnwindFrame g = f;
        if (!step_by_frame_record(g)) return false;
        f = g;
        return true;
    }
    if (unw_step(&cur) <= 0) return false;
    unw_word_t ip = 0, sp = 0, fp = 0;
    if (unw_get_reg(&cur, UNW_REG_IP, &ip) != 0 || unw_get_reg(&cur, UNW_REG_SP, &sp) != 0 ||
        unw_get_reg(&cur, kFpReg, &fp) != 0 || ip == 0 || sp <= f.sp) {
        return false;
    }
    f.ip = static_cast<uintptr_t>(ip);
    f.sp = static_cast<uintptr_t>(sp);
    f.fp = static_cast<uintptr_t>(fp);
    return true;
#else
    NativeUnwindFrame g = f;
    switch (step_by_cfi(g, lookup)) {
    case CfiStep::Stepped: f = g; return true;
    case CfiStep::End: return false;
    case CfiStep::NoCfi:
        if (!step_by_frame_record(g)) return false;
        f = g;
        return true;
    default: return false;
    }
#endif
}

namespace {

struct Capture {
    unsigned want;
    unsigned index = 0;
    uintptr_t callee_cfa = 0;
    NativeUnwindFrame out;
    bool found = false;
};

_Unwind_Reason_Code capture_one(_Unwind_Context* ctx, void* arg) {
    auto* c = static_cast<Capture*>(arg);
    const uintptr_t cfa = static_cast<uintptr_t>(_Unwind_GetCFA(ctx));
    if (c->index == c->want) {
        c->out.ip = static_cast<uintptr_t>(_Unwind_GetIP(ctx));
        c->out.sp = c->callee_cfa;
        c->out.fp = static_cast<uintptr_t>(_Unwind_GetGR(ctx, static_cast<int>(kFpReg)));
        c->found = c->out.ip != 0 && c->out.sp != 0;
        return _URC_END_OF_STACK;
    }
    c->callee_cfa = cfa;
    ++c->index;
    return _URC_NO_REASON;
}

} // namespace

__attribute__((noinline)) bool brass_capture_frame(NativeUnwindFrame& f, unsigned skip) noexcept {
    // Frame 0 is this function, 1 its caller, 2 that caller's caller.
    Capture c{skip + 2};
    _Unwind_Backtrace(capture_one, &c);
    if (!c.found) return false;
    f = c.out;
    return true;
}

bool brass_ip_in_image(uintptr_t ip) noexcept {
    Dl_info info;
    return dladdr(reinterpret_cast<void*>(ip), &info) != 0 && info.dli_fbase != nullptr;
}

} // namespace brass

#endif // BRASS_NATIVE_UNWIND
