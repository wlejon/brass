// A C++ exception thrown by a host function that JIT code called has to
// unwind through the JIT frames to the C++ caller. On ELF and Mach-O hosts
// that needs the engine's .eh_frame registered with the unwinder; on Windows
// its .pdata. The JIT functions are called through plain function pointers so
// that nothing but their own unwind info stands between the throw and the
// catch.

#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace brass;

namespace {

struct HostError : std::runtime_error {
    int64_t value;
    explicit HostError(int64_t v) : std::runtime_error("host error"), value(v) {}
};

int64_t host_maybe_throw(int64_t v) {
    if (v > 1000) throw HostError(v);
    return v + 1;
}

double host_maybe_throw_f64(double v) {
    if (v > 1000.0) throw HostError(static_cast<int64_t>(v));
    return v * 0.5;
}

// Values live across the call force callee-saved registers (x19.., d8.. on
// AArch64; rbx, r12.. on x64) into the JIT frames, so the unwinder has to
// restore them from the CFI on the way out.
const char* kUnwindModule = R"(
module @jit_unwind

extern @host_maybe_throw
extern @host_maybe_throw_f64

func @inner(%0: i64, %1: f64) -> i64 {
bb0:
  %2 = iconst.i64 3
  %3 = mul.i64 %0, %2
  %4 = iconst.i64 5
  %5 = add.i64 %0, %4
  %6 = iconst.i64 7
  %7 = xor.i64 %0, %6
  %8 = fconst.f64 2.5
  %9 = mul.f64 %1, %8
  %10 = call.f64 @host_maybe_throw_f64(%1)
  %11 = call.i64 @host_maybe_throw(%3)
  %12 = add.i64 %11, %5
  %13 = add.i64 %12, %7
  %14 = add.f64 %9, %10
  %15 = fptosi.i64 %14
  %16 = add.i64 %13, %15
  ret %16
}

func @outer(%0: i64, %1: f64) -> i64 {
bb0:
  %2 = iconst.i64 11
  %3 = mul.i64 %0, %2
  %4 = call.i64 @inner(%0, %1)
  %5 = add.i64 %4, %3
  ret %5
}
)";

using OuterFn = int64_t (*)(int64_t, double);

int64_t expected_outer(int64_t x, double d) {
    const int64_t inner = (x * 3 + 1) + (x + 5) + (x ^ 7) + static_cast<int64_t>(d * 2.5 + d * 0.5);
    return inner + x * 11;
}

// Kept out of line so that its locals live in callee-saved registers across
// the call that throws.
#if defined(_MSC_VER) && !defined(__clang__)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int64_t call_and_catch(OuterFn fn, int64_t x, double d, int64_t& caught) {
    volatile int64_t seed = x;
    int64_t a = seed * 17 + 3;
    int64_t b = seed ^ 0x5a5a;
    int64_t c = seed - 99;
    double f = static_cast<double>(seed) * 1.25;
    caught = -1;
    try {
        (void)fn(x, d);
    } catch (const HostError& e) {
        caught = e.value;
    }
    return a + b + c + static_cast<int64_t>(f);
}

uint32_t rd32(const std::vector<uint8_t>& b, size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, 4);
    return v;
}

uint64_t rd64(const std::vector<uint8_t>& b, size_t off) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + off, 8);
    return v;
}

uint64_t uleb(const std::vector<uint8_t>& b, size_t& p) {
    uint64_t v = 0;
    unsigned shift = 0;
    for (;;) {
        const uint8_t byte = b[p++];
        v |= uint64_t{byte & 0x7Fu} << shift;
        if (!(byte & 0x80)) return v;
        shift += 7;
    }
}

struct ImageSection {
    uint64_t vaddr = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct FdeInfo {
    uint64_t fde_vaddr = 0;
    uint64_t pc_begin = 0;
    uint64_t pc_range = 0;
    std::vector<uint64_t> saved_regs;   // DWARF numbers given a save slot
};

// Every FDE in an image's .eh_frame (offset == file offset), its pc range
// resolved from the pcrel sdata4 field, and the registers its CFA program
// saves.
std::vector<FdeInfo> parse_eh_frame(const std::vector<uint8_t>& img, const ImageSection& eh) {
    std::vector<FdeInfo> out;
    size_t p = static_cast<size_t>(eh.offset);
    const size_t end = static_cast<size_t>(eh.offset + eh.size);
    while (p + 8 <= end) {
        const uint32_t len = rd32(img, p);
        if (len == 0) break;
        REQUIRE(len != 0xFFFFFFFFu);
        const size_t next = p + 4 + len;
        REQUIRE(next <= end);
        if (rd32(img, p + 4) != 0) {
            FdeInfo f;
            f.fde_vaddr = eh.vaddr + (p - eh.offset);
            const uint64_t field = f.fde_vaddr + 8;
            f.pc_begin = field + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(rd32(img, p + 8))));
            f.pc_range = rd32(img, p + 12);
            size_t q = p + 16;
            q += static_cast<size_t>(uleb(img, q));   // augmentation data
            while (q < next) {
                const uint8_t op = img[q++];
                if ((op & 0xC0) == 0x40) continue;                       // advance_loc
                if ((op & 0xC0) == 0x80) { f.saved_regs.push_back(op & 0x3Fu); (void)uleb(img, q); continue; }
                if (op == 0x00) continue;                                // nop
                if (op == 0x0C) { (void)uleb(img, q); (void)uleb(img, q); continue; }   // def_cfa
                if (op == 0x0D || op == 0x0E) { (void)uleb(img, q); continue; }          // def_cfa_register/offset
                if (op == 0x05) { f.saved_regs.push_back(uleb(img, q)); (void)uleb(img, q); continue; }   // offset_extended
                CHECK(!"unexpected CFA opcode in a brass FDE");
                break;
            }
            out.push_back(std::move(f));
        }
        p = next;
    }
    return out;
}

bool find_elf_section(const std::vector<uint8_t>& img, const std::string& want, ImageSection& out) {
    const uint64_t shoff = rd64(img, 40);
    uint16_t shnum = 0, shstrndx = 0;
    std::memcpy(&shnum, img.data() + 60, 2);
    std::memcpy(&shstrndx, img.data() + 62, 2);
    const uint64_t strtab_off = rd64(img, static_cast<size_t>(shoff + shstrndx * 64u + 24));
    for (uint16_t i = 0; i < shnum; ++i) {
        const size_t sh = static_cast<size_t>(shoff + i * 64u);
        const char* name = reinterpret_cast<const char*>(img.data() + strtab_off + rd32(img, sh));
        if (want == name) {
            out = {rd64(img, sh + 16), rd64(img, sh + 24), rd64(img, sh + 32)};
            return true;
        }
    }
    return false;
}

bool find_macho_section(const std::vector<uint8_t>& img, const std::string& seg, const std::string& sect,
                        ImageSection& out) {
    const uint32_t ncmds = rd32(img, 16);
    size_t p = 32;
    for (uint32_t c = 0; c < ncmds; ++c) {
        const uint32_t cmd = rd32(img, p);
        const uint32_t cmdsize = rd32(img, p + 4);
        if (cmd == 0x19) {   // LC_SEGMENT_64
            const uint32_t nsects = rd32(img, p + 64);
            for (uint32_t s = 0; s < nsects; ++s) {
                const size_t sp = p + 72 + s * 80u;
                const std::string sn(reinterpret_cast<const char*>(img.data() + sp), strnlen(reinterpret_cast<const char*>(img.data() + sp), 16));
                const std::string gn(reinterpret_cast<const char*>(img.data() + sp + 16), strnlen(reinterpret_cast<const char*>(img.data() + sp + 16), 16));
                if (sn == sect && gn == seg) {
                    out = {rd64(img, sp + 32), rd32(img, sp + 48), rd64(img, sp + 40)};
                    return true;
                }
            }
        }
        p += cmdsize;
    }
    return false;
}

// Each function has exactly one FDE, inside the text section, and the
// AArch64 FPR saves are numbered as DWARF numbers them (V8 = 72).
void check_image_fdes(const std::vector<FdeInfo>& fdes, const ImageSection& text, size_t num_functions,
                      bool aarch64) {
    CHECK_EQ(fdes.size(), num_functions);
    std::vector<uint64_t> starts;
    bool fpr_saved = false;
    for (const auto& f : fdes) {
        CHECK(f.pc_begin >= text.vaddr);
        CHECK(f.pc_begin + f.pc_range <= text.vaddr + text.size);
        CHECK(f.pc_range > 0);
        starts.push_back(f.pc_begin);
        for (uint64_t r : f.saved_regs) {
            if (aarch64) {
                CHECK((r <= 30 || (r >= 72 && r <= 79)));
                if (r >= 72) fpr_saved = true;
            } else {
                CHECK(r <= 16);
            }
        }
    }
    // @inner keeps doubles live across its calls, in D8.. on AArch64.
    if (aarch64) CHECK(fpr_saved);
    std::sort(starts.begin(), starts.end());
    CHECK(std::adjacent_find(starts.begin(), starts.end()) == starts.end());
}

} // namespace

TEST_CASE("Image unwind - .so and .dylib outputs carry an FDE per function") {
    DiagnosticReporter diag;
    auto mod = parse_module(kUnwindModule, &diag);
    REQUIRE(mod != nullptr);
    const size_t num_functions = 2;

    for (const Target t : {Target::x64_linux(), Target::aarch64_linux()}) {
        target::LinkerOptions opts;
        opts.format = target::OutputFormat::LinuxElfSo;
        opts.export_all_functions = true;
        opts.imports = {{"libhost.so", {"host_maybe_throw", "host_maybe_throw_f64"}}};
        const std::vector<uint8_t> so = target::AotLinker::link(*mod, t, opts);
        REQUIRE(!so.empty());
        ImageSection text, eh, hdr;
        REQUIRE(find_elf_section(so, ".text", text));
        REQUIRE(find_elf_section(so, ".eh_frame", eh));
        REQUIRE(find_elf_section(so, ".eh_frame_hdr", hdr));
        const auto fdes = parse_eh_frame(so, eh);
        check_image_fdes(fdes, text, num_functions, t.is_aarch64());

        // PT_GNU_EH_FRAME names the header, whose sorted table maps each
        // function start to its FDE.
        const uint64_t phoff = rd64(so, 32);
        uint16_t phnum = 0;
        std::memcpy(&phnum, so.data() + 56, 2);
        bool found = false;
        for (uint16_t i = 0; i < phnum; ++i) {
            const size_t ph = static_cast<size_t>(phoff + i * 56u);
            if (rd32(so, ph) == target::elf64::PT_GNU_EH_FRAME) {
                found = true;
                CHECK_EQ(rd64(so, ph + 16), hdr.vaddr);
            }
        }
        CHECK(found);
        const size_t h = static_cast<size_t>(hdr.offset);
        CHECK_EQ(so[h], uint8_t{1});
        const int64_t eh_ptr = static_cast<int32_t>(rd32(so, h + 4));
        CHECK_EQ(static_cast<uint64_t>(static_cast<int64_t>(hdr.vaddr + 4) + eh_ptr), eh.vaddr);
        const uint32_t count = rd32(so, h + 8);
        REQUIRE_EQ(count, static_cast<uint32_t>(fdes.size()));
        int64_t prev = INT64_MIN;
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t pc = static_cast<int64_t>(hdr.vaddr) + static_cast<int32_t>(rd32(so, h + 12 + i * 8u));
            const int64_t fde = static_cast<int64_t>(hdr.vaddr) + static_cast<int32_t>(rd32(so, h + 16 + i * 8u));
            CHECK(pc > prev);
            prev = pc;
            bool matches = false;
            for (const auto& f : fdes) {
                if (static_cast<int64_t>(f.fde_vaddr) == fde && static_cast<int64_t>(f.pc_begin) == pc) matches = true;
            }
            CHECK(matches);
        }
    }

    for (const Target t : {Target::x64_macos(), Target::aarch64_macos()}) {
        target::LinkerOptions opts;
        opts.format = target::OutputFormat::MacOSMachODylib;
        opts.export_all_functions = true;
        opts.imports = {{"@rpath/libhost.dylib", {"host_maybe_throw", "host_maybe_throw_f64"}}};
        const std::vector<uint8_t> dylib = target::AotLinker::link(*mod, t, opts);
        REQUIRE(!dylib.empty());
        ImageSection text, eh;
        REQUIRE(find_macho_section(dylib, "__TEXT", "__text", text));
        REQUIRE(find_macho_section(dylib, "__TEXT", "__eh_frame", eh));
        check_image_fdes(parse_eh_frame(dylib, eh), text, num_functions, t.is_aarch64());
    }
}

namespace {

// A Windows ARM64 unwinder over the codes brass emits, run against a frame
// laid out the way AArch64FrameLayout's prologue lays it out.
struct A64Machine {
    uint64_t sp = 0;
    uint64_t x[31] = {};    // x0..x30; x29 = fp, x30 = lr
    uint64_t d[32] = {};
    std::unordered_map<uint64_t, uint64_t> mem;
    uint64_t load(uint64_t addr) {
        auto it = mem.find(addr);
        REQUIRE(it != mem.end());
        return it->second;
    }
};

void run_arm64_unwind_codes(const std::vector<uint8_t>& codes, A64Machine& m) {
    size_t i = 0;
    for (;;) {
        REQUIRE(i < codes.size());
        const uint8_t b = codes[i];
        if (b == 0xE4) return;                                         // end
        if (b < 0x20) { m.sp += uint64_t{b} * 16; ++i; continue; }     // alloc_s
        if ((b & 0xC0) == 0x40) {                                      // save_fplr
            const uint64_t at = m.sp + uint64_t(b & 0x3F) * 8;
            m.x[29] = m.load(at);
            m.x[30] = m.load(at + 8);
            ++i;
            continue;
        }
        if (b == 0xE1) { m.sp = m.x[29]; ++i; continue; }              // set_fp
        if (b == 0xE3) { ++i; continue; }                              // nop
        if (b == 0xE0) {                                               // alloc_l
            m.sp += ((uint64_t{codes[i + 1]} << 16) | (uint64_t{codes[i + 2]} << 8) | codes[i + 3]) * 16;
            i += 4;
            continue;
        }
        const uint16_t w = static_cast<uint16_t>((b << 8) | codes[i + 1]);
        const uint64_t z = w & 0x3F;
        if ((w & 0xF800) == 0xC000) { m.sp += uint64_t(w & 0x7FF) * 16; i += 2; continue; }   // alloc_m
        if ((w & 0xFC00) == 0xC800) {                                  // save_regp
            const unsigned r = 19 + ((w >> 6) & 0xF);
            m.x[r] = m.load(m.sp + z * 8);
            m.x[r + 1] = m.load(m.sp + z * 8 + 8);
            i += 2;
            continue;
        }
        if ((w & 0xFC00) == 0xD000) {                                  // save_reg
            m.x[19 + ((w >> 6) & 0xF)] = m.load(m.sp + z * 8);
            i += 2;
            continue;
        }
        if ((w & 0xFE00) == 0xD800) {                                  // save_fregp
            const unsigned r = 8 + ((w >> 6) & 0x7);
            m.d[r] = m.load(m.sp + z * 8);
            m.d[r + 1] = m.load(m.sp + z * 8 + 8);
            i += 2;
            continue;
        }
        if ((w & 0xFE00) == 0xDC00) {                                  // save_freg
            m.d[8 + ((w >> 6) & 0x7)] = m.load(m.sp + z * 8);
            i += 2;
            continue;
        }
        CHECK(!"unwind code the simulator does not know");
        return;
    }
}

} // namespace

TEST_CASE("Windows ARM64 unwind - .xdata restores every saved register and the caller's SP") {
    // Frames with and without outgoing argument space, callee-saved GPRs
    // that do and do not pair, FPRs, and a frame too large for alloc_s.
    const char* src = R"(
module @a64_xdata

extern @host_maybe_throw
extern @host_maybe_throw_f64
extern @sink9

func @many(%0: i64, %1: f64) -> i64 {
bb0:
  %2 = iconst.i64 3
  %3 = mul.i64 %0, %2
  %4 = add.i64 %0, %2
  %5 = xor.i64 %0, %2
  %6 = sub.i64 %0, %2
  %7 = mul.i64 %3, %4
  %8 = fconst.f64 1.5
  %9 = mul.f64 %1, %8
  %10 = add.f64 %1, %8
  %11 = sub.f64 %1, %8
  %12 = call.i64 @sink9(%0, %2, %3, %4, %5, %6, %7, %0, %2, %3)
  %13 = call.f64 @host_maybe_throw_f64(%1)
  %14 = add.i64 %12, %3
  %15 = add.i64 %14, %4
  %16 = add.i64 %15, %5
  %17 = add.i64 %16, %6
  %18 = add.i64 %17, %7
  %19 = add.f64 %9, %10
  %20 = add.f64 %19, %11
  %21 = add.f64 %20, %13
  %22 = fptosi.i64 %21
  %23 = add.i64 %18, %22
  ret %23
}

func @big(%0: i64) -> i64 {
bb0:
  %1 = alloca 4096, 16
  store.i64 %1, %0
  %2 = call.i64 @host_maybe_throw(%0)
  %3 = load.i64 %1
  %4 = add.i64 %2, %3
  ret %4
}

func @leaf(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 1
  %2 = add.i64 %0, %1
  ret %2
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    object::ObjectFile obj = object::compile_module_to_object(*mod, Target::aarch64_windows());
    auto& xdata = obj.get_or_create_section(".xdata", object::SectionKind::XData,
                                            object::SectionFlags::Read | object::SectionFlags::Alloc, 4);
    (void)xdata;
    auto& pdata_ref = obj.get_or_create_section(".pdata", object::SectionKind::PData,
                                                object::SectionFlags::Read | object::SectionFlags::Alloc, 4);
    (void)pdata_ref;
    object::CoffUnwindBuilder::build_unwind_info(obj, *obj.get_section(".pdata"), *obj.get_section(".xdata"));
    const auto* pdata = obj.get_section(".pdata");
    const auto* xd = obj.get_section(".xdata");
    REQUIRE(pdata != nullptr);
    REQUIRE(xd != nullptr);
    REQUIRE_EQ(pdata->data.size(), obj.functions.size() * 8);

    bool saw_outgoing = false, saw_fpr = false, saw_big = false, saw_leaf = false;
    for (size_t f = 0; f < obj.functions.size(); ++f) {
        const auto& fn = obj.functions[f];
        const auto& fi = fn.frame_info;
        const uint32_t word1 = rd32(pdata->data, f * 8 + 4);
        if (fi.is_leaf) {
            // Packed: FunctionLength only; no frame, LR not saved.
            saw_leaf = true;
            CHECK_EQ(word1, 0x1u | static_cast<uint32_t>((fn.text_size / 4) << 2));
            continue;
        }
        REQUIRE_EQ(word1 & 3u, 0u);   // a full .xdata record
        const size_t xoff = word1;
        const uint32_t header = rd32(xd->data, xoff);
        CHECK_EQ(header & 0x3FFFFu, static_cast<uint32_t>(fn.text_size / 4));
        CHECK_EQ((header >> 18) & 3u, 0u);    // Vers
        CHECK_EQ((header >> 21) & 1u, 0u);    // E
        CHECK_EQ((header >> 22) & 0x1Fu, 0u); // no epilog scopes
        const uint32_t words = header >> 27;
        REQUIRE(words > 0);
        const std::vector<uint8_t> codes(xd->data.begin() + static_cast<std::ptrdiff_t>(xoff + 4),
                                         xd->data.begin() + static_cast<std::ptrdiff_t>(xoff + 4 + words * 4));

        A64Machine m;
        const uint64_t caller_sp = 0x100000;
        const uint64_t caller_lr = 0xAAAA0000;
        const uint64_t caller_fp = 0xBBBB0000;
        const uint64_t outgoing = (fi.outgoing_arg_space + 15) & ~uint64_t{15};
        if (outgoing) saw_outgoing = true;
        if (fi.total_frame_size >= 512) saw_big = true;
        m.sp = caller_sp - fi.total_frame_size;
        m.x[29] = m.sp + outgoing;
        m.mem[m.x[29]] = caller_fp;
        m.mem[m.x[29] + 8] = caller_lr;
        const auto gprs = aarch64::AArch64FrameLayout::get_saved_callee_gprs(fi);
        const auto fprs = aarch64::AArch64FrameLayout::get_saved_callee_fprs(fi);
        if (!fprs.empty()) saw_fpr = true;
        for (auto r : gprs) {
            const auto a = aarch64::AArch64FrameLayout::callee_gpr_address(r, fi);
            m.mem[m.x[29] + static_cast<uint64_t>(a.offset)] = 0x1000 + static_cast<uint64_t>(r);
        }
        for (auto r : fprs) {
            const auto a = aarch64::AArch64FrameLayout::callee_fpr_address(r, fi);
            m.mem[m.x[29] + static_cast<uint64_t>(a.offset)] = 0x2000 + static_cast<uint64_t>(r);
        }
        run_arm64_unwind_codes(codes, m);
        CHECK_EQ(m.sp, caller_sp);
        CHECK_EQ(m.x[29], caller_fp);
        CHECK_EQ(m.x[30], caller_lr);
        for (auto r : gprs) CHECK_EQ(m.x[static_cast<int>(r)], 0x1000 + static_cast<uint64_t>(r));
        for (auto r : fprs) CHECK_EQ(m.d[static_cast<int>(r)], 0x2000 + static_cast<uint64_t>(r));
    }
    CHECK(saw_outgoing);
    CHECK(saw_fpr);
    CHECK(saw_big);
    CHECK(saw_leaf);
}

TEST_CASE("JIT unwind - a host C++ exception propagates through JIT frames") {
    DiagnosticReporter diag;
    auto mod = parse_module(kUnwindModule, &diag);
    REQUIRE(mod != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_maybe_throw", reinterpret_cast<void*>(&host_maybe_throw));
    jit.register_external_symbol("host_maybe_throw_f64", reinterpret_cast<void*>(&host_maybe_throw_f64));
    REQUIRE(jit.compile_and_load(*mod));

    auto outer = jit.get_function_ptr<OuterFn>("outer");
    REQUIRE(outer != nullptr);

    // No throw: the ordinary path.
    CHECK_EQ(outer(4, 8.0), expected_outer(4, 8.0));

    // The i64 host call throws: two JIT frames between the throw and catch.
    int64_t caught = 0;
    const int64_t x = 500;
    const int64_t locals = call_and_catch(outer, x, 3.0, caught);
    CHECK_EQ(caught, x * 3);
    CHECK_EQ(locals, (x * 17 + 3) + (x ^ 0x5a5a) + (x - 99) + static_cast<int64_t>(x * 1.25));

    // The f64 host call throws, before the i64 one.
    caught = 0;
    const int64_t locals2 = call_and_catch(outer, 2, 4096.0, caught);
    CHECK_EQ(caught, int64_t{4096});
    CHECK_EQ(locals2, (2 * 17 + 3) + (2 ^ 0x5a5a) + (2 - 99) + static_cast<int64_t>(2 * 1.25));

    // And the engine still runs normally afterwards.
    CHECK_EQ(outer(7, 1.0), expected_outer(7, 1.0));
}

// The same through a shared library brass linked itself: the callback comes
// in as a pointer, so the image needs no imports.
TEST_CASE("AOT unwind - a host C++ exception propagates through a brass-linked library") {
    const Target host = Target::host();
    if (!host.is_linux() && !host.is_macos() && !host.is_windows()) return;

    const char* src = R"(
module @aot_unwind

func @through(%0: ptr, %1: i64) -> i64 {
bb0:
  %2 = iconst.i64 3
  %3 = mul.i64 %1, %2
  %4 = iconst.i64 9
  %5 = add.i64 %1, %4
  %6 = call_indirect.i64 %0(%3)
  %7 = add.i64 %6, %5
  ret %7
}

func @entry(%0: ptr, %1: i64) -> i64 {
bb0:
  %2 = iconst.i64 2
  %3 = mul.i64 %1, %2
  %4 = call.i64 @through(%0, %1)
  %5 = add.i64 %4, %3
  ret %5
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);

    const std::string ext = host.is_windows() ? ".dll" : (host.is_macos() ? ".dylib" : ".so");
    const std::filesystem::path lib_path = brass::test::scratch_dir() / ("test_aot_unwind" + ext);
    std::error_code ec;
    std::filesystem::remove(lib_path, ec);

    target::LinkerOptions opts;
    opts.module_name = "test_aot_unwind" + ext;
    opts.export_all_functions = true;
    REQUIRE(target::AotLinker::link_to_file(*mod, lib_path.string(), host, opts));

    std::string err;
    auto lib = target::DynamicLibrary::open(lib_path.string(), &err);
    if (!lib) std::cerr << "DynamicLibrary::open failed: " << err << "\n";
    REQUIRE(lib != nullptr);
    using EntryFn = int64_t (*)(int64_t (*)(int64_t), int64_t);
    auto entry = lib->get_function<EntryFn>("entry");
    REQUIRE(entry != nullptr);

    CHECK_EQ(entry(&host_maybe_throw, 10), (10 * 3 + 1) + (10 + 9) + 10 * 2);

    int64_t caught = -1;
    try {
        (void)entry(&host_maybe_throw, 700);
    } catch (const HostError& e) {
        caught = e.value;
    }
    CHECK_EQ(caught, int64_t{2100});

    lib.reset();
    std::filesystem::remove(lib_path, ec);
}

// Through JitExecutionEngine::invoke, whose argument-marshalling thunk is
// hand-written assembly with its own CFI on ELF / Mach-O. (On Windows the
// thunks carry no SEH unwind info; they are not covered here.)
TEST_CASE("JIT unwind - a host C++ exception propagates through invoke()") {
    if (Target::host().is_windows()) return;
    // Three or more arguments take the thunk on every host.
    const char* src = R"(
module @invoke_unwind

extern @host_maybe_throw

func @three(%0: i64, %1: f64, %2: i64) -> i64 {
bb0:
  %3 = mul.i64 %0, %2
  %4 = call.i64 @host_maybe_throw(%3)
  %5 = fptosi.i64 %1
  %6 = add.i64 %4, %5
  ret %6
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_maybe_throw", reinterpret_cast<void*>(&host_maybe_throw));
    REQUIRE(jit.compile_and_load(*mod));

    const RuntimeValue ok = jit.invoke(
        "three", {RuntimeValue::from_i64(4), RuntimeValue::from_f64(8.0), RuntimeValue::from_i64(5)});
    CHECK_EQ(ok.as_i64(), int64_t{4 * 5 + 1 + 8});

    int64_t caught = -1;
    try {
        (void)jit.invoke("three", {RuntimeValue::from_i64(800), RuntimeValue::from_f64(1.0), RuntimeValue::from_i64(3)});
    } catch (const HostError& e) {
        caught = e.value;
    }
    CHECK_EQ(caught, int64_t{2400});
}

TEST_CASE("JIT unwind - reloading an engine replaces its registration") {
    DiagnosticReporter diag;
    auto mod = parse_module(kUnwindModule, &diag);
    REQUIRE(mod != nullptr);

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_maybe_throw", reinterpret_cast<void*>(&host_maybe_throw));
    jit.register_external_symbol("host_maybe_throw_f64", reinterpret_cast<void*>(&host_maybe_throw_f64));
    for (int round = 0; round < 3; ++round) {
        REQUIRE(jit.compile_and_load(*mod));
        auto outer = jit.get_function_ptr<OuterFn>("outer");
        REQUIRE(outer != nullptr);
        int64_t caught = 0;
        (void)call_and_catch(outer, 600 + round, 1.0, caught);
        CHECK_EQ(caught, int64_t{(600 + round) * 3});
    }
}
