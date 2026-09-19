// Position-independent x64 code: no symbol address is an absolute word in
// .text. A function's address and a data symbol's are `lea`s, an import's
// address is a load from the image's GOT (IAT on PE, a JIT-owned slot
// in-process), and a switch is a compare chain with no table of code
// pointers. The same module goes through every writer and the JIT.

#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <brass/target/pe_dll_writer.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::target;

namespace {

uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
int32_t rdi32(const uint8_t* p) { int32_t v; std::memcpy(&v, p, 4); return v; }

// helper(n) = n + 1
// probe(n)  = [table] + [ext_data] + helper(n) + ext_fn(n) + (n < 3 ? n : 100)
//             — helper through its address, ext_fn through the import, and
//             the tail through a switch on n.
std::unique_ptr<Module> build_module() {
    auto mod = std::make_unique<Module>("pic_x64");
    mod->add_external_symbol("ext_data");
    mod->add_external_symbol("ext_fn");
    mod->add_external_symbol("table");
    Builder b(*mod);

    Function* helper = mod->create_function("helper", Type::i64(), {Type::i64()});
    b.set_function(helper);
    BasicBlock* hentry = b.append_block("entry");
    Value* hn = b.add_block_param(hentry, Type::i64());
    b.build_ret(b.build_add(hn, b.build_iconst_i64(1)));
    helper->rebuild_cfg_predecessors();

    // far_addr() = &ext_far, never dereferenced: the JIT registers it at an
    // address no mapping is near, so the load has to go through a slot.
    mod->add_external_symbol("ext_far");
    Function* far_fn = mod->create_function("far_addr", Type::ptr());
    b.set_function(far_fn);
    b.append_block("entry");
    b.build_ret(b.build_func_addr("ext_far"));
    far_fn->rebuild_cfg_predecessors();

    Function* probe = mod->create_function("probe", Type::i64(), {Type::i64()});
    b.set_function(probe);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b0 = b.append_block("case0");
    BasicBlock* b1 = b.append_block("case1");
    BasicBlock* b2 = b.append_block("case2");
    BasicBlock* bd = b.append_block("default");
    BasicBlock* merge = b.append_block("merge");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* tail = b.add_block_param(merge, Type::i64());

    b.position_at_end(entry);
    Value* f = b.build_func_addr("helper");
    Value* d = b.build_func_addr("table");
    Value* e = b.build_func_addr("ext_data");
    Value* dv = b.build_load(Type::i64(), d, 0);
    Value* ev = b.build_load(Type::i64(), e, 0);
    Value* fv = b.build_call_indirect(f, Type::i64(), {n});
    Value* xv = b.build_call("ext_fn", Type::i64(), {n});
    Value* sum = b.build_add(b.build_add(dv, ev), b.build_add(fv, xv));
    b.build_switch(n, bd, {SwitchCase(0, b0), SwitchCase(1, b1), SwitchCase(2, b2)});
    b.position_at_end(b0);
    b.build_br(merge, {b.build_iconst_i64(0)});
    b.position_at_end(b1);
    b.build_br(merge, {b.build_iconst_i64(1)});
    b.position_at_end(b2);
    b.build_br(merge, {b.build_iconst_i64(2)});
    b.position_at_end(bd);
    b.build_br(merge, {b.build_iconst_i64(100)});
    b.position_at_end(merge);
    b.build_ret(b.build_add(sum, tail));
    probe->rebuild_cfg_predecessors();

    REQUIRE(verify_module(*mod));
    return mod;
}

// `table` the way a host defines data after code generation: one word of
// 1000, then a pointer to helper (an absolute word the loader relocates).
void define_table(object::ObjectFile& obj) {
    object::Section& ro = obj.get_or_create_section(
        ".rodata", object::SectionKind::RoData,
        object::SectionFlags::Read | object::SectionFlags::Alloc, 16);
    ro.align_to(16);
    const size_t at = ro.data.size();
    ro.emit64(1000);
    ro.relocations.push_back({ro.data.size(), object::RelocKind::Abs64, "helper", 0, 0});
    ro.emit64(0);
    obj.add_symbol({"table", obj.get_section_index(".rodata"), at, 16,
                    object::SymbolBinding::Global, object::SymbolType::Object});
}

object::ObjectFile build_object(const Target& target) {
    auto mod = build_module();
    object::ObjectFile obj = object::compile_module_to_object(*mod, target);
    define_table(obj);
    return obj;
}

// The GOT loads of .text, and the proof that nothing else in .text needs a
// loader's hand.
std::vector<object::ObjectRelocation> got_loads(const object::ObjectFile& obj) {
    std::vector<object::ObjectRelocation> loads;
    const object::Section* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    for (const auto& r : text->relocations) {
        CHECK(r.kind != object::RelocKind::Abs64);
        if (r.kind == object::RelocKind::GotPCRel32) {
            CHECK_EQ(r.addend, int64_t(-4));
            CHECK_EQ(text->data[r.offset - 2], uint8_t(0x8B));
            loads.push_back(r);
        }
    }
    return loads;
}

bool has_load_of(const std::vector<object::ObjectRelocation>& loads, const char* name) {
    for (const auto& r : loads) if (r.symbol_name == name) return true;
    return false;
}

LinkerOptions link_options(const std::string& library) {
    LinkerOptions opts;
    opts.export_all_functions = true;
    opts.imports.push_back({library, {"ext_data", "ext_fn", "ext_far"}});
    return opts;
}

bool is_import(const std::string& name) { return name == "ext_data" || name == "ext_far"; }

struct Range { uint64_t lo = 0, hi = 0; bool holds(uint64_t a) const { return a >= lo && a < hi; } };

int64_t ext_fn_impl(int64_t n) { return n * 10; }

} // namespace

TEST_CASE("PIC x64 - Every Symbol Address Is a GOT Load Until the Object Is Placed") {
    object::ObjectFile obj = build_object(Target::x64_linux());
    std::vector<object::ObjectRelocation> loads = got_loads(obj);
    CHECK(has_load_of(loads, "helper"));
    CHECK(has_load_of(loads, "table"));
    CHECK(has_load_of(loads, "ext_data"));
    CHECK_FALSE(has_load_of(loads, "ext_fn"));   // a call, not an address

    // Placing it: the defined ones become leas, the imports stay loads.
    object::ObjectFile placed = obj;
    CHECK_EQ(object::relax_got_loads(placed), size_t(2));
    const object::Section* text = placed.get_section(".text");
    size_t leas = 0, gots = 0;
    for (const auto& r : text->relocations) {
        if (r.kind == object::RelocKind::PCRel32 && (r.symbol_name == "helper" || r.symbol_name == "table")) {
            CHECK_EQ(text->data[r.offset - 2], uint8_t(0x8D));
            ++leas;
        }
        if (r.kind == object::RelocKind::GotPCRel32) {
            CHECK(is_import(r.symbol_name));
            CHECK_EQ(text->data[r.offset - 2], uint8_t(0x8B));
            ++gots;
        }
    }
    CHECK(leas >= 2);
    CHECK_EQ(gots, size_t(2));
}

TEST_CASE("PIC x64 - A COFF Object Carries Its Own Slot For an Import's Address") {
    object::ObjectFile obj = build_object(Target::x64_windows());
    object::ObjectFile placed = obj;
    object::materialize_got_slots(placed, ".rdata", object::SectionKind::RoData,
                                  object::SectionFlags::Read | object::SectionFlags::Alloc);
    const object::ObjectSymbol* slot = placed.find_symbol("ext_data$got");
    REQUIRE(slot != nullptr);
    CHECK(slot->binding == object::SymbolBinding::Local);
    CHECK_EQ(slot->section_index, placed.get_section_index(".rdata"));
    bool slot_bound = false;
    for (const auto& r : placed.get_section(".rdata")->relocations) {
        if (r.offset == slot->value && r.kind == object::RelocKind::Abs64 && r.symbol_name == "ext_data") slot_bound = true;
    }
    CHECK(slot_bound);
    bool load_rewritten = false;
    for (const auto& r : placed.get_section(".text")->relocations) {
        CHECK(r.kind != object::RelocKind::GotPCRel32);
        if (r.kind == object::RelocKind::PCRel32 && r.symbol_name == "ext_data$got") load_rewritten = true;
    }
    CHECK(load_rewritten);
    CHECK_FALSE(object::emit_coff_object(obj).empty());
}

TEST_CASE("PIC x64 - Mach-O Dylib: Leas Into __TEXT and __const, GOT Loads Into __got, Rebases Only in Data") {
    object::ObjectFile obj = build_object(Target::x64_macos());
    std::vector<object::ObjectRelocation> loads = got_loads(obj);
    REQUIRE(loads.size() >= 3);

    LinkerOptions opts = link_options("@rpath/libext.dylib");
    std::string err;
    std::vector<uint8_t> dylib = AotLinker::link(obj, opts, &err);
    if (dylib.empty()) std::fprintf(stderr, "Mach-O link failed: %s\n", err.c_str());
    REQUIRE(!dylib.empty());
    CHECK(err.empty());

    using namespace brass::object;
    const uint32_t ncmds = rd32(dylib.data() + 16);
    Range text, stubs, got, konst;
    uint32_t text_fileoff = 0, text_seg = 0, nsegs = 0;
    uint32_t rebase_off = 0, rebase_size = 0, bind_off = 0, bind_size = 0;
    const uint8_t* p = dylib.data() + 32;
    for (uint32_t c = 0; c < ncmds; ++c) {
        const uint32_t cmd = rd32(p);
        const uint32_t size = rd32(p + 4);
        if (cmd == macho::LC_SEGMENT_64) {
            const bool is_text = std::strncmp(reinterpret_cast<const char*>(p + 8), "__TEXT", 16) == 0;
            if (is_text) text_seg = nsegs;
            ++nsegs;
            const uint32_t nsects = rd32(p + 64);
            for (uint32_t s = 0; s < nsects; ++s) {
                const uint8_t* sect = p + 72 + s * 80;
                const char* name = reinterpret_cast<const char*>(sect);
                Range r{rd64(sect + 32), rd64(sect + 32) + rd64(sect + 40)};
                if (std::strncmp(name, "__text", 16) == 0) { text = r; text_fileoff = rd32(sect + 48); }
                else if (std::strncmp(name, "__stubs", 16) == 0) stubs = r;
                else if (std::strncmp(name, "__got", 16) == 0) got = r;
                else if (std::strncmp(name, "__const", 16) == 0) konst = r;
            }
        } else if (cmd == macho::LC_DYLD_INFO_ONLY) {
            rebase_off = rd32(p + 8); rebase_size = rd32(p + 12);
            bind_off = rd32(p + 16); bind_size = rd32(p + 20);
        }
        p += size;
    }
    REQUIRE(text.hi > text.lo);
    REQUIRE(got.hi > got.lo);
    REQUIRE(konst.hi > konst.lo);
    CHECK(stubs.hi > stubs.lo);

    // Every load site, as the image holds it.
    for (const auto& r : loads) {
        const uint8_t* site = dylib.data() + text_fileoff + r.offset;
        const uint64_t pc = text.lo + r.offset + 4;
        const uint64_t target = static_cast<uint64_t>(static_cast<int64_t>(pc) + rdi32(site));
        if (is_import(r.symbol_name)) {
            CHECK_EQ(site[-2], uint8_t(0x8B));
            CHECK(got.holds(target));
            CHECK_EQ((target - got.lo) % 8, uint64_t(0));
        } else {
            CHECK_EQ(site[-2], uint8_t(0x8D));
            if (r.symbol_name == "helper") CHECK(text.holds(target));
            if (r.symbol_name == "table") CHECK(konst.holds(target));
        }
    }

    // Rebases: the one pointer in `table`, in __DATA_CONST, none in __TEXT.
    REQUIRE(rebase_size > 0);
    size_t rebases = 0;
    for (uint32_t i = 0; i < rebase_size;) {
        const uint8_t op = dylib[rebase_off + i];
        const uint8_t imm = op & 0x0F;
        ++i;
        if ((op & 0xF0) == 0x20) {   // SET_SEGMENT_AND_OFFSET_ULEB
            CHECK(imm != text_seg);
            while (dylib[rebase_off + i] & 0x80) ++i;
            ++i;
        } else if ((op & 0xF0) == 0x50) {   // DO_REBASE_IMM_TIMES
            rebases += imm;
        } else if (op == 0) {
            break;
        }
    }
    CHECK_EQ(rebases, size_t(1));

    // Binds: one GOT entry per import, both in __DATA_CONST.
    REQUIRE(bind_size > 0);
    size_t binds = 0;
    for (uint32_t i = 0; i < bind_size;) {
        const uint8_t op = dylib[bind_off + i];
        const uint8_t imm = op & 0x0F;
        ++i;
        switch (op & 0xF0) {
            case 0x40:   // SET_SYMBOL_TRAILING_FLAGS_IMM
                while (dylib[bind_off + i]) ++i;
                ++i;
                break;
            case 0x20:   // SET_DYLIB_ORDINAL_ULEB
            case 0x60:   // SET_ADDEND_SLEB
                while (dylib[bind_off + i] & 0x80) ++i;
                ++i;
                break;
            case 0x70:   // SET_SEGMENT_AND_OFFSET_ULEB
                CHECK(imm != text_seg);
                while (dylib[bind_off + i] & 0x80) ++i;
                ++i;
                break;
            case 0x90:
                ++binds;
                break;
            default:
                break;
        }
        if (op == 0) break;
    }
    CHECK_EQ(binds, size_t(3));
}

TEST_CASE("PIC x64 - ELF Shared Object Needs No DT_TEXTREL and Loads the Import's GOT Slot") {
    object::ObjectFile obj = build_object(Target::x64_linux());
    std::vector<object::ObjectRelocation> loads = got_loads(obj);
    std::string err;
    std::vector<uint8_t> so = AotLinker::link(obj, link_options("libext.so"), &err);
    if (so.empty()) std::fprintf(stderr, "ELF link failed: %s\n", err.c_str());
    REQUIRE(!so.empty());

    const uint64_t shoff = rd64(so.data() + 40);
    const uint16_t shnum = rd16(so.data() + 60);
    const uint16_t shstrndx = rd16(so.data() + 62);
    auto shdr = [&](uint16_t i) { return so.data() + shoff + i * 64ULL; };
    const uint8_t* shstr = so.data() + rd64(shdr(shstrndx) + 24);
    Range text, got;
    uint64_t text_off = 0;
    const uint8_t* dynamic = nullptr; uint64_t dynamic_size = 0;
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* h = shdr(i);
        const std::string name(reinterpret_cast<const char*>(shstr + rd32(h)));
        Range r{rd64(h + 16), rd64(h + 16) + rd64(h + 32)};
        if (name == ".text") { text = r; text_off = rd64(h + 24); }
        else if (name == ".got") got = r;
        else if (rd32(h + 4) == elf64::SHT_DYNAMIC) { dynamic = so.data() + rd64(h + 24); dynamic_size = rd64(h + 32); }
    }
    REQUIRE(dynamic != nullptr);
    REQUIRE(got.hi > got.lo);
    for (uint64_t off = 0; off + 16 <= dynamic_size; off += 16) {
        const int64_t tag = static_cast<int64_t>(rd64(dynamic + off));
        const uint64_t val = rd64(dynamic + off + 8);
        CHECK(tag != elf64::DT_TEXTREL);
        if (tag == elf64::DT_FLAGS) CHECK((val & elf64::DF_TEXTREL) == 0);
        if (tag == elf64::DT_NULL) break;
    }
    for (const auto& r : loads) {
        const uint8_t* site = so.data() + text_off + r.offset;
        const uint64_t target = static_cast<uint64_t>(static_cast<int64_t>(text.lo + r.offset + 4) + rdi32(site));
        if (is_import(r.symbol_name)) {
            CHECK_EQ(site[-2], uint8_t(0x8B));
            CHECK(got.holds(target));
        } else {
            CHECK_EQ(site[-2], uint8_t(0x8D));
        }
    }
}

TEST_CASE("PIC x64 - PE DLL Loads the Import's IAT Slot, Not Its Thunk") {
    object::ObjectFile obj = build_object(Target::x64_windows());
    std::vector<object::ObjectRelocation> loads = got_loads(obj);
    std::string err;
    std::vector<uint8_t> dll = AotLinker::link(obj, link_options("ext.dll"), &err);
    if (dll.empty()) std::fprintf(stderr, "PE link failed: %s\n", err.c_str());
    REQUIRE(!dll.empty());

    const uint32_t pe_off = rd32(dll.data() + 0x3C);
    const uint8_t* file_hdr = dll.data() + pe_off + 4;
    const uint16_t nsections = rd16(file_hdr + 2);
    const uint16_t opt_size = rd16(file_hdr + 16);
    const uint8_t* opt = file_hdr + 20;
    const uint8_t* dirs = opt + 112;
    const Range iat{rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IAT * 8),
                    rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IAT * 8) + rd32(dirs + pe::IMAGE_DIRECTORY_ENTRY_IAT * 8 + 4)};
    REQUIRE(iat.hi > iat.lo);
    uint32_t text_rva = 0, text_raw = 0;
    const uint8_t* sh = opt + opt_size;
    for (uint16_t i = 0; i < nsections; ++i, sh += 40) {
        if (std::strncmp(reinterpret_cast<const char*>(sh), ".text", 8) == 0) {
            text_rva = rd32(sh + 12);
            text_raw = rd32(sh + 20);
        }
    }
    REQUIRE(text_rva > 0);
    for (const auto& r : loads) {
        const uint8_t* site = dll.data() + text_raw + r.offset;
        const uint64_t target = static_cast<uint64_t>(static_cast<int64_t>(text_rva + r.offset + 4) + rdi32(site));
        if (is_import(r.symbol_name)) {
            CHECK_EQ(site[-2], uint8_t(0x8B));
            CHECK(iat.holds(target));
        } else {
            CHECK_EQ(site[-2], uint8_t(0x8D));
        }
    }
}

TEST_CASE("PIC x64 - JIT Binds an Import's Address Through Its Own Slot") {
    if (!Target::host().is_x64()) return;
    auto mod = build_module();
    object::ObjectFile obj = object::compile_module_to_object(*mod, Target::host());
    define_table(obj);

    int64_t ext_data = 7;
    // An address no mapping is anywhere near: out of a lea's reach, so its
    // load has to come through a slot; ext_data may go either way.
    void* const far_ptr = reinterpret_cast<void*>(uintptr_t{0x0000123456789AB8});
    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("ext_data", &ext_data);
    jit.register_external_symbol("ext_fn", reinterpret_cast<void*>(&ext_fn_impl));
    jit.register_external_symbol("ext_far", far_ptr);
    REQUIRE(jit.load_object(obj));
    auto probe = jit.get_function_ptr<int64_t (*)(int64_t)>("probe");
    REQUIRE(probe != nullptr);
    // 1000 + 7 + (n + 1) + 10n + tail
    CHECK_EQ(probe(0), int64_t(1008));
    CHECK_EQ(probe(2), int64_t(1032));
    CHECK_EQ(probe(5), int64_t(1163));
    ext_data = 100;
    CHECK_EQ(probe(1), int64_t(1113));   // the address of the variable, not a copy of it
    auto far_addr = jit.get_function_ptr<void* (*)()>("far_addr");
    REQUIRE(far_addr != nullptr);
    CHECK_EQ(far_addr(), far_ptr);
}
