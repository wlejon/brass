#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <cstring>
#include <vector>
#include <string>

using namespace brass;
using namespace brass::target;

namespace {

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] | (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

int64_t read_i64(const uint8_t* p) {
    return static_cast<int64_t>(read_u64(p));
}

uint32_t compute_elf_hash(const char* name) {
    uint32_t h = 0, g = 0;
    while (*name) {
        h = (h << 4) + static_cast<uint8_t>(*name++);
        g = h & 0xF0000000u;
        if (g) h ^= (g >> 24);
        h &= ~g;
    }
    return h;
}

} // namespace

TEST_CASE("ELF SO Writer - Header and Program Header Verification") {
    Module mod("test_elf_hdr");
    Function* fn = mod.create_function("simple_mult", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* prod = b.build_mul(a, c);
    b.build_ret(prod);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_linux());
    ElfSoOptions opts;
    opts.soname = "libsimple_mult.so";
    std::vector<uint8_t> so = ElfSoWriter::emit(obj, opts);

    REQUIRE(so.size() >= 64 + 5 * 56);

    // 1. ELF Header (64 bytes)
    CHECK_EQ(so[0], 0x7F);
    CHECK_EQ(so[1], 'E');
    CHECK_EQ(so[2], 'L');
    CHECK_EQ(so[3], 'F');
    CHECK_EQ(so[4], elf64::ELFCLASS64);
    CHECK_EQ(so[5], elf64::ELFDATA2LSB);
    CHECK_EQ(so[6], elf64::EV_CURRENT);

    uint16_t e_type = read_u16(so.data() + 16);
    uint16_t e_machine = read_u16(so.data() + 18);
    uint32_t e_version = read_u32(so.data() + 20);
    uint64_t e_phoff = read_u64(so.data() + 32);
    uint64_t e_shoff = read_u64(so.data() + 40);
    uint16_t e_phnum = read_u16(so.data() + 56);
    uint16_t e_shnum = read_u16(so.data() + 60);

    CHECK_EQ(e_type, elf64::ET_DYN); // 3 (Shared object file)
    CHECK_EQ(e_machine, elf64::EM_X86_64); // 62 (x86-64)
    CHECK_EQ(e_version, uint32_t(elf64::EV_CURRENT));
    CHECK_EQ(e_phoff, 64ULL); // Right after Ehdr
    CHECK(e_shoff > 0);
    CHECK_EQ(e_phnum, 5u); // PT_PHDR, PT_LOAD(R), PT_LOAD(RX), PT_LOAD(RW), PT_DYNAMIC
    CHECK(e_shnum >= 7u);

    // 2. Program Headers (5 entries, 56 bytes each)
    const uint8_t* phdrs = so.data() + e_phoff;

    bool found_phdr = false;
    bool found_load_rx = false;
    bool found_dynamic = false;

    for (uint16_t i = 0; i < e_phnum; ++i) {
        const uint8_t* ph = phdrs + i * 56;
        uint32_t p_type = read_u32(ph + 0);
        uint32_t p_flags = read_u32(ph + 4);
        uint64_t p_offset = read_u64(ph + 8);
        uint64_t p_vaddr = read_u64(ph + 16);
        uint64_t p_align = read_u64(ph + 48);

        if (p_type == elf64::PT_PHDR) {
            found_phdr = true;
            CHECK_EQ(p_flags, elf64::PF_R);
            CHECK_EQ(p_offset, 64ULL);
            CHECK_EQ(p_vaddr, 64ULL);
        } else if (p_type == elf64::PT_LOAD) {
            CHECK_EQ(p_align, 0x1000ULL); // Page aligned!
            if ((p_flags & elf64::PF_X) != 0) {
                found_load_rx = true;
                CHECK((p_flags & elf64::PF_R) != 0);
            }
        } else if (p_type == elf64::PT_DYNAMIC) {
            found_dynamic = true;
            CHECK((p_flags & elf64::PF_R) != 0);
            CHECK((p_flags & elf64::PF_W) != 0);
        }
    }

    CHECK(found_phdr);
    CHECK(found_load_rx);
    CHECK(found_dynamic);
}

TEST_CASE("ELF SO Writer - Dynamic Section Tags and DT_HASH Lookup") {
    Module mod("test_elf_dyn");
    std::vector<std::string> names = {"calc_sum", "calc_diff", "calc_prod"};

    for (const auto& name : names) {
        Function* f = mod.create_function(name, Type::i64(), {Type::i64(), Type::i64()});
        Builder b(mod);
        b.set_function(f);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        Value* res = b.build_add(x, y);
        b.build_ret(res);
        f->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f));
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_linux());
    ElfSoOptions opts;
    opts.soname = "libcalc.so.1";
    std::vector<uint8_t> so = ElfSoWriter::emit(obj, opts);

    uint64_t e_shoff = read_u64(so.data() + 40);
    uint16_t e_shnum = read_u16(so.data() + 60);
    uint16_t e_shstrndx = read_u16(so.data() + 62);

    // Locate .shstrtab
    const uint8_t* shstrtab_hdr = so.data() + e_shoff + e_shstrndx * 64;
    uint64_t shstrtab_off = read_u64(shstrtab_hdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(so.data() + shstrtab_off);

    // Find .dynamic and .hash sections
    uint64_t dyn_off = 0;
    uint64_t dyn_size = 0;
    uint64_t hash_off = 0;
    uint64_t dynstr_off = 0;
    uint64_t dynstr_size = 0;
    uint64_t dynsym_off = 0;
    uint64_t dynsym_size = 0;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* sh = so.data() + e_shoff + i * 64;
        uint32_t sh_name = read_u32(sh + 0);
        uint32_t sh_type = read_u32(sh + 4);
        uint64_t s_off = read_u64(sh + 24);
        uint64_t s_sz = read_u64(sh + 32);
        const char* sec_name = shstrtab + sh_name;

        if (sh_type == elf64::SHT_DYNAMIC) {
            dyn_off = s_off;
            dyn_size = s_sz;
        } else if (sh_type == elf64::SHT_HASH) {
            hash_off = s_off;
        } else if (sh_type == elf64::SHT_STRTAB && std::strcmp(sec_name, ".dynstr") == 0) {
            dynstr_off = s_off;
            dynstr_size = s_sz;
        } else if (sh_type == elf64::SHT_DYNSYM) {
            dynsym_off = s_off;
            dynsym_size = s_sz;
        }
    }

    REQUIRE(dyn_off > 0);
    REQUIRE(hash_off > 0);
    REQUIRE(dynstr_off > 0);
    REQUIRE(dynsym_off > 0);

    // 1. Verify .dynamic tags
    bool found_dt_hash = false;
    bool found_dt_strtab = false;
    bool found_dt_symtab = false;
    bool found_dt_strsz = false;
    bool found_dt_soname = false;
    bool found_dt_null = false;

    const char* dynstr = reinterpret_cast<const char*>(so.data() + dynstr_off);

    for (uint64_t off = 0; off < dyn_size; off += 16) {
        int64_t tag = read_i64(so.data() + dyn_off + off);
        uint64_t val = read_u64(so.data() + dyn_off + off + 8);

        if (tag == elf64::DT_HASH) found_dt_hash = true;
        else if (tag == elf64::DT_STRTAB) found_dt_strtab = true;
        else if (tag == elf64::DT_SYMTAB) found_dt_symtab = true;
        else if (tag == elf64::DT_STRSZ) {
            found_dt_strsz = true;
            CHECK_EQ(val, dynstr_size);
        } else if (tag == elf64::DT_SONAME) {
            found_dt_soname = true;
            CHECK_EQ(std::string(dynstr + val), "libcalc.so.1");
        } else if (tag == elf64::DT_NULL) {
            found_dt_null = true;
            break;
        }
    }

    CHECK(found_dt_hash);
    CHECK(found_dt_strtab);
    CHECK(found_dt_symtab);
    CHECK(found_dt_strsz);
    CHECK(found_dt_soname);
    CHECK(found_dt_null);

    // 2. Verify .hash table and DT_HASH lookups
    const uint8_t* hash_ptr = so.data() + hash_off;
    uint32_t nbucket = read_u32(hash_ptr + 0);
    uint32_t nchain = read_u32(hash_ptr + 4);

    CHECK(nbucket > 0);
    CHECK_EQ(nchain, static_cast<uint32_t>(dynsym_size / 24));

    const uint32_t* buckets = reinterpret_cast<const uint32_t*>(hash_ptr + 8);
    const uint32_t* chains = buckets + nbucket;

    // Verify each function can be found by standard ELF hash lookup!
    for (const auto& name : names) {
        uint32_t h = compute_elf_hash(name.c_str()) % nbucket;
        uint32_t sym_idx = buckets[h];
        bool found = false;

        while (sym_idx != 0 && sym_idx < nchain) {
            const uint8_t* sym = so.data() + dynsym_off + sym_idx * 24;
            uint32_t st_name = read_u32(sym + 0);
            const char* sym_name = dynstr + st_name;
            if (name == sym_name) {
                found = true;
                break;
            }
            sym_idx = chains[sym_idx];
        }
        CHECK(found);
    }
}

TEST_CASE("ELF SO Writer - DynSym and DynStr String Table Indices") {
    Module mod("test_elf_syms");
    Function* f1 = mod.create_function("first_kernel", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("second_kernel", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(x);
        f1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f1));
    }
    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(x);
        f2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f2));
    }

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::x64_linux());
    std::vector<uint8_t> so = ElfSoWriter::emit(obj);

    uint64_t e_shoff = read_u64(so.data() + 40);
    uint16_t e_shnum = read_u16(so.data() + 60);
    uint16_t e_shstrndx = read_u16(so.data() + 62);

    const uint8_t* shstrtab_hdr = so.data() + e_shoff + e_shstrndx * 64;
    uint64_t shstrtab_off = read_u64(shstrtab_hdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(so.data() + shstrtab_off);

    uint64_t dynstr_off = 0;
    uint64_t dynsym_off = 0;
    uint64_t dynsym_size = 0;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* sh = so.data() + e_shoff + i * 64;
        uint32_t sh_name = read_u32(sh + 0);
        uint32_t sh_type = read_u32(sh + 4);
        const char* sec_name = shstrtab + sh_name;

        if (sh_type == elf64::SHT_STRTAB && std::strcmp(sec_name, ".dynstr") == 0) {
            dynstr_off = read_u64(sh + 24);
        } else if (sh_type == elf64::SHT_DYNSYM) {
            dynsym_off = read_u64(sh + 24);
            dynsym_size = read_u64(sh + 32);
        }
    }

    REQUIRE(dynstr_off > 0);
    REQUIRE(dynsym_off > 0);

    const char* dynstr = reinterpret_cast<const char*>(so.data() + dynstr_off);
    size_t num_syms = dynsym_size / 24;
    CHECK(num_syms >= 3); // NULL, first_kernel, second_kernel

    // Symbol 0 must be all zeroes
    const uint8_t* sym0 = so.data() + dynsym_off;
    for (int i = 0; i < 24; ++i) {
        CHECK_EQ(sym0[i], uint8_t(0));
    }

    // Symbols 1..N
    std::vector<std::string> discovered_names;
    for (size_t i = 1; i < num_syms; ++i) {
        const uint8_t* sym = so.data() + dynsym_off + i * 24;
        uint32_t st_name = read_u32(sym + 0);
        uint8_t st_info = sym[4];
        uint16_t st_shndx = read_u16(sym + 6);
        uint64_t st_value = read_u64(sym + 8);
        uint64_t st_size = read_u64(sym + 16);

        const char* name = dynstr + st_name;
        discovered_names.push_back(std::string(name));

        uint8_t binding = st_info >> 4;
        uint8_t type = st_info & 0xF;
        CHECK_EQ(binding, elf64::STB_GLOBAL);
        CHECK_EQ(type, elf64::STT_FUNC);
        CHECK(st_shndx > 0);
        CHECK(st_value > 0);
        CHECK(st_size > 0);
    }

    bool has_f1 = std::find(discovered_names.begin(), discovered_names.end(), "first_kernel") != discovered_names.end();
    bool has_f2 = std::find(discovered_names.begin(), discovered_names.end(), "second_kernel") != discovered_names.end();
    CHECK(has_f1);
    CHECK(has_f2);
}
