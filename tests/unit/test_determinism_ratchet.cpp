#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <cstring>
#include <vector>
#include <random>

using namespace brass;
using namespace brass::object;

namespace {

void scramble_heap() {
    static std::mt19937_64 rng(0x123456789ABCDEF0ULL);
    std::vector<std::vector<uint8_t>> garbage;
    for (int i = 0; i < 50; ++i) {
        size_t sz = static_cast<size_t>(rng() % 4096) + 16;
        garbage.emplace_back(sz, static_cast<uint8_t>(rng() & 0xFF));
    }
}

std::unique_ptr<Module> build_complex_numeric_module() {
    auto mod = std::make_unique<Module>("numeric_determinism");
    Builder b(*mod);

    // 1. Math and Bitwise Function
    Function* fn1 = mod->create_function("math_kernel", Type::i64(), {Type::i64(), Type::i64(), Type::f64()});
    b.set_function(fn1);
    BasicBlock* entry1 = b.append_block("entry");
    Value* x = b.add_block_param(entry1, Type::i64());
    Value* y = b.add_block_param(entry1, Type::i64());
    Value* z = b.add_block_param(entry1, Type::f64());

    Value* sum = b.build_add(x, y);
    Value* diff = b.build_sub(x, y);
    Value* prod = b.build_mul(sum, diff);
    Value* xor_v = b.build_xor(prod, x);
    Value* popc = b.build_popcnt(xor_v);

    Value* fprod = b.build_mul(z, z);
    Value* f_as_i = b.build_fptosi_i64(fprod);
    Value* final_v = b.build_add(popc, f_as_i);
    b.build_ret(final_v);
    fn1->rebuild_cfg_predecessors();

    // 2. Loop Function with Block Parameters
    Function* fn2 = mod->create_function("loop_kernel", Type::i64(), {Type::i64()});
    b.set_function(fn2);
    BasicBlock* entry2 = b.append_block("entry");
    Value* n = b.add_block_param(entry2, Type::i64());

    BasicBlock* loop_header = b.create_block("loop_header");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* init_i = b.build_iconst_i64(0);
    Value* init_acc = b.build_iconst_i64(1);
    b.build_br(loop_header, {init_i, init_acc});

    fn2->append_block(loop_header);
    b.position_at_end(loop_header);
    Value* cur_i = b.add_block_param(loop_header, Type::i64());
    Value* cur_acc = b.add_block_param(loop_header, Type::i64());
    Value* cmp = b.build_slt(cur_i, n);
    b.build_br_if(cmp, loop_body, {}, exit_bb, {cur_acc});

    fn2->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(cur_i, one);
    Value* three = b.build_iconst_i64(3);
    Value* next_acc = b.build_add(b.build_mul(cur_acc, three), cur_i);
    b.build_br(loop_header, {next_i, next_acc});

    fn2->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* exit_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(exit_res);
    fn2->rebuild_cfg_predecessors();

    return mod;
}

std::unique_ptr<Module> build_gc_and_speculation_module() {
    auto mod = std::make_unique<Module>("gc_spec_determinism");
    mod->add_external_symbol("brass_gc_alloc");
    mod->add_external_symbol("brass_gc_safepoint");
    mod->add_external_symbol("fallback_twin");
    Builder b(*mod);

    // 1. GC Safepoint and Allocation Function
    Function* fn_gc = mod->create_function("gc_worker", Type::i64(), {Type::gcref(), Type::i64()});
    b.set_function(fn_gc);
    BasicBlock* entry = b.append_block("entry");
    Value* ref = b.add_block_param(entry, Type::gcref());
    Value* count = b.add_block_param(entry, Type::i64());

    Value* loaded = b.build_load(Type::i64(), ref, 8);
    b.build_safepoint();
    Value* stored_val = b.build_add(loaded, count);
    b.build_store(Type::i64(), ref, 8, stored_val);
    b.build_ret(stored_val);
    fn_gc->rebuild_cfg_predecessors();

    // 2. Speculative Function with Guard, Patchable Const, and Resume Point
    Function* fn_spec = mod->create_function("spec_kernel", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn_spec);
    BasicBlock* spec_entry = b.append_block("entry");
    Value* a = b.add_block_param(spec_entry, Type::i64());
    Value* c = b.add_block_param(spec_entry, Type::i64());

    Value* pconst = b.build_patchable_const_i32("ic_type_tag", 100);
    Value* pconst64 = b.build_zext_i64(pconst);
    Value* cond = b.build_eq(a, pconst64);

    b.build_guard(cond, "deopt_stub_1", {a, c, pconst64});
    Value* fast_res = b.build_mul(a, c);
    b.build_ret(fast_res);

    BasicBlock* resume_bb = b.append_block("resume_target");
    b.position_at_end(resume_bb);
    b.build_resume_point(10);
    b.build_ret(c);

    fn_spec->add_resume_point(10, resume_bb);
    fn_spec->rebuild_cfg_predecessors();

    return mod;
}

} // namespace

TEST_CASE("Determinism Ratchet - Numeric & CFG Module (COFF & ELF64)") {
    std::vector<uint8_t> first_coff;
    std::vector<uint8_t> first_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = build_complex_numeric_module();
        CHECK(verify_module(*mod));

        // COFF Win64 Emission
        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        // ELF64 Linux Emission
        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            first_coff = std::move(coff_bytes);
            first_elf = std::move(elf_bytes);
            CHECK(!first_coff.empty());
            CHECK(!first_elf.empty());
        } else {
            REQUIRE_EQ(coff_bytes.size(), first_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), first_coff.data(), first_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), first_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), first_elf.data(), first_elf.size()) == 0);
        }
    }
}

TEST_CASE("Determinism Ratchet - GC & Speculation Module (COFF & ELF64)") {
    std::vector<uint8_t> first_coff;
    std::vector<uint8_t> first_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = build_gc_and_speculation_module();
        CHECK(verify_module(*mod));

        // COFF Win64 Emission
        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        // ELF64 Linux Emission
        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            first_coff = std::move(coff_bytes);
            first_elf = std::move(elf_bytes);
            CHECK(!first_coff.empty());
            CHECK(!first_elf.empty());
        } else {
            REQUIRE_EQ(coff_bytes.size(), first_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), first_coff.data(), first_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), first_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), first_elf.data(), first_elf.size()) == 0);
        }
    }
}

TEST_CASE("Determinism Ratchet - Text Parser Roundtrip Compilation") {
    const char* kMirText = R"(module @parsed_determinism

func @fibonacci(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 2
  %2 = slt.i64 %0, %1
  br_if %2, bb_base, bb_rec

bb_base:
  ret %0

bb_rec:
  %3 = iconst.i64 1
  %4 = sub.i64 %0, %3
  %5 = call.i64 @fibonacci(%4)
  %6 = iconst.i64 2
  %7 = sub.i64 %0, %6
  %8 = call.i64 @fibonacci(%7)
  %9 = add.i64 %5, %8
  ret %9
}
)";

    std::vector<uint8_t> base_coff;
    std::vector<uint8_t> base_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = parse_module(kMirText);
        REQUIRE(mod != nullptr);
        CHECK(verify_module(*mod));

        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            base_coff = std::move(coff_bytes);
            base_elf = std::move(elf_bytes);
        } else {
            REQUIRE_EQ(coff_bytes.size(), base_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), base_coff.data(), base_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), base_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), base_elf.data(), base_elf.size()) == 0);
        }
    }
}
