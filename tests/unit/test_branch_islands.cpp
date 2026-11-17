// Branch islands (function_codegen.hpp, BranchIslands): an AArch64 module
// whose code is larger than a direct branch reaches gets veneers between its
// functions, and a call out of reach goes through the nearest one. Exercised
// on a small module with the reach set small (ModuleCompiler::
// set_branch_reach): the far call loses its relocation, the near ones keep
// theirs, x64 and a module in reach are untouched, and on an AArch64 Mac the
// linked dylib calls through the island and gets the callee's answer.
#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;

namespace {

constexpr int kPads = 40;

// leaf(x) = x + 1, then kPads functions of a small loop each, then
// caller(x) = leaf(x) * 2 and near(x) = caller(x) + 3 right after it.
std::unique_ptr<Module> make_module() {
    auto mod = std::make_unique<Module>("islands");
    Builder b(*mod);
    {
        Function* fn = mod->create_function("leaf", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(1)));
    }
    for (int k = 0; k < kPads; ++k) {
        Function* fn = mod->create_function("pad" + std::to_string(k), Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());
        BasicBlock* loop = b.create_block("loop");
        BasicBlock* done = b.create_block("done");
        b.build_br(loop, {b.build_iconst_i64(0), b.build_iconst_i64(k)});
        fn->append_block(loop);
        b.position_at_end(loop);
        Value* i = b.add_block_param(loop, Type::i64());
        Value* acc = b.add_block_param(loop, Type::i64());
        Value* next_acc = b.build_add(b.build_mul(acc, b.build_iconst_i64(31)), i);
        Value* next_i = b.build_add(i, b.build_iconst_i64(1));
        b.build_br_if(b.build_slt(next_i, n), loop, {next_i, next_acc}, done, {next_acc});
        fn->append_block(done);
        b.position_at_end(done);
        b.build_ret(b.add_block_param(done, Type::i64()));
        fn->rebuild_cfg_predecessors();
    }
    {
        Function* fn = mod->create_function("caller", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* r = b.build_call("leaf", Type::i64(), {x});
        b.build_ret(b.build_mul(r, b.build_iconst_i64(2)));
    }
    {
        Function* fn = mod->create_function("near", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* r = b.build_call("caller", Type::i64(), {x});
        b.build_ret(b.build_add(r, b.build_iconst_i64(3)));
    }
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

object::ObjectFile compile(const Module& mod, const Target& target, uint64_t reach) {
    object::ModuleCompiler compiler(target);
    compiler.set_threads(1);
    compiler.set_branch_reach(reach);
    return compiler.compile(mod);
}

bool has_call_reloc(const object::ObjectFile& obj, const std::string& callee) {
    const object::Section* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    for (const object::ObjectRelocation& r : text->relocations) {
        if (r.kind == object::RelocKind::Plt32 && r.symbol_name == callee) return true;
    }
    return false;
}

} // namespace

TEST_CASE("branch islands: a call out of reach goes through a veneer, a near one stays direct") {
    auto mod = make_module();
    const Target target = Target::aarch64_macos();
    const object::ObjectFile plain = compile(*mod, target, uint64_t{1} << 27);
    const object::ObjectFile islanded = compile(*mod, target, 1024);

    // In reach: placed as ever, both calls by relocation.
    CHECK(has_call_reloc(plain, "leaf"));
    CHECK(has_call_reloc(plain, "caller"));
    // caller -> leaf crosses the pads (several KB): encoded to a veneer.
    // near -> caller is a few bytes: still a relocation.
    CHECK_FALSE(has_call_reloc(islanded, "leaf"));
    CHECK(has_call_reloc(islanded, "caller"));
    CHECK(islanded.get_section(".text")->data.size() > plain.get_section(".text")->data.size());
    // The veneers' loads of their targets.
    bool veneer = false;
    for (const object::ObjectRelocation& r : islanded.get_section(".text")->relocations) {
        veneer = veneer || (r.kind == object::RelocKind::GotPage21 && r.symbol_name == "leaf");
    }
    CHECK(veneer);

    // x64's rel32 reaches 2 GB: nothing changes there at any reach.
    const object::ObjectFile x64 = compile(*mod, Target::x64_linux(), 1024);
    CHECK(has_call_reloc(x64, "leaf"));
}

TEST_CASE("branch islands: the linked dylib calls through its island") {
    const Target host = Target::host();
    if (!host.is_aarch64() || !host.is_macos()) return;
    auto mod = make_module();
    const object::ObjectFile obj = compile(*mod, host, 1024);
    REQUIRE_FALSE(has_call_reloc(obj, "leaf"));
    std::string err;
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "brass_branch_islands.dylib";
    // Linked, written and signed as an AOT image is (AotLinker::link_to_file).
    const bool linked = target::AotLinker::link_to_file(obj, path.string(), target::LinkerOptions{}, &err);
    if (!linked) std::cerr << "link: " << err << "\n";
    REQUIRE(linked);
    auto lib = target::DynamicLibrary::open(path.string(), &err);
    if (lib == nullptr) std::cerr << "open: " << err << "\n";
    REQUIRE(lib != nullptr);
    using Fn = int64_t (*)(int64_t);
    auto caller = reinterpret_cast<Fn>(lib->get_symbol("caller"));
    auto near = reinterpret_cast<Fn>(lib->get_symbol("near"));
    REQUIRE(caller != nullptr);
    REQUIRE(near != nullptr);
    CHECK(caller(5) == 12);
    CHECK(near(5) == 15);
    lib.reset();
    std::filesystem::remove(path);
}
