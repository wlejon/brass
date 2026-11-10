// Function placement (object_writer.hpp, FunctionPlacement): a front end's
// hot functions lead .text in the order it gives, its cold ones close it, the rest keep module order; every
// function's own bytes and relocations are what they are without placement,
// every object format carries the placed layout, and the object is the same
// at any thread count.
#include "test_framework.hpp"
#include <brass/fuzz/program_generator.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> make_module(size_t programs) {
    auto mod = std::make_unique<Module>("placement");
    fuzz::ProgramGeneratorOptions o;
    o.max_depth = 3;
    o.max_statements = 30;
    fuzz::ProgramGenerator gen(o);
    for (size_t k = 0; k < programs; ++k) {
        gen.generate(*mod, "pl" + std::to_string(k), 0x2545f4914f6cdd1dull * (k + 1));
    }
    DiagnosticReporter diag;
    if (!verify_module(*mod, &diag)) throw std::runtime_error(diag.format_all());
    return mod;
}

std::vector<std::string> names_of(const Module& mod) {
    std::vector<std::string> out;
    for (const Function* fn : mod.functions()) {
        if (fn) out.emplace_back(fn->name());
    }
    return out;
}

object::ObjectFile compile(const Module& mod, const Target& target, const object::FunctionPlacement& placement,
                           size_t threads) {
    object::ModuleCompiler compiler(target);
    compiler.set_threads(threads);
    compiler.set_min_blocks_per_thread(1);
    compiler.set_placement(placement);
    return compiler.compile(mod);
}

std::vector<uint8_t> emit(const object::ObjectFile& obj) {
    if (obj.target.is_windows()) return object::emit_coff_object(obj);
    if (obj.target.is_macos()) return object::emit_macho_object(obj);
    return object::emit_elf_object(obj);
}

// A function's code and its relocations, relative to its first byte.
struct Body {
    std::vector<uint8_t> code;
    std::vector<std::pair<size_t, std::string>> relocs;
    bool operator==(const Body&) const = default;
};

std::map<std::string, Body> bodies(const object::ObjectFile& obj) {
    std::map<std::string, Body> out;
    const object::Section* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    for (const object::CompiledFunctionInfo& f : obj.functions) {
        Body& b = out[f.name];
        b.code.assign(text->data.begin() + static_cast<std::ptrdiff_t>(f.text_offset),
                      text->data.begin() + static_cast<std::ptrdiff_t>(f.text_offset + f.text_size));
        for (const object::ObjectRelocation& r : text->relocations) {
            if (r.offset >= f.text_offset && r.offset < f.text_offset + f.text_size) {
                b.relocs.emplace_back(r.offset - f.text_offset, r.symbol_name);
            }
        }
    }
    return out;
}

} // namespace

TEST_CASE("placement: hot functions lead in the given order, cold ones close, the rest keep module order") {
    std::unique_ptr<Module> mod = make_module(24);
    const std::vector<std::string> names = names_of(*mod);
    REQUIRE(names.size() >= 24u);
    object::FunctionPlacement placement;
    // Out of module order, with a name the module lacks and a repeat.
    placement.hot = {names[17], names[3], "no_such_function", names[9], names[3]};
    placement.cold = {names[0], names[9], names[5]};

    std::vector<std::string> expected = {names[17], names[3], names[9]};
    for (const std::string& n : names) {
        if (n != names[17] && n != names[3] && n != names[9] && n != names[0] && n != names[5]) expected.push_back(n);
    }
    expected.push_back(names[0]);
    expected.push_back(names[5]);

    for (const Target& target : {Target::x64_linux(), Target::x64_windows(), Target::x64_macos(),
                                 Target::aarch64_linux(), Target::aarch64_macos()}) {
        const object::ObjectFile plain = compile(*mod, target, {}, 1);
        const object::ObjectFile placed = compile(*mod, target, placement, 1);
        REQUIRE_EQ(placed.functions.size(), expected.size());
        size_t last = 0;
        for (size_t k = 0; k < expected.size(); ++k) {
            const object::CompiledFunctionInfo& f = placed.functions[k];
            CHECK(f.name == expected[k]);
            CHECK(k == 0 || f.text_offset > last);
            last = f.text_offset;
            CHECK_EQ(f.text_offset % (target.arch() == Arch::x64 ? 32u : 16u), size_t{0});
            const object::ObjectSymbol* sym = placed.find_symbol(f.name);
            REQUIRE(sym != nullptr);
            CHECK_EQ(sym->value, static_cast<uint64_t>(f.text_offset));
        }
        // Placement only moves code.
        CHECK(bodies(placed) == bodies(plain));
        // Every format writes the placed object.
        CHECK(emit(placed).size() > 1000u);
        // No placement is module order.
        for (size_t k = 0; k < names.size(); ++k) CHECK(plain.functions[k].name == names[k]);
    }
}

TEST_CASE("placement: the placed object is byte-identical at any thread count") {
    std::unique_ptr<Module> mod = make_module(120);
    const std::vector<std::string> names = names_of(*mod);
    object::FunctionPlacement placement;
    for (size_t i = names.size(); i-- > 0;) {
        if (i % 3 == 0) placement.hot.push_back(names[i]);
        else if (i % 7 == 0) placement.cold.push_back(names[i]);
    }
    for (const Target& target : {Target::x64_linux(), Target::aarch64_macos()}) {
        const std::vector<uint8_t> serial = emit(compile(*mod, target, placement, 1));
        CHECK(serial.size() > 50000u);
        for (size_t threads : {size_t{2}, size_t{8}, size_t{32}}) {
            CHECK(emit(compile(*mod, target, placement, threads)) == serial);
        }
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
