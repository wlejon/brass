#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>
#include <cstring>

// A raw word and the tagged value made of it share one register where the
// word is made in the tagged value's block with no GC point between
// (x64_isel_mem.cpp, name_tagged_sources): a join's number test reads the raw
// word after the tagged value stands, and as two registers the copy between
// them could not be coalesced.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// f: a join receives a word, makes the tagged value, tests the raw word for a
//    number and reads the tagged value's bits on both arms.
// g: the word is made, a call runs, and only then is it tagged: two registers.
const char* kSrc = R"(module @m
func @id(%a: i64) -> i64 {
entry:
  ret %a
}
func @f(%c: i64, %a: i64, %b: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %t = ne.i64 %c, %zero
  br_if %t, left, right
left:
  br join(%a)
right:
  br join(%b)
join(%x: i64):
  %tv = bitcast.tagged.i64 %x
  %k = iconst.i64 -4503599627370496
  %num = ule.i64 %x, %k
  br_if %num, fast, slow
fast:
  %bits = bitcast.i64.tagged %tv
  %d = bitcast.f64.i64 %bits
  %s = add.f64 %d, %d
  %r = bitcast.i64.f64 %s
  ret %r
slow:
  %bits2 = bitcast.i64.tagged %tv
  %one = iconst.i64 1
  %r2 = add.i64 %bits2, %one
  ret %r2
}
func @g(%x: i64) -> i64 {
entry:
  %one = iconst.i64 1
  %y = add.i64 %x, %one
  %z = call.i64 @id(%x)
  %tv = bitcast.tagged.i64 %y
  %bits = bitcast.i64.tagged %tv
  %r = add.i64 %bits, %z
  ret %r
}
)";

// The plain register-to-register moves of a block that define a tagged
// register.
size_t tagged_copies(const LirFunction& lir, const std::string& block) {
    size_t n = 0;
    for (const auto& bb : lir.blocks) {
        if (bb->name != block) continue;
        for (const auto& inst : bb->instructions) {
            if (inst->opcode != LirOpcode::Mov || inst->defs.size() != 1 || inst->uses.size() != 1) continue;
            if (!inst->defs[0].is_vreg() || !inst->uses[0].is_vreg()) continue;
            n += inst->defs[0].vreg_val.is_tagged ? 1 : 0;
        }
    }
    return n;
}

std::unique_ptr<LirFunction> lower(Module& mod, const char* name) {
    for (Function* fn : mod.functions()) {
        if (fn->name() != name) continue;
        fn->rebuild_cfg_predecessors();
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        return isel.lower(*fn);
    }
    return nullptr;
}

}  // namespace

TEST_CASE("x64 tagged source - a join's word and its tagged value are one register") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    auto f = lower(*mod, "f");
    REQUIRE(f != nullptr);
    CHECK_EQ(tagged_copies(*f, "join"), 0ULL);
}

TEST_CASE("x64 tagged source - a word made before a GC point keeps its own register") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    auto g = lower(*mod, "g");
    REQUIRE(g != nullptr);
    CHECK_EQ(tagged_copies(*g, "entry"), 1ULL);
}

TEST_CASE("x64 tagged source - the shared register answers through the JIT") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("f");
    auto g = jit.get_function_ptr<int64_t (*)(int64_t)>("g");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    const double half = 1.5;
    int64_t half_bits = 0;
    std::memcpy(&half_bits, &half, sizeof half_bits);
    const double three = 3.0;
    int64_t three_bits = 0;
    std::memcpy(&three_bits, &three, sizeof three_bits);
    const int64_t boxed = static_cast<int64_t>(0xFFF1000000001234ULL);
    CHECK_EQ(f(1, half_bits, boxed), three_bits);
    CHECK_EQ(f(0, half_bits, boxed), boxed + 1);
    CHECK_EQ(f(0, boxed, half_bits), three_bits);
    CHECK_EQ(g(41), 42 + 41);
}
