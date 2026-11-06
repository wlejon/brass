// Branch relays (codegen/branch_relay.hpp): a long branch reaches its target
// through a nearby branch to the same target that is sure to be taken from
// where it stands, in two bytes instead of six.
#include "test_framework.hpp"
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/branch_relay.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>

using namespace brass;
using namespace brass::codegen;
using x64::Condition;

TEST_CASE("Branch relay - a condition implies the ones its flags always satisfy") {
    // Itself and the unconditional jump, always.
    for (uint8_t c = 0; c < 16; ++c) {
        CHECK(condition_implies(static_cast<Condition>(c), static_cast<Condition>(c)));
        CHECK(condition_implies(static_cast<Condition>(c), Condition::None));
    }
    CHECK(condition_implies(Condition::None, Condition::None));
    CHECK(!condition_implies(Condition::None, Condition::E));
    CHECK(condition_implies(Condition::A, Condition::AE));
    CHECK(condition_implies(Condition::A, Condition::NE));
    CHECK(condition_implies(Condition::G, Condition::GE));
    CHECK(condition_implies(Condition::G, Condition::NE));
    CHECK(condition_implies(Condition::E, Condition::BE));
    CHECK(condition_implies(Condition::E, Condition::LE));
    CHECK(condition_implies(Condition::B, Condition::BE));
    CHECK(condition_implies(Condition::L, Condition::LE));
    // ZF=1 with CF=1 is a flags state (an add carrying out to zero), so an
    // equal branch does not reach through an above-or-equal one.
    CHECK(!condition_implies(Condition::E, Condition::AE));
    CHECK(!condition_implies(Condition::E, Condition::GE));
    CHECK(!condition_implies(Condition::AE, Condition::A));
    CHECK(!condition_implies(Condition::NE, Condition::A));
    CHECK(!condition_implies(Condition::B, Condition::NE));
    CHECK(!condition_implies(Condition::BE, Condition::B));
    // Each with its inverse: never.
    for (uint8_t c = 0; c < 16; ++c) {
        CHECK(!condition_implies(static_cast<Condition>(c), x64::invert(static_cast<Condition>(c))));
    }
}

TEST_CASE("Branch relay - the plan relays long branches onto near ones to the same target") {
    std::vector<BranchSite> s = {
        {Condition::AE, 9, 0, true},     // 0: hub for 1 (1's `ja` implies `jae`)
        {Condition::A, 9, 40, true},     // 1: relays back to 0
        {Condition::NE, 9, 160, true},   // 2: no `jne` near: stays long
        {Condition::AE, 8, 170, true},   // 3: another target: nothing to relay to
        {Condition::A, 9, 400, true},    // 4: too far from 0 and 1; 5 is near
        {Condition::NE, 9, 450, true},   // 5: hub for 4 (`ja` implies `jne`)
        {Condition::E, 9, 470, false},   // 6: short already: stays
    };
    std::vector<uint32_t> r = plan_branch_relays(s, 104);
    REQUIRE_EQ(r.size(), s.size());
    CHECK_EQ(r[0], UINT32_MAX);
    CHECK_EQ(r[1], 0u);
    CHECK_EQ(r[2], UINT32_MAX);
    CHECK_EQ(r[3], UINT32_MAX);
    CHECK_EQ(r[4], 5u);
    CHECK_EQ(r[5], UINT32_MAX);
    CHECK_EQ(r[6], UINT32_MAX);
    // A hub never relays itself, so no relay chains or cycles.
    for (uint32_t i = 0; i < r.size(); ++i) {
        if (r[i] != UINT32_MAX) CHECK_EQ(r[r[i]], UINT32_MAX);
    }
}

namespace {

// f(x, y): four guards that miss to one cold block after a long hot body
// (`x <u 100`, `y <=u 50`, `x != y`, `y >u 3`), so every miss branch is far
// from its target and near the others.
Function* guarded(Module& mod, Builder& b) {
    Function* fn = mod.create_function("f", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* g2 = b.append_block("g2");
    BasicBlock* g3 = b.append_block("g3");
    BasicBlock* g4 = b.append_block("g4");
    BasicBlock* body = b.append_block("body");
    BasicBlock* miss = b.append_block("guard_miss_cold");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);
    b.build_br_if(b.build_ult(x, b.build_iconst_i64(100)), g2, {}, miss, {});
    b.position_at_end(g2);
    b.build_br_if(b.build_ule(y, b.build_iconst_i64(50)), g3, {}, miss, {});
    b.position_at_end(g3);
    b.build_br_if(b.build_ne(x, y), g4, {}, miss, {});
    b.position_at_end(g4);
    b.build_br_if(b.build_ugt(y, b.build_iconst_i64(3)), body, {}, miss, {});
    b.position_at_end(body);
    Value* acc = x;
    for (int k = 0; k < 48; ++k) acc = b.build_xor(b.build_add(acc, b.build_iconst_i64(k * 7919 + 3)), y);
    b.build_ret(acc);
    b.position_at_end(miss);
    b.build_ret(b.build_add(b.build_mul(x, b.build_iconst_i64(3)), y));
    fn->rebuild_cfg_predecessors();
    return fn;
}

int64_t guarded_ref(int64_t x, int64_t y) {
    const uint64_t ux = static_cast<uint64_t>(x), uy = static_cast<uint64_t>(y);
    if (!(ux < 100) || !(uy <= 50) || x == y || !(uy > 3)) return x * 3 + y;
    int64_t acc = x;
    for (int k = 0; k < 48; ++k) acc = (acc + (k * 7919 + 3)) ^ y;
    return acc;
}

} // namespace

TEST_CASE("Branch relay - guards missing to one far block relay through each other") {
    Module mod("relay");
    Builder b(mod);
    Function* fn = guarded(mod, b);
    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    optimize_block_layout(*lir);
    EmitContext emit(*lir, Target::x64_linux());
    CompilationResult res = emit.compile();
    REQUIRE(!res.code_buffer.has_unresolved_labels());

    // A relay is a 2-byte jcc landing on the first byte of another branch.
    const auto& code = res.code_buffer.bytes();
    auto is_branch_at = [&](int64_t at) {
        if (at < 0 || static_cast<size_t>(at) + 1 >= code.size()) return false;
        const uint8_t op = code[static_cast<size_t>(at)];
        if (op == 0xE9 || op == 0xEB || (op >= 0x70 && op <= 0x7F)) return true;
        return op == 0x0F && code[static_cast<size_t>(at) + 1] >= 0x80 && code[static_cast<size_t>(at) + 1] <= 0x8F;
    };
    size_t relays = 0, near_jcc = 0;
    for (size_t p = 0; p + 1 < code.size(); ++p) {
        const uint8_t op = code[p];
        if (op >= 0x70 && op <= 0x7F) {
            const int64_t to = static_cast<int64_t>(p) + 2 + static_cast<int8_t>(code[p + 1]);
            if (is_branch_at(to)) ++relays;
        }
        if (op == 0x0F && code[p + 1] >= 0x80 && code[p + 1] <= 0x8F) ++near_jcc;
    }
    // Of the four misses at least two relay (a `ja` through the `jae`, an
    // `je` through the `jbe`), whichever way isel orders the compares.
    CHECK(relays >= 2);
    CHECK(near_jcc <= 2);
}

TEST_CASE("Branch relay - every guard's miss runs the miss block") {
    Module mod("relay_run");
    Builder b(mod);
    guarded(mod, b);
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("f");
    REQUIRE(f != nullptr);
    const int64_t xs[] = {0, 3, 7, 50, 99, 100, 1000, -1};
    const int64_t ys[] = {0, 3, 4, 7, 50, 51, 99, -2};
    for (int64_t x : xs) {
        for (int64_t y : ys) CHECK_EQ(f(x, y), guarded_ref(x, y));
    }
}
