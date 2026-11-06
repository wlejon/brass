#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

// Values that live long in a function that calls: a float equality's
// `jp F; je T; jmp F` reaches F from two branches of one block, which once
// made every value live into F unsplittable (one location for its whole
// life); the edge now reads them where the first branch is and the scan
// keeps splits out from between the two. And a value read all over such a
// function (the address of a table its sites index) takes a callee-saved
// register for its whole life before the scan starts.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// Twelve values loaded off %p, live across a float equality branch into
// both arms, each of which calls and then sums them all.
std::string float_arms_text() {
    std::string s = "module @m\n\nfunc @mix(%a: i64) -> i64 {\nentry:\n  %k = iconst.i64 3\n"
                    "  %r = mul.i64 %a, %k\n  ret %r\n}\n\n"
                    "func @arms(%x: f64, %y: f64, %p: ptr) -> i64 {\nentry:\n";
    for (int i = 0; i < 12; ++i) {
        s += "  %v" + std::to_string(i) + " = load.i64 %p, " + std::to_string(i * 8) + "\n";
    }
    s += "  %c = eq.f64 %x, %y\n  br_if %c, t, f\n";
    for (const char* arm : {"t", "f"}) {
        const std::string a = arm;
        s += a + ":\n  %" + a + "c = call.i64 @mix(%v" + (a == "t" ? "0" : "1") + ")\n";
        std::string prev = "%" + a + "c";
        for (int i = 0; i < 12; ++i) {
            const std::string n = "%" + a + "s" + std::to_string(i);
            s += "  " + n + " = add.i64 " + prev + ", %v" + std::to_string(i) + "\n";
            prev = n;
        }
        s += "  br out(" + prev + ")\n";
    }
    s += "out(%r: i64):\n  ret %r\n}\n";
    return s;
}

int64_t arms_reference(double x, double y, const int64_t* p) {
    int64_t s = x == y ? p[0] * 3 : p[1] * 3;
    for (int i = 0; i < 12; ++i) s += p[i];
    return s;
}

// %p read forty times between four calls: a callee-saved register holds it
// throughout.
std::string table_text() {
    std::string s = "module @m\n\nfunc @mix(%a: i64) -> i64 {\nentry:\n  %k = iconst.i64 3\n"
                    "  %r = mul.i64 %a, %k\n  ret %r\n}\n\n"
                    "func @table(%p: ptr, %x: i64) -> i64 {\nentry:\n";
    std::string prev = "%x";
    int n = 0;
    for (int call = 0; call < 4; ++call) {
        for (int i = 0; i < 10; ++i, ++n) {
            const std::string id = std::to_string(n);
            s += "  %l" + id + " = load.i64 %p, " + std::to_string((n % 16) * 8) + "\n";
            s += "  %s" + id + " = add.i64 " + prev + ", %l" + id + "\n";
            prev = "%s" + id;
        }
        const std::string id = std::to_string(call);
        s += "  %c" + id + " = call.i64 @mix(" + prev + ")\n";
        prev = "%c" + id;
    }
    s += "  ret " + prev + "\n}\n";
    return s;
}

int64_t table_reference(const int64_t* p, int64_t x) {
    int64_t s = x;
    int n = 0;
    for (int call = 0; call < 4; ++call) {
        for (int i = 0; i < 10; ++i, ++n) s += p[n % 16];
        s *= 3;
    }
    return s;
}

} // namespace

TEST_CASE("Linear Scan - values live into both edges of a float equality branch are allocated and split") {
    const std::string text = float_arms_text();
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("arms");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    // Some block reaches one target from two branches (the parity jump and
    // the other one), which is the case this is about.
    bool two_ways = false;
    for (const auto& bb : lir->blocks) {
        for (const auto& s : bb->successors) {
            int ways = 0;
            for (const auto& inst : bb->instructions) {
                if (inst->opcode != LirOpcode::Jcc && inst->opcode != LirOpcode::Jmp) continue;
                for (const auto& u : inst->uses) ways += u.is_label() && u.label_id == s->id;
            }
            two_ways = two_ways || ways >= 2;
        }
    }
    CHECK(two_ways);
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();

    JitExecutionEngine jit;
    auto run = parse_module(text.c_str());
    REQUIRE(run != nullptr);
    REQUIRE(jit.compile_and_load(*run));
    auto f = jit.get_function_ptr<int64_t (*)(double, double, const int64_t*)>("arms");
    REQUIRE(f != nullptr);
    int64_t p[12];
    for (int i = 0; i < 12; ++i) p[i] = (i + 1) * 1000 + i * i;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double vals[] = {0.0, -0.0, 1.5, nan};
    for (double x : vals) {
        for (double y : vals) CHECK_EQ(f(x, y, p), arms_reference(x, y, p));
    }
}

TEST_CASE("Linear Scan - a value read all over a function that calls keeps a callee-saved register") {
    const std::string text = table_text();
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("table");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    // The one interval read forty times and live across calls: %p.
    size_t found = 0;
    for (const auto& iv : liveness.intervals()) {
        if (!iv.vreg.is_valid() || !iv.spans_call || iv.use_positions.size() < 40) continue;
        ++found;
        for (const UsePosition& u : iv.use_positions) {
            if (u.is_def) continue;
            const LirOperand at = regalloc.location_at(iv.vreg, u.inst_id, 8);
            REQUIRE(at.is_preg());
            const GPR g = at.preg_val.as_gpr();
            CHECK((g == GPR::RBX || g == GPR::R12));
        }
    }
    CHECK_EQ(found, size_t(1));

    JitExecutionEngine jit;
    auto run = parse_module(text.c_str());
    REQUIRE(run != nullptr);
    REQUIRE(jit.compile_and_load(*run));
    auto f = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("table");
    REQUIRE(f != nullptr);
    int64_t p[16];
    for (int i = 0; i < 16; ++i) p[i] = 7 * i - 20;
    for (int64_t x : {int64_t{0}, int64_t{5}, int64_t{-123456}}) CHECK_EQ(f(p, x), table_reference(p, x));
}
