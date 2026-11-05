// Cold code out of the hot line, and the hot line's encodings short:
// a cold trampoline survives CFG simplification, what only cold code enters
// is laid out with it, a slow call entered from many tests does not fall
// through, and block-to-block branches take their 2-byte form where they
// reach (docs in bronze: docs/aot-codesize.md).
#include "test_framework.hpp"
#include <brass/codegen/block_layout.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>

using namespace brass;
using namespace brass::codegen;

namespace {

void jmp_to(LirBlock* from, LirBlock* to) {
    auto j = std::make_unique<LirInst>(LirOpcode::Jmp);
    j->add_use(LirOperand::label(to->id));
    from->append_inst(std::move(j));
}

void branch(LirBlock* from, LirBlock* taken, LirBlock* other) {
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = x64::Condition::NE;
    jcc->add_use(LirOperand::label(taken->id));
    from->append_inst(std::move(jcc));
    jmp_to(from, other);
}

void call(LirBlock* b, const char* symbol) {
    auto c = std::make_unique<LirInst>(LirOpcode::Call);
    c->callee_symbol = symbol;
    b->append_inst(std::move(c));
}

void work(LirBlock* b) {
    auto m = std::make_unique<LirInst>(LirOpcode::Add);
    m->add_def(LirOperand::preg_gpr(x64::GPR::RAX));
    m->add_use(LirOperand::preg_gpr(x64::GPR::RCX));
    b->append_inst(std::move(m));
}

size_t index_of(const LirFunction& fn, const LirBlock* b) {
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        if (fn.blocks[i].get() == b) return i;
    }
    return fn.blocks.size();
}

bool branches_to(const BasicBlock* from, const BasicBlock* to) {
    for (const BasicBlock* s : from->successors()) {
        if (s == to) return true;
    }
    return false;
}

} // namespace

// A static guard's miss whose own receiver test a dominating one made
// redundant is an empty `br` to the fallback read. Bypassed, the guard's
// header test branched straight to the fallback, which nothing marked cold,
// and the layout let it fall through with the hit out of line (bronze's
// recv_N_obj -> recv_N_plain, 154K taken branches a frame on three_meshes).
TEST_CASE("Cold layout - a cold trampoline stays between a hot branch and its target") {
    Module mod("cold_trampoline");
    Function* fn = mod.create_function("f", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hit = b.append_block("hit");
    BasicBlock* miss = b.append_block("get_miss_cold");
    BasicBlock* fallback = b.append_block("fallback");
    BasicBlock* merge = b.append_block("merge");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* r = b.add_block_param(merge, Type::i64());
    b.position_at_end(entry);
    b.build_br_if(b.build_slt(x, b.build_iconst_i64(10)), hit, {}, miss, {});
    b.position_at_end(hit);
    b.build_br(merge, {b.build_add(x, b.build_iconst_i64(1))});
    b.position_at_end(miss);
    b.build_br(fallback);
    b.position_at_end(fallback);
    b.build_br(merge, {b.build_mul(x, b.build_iconst_i64(3))});
    b.position_at_end(merge);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    CfgSimplifyOptions opts;
    opts.enable_block_merge = false;
    cfg_simplify_function(*fn, opts);
    CHECK(verify_module(mod, &diag));
    CHECK(branches_to(entry, miss));
    CHECK(!branches_to(entry, fallback));

    // A trampoline that is not cold is still bypassed.
    Module mod2("warm_trampoline");
    Function* fn2 = mod2.create_function("g", Type::i64(), {Type::i64()});
    Builder b2(mod2);
    b2.set_function(fn2);
    BasicBlock* e2 = b2.append_block("entry");
    BasicBlock* t2 = b2.append_block("hop");
    BasicBlock* s2 = b2.append_block("target");
    BasicBlock* o2 = b2.append_block("other");
    Value* y = b2.add_block_param(e2, Type::i64());
    b2.position_at_end(e2);
    b2.build_br_if(b2.build_slt(y, b2.build_iconst_i64(10)), o2, {}, t2, {});
    b2.position_at_end(t2);
    b2.build_br(s2);
    b2.position_at_end(s2);
    b2.build_ret(y);
    b2.position_at_end(o2);
    b2.build_ret(b2.build_iconst_i64(0));
    fn2->rebuild_cfg_predecessors();
    REQUIRE(verify_module(mod2, &diag));
    cfg_simplify_function(*fn2, opts);
    CHECK(verify_module(mod2, &diag));
    CHECK(branches_to(e2, s2));
}

// The code a miss runs is named cold at its first block only; the rest (an
// inline cache read's own tests and hit) is entered from nothing but that
// block, and goes with it after the hot code rather than between traces.
TEST_CASE("Cold layout - a block only cold blocks enter is laid out with them") {
    LirFunction fn;
    fn.name = "cold_region";
    LirBlock* entry = fn.create_block("entry");
    LirBlock* miss = fn.create_block("get_miss_cold");
    LirBlock* fb_test = fn.create_block("fallback_test");
    LirBlock* fb_hit = fn.create_block("fallback_hit");
    LirBlock* hit = fn.create_block("hit");
    LirBlock* merge = fn.create_block("merge");
    branch(entry, miss, hit);
    work(miss);
    jmp_to(miss, fb_test);
    branch(fb_test, fb_hit, merge);
    work(fb_hit);
    jmp_to(fb_hit, merge);
    work(hit);
    jmp_to(hit, merge);
    merge->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(index_of(fn, entry), 0u);
    CHECK_EQ(index_of(fn, hit), 1u);
    CHECK_EQ(index_of(fn, merge), 2u);
    CHECK(index_of(fn, fb_test) > index_of(fn, miss));
    CHECK(index_of(fn, fb_hit) > index_of(fn, miss));
    CHECK(fb_test->cold);
    CHECK(fb_hit->cold);
    CHECK(!merge->cold);
}

// An inline cache stub: the object test and the shape test both refuse to
// one slow call. The call heuristic held only for a side entered once, so
// this call (entered twice) fell through and the object case jumped out of
// line (__bronze_prop_get_ic, 78K taken branches a frame).
TEST_CASE("Cold layout - a slow call entered from several tests does not fall through") {
    LirFunction fn;
    fn.name = "stub";
    LirBlock* entry = fn.create_block("entry");
    LirBlock* slow = fn.create_block("slow");
    LirBlock* object = fn.create_block("object");
    LirBlock* hit = fn.create_block("hit");
    branch(entry, object, slow);
    work(object);
    branch(object, hit, slow);
    call(slow, "slow_helper");
    slow->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    work(hit);
    hit->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(index_of(fn, object), 1u);
    CHECK_EQ(index_of(fn, hit), 2u);
    CHECK_EQ(index_of(fn, slow), 3u);
}

namespace {

// f(x): a short diamond (x < 0 ? -x : x), then a loop adding `k` to the
// result `n` times, its body padded with `pad` dependent adds so its back
// edge is short or long as `pad` says.
Function* diamond_loop(Module& mod, Builder& b, const std::string& name, int pad) {
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* neg = b.append_block("neg");
    BasicBlock* pos = b.append_block("pos");
    BasicBlock* head = b.append_block("head");
    BasicBlock* body = b.append_block("body");
    BasicBlock* done = b.append_block("done");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i64());
    Value* a = b.add_block_param(head, Type::i64());
    Value* i = b.add_block_param(head, Type::i64());
    b.position_at_end(entry);
    b.build_br_if(b.build_slt(x, b.build_iconst_i64(0)), neg, {}, pos, {});
    b.position_at_end(neg);
    b.build_br(head, {b.build_sub(b.build_iconst_i64(0), x), b.build_iconst_i64(0)});
    b.position_at_end(pos);
    b.build_br(head, {x, b.build_iconst_i64(0)});
    b.position_at_end(head);
    b.build_br_if(b.build_slt(i, n), body, {}, done, {});
    b.position_at_end(body);
    Value* acc = a;
    for (int k = 0; k < pad; ++k) acc = b.build_xor(b.build_add(acc, b.build_iconst_i64(k * 7 + 3)), i);
    b.build_br(head, {acc, b.build_add(i, b.build_iconst_i64(1))});
    b.position_at_end(done);
    b.build_ret(a);
    fn->rebuild_cfg_predecessors();
    return fn;
}

int64_t diamond_loop_ref(int64_t x, int64_t n, int pad) {
    int64_t a = x < 0 ? -x : x;
    for (int64_t i = 0; i < n; ++i) {
        for (int k = 0; k < pad; ++k) a = (a + (k * 7 + 3)) ^ i;
    }
    return a;
}

} // namespace

// Every block-to-block branch was rel32 (a jcc 6 bytes, a jmp 5); one that
// reaches is now 2. Emitted short, a branch lands on its block's first byte.
TEST_CASE("Cold layout - a branch that reaches its block is emitted short") {
    Module mod("relax");
    Builder b(mod);
    Function* fn = diamond_loop(mod, b, "f", 1);
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

    const auto& code = res.code_buffer.bytes();
    std::vector<size_t> starts;
    for (const auto& [id, off] : res.block_offsets) starts.push_back(off);
    auto is_start = [&](int64_t at) {
        for (size_t s : starts) {
            if (static_cast<int64_t>(s) == at) return true;
        }
        return false;
    };
    size_t short_to_block = 0, near_jcc = 0;
    for (size_t p = 0; p + 1 < code.size(); ++p) {
        const uint8_t op = code[p];
        if ((op >= 0x70 && op <= 0x7F) || op == 0xEB) {
            const int64_t to = static_cast<int64_t>(p) + 2 + static_cast<int8_t>(code[p + 1]);
            if (is_start(to)) ++short_to_block;
        }
        if (op == 0x0F && code[p + 1] >= 0x80 && code[p + 1] <= 0x8F) ++near_jcc;
    }
    // The diamond's test and the loop's exit test and back edge all reach.
    CHECK(short_to_block >= 2);
    CHECK_EQ(near_jcc, 0u);
}

// Short and long forms side by side, run: loop bodies from a few bytes to
// far past a rel8's reach, so the back edge and the loop's exit change form
// across them, and alignment padding moves the blocks between the passes.
TEST_CASE("Cold layout - short and long branches run the same") {
    Module mod("relax_run");
    Builder b(mod);
    const int pads[] = {0, 1, 3, 6, 9, 12, 20, 40};
    for (int pad : pads) diamond_loop(mod, b, "f" + std::to_string(pad), pad);
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    for (int pad : pads) {
        auto f = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("f" + std::to_string(pad));
        REQUIRE(f != nullptr);
        for (int64_t x : {int64_t{-5}, int64_t{0}, int64_t{17}}) {
            for (int64_t n : {int64_t{0}, int64_t{1}, int64_t{9}}) CHECK_EQ(f(x, n), diamond_loop_ref(x, n, pad));
        }
    }
}

// A cold block starts where the code before it ends: 16-byte alignment padded
// every block nothing falls into, the misses at a function's end too, which
// spread the hot code over more lines for blocks that seldom run.
TEST_CASE("Cold layout - a cold block is not aligned") {
    for (int pad = 0; pad < 16; ++pad) {
        Module mod("cold_align");
        Function* fn = mod.create_function("f", Type::i64(), {Type::i64()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hot = b.append_block("hot");
        BasicBlock* miss = b.append_block("get_miss_cold");
        Value* x = b.add_block_param(entry, Type::i64());
        b.position_at_end(entry);
        b.build_br_if(b.build_slt(x, b.build_iconst_i64(10)), hot, {}, miss, {});
        b.position_at_end(hot);
        Value* v = x;
        for (int i = 0; i < pad; ++i) v = b.build_add(v, b.build_iconst_i64(i + 1));
        b.build_ret(v);
        b.position_at_end(miss);
        b.build_ret(b.build_mul(x, b.build_iconst_i64(3)));
        fn->rebuild_cfg_predecessors();
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
        const LirBlock* cold = nullptr;
        for (const auto& lb : lir->blocks) {
            if (lb && lb->name.find("cold") != std::string::npos) cold = lb.get();
        }
        REQUIRE(cold != nullptr);
        CHECK(index_of(*lir, cold) == lir->blocks.size() - 1);
        EmitContext emit(*lir, Target::x64_linux());
        CompilationResult res = emit.compile();
        const size_t at = res.block_offsets.at(cold->id);
        REQUIRE(at > 0);
        // The hot side's `ret` ends right where the cold block begins.
        CHECK_EQ(res.code_buffer.bytes()[at - 1], 0xC3);
    }
}
