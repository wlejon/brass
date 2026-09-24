#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <memory>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

namespace {

// @f has a resume table; @get_bias a patchable constant; @try_catch catches
// what @fail throws, which needs the code's global exception mappings.
const char* kMoveSrc = R"(module @engine_move
func @f(%n: i64) -> i64 {
bb0:
  %lim = iconst.i64 150
  %ok = slt %n, %lim
  guard %ok, @exit_stub, [%n]
  ret %n
bres(%a: i64):
  %k = iconst.i64 1000
  %m = mul %a, %k
  ret %m
resume_table {
  entry 0 -> bres
}
}
func @exit_stub(%a: i64) -> i64 {
b0:
  ret %a
}
)";

std::unique_ptr<Module> build_move_module() {
    DiagnosticReporter diag;
    auto mod = parse_module(kMoveSrc, &diag);
    if (!mod) return nullptr;
    Builder b(*mod);

    Function* bias = mod->create_function("get_bias", Type::i64(), {Type::i64()});
    b.set_function(bias);
    BasicBlock* bias_entry = b.append_block("entry");
    b.position_at_end(bias_entry);
    Value* bx = b.add_block_param(bias_entry, Type::i64());
    Value* k = b.build_patchable_const_i64("bias_site", 10);
    b.build_ret(b.build_add(bx, k));

    Function* callee = mod->create_function("fail", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_bb = b.append_block("entry");
    b.position_at_end(c_bb);
    b.build_throw(b.add_block_param(c_bb, Type::i64()));
    callee->rebuild_cfg_predecessors();

    Function* caller = mod->create_function("try_catch", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("fail", Type::i64(), {x}, normal_bb, unwind_bb);
    b.position_at_end(normal_bb);
    b.build_ret(inv->result());
    b.position_at_end(unwind_bb);
    Value* caught = b.build_landing_pad(Type::i64());
    b.build_ret(b.build_add(caught, b.build_iconst_i64(1)));
    caller->rebuild_cfg_predecessors();

    if (!verify_module(*mod)) return nullptr;
    return mod;
}

// A second module, loaded where code is to be replaced.
std::unique_ptr<Module> build_other_module() {
    auto mod = std::make_unique<Module>("engine_move_other");
    Builder b(*mod);
    Function* fn = mod->create_function("other", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_ret(b.build_add(x, b.build_iconst_i64(7)));
    return mod;
}

bool has_exception_mapping(void* fn) {
    return get_global_exception_registry().find_function_by_pc(reinterpret_cast<uintptr_t>(fn) + 1) != nullptr;
}

// Everything a load set up works through `e`; `bias` is the patchable
// constant's current value, which this patches to `bias + 1`.
void check_loaded(JitExecutionEngine& e, int64_t bias) {
    CHECK_EQ(e.invoke("f", {RuntimeValue::from_i64(5)}).as_i64(), 5);

    void* f = e.get_symbol_address("f");
    REQUIRE(f != nullptr);
    CHECK(e.get_resume_table("f") != nullptr);
    void* resume = e.get_resume_target_address("f", 0);
    CHECK(resume != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(resume) > reinterpret_cast<uintptr_t>(f));

    CHECK_EQ(e.invoke("get_bias", {RuntimeValue::from_i64(1)}).as_i64(), 1 + bias);
    CHECK(e.patch_const64("bias_site", bias + 1));
    CHECK_EQ(e.invoke("get_bias", {RuntimeValue::from_i64(1)}).as_i64(), 2 + bias);

    void* tc = e.get_symbol_address("try_catch");
    REQUIRE(tc != nullptr);
    CHECK(has_exception_mapping(tc));
    CHECK(e.exception_tables().get_table("try_catch") != nullptr);
    auto fn = e.get_function_ptr<int64_t (*)(int64_t)>("try_catch");
    CHECK_EQ(fn(41), 42);
}

// A moved-from engine owns nothing.
void check_empty(JitExecutionEngine& e) {
    CHECK(e.get_symbol_address("f") == nullptr);
    CHECK(e.get_resume_table("f") == nullptr);
    CHECK(!e.patch_const64("bias_site", 0));
}

} // namespace

TEST_CASE("JitExecutionEngine - move construction and assignment carry the loaded code") {
    auto mod = build_move_module();
    REQUIRE(mod != nullptr);
    auto other_mod = build_other_module();

    auto a = std::make_unique<JitExecutionEngine>();
    REQUIRE(a->compile_and_load(*mod));
    check_loaded(*a, 10);
    void* try_catch = a->get_symbol_address("try_catch");

    // Move construction; the source is destroyed while the target runs.
    auto b = std::make_unique<JitExecutionEngine>(std::move(*a));
    check_empty(*a);
    check_loaded(*b, 11);
    a.reset();
    check_loaded(*b, 12);
    CHECK_EQ(b->get_symbol_address("try_catch"), try_catch);

    // Move assignment onto an engine with code of its own: that code's
    // exception mappings go with it.
    auto c = std::make_unique<JitExecutionEngine>();
    REQUIRE(c->compile_and_load(*other_mod));
    void* other = c->get_symbol_address("other");
    REQUIRE(other != nullptr);
    CHECK(has_exception_mapping(other));
    *c = std::move(*b);
    CHECK(!has_exception_mapping(other));
    CHECK(c->get_symbol_address("other") == nullptr);
    check_empty(*b);
    check_loaded(*c, 13);
    b.reset();
    check_loaded(*c, 14);

    // Move assignment onto an empty engine.
    JitExecutionEngine d;
    d = std::move(*c);
    check_empty(*c);
    c.reset();
    check_loaded(d, 15);

    // Another hop, with the moved-from engine still alive.
    JitExecutionEngine e;
    e = std::move(d);
    check_empty(d);
    check_loaded(e, 16);

    // The last owner releases the mappings once.
    {
        JitExecutionEngine last(std::move(e));
        check_loaded(last, 17);
    }
    CHECK(!has_exception_mapping(try_catch));
}

TEST_CASE("JitExecutionEngine - reloading drops the previous code's exception mappings") {
    auto mod = build_move_module();
    REQUIRE(mod != nullptr);
    auto other_mod = build_other_module();

    JitExecutionEngine e;
    REQUIRE(e.compile_and_load(*mod));
    void* try_catch = e.get_symbol_address("try_catch");
    REQUIRE(try_catch != nullptr);
    CHECK(has_exception_mapping(try_catch));

    REQUIRE(e.compile_and_load(*other_mod));
    CHECK(!has_exception_mapping(try_catch));
    CHECK(e.get_resume_table("f") == nullptr);
    CHECK(!e.patch_const64("bias_site", 0));
    CHECK_EQ(e.invoke("other", {RuntimeValue::from_i64(1)}).as_i64(), 8);

    REQUIRE(e.compile_and_load(*mod));
    check_loaded(e, 10);
}
