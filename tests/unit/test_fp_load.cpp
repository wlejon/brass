// Loads read only as doubles (fp_load.hpp): the load is retyped in place and
// stands for the bitcasts; a load with any other use keeps its type.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/fp_load.hpp>
#include <brass/mir/verifier.hpp>

#include <cstring>

using namespace brass;

namespace {

size_t count_opcodes(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) n += inst->opcode() == op;
    }
    return n;
}

}  // namespace

TEST_CASE("fp_load - a word read only as a double is loaded as one") {
    Module mod("fp_load");
    Builder b(mod);
    Function* fn = mod.create_function("f", Type::f64(), {Type::ptr()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* next = b.append_block("next");
    b.position_at_end(entry);
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* x = b.build_load(Type::i64(), p, 0);
    Value* y = b.build_load_indexed(Type::i64(), p, b.build_iconst_i64(1), 8, 0);
    Value* fx = b.build_bitcast_f64_i64(x);
    b.build_br(next);
    b.position_at_end(next);
    Value* fx2 = b.build_bitcast_f64_i64(x);
    Value* fy = b.build_bitcast_f64_i64(y);
    b.build_ret(b.build_add(b.build_add(fx, fx2), fy));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(read_doubles_as_doubles(*fn), 2ULL);
    REQUIRE(verify_function(*fn));
    CHECK_EQ(count_opcodes(*fn, Opcode::bitcast_f64_i64), 0ULL);
    CHECK(x->type() == Type::f64());
    CHECK(y->type() == Type::f64());

    double mem[2] = {1.5, 2.25};
    Interpreter interp;
    RuntimeValue v = interp.run(mod, "f", {RuntimeValue::from_ptr(mem)});
    CHECK_EQ(v.as_f64(), 1.5 + 1.5 + 2.25);
}

TEST_CASE("fp_load - a word with another use stays a word") {
    Module mod("fp_load_kept");
    Builder b(mod);
    Function* fn = mod.create_function("f", Type::i64(), {Type::ptr()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* x = b.build_load(Type::i64(), p, 0);
    Value* fx = b.build_bitcast_f64_i64(x);
    b.build_ret(b.build_add(x, b.build_fptosi_i64(fx)));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(read_doubles_as_doubles(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::bitcast_f64_i64), 1ULL);
    CHECK(x->type() == Type::i64());
}
