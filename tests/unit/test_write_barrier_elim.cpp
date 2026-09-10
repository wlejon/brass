#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/write_barrier_elim.hpp>

using namespace brass;

namespace {

size_t count_write_barriers(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::write_barrier) {
                count++;
            }
        }
    }
    return count;
}

} // namespace

TEST_CASE("WBE - Non-pointer value elimination") {
    Module mod("test_wbe_non_ptr");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());

    Value* int_val = b.build_iconst_i64(42);
    // Write barrier on integer value
    b.build_write_barrier(obj, int_val);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 0ULL);
    CHECK_EQ(wbe.stats().eliminated_non_pointer, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Young provenance allocation elimination") {
    Module mod("test_wbe_young");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::gcref(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* child = b.add_block_param(entry, Type::gcref());

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask1 = b.build_iconst_i64(1);
    Value* tag1 = b.build_iconst_i32(1);

    // obj is freshly allocated in nursery
    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask1, tag1});

    // Write barrier storing child into young obj
    b.build_write_barrier(obj, child);
    b.build_ret(obj);

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 0ULL);
    CHECK_EQ(wbe.stats().eliminated_young_provenance, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Redundant intra-block write barrier elimination") {
    Module mod("test_wbe_redundant");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref(), Type::gcref(), Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());
    Value* val1 = b.add_block_param(entry, Type::gcref());
    Value* val2 = b.add_block_param(entry, Type::gcref());

    // First barrier on obj
    b.build_write_barrier(obj, val1);
    // Second barrier on obj with different value in same block
    b.build_write_barrier(obj, val2);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 2ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(changed);
    CHECK_EQ(count_write_barriers(*fn), 1ULL);
    CHECK_EQ(wbe.stats().eliminated_redundant, 1U);
    CHECK_EQ(wbe.stats().remaining_barriers, 1U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}

TEST_CASE("WBE - Preserving required cross-generation barrier") {
    Module mod("test_wbe_preserve");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::void_type(), {Type::gcref(), Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());
    Value* child = b.add_block_param(entry, Type::gcref());

    b.build_write_barrier(obj, child);
    b.build_ret_void();

    CHECK_EQ(count_write_barriers(*fn), 1ULL);

    WriteBarrierElimination wbe;
    bool changed = wbe.run_on_function(*fn);

    CHECK(!changed);
    CHECK_EQ(count_write_barriers(*fn), 1ULL);
    CHECK_EQ(wbe.stats().remaining_barriers, 1U);
    CHECK_EQ(wbe.stats().total_eliminated(), 0U);

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
}
