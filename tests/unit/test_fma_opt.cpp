#include "test_framework.hpp"
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>

using namespace brass;

TEST_CASE("FMA Opt - Scalar Float Pattern Matching") {
    Module mod("scalar_fma_test");
    Function* fn = mod.create_function("scalar_fma_fn", Type::f32(), {Type::f32(), Type::f32(), Type::f32()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* a = b.add_block_param(entry, Type::f32());
    Value* b_val = b.add_block_param(entry, Type::f32());
    Value* c = b.add_block_param(entry, Type::f32());

    // mul(a, b) + c
    Value* prod = b.build_mul(a, b_val);
    Value* sum1 = b.build_add(prod, c);

    // c + mul(a, b) (commutative)
    Value* sum2 = b.build_add(c, prod);

    Value* total = b.build_add(sum1, sum2);
    b.build_ret(total);

    FmaOptStats stats;
    FmaOptOptions opts;
    opts.stats = &stats;
    bool changed = fma_opt_pass(*fn, opts);

    CHECK(changed);
    // Both sum1 and sum2 should be converted to fma_f32
    CHECK_EQ(stats.scalar_fma_f32_count, 2u);
    CHECK_EQ(stats.total_fused(), 2u);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Verify instructions in block
    size_t fma_count = 0;
    for (const Instruction* inst : *entry) {
        if (inst && inst->opcode() == Opcode::fma_f32) {
            fma_count++;
            CHECK_EQ(inst->operand(0), a);
            CHECK_EQ(inst->operand(1), b_val);
            CHECK_EQ(inst->operand(2), c);
        }
    }
    CHECK_EQ(fma_count, 2u);
}

TEST_CASE("FMA Opt - Vector Float Pattern Matching") {
    Module mod("vec_fma_test");
    Function* fn = mod.create_function("vec_fma_fn", Type::f32x8(), {Type::f32x8(), Type::f32x8(), Type::f32x8()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* v0 = b.add_block_param(entry, Type::f32x8());
    Value* v1 = b.add_block_param(entry, Type::f32x8());
    Value* v2 = b.add_block_param(entry, Type::f32x8());

    Value* vprod = b.build_vmul(v0, v1);
    Value* vsum = b.build_vadd(vprod, v2);
    b.build_ret(vsum);

    FmaOptStats stats;
    bool changed = run_fma_opt(*fn, &stats);

    CHECK(changed);
    CHECK_EQ(stats.vector_vfma_count, 1u);
    CHECK_EQ(stats.total_fused(), 1u);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Verify inst is vfma
    Instruction* fma_inst = vsum->defining_instruction();
    REQUIRE(fma_inst != nullptr);
    CHECK_EQ(fma_inst->opcode(), Opcode::vfma);
    CHECK_EQ(fma_inst->operand(0), v0);
    CHECK_EQ(fma_inst->operand(1), v1);
    CHECK_EQ(fma_inst->operand(2), v2);

    std::string report = stats.format_report();
    CHECK(report.find("Vector vfma fused:    1") != std::string::npos);
}
