#include "test_framework.hpp"
#include <brass/fuzz/delta_reducer.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <fstream>
#include <cstdio>

using namespace brass;
using namespace brass::fuzz;

TEST_CASE("DeltaReducer_InstructionReduction") {
    // Build a synthetic module with dead/unrelated instructions
    Module mod("reducer_test");
    Builder b(mod);

    Function* fn = mod.create_function("test_fn", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, Type::i64());
    Value* p1 = b.add_block_param(entry, Type::i64());

    // Unrelated dead instructions
    Value* c1 = b.build_iconst_i64(10);
    Value* c2 = b.build_iconst_i64(20);
    Value* d1 = b.build_add(c1, c2);
    Value* d2 = b.build_mul(d1, c1);
    (void)d2;

    // Critical path instructions
    Value* sum = b.build_add(p0, p1);
    Value* c3 = b.build_iconst_i64(42);
    Value* target = b.build_mul(sum, c3);
    b.build_ret(target);
    fn->rebuild_cfg_predecessors();

    CHECK(verify_function(*fn));

    // Oracle: succeeds (reproduces) only if the function still computes target involving sum and c3
    auto oracle = [](const Module& m, std::string_view fn_name) -> bool {
        const Function* f = m.get_function(fn_name);
        if (!f) return false;
        bool has_target_mul = false;
        for (const BasicBlock* bb : f->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst && inst->opcode() == Opcode::mul) {
                    has_target_mul = true;
                }
            }
        }
        return has_target_mul;
    };

    DeltaReducer reducer;
    ReductionResult res = reducer.reduce(mod, "test_fn", oracle);

    CHECK(res.success);
    CHECK(res.minimized_module != nullptr);
    CHECK(res.final_instructions < res.initial_instructions);

    const Function* min_fn = res.minimized_module->get_function("test_fn");
    CHECK(min_fn != nullptr);
    CHECK(verify_function(*min_fn));
}

TEST_CASE("DeltaReducer_Exporters") {
    Module mod("export_test");
    Builder b(mod);

    Function* fn = mod.create_function("main", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");
    Value* zero = b.build_iconst_i64(0);
    b.build_ret(zero);
    fn->rebuild_cfg_predecessors();

    std::string mir_path = "test_export_min.mir";
    std::string il_path = "test_export_min.il";

    CHECK(DeltaReducer::export_mir(mod, mir_path));
    CHECK(DeltaReducer::export_il(mod, il_path));

    std::ifstream mir_file(mir_path);
    CHECK(mir_file.is_open());
    std::string mir_line;
    std::getline(mir_file, mir_line);
    CHECK_FALSE(mir_line.empty());
    mir_file.close();

    std::ifstream il_file(il_path);
    CHECK(il_file.is_open());
    std::string il_line;
    std::getline(il_file, il_line);
    CHECK_FALSE(il_line.empty());
    il_file.close();

    std::remove(mir_path.c_str());
    std::remove(il_path.c_str());
}
