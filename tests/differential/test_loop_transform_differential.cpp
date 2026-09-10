#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <random>

using namespace brass;

namespace {

// Helper to build a two-loop kernel:
// Loop 1: tmp[i] = in[i] * 3 + 1
// Loop 2: out[i] = tmp[i] * 2 + 5
std::unique_ptr<Module> build_fusion_kernel(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* fn = mod->create_function(fn_name, Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* tmp_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    Value* three = b.build_iconst_i64(3);
    Value* five = b.build_iconst_i64(5);

    b.build_br(l1_hdr, {zero});

    // Loop 1
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* val1 = b.build_load_indexed(Type::i64(), in_ptr, i1, 8, 0);
    Value* m1 = b.build_mul(val1, three);
    Value* a1 = b.build_add(m1, one);
    b.build_store_indexed(Type::i64(), tmp_ptr, i1, 8, 0, a1);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    // Bridge / Exit 1
    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero});

    // Loop 2
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* val2 = b.build_load_indexed(Type::i64(), tmp_ptr, i2, 8, 0);
    Value* m2 = b.build_mul(val2, two);
    Value* a2 = b.build_add(m2, five);
    b.build_store_indexed(Type::i64(), out_ptr, i2, 8, 0, a2);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2});

    // Exit 2
    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

// Helper to build a map-reduce kernel with intermediate heap buffer:
// arr = brass_gc_alloc(n * 8)
// Loop 1: arr[i] = in[i] * 3 + 7
// Loop 2: sum += arr[i]
// ret sum
std::unique_ptr<Module> build_contraction_kernel(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    mod->add_external_symbol("brass_gc_alloc");

    Function* fn = mod->create_function(fn_name, Type::i64(), {
        Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    Value* eight = b.build_iconst_i64(8);
    Value* byte_size = b.build_mul(n, eight);
    Value* arr = b.build_call("brass_gc_alloc", Type::gcref(), {byte_size});

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* three = b.build_iconst_i64(3);
    Value* seven = b.build_iconst_i64(7);

    b.build_br(l1_hdr, {zero});

    // Loop 1: arr[i1] = in[i1] * 3 + 7
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* in_val = b.build_load_indexed(Type::i64(), in_ptr, i1, 8, 0);
    Value* prod = b.build_mul(in_val, three);
    Value* term = b.build_add(prod, seven);
    b.build_store_indexed(Type::i64(), arr, i1, 8, 0, term);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    // Exit 1 -> Loop 2
    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero, zero});

    // Loop 2: sum += arr[i2]
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* sum2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {sum2});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* loaded = b.build_load_indexed(Type::i64(), arr, i2, 8, 0);
    Value* next_sum2 = b.build_add(sum2, loaded);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2, next_sum2});

    // Exit 2
    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    Value* final_sum = b.add_block_param(l2_exit, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();
    return mod;
}

// Helper to build a distributable loop kernel:
// for i = 0..n:
//   out1[i] = in[i] * 5 + 3
//   out2[i] = in[i] * 7 + 1
std::unique_ptr<Module> build_distribution_kernel(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* sink_fn = mod->create_function("dummy_sink", Type::void_type(), {Type::i64()});
    {
        Builder b_sink(*mod);
        b_sink.set_function(sink_fn);
        BasicBlock* s_entry = b_sink.append_block("entry");
        b_sink.position_at_end(s_entry);
        b_sink.build_ret_void();
        sink_fn->rebuild_cfg_predecessors();
    }

    Function* fn = mod->create_function(fn_name, Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out1_ptr = b.add_block_param(entry, Type::ptr());
    Value* out2_ptr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* three = b.build_iconst_i64(3);
    Value* five = b.build_iconst_i64(5);
    Value* seven = b.build_iconst_i64(7);

    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* i = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* val = b.build_load_indexed(Type::i64(), in_ptr, i, 8, 0);
    Value* res1 = b.build_add(b.build_mul(val, five), three);
    b.build_store_indexed(Type::i64(), out1_ptr, i, 8, 0, res1);

    Value* res2 = b.build_add(b.build_mul(val, seven), one);
    b.build_store_indexed(Type::i64(), out2_ptr, i, 8, 0, res2);

    // Call that causes unvectorizable partition
    b.build_call("dummy_sink", Type::void_type(), {res2});

    Value* next_i = b.build_add(i, one);
    b.build_br(hdr, {next_i});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Differential - Loop Fusion Execution Across Trip Counts") {
    std::mt19937_64 rng(42);

    for (int64_t n : {1, 3, 8, 15, 32, 64, 127, 256}) {
        size_t count = static_cast<size_t>(n);
        std::vector<int64_t> in_arr(count);
        std::vector<int64_t> tmp_ref(count, 0);
        std::vector<int64_t> out_ref(count, 0);
        std::vector<int64_t> tmp_jit(count, 0);
        std::vector<int64_t> out_jit(count, 0);

        for (size_t i = 0; i < count; ++i) {
            in_arr[i] = static_cast<int64_t>((rng() % 200) - 100);
        }

        // 1. Reference interpreter execution on un-fused module
        auto ref_mod = build_fusion_kernel("ref_mod", "fusion_kernel");
        Interpreter interp;
        interp.run(*ref_mod->get_function("fusion_kernel"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(tmp_ref.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_ref.data())),
            RuntimeValue::from_i64(n)
        });

        // 2. Native JIT execution on fused module
        auto opt_mod = build_fusion_kernel("opt_mod", "fusion_kernel");
        Function* opt_fn = opt_mod->get_function("fusion_kernel");
        DominatorTree dom(*opt_fn);
        LoopFusionOptions fuse_opts;
        bool fused = loop_fusion_pass(*opt_fn, dom, fuse_opts);
        CHECK(fused);
        CHECK(verify_function(*opt_fn));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("fusion_kernel");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(in_arr.data(), tmp_jit.data(), out_jit.data(), n);

        // 3. Verify exact output match
        CHECK(out_ref == out_jit);
        CHECK(tmp_ref == tmp_jit);
    }
}

TEST_CASE("Differential - Array Contraction Intermediate Buffer Elimination") {
    std::mt19937_64 rng(101);

    for (int64_t n : {1, 4, 11, 23, 45, 64, 100}) {
        size_t count = static_cast<size_t>(n);
        std::vector<int64_t> in_arr(count);
        for (size_t i = 0; i < count; ++i) {
            in_arr[i] = static_cast<int64_t>((rng() % 50) - 25);
        }

        // 1. Reference Interpreter on uncontracted module
        auto ref_mod = build_contraction_kernel("ref_mod", "map_reduce");
        Interpreter interp_ref;
        auto ref_res = interp_ref.run(*ref_mod->get_function("map_reduce"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_arr.data())),
            RuntimeValue::from_i64(n)
        });

        // 2. Optimized module with array contraction
        auto opt_mod = build_contraction_kernel("opt_mod", "map_reduce");
        Function* opt_fn = opt_mod->get_function("map_reduce");
        DominatorTree dom(*opt_fn);
        ArrayContractionOptions contract_opts;
        ArrayContractionStats stats;
        contract_opts.stats = &stats;

        bool contracted = array_contraction_pass(*opt_fn, dom, contract_opts);
        CHECK(contracted);
        CHECK_EQ(stats.arrays_contracted, 1);
        CHECK_EQ(stats.allocations_eliminated, 1);
        CHECK(verify_function(*opt_fn));

        // JIT execute
        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t)>("map_reduce");
        REQUIRE(fn_ptr != nullptr);
        int64_t jit_res = fn_ptr(in_arr.data(), n);

        // 3. Verify exact match
        CHECK_EQ(ref_res.as_i64(), jit_res);
    }
}

TEST_CASE("Differential - Loop Distribution Partitioned Execution") {
    std::mt19937_64 rng(202);

    for (int64_t n : {1, 5, 16, 32, 64}) {
        size_t count = static_cast<size_t>(n);
        std::vector<int64_t> in_arr(count);
        std::vector<int64_t> out1_ref(count, 0);
        std::vector<int64_t> out2_ref(count, 0);
        std::vector<int64_t> out1_jit(count, 0);
        std::vector<int64_t> out2_jit(count, 0);

        for (size_t i = 0; i < count; ++i) {
            in_arr[i] = static_cast<int64_t>((rng() % 100) - 50);
        }

        auto ref_mod = build_distribution_kernel("ref_mod", "dist_kernel");
        Interpreter interp;
        interp.run(*ref_mod->get_function("dist_kernel"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out1_ref.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out2_ref.data())),
            RuntimeValue::from_i64(n)
        });

        auto opt_mod = build_distribution_kernel("opt_mod", "dist_kernel");
        Function* opt_fn = opt_mod->get_function("dist_kernel");
        DominatorTree dom(*opt_fn);
        LoopDistributionOptions dist_opts;
        bool distributed = loop_distribution_pass(*opt_fn, dom, dist_opts);
        CHECK(distributed);
        CHECK(verify_function(*opt_fn));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("dist_kernel");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(in_arr.data(), out1_jit.data(), out2_jit.data(), n);

        CHECK(out1_ref == out1_jit);
        CHECK(out2_ref == out2_jit);
    }
}
