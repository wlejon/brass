#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <random>
#include <vector>
#include <cmath>

using namespace brass;
using namespace brass::test;

namespace {

// Helper to generate arithmetic and LEA patterns
void generate_fuzz_arithmetic_lea(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb_entry");
    b.position_at_end(entry);

    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* z = b.add_block_param(entry, Type::i64());

    std::vector<Value*> pool = {x, y, z};
    size_t num_ops = 16 + (seed % 15);

    for (size_t i = 0; i < num_ops; ++i) {
        uint32_t choice = static_cast<uint32_t>(rng() % 10);
        Value* v1 = pool[rng() % pool.size()];
        Value* v2 = pool[rng() % pool.size()];

        Value* res = nullptr;
        switch (choice) {
            case 0: { // Scaled add via mul (x * 2, 3, 4, 5, 8, 9 + y) -> LEA candidates
                static const int64_t scales[] = {2, 3, 4, 5, 8, 9};
                int64_t s = scales[rng() % 6];
                Value* scaled = b.build_mul(v1, b.build_iconst_i64(s));
                res = b.build_add(scaled, v2);
                break;
            }
            case 1: { // Scaled add via shift ((x << 1, 2, 3) + y) -> LEA candidates
                int64_t sh = 1 + static_cast<int64_t>(rng() % 3);
                Value* shifted = b.build_shl(v1, b.build_iconst_i64(sh));
                res = b.build_add(shifted, v2);
                break;
            }
            case 2: { // Add with displacement / imm32
                int64_t disp = static_cast<int64_t>((rng() % 2000) - 1000);
                res = b.build_add(v1, b.build_iconst_i64(disp));
                break;
            }
            case 3: { // Arithmetic simplification candidates: add 0, sub 0, mul 1
                uint32_t simp_type = static_cast<uint32_t>(rng() % 3);
                if (simp_type == 0) {
                    res = b.build_add(v1, b.build_iconst_i64(0));
                } else if (simp_type == 1) {
                    res = b.build_sub(v1, b.build_iconst_i64(0));
                } else {
                    res = b.build_mul(v1, b.build_iconst_i64(1));
                }
                break;
            }
            case 4: { // Three-term add: (v1 + v2) + disp
                Value* sum = b.build_add(v1, v2);
                int64_t disp = static_cast<int64_t>((rng() % 500) - 250);
                res = b.build_add(sum, b.build_iconst_i64(disp));
                break;
            }
            case 5: { // Subtraction
                res = b.build_sub(v1, v2);
                break;
            }
            case 6: { // Multiplication by constant
                int64_t c = static_cast<int64_t>((rng() % 15) + 2);
                res = b.build_mul(v1, b.build_iconst_i64(c));
                break;
            }
            case 7: { // Compound: ((v1 * scale) + disp) + v2
                static const int64_t scales[] = {2, 4, 8};
                int64_t s = scales[rng() % 3];
                Value* scaled = b.build_mul(v1, b.build_iconst_i64(s));
                Value* with_disp = b.build_add(scaled, b.build_iconst_i64(static_cast<int64_t>(rng() % 128)));
                res = b.build_add(with_disp, v2);
                break;
            }
            case 8: { // XOR zeroing / mask candidate
                res = b.build_xor(v1, b.build_iconst_i64(0));
                break;
            }
            case 9: { // Regular add
                res = b.build_add(v1, v2);
                break;
            }
        }

        if (res) pool.push_back(res);
    }

    b.build_ret(pool.back());
    fn->rebuild_cfg_predecessors();
}

// Helper to generate bitwise and shift optimization patterns
void generate_fuzz_bitwise_shifts(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb_entry");
    b.position_at_end(entry);

    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    std::vector<Value*> pool = {a, c};
    size_t num_ops = 18 + (seed % 12);

    for (size_t i = 0; i < num_ops; ++i) {
        uint32_t choice = static_cast<uint32_t>(rng() % 13);
        Value* v1 = pool[rng() % pool.size()];
        Value* v2 = pool[rng() % pool.size()];

        Value* res = nullptr;
        switch (choice) {
            case 0: { // AND with bitmask
                static const int64_t masks[] = {
                    0x0, 0x1, 0xFF, 0xFFFF, 0x7FFFFFFF,
                    static_cast<int64_t>(0xFFFFFFFFULL),
                    static_cast<int64_t>(0x5555555555555555ULL),
                    static_cast<int64_t>(0xAAAAAAAAAAAAAAAAULL),
                    -1
                };
                int64_t mask = masks[rng() % 9];
                res = b.build_and(v1, b.build_iconst_i64(mask));
                break;
            }
            case 1: { // OR with bitmask
                int64_t mask = static_cast<int64_t>(rng() % 0xFFFF);
                res = b.build_or(v1, b.build_iconst_i64(mask));
                break;
            }
            case 2: { // XOR with bitmask
                int64_t mask = static_cast<int64_t>(rng() % 0xFFFFFF);
                res = b.build_xor(v1, b.build_iconst_i64(mask));
                break;
            }
            case 3: { // NOT
                res = b.build_not(v1);
                break;
            }
            case 4: { // SHL
                int64_t sh = static_cast<int64_t>(rng() % 32);
                res = b.build_shl(v1, b.build_iconst_i64(sh));
                break;
            }
            case 5: { // LSHR
                int64_t sh = static_cast<int64_t>(rng() % 32);
                res = b.build_lshr(v1, b.build_iconst_i64(sh));
                break;
            }
            case 6: { // ASHR
                int64_t sh = static_cast<int64_t>(rng() % 32);
                res = b.build_ashr(v1, b.build_iconst_i64(sh));
                break;
            }
            case 7: { // POPCNT
                res = b.build_popcnt(v1);
                break;
            }
            case 8: { // CLZ
                res = b.build_clz(v1);
                break;
            }
            case 9: { // CTZ
                res = b.build_ctz(v1);
                break;
            }
            case 10: { // Reg-Reg AND
                res = b.build_and(v1, v2);
                break;
            }
            case 11: { // Reg-Reg OR
                res = b.build_or(v1, v2);
                break;
            }
            case 12: { // Reg-Reg XOR
                res = b.build_xor(v1, v2);
                break;
            }
        }

        if (res) pool.push_back(res);
    }

    b.build_ret(pool.back());
    fn->rebuild_cfg_predecessors();
}

// Helper to generate complex store-to-load forwarding and redundant move sequences
void generate_fuzz_store_load_forwarding(Module& mod, std::string_view fn_name, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Function* fn = mod.create_function(fn_name, Type::i64(), {Type::ptr(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("bb_entry");
    b.position_at_end(entry);

    Value* mem_buf = b.add_block_param(entry, Type::ptr());
    Value* v1 = b.add_block_param(entry, Type::i64());
    Value* v2 = b.add_block_param(entry, Type::i64());

    // 1. Store v1 to offset 0, store v2 to offset 8
    b.build_store(Type::i64(), mem_buf, 0, v1);
    b.build_store(Type::i64(), mem_buf, 8, v2);

    // 2. Redundant load immediately from offset 0
    Value* l0 = b.build_load(Type::i64(), mem_buf, 0);

    // 3. Computed store to offset 16
    Value* sum12 = b.build_add(l0, v2);
    b.build_store(Type::i64(), mem_buf, 16, sum12);

    // 4. Overwrite offset 0 with new value
    Value* v1_mod = b.build_xor(v1, b.build_iconst_i64(0x5A5A));
    b.build_store(Type::i64(), mem_buf, 0, v1_mod);

    // 5. Load both offset 0 and offset 16
    Value* l0_new = b.build_load(Type::i64(), mem_buf, 0);
    Value* l16 = b.build_load(Type::i64(), mem_buf, 16);

    // 6. Indexed store & load pattern
    Value* idx_1 = b.build_iconst_i64(1);
    Value* prod = b.build_mul(l0_new, l16);
    b.build_store_indexed(Type::i64(), mem_buf, idx_1, 8, 16, prod); // slot 3 (offset 24)

    Value* l_indexed = b.build_load_indexed(Type::i64(), mem_buf, idx_1, 8, 16);

    Value* total = b.build_add(b.build_add(l0_new, l16), l_indexed);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
}

} // namespace

TEST_CASE("Differential Fuzzer - Arithmetic and LEA-to-ADD Optimizations") {
    std::mt19937_64 rng(10101);

    for (uint64_t seed = 0; seed < 50; ++seed) {
        std::string mod_name = "fuzz_arith_lea_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_arith_lea_fn";

        generate_fuzz_arithmetic_lea(mod, fn_name, seed + 100);

        int64_t x = static_cast<int64_t>((rng() % 2000) - 1000);
        int64_t y = static_cast<int64_t>((rng() % 2000) - 1000);
        int64_t z = static_cast<int64_t>((rng() % 2000) - 1000);

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_i64(x),
            RuntimeValue::from_i64(y),
            RuntimeValue::from_i64(z)
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t, int64_t)>(fn_name);
        REQUIRE(fn_ptr != nullptr);
        int64_t jit_res = fn_ptr(x, y, z);

        CHECK_EQ(interp_res.as_i64(), jit_res);
    }
}

TEST_CASE("Differential Fuzzer - Bitwise and Shift Pattern Optimizations") {
    std::mt19937_64 rng(20202);

    for (uint64_t seed = 0; seed < 50; ++seed) {
        std::string mod_name = "fuzz_bitwise_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_bitwise_fn";

        generate_fuzz_bitwise_shifts(mod, fn_name, seed + 500);

        int64_t a = static_cast<int64_t>(rng());
        int64_t c = static_cast<int64_t>(rng());

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_i64(a),
            RuntimeValue::from_i64(c)
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>(fn_name);
        REQUIRE(fn_ptr != nullptr);
        int64_t jit_res = fn_ptr(a, c);

        CHECK_EQ(interp_res.as_i64(), jit_res);
    }
}

TEST_CASE("Differential Fuzzer - Load-After-Store Forwarding and Redundant Moves") {
    std::mt19937_64 rng(30303);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        std::string mod_name = "fuzz_forwarding_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_forwarding_fn";

        generate_fuzz_store_load_forwarding(mod, fn_name, seed + 700);

        int64_t v1 = static_cast<int64_t>((rng() % 1000) - 500);
        int64_t v2 = static_cast<int64_t>((rng() % 1000) - 500);

        std::vector<int64_t> buf_interp(8, 0);
        std::vector<int64_t> buf_jit(8, 0);

        Interpreter interp;
        RuntimeValue interp_res = interp.run(mod, fn_name, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(buf_interp.data())),
            RuntimeValue::from_i64(v1),
            RuntimeValue::from_i64(v2)
        });

        codegen::JitExecutionEngine jit(Target::host());
        bool ok = jit.compile_and_load(mod);
        REQUIRE(ok);

        auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t, int64_t)>(fn_name);
        REQUIRE(fn_ptr != nullptr);
        int64_t jit_res = fn_ptr(buf_jit.data(), v1, v2);

        CHECK_EQ(interp_res.as_i64(), jit_res);
        CHECK(buf_interp == buf_jit);
    }
}

TEST_CASE("Differential Fuzzer - Select and Branchless CMOV") {
    std::mt19937_64 rng(40404);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        Module mod("fuzz_select_mod_" + std::to_string(seed));
        Builder b(mod);

        Function* fn_i64 = mod.create_function("fuzz_select_i64", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
        b.set_function(fn_i64);
        BasicBlock* entry_i64 = b.append_block("entry");
        b.position_at_end(entry_i64);
        Value* a = b.add_block_param(entry_i64, Type::i64());
        Value* x = b.add_block_param(entry_i64, Type::i64());
        Value* y = b.add_block_param(entry_i64, Type::i64());
        Value* cond_i64 = b.build_slt(a, b.build_iconst_i64(50));
        Value* sel_i64 = b.build_select(cond_i64, x, y);
        b.build_ret(sel_i64);

        Function* fn_f64 = mod.create_function("fuzz_select_f64", Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
        b.set_function(fn_f64);
        BasicBlock* entry_f64 = b.append_block("entry");
        b.position_at_end(entry_f64);
        Value* fa = b.add_block_param(entry_f64, Type::f64());
        Value* fx = b.add_block_param(entry_f64, Type::f64());
        Value* fy = b.add_block_param(entry_f64, Type::f64());
        Value* cond_f64 = b.build_slt(fa, b.build_fconst_f64(0.0));
        Value* sel_f64 = b.build_select(cond_f64, fx, fy);
        b.build_ret(sel_f64);

        fn_i64->rebuild_cfg_predecessors();
        fn_f64->rebuild_cfg_predecessors();

        int64_t v_a = static_cast<int64_t>(rng() % 100);
        int64_t v_x = static_cast<int64_t>((rng() % 2000) - 1000);
        int64_t v_y = static_cast<int64_t>((rng() % 2000) - 1000);
        assert_diff(mod, "fuzz_select_i64", {
            RuntimeValue::from_i64(v_a),
            RuntimeValue::from_i64(v_x),
            RuntimeValue::from_i64(v_y)
        });

        double d_a = static_cast<double>(static_cast<int64_t>(rng() % 200) - 100) / 10.0;
        double d_x = static_cast<double>(static_cast<int64_t>(rng() % 2000) - 1000) / 10.0;
        double d_y = static_cast<double>(static_cast<int64_t>(rng() % 2000) - 1000) / 10.0;
        assert_diff(mod, "fuzz_select_f64", {
            RuntimeValue::from_f64(d_a),
            RuntimeValue::from_f64(d_x),
            RuntimeValue::from_f64(d_y)
        });
    }
}

