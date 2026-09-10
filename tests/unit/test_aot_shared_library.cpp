#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>

#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <memory>

using namespace brass;
using namespace brass::target;

namespace {

std::unique_ptr<Module> build_aot_roundtrip_module() {
    auto mod = std::make_unique<Module>("aot_roundtrip_test");
    Builder b(*mod);

    // 1. fib(n: i64) -> i64 (iterative)
    Function* fn_fib = mod->create_function("fib", Type::i64(), {Type::i64()});
    {
        b.set_function(fn_fib);
        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* base_case = b.create_block("base_case");
        BasicBlock* loop_init = b.create_block("loop_init");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* c2 = b.build_iconst_i64(2);
        Value* is_small = b.build_slt(n, c2);
        b.build_br_if(is_small, base_case, {}, loop_init, {});

        fn_fib->append_block(base_case);
        b.position_at_end(base_case);
        b.build_ret(n);

        fn_fib->append_block(loop_init);
        b.position_at_end(loop_init);
        Value* i_init = b.build_iconst_i64(2);
        Value* a_init = b.build_iconst_i64(0);
        Value* b_init = b.build_iconst_i64(1);
        b.build_br(loop_body, {i_init, a_init, b_init});

        fn_fib->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* i = b.add_block_param(loop_body, Type::i64());
        Value* a = b.add_block_param(loop_body, Type::i64());
        Value* cur_b = b.add_block_param(loop_body, Type::i64());
        Value* c = b.build_add(a, cur_b);
        Value* one = b.build_iconst_i64(1);
        Value* next_i = b.build_add(i, one);
        Value* in_range = b.build_sle(next_i, n);
        b.build_br_if(in_range, loop_body, {next_i, cur_b, c}, exit_bb, {c});

        fn_fib->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* ret_val = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(ret_val);

        fn_fib->rebuild_cfg_predecessors();
        if (!verify_function(*fn_fib)) return nullptr;
    }

    // 2. vec3_length_sq(x: f64, y: f64, z: f64) -> f64
    Function* fn_len = mod->create_function("vec3_length_sq", Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
    {
        b.set_function(fn_len);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::f64());
        Value* y = b.add_block_param(entry, Type::f64());
        Value* z = b.add_block_param(entry, Type::f64());

        Value* xx = b.build_mul(x, x);
        Value* yy = b.build_mul(y, y);
        Value* zz = b.build_mul(z, z);
        Value* sum = b.build_add(xx, yy);
        Value* total = b.build_add(sum, zz);
        b.build_ret(total);

        fn_len->rebuild_cfg_predecessors();
        if (!verify_function(*fn_len)) return nullptr;
    }

    // 3. dot_product(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64
    Function* fn_dot = mod->create_function("dot_product", Type::f64(), {Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64()});
    {
        b.set_function(fn_dot);
        BasicBlock* entry = b.append_block("entry");
        Value* x1 = b.add_block_param(entry, Type::f64());
        Value* y1 = b.add_block_param(entry, Type::f64());
        Value* z1 = b.add_block_param(entry, Type::f64());
        Value* x2 = b.add_block_param(entry, Type::f64());
        Value* y2 = b.add_block_param(entry, Type::f64());
        Value* z2 = b.add_block_param(entry, Type::f64());

        Value* p1 = b.build_mul(x1, x2);
        Value* p2 = b.build_mul(y1, y2);
        Value* p3 = b.build_mul(z1, z2);
        Value* sum1 = b.build_add(p1, p2);
        Value* dot = b.build_add(sum1, p3);
        b.build_ret(dot);

        fn_dot->rebuild_cfg_predecessors();
        if (!verify_function(*fn_dot)) return nullptr;
    }

    // 4. loop_kernel(n: i64) -> i64: computes sum(i * 3 + 1) for i = 0..n-1
    Function* fn_loop = mod->create_function("loop_kernel", Type::i64(), {Type::i64()});
    {
        b.set_function(fn_loop);
        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* check_bb = b.create_block("check");
        BasicBlock* body_bb = b.create_block("body");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero = b.build_iconst_i64(0);
        b.build_br(check_bb, {zero, zero});

        fn_loop->append_block(check_bb);
        b.position_at_end(check_bb);
        Value* i = b.add_block_param(check_bb, Type::i64());
        Value* acc = b.add_block_param(check_bb, Type::i64());
        Value* cond = b.build_slt(i, n);
        b.build_br_if(cond, body_bb, {}, exit_bb, {acc});

        fn_loop->append_block(body_bb);
        b.position_at_end(body_bb);
        Value* c3 = b.build_iconst_i64(3);
        Value* c1 = b.build_iconst_i64(1);
        Value* term = b.build_add(b.build_mul(i, c3), c1);
        Value* next_acc = b.build_add(acc, term);
        Value* next_i = b.build_add(i, c1);
        b.build_br(check_bb, {next_i, next_acc});

        fn_loop->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* res = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(res);

        fn_loop->rebuild_cfg_predecessors();
        if (!verify_function(*fn_loop)) return nullptr;
    }

    // 5. collatz(n: i64) -> i64
    Function* fn_collatz = mod->create_function("collatz", Type::i64(), {Type::i64()});
    {
        b.set_function(fn_collatz);
        BasicBlock* entry = b.append_block("entry");
        Value* n = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_bb = b.create_block("loop");
        BasicBlock* check_parity = b.create_block("check_parity");
        BasicBlock* step_odd = b.create_block("step_odd");
        BasicBlock* step_even = b.create_block("step_even");
        BasicBlock* next_step_bb = b.create_block("next_step");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop_bb, {n, zero});

        fn_collatz->append_block(loop_bb);
        b.position_at_end(loop_bb);
        Value* curr = b.add_block_param(loop_bb, Type::i64());
        Value* steps = b.add_block_param(loop_bb, Type::i64());

        Value* c1 = b.build_iconst_i64(1);
        Value* is_gt_1 = b.build_sgt(curr, c1);
        b.build_br_if(is_gt_1, check_parity, {}, exit_bb, {steps});

        fn_collatz->append_block(check_parity);
        b.position_at_end(check_parity);
        Value* rem = b.build_and(curr, c1);
        Value* is_odd = b.build_eq(rem, c1);
        b.build_br_if(is_odd, step_odd, {}, step_even, {});

        fn_collatz->append_block(step_odd);
        b.position_at_end(step_odd);
        Value* c3 = b.build_iconst_i64(3);
        Value* mul3 = b.build_mul(curr, c3);
        Value* next_odd = b.build_add(mul3, c1);
        b.build_br(next_step_bb, {next_odd});

        fn_collatz->append_block(step_even);
        b.position_at_end(step_even);
        Value* c2 = b.build_iconst_i64(2);
        Value* next_even = b.build_sdiv(curr, c2);
        b.build_br(next_step_bb, {next_even});

        fn_collatz->append_block(next_step_bb);
        b.position_at_end(next_step_bb);
        Value* next_val = b.add_block_param(next_step_bb, Type::i64());
        Value* next_steps = b.build_add(steps, c1);
        b.build_br(loop_bb, {next_val, next_steps});

        fn_collatz->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* res_steps = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(res_steps);

        fn_collatz->rebuild_cfg_predecessors();
        if (!verify_function(*fn_collatz)) return nullptr;
    }

    return mod;
}

} // namespace

TEST_CASE("AOT Linker - Roundtrip Native Execution with DynamicLibrary") {
    auto mod = build_aot_roundtrip_module();
    REQUIRE(mod != nullptr);

    Interpreter interp;
    interp.set_module(mod.get());

    Function* fn_fib = mod->get_function("fib");
    Function* fn_len = mod->get_function("vec3_length_sq");
    Function* fn_dot = mod->get_function("dot_product");
    Function* fn_loop = mod->get_function("loop_kernel");
    Function* fn_collatz = mod->get_function("collatz");
    REQUIRE(fn_fib != nullptr);
    REQUIRE(fn_len != nullptr);
    REQUIRE(fn_dot != nullptr);
    REQUIRE(fn_loop != nullptr);
    REQUIRE(fn_collatz != nullptr);

    // Compute Interpreter oracle values
    std::vector<int64_t> fib_inputs = {0, 1, 2, 3, 5, 10, 20};
    std::vector<int64_t> fib_oracle;
    for (int64_t in : fib_inputs) {
        RuntimeValue res = interp.run(*fn_fib, {RuntimeValue::from_i64(in)});
        fib_oracle.push_back(res.as_i64());
    }

    struct Vec3Test {
        double x, y, z;
    };
    std::vector<Vec3Test> vec_tests = {
        {0.0, 0.0, 0.0},
        {1.0, 2.0, 3.0},
        {-3.5, 4.2, 1.8},
        {10.0, -20.0, 30.0}
    };
    std::vector<double> len_oracle;
    for (const auto& t : vec_tests) {
        RuntimeValue res = interp.run(*fn_len, {
            RuntimeValue::from_f64(t.x),
            RuntimeValue::from_f64(t.y),
            RuntimeValue::from_f64(t.z)
        });
        len_oracle.push_back(res.as_f64());
    }

    struct DotTest {
        double x1, y1, z1, x2, y2, z2;
    };
    std::vector<DotTest> dot_tests = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {0.5, -1.5, 2.0, 3.0, 2.0, -1.0},
        {0.0, 0.0, 0.0, 10.0, 20.0, 30.0},
        {-2.0, -4.0, 6.0, 1.5, 0.5, -2.5}
    };
    std::vector<double> dot_oracle;
    for (const auto& t : dot_tests) {
        RuntimeValue res = interp.run(*fn_dot, {
            RuntimeValue::from_f64(t.x1),
            RuntimeValue::from_f64(t.y1),
            RuntimeValue::from_f64(t.z1),
            RuntimeValue::from_f64(t.x2),
            RuntimeValue::from_f64(t.y2),
            RuntimeValue::from_f64(t.z2)
        });
        dot_oracle.push_back(res.as_f64());
    }

    std::vector<int64_t> loop_inputs = {0, 1, 5, 10, 50, 100};
    std::vector<int64_t> loop_oracle;
    for (int64_t in : loop_inputs) {
        RuntimeValue res = interp.run(*fn_loop, {RuntimeValue::from_i64(in)});
        loop_oracle.push_back(res.as_i64());
    }

    std::vector<int64_t> collatz_inputs = {1, 2, 3, 6, 12, 19, 27};
    std::vector<int64_t> collatz_oracle;
    for (int64_t in : collatz_inputs) {
        RuntimeValue res = interp.run(*fn_collatz, {RuntimeValue::from_i64(in)});
        collatz_oracle.push_back(res.as_i64());
    }

    // Now test AOT Linker & native loading on Windows and macOS
    if (Target::host().is_windows() || Target::host().is_macos()) {
        std::string ext = Target::host().is_windows() ? ".dll" : ".dylib";
        std::filesystem::path dll_path = std::filesystem::temp_directory_path() / ("test_aot_roundtrip_module" + ext);
        std::error_code ec;
        std::filesystem::remove(dll_path, ec);

        LinkerOptions opts;
        opts.module_name = "test_aot_roundtrip_module" + ext;
        opts.export_all_functions = true;

        bool linked = AotLinker::link_to_file(*mod, dll_path.string(), Target::host(), opts);
        REQUIRE(linked);
        REQUIRE(std::filesystem::exists(dll_path));

        std::string err;
        auto lib = DynamicLibrary::open(dll_path.string(), &err);
        if (!lib) {
            std::cerr << "DynamicLibrary::open failed: " << err << "\n";
        }
        REQUIRE(lib != nullptr);
        REQUIRE(lib->is_valid());

        // Test fib
        auto native_fib = lib->get_function<int64_t(*)(int64_t)>("fib");
        REQUIRE(native_fib != nullptr);
        for (size_t i = 0; i < fib_inputs.size(); ++i) {
            int64_t actual = native_fib(fib_inputs[i]);
            CHECK_EQ(actual, fib_oracle[i]);
        }

        // Test vec3_length_sq
        auto native_len = lib->get_function<double(*)(double, double, double)>("vec3_length_sq");
        REQUIRE(native_len != nullptr);
        for (size_t i = 0; i < vec_tests.size(); ++i) {
            double actual = native_len(vec_tests[i].x, vec_tests[i].y, vec_tests[i].z);
            CHECK_EQ(actual, len_oracle[i]);
        }

        // Test dot_product
        auto native_dot = lib->get_function<double(*)(double, double, double, double, double, double)>("dot_product");
        REQUIRE(native_dot != nullptr);
        for (size_t i = 0; i < dot_tests.size(); ++i) {
            double actual = native_dot(dot_tests[i].x1, dot_tests[i].y1, dot_tests[i].z1,
                                       dot_tests[i].x2, dot_tests[i].y2, dot_tests[i].z2);
            CHECK_EQ(actual, dot_oracle[i]);
        }

        // Test loop_kernel
        auto native_loop = lib->get_function<int64_t(*)(int64_t)>("loop_kernel");
        REQUIRE(native_loop != nullptr);
        for (size_t i = 0; i < loop_inputs.size(); ++i) {
            int64_t actual = native_loop(loop_inputs[i]);
            CHECK_EQ(actual, loop_oracle[i]);
        }

        // Test collatz
        auto native_collatz = lib->get_function<int64_t(*)(int64_t)>("collatz");
        REQUIRE(native_collatz != nullptr);
        for (size_t i = 0; i < collatz_inputs.size(); ++i) {
            int64_t actual = native_collatz(collatz_inputs[i]);
            CHECK_EQ(actual, collatz_oracle[i]);
        }

        // Unload and clean up
        lib.reset();
        std::filesystem::remove(dll_path, ec);
    }
}

TEST_CASE("AOT Linker - Selective Function Exports") {
    auto mod = build_aot_roundtrip_module();
    REQUIRE(mod != nullptr);

    if (Target::host().is_windows() || Target::host().is_macos()) {
        std::string ext = Target::host().is_windows() ? ".dll" : ".dylib";
        std::filesystem::path dll_path = std::filesystem::temp_directory_path() / ("test_aot_selective_exports" + ext);
        std::error_code ec;
        std::filesystem::remove(dll_path, ec);

        LinkerOptions opts;
        opts.module_name = "test_aot_selective_exports" + ext;
        opts.export_all_functions = false;
        opts.explicit_exports = {"fib", "collatz"};

        bool linked = AotLinker::link_to_file(*mod, dll_path.string(), Target::host(), opts);
        REQUIRE(linked);

        auto lib = DynamicLibrary::open(dll_path.string());
        REQUIRE(lib != nullptr);
        REQUIRE(lib->is_valid());

        // Explicitly exported symbols should be present
        CHECK(lib->get_symbol("fib") != nullptr);
        CHECK(lib->get_symbol("collatz") != nullptr);

        // Non-exported symbols should NOT be present
        CHECK(lib->get_symbol("vec3_length_sq") == nullptr);
        CHECK(lib->get_symbol("dot_product") == nullptr);
        CHECK(lib->get_symbol("loop_kernel") == nullptr);

        // fib should still be callable and produce correct answer
        auto fn_fib = lib->get_function<int64_t(*)(int64_t)>("fib");
        REQUIRE(fn_fib != nullptr);
        CHECK_EQ(fn_fib(10), 55);

        lib.reset();
        std::filesystem::remove(dll_path, ec);
    }
}

TEST_CASE("AOT Linker - ELF Shared Object Generation Parity") {
    auto mod = build_aot_roundtrip_module();
    REQUIRE(mod != nullptr);

    LinkerOptions opts;
    opts.format = OutputFormat::LinuxElfSo;
    opts.soname = "libaot_test.so";
    opts.export_all_functions = true;

    std::vector<uint8_t> elf_bytes = AotLinker::link(*mod, Target::x64_linux(), opts);
    REQUIRE(elf_bytes.size() >= 0x400);

    // Verify ELF header
    CHECK_EQ(elf_bytes[0], 0x7F);
    CHECK_EQ(elf_bytes[1], 'E');
    CHECK_EQ(elf_bytes[2], 'L');
    CHECK_EQ(elf_bytes[3], 'F');
    CHECK_EQ(elf_bytes[4], 2); // ELFCLASS64
    CHECK_EQ(elf_bytes[5], 1); // ELFDATA2LSB

    uint16_t e_type = static_cast<uint16_t>(elf_bytes[16] | (elf_bytes[17] << 8));
    CHECK_EQ(e_type, 3); // ET_DYN
}

TEST_CASE("AOT Linker - Mach-O Dynamic Library Generation Parity") {
    auto mod = build_aot_roundtrip_module();
    REQUIRE(mod != nullptr);

    LinkerOptions opts;
    opts.format = OutputFormat::MacOSMachODylib;
    opts.soname = "libaot_test.dylib";
    opts.export_all_functions = true;

    std::vector<uint8_t> dylib_bytes = AotLinker::link(*mod, Target::x64_macos(), opts);
    REQUIRE(dylib_bytes.size() >= 4096);

    uint32_t magic = static_cast<uint32_t>(dylib_bytes[0] | (dylib_bytes[1] << 8) | (dylib_bytes[2] << 16) | (dylib_bytes[3] << 24));
    CHECK_EQ(magic, 0xFEEDFACF); // MH_MAGIC_64

    uint32_t filetype = static_cast<uint32_t>(dylib_bytes[12] | (dylib_bytes[13] << 8) | (dylib_bytes[14] << 16) | (dylib_bytes[15] << 24));
    CHECK_EQ(filetype, 6); // MH_DYLIB
}

TEST_CASE("DynamicLibrary - Error Handling on Non-Existent Files and Missing Symbols") {
    std::string err;
    auto lib_bad = DynamicLibrary::open("non_existent_file_definitely_not_here_404.dll", &err);
    CHECK(lib_bad == nullptr);
    CHECK(!err.empty());

    if (Target::host().is_windows() || Target::host().is_macos()) {
        std::string ext = Target::host().is_windows() ? ".dll" : ".dylib";
        std::filesystem::path dll_path = std::filesystem::temp_directory_path() / ("test_aot_err_handling" + ext);
        std::error_code ec;
        std::filesystem::remove(dll_path, ec);

        auto mod = build_aot_roundtrip_module();
        REQUIRE(mod != nullptr);

        LinkerOptions opts;
        opts.module_name = "test_aot_err_handling" + ext;
        bool linked = AotLinker::link_to_file(*mod, dll_path.string(), Target::host(), opts);
        REQUIRE(linked);

        auto lib = DynamicLibrary::open(dll_path.string());
        REQUIRE(lib != nullptr);

        CHECK(lib->get_symbol("unknown_symbol_xyz_123") == nullptr);

        lib.reset();
        std::filesystem::remove(dll_path, ec);
    }
}
