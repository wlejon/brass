// C API handle lifetime and misuse (bug sweep 5): module-scoped invalidation,
// builders after module destroy, any-order destroy, builder misuse errors,
// per-module JIT symbols, and the embedding API's DLL exports.
#include "test_framework.hpp"
#include <brass/brass_c_api.h>

#include <brass/target/dynamic_library.hpp>
#include <cstring>
#include <string>

using namespace brass::test;

namespace {

using AddFn = int64_t (*)(int64_t);

bool has_error(BrassContext ctx) {
    const char* e = brass_context_get_last_error(ctx);
    return e && e[0] != '\0';
}

// i64 name(i64 x) { return x + k; }. Returns the builder (not destroyed).
BrassBuilder build_addk(BrassContext ctx, BrassModule mod, const char* name, int64_t k) {
    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(mod, name, i64_t, &i64_t, 1);
    REQUIRE(fn != nullptr);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    REQUIRE(entry != nullptr);
    BrassBuilder b = brass_builder_create(ctx, fn);
    REQUIRE(b != nullptr);
    brass_builder_position_at_end(b, entry);
    BrassValue x = brass_function_get_param(fn, 0);
    BrassValue c = brass_build_iconst_i64(b, k);
    BrassValue s = brass_build_add(b, x, c);
    REQUIRE(s != nullptr);
    REQUIRE_EQ(brass_build_ret(b, s), BRASS_OK);
    return b;
}

} // namespace

// Finding 1: destroying module A must not touch module B's handles.
TEST_CASE("C-API lifetime - destroying one module leaves another module's handles valid") {
    BrassContext ctx = brass_context_create();
    BrassModule a = brass_module_create(ctx, "a");
    BrassModule b = brass_module_create(ctx, "b");
    BrassBuilder ba = build_addk(ctx, a, "fa", 1);

    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(b, "fb", i64_t, &i64_t, 1);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassBuilder bb = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(bb, entry);
    BrassValue x = brass_function_get_param(fn, 0);
    BrassValue c = brass_build_iconst_i64(bb, 5);
    REQUIRE(c != nullptr);

    brass_builder_destroy(ba);
    brass_module_destroy(a);

    BrassValue s = brass_build_add(bb, x, c);
    CHECK(s != nullptr);
    CHECK(brass_build_iconst_i64(bb, 7) != nullptr);
    CHECK_EQ(brass_build_ret(bb, s), BRASS_OK);
    CHECK(!has_error(ctx));

    BrassJitEngine jit = brass_jit_create(ctx);
    BrassCompiledModule cb = brass_jit_compile_module(jit, b);
    REQUIRE(cb != nullptr);
    auto f = reinterpret_cast<AddFn>(brass_compiled_module_get_symbol(cb, "fb"));
    REQUIRE(f != nullptr);
    CHECK_EQ(f(10), int64_t{15});

    brass_builder_destroy(bb);
    brass_jit_destroy(jit);
    brass_module_destroy(b);
    brass_context_destroy(ctx);
}

// Finding 2: every builder entry point rejects a builder whose module is gone.
TEST_CASE("C-API lifetime - builder use after its module is destroyed is an error") {
    BrassContext ctx = brass_context_create();
    BrassModule a = brass_module_create(ctx, "a");
    BrassType i64_t = brass_type_i64();
    BrassFunction fn = brass_function_create(a, "g", i64_t, &i64_t, 1);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassBlock other = brass_function_append_block(fn, "other");
    BrassBuilder b = brass_builder_create(ctx, fn);
    brass_builder_position_at_end(b, entry);
    BrassValue x = brass_function_get_param(fn, 0);
    brass_module_destroy(a);

    brass_context_set_error(ctx, "");
    CHECK(brass_build_iconst_i64(b, 1) == nullptr);
    CHECK(has_error(ctx));
    for (int i = 0; i < 64; ++i) {
        brass_context_set_error(ctx, "");
        CHECK(brass_build_fconst_f64(b, 2.0) == nullptr);
        CHECK(has_error(ctx));
    }
    CHECK(brass_build_fconst_f32(b, 1.0f) == nullptr);
    CHECK(brass_build_add(b, x, x) == nullptr);
    CHECK(brass_build_fdiv(b, x, x) == nullptr);
    CHECK(brass_build_call(b, "g", i64_t, &x, 1) == nullptr);
    CHECK(brass_build_func_addr(b, "g") == nullptr);
    CHECK(brass_build_vzero(b, brass_type_v128(BRASS_LANE_F32)) == nullptr);
    CHECK(brass_build_load(b, i64_t, x, 0) == nullptr);
    CHECK_EQ(brass_build_br(b, other, nullptr, 0), BRASS_ERR_INVALID_ARGUMENT);
    CHECK_EQ(brass_build_unreachable(b), BRASS_ERR_INVALID_ARGUMENT);
    brass_context_set_error(ctx, "");
    CHECK_EQ(brass_build_ret(b, nullptr), BRASS_ERR_INVALID_ARGUMENT);
    CHECK(has_error(ctx));

    brass_builder_destroy(b);
    brass_context_destroy(ctx);
}

// Finding 3: any destroy order is safe (the context outlives its dependents).
TEST_CASE("C-API lifetime - context may be destroyed before its modules, builders and JITs") {
    for (int i = 0; i < 50; ++i) {
        BrassContext ctx = brass_context_create();
        BrassModule a = brass_module_create(ctx, "a");
        BrassBuilder b1 = build_addk(ctx, a, "f", i);
        BrassJitEngine jit = brass_jit_create(ctx);
        BrassCompiledModule ca = brass_jit_compile_module(jit, a);
        REQUIRE(ca != nullptr);
        BrassKernelJit kj = brass_kernel_jit_create(ctx, nullptr);
        REQUIRE(kj != nullptr);

        brass_context_destroy(ctx);
        // Dependents keep working after the caller's context reference is gone.
        char err[128] = {0};
        CHECK_EQ(brass_module_verify(a, err, sizeof(err)), BRASS_OK);
        auto f = reinterpret_cast<AddFn>(brass_compiled_module_get_symbol(ca, "f"));
        REQUIRE(f != nullptr);
        CHECK_EQ(f(1), int64_t{i + 1});

        brass_builder_destroy(b1);
        brass_kernel_jit_destroy(kj);
        brass_module_destroy(a);
        brass_jit_destroy(jit);
    }
}

// Finding 4: builder misuse is an error with a message, not a silent success.
TEST_CASE("C-API lifetime - ret and builder misuse report errors") {
    BrassContext ctx = brass_context_create();
    BrassModule a = brass_module_create(ctx, "a");
    BrassModule other = brass_module_create(ctx, "other");
    BrassType i64_t = brass_type_i64();

    // A builder needs a live function.
    CHECK(brass_builder_create(ctx, nullptr) == nullptr);
    CHECK(has_error(ctx));

    BrassFunction fn = brass_function_create(a, "g", i64_t, &i64_t, 1);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassBuilder b = brass_builder_create(ctx, fn);
    REQUIRE(b != nullptr);

    // Not positioned yet: nothing may be built.
    brass_context_set_error(ctx, "");
    CHECK(brass_build_fconst_f32(b, 1.0f) == nullptr);
    CHECK(has_error(ctx));

    brass_builder_position_at_end(b, entry);
    BrassValue x = brass_function_get_param(fn, 0);

    // A value from another module is rejected.
    BrassFunction ofn = brass_function_create(other, "h", i64_t, &i64_t, 1);
    BrassValue ox = brass_function_get_param(ofn, 0);
    brass_context_set_error(ctx, "");
    CHECK(brass_build_add(b, x, ox) == nullptr);
    CHECK(has_error(ctx));

    // ret(NULL) in a non-void function, and a mistyped return value.
    brass_context_set_error(ctx, "");
    CHECK_EQ(brass_build_ret(b, nullptr), BRASS_ERR_INVALID_ARGUMENT);
    CHECK(has_error(ctx));
    BrassValue f = brass_build_fconst_f64(b, 1.0);
    REQUIRE(f != nullptr);
    brass_context_set_error(ctx, "");
    CHECK_EQ(brass_build_ret(b, f), BRASS_ERR_INVALID_ARGUMENT);
    CHECK(has_error(ctx));
    CHECK_EQ(brass_build_ret(b, x), BRASS_OK);

    // ret(NULL) is accepted in a void function; a value there is not.
    BrassFunction vfn = brass_function_create(a, "v", brass_type_void(), nullptr, 0);
    BrassBlock ventry = brass_function_append_block(vfn, "entry");
    BrassBuilder vb = brass_builder_create(ctx, vfn);
    brass_builder_position_at_end(vb, ventry);
    BrassValue one = brass_build_iconst_i64(vb, 1);
    CHECK_EQ(brass_build_ret(vb, one), BRASS_ERR_INVALID_ARGUMENT);
    CHECK_EQ(brass_build_ret(vb, nullptr), BRASS_OK);

    char err[256] = {0};
    CHECK_EQ(brass_module_verify(a, err, sizeof(err)), BRASS_OK);

    brass_builder_destroy(vb);
    brass_builder_destroy(b);
    brass_module_destroy(other);
    brass_module_destroy(a);
    brass_context_destroy(ctx);
}

// Finding 5: per-module symbols, duplicate definitions rejected, destroy frees.
TEST_CASE("C-API lifetime - JIT symbols are per compiled module and collisions are errors") {
    BrassContext ctx = brass_context_create();
    BrassModule a = brass_module_create(ctx, "a");
    BrassModule b = brass_module_create(ctx, "b");
    BrassModule c = brass_module_create(ctx, "c");
    BrassBuilder b1 = build_addk(ctx, a, "f", 1);
    BrassBuilder b2 = build_addk(ctx, b, "f", 100);
    BrassBuilder b3 = build_addk(ctx, c, "g", 1000);

    BrassJitEngine jit = brass_jit_create(ctx);
    BrassCompiledModule ca = brass_jit_compile_module(jit, a);
    REQUIRE(ca != nullptr);

    // Same function name in a second module: hard error.
    brass_context_set_error(ctx, "");
    CHECK(brass_jit_compile_module(jit, b) == nullptr);
    CHECK(std::strstr(brass_context_get_last_error(ctx), "'f'") != nullptr);

    // Recompiling a live module: hard error.
    brass_context_set_error(ctx, "");
    CHECK(brass_jit_compile_module(jit, a) == nullptr);
    CHECK(has_error(ctx));

    // A module with distinct names coexists, and lookups are module-scoped;
    // compiling it must not disturb module a's code.
    BrassCompiledModule cc = brass_jit_compile_module(jit, c);
    REQUIRE(cc != nullptr);
    auto fa = reinterpret_cast<AddFn>(brass_compiled_module_get_symbol(ca, "f"));
    auto gc = reinterpret_cast<AddFn>(brass_compiled_module_get_symbol(cc, "g"));
    REQUIRE(fa != nullptr);
    REQUIRE(gc != nullptr);
    CHECK_EQ(fa(1), int64_t{2});
    CHECK_EQ(gc(1), int64_t{1001});
    CHECK(brass_compiled_module_get_symbol(cc, "f") == nullptr);
    CHECK(brass_compiled_module_get_symbol(ca, "g") == nullptr);
    CHECK(brass_jit_get_function_address(jit, "f") == reinterpret_cast<void*>(fa));
    CHECK(brass_jit_get_function_address(jit, "g") == reinterpret_cast<void*>(gc));

    // Destroy releases the module: lookups through it fail, its names are free.
    brass_compiled_module_destroy(ca);
    brass_context_set_error(ctx, "");
    CHECK(brass_compiled_module_get_symbol(ca, "f") == nullptr);
    CHECK(has_error(ctx));
    CHECK(brass_jit_get_function_address(jit, "f") == nullptr);

    BrassCompiledModule cb = brass_jit_compile_module(jit, b);
    REQUIRE(cb != nullptr);
    auto fb = reinterpret_cast<AddFn>(brass_compiled_module_get_symbol(cb, "f"));
    REQUIRE(fb != nullptr);
    CHECK_EQ(fb(1), int64_t{101});
    CHECK_EQ(gc(2), int64_t{1002});

    brass_builder_destroy(b1);
    brass_builder_destroy(b2);
    brass_builder_destroy(b3);
    brass_jit_destroy(jit);
    brass_module_destroy(a);
    brass_module_destroy(b);
    brass_module_destroy(c);
    brass_context_destroy(ctx);
}

#if defined(BRASS_TEST_SHARED_LIB)
// Finding 6: both C headers' functions are exported from the shared library.
TEST_CASE("C-API lifetime - embedding and public C API are exported from the shared library") {
    std::string load_err;
    auto lib = brass::target::DynamicLibrary::open(BRASS_TEST_SHARED_LIB, &load_err);
    if (!lib) std::cerr << "DynamicLibrary::open error: " << load_err << "\n";
    REQUIRE(lib != nullptr);

    static const char* const kNames[] = {
        // include/brass/embedding/brass_c_api.h (all 48)
        "brass_engine_create", "brass_engine_destroy", "brass_engine_register_symbol",
        "brass_embed_module_create", "brass_embed_module_destroy",
        "brass_embed_module_add_external_symbol", "brass_engine_compile_module",
        "brass_embed_compiled_module_destroy", "brass_embed_compiled_module_get_symbol",
        "brass_compiled_module_patch_const32", "brass_compiled_module_patch_const64",
        "brass_compiled_module_patch_call", "brass_compiled_module_patch_call_target",
        "brass_compiled_module_walk_stack", "brass_heap_create", "brass_heap_destroy",
        "brass_heap_bind", "brass_heap_allocate", "brass_heap_allocate_value", "brass_heap_collect",
        "brass_heap_add_root", "brass_heap_remove_root", "brass_heap_write_barrier",
        "brass_heap_set_stress", "brass_heap_get_stress", "brass_heap_collection_count",
        "brass_value_from_f64",
        "brass_value_from_i32", "brass_value_from_bool", "brass_value_null", "brass_value_undefined",
        "brass_value_from_gcref", "brass_value_from_pointer", "brass_value_from_raw",
        "brass_value_is_f64", "brass_value_is_i32", "brass_value_is_bool", "brass_value_is_null",
        "brass_value_is_undefined", "brass_value_is_gcref", "brass_value_is_pointer",
        "brass_value_as_f64", "brass_value_as_i32", "brass_value_as_bool", "brass_value_as_gcref",
        "brass_value_as_pointer", "brass_value_raw", "brass_value_update_gcref",
        // a sample of include/brass/brass_c_api.h
        "brass_context_create", "brass_builder_create", "brass_build_ret", "brass_jit_compile_module",
    };
    for (const char* name : kNames) {
        if (!lib->get_symbol(name)) std::cerr << "not exported: " << name << "\n";
        CHECK(lib->get_symbol(name) != nullptr);
    }

    using FromF64 = uint64_t (*)(double);
    using AsF64 = double (*)(uint64_t);
    auto from_f64 = lib->get_function<FromF64>("brass_value_from_f64");
    auto as_f64 = lib->get_function<AsF64>("brass_value_as_f64");
    REQUIRE(from_f64 != nullptr);
    REQUIRE(as_f64 != nullptr);
    CHECK_EQ(as_f64(from_f64(2.5)), 2.5);
}
#endif
