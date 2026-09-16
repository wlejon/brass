#include "test_framework.hpp"
#include <brass/target/ptx_target.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/brass_c_api.h>

#include <string>
#include <cstring>

using namespace brass;
using namespace brass::target;

TEST_CASE("PTX Target - Version, SM Architecture, and Function Signature") {
    Module mod("test_mod");
    Function* fn = mod.create_function("vector_scale", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::f32(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.build_ret_void();

    PtxOptions opts;
    opts.sm_arch = "sm_80";
    opts.ptx_version_major = 7;
    opts.ptx_version_minor = 5;

    std::string ptx = PtxTarget::emit_function(*fn, opts);

    CHECK(ptx.find(".version 7.5") != std::string::npos);
    CHECK(ptx.find(".target sm_80") != std::string::npos);
    CHECK(ptx.find(".address_size 64") != std::string::npos);
    CHECK(ptx.find(".visible .entry vector_scale(") != std::string::npos);
    CHECK(ptx.find(".param .u64 param_0") != std::string::npos);
    CHECK(ptx.find(".param .u64 param_1") != std::string::npos);
    CHECK(ptx.find(".param .f32 param_2") != std::string::npos);
    CHECK(ptx.find(".param .u64 param_3") != std::string::npos);
    CHECK(ptx.find("ret;") != std::string::npos);
}

// The results of these kernels are stored through a pointer parameter: the
// ptx::cleanup pass deletes side-effect-free instructions whose result is
// never read, so an unused fma/rsqrt/%tid.x would (correctly) not be printed.

TEST_CASE("PTX Target - Arithmetic and Exact Hex Floating Literals") {
    Module mod("test_math");
    Function* fn = mod.create_function("fma_kernel", Type::void_type(), {
        Type::f32(), Type::f32(), Type::ptr()
    });

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::f32());
    b.add_block_param(entry, Type::f32());
    b.add_block_param(entry, Type::ptr());

    Value* a = entry->param(0);
    Value* scale = entry->param(1);
    Value* one = b.build_fconst_f32(1.0f);
    Value* res = b.build_fma_f32(a, scale, one);
    b.build_store(Type::f32(), entry->param(2), 0, res);
    b.build_ret_void();

    std::string ptx = PtxTarget::emit_function(*fn);

    // Exact IEEE 754 float format: 1.0f is 0f3F800000, folded into the fma
    // as an immediate (no mov.f32 of the constant remains).
    CHECK(ptx.find("fma.rn.f32 %f2, %f0, %f1, 0f3F800000;") != std::string::npos);
    CHECK(ptx.find("mov.f32") == std::string::npos);
}

TEST_CASE("PTX Target - Fast Math Approximations") {
    Module mod("test_transcendental");
    Function* fn = mod.create_function("math_kernel", Type::void_type(), {
        Type::f32(), Type::ptr()
    });

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::f32());
    b.add_block_param(entry, Type::ptr());

    Value* x = entry->param(0);
    Value* r1 = b.build_call("rsqrtf", Type::f32(), {x});
    Value* r2 = b.build_call("sqrtf", Type::f32(), {r1});
    Value* r3 = b.build_call("expf", Type::f32(), {r2});
    Value* r4 = b.build_call("sinf", Type::f32(), {r3});
    Value* r5 = b.build_call("cosf", Type::f32(), {r4});
    b.build_store(Type::f32(), entry->param(1), 0, r5);
    b.build_ret_void();

    std::string ptx = PtxTarget::emit_function(*fn);

    CHECK(ptx.find("rsqrt.approx.f32") != std::string::npos);
    CHECK(ptx.find("sqrt.approx.f32") != std::string::npos);
    CHECK(ptx.find("ex2.approx.f32") != std::string::npos);
    CHECK(ptx.find("sin.approx.f32") != std::string::npos);
    CHECK(ptx.find("cos.approx.f32") != std::string::npos);
    // Exponential constant log2(e) = 1.44269504f = 0f3FB8AA3B
    CHECK(ptx.find("0f3FB8AA3B") != std::string::npos);
}

TEST_CASE("PTX Target - Thread Indexing Builtins") {
    Module mod("test_indexing");
    Function* fn = mod.create_function("grid_kernel", Type::void_type(), {
        Type::ptr()
    });

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());

    Value* tid = b.build_call("ptx_tid_x", Type::i32(), {});
    Value* gid = b.build_call("ptx_global_tid_x", Type::i32(), {});
    b.build_store(Type::i32(), out, 0, tid);
    b.build_store(Type::i32(), out, 4, gid);
    b.build_ret_void();

    std::string ptx = PtxTarget::emit_function(*fn);

    // Each special register is read exactly once (in the $L_params prologue)
    // even though tid_x and global_tid_x both need %tid.x.
    auto count = [&](const char* needle) {
        size_t n = 0;
        for (size_t p = ptx.find(needle); p != std::string::npos; p = ptx.find(needle, p + 1)) ++n;
        return n;
    };
    CHECK_EQ(count("%tid.x"), size_t(1));
    CHECK_EQ(count("%ctaid.x"), size_t(1));
    CHECK_EQ(count("%ntid.x"), size_t(1));
    CHECK(ptx.find("$L_params:") != std::string::npos);
    CHECK(ptx.find("mad.lo.s32") != std::string::npos);
}

TEST_CASE("PTX Target - C-API Emission and Cleanup") {
    BrassContext ctx = brass_context_create();
    CHECK(ctx != nullptr);

    BrassModule mod = brass_module_create(ctx, "c_api_ptx_mod");
    CHECK(mod != nullptr);

    BrassType ptr_t = brass_type_ptr();
    BrassType f32_t = brass_type_f32();
    BrassType i64_t = brass_type_i64();
    BrassType params[3] = {ptr_t, f32_t, i64_t};

    BrassFunction fn = brass_function_create(
        mod,
        "c_api_kernel",
        brass_type_void(),
        params,
        3
    );
    CHECK(fn != nullptr);

    BrassBuilder b = brass_builder_create(ctx, fn);
    BrassBlock entry = brass_function_append_block(fn, "entry");
    brass_builder_position_at_end(b, entry);
    brass_build_ret(b, nullptr);

    char* ptx_out = nullptr;
    size_t ptx_len = 0;
    BrassStatus st = brass_kernel_emit_ptx(fn, "sm_89", &ptx_out, &ptx_len);
    CHECK(st == BRASS_OK);
    CHECK(ptx_out != nullptr);
    CHECK(ptx_len > 0);
    CHECK(std::strstr(ptx_out, ".target sm_89") != nullptr);
    CHECK(std::strstr(ptx_out, ".visible .entry c_api_kernel(") != nullptr);

    brass_free_string(ptx_out);
    brass_builder_destroy(b);
    brass_module_destroy(mod);
    brass_context_destroy(ctx);
}

TEST_CASE("PTX Target - Regular Device Function Calls and Modulo/Bitwise Ops") {
    Module mod("test_ptx_calls");
    Function* fn = mod.create_function("caller_kernel", Type::void_type(), {Type::i32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::i32());
    b.add_block_param(entry, Type::i32());

    Value* x = entry->param(0);
    Value* y = entry->param(1);
    Value* smod_val = b.build_smod(x, y);
    Value* not_val = b.build_not(smod_val);
    Value* res = b.build_call("custom_device_helper", Type::i32(), {not_val, y});
    (void)res;
    b.build_ret_void();

    std::string ptx = PtxTarget::emit_function(*fn);

    CHECK(ptx.find("rem.s32") != std::string::npos);
    CHECK(ptx.find("not.b32") != std::string::npos);
    CHECK(ptx.find("call (%r") != std::string::npos);
    CHECK(ptx.find("custom_device_helper, (%r") != std::string::npos);
}

TEST_CASE("PTX Target - Unsupported Opcode Diagnostic Error") {
    Module mod("test_ptx_unsupported");
    Function* fn = mod.create_function("invalid_kernel", Type::void_type(), {});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    // coro_create is not supported on PTX
    Instruction* bad_inst = mod.arena().make<Instruction>(Opcode::coro_create, Type::ptr());
    Value* bad_val = mod.arena().make<Value>(fn->next_value_id(), Type::ptr(), ValueKind::InstructionResult);
    bad_val->set_defining_instruction(bad_inst);
    bad_inst->set_result(bad_val);
    entry->append_instruction(bad_inst);
    b.build_ret_void();

    bool threw = false;
    try {
        PtxTarget::emit_function(*fn);
    } catch (const std::runtime_error& err) {
        threw = true;
        CHECK(std::string(err.what()).find("unsupported opcode") != std::string::npos);
    }
    CHECK(threw);
}

TEST_CASE("PTX Target - Non-void kernels are rejected") {
    // PTX `ret` takes no operand and .entry kernels cannot return values, so
    // emitting `ret %r1` would be rejected by ptxas. Refuse up front instead.
    Module mod("test_mod");
    Function* fn = mod.create_function("ret_val", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::i32());
    b.build_ret(entry->param(0));

    bool threw = false;
    try {
        PtxTarget::emit_function(*fn);
    } catch (const std::runtime_error& err) {
        threw = true;
        CHECK(std::string(err.what()).find("cannot return values") != std::string::npos);
    }
    CHECK(threw);
}

