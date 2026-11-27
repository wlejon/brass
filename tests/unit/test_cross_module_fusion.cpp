#include "test_framework.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cross_module.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#if BRASS_WITH_SPIRV
#include <brass/target/spirv_target.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <string>

using namespace brass;

namespace {

bool contains_tagged_or_gcref(const Function& fn) {
    auto is_gc = [](Type t) {
        return t.kind() == TypeKind::Tagged || t.kind() == TypeKind::GCRef;
    };
    if (is_gc(fn.return_type())) return true;
    for (Type pt : fn.param_types()) {
        if (is_gc(pt)) return true;
    }
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Value* p : bb->params()) {
            if (p && is_gc(p->type())) return true;
        }
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (is_gc(inst->type())) return true;
            for (const Value* op : inst->operands()) {
                if (op && is_gc(op->type())) return true;
            }
        }
    }
    return false;
}

} // namespace

TEST_CASE("CrossModule - Import Transitive Closure, Symbols, and DebugContext") {
    Module src("bronze_js");
    const uint32_t f_id = src.debug_context().get_or_add_file("script.js");

    src.add_external_symbol("pure_helper");
    src.add_symbol_role("pure_helper", SymbolRole::Pure);
    src.define_string_symbol("str_tag", "js_lambda");

    // Leaf helper: f(x) = x * 2.0 + pure_helper(x)
    Function* leaf = src.create_function("leaf_fn", Type::f64(), {Type::f64()});
    {
        Builder b(src);
        b.set_function(leaf);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        b.set_current_loc(f_id, 12, 4);
        Value* x = b.add_block_param(entry, Type::f64());
        Value* two = b.build_fconst_f64(2.0);
        Value* p = b.build_call("pure_helper", Type::f64(), {x});
        Value* res = b.build_fadd(b.build_mul(x, two), p);
        b.build_ret(res);
    }

    // Root lambda calls leaf_fn and references str_tag
    Function* lambda = src.create_function("helper_chain", Type::f64(), {Type::f64()});
    {
        Builder b(src);
        b.set_function(lambda);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        b.set_current_loc(f_id, 25, 8);
        Value* x = b.add_block_param(entry, Type::f64());
        Value* msg = b.build_func_addr("str_tag");
        (void)msg;
        Value* y = b.build_call("leaf_fn", Type::f64(), {x});
        b.build_ret(b.build_fadd(y, b.build_fconst_f64(1.0)));
    }

    Module dst("bro_table");
    const uint32_t tbl_file = dst.debug_context().get_or_add_file("query.sql");
    // Deliberate collision on name `leaf_fn` in dst
    Function* existing = dst.create_function("leaf_fn", Type::i32(), {Type::i32()});
    {
        Builder b(dst);
        b.set_function(existing);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        b.set_current_loc(tbl_file, 1, 1);
        Value* v = b.add_block_param(entry, Type::i32());
        b.build_ret(v);
    }

    CrossModuleLinkOptions link_opts;
    link_opts.symbol_prefix = "js_";
    const CrossModuleLinkResult lr = import_function_into_module(dst, src, "helper_chain", link_opts);
    CHECK(lr.ok);
    CHECK(lr.imported != nullptr);
    CHECK(lr.imported->name() == "js_helper_chain");
    CHECK(lr.imported_callees.size() == 1);
    CHECK(lr.imported_callees[0]->name() == "js_leaf_fn");
    CHECK(dst.has_external_symbol("pure_helper"));
    CHECK(dst.has_symbol_role("pure_helper", SymbolRole::Pure));
    CHECK(dst.string_symbol("str_tag") != nullptr);
    CHECK(std::string_view(dst.string_symbol("str_tag")) == "js_lambda");

    // Check DebugContext remapped `script.js` cleanly without colliding with `query.sql`
    const uint32_t remapped_file_id = dst.debug_context().get_file_id("script.js");
    CHECK(remapped_file_id != 0);
    CHECK(remapped_file_id != tbl_file);

    bool saw_remapped_loc = false;
    for (const BasicBlock* bb : lr.imported_callees[0]->blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->loc().is_valid()) {
                CHECK(inst->loc().file_id == remapped_file_id);
                const std::string fmt = format_debug_loc(dst.debug_context(), inst->loc());
                CHECK(fmt.find("script.js:12") != std::string::npos);
                saw_remapped_loc = true;
            }
        }
    }
    CHECK(saw_remapped_loc);
    CHECK(verify_module(dst));
}

TEST_CASE("CrossModule - Fuse NaN-Boxed Tagged JS Lambda into Unboxed Loop & JIT") {
    Module js_mod("bronze_runtime");
    const uint32_t js_file = js_mod.debug_context().get_or_add_file("transform.js");

    // NaN-boxed lambda: (Tagged x) -> Tagged, computes x * 1.5 + 2.0 unboxed inside
    Function* js_fn = js_mod.create_function("js_scale_bias", Type::tagged(), {Type::tagged()});
    {
        Builder b(js_mod);
        b.set_function(js_fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        b.set_current_loc(js_file, 3, 2);
        Value* xt = b.add_block_param(entry, Type::tagged());
        Value* xd = b.build_bitcast_f64_i64(b.build_bitcast_i64_tagged(xt));
        Value* yd = b.build_fadd(b.build_mul(xd, b.build_fconst_f64(1.5)), b.build_fconst_f64(2.0));
        Value* yt = b.build_bitcast_tagged_i64(b.build_bitcast_i64_f64(yd));
        b.build_ret(yt);
    }

    // Host numeric loop module (raw f64 array sum over lambda(arr[i]))
    Module host_mod("table_scan");
    Function* scan = host_mod.create_function("sum_transformed", Type::f64(), {Type::ptr(), Type::i64()});
    Instruction* call_to_fuse = nullptr;
    {
        Builder b(host_mod);
        b.set_function(scan);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop = b.append_block("loop");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        Value* arr = b.add_block_param(entry, Type::ptr());
        Value* count = b.add_block_param(entry, Type::i64());
        b.build_br(loop, {b.build_iconst_i64(0), b.build_fconst_f64(0.0)});

        b.position_at_end(loop);
        Value* i = b.add_block_param(loop, Type::i64());
        Value* acc = b.add_block_param(loop, Type::f64());
        b.build_br_if(b.build_slt(i, count), body, {}, exit, {});

        b.position_at_end(body);
        Value* v = b.build_load_indexed(Type::f64(), arr, i, 8);
        Value* mapped = b.build_call("js_scale_bias", Type::f64(), {v});
        call_to_fuse = mapped->defining_instruction();
        Value* next_acc = b.build_fadd(acc, mapped);
        Value* next_i = b.build_add(i, b.build_iconst_i64(1));
        b.build_br(loop, {next_i, next_acc});

        b.position_at_end(exit);
        b.build_ret(acc);
    }
    scan->rebuild_cfg_predecessors();

    CrossDomainFusionOptions fopts;
    fopts.audit_opts.require_no_gc = true;
    fopts.audit_opts.require_no_pinned_tls = true;
    fopts.audit_opts.require_no_alloc = true;
    const CrossDomainFusionResult fres = fuse_cross_module_call(
        *scan, call_to_fuse, host_mod, js_mod, "js_scale_bias", fopts);
    CHECK(fres.fused);
    CHECK(fres.audit.ok);
    CHECK(!contains_tagged_or_gcref(*scan));
    CHECK(verify_function(*scan));

    // Compile fused numeric loop natively with KernelJit and execute
    codegen::KernelJit jit;
    auto compiled = jit.compile(*scan);
    CHECK(compiled.is_valid());
    using FnPtr = double (*)(const double*, int64_t);
    FnPtr fn = compiled.as<FnPtr>();
    const double input[4] = {1.0, 2.0, 3.0, 4.0};
    // (1.5*1+2) + (1.5*2+2) + (1.5*3+2) + (1.5*4+2) = 3.5 + 5.0 + 6.5 + 8.0 = 23.0
    const double got = fn(input, 4);
    CHECK(std::fabs(got - 23.0) < 1e-12);
}

TEST_CASE("CrossModule - Reject Allocation/TLS JS Lambda with Source Location") {
    Module js_mod("bronze_runtime");
    js_mod.set_pinned_tls_register(true);
    js_mod.add_external_symbol("gc_alloc_obj");
    js_mod.add_symbol_role("gc_alloc_obj", SymbolRole::Allocator);
    const uint32_t js_file = js_mod.debug_context().get_or_add_file("script.js");

    Function* bad_lambda = js_mod.create_function("bad_alloc_fn", Type::tagged(), {Type::tagged()});
    {
        Builder b(js_mod);
        b.set_function(bad_lambda);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* xt = b.add_block_param(entry, Type::tagged());
        b.set_current_loc(js_file, 42, 7);
        Value* tls = b.build_pinned_tls_read();
        (void)tls;
        Value* obj = b.build_call("gc_alloc_obj", Type::tagged(), {xt});
        b.build_ret(obj);
    }

    Module host_mod("bro_dsp");
    Function* host_fn = host_mod.create_function("dsp_step", Type::f64(), {Type::f64()});
    Instruction* call_inst = nullptr;
    {
        Builder b(host_mod);
        b.set_function(host_fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::f64());
        Value* r = b.build_call("bad_alloc_fn", Type::f64(), {x});
        call_inst = r->defining_instruction();
        b.build_ret(r);
    }

    const size_t before_blocks = host_fn->block_count();
    CrossDomainFusionOptions fopts;
    const CrossDomainFusionResult fres = fuse_cross_module_call(
        *host_fn, call_inst, host_mod, js_mod, "bad_alloc_fn", fopts);
    CHECK(!fres.fused);
    CHECK(!fres.audit.ok);
    CHECK(fres.audit.violations.size() >= 2);
    CHECK(host_fn->block_count() == before_blocks);
    CHECK(fres.diagnostic.find("script.js:42") != std::string::npos);
}

TEST_CASE("CrossModule - Fuse Helper from CPU Module into Portable GPU Kernel") {
    Module math_mod("shared_math");
    Function* act = math_mod.create_function("fast_gelu_approx", Type::f32(), {Type::f32()});
    {
        Builder b(math_mod);
        b.set_function(act);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* x = b.add_block_param(entry, Type::f32());
        Value* half = b.build_fconst_f32(0.5f);
        Value* one = b.build_fconst_f32(1.0f);
        Value* clamped = b.build_fmax_f32(b.build_fconst_f32(-3.0f), b.build_fmin_f32(b.build_fconst_f32(3.0f), x));
        Value* s = b.build_mul(x, b.build_mul(half, b.build_fadd(one, b.build_mul(clamped, b.build_fconst_f32(0.25f)))));
        b.build_ret(s);
    }

    Module gpu_mod("kernel_mod");
    Function* k = gpu_mod.create_function("apply_act", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    Instruction* call_inst = nullptr;
    {
        Builder b(gpu_mod);
        b.set_function(k);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* work = b.append_block("work");
        BasicBlock* done = b.append_block("done");

        b.position_at_end(entry);
        Value* dst = b.add_block_param(entry, Type::ptr());
        Value* src = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        Value* gid = b.build_call("gpu_global_tid_x", Type::i32());
        b.build_br_if(b.build_slt(gid, n), work, {}, done, {});

        b.position_at_end(work);
        Value* v = b.build_load_indexed(Type::f32(), src, gid, 4);
        Value* y = b.build_call("fast_gelu_approx", Type::f32(), {v});
        call_inst = y->defining_instruction();
        b.build_store_indexed(Type::f32(), dst, gid, 4, y);
        b.build_br(done);

        b.position_at_end(done);
        b.build_ret_void();
    }
    k->rebuild_cfg_predecessors();

    CrossDomainFusionOptions fopts;
    fopts.audit_opts.require_gpu_portable = true;
    const CrossDomainFusionResult fres = fuse_cross_module_call(
        *k, call_inst, gpu_mod, math_mod, "fast_gelu_approx", fopts);
    CHECK(fres.fused);
    CHECK(verify_gpu_portability(*k));
#if BRASS_WITH_SPIRV
    const auto spv = target::SpirvTarget::compile(*k);
    CHECK(!spv.words.empty());
#endif
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
