#include "test_framework.hpp"

#include <brass/cache/artifact_cache.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/verify/gpu_kernel_sandbox.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::cache;

namespace {

Function* build_valid_shared_kernel(Module& mod, const char* name, uint32_t local_x = 16) {
    Function* fn = mod.create_function(name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry   = b.append_block("entry");
    BasicBlock* in_b    = b.append_block("in_bounds");
    BasicBlock* out_b   = b.append_block("out_bounds");
    BasicBlock* store_b = b.append_block("store_ok");
    BasicBlock* done_b  = b.append_block("done");

    b.position_at_end(entry);
    Value* dst  = b.add_block_param(entry, Type::ptr());
    Value* src  = b.add_block_param(entry, Type::ptr());
    Value* n    = b.add_block_param(entry, Type::i32());

    Value* c0   = b.build_iconst_i32(0);
    Value* c1   = b.build_iconst_i32(1);
    Value* smem = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(static_cast<int32_t>(local_x))});
    Value* tid  = b.build_call("gpu_tid_x", Type::i32());
    Value* gid  = b.build_call("gpu_global_tid_x", Type::i32());

    Value* in_range = b.build_and(b.build_sge(gid, c0), b.build_slt(gid, n));
    b.build_br_if(in_range, in_b, {}, out_b, {});

    b.position_at_end(in_b);
    Value* x = b.build_load_indexed(Type::f32(), src, gid, 4);
    b.build_call("gpu_shared_store_f32_indexed", Type::void_type(), {smem, tid, x});
    b.build_br(out_b, {});

    b.position_at_end(out_b);
    b.build_call("gpu_sync", Type::void_type());
    b.build_br_if(in_range, store_b, {}, done_b, {});

    b.position_at_end(store_b);
    Value* peer = b.build_xor(tid, c1);
    Value* y = b.build_call("gpu_shared_load_f32_indexed", Type::f32(), {smem, peer});
    b.build_store_indexed(Type::f32(), dst, gid, 4, y);
    b.build_br(done_b, {});

    b.position_at_end(done_b);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

Function* build_divergent_barrier_kernel(Module& mod) {
    Function* fn = mod.create_function("div_bar", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry  = b.append_block("entry");
    BasicBlock* then_b = b.append_block("then");
    BasicBlock* merge  = b.append_block("merge");

    b.position_at_end(entry);
    Value* dst  = b.add_block_param(entry, Type::ptr());
    Value* tid  = b.build_call("gpu_tid_x", Type::i32());
    Value* cond = b.build_slt(tid, b.build_iconst_i32(8));
    b.build_br_if(cond, then_b, {}, merge, {});

    b.position_at_end(then_b);
    b.build_call("gpu_sync", Type::void_type());
    b.build_store_indexed(Type::f32(), dst, tid, 4, b.build_fconst_f32(1.0f));
    b.build_br(merge, {});

    b.position_at_end(merge);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

Function* build_shared_oob_kernel(Module& mod) {
    Function* fn = mod.create_function("shared_oob", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* dst  = b.add_block_param(entry, Type::ptr());
    Value* smem = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(8)});
    Value* tid  = b.build_call("gpu_tid_x", Type::i32()); // local_size_x=16 -> OOB for size-8 smem
    b.build_call("gpu_shared_store_f32_indexed", Type::void_type(), {smem, tid, b.build_fconst_f32(2.0f)});
    b.build_call("gpu_sync", Type::void_type());
    Value* v = b.build_call("gpu_shared_load_f32_indexed", Type::f32(), {smem, tid});
    b.build_store_indexed(Type::f32(), dst, tid, 4, v);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

Function* build_simple_add_mul(Module& mod, const char* name, bool mul_add) {
    Function* fn = mod.create_function(name, Type::i32(), {Type::i32(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i32());
    Value* y = b.add_block_param(entry, Type::i32());
    Value* r = mul_add
        ? b.build_add(x, y)
        : b.build_sub(b.build_add(x, y), b.build_iconst_i32(1));
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    return fn;
}

} // namespace

TEST_CASE("GPU Sandbox - Portability, Safety Proofs, and Equivalence Gate") {
    GpuSandboxOptions opts;
    opts.prover_opts.local_size_x = 16;
    opts.prover_opts.local_size_y = 1;
    opts.prover_opts.local_size_z = 1;
    opts.prover_opts.buffers = {
        GpuBufferSpec{0, 0, 2, 4, 4},
        GpuBufferSpec{1, 0, 2, 4, 4},
    };
    opts.spirv_opts.local_size_x = 16;

    // 1. Valid kernel passes full portability + SPIR-V + SMT memory/barrier gate.
    {
        Module mod("sb_ok");
        Function* fn = build_valid_shared_kernel(mod, "sb_ok", 16);
        const auto res = verify_gpu_kernel_sandbox(*fn, opts);
        if (!res.passed()) std::cerr << "[test_gpu_sandbox] diag: " << res.diagnostic << "\n";
        CHECK(res.portability_ok);
        CHECK(res.spirv_compiled);
        CHECK(res.spirv_parsed);
        CHECK(res.memory_safe);
        CHECK(res.barrier_uniform);
        CHECK(res.global_bounds_proven);
        CHECK(res.proof_tier == SandboxProofTier::FullSmt);
        CHECK(res.shared_accesses_proven >= 2);
        CHECK(res.barriers_proven >= 1);
        CHECK(res.passed());
        CHECK(res.violations.empty());
    }

    // 1b. StaticOnly tier when SMT prover is disabled; require_full_smt rejects it.
    {
        Module mod("sb_static");
        Function* fn = build_valid_shared_kernel(mod, "sb_static", 16);
        GpuSandboxOptions static_opts = opts;
        static_opts.run_smt_prover = false;
        const auto res = verify_gpu_kernel_sandbox(*fn, static_opts);
        CHECK(res.proof_tier == SandboxProofTier::StaticOnly);
        CHECK(!res.global_bounds_proven);
        CHECK(res.passed());

        static_opts.require_full_smt = true;
        const auto strict_res = verify_gpu_kernel_sandbox(*fn, static_opts);
        CHECK(strict_res.proof_tier == SandboxProofTier::StaticOnly);
        CHECK(!strict_res.passed());
        CHECK(!strict_res.violations.empty());
    }

    // 1c. StaticFallbackLoopBound tier when loop exceeds loop_unroll_bound.
    {
        Module mod("sb_loop");
        Function* fn = mod.create_function("sb_loop", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hdr   = b.create_block("hdr");
        BasicBlock* body  = b.create_block("body");
        BasicBlock* done  = b.create_block("done");
        fn->append_block(hdr);
        fn->append_block(body);
        fn->append_block(done);

        b.position_at_end(entry);
        Value* dst  = b.add_block_param(entry, Type::ptr());
        Value* smem = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(16)});
        Value* tid  = b.build_call("gpu_tid_x", Type::i32());
        b.build_br(hdr, {b.build_iconst_i32(0)});

        b.position_at_end(hdr);
        Value* idx = b.add_block_param(hdr, Type::i32());
        b.build_br_if(b.build_slt(idx, b.build_iconst_i32(20)), body, {}, done, {});

        b.position_at_end(body);
        b.build_call("gpu_shared_store_f32_indexed", Type::void_type(), {smem, tid, b.build_fconst_f32(1.0f)});
        b.build_br(hdr, {b.build_add(idx, b.build_iconst_i32(1))});

        b.position_at_end(done);
        Value* v = b.build_call("gpu_shared_load_f32_indexed", Type::f32(), {smem, tid});
        b.build_store_indexed(Type::f32(), dst, tid, 4, v);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        GpuSandboxOptions loop_opts = opts;
        loop_opts.prover_opts.buffers = {GpuBufferSpec{0, 64, -1, 4, 4}};
        loop_opts.prover_opts.loop_unroll_bound = 2;
        const auto res = verify_gpu_kernel_sandbox(*fn, loop_opts);
        CHECK(res.proof_tier == SandboxProofTier::StaticFallbackLoopBound);
        CHECK(!res.global_bounds_proven);
        CHECK(res.passed());

        loop_opts.require_full_smt = true;
        const auto strict_res = verify_gpu_kernel_sandbox(*fn, loop_opts);
        CHECK(strict_res.proof_tier == SandboxProofTier::StaticFallbackLoopBound);
        CHECK(!strict_res.passed());
        CHECK(!strict_res.violations.empty());
    }

    // 2. Divergent barrier rejected.
    {
        Module mod("sb_div");
        Function* fn = build_divergent_barrier_kernel(mod);
        const auto res = verify_gpu_kernel_sandbox(*fn, opts);
        CHECK(!res.portability_ok);
        CHECK(!res.passed());
        CHECK(!res.violations.empty());
    }

    // 3. Non-portable intrinsic rejected.
    {
        Module mod("sb_warp");
        Function* fn = mod.create_function("sb_warp", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* ptr = b.add_block_param(entry, Type::ptr());
        Value* wid = b.build_call("gpu_warpid", Type::i32());
        b.build_store(Type::i32(), ptr, 0, wid);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        const auto res = verify_gpu_kernel_sandbox(*fn, opts);
        CHECK(!res.portability_ok);
        CHECK(!res.passed());
    }

    // 4. Out-of-bounds shared-memory access caught by SMT prover.
    {
        Module mod("sb_oob");
        Function* fn = build_shared_oob_kernel(mod);
        GpuSandboxOptions oob_opts = opts;
        oob_opts.prover_opts.buffers = {GpuBufferSpec{0, 256, -1, 4, 4}};
        const auto res = verify_gpu_kernel_sandbox(*fn, oob_opts);
        CHECK(res.portability_ok);
        CHECK(!res.memory_safe);
        CHECK(!res.passed());
        CHECK(!res.violations.empty());
    }

    // 5. MIR equivalence reference check passes when equivalent and fails when altered.
    {
        Module m_ref("ref");
        Function* ref_fn = build_simple_add_mul(m_ref, "f", true);

        Module m_same("same");
        Function* same_fn = m_same.create_function("f", Type::i32(), {Type::i32(), Type::i32()});
        {
            Builder b(m_same);
            b.set_function(same_fn);
            BasicBlock* entry = b.append_block("entry");
            b.position_at_end(entry);
            Value* x = b.add_block_param(entry, Type::i32());
            Value* y = b.add_block_param(entry, Type::i32());
            b.build_ret(b.build_add(y, x));
            same_fn->rebuild_cfg_predecessors();
        }
        Module m_diff("diff");
        Function* diff_fn = build_simple_add_mul(m_diff, "f", false);

        GpuSandboxOptions eq_opts;
        eq_opts.verify_spirv = false;
        eq_opts.reference_mir = ref_fn;

        const auto ok_res = verify_gpu_kernel_sandbox(*same_fn, eq_opts);
        CHECK(ok_res.equivalence_checked);
        CHECK(ok_res.equivalence_ok);
        CHECK(ok_res.passed());

        const auto bad_res = verify_gpu_kernel_sandbox(*diff_fn, eq_opts);
        CHECK(bad_res.equivalence_checked);
        CHECK(!bad_res.equivalence_ok);
        CHECK(!bad_res.passed());
    }
}

TEST_CASE("ArtifactCache - Key Listing and Portable Bundle Export/Import") {
    const std::filesystem::path root = brass::test::scratch_dir() / "artifact_bundle_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    const std::filesystem::path src_dir = root / "src_cache";
    const std::filesystem::path dst_dir = root / "dst_cache";
    const std::filesystem::path bundle_path = root / "test.bacb";

    const CacheKey k1 = KeyBuilder("bundle/1").add_brass_codegen().add("k", "alpha").finish();
    const CacheKey k2 = KeyBuilder("bundle/1").add_brass_codegen().add("k", "beta").finish();
    const std::vector<uint8_t> raw1{1, 2, 3, 4, 5, 6, 7, 8};

    {
        ArtifactCache src(src_dir, 0);
        CHECK(src.put(k1, {BlobRef("spirv", std::span<const uint8_t>(raw1)), BlobRef("meta", "v1")}));
        CHECK(src.put(k2, {BlobRef("ptx", ".entry k() { ret; }")}));

        const auto keys = src.list_keys();
        CHECK(keys.size() == 2);

        BundleStats exp_stats;
        std::string err;
        CHECK(src.export_bundle(bundle_path, "fp=demo;arch=vulkan", &exp_stats, &err));
        CHECK(exp_stats.entries_written == 2);
        CHECK(exp_stats.bytes > 0);
    }

    {
        ArtifactCache dst(dst_dir, 0);
        BundleStats imp_stats;
        std::string meta;
        std::string err;
        CHECK(dst.import_bundle(bundle_path, &meta, &imp_stats, &err));
        CHECK(meta == "fp=demo;arch=vulkan");
        CHECK(imp_stats.entries_imported == 2);
        CHECK(imp_stats.entries_skipped == 0);

        const auto a1 = dst.get(k1);
        const auto a2 = dst.get(k2);
        CHECK(a1.has_value());
        CHECK(a2.has_value());
        CHECK(a1->text("meta") == "v1");
        CHECK(a1->bytes("spirv").size() == raw1.size());
        CHECK(a2->text("ptx") == ".entry k() { ret; }");

        // Second import skips already-present valid entries.
        BundleStats imp2;
        CHECK(dst.import_bundle(bundle_path, nullptr, &imp2, &err));
        CHECK(imp2.entries_imported == 0);
        CHECK(imp2.entries_skipped == 2);
    }

    // Corrupt bundle trailer/body must be rejected cleanly.
    {
        std::ifstream in(bundle_path, std::ios::binary);
        std::vector<uint8_t> corrupted((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(!corrupted.empty());
        corrupted[corrupted.size() / 2] ^= 0xFFu;
        const std::filesystem::path bad_bundle = root / "bad.bacb";
        std::ofstream out(bad_bundle, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(corrupted.data()), static_cast<std::streamsize>(corrupted.size()));
        out.close();

        ArtifactCache dst2(root / "dst_bad", 0);
        std::string err;
        CHECK(!dst2.import_bundle(bad_bundle, nullptr, nullptr, &err));
        CHECK(!err.empty());
        CHECK(dst2.list_keys().empty());
    }

    std::filesystem::remove_all(root, ec);
}
