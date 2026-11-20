#include "test_framework.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/verify/bv_solver.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::verify;

namespace {

ptx::Function lower_ptx(const Function& fn) {
    ptx::PtxISel isel;
    return isel.lower(fn);
}

spirv::Module lower_spirv(const Function& fn, const target::SpirvOptions& opts = {}) {
    spirv::Module m(opts.spirv_version);
    spirv::SpirvISel isel(m, opts);
    isel.lower(fn);
    return m;
}

spirv::Module make_2d_shared_spirv(bool out_of_bounds, bool chained_ac) {
    spirv::Module m;
    m.add_capability(spv::CapabilityShader);
    m.set_memory_model(spv::AddressingModelLogical, spv::MemoryModelGLSL450);

    spirv::Id void_t = m.t_void();
    spirv::Id u32_t = m.t_int(32, 0);
    spirv::Id uvec3_t = m.t_vector(u32_t, 3);
    spirv::Id ptr_in_uvec3 = m.t_pointer(spv::StorageClassInput, uvec3_t);
    spirv::Id ptr_in_u32 = m.t_pointer(spv::StorageClassInput, u32_t);
    spirv::Id lid_var = m.variable(ptr_in_uvec3, spv::StorageClassInput);
    m.decorate(lid_var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInLocalInvocationId)});

    spirv::Id row_t = m.t_array(u32_t, 33);
    spirv::Id tile_t = m.t_array(row_t, 32);
    spirv::Id ptr_wg_tile = m.t_pointer(spv::StorageClassWorkgroup, tile_t);
    spirv::Id ptr_wg_row = m.t_pointer(spv::StorageClassWorkgroup, row_t);
    spirv::Id ptr_wg_u32 = m.t_pointer(spv::StorageClassWorkgroup, u32_t);
    spirv::Id tile_var = m.variable(ptr_wg_tile, spv::StorageClassWorkgroup);

    spirv::Function& fn = m.add_function("main", void_t, m.t_function(void_t, {}));
    m.add_entry_point(spv::ExecutionModelGLCompute, fn.id, "main", {lid_var, tile_var});
    m.add_execution_mode(fn.id, spv::ExecutionModeLocalSize, {32, 8, 1});

    spirv::Block entry;
    entry.label = m.new_id();

    spirv::Id ac_x = m.new_id();
    entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_in_u32, ac_x).id(lid_var).id(m.c_u32(0)));
    spirv::Id lx = m.new_id();
    entry.insts.push_back(spirv::Inst(spv::OpLoad, u32_t, lx).id(ac_x));

    spirv::Id ac_y = m.new_id();
    entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_in_u32, ac_y).id(lid_var).id(m.c_u32(1)));
    spirv::Id ly = m.new_id();
    entry.insts.push_back(spirv::Inst(spv::OpLoad, u32_t, ly).id(ac_y));

    spirv::Id row_idx = out_of_bounds ? m.c_u32(32) : ly;
    spirv::Id elem_ptr = m.new_id();
    if (chained_ac) {
        spirv::Id row_ptr = m.new_id();
        entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_wg_row, row_ptr).id(tile_var).id(row_idx));
        entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_wg_u32, elem_ptr).id(row_ptr).id(lx));
    } else {
        entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_wg_u32, elem_ptr).id(tile_var).id(row_idx).id(lx));
    }
    entry.insts.push_back(spirv::Inst(spv::OpStore).id(elem_ptr).id(lx));
    entry.insts.push_back(spirv::Inst(spv::OpReturn));

    fn.blocks.push_back(std::move(entry));
    return m;
}

} // namespace

TEST_CASE("GPU Mem Safety - 2D Row-Major Bounds and OOB Detection") {
    Module mod("gpu_2d_row_major");
    Function* fn = mod.create_function(
        "write_2d", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    {
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* buf = b.add_block_param(entry, Type::ptr());
        Value* rows = b.add_block_param(entry, Type::i32());
        Value* cols = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* r = kb.tid_y();
        Value* c = kb.tid_x();
        Value* in_bounds = b.build_and(b.build_ult(r, rows), b.build_ult(c, cols));
        kb.if_then(in_bounds, [&] {
            Value* idx = b.build_add(b.build_mul(r, cols), c);
            kb.store_i32_indexed(buf, idx, kb.const_i32(1));
        });
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
    }

    GpuProverOptions opts;
    opts.local_size_x = 16;
    opts.local_size_y = 8;
    opts.grid_size_x = 1;
    opts.buffers = {GpuBufferSpec{0, 0, 1, 4, 16, 2, -1, 1}};

    GpuMemorySafetyResult r_mir = verify_gpu_memory_safety(*fn, opts);
    CHECK(r_mir.is_safe());
    CHECK_EQ(r_mir.global_accesses_proven, 1u);

    target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 16;
    spv_opts.local_size_y = 8;
    GpuMemorySafetyResult r_spv = verify_spirv_memory_safety(lower_spirv(*fn, spv_opts), opts);
    CHECK(r_spv.is_safe());
    CHECK_EQ(r_spv.global_accesses_proven, 1u);

    GpuMemorySafetyResult r_ptx = verify_ptx_memory_safety(lower_ptx(*fn), opts);
    CHECK(r_ptx.is_safe());
    CHECK_EQ(r_ptx.global_accesses_proven, 1u);

    // Negative test: off-by-one r <= rows should produce a Counterexample
    Module oob_mod("gpu_2d_oob");
    Function* oob_fn = oob_mod.create_function(
        "write_2d_oob", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    {
        KernelBuilder kb(oob_mod, oob_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* buf = b.add_block_param(entry, Type::ptr());
        Value* rows = b.add_block_param(entry, Type::i32());
        Value* cols = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* r = kb.tid_y();
        Value* c = kb.tid_x();
        Value* bad_guard = b.build_and(b.build_ule(r, rows), b.build_ult(c, cols));
        kb.if_then(bad_guard, [&] {
            Value* idx = b.build_add(b.build_mul(r, cols), c);
            kb.store_i32_indexed(buf, idx, kb.const_i32(1));
        });
        b.build_ret_void();
        oob_fn->rebuild_cfg_predecessors();
    }

    GpuMemorySafetyResult oob_mir = verify_gpu_memory_safety(*oob_fn, opts);
    CHECK(oob_mir.is_counterexample());
    CHECK(!oob_mir.violations.empty());
    CHECK(verify_spirv_memory_safety(lower_spirv(*oob_fn, spv_opts), opts).is_counterexample());
    CHECK(verify_ptx_memory_safety(lower_ptx(*oob_fn), opts).is_counterexample());
}

TEST_CASE("GPU Mem Safety - 2D Pitch-Strided Copy with GpuParamConstraint") {
    Module mod("gpu_copy2d");
    Function* fn = mod.create_function(
        "copy_2d",
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    {
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* src = b.add_block_param(entry, Type::ptr());
        Value* dst = b.add_block_param(entry, Type::ptr());
        Value* src_pitch = b.add_block_param(entry, Type::i32());
        Value* dst_pitch = b.add_block_param(entry, Type::i32());
        Value* width = b.add_block_param(entry, Type::i32());
        Value* height = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* r = kb.tid_y();
        Value* c = kb.tid_x();
        Value* in_bounds = b.build_and(b.build_ult(r, height), b.build_ult(c, width));
        kb.if_then(in_bounds, [&] {
            Value* si = b.build_add(b.build_mul(r, src_pitch), c);
            Value* di = b.build_add(b.build_mul(r, dst_pitch), c);
            Value* v = kb.load_f32_indexed(src, si);
            kb.store_f32_indexed(dst, di, v);
        });
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
    }

    GpuProverOptions opts;
    opts.local_size_x = 16;
    opts.local_size_y = 8;
    opts.grid_size_x = 1;
    opts.buffers = {
        GpuBufferSpec{0, 0, 5, 4, 16, 2, -1, 1},
        GpuBufferSpec{1, 0, 5, 4, 16, 3, -1, 1},
    };
    opts.param_constraints = {
        GpuParamConstraint{4, 2, 0, 0}, // width <= src_pitch
        GpuParamConstraint{4, 3, 0, 0}, // width <= dst_pitch
    };

    GpuMemorySafetyResult r_mir = verify_gpu_memory_safety(*fn, opts);
    CHECK(r_mir.is_safe());
    CHECK_EQ(r_mir.global_accesses_proven, 2u);

    target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 16;
    spv_opts.local_size_y = 8;
    GpuMemorySafetyResult r_spv = verify_spirv_memory_safety(lower_spirv(*fn, spv_opts), opts);
    CHECK(r_spv.is_safe());
    CHECK_EQ(r_spv.global_accesses_proven, 2u);

    // Without width <= dst_pitch constraint, dst write can exceed height * dst_pitch
    opts.param_constraints.pop_back();
    CHECK(verify_gpu_memory_safety(*fn, opts).is_counterexample());
}

TEST_CASE("GPU Mem Safety - Modulo Broadcast and 3D Batched Row-Major") {
    // 1. Modulo broadcast: y[i] = x[i] + bias[i % d] with d >= 1
    {
        Module mod("gpu_mod_bcast");
        Function* fn = mod.create_function(
            "add_bias", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* y = b.add_block_param(entry, Type::ptr());
        Value* x = b.add_block_param(entry, Type::ptr());
        Value* bias = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        Value* d = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* i = kb.tid_x();
        kb.if_then(b.build_ult(i, n), [&] {
            Value* j = b.build_umod(i, d);
            Value* xv = kb.load_f32_indexed(x, i);
            Value* bv = kb.load_f32_indexed(bias, j);
            kb.store_f32_indexed(y, i, b.build_add(xv, bv));
        });
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        GpuProverOptions opts;
        opts.local_size_x = 64;
        opts.grid_size_x = 1;
        opts.buffers = {
            GpuBufferSpec{0, 0, 3, 4, 16},
            GpuBufferSpec{1, 0, 3, 4, 16},
            GpuBufferSpec{2, 0, 4, 4, 16},
        };
        opts.param_constraints = {GpuParamConstraint{4, UINT32_MAX, 1, 0}}; // d >= 1

        GpuMemorySafetyResult r_mir = verify_gpu_memory_safety(*fn, opts);
        CHECK(r_mir.is_safe());
        CHECK_EQ(r_mir.global_accesses_proven, 3u);

        target::SpirvOptions spv_opts;
        spv_opts.local_size_x = 64;
        GpuMemorySafetyResult r_spv = verify_spirv_memory_safety(lower_spirv(*fn, spv_opts), opts);
        CHECK(r_spv.is_safe());
        CHECK_EQ(r_spv.global_accesses_proven, 3u);

        GpuMemorySafetyResult r_ptx = verify_ptx_memory_safety(lower_ptx(*fn), opts);
        CHECK(r_ptx.is_safe());
        CHECK_EQ(r_ptx.global_accesses_proven, 3u);
    }

    // 2. 3D batched row-major: buf[b * (rows * cols) + r * cols + c]
    {
        Module mod("gpu_3d_batch");
        Function* fn = mod.create_function(
            "write_3d", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32(), Type::i32()});
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* buf = b.add_block_param(entry, Type::ptr());
        Value* batch = b.add_block_param(entry, Type::i32());
        Value* rows = b.add_block_param(entry, Type::i32());
        Value* cols = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* bz = kb.ctaid_z();
        Value* r = kb.tid_y();
        Value* c = kb.tid_x();
        Value* guard = b.build_and(
            b.build_ult(bz, batch),
            b.build_and(b.build_ult(r, rows), b.build_ult(c, cols)));
        kb.if_then(guard, [&] {
            Value* plane = b.build_mul(rows, cols);
            Value* idx = b.build_add(b.build_add(b.build_mul(bz, plane), b.build_mul(r, cols)), c);
            kb.store_i32_indexed(buf, idx, kb.const_i32(7));
        });
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        GpuProverOptions opts;
        opts.local_size_x = 16;
        opts.local_size_y = 8;
        opts.grid_size_x = 1;
        opts.grid_size_z = 4;
        opts.buffers = {GpuBufferSpec{0, 0, 1, 4, 16, 2, 3, 1}};

        GpuMemorySafetyResult r_mir = verify_gpu_memory_safety(*fn, opts);
        CHECK(r_mir.is_safe());
        CHECK_EQ(r_mir.global_accesses_proven, 1u);

        target::SpirvOptions spv_opts;
        spv_opts.local_size_x = 16;
        spv_opts.local_size_y = 8;
        GpuMemorySafetyResult r_spv = verify_spirv_memory_safety(lower_spirv(*fn, spv_opts), opts);
        CHECK(r_spv.is_safe());
        CHECK_EQ(r_spv.global_accesses_proven, 1u);

        GpuMemorySafetyResult r_ptx = verify_ptx_memory_safety(lower_ptx(*fn), opts);
        CHECK(r_ptx.is_safe());
        CHECK_EQ(r_ptx.global_accesses_proven, 1u);
    }
}

TEST_CASE("GPU Mem Safety - Multi-Dimensional SPIR-V Workgroup Shared Arrays") {
    GpuProverOptions opts;
    opts.local_size_x = 32;
    opts.local_size_y = 8;
    opts.grid_size_x = 1;
    opts.check_global_bounds = false;

    GpuMemorySafetyResult safe_direct = verify_spirv_memory_safety(make_2d_shared_spirv(false, false), opts);
    CHECK(safe_direct.is_safe());
    CHECK_EQ(safe_direct.shared_accesses_proven, 1u);

    GpuMemorySafetyResult safe_chained = verify_spirv_memory_safety(make_2d_shared_spirv(false, true), opts);
    CHECK(safe_chained.is_safe());
    CHECK_EQ(safe_chained.shared_accesses_proven, 1u);

    GpuMemorySafetyResult oob = verify_spirv_memory_safety(make_2d_shared_spirv(true, false), opts);
    CHECK(oob.is_counterexample());
    CHECK(!oob.violations.empty());
    CHECK(oob.violations[0].is_shared);
}

TEST_CASE("GPU Mem Safety - BVContext make_extract Memoizes Reconvergent ITE Diamond DAGs") {
    verify::BVContext ctx;
    verify::BVNodeId x = ctx.make_var("leaf_x", 32);
    verify::BVNodeId cur = x;
    for (int level = 0; level < 32; ++level) {
        verify::BVNodeId cond = ctx.make_var("c_" + std::to_string(level), 1);
        verify::BVNodeId next = ctx.make_binary(verify::BVOp::Add, cur, ctx.make_const(1, 32), 32);
        cur = ctx.make_ite(cond, next, cur);
    }
    verify::BVNodeId lo_byte = ctx.make_extract(cur, 7, 0);
    verify::BVNodeId hi_byte = ctx.make_extract(cur, 15, 8);
    CHECK_EQ(ctx.width(lo_byte), 8u);
    CHECK_EQ(ctx.width(hi_byte), 8u);
    CHECK(ctx.nodes().size() < 2000u);

    // Every path adds at most 32 to x, so zeroed lower bits of x can never exceed 32
    verify::BVNodeId even_x = ctx.make_binary(verify::BVOp::And, x, ctx.make_const(0xFFFFFF00u, 32), 32);
    verify::BVNodeId dcur = even_x;
    for (int level = 0; level < 32; ++level) {
        verify::BVNodeId cond = ctx.make_var("d_" + std::to_string(level), 1);
        verify::BVNodeId next = ctx.make_binary(verify::BVOp::Add, dcur, ctx.make_const(1, 32), 32);
        dcur = ctx.make_ite(cond, next, dcur);
    }
    verify::BVNodeId d_lo = ctx.make_extract(dcur, 7, 0);
    verify::BVNodeId bad = ctx.make_binary(verify::BVOp::Ugt, d_lo, ctx.make_const(32, 8), 1);
    CHECK_EQ(verify::solve_nonzero(ctx, bad, 2000).status, verify::SolveOutcome::Unsat);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
