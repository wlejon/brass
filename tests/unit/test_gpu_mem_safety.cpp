#include "test_framework.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/verify/bv_solver.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <spirv/unified1/GLSL.std.450.h>

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

TEST_CASE("GPU Mem Safety - SPIR-V CHK Dominators Multi-Exit and Loop Control Dependence") {
    auto build_cfg_shader = [](bool divergent_barrier) {
        spirv::Module m;
        m.add_capability(spv::CapabilityShader);
        m.set_memory_model(spv::AddressingModelLogical, spv::MemoryModelGLSL450);

        spirv::Id void_t = m.t_void();
        spirv::Id bool_t = m.t_bool();
        spirv::Id u32_t = m.t_int(32, 0);
        spirv::Id uvec3_t = m.t_vector(u32_t, 3);
        spirv::Id ptr_in_uvec3 = m.t_pointer(spv::StorageClassInput, uvec3_t);
        spirv::Id ptr_in_u32 = m.t_pointer(spv::StorageClassInput, u32_t);
        spirv::Id lid_var = m.variable(ptr_in_uvec3, spv::StorageClassInput);
        m.decorate(lid_var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInLocalInvocationId)});

        spirv::Id arr_t = m.t_array(u32_t, 32);
        spirv::Id ptr_wg_arr = m.t_pointer(spv::StorageClassWorkgroup, arr_t);
        spirv::Id ptr_wg_u32 = m.t_pointer(spv::StorageClassWorkgroup, u32_t);
        spirv::Id sh_var = m.variable(ptr_wg_arr, spv::StorageClassWorkgroup);

        spirv::Function& fn = m.add_function("main", void_t, m.t_function(void_t, {}));
        m.add_entry_point(spv::ExecutionModelGLCompute, fn.id, "main", {lid_var, sh_var});
        m.add_execution_mode(fn.id, spv::ExecutionModeLocalSize, {32, 1, 1});

        spirv::Id b_entry = m.new_id(), b_early = m.new_id(), b_div_hdr = m.new_id(),
                  b_then = m.new_id(), b_loop_hdr = m.new_id(), b_loop_body = m.new_id(),
                  b_loop_cont = m.new_id(), b_after = m.new_id(), b_inf = m.new_id(),
                  b_exit2 = m.new_id();

        spirv::Block entry{b_entry, {}};
        spirv::Id ac_x = m.new_id(), lx = m.new_id(), c_unif_entry = m.new_id(), c_div = m.new_id();
        entry.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_in_u32, ac_x).id(lid_var).id(m.c_u32(0)));
        entry.insts.push_back(spirv::Inst(spv::OpLoad, u32_t, lx).id(ac_x));
        entry.insts.push_back(spirv::Inst(spv::OpIEqual, bool_t, c_unif_entry).id(m.c_u32(0)).id(m.c_u32(0)));
        entry.insts.push_back(spirv::Inst(spv::OpBranchConditional, 0, 0).id(c_unif_entry).id(b_div_hdr).id(b_early));

        spirv::Block early_blk{b_early, {}};
        early_blk.insts.push_back(spirv::Inst(spv::OpReturn, 0, 0));

        spirv::Block div_hdr{b_div_hdr, {}};
        div_hdr.insts.push_back(spirv::Inst(spv::OpULessThan, bool_t, c_div).id(lx).id(m.c_u32(16)));
        div_hdr.insts.push_back(spirv::Inst(spv::OpBranchConditional, 0, 0).id(c_div).id(b_then).id(b_loop_hdr));

        // Divergent diamond arm reconverges cleanly at b_after
        spirv::Block then_blk{b_then, {}};
        if (divergent_barrier) {
            then_blk.insts.push_back(
                spirv::Inst(spv::OpControlBarrier, 0, 0).id(m.c_u32(2)).id(m.c_u32(2)).id(m.c_u32(0x108)));
        }
        then_blk.insts.push_back(spirv::Inst(spv::OpBranch, 0, 0).id(b_after));

        // Loop arm with shared store, also reconverging at b_after
        spirv::Block hdr_blk{b_loop_hdr, {}};
        spirv::Id c_unif = m.new_id();
        hdr_blk.insts.push_back(spirv::Inst(spv::OpULessThan, bool_t, c_unif).id(m.c_u32(1)).id(m.c_u32(0)));
        hdr_blk.insts.push_back(spirv::Inst(spv::OpBranchConditional, 0, 0).id(c_unif).id(b_loop_body).id(b_after));

        spirv::Block lbody_blk{b_loop_body, {}};
        spirv::Id ac_s = m.new_id();
        lbody_blk.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_wg_u32, ac_s).id(sh_var).id(lx));
        lbody_blk.insts.push_back(spirv::Inst(spv::OpStore, 0, 0).id(ac_s).id(lx));
        lbody_blk.insts.push_back(spirv::Inst(spv::OpBranch, 0, 0).id(b_loop_cont));

        spirv::Block lcont_blk{b_loop_cont, {}};
        lcont_blk.insts.push_back(spirv::Inst(spv::OpBranch, 0, 0).id(b_loop_hdr));

        spirv::Block after_blk{b_after, {}};
        if (!divergent_barrier) {
            after_blk.insts.push_back(
                spirv::Inst(spv::OpControlBarrier, 0, 0).id(m.c_u32(2)).id(m.c_u32(2)).id(m.c_u32(0x108)));
        }
        spirv::Id c_never = m.new_id();
        after_blk.insts.push_back(spirv::Inst(spv::OpIEqual, bool_t, c_never).id(m.c_u32(1)).id(m.c_u32(0)));
        after_blk.insts.push_back(spirv::Inst(spv::OpBranchConditional, 0, 0).id(c_never).id(b_inf).id(b_exit2));

        spirv::Block inf_blk{b_inf, {}};
        inf_blk.insts.push_back(spirv::Inst(spv::OpBranch, 0, 0).id(b_inf));

        spirv::Block exit2_blk{b_exit2, {}};
        exit2_blk.insts.push_back(spirv::Inst(spv::OpReturn, 0, 0));

        fn.blocks = {entry, early_blk, div_hdr, then_blk, hdr_blk, lbody_blk, lcont_blk, after_blk, inf_blk, exit2_blk};
        return m;
    };

    const auto safe_rep = brass::verify::prove_spirv_shader_safety(build_cfg_shader(false));
    CHECK(safe_rep.barriers_uniform);
    CHECK(safe_rep.dominance_ok);
    CHECK(safe_rep.ok());
    CHECK(safe_rep.num_blocks >= 9u);

    const auto bad_rep = brass::verify::prove_spirv_shader_safety(build_cfg_shader(true));
    CHECK(!bad_rep.barriers_uniform);
    CHECK(!bad_rep.ok());
}

TEST_CASE("GPU Mem Safety - SPIR-V Local Array SClamp and OpFunctionCall Out-Param") {
    spirv::Module m;
    m.add_capability(spv::CapabilityShader);
    m.set_memory_model(spv::AddressingModelLogical, spv::MemoryModelGLSL450);

    spirv::Id void_t = m.t_void();
    spirv::Id u32_t = m.t_int(32, 0);
    spirv::Id s32_t = m.t_int(32, 1);
    spirv::Id f32_t = m.t_float(32);
    spirv::Id uvec3_t = m.t_vector(u32_t, 3);
    spirv::Id ptr_in_uvec3 = m.t_pointer(spv::StorageClassInput, uvec3_t);
    spirv::Id ptr_in_u32 = m.t_pointer(spv::StorageClassInput, u32_t);
    spirv::Id gid_var = m.variable(ptr_in_uvec3, spv::StorageClassInput);
    m.decorate(gid_var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInGlobalInvocationId)});

    spirv::Id sh_arr_t = m.t_array(f32_t, 4);
    spirv::Id ptr_wg_arr = m.t_pointer(spv::StorageClassWorkgroup, sh_arr_t);
    spirv::Id ptr_wg_f32 = m.t_pointer(spv::StorageClassWorkgroup, f32_t);
    spirv::Id sh_var = m.variable(ptr_wg_arr, spv::StorageClassWorkgroup);

    spirv::Id fn_arr_t = m.t_array(f32_t, 4);
    spirv::Id ptr_fn_arr = m.t_pointer(spv::StorageClassFunction, fn_arr_t);
    spirv::Id ptr_fn_f32 = m.t_pointer(spv::StorageClassFunction, f32_t);
    spirv::Id ptr_fn_u32 = m.t_pointer(spv::StorageClassFunction, u32_t);

    // Pure helper: clamp_idx(u32 raw) -> u32 in [0, 3]
    spirv::Id p_raw = m.new_id();
    spirv::Function& f_clamp = m.add_function("clamp_idx", u32_t, m.t_function(u32_t, {u32_t}));
    const spirv::Id f_clamp_id = f_clamp.id;
    f_clamp.params.push_back(spirv::Inst(spv::OpFunctionParameter, u32_t, p_raw));
    {
        spirv::Block b{m.new_id(), {}};
        spirv::Id sraw = m.new_id(), sclamped = m.new_id(), uclamped = m.new_id();
        b.insts.push_back(spirv::Inst(spv::OpBitcast, s32_t, sraw).id(p_raw));
        b.insts.push_back(spirv::Inst(spv::OpExtInst, s32_t, sclamped)
                              .id(m.glsl_std_450()).lit(GLSLstd450SClamp)
                              .id(sraw).id(m.c_sint(32, 0)).id(m.c_sint(32, 3)));
        b.insts.push_back(spirv::Inst(spv::OpBitcast, u32_t, uclamped).id(sclamped));
        b.insts.push_back(spirv::Inst(spv::OpReturnValue, 0, 0).id(uclamped));
        f_clamp.blocks.push_back(b);
    }

    // Out-param helper: write_slot(ptr Function u32 out, u32 val)
    spirv::Id p_out = m.new_id(), p_val = m.new_id();
    spirv::Function& f_write = m.add_function("write_slot", void_t, m.t_function(void_t, {ptr_fn_u32, u32_t}));
    const spirv::Id f_write_id = f_write.id;
    f_write.params.push_back(spirv::Inst(spv::OpFunctionParameter, ptr_fn_u32, p_out));
    f_write.params.push_back(spirv::Inst(spv::OpFunctionParameter, u32_t, p_val));
    {
        spirv::Block b{m.new_id(), {}};
        b.insts.push_back(spirv::Inst(spv::OpStore, 0, 0).id(p_out).id(p_val));
        b.insts.push_back(spirv::Inst(spv::OpReturn, 0, 0));
        f_write.blocks.push_back(b);
    }

    spirv::Function& main_fn = m.add_function("main", void_t, m.t_function(void_t, {}));
    m.add_entry_point(spv::ExecutionModelGLCompute, main_fn.id, "main", {gid_var, sh_var});
    m.add_execution_mode(main_fn.id, spv::ExecutionModeLocalSize, {64, 1, 1});
    {
        spirv::Block b{m.new_id(), {}};
        spirv::Id local_arr = m.new_id(), out_slot = m.new_id();
        b.insts.push_back(spirv::Inst(spv::OpVariable, ptr_fn_arr, local_arr).lit(spv::StorageClassFunction));
        b.insts.push_back(spirv::Inst(spv::OpVariable, ptr_fn_u32, out_slot).lit(spv::StorageClassFunction));

        spirv::Id ac_gx = m.new_id(), gx = m.new_id(), cidx = m.new_id(), call_void = m.new_id(), loaded_idx = m.new_id();
        b.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_in_u32, ac_gx).id(gid_var).id(m.c_u32(0)));
        b.insts.push_back(spirv::Inst(spv::OpLoad, u32_t, gx).id(ac_gx));
        b.insts.push_back(spirv::Inst(spv::OpFunctionCall, u32_t, cidx).id(f_clamp_id).id(gx));
        b.insts.push_back(spirv::Inst(spv::OpFunctionCall, void_t, call_void).id(f_write_id).id(out_slot).id(cidx));
        b.insts.push_back(spirv::Inst(spv::OpLoad, u32_t, loaded_idx).id(out_slot));

        spirv::Id ac_loc = m.new_id(), loc_v = m.new_id(), ac_sh = m.new_id();
        b.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_fn_f32, ac_loc).id(local_arr).id(loaded_idx));
        b.insts.push_back(spirv::Inst(spv::OpStore, 0, 0).id(ac_loc).id(m.c_f32(2.5f)));
        b.insts.push_back(spirv::Inst(spv::OpLoad, f32_t, loc_v).id(ac_loc));
        b.insts.push_back(spirv::Inst(spv::OpAccessChain, ptr_wg_f32, ac_sh).id(sh_var).id(loaded_idx));
        b.insts.push_back(spirv::Inst(spv::OpStore, 0, 0).id(ac_sh).id(loc_v));
        b.insts.push_back(spirv::Inst(spv::OpReturn, 0, 0));
        main_fn.blocks.push_back(b);
    }

    GpuProverOptions opts;
    opts.local_size_x = 64;
    opts.grid_size_x = 4;
    opts.check_global_bounds = false;
    GpuMemorySafetyResult r = verify_spirv_memory_safety(m, opts);
    CHECK(r.is_safe());
    CHECK_EQ(r.shared_accesses_proven, 1u);
}

TEST_CASE("GPU Mem Safety - Divisibility ParamConstraint Proves GLU-Style Indexing") {
    Module mod("gpu_div_constraint");
    Function* fn = mod.create_function(
        "glu_write", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    {
        KernelBuilder kb(mod, fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* y = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        Value* d = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* i = kb.global_tid_x();
        kb.if_then(b.build_ult(i, n), [&] {
            Value* blk = b.build_udiv(i, d);
            Value* off = b.build_umod(i, d);
            Value* two_d = b.build_mul(b.build_iconst_i32(2), d);
            Value* idx = b.build_add(b.build_add(b.build_mul(blk, two_d), off), d);
            kb.store_f32_indexed(y, idx, b.build_fconst_f32(1.0f));
        });
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
    }

    target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 64;
    spirv::Module spv_mod = lower_spirv(*fn, spv_opts);

    GpuProverOptions opts;
    opts.local_size_x = 64;
    opts.grid_size_x = 8;
    opts.buffers = {GpuBufferSpec{0, 0, 1, 4, 16, -1, -1, 2}}; // y holds 2 * n floats

    GpuParamConstraint d_pos;
    d_pos.param_index = 2;
    d_pos.min_val = 1;
    opts.param_constraints = {d_pos};
    // Without n % d == 0, last partial block pushes (n/d)*2*d + d past 2*n
    CHECK(verify_spirv_memory_safety(spv_mod, opts).is_counterexample());

    GpuParamConstraint n_div_d;
    n_div_d.param_index = 1;
    n_div_d.divisor_param = 2;
    opts.param_constraints.push_back(n_div_d);
    GpuMemorySafetyResult r_safe = verify_spirv_memory_safety(spv_mod, opts);
    CHECK(r_safe.is_safe());
    CHECK_EQ(r_safe.global_accesses_proven, 1u);

    // Also check constant divisibility (divisor_const = 4 on d -> 4 | d also keeps 2*n safe with n % d == 0)
    GpuParamConstraint n_div_4;
    n_div_4.param_index = 1;
    n_div_4.divisor_const = 4;
    opts.param_constraints.push_back(n_div_4);
    CHECK(verify_spirv_memory_safety(spv_mod, opts).is_safe());
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
