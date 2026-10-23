#include "test_framework.hpp"

#include <brass/brass.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gpu_superopt.h>
#include <brass/mir/gpu_superopt.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_equiv.h>
#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/gpu_kernel_prover.h>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <brass/verify/smt_equiv.hpp>

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

bool has_intrinsic_call(const Function& fn, std::string_view callee) {
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == callee) {
                return true;
            }
        }
    }
    return false;
}

size_t count_opcode(const Function& fn, Opcode op, TypeKind tk = TypeKind::Void) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) {
                if (tk == TypeKind::Void ||
                    (inst->result() && inst->result()->type().kind() == tk)) {
                    ++n;
                }
            }
        }
    }
    return n;
}

} // namespace

// ============================================================================
// Issue #85 Part 1: Symbolic Backend Equivalence (MIR == SPIR-V == PTX)
// ============================================================================

TEST_CASE("GPU Verify - MIR == SPIR-V == PTX Equivalence on Vector Add & FMA Kernel") {
    Module mod("gpu_vec_fma");
    Function* fn = mod.create_function(
        "vec_fma", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::ptr());
    Value* bb_ptr = b.add_block_param(entry, Type::ptr());
    Value* c = b.add_block_param(entry, Type::ptr());
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* gid = kb.global_tid_x();
    kb.if_then(b.build_slt(gid, n), [&] {
        Value* va = kb.load_f32_indexed(a, gid);
        Value* vb = kb.load_f32_indexed(bb_ptr, gid);
        Value* vc = kb.load_f32_indexed(c, gid);
        Value* res = kb.fma(va, vb, vc);
        kb.store_f32_indexed(out, gid, res);
    });
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto spv = lower_spirv(*fn);
    auto ptx_fn = lower_ptx(*fn);

    EquivResult r_spv = verify_spirv_equivalence(*fn, spv);
    CHECK_EQ(r_spv.status, EquivStatus::ProvenEquivalent);

    EquivResult r_ptx = verify_ptx_equivalence(*fn, ptx_fn);
    CHECK_EQ(r_ptx.status, EquivStatus::ProvenEquivalent);

    EquivResult r_cross = verify_spirv_ptx_equivalence(*fn, spv, ptx_fn);
    CHECK_EQ(r_cross.status, EquivStatus::ProvenEquivalent);

    // Also test the convenience overload & C API (0 == ProvenEquivalent)
    CHECK_EQ(verify_spirv_ptx_equivalence(*fn).status, EquivStatus::ProvenEquivalent);
    char err_buf[256] = {0};
    CHECK_EQ(brass_verify_spirv_equivalence(reinterpret_cast<const BrassFunction_T*>(fn), err_buf, sizeof(err_buf)), 0);
    CHECK_EQ(brass_verify_ptx_equivalence(reinterpret_cast<const BrassFunction_T*>(fn), err_buf, sizeof(err_buf)), 0);
    CHECK_EQ(brass_verify_spirv_ptx_equivalence(reinterpret_cast<const BrassFunction_T*>(fn), err_buf, sizeof(err_buf)), 0);
}

TEST_CASE("GPU Verify - MIR == SPIR-V == PTX Equivalence on Control Flow & Bounded Loop") {
    Module mod("gpu_cf_loop");
    Function* fn = mod.create_function(
        "cf_loop", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* scale = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* tid = kb.tid_x();
    Value* acc = kb.for_range_reduce(
        kb.const_i32(0), kb.const_i32(3), kb.const_i32(1), kb.const_i32(0),
        [&](Value* i, Value* cur) {
            Value* idx = b.build_add(tid, i);
            Value* v = kb.load_i32_indexed(in_ptr, idx);
            Value* prod = b.build_mul(v, scale);
            return b.build_add(cur, prod);
        });
    kb.store_i32_indexed(out_ptr, tid, acc);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto spv = lower_spirv(*fn);
    auto ptx_fn = lower_ptx(*fn);

    EquivOptions opts;
    opts.loop_unroll_bound = 6;

    CHECK_EQ(verify_spirv_equivalence(*fn, spv, opts).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*fn, ptx_fn, opts).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_spirv_ptx_equivalence(*fn, spv, ptx_fn, nullptr, opts).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("GPU Verify - MIR == SPIR-V == PTX Equivalence on Workgroup Shared Memory & Barrier") {
    Module mod("gpu_shared_barrier");
    Function* fn = mod.create_function(
        "shared_reverse", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());

    b.position_at_end(entry);
    Value* smem = kb.shared_alloc_i32(64);
    Value* tid = kb.tid_x();
    Value* val = kb.load_i32_indexed(in_ptr, tid);
    kb.shared_store_i32_indexed(smem, tid, val);
    kb.sync();
    Value* rev_idx = b.build_sub(kb.const_i32(63), tid);
    Value* loaded = kb.shared_load_i32_indexed(smem, rev_idx);
    kb.store_i32_indexed(out_ptr, tid, loaded);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto spv = lower_spirv(*fn);
    auto ptx_fn = lower_ptx(*fn);

    CHECK_EQ(verify_spirv_equivalence(*fn, spv).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*fn, ptx_fn).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_spirv_ptx_equivalence(*fn, spv, ptx_fn).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("GPU Verify - MIR == SPIR-V == PTX Equivalence on Subgroup Shuffle & Atomics") {
    Module mod("gpu_shfl_atomic");
    Function* fn = mod.create_function(
        "shfl_atomic", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());

    b.position_at_end(entry);
    Value* tid = kb.tid_x();
    Value* x = kb.load_i32_indexed(in_ptr, tid);
    Value* s_down = kb.shfl_down_i32(x, 1);
    Value* s_bfly = kb.shfl_bfly_i32(s_down, 2);
    Value* s_idx = kb.shfl_idx_i32(s_bfly, 0);
    Value* combined = b.build_add(s_bfly, s_idx);
    kb.atom_add_i32(out_ptr, combined);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto spv = lower_spirv(*fn);
    auto ptx_fn = lower_ptx(*fn);

    CHECK_EQ(verify_spirv_equivalence(*fn, spv).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*fn, ptx_fn).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_spirv_ptx_equivalence(*fn, spv, ptx_fn).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("GPU Verify - Detects Mutated SPIR-V and Mutated PTX with Concrete Counterexamples") {
    Module mod("gpu_mut_detect");
    Function* fn = mod.create_function(
        "add_store", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* x = b.add_block_param(entry, Type::i32());
    Value* y = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    Value* sum = b.build_add(x, y);
    kb.store_i32(out, sum, 0);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto spv = lower_spirv(*fn);
    auto bad_spv = lower_spirv(*fn);

    // Mutate OpIAdd -> OpISub in spirv::Module
    bool mutated_spv = false;
    for (auto& f : bad_spv.functions) {
        for (auto& blk : f.blocks) {
            for (auto& inst : blk.insts) {
                if (inst.op == spv::Op::OpIAdd) {
                    inst.op = spv::Op::OpISub;
                    mutated_spv = true;
                    break;
                }
            }
        }
    }
    REQUIRE(mutated_spv);

    EquivResult bad_spv_res = verify_spirv_equivalence(*fn, bad_spv);
    CHECK_EQ(bad_spv_res.status, EquivStatus::Counterexample);

    // Mutate PTX add -> sub
    ptx::Function bad_ptx = lower_ptx(*fn);
    bool mutated_ptx = false;
    for (auto& blk : bad_ptx.blocks) {
        for (auto& inst : blk->insts) {
            if (inst.op == ptx::Opcode::add && inst.type == ptx::Type::s32) {
                inst.op = ptx::Opcode::sub;
                mutated_ptx = true;
                break;
            }
        }
    }
    REQUIRE(mutated_ptx);

    EquivResult bad_ptx_res = verify_ptx_equivalence(*fn, bad_ptx);
    CHECK_EQ(bad_ptx_res.status, EquivStatus::Counterexample);

    EquivResult bad_cross_res = verify_spirv_ptx_equivalence(*fn, spv, bad_ptx);
    CHECK_EQ(bad_cross_res.status, EquivStatus::Counterexample);
}

// ============================================================================
// Issue #85 Part 2: GPU Superoptimization (Strength Reduction & Index Narrowing)
// ============================================================================

TEST_CASE("GPU Superopt - 32-bit Constant udiv/umod and Guarded sdiv/smod Strength Reduction") {
    MagicU32 m10 = compute_magic_u32(10);
    CHECK(verify_magic_u32_with_solver(10, m10));
    MagicU32 m7 = compute_magic_u32(7);
    CHECK(verify_magic_u32_with_solver(7, m7));
    MagicU32 bad_m7 = m7;
    bad_m7.multiplier ^= 1u;
    CHECK(!verify_magic_u32_with_solver(7, bad_m7));

    Module mod("gpu_div_superopt");
    Function* fn = mod.create_function(
        "div_mod_kernel", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* u_in = b.add_block_param(entry, Type::i32());
    Value* s_in = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    // Dominated by non-negative branch on s_in: s_in >= 0
    kb.if_then(b.build_sge(s_in, kb.const_i32(0)), [&] {
        Value* uq = b.build_udiv(u_in, kb.const_i32(10));
        Value* ur = b.build_umod(u_in, kb.const_i32(10));
        Value* sq = b.build_sdiv(s_in, kb.const_i32(7));
        Value* sr = b.build_smod(s_in, kb.const_i32(7));

        kb.store_i32(out, uq, 0);
        kb.store_i32(out, ur, 4);
        kb.store_i32(out, sq, 8);
        kb.store_i32(out, sr, 12);
    });
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto orig_mod = clone_module(mod);
    Function* orig_fn = orig_mod->get_function("div_mod_kernel");

    SuperoptStats stats;
    SuperoptOptions sopts;
    sopts.enable_gpu_superopt = true;
    sopts.stats = &stats;
    bool changed = run_gpu_mir_superopt(*fn, mod, sopts);
    CHECK(changed);
    CHECK_EQ(stats.magic_divs_reduced, 4u);
    CHECK(has_intrinsic_call(*fn, "ptx_mul_hi_u32"));
    CHECK_EQ(count_opcode(*fn, Opcode::udiv), 0u);
    CHECK_EQ(count_opcode(*fn, Opcode::umod), 0u);
    CHECK_EQ(count_opcode(*fn, Opcode::sdiv), 0u);
    CHECK_EQ(count_opcode(*fn, Opcode::smod), 0u);

    // Prove SMT equivalence of the superoptimized MIR against original MIR, SPIR-V, and PTX
    CHECK_EQ(verify_mir_equivalence(*orig_fn, *fn).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_spirv_equivalence(*orig_fn, lower_spirv(*fn)).status, EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*orig_fn, lower_ptx(*fn)).status, EquivStatus::ProvenEquivalent);
}

TEST_CASE("GPU Superopt - Refuses Signed Division Strength Reduction When Dividend Can Be Negative") {
    Module mod("gpu_sdiv_negative");
    Function* fn = mod.create_function(
        "sdiv_unproven", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    Value* q = b.build_sdiv(x, b.build_iconst_i32(7));
    b.build_ret(q);
    fn->rebuild_cfg_predecessors();

    SuperoptStats stats;
    SuperoptOptions sopts;
    sopts.enable_gpu_superopt = true;
    sopts.stats = &stats;
    run_gpu_mir_superopt(*fn, mod, sopts);
    CHECK_EQ(stats.magic_divs_reduced, 0u);
    CHECK_EQ(count_opcode(*fn, Opcode::sdiv), 1u);
}

TEST_CASE("GPU Superopt - 32-bit Index Narrowing of 64-bit Coordinate Arithmetic") {
    Module mod("gpu_idx_narrow");
    Function* fn = mod.create_function(
        "narrow_indexing", Type::void_type(), {Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* base_ptr = b.add_block_param(entry, Type::ptr());
    Value* raw_stride = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    // Bound stride in [0, 1023] so 64-bit index arithmetic cannot overflow 31 bits
    Value* stride = b.build_and(raw_stride, kb.const_i32(1023));
    Value* tid = kb.tid_x();
    Value* tid64 = b.build_zext_i64(tid);
    Value* stride64 = b.build_zext_i64(stride);
    Value* prod64 = b.build_mul(tid64, stride64);
    Value* sum64 = b.build_add(prod64, b.build_iconst_i64(16));
    Value* byte_off64 = b.build_shl(sum64, b.build_iconst_i64(2));
    Value* elem_idx = b.build_trunc_i32(byte_off64);

    kb.store_i32(base_ptr, elem_idx, 0);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto orig_mod = clone_module(mod);
    Function* orig_fn = orig_mod->get_function("narrow_indexing");

    SuperoptStats stats;
    SuperoptOptions sopts;
    sopts.enable_gpu_superopt = true;
    sopts.stats = &stats;
    bool changed = run_gpu_mir_superopt(*fn, mod, sopts);
    CHECK(changed);
    CHECK(stats.indices_narrowed >= 3u);

    CHECK_EQ(verify_spirv_equivalence(*orig_fn, lower_spirv(*fn)).status,
             EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*orig_fn, lower_ptx(*fn)).status,
             EquivStatus::ProvenEquivalent);
}

// ============================================================================
// Issue #86 Part 1: GPU Memory Safety & Alignment Prover
// ============================================================================

TEST_CASE("GPU Prover - Proves Safe Shared & Global Memory Accesses and Catches OOB / Misalignment") {
    // 1. Safe kernel: shared memory [64 x i32] indexed by tid_x < 64, global buffer guarded by gid < 256
    Module safe_mod("gpu_mem_safe");
    Function* safe_fn = safe_mod.create_function(
        "safe_kernel", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    {
        KernelBuilder kb(safe_mod, safe_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* in_ptr = b.add_block_param(entry, Type::ptr());
        Value* out_ptr = b.add_block_param(entry, Type::ptr());
        (void)b.add_block_param(entry, Type::i32());

        b.position_at_end(entry);
        Value* smem = kb.shared_alloc_i32(64);
        Value* tid = kb.tid_x();
        kb.shared_store_i32_indexed(smem, tid, tid);
        kb.sync();

        Value* gid = kb.global_tid_x();
        kb.if_then(b.build_slt(gid, kb.const_i32(256)), [&] {
            Value* v = kb.load_i32_indexed(in_ptr, gid);
            Value* sv = kb.shared_load_i32_indexed(smem, tid);
            kb.store_i32_indexed(out_ptr, gid, b.build_add(v, sv));
        });
        b.build_ret_void();
        safe_fn->rebuild_cfg_predecessors();
    }

    GpuProverOptions opts;
    opts.local_size_x = 64;
    opts.grid_size_x = 4;
    opts.buffers = {
        GpuBufferSpec{0, 1024, -1, 4, 16},
        GpuBufferSpec{1, 1024, -1, 4, 16},
    };

    CHECK(verify_gpu_memory_safety(*safe_fn, opts).is_safe());
    CHECK(verify_spirv_memory_safety(lower_spirv(*safe_fn), opts).is_safe());
    CHECK(verify_ptx_memory_safety(lower_ptx(*safe_fn), opts).is_safe());

    // 2. Shared memory OOB: accesses smem[tid_x + 1] with local_size_x = 64 on 64-element array
    Module oob_shared_mod("gpu_oob_shared");
    Function* oob_shared_fn = oob_shared_mod.create_function(
        "oob_shared", Type::void_type(), {Type::ptr()});
    {
        KernelBuilder kb(oob_shared_mod, oob_shared_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* out_ptr = b.add_block_param(entry, Type::ptr());
        b.position_at_end(entry);
        Value* smem = kb.shared_alloc_i32(64);
        Value* bad_idx = b.build_add(kb.tid_x(), kb.const_i32(1));
        Value* v = kb.shared_load_i32_indexed(smem, bad_idx);
        kb.store_i32(out_ptr, v, 0);
        b.build_ret_void();
        oob_shared_fn->rebuild_cfg_predecessors();
    }

    GpuMemorySafetyResult r_shared = verify_gpu_memory_safety(*oob_shared_fn, opts);
    CHECK(r_shared.is_counterexample());
    REQUIRE(!r_shared.violations.empty());
    CHECK(r_shared.violations[0].is_shared);
    CHECK(!r_shared.violations[0].is_alignment);
    CHECK_EQ(r_shared.violations[0].tid_x, 63u);

    CHECK(verify_spirv_memory_safety(lower_spirv(*oob_shared_fn), opts).is_counterexample());
    CHECK(verify_ptx_memory_safety(lower_ptx(*oob_shared_fn), opts).is_counterexample());

    // 3. Global memory OOB: gid ranges up to 255, buffer is 128 elements = 512 bytes
    GpuProverOptions small_buf_opts = opts;
    small_buf_opts.buffers = {
        GpuBufferSpec{0, 512, -1, 4, 16},
        GpuBufferSpec{1, 512, -1, 4, 16},
    };
    GpuMemorySafetyResult r_glob = verify_gpu_memory_safety(*safe_fn, small_buf_opts);
    CHECK(r_glob.is_counterexample());
    REQUIRE(!r_glob.violations.empty());
    CHECK(!r_glob.violations[0].is_shared);
    CHECK(r_glob.violations[0].offending_index_or_offset >= 512);

    // 4. Misaligned global access: 4-byte store at byte offset 2
    Module misalign_mod("gpu_misalign");
    Function* misalign_fn = misalign_mod.create_function(
        "misaligned_store", Type::void_type(), {Type::ptr()});
    {
        KernelBuilder kb(misalign_mod, misalign_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* out_ptr = b.add_block_param(entry, Type::ptr());
        b.position_at_end(entry);
        kb.store_i32(out_ptr, kb.const_i32(42), 2);
        b.build_ret_void();
        misalign_fn->rebuild_cfg_predecessors();
    }
    GpuMemorySafetyResult r_align = verify_gpu_memory_safety(*misalign_fn, opts);
    CHECK(r_align.is_counterexample());
    REQUIRE(!r_align.violations.empty());
    CHECK(r_align.violations[0].is_alignment);
    CHECK_EQ(r_align.violations[0].offending_index_or_offset, 2);

    // C API check (0 == ProvenSafe, 1 == Counterexample)
    char err_buf[256] = {0};
    CHECK_EQ(brass_verify_gpu_memory_safety(
                 reinterpret_cast<const BrassFunction_T*>(oob_shared_fn), 64, 4, 1024, err_buf, sizeof(err_buf)),
             1);
}

// ============================================================================
// Issue #86 Part 2: Workgroup Barrier & Subgroup Collective Uniformity Prover
// ============================================================================

TEST_CASE("GPU Prover - Workgroup Barrier & Subgroup Shuffle Uniformity Verification") {
    GpuProverOptions opts;
    opts.local_size_x = 64;
    opts.grid_size_x = 2;

    // 1. Uniform barrier and uniform warp shuffle (guarded by uniform grid/param condition)
    Module uni_mod("gpu_uniform_bar");
    Function* uni_fn = uni_mod.create_function(
        "uniform_kernel", Type::void_type(), {Type::ptr(), Type::i32()});
    {
        KernelBuilder kb(uni_mod, uni_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* flag = b.add_block_param(entry, Type::i32());

        b.position_at_end(entry);
        kb.if_then(b.build_sgt(flag, kb.const_i32(0)), [&] {
            kb.sync();
            Value* s = kb.shfl_down_i32(kb.tid_x(), 1);
            kb.store_i32_indexed(out, kb.tid_x(), s);
        });
        b.build_ret_void();
        uni_fn->rebuild_cfg_predecessors();
    }

    CHECK(verify_gpu_barrier_uniformity(*uni_fn, opts).is_uniform());
    CHECK(verify_spirv_barrier_uniformity(lower_spirv(*uni_fn), opts).is_uniform());
    CHECK(verify_ptx_barrier_uniformity(lower_ptx(*uni_fn), opts).is_uniform());

    // 2. Divergent workgroup barrier: `if (tid_x < 16) bar.sync`
    Module div_bar_mod("gpu_div_bar");
    Function* div_bar_fn = div_bar_mod.create_function(
        "divergent_barrier", Type::void_type(), {Type::ptr()});
    {
        KernelBuilder kb(div_bar_mod, div_bar_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        (void)b.add_block_param(entry, Type::ptr());

        b.position_at_end(entry);
        kb.if_then(b.build_slt(kb.tid_x(), kb.const_i32(16)), [&] {
            kb.sync();
        });
        b.build_ret_void();
        div_bar_fn->rebuild_cfg_predecessors();
    }

    BarrierUniformityResult r_bar = verify_gpu_barrier_uniformity(*div_bar_fn, opts);
    CHECK(r_bar.is_counterexample());
    REQUIRE(!r_bar.divergences.empty());
    CHECK(!r_bar.divergences[0].is_subgroup_collective);
    CHECK_NE(r_bar.divergences[0].local_id_a[0] < 16u, r_bar.divergences[0].local_id_b[0] < 16u);

    CHECK(verify_spirv_barrier_uniformity(lower_spirv(*div_bar_fn), opts).is_counterexample());
    CHECK(verify_ptx_barrier_uniformity(lower_ptx(*div_bar_fn), opts).is_counterexample());

    // 3. Divergent subgroup shuffle: `if (tid_x == 0) shfl_down_i32(...)`
    Module div_shfl_mod("gpu_div_shfl");
    Function* div_shfl_fn = div_shfl_mod.create_function(
        "divergent_shfl", Type::void_type(), {Type::ptr()});
    {
        KernelBuilder kb(div_shfl_mod, div_shfl_fn);
        Builder& b = kb.builder();
        BasicBlock* entry = b.append_block("entry");
        Value* out = b.add_block_param(entry, Type::ptr());

        b.position_at_end(entry);
        kb.if_then(b.build_eq(kb.tid_x(), kb.const_i32(0)), [&] {
            Value* s = kb.shfl_down_i32(kb.tid_x(), 1);
            kb.store_i32(out, s, 0);
        });
        b.build_ret_void();
        div_shfl_fn->rebuild_cfg_predecessors();
    }

    BarrierUniformityResult r_shfl = verify_gpu_barrier_uniformity(*div_shfl_fn, opts);
    CHECK(r_shfl.is_counterexample());
    REQUIRE(!r_shfl.divergences.empty());
    CHECK(r_shfl.divergences[0].is_subgroup_collective);
    CHECK_EQ(r_shfl.divergences[0].local_id_a[0] / 32u, r_shfl.divergences[0].local_id_b[0] / 32u);

    CHECK(verify_spirv_barrier_uniformity(lower_spirv(*div_shfl_fn), opts).is_counterexample());
    CHECK(verify_ptx_barrier_uniformity(lower_ptx(*div_shfl_fn), opts).is_counterexample());

    char err_buf[256] = {0};
    CHECK_EQ(brass_verify_gpu_barrier_uniformity(
                 reinterpret_cast<const BrassFunction_T*>(uni_fn), 64, 32, err_buf, sizeof(err_buf)),
             0);
    CHECK_EQ(brass_verify_gpu_barrier_uniformity(
                 reinterpret_cast<const BrassFunction_T*>(div_bar_fn), 64, 32, err_buf, sizeof(err_buf)),
             1);
}

// ============================================================================
// Issue #86 Part 3: Tile-Interior Specialization & Bounds-Check Elimination
// ============================================================================

TEST_CASE("GPU Prover - Tile-Interior Bounds Prover and Redundant Check Elimination") {
    // 1. Interior tile [0..960] + [0..64) < 1024 is proven safe
    TileBoundsProofResult r_ok = prove_tile_interior_bounds(0, 960, 64, 1024);
    CHECK(r_ok.is_proven());

    // 2. Boundary tile where tile_base_max (992) + tile_extent (64) > 1024 produces counterexample
    TileBoundsProofResult r_bad = prove_tile_interior_bounds(0, 992, 64, 1024);
    CHECK_EQ(r_bad.status, EquivStatus::Counterexample);
    CHECK(r_bad.counterexample_tile_base + r_bad.counterexample_local_offset >= 1024u);

    // 3. Eliminate redundant per-element bounds comparisons inside an interior tile block
    Module mod("gpu_tile_bce");
    Function* fn = mod.create_function(
        "tiled_kernel", Type::void_type(), {Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());

    b.position_at_end(entry);
    Value* tid = b.build_and(kb.tid_x(), kb.const_i32(63));
    Value* tile_base = b.build_mul(kb.ctaid_x(), kb.const_i32(64));
    Value* idx = b.build_add(tile_base, tid);
    Value* dim_n = kb.const_i32(1024);

    // Interior tile branch: tile_base + 64 <= 1024
    Value* tile_end = b.build_add(tile_base, kb.const_i32(64));
    kb.if_then(b.build_ule(tile_end, dim_n), [&] {
        // Inside the interior tile, `idx < 1024` and `idx <= 1024` are provably always true!
        Value* in_bounds_hi = b.build_ult(idx, dim_n);
        Value* in_bounds_le = b.build_ule(idx, dim_n);
        Value* both_ok = b.build_and(in_bounds_hi, in_bounds_le);
        Value* v = kb.load_i32_indexed(in_ptr, idx);
        Value* masked = b.build_mul(v, both_ok);
        kb.store_i32_indexed(out_ptr, idx, masked);
    });
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    auto orig_mod = clone_module(mod);
    Function* orig_fn = orig_mod->get_function("tiled_kernel");

    uint32_t elided = eliminate_interior_tile_bounds(*fn, mod, 64, 1024);
    CHECK(elided >= 2u);

    target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 64;
    CHECK_EQ(verify_spirv_equivalence(*orig_fn, lower_spirv(*fn, spv_opts)).status,
             EquivStatus::ProvenEquivalent);
    CHECK_EQ(verify_ptx_equivalence(*orig_fn, lower_ptx(*fn)).status,
             EquivStatus::ProvenEquivalent);

    // Verify C API wrappers as well (0 == ProvenEquivalent, 1 == Counterexample)
    char err_buf[256] = {0};
    CHECK_EQ(brass_prove_tile_interior_bounds(0, 960, 64, 1024, err_buf, sizeof(err_buf)), 0);
    CHECK_EQ(brass_prove_tile_interior_bounds(0, 992, 64, 1024, err_buf, sizeof(err_buf)), 1);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
