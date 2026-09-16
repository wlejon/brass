#pragma once

// Internal helpers shared by the MIR GPU kernel builders
// (ml_fusion_ptx_kernels*.cpp). Each is a few lines of MIR around
// KernelBuilder; none knows PTX syntax. Not installed, not part of the public
// brass API -- include only from src/codegen.

#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>

#include <vector>

namespace brass::codegen::ptx_kernels {

// Byte offset of f32 element `i` (i32): (u64)i << 2.
inline Value* f32_offset(KernelBuilder& kb, Value* i) {
    Builder& b = kb.builder();
    return b.build_shl(b.build_zext_i64(i), kb.const_i32(2));
}

inline Value* at(KernelBuilder& kb, Value* base, Value* byte_off) {
    return kb.builder().build_add(base, byte_off);
}

// silu(g) = g * rcp.approx(1 + ex2.approx(-g * log2 e)), unclamped (the
// element-wise SwiGLU; the GEMV SwiGLU clamps the exponent, see silu_fast_clamped)
inline Value* silu_fast(KernelBuilder& kb, Value* g) {
    Value* e = kb.exp_fast(kb.builder().build_neg(g));
    Value* r = kb.rcp_approx(kb.add(e, kb.const_f32(1.0f)));
    return kb.mul(g, r);
}

// Row-per-block prologue: returns after emitting `if (row >= rows) return;`.
// vec_d is d & ~3, or 0 when d % 4 != 0 (rows are then not 16-byte aligned
// and the whole row takes the scalar path). vstart/vstep are the float4 loop
// bounds for this thread; the scalar remainder runs from vec_d + tid by ntid.
struct RowBlock {
    Value* row;       // i32 block row index (ctaid.x)
    Value* row_off;   // i64 byte offset of this block's row
    Value* tid;
    Value* ntid;
    Value* vec_d;
    Value* vstart;    // tid * 4
    Value* vstep;     // ntid * 4
    Value* sstart;    // vec_d + tid
};

inline RowBlock row_block_prologue(KernelBuilder& kb, Value* rows, Value* d) {
    Builder& b = kb.builder();
    Value* row = kb.ctaid_x();
    kb.if_then(b.build_uge(row, rows), [&] { b.build_ret_void(); });

    RowBlock rb;
    rb.row = row;
    rb.row_off = b.build_shl(b.build_mul(b.build_zext_i64(row), b.build_zext_i64(d)), kb.const_i32(2));
    rb.tid = kb.tid_x();
    rb.ntid = kb.ntid_x();
    Value* aligned = b.build_and(d, kb.const_i32(~3));
    Value* misaligned = b.build_ne(b.build_and(d, kb.const_i32(3)), kb.const_i32(0));
    rb.vec_d = b.build_select(misaligned, kb.const_i32(0), aligned);
    rb.vstart = b.build_shl(rb.tid, kb.const_i32(2));
    rb.vstep = b.build_shl(rb.ntid, kb.const_i32(2));
    rb.sstart = kb.add(rb.vec_d, rb.tid);
    return rb;
}

// Creates the entry block with one block parameter per kernel parameter.
inline std::vector<Value*> entry_params(KernelBuilder& kb, Function* fn) {
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    std::vector<Value*> out;
    for (Type t : fn->param_types()) out.push_back(b.add_block_param(entry, t));
    return out;
}

} // namespace brass::codegen::ptx_kernels
