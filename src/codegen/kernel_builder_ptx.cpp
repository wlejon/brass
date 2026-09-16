// KernelBuilder GPU helpers: thin MIR wrappers over the `ptx_*` builtins that
// PtxISel lowers (src/target/ptx/ptx_isel_intrinsics*.cpp). Kernels written
// with these read like CUDA; none of them knows PTX syntax. The builtin
// signatures are documented in docs/ptx_backend_design.md ("Stage 4
// implementation notes").

#include <brass/codegen/kernel_jit.hpp>

namespace brass::codegen {

namespace {

constexpr float kLog2E = 1.44269504f;
constexpr float kLn2 = 0.69314718f;

} // namespace

// ---------------------------------------------------------------------------
// Indices
// ---------------------------------------------------------------------------

Value* KernelBuilder::tid_x()    { return b_.build_call("ptx_tid_x", Type::i32()); }
Value* KernelBuilder::tid_y()    { return b_.build_call("ptx_tid_y", Type::i32()); }
Value* KernelBuilder::tid_z()    { return b_.build_call("ptx_tid_z", Type::i32()); }
Value* KernelBuilder::ctaid_x()  { return b_.build_call("ptx_ctaid_x", Type::i32()); }
Value* KernelBuilder::ctaid_y()  { return b_.build_call("ptx_ctaid_y", Type::i32()); }
Value* KernelBuilder::ctaid_z()  { return b_.build_call("ptx_ctaid_z", Type::i32()); }
Value* KernelBuilder::ntid_x()   { return b_.build_call("ptx_ntid_x", Type::i32()); }
Value* KernelBuilder::ntid_y()   { return b_.build_call("ptx_ntid_y", Type::i32()); }
Value* KernelBuilder::ntid_z()   { return b_.build_call("ptx_ntid_z", Type::i32()); }
Value* KernelBuilder::nctaid_x() { return b_.build_call("ptx_nctaid_x", Type::i32()); }
Value* KernelBuilder::nctaid_y() { return b_.build_call("ptx_nctaid_y", Type::i32()); }
Value* KernelBuilder::nctaid_z() { return b_.build_call("ptx_nctaid_z", Type::i32()); }
Value* KernelBuilder::lane_id()  { return b_.build_call("ptx_laneid", Type::i32()); }
Value* KernelBuilder::warp_id()  { return b_.build_lshr(tid_x(), const_i32(5)); }
Value* KernelBuilder::global_tid_x() { return b_.build_call("ptx_global_tid_x", Type::i32()); }

// ---------------------------------------------------------------------------
// Barriers and shuffles
// ---------------------------------------------------------------------------

Instruction* KernelBuilder::sync() {
    b_.build_call("ptx_sync", Type::void_type());
    return b_.current_block()->tail();
}

Instruction* KernelBuilder::bar_sync(uint32_t id) {
    b_.build_call("ptx_bar_sync", Type::void_type(), {const_i32(static_cast<int32_t>(id))});
    return b_.current_block()->tail();
}

Value* KernelBuilder::shfl_down_f32(Value* v, uint32_t delta) { return shfl_down_f32(v, const_i32(static_cast<int32_t>(delta))); }
Value* KernelBuilder::shfl_down_f32(Value* v, Value* delta)   { return b_.build_call("ptx_shfl_down_f32", Type::f32(), {v, delta}); }
Value* KernelBuilder::shfl_up_f32(Value* v, uint32_t delta)   { return b_.build_call("ptx_shfl_up_f32", Type::f32(), {v, const_i32(static_cast<int32_t>(delta))}); }
Value* KernelBuilder::shfl_bfly_f32(Value* v, uint32_t mask)  { return b_.build_call("ptx_shfl_bfly_f32", Type::f32(), {v, const_i32(static_cast<int32_t>(mask))}); }
Value* KernelBuilder::shfl_idx_f32(Value* v, uint32_t lane)   { return shfl_idx_f32(v, const_i32(static_cast<int32_t>(lane))); }
Value* KernelBuilder::shfl_idx_f32(Value* v, Value* lane)     { return b_.build_call("ptx_shfl_idx_f32", Type::f32(), {v, lane}); }
Value* KernelBuilder::shfl_down_i32(Value* v, uint32_t delta) { return b_.build_call("ptx_shfl_down_i32", Type::i32(), {v, const_i32(static_cast<int32_t>(delta))}); }
Value* KernelBuilder::shfl_bfly_i32(Value* v, uint32_t mask)  { return b_.build_call("ptx_shfl_bfly_i32", Type::i32(), {v, const_i32(static_cast<int32_t>(mask))}); }
Value* KernelBuilder::shfl_idx_i32(Value* v, uint32_t lane)   { return b_.build_call("ptx_shfl_idx_i32", Type::i32(), {v, const_i32(static_cast<int32_t>(lane))}); }

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

// The 5-step shfl.down butterfly: after it, lane 0 holds the sum of all 32.
Value* KernelBuilder::warp_reduce_sum_f32(Value* v) {
    for (uint32_t delta = 16; delta >= 1; delta >>= 1) {
        v = b_.build_add(v, shfl_down_f32(v, delta));
    }
    return v;
}

Value* KernelBuilder::warp_reduce_max_f32(Value* v) {
    for (uint32_t delta = 16; delta >= 1; delta >>= 1) {
        v = fmax(v, shfl_down_f32(v, delta));
    }
    return v;
}

// warp reduce -> lane 0 of each warp writes scratch[warp] -> bar.sync ->
// every warp reduces scratch[0..nwarps) redundantly and broadcasts lane 0
// with shfl.idx, so all threads leave with the block total -> bar.sync so
// the scratch can be reused immediately.
Value* KernelBuilder::block_reduce_sum_f32(Value* v, Value* shared_scratch) {
    Value* lane = lane_id();
    Value* warp = warp_id();
    Value* partial = warp_reduce_sum_f32(v);

    Value* is_lane0 = b_.build_eq(lane, const_i32(0));
    if_then(is_lane0, [&] { shared_store_f32_indexed(shared_scratch, warp, partial); });
    sync();

    Value* nwarps = b_.build_lshr(b_.build_add(ntid_x(), const_i32(31)), const_i32(5));
    Value* in_range = b_.build_ult(lane, nwarps);
    Value* loaded = shared_load_f32_indexed(shared_scratch, lane); // in bounds: scratch has >= 32 entries
    Value* mine = b_.build_select(in_range, loaded, const_f32(0.0f));
    Value* total = shfl_idx_f32(warp_reduce_sum_f32(mine), 0u);
    sync();
    return total;
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

Value* KernelBuilder::shared_alloc_f32(uint32_t count) {
    return b_.build_call("ptx_shared_alloc_f32", Type::ptr(), {const_i32(static_cast<int32_t>(count))});
}
Value* KernelBuilder::shared_alloc_i32(uint32_t count) {
    return b_.build_call("ptx_shared_alloc_i32", Type::ptr(), {const_i32(static_cast<int32_t>(count))});
}
Value* KernelBuilder::shared_load_f32(Value* smem, int32_t byte_offset) {
    return b_.build_call("ptx_shared_load_f32", Type::f32(), {smem, const_i32(byte_offset)});
}
Value* KernelBuilder::shared_load_f32_indexed(Value* smem, Value* index) {
    return b_.build_call("ptx_shared_load_f32_indexed", Type::f32(), {smem, index});
}
Instruction* KernelBuilder::shared_store_f32(Value* smem, Value* val, int32_t byte_offset) {
    b_.build_call("ptx_shared_store_f32", Type::void_type(), {smem, val, const_i32(byte_offset)});
    return b_.current_block()->tail();
}
Instruction* KernelBuilder::shared_store_f32_indexed(Value* smem, Value* index, Value* val) {
    b_.build_call("ptx_shared_store_f32_indexed", Type::void_type(), {smem, index, val});
    return b_.current_block()->tail();
}
Value* KernelBuilder::shared_load_i32(Value* smem, int32_t byte_offset) {
    return b_.build_call("ptx_shared_load_i32", Type::i32(), {smem, const_i32(byte_offset)});
}
Value* KernelBuilder::shared_load_i32_indexed(Value* smem, Value* index) {
    return b_.build_call("ptx_shared_load_i32_indexed", Type::i32(), {smem, index});
}
Instruction* KernelBuilder::shared_store_i32(Value* smem, Value* val, int32_t byte_offset) {
    b_.build_call("ptx_shared_store_i32", Type::void_type(), {smem, val, const_i32(byte_offset)});
    return b_.current_block()->tail();
}
Instruction* KernelBuilder::shared_store_i32_indexed(Value* smem, Value* index, Value* val) {
    b_.build_call("ptx_shared_store_i32_indexed", Type::void_type(), {smem, index, val});
    return b_.current_block()->tail();
}

// ---------------------------------------------------------------------------
// Fast math and conversions
// ---------------------------------------------------------------------------

Value* KernelBuilder::rcp_approx(Value* x)          { return b_.build_call("ptx_rcp_approx", Type::f32(), {x}); }
Value* KernelBuilder::div_approx(Value* a, Value* b) { return b_.build_call("ptx_div_approx", Type::f32(), {a, b}); }
Value* KernelBuilder::ex2_approx(Value* x)          { return b_.build_call("ptx_ex2", Type::f32(), {x}); }
Value* KernelBuilder::lg2_approx(Value* x)          { return b_.build_call("ptx_lg2", Type::f32(), {x}); }
Value* KernelBuilder::exp_fast(Value* x)            { return ex2_approx(b_.build_mul(x, const_f32(kLog2E))); }
Value* KernelBuilder::log_fast(Value* x)            { return b_.build_mul(lg2_approx(x), const_f32(kLn2)); }
Value* KernelBuilder::rsqrt_approx(Value* x)        { return b_.build_call("ptx_rsqrt", Type::f32(), {x}); }
Value* KernelBuilder::sqrt_approx(Value* x)         { return b_.build_call("ptx_sqrt", Type::f32(), {x}); }
Value* KernelBuilder::fabs(Value* x)                { return b_.build_call("ptx_fabs", x->type(), {x}); }
Value* KernelBuilder::fmin(Value* a, Value* b)      { return b_.build_call("ptx_fmin", a->type(), {a, b}); }
Value* KernelBuilder::fmax(Value* a, Value* b)      { return b_.build_call("ptx_fmax", a->type(), {a, b}); }

Value* KernelBuilder::f16_to_f32(Value* bits) { return b_.build_call("ptx_f16_to_f32", Type::f32(), {bits}); }
Value* KernelBuilder::f32_to_f16(Value* x)    { return b_.build_call("ptx_f32_to_f16", Type::i32(), {x}); }
Value* KernelBuilder::u32_to_f32(Value* x)    { return b_.build_call("ptx_u32_to_f32", Type::f32(), {x}); }
Value* KernelBuilder::i32_to_f32(Value* x)    { return b_.build_call("ptx_i32_to_f32", Type::f32(), {x}); }
Value* KernelBuilder::f32_to_u32(Value* x)    { return b_.build_call("ptx_f32_to_u32", Type::i32(), {x}); }
Value* KernelBuilder::f32_to_i32(Value* x)    { return b_.build_call("ptx_f32_to_i32", Type::i32(), {x}); }

// ---------------------------------------------------------------------------
// Narrow loads/stores and atomics
// ---------------------------------------------------------------------------

Value* KernelBuilder::load_u8(Value* ptr, int32_t off)  { return b_.build_call("ptx_load_u8", Type::i32(), {ptr, const_i32(off)}); }
Value* KernelBuilder::load_s8(Value* ptr, int32_t off)  { return b_.build_call("ptx_load_s8", Type::i32(), {ptr, const_i32(off)}); }
Value* KernelBuilder::load_u16(Value* ptr, int32_t off) { return b_.build_call("ptx_load_u16", Type::i32(), {ptr, const_i32(off)}); }
Value* KernelBuilder::load_s16(Value* ptr, int32_t off) { return b_.build_call("ptx_load_s16", Type::i32(), {ptr, const_i32(off)}); }
Instruction* KernelBuilder::store_u8(Value* ptr, Value* val, int32_t off) {
    b_.build_call("ptx_store_u8", Type::void_type(), {ptr, val, const_i32(off)});
    return b_.current_block()->tail();
}
Instruction* KernelBuilder::store_u16(Value* ptr, Value* val, int32_t off) {
    b_.build_call("ptx_store_u16", Type::void_type(), {ptr, val, const_i32(off)});
    return b_.current_block()->tail();
}

Value* KernelBuilder::atom_add_f32(Value* ptr, Value* val) { return b_.build_call("ptx_atom_add_f32", Type::f32(), {ptr, val}); }
Value* KernelBuilder::atom_add_i32(Value* ptr, Value* val) { return b_.build_call("ptx_atom_add_i32", Type::i32(), {ptr, val}); }

// ---------------------------------------------------------------------------
// Structured control flow
// ---------------------------------------------------------------------------

void KernelBuilder::if_then(Value* cond, const std::function<void()>& body) {
    BasicBlock* from = b_.current_block();
    BasicBlock* then_bb = b_.append_block("if_then");
    BasicBlock* join = b_.append_block("if_join");

    b_.position_at_end(from);
    b_.build_br_if(cond, then_bb, join);

    b_.position_at_end(then_bb);
    body();
    if (!b_.current_block()->terminator()) b_.build_br(join);

    b_.position_at_end(join);
}

// head(i): if i < end -> body else exit;  body: ...; br head(i + step)
void KernelBuilder::for_range(Value* start, Value* end, Value* step, const std::function<void(Value*)>& body) {
    BasicBlock* from = b_.current_block();
    BasicBlock* head = b_.append_block("for_head");
    BasicBlock* body_bb = b_.append_block("for_body");
    BasicBlock* exit = b_.append_block("for_exit");
    Value* i = b_.add_block_param(head, start->type());

    b_.position_at_end(from);
    b_.build_br(head, {start});

    b_.position_at_end(head);
    b_.build_br_if(b_.build_slt(i, end), body_bb, exit);

    b_.position_at_end(body_bb);
    body(i);
    if (!b_.current_block()->terminator()) b_.build_br(head, {b_.build_add(i, step)});

    b_.position_at_end(exit);
}

} // namespace brass::codegen
