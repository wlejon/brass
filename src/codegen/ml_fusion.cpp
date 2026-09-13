#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>

namespace brass::codegen {

MlFusionCompiler::MlFusionCompiler()
    : jit_(KernelOptions(), Target::host()) {}

MlFusionCompiler::MlFusionCompiler(KernelJit jit)
    : jit_(std::move(jit)) {}

// =========================================================================
// 1. FusedResidualRmsNorm
// =========================================================================

Function* MlFusionCompiler::build_residual_rms_norm(Module& mod, std::string_view name) {
    mod.add_external_symbol("rsqrtf");
    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32(), Type::f32()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* residual = kb.builder().add_block_param(entry, Type::ptr());
    Value* weight = kb.builder().add_block_param(entry, Type::ptr());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());
    Value* eps = kb.builder().add_block_param(entry, Type::f32());
    Value* inv_n = kb.builder().add_block_param(entry, Type::f32());

    BasicBlock* loop1_head = kb.builder().create_block("rms_l1_head");
    BasicBlock* loop1_body = kb.builder().create_block("rms_l1_body");
    BasicBlock* loop1_exit = kb.builder().create_block("rms_l1_exit");
    BasicBlock* loop2_head = kb.builder().create_block("rms_l2_head");
    BasicBlock* loop2_body = kb.builder().create_block("rms_l2_body");
    BasicBlock* loop2_exit = kb.builder().create_block("rms_l2_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop1_head, {zero_i64, zero_f32});

    // loop1_head: params [i1, sum_sq]
    fn->append_block(loop1_head);
    Value* i1 = kb.builder().add_block_param(loop1_head, Type::i64());
    Value* sum_sq = kb.builder().add_block_param(loop1_head, Type::f32());
    kb.position_at_end(loop1_head);
    Value* cond1 = kb.builder().build_slt(i1, n);
    kb.builder().build_br_if(cond1, loop1_body, {}, loop1_exit, {sum_sq});

    // loop1_body: residual[i] = x[i] + residual[i]; sum_sq += residual[i]^2
    fn->append_block(loop1_body);
    kb.position_at_end(loop1_body);
    Value* xi = kb.load_f32_indexed(x, i1, 4, 0);
    Value* ri = kb.load_f32_indexed(residual, i1, 4, 0);
    Value* yi = kb.add(xi, ri);
    kb.store_f32_indexed(residual, i1, yi, 4, 0);
    Value* yi_sq = kb.mul(yi, yi);
    Value* next_sum = kb.add(sum_sq, yi_sq);
    Value* next_i1 = kb.add(i1, one_i64);
    kb.builder().build_br(loop1_head, {next_i1, next_sum});

    // loop1_exit: param [final_sum_sq]
    fn->append_block(loop1_exit);
    Value* final_sum = kb.builder().add_block_param(loop1_exit, Type::f32());
    kb.position_at_end(loop1_exit);
    Value* mean_sq = kb.mul(final_sum, inv_n);
    Value* denom = kb.add(mean_sq, eps);
    Value* inv_rms = kb.builder().build_call("rsqrtf", Type::f32(), {denom});
    kb.builder().build_br(loop2_head, {zero_i64, inv_rms});

    // loop2_head: params [i2, inv_rms_val]
    fn->append_block(loop2_head);
    Value* i2 = kb.builder().add_block_param(loop2_head, Type::i64());
    Value* inv_rms_val = kb.builder().add_block_param(loop2_head, Type::f32());
    kb.position_at_end(loop2_head);
    Value* cond2 = kb.builder().build_slt(i2, n);
    kb.builder().build_br_if(cond2, loop2_body, loop2_exit);

    // loop2_body: out[i] = residual[i] * inv_rms * weight[i]
    fn->append_block(loop2_body);
    kb.position_at_end(loop2_body);
    Value* yj = kb.load_f32_indexed(residual, i2, 4, 0);
    Value* wj = kb.load_f32_indexed(weight, i2, 4, 0);
    Value* norm_val = kb.mul(yj, inv_rms_val);
    Value* out_val = kb.mul(norm_val, wj);
    kb.store_f32_indexed(out, i2, out_val, 4, 0);
    Value* next_i2 = kb.add(i2, one_i64);
    kb.builder().build_br(loop2_head, {next_i2, inv_rms_val});

    // loop2_exit
    fn->append_block(loop2_exit);
    kb.position_at_end(loop2_exit);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_residual_rms_norm() {
    Module mod("mod_residual_rms_norm");
    build_residual_rms_norm(mod);
    return jit_.compile(mod, "fused_residual_rms_norm");
}

// =========================================================================
// 2. FusedSwiGLU
// =========================================================================

Function* MlFusionCompiler::build_swiglu(Module& mod, std::string_view name) {
    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* gate = kb.builder().add_block_param(entry, Type::ptr());
    Value* up = kb.builder().add_block_param(entry, Type::ptr());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("swiglu_head");
    BasicBlock* loop_body = kb.builder().create_block("swiglu_body");
    BasicBlock* loop_exit = kb.builder().create_block("swiglu_exit");

    kb.position_at_end(entry);
    Value* zero = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero});

    // loop_head
    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, loop_exit);

    // loop_body: silu(g) = g / (1.0f + exp(-g)); out[i] = silu(g) * up[i]
    // Inlined 5th-degree polynomial approximation for exp(-g) with scaling & squaring
    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* gi = kb.load_f32_indexed(gate, i, 4, 0);
    Value* ui = kb.load_f32_indexed(up, i, 4, 0);

    Value* neg_gi = kb.builder().build_neg(gi);
    Value* c_inv16 = kb.builder().build_fconst_f32(0.0625f);
    Value* u = kb.mul(neg_gi, c_inv16);

    Value* c_1_120 = kb.builder().build_fconst_f32(1.0f / 120.0f);
    Value* c_1_24  = kb.builder().build_fconst_f32(1.0f / 24.0f);
    Value* c_1_6   = kb.builder().build_fconst_f32(1.0f / 6.0f);
    Value* c_1_2   = kb.builder().build_fconst_f32(0.5f);
    Value* c_1_0   = kb.builder().build_fconst_f32(1.0f);

    Value* p5 = kb.builder().build_fma_f32(u, c_1_120, c_1_24);
    Value* p4 = kb.builder().build_fma_f32(u, p5, c_1_6);
    Value* p3 = kb.builder().build_fma_f32(u, p4, c_1_2);
    Value* p2 = kb.builder().build_fma_f32(u, p3, c_1_0);
    Value* p1 = kb.builder().build_fma_f32(u, p2, c_1_0);

    Value* sq1 = kb.mul(p1, p1);
    Value* sq2 = kb.mul(sq1, sq1);
    Value* sq3 = kb.mul(sq2, sq2);
    Value* exp_neg_g = kb.mul(sq3, sq3);

    Value* denom = kb.add(c_1_0, exp_neg_g);
    Value* silu_g = kb.div(gi, denom);
    Value* res = kb.mul(silu_g, ui);
    kb.store_f32_indexed(out, i, res, 4, 0);

    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i});

    // loop_exit
    fn->append_block(loop_exit);
    kb.position_at_end(loop_exit);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_swiglu() {
    Module mod("mod_swiglu");
    build_swiglu(mod);
    return jit_.compile(mod, "fused_swiglu");
}

// =========================================================================
// 3. FusedAdaLNModulate
// =========================================================================

Function* MlFusionCompiler::build_adaln_modulate(Module& mod, bool gated, std::string_view name) {
    std::vector<Type> params;
    if (gated) {
        params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()};
    } else {
        params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()};
    }
    Function* fn = mod.create_function(name, Type::void_type(), params);
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* scale = kb.builder().add_block_param(entry, Type::ptr());
    Value* shift = kb.builder().add_block_param(entry, Type::ptr());
    Value* gate = gated ? kb.builder().add_block_param(entry, Type::ptr()) : nullptr;
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("adaln_head");
    BasicBlock* loop_body = kb.builder().create_block("adaln_body");
    BasicBlock* loop_exit = kb.builder().create_block("adaln_exit");

    kb.position_at_end(entry);
    Value* zero = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero});

    // loop_head
    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, loop_exit);

    // loop_body: modulated = x * (1.0 + scale) + shift; [optional: * gate]
    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* scale_i = kb.load_f32_indexed(scale, i, 4, 0);
    Value* shift_i = kb.load_f32_indexed(shift, i, 4, 0);

    Value* one_f32 = kb.builder().build_fconst_f32(1.0f);
    Value* one_plus_scale = kb.add(one_f32, scale_i);
    Value* modulated = kb.builder().build_fma_f32(xi, one_plus_scale, shift_i);

    if (gated) {
        Value* gate_i = kb.load_f32_indexed(gate, i, 4, 0);
        modulated = kb.mul(modulated, gate_i);
    }

    kb.store_f32_indexed(out, i, modulated, 4, 0);

    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i});

    // loop_exit
    fn->append_block(loop_exit);
    kb.position_at_end(loop_exit);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_adaln_modulate(bool gated) {
    Module mod(gated ? "mod_adaln_gated" : "mod_adaln");
    std::string name = gated ? "fused_adaln_modulate_gated" : "fused_adaln_modulate";
    build_adaln_modulate(mod, gated, name);
    return jit_.compile(mod, name);
}

// =========================================================================
// 4. QuantizedQ8DotProduct (Flat & Block Q8_0)
// =========================================================================

Function* MlFusionCompiler::build_q8_dot(Module& mod, std::string_view name) {
    mod.add_external_symbol("i32_to_f32");
    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::f32(), Type::f32(), Type::ptr(), Type::i64()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* a = kb.builder().add_block_param(entry, Type::ptr());
    Value* b = kb.builder().add_block_param(entry, Type::ptr());
    Value* scale_a = kb.builder().add_block_param(entry, Type::f32());
    Value* scale_b = kb.builder().add_block_param(entry, Type::f32());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("q8_head");
    BasicBlock* loop_body = kb.builder().create_block("q8_body");
    BasicBlock* loop_exit = kb.builder().create_block("q8_exit");

    kb.position_at_end(entry);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* num_words = kb.builder().build_lshr(n, two_i64); // n / 4
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_i32 = kb.builder().build_iconst_i32(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64, zero_i32});

    // loop_head: params [w, acc]
    fn->append_block(loop_head);
    Value* w = kb.builder().add_block_param(loop_head, Type::i64());
    Value* acc = kb.builder().add_block_param(loop_head, Type::i32());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(w, num_words);
    kb.builder().build_br_if(cond, loop_body, {}, loop_exit, {acc});

    // loop_body: unpack 4 signed int8 values from each 32-bit word and accumulate
    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* wa = kb.load_i32_indexed(a, w, 4, 0);
    Value* wb = kb.load_i32_indexed(b, w, 4, 0);

    Value* c24 = kb.builder().build_iconst_i32(24);
    Value* c16 = kb.builder().build_iconst_i32(16);
    Value* c8  = kb.builder().build_iconst_i32(8);

    // Byte 0: (w << 24) >> 24
    Value* a0 = kb.builder().build_ashr(kb.builder().build_shl(wa, c24), c24);
    Value* b0 = kb.builder().build_ashr(kb.builder().build_shl(wb, c24), c24);
    Value* p0 = kb.mul(a0, b0);

    // Byte 1: (w << 16) >> 24
    Value* a1 = kb.builder().build_ashr(kb.builder().build_shl(wa, c16), c24);
    Value* b1 = kb.builder().build_ashr(kb.builder().build_shl(wb, c16), c24);
    Value* p1 = kb.mul(a1, b1);

    // Byte 2: (w << 8) >> 24
    Value* a2 = kb.builder().build_ashr(kb.builder().build_shl(wa, c8), c24);
    Value* b2 = kb.builder().build_ashr(kb.builder().build_shl(wb, c8), c24);
    Value* p2 = kb.mul(a2, b2);

    // Byte 3: w >> 24
    Value* a3 = kb.builder().build_ashr(wa, c24);
    Value* b3 = kb.builder().build_ashr(wb, c24);
    Value* p3 = kb.mul(a3, b3);

    Value* s01 = kb.add(p0, p1);
    Value* s23 = kb.add(p2, p3);
    Value* s4 = kb.add(s01, s23);
    Value* next_acc = kb.add(acc, s4);

    Value* next_w = kb.add(w, one_i64);
    kb.builder().build_br(loop_head, {next_w, next_acc});

    // loop_exit: final dot product = acc * scale_a * scale_b
    fn->append_block(loop_exit);
    Value* final_acc = kb.builder().add_block_param(loop_exit, Type::i32());
    kb.position_at_end(loop_exit);
    Value* acc_f = kb.builder().build_call("i32_to_f32", Type::f32(), {final_acc});
    Value* combined_scale = kb.mul(scale_a, scale_b);
    Value* dot_val = kb.mul(acc_f, combined_scale);
    kb.store_f32(out, dot_val, 0);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_q8_dot() {
    Module mod("mod_q8_dot");
    build_q8_dot(mod);
    return jit_.compile(mod, "fused_q8_dot");
}

std::string MlFusionCompiler::emit_ptx_q8_dot(const target::PtxOptions& opts) {
    Module mod("mod_ptx_q8_dot");
    Function* fn = build_q8_dot(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

Function* MlFusionCompiler::build_block_q8_dot(Module& mod, std::string_view name) {
    mod.add_external_symbol("i32_to_f32");
    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* a = kb.builder().add_block_param(entry, Type::ptr());
    Value* b = kb.builder().add_block_param(entry, Type::ptr());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* num_blocks = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* blk_head = kb.builder().create_block("blk_head");
    BasicBlock* blk_body = kb.builder().create_block("blk_body");
    BasicBlock* wrd_head = kb.builder().create_block("wrd_head");
    BasicBlock* wrd_body = kb.builder().create_block("wrd_body");
    BasicBlock* wrd_exit = kb.builder().create_block("wrd_exit");
    BasicBlock* blk_exit = kb.builder().create_block("blk_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(blk_head, {zero_i64, zero_f32});

    // blk_head: params [blk_idx, total_dot]
    fn->append_block(blk_head);
    Value* blk_idx = kb.builder().add_block_param(blk_head, Type::i64());
    Value* total_dot = kb.builder().add_block_param(blk_head, Type::f32());
    kb.position_at_end(blk_head);
    Value* cond_blk = kb.builder().build_slt(blk_idx, num_blocks);
    kb.builder().build_br_if(cond_blk, blk_body, {}, blk_exit, {total_dot});

    // blk_body: compute block pointer and scales (each block is 36 bytes)
    fn->append_block(blk_body);
    kb.position_at_end(blk_body);
    Value* c36 = kb.builder().build_iconst_i64(36);
    Value* blk_offset = kb.mul(blk_idx, c36);
    Value* blk_a = kb.add(a, blk_offset);
    Value* blk_b = kb.add(b, blk_offset);

    // Scale is float d at offset 0
    Value* da = kb.load_f32(blk_a, 0);
    Value* db = kb.load_f32(blk_b, 0);
    Value* blk_scale = kb.mul(da, db);

    // 32 int8 values (8 i32 words) start at offset 4
    Value* c4 = kb.builder().build_iconst_i64(4);
    Value* qs_a = kb.add(blk_a, c4);
    Value* qs_b = kb.add(blk_b, c4);

    Value* zero_i32 = kb.builder().build_iconst_i32(0);
    kb.builder().build_br(wrd_head, {zero_i64, zero_i32});

    // wrd_head: params [w, block_acc]
    fn->append_block(wrd_head);
    Value* w = kb.builder().add_block_param(wrd_head, Type::i64());
    Value* block_acc = kb.builder().add_block_param(wrd_head, Type::i32());
    kb.position_at_end(wrd_head);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* cond_w = kb.builder().build_slt(w, eight_i64);
    kb.builder().build_br_if(cond_w, wrd_body, {}, wrd_exit, {block_acc});

    // wrd_body: load i32 word from qs_a and qs_b, unpack 4 bytes and accumulate
    fn->append_block(wrd_body);
    kb.position_at_end(wrd_body);
    Value* wa = kb.load_i32_indexed(qs_a, w, 4, 0);
    Value* wb = kb.load_i32_indexed(qs_b, w, 4, 0);

    Value* c24 = kb.builder().build_iconst_i32(24);
    Value* c16 = kb.builder().build_iconst_i32(16);
    Value* c8  = kb.builder().build_iconst_i32(8);

    Value* a0 = kb.builder().build_ashr(kb.builder().build_shl(wa, c24), c24);
    Value* b0 = kb.builder().build_ashr(kb.builder().build_shl(wb, c24), c24);
    Value* p0 = kb.mul(a0, b0);

    Value* a1 = kb.builder().build_ashr(kb.builder().build_shl(wa, c16), c24);
    Value* b1 = kb.builder().build_ashr(kb.builder().build_shl(wb, c16), c24);
    Value* p1 = kb.mul(a1, b1);

    Value* a2 = kb.builder().build_ashr(kb.builder().build_shl(wa, c8), c24);
    Value* b2 = kb.builder().build_ashr(kb.builder().build_shl(wb, c8), c24);
    Value* p2 = kb.mul(a2, b2);

    Value* a3 = kb.builder().build_ashr(wa, c24);
    Value* b3 = kb.builder().build_ashr(wb, c24);
    Value* p3 = kb.mul(a3, b3);

    Value* s01 = kb.add(p0, p1);
    Value* s23 = kb.add(p2, p3);
    Value* s4 = kb.add(s01, s23);
    Value* next_block_acc = kb.add(block_acc, s4);

    Value* next_w = kb.add(w, one_i64);
    kb.builder().build_br(wrd_head, {next_w, next_block_acc});

    // wrd_exit: param [final_block_acc]
    fn->append_block(wrd_exit);
    Value* final_block_acc = kb.builder().add_block_param(wrd_exit, Type::i32());
    kb.position_at_end(wrd_exit);
    Value* block_acc_f = kb.builder().build_call("i32_to_f32", Type::f32(), {final_block_acc});
    Value* block_dot = kb.mul(block_acc_f, blk_scale);
    Value* next_total_dot = kb.add(total_dot, block_dot);

    Value* next_blk_idx = kb.add(blk_idx, one_i64);
    kb.builder().build_br(blk_head, {next_blk_idx, next_total_dot});

    // blk_exit: param [final_total_dot]
    fn->append_block(blk_exit);
    Value* final_total_dot = kb.builder().add_block_param(blk_exit, Type::f32());
    kb.position_at_end(blk_exit);
    kb.store_f32(out, final_total_dot, 0);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_block_q8_dot() {
    Module mod("mod_block_q8_dot");
    build_block_q8_dot(mod);
    return jit_.compile(mod, "fused_block_q8_dot");
}

std::string MlFusionCompiler::emit_ptx_block_q8_dot(const target::PtxOptions& opts) {
    Module mod("mod_ptx_block_q8_dot");
    Function* fn = build_block_q8_dot(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

} // namespace brass::codegen
