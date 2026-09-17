#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>

#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
// The AVX2 / F16C paths below are already behind __AVX2__ / __F16C__; the
// header has to be gated the same way, since <immintrin.h> is an #error on
// arm64 (AppleClang 15 on an Apple Silicon runner).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace brass::codegen {

namespace {

// Half-precision float (FP16) to single-precision float (FP32) conversion
inline float fp16_to_f32_bits(uint16_t h) {
#if defined(__F16C__) || defined(__AVX2__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            float f = 0.0f;
            uint32_t u = sign;
            std::memcpy(&f, &u, 4);
            return f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    exp = exp + (127 - 15);
    uint32_t u = sign | (static_cast<uint32_t>(exp) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

// Thread-local scratch buffer for dequantized block weights (256 floats max)
extern "C" float* brass_get_tls_scratch() {
    static thread_local float scratch[256];
    return scratch;
}

// Dequantize a Q8_0 block (32 weights): FP16 scale d + 32 signed int8s
extern "C" void brass_dequant_q8_0_block(const void* blk_ptr, float* out32) {
    uint16_t d_raw;
    std::memcpy(&d_raw, blk_ptr, 2);
    const float d = fp16_to_f32_bits(d_raw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(static_cast<const char*>(blk_ptr) + 2);
#if defined(__AVX2__)
    __m256 d_vec = _mm256_set1_ps(d);
    for (int i = 0; i < 4; ++i) {
        __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i * 8));
        __m256i q32 = _mm256_cvtepi8_epi32(q8);
        __m256 qf = _mm256_cvtepi32_ps(q32);
        __m256 w = _mm256_mul_ps(qf, d_vec);
        _mm256_storeu_ps(out32 + i * 8, w);
    }
#else
    for (int i = 0; i < 32; ++i) {
        out32[i] = static_cast<float>(qs[i]) * d;
    }
#endif
}

// Dequantize a Q4_K block (256 weights): 16-byte header + 128 bytes 4-bit nibbles
extern "C" void brass_dequant_q4k_block(const void* blk_ptr, float* out256) {
    const uint8_t* p = static_cast<const uint8_t*>(blk_ptr);
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw, p, 2);
    std::memcpy(&dmin_raw, p + 2, 2);
    const float d = fp16_to_f32_bits(d_raw);
    const float dmin = fp16_to_f32_bits(dmin_raw);

    const uint8_t* scales = p + 4;
    const uint8_t* qs = p + 16;

    uint8_t sc[8], m[8];
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = scales[j]     & 0x3F;
            m[j]  = scales[j + 4] & 0x3F;
        } else {
            sc[j] = static_cast<uint8_t>((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            m[j]  = static_cast<uint8_t>((scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4));
        }
    }

    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        const int is_lo = 2 * p_idx;
        const int is_hi = 2 * p_idx + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;

        for (int l = 0; l < 32; ++l) {
            const uint8_t byte = qs[p_idx * 32 + l];
            const int n_lo = byte & 0x0F;
            const int n_hi = (byte >> 4) & 0x0F;
            out256[is_lo * 32 + l] = w_lo * static_cast<float>(n_lo) - b_lo;
            out256[is_hi * 32 + l] = w_hi * static_cast<float>(n_hi) - b_hi;
        }
    }
}

// Horizontal sum of an 8-wide float vector using destination scratch buffer
static Value* reduce_vsum8(KernelBuilder& kb, Value* scratch, Value* vsum) {
    kb.vstore_f32x8(scratch, vsum);
    Value* s0 = kb.load_f32(scratch, 0);  Value* s1 = kb.load_f32(scratch, 4);
    Value* s2 = kb.load_f32(scratch, 8);  Value* s3 = kb.load_f32(scratch, 12);
    Value* s4 = kb.load_f32(scratch, 16); Value* s5 = kb.load_f32(scratch, 20);
    Value* s6 = kb.load_f32(scratch, 24); Value* s7 = kb.load_f32(scratch, 28);
    Value* sum0123 = kb.add(kb.add(s0, s1), kb.add(s2, s3));
    Value* sum4567 = kb.add(kb.add(s4, s5), kb.add(s6, s7));
    return kb.add(sum0123, sum4567);
}

} // namespace

// =========================================================================
// 1. Q8_0 AVX2 GEMV MIR Builder & Compiler
// =========================================================================

Function* MlFusionCompiler::build_gemv_q8_0(Module& mod, std::string_view name) {
    mod.add_external_symbol("brass_get_tls_scratch");
    mod.add_external_symbol("brass_dequant_q8_0_block");

    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::i64()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry    = kb.builder().append_block("entry");
    BasicBlock* row_head = kb.builder().create_block("row_head");
    BasicBlock* row_body = kb.builder().create_block("row_body");
    BasicBlock* blk_head = kb.builder().create_block("blk_head");
    BasicBlock* blk_body = kb.builder().create_block("blk_body");
    BasicBlock* blk_exit = kb.builder().create_block("blk_exit");
    BasicBlock* row_exit = kb.builder().create_block("row_exit");

    kb.position_at_end(entry);
    Value* w = kb.builder().add_block_param(entry, Type::ptr());
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());
    Value* k = kb.builder().add_block_param(entry, Type::i64());

    Value* scratch = kb.builder().build_call("brass_get_tls_scratch", Type::ptr(), {});

    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64  = kb.builder().build_iconst_i64(1);
    Value* five_i64 = kb.builder().build_iconst_i64(5);
    Value* c34_i64  = kb.builder().build_iconst_i64(34);
    Value* vzero    = kb.vzero(Type::f32x8());

    // num_blocks = K / 32
    Value* num_blocks = kb.builder().build_lshr(k, five_i64);
    // row_bytes = num_blocks * 34
    Value* row_bytes = kb.mul(num_blocks, c34_i64);

    kb.builder().build_br(row_head, {zero_i64});

    // --- row_head: params [row] ---
    fn->append_block(row_head);
    Value* row = kb.builder().add_block_param(row_head, Type::i64());
    kb.position_at_end(row_head);
    Value* cond_row = kb.builder().build_slt(row, n);
    kb.builder().build_br_if(cond_row, row_body, row_exit);

    // --- row_body ---
    fn->append_block(row_body);
    kb.position_at_end(row_body);
    Value* row_w_offset = kb.mul(row, row_bytes);
    Value* row_w = kb.add(w, row_w_offset);
    kb.builder().build_br(blk_head, {zero_i64, vzero});

    // --- blk_head: params [blk, vsum] ---
    fn->append_block(blk_head);
    Value* blk = kb.builder().add_block_param(blk_head, Type::i64());
    Value* vsum = kb.builder().add_block_param(blk_head, Type::f32x8());
    kb.position_at_end(blk_head);
    Value* cond_blk = kb.builder().build_slt(blk, num_blocks);
    kb.builder().build_br_if(cond_blk, blk_body, {}, blk_exit, {vsum});

    // --- blk_body ---
    fn->append_block(blk_body);
    kb.position_at_end(blk_body);
    Value* blk_w_offset = kb.mul(blk, c34_i64);
    Value* blk_ptr = kb.add(row_w, blk_w_offset);

    // Dequantize 32 weights to scratch
    kb.builder().build_call("brass_dequant_q8_0_block", Type::void_type(), {blk_ptr, scratch});

    // 4 vector FMA steps (each 8 floats)
    Value* blk_k_base = kb.builder().build_shl(blk, five_i64); // blk * 32
    Value* cur_vsum = vsum;
    for (int c = 0; c < 4; ++c) {
        Value* vw = kb.vload_f32x8(scratch, c * 32);

        Value* sub_elem = kb.builder().build_iconst_i64(c * 8);
        Value* k_idx = kb.add(blk_k_base, sub_elem);
        Value* x_byte_off = kb.builder().build_shl(k_idx, kb.builder().build_iconst_i64(2)); // * 4
        Value* px = kb.add(x, x_byte_off);
        Value* vx = kb.vload_f32x8(px);

        cur_vsum = kb.vfma(vw, vx, cur_vsum);
    }

    Value* next_blk = kb.add(blk, one_i64);
    kb.builder().build_br(blk_head, {next_blk, cur_vsum});

    // --- blk_exit: params [final_vsum] ---
    fn->append_block(blk_exit);
    Value* final_vsum = kb.builder().add_block_param(blk_exit, Type::f32x8());
    kb.position_at_end(blk_exit);
    Value* row_dot = reduce_vsum8(kb, scratch, final_vsum);
    kb.store_f32_indexed(y, row, row_dot, 4, 0);

    Value* next_row = kb.add(row, one_i64);
    kb.builder().build_br(row_head, {next_row});

    // --- row_exit ---
    fn->append_block(row_exit);
    kb.position_at_end(row_exit);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_gemv_q8_0() {
    jit_.register_external_symbol("brass_get_tls_scratch", reinterpret_cast<void*>(&brass_get_tls_scratch));
    jit_.register_external_symbol("brass_dequant_q8_0_block", reinterpret_cast<void*>(&brass_dequant_q8_0_block));
    Module mod("mod_gemv_q8_0");
    build_gemv_q8_0(mod);
    return jit_.compile(mod, "gemv_q8_0");
}

// =========================================================================
// 2. Q4_K AVX2 GEMV MIR Builder & Compiler
// =========================================================================

Function* MlFusionCompiler::build_gemv_q4_k(Module& mod, std::string_view name) {
    mod.add_external_symbol("brass_get_tls_scratch");
    mod.add_external_symbol("brass_dequant_q4k_block");

    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::i64()}
    );
    KernelBuilder kb(mod, fn);

    BasicBlock* entry    = kb.builder().append_block("entry");
    BasicBlock* row_head = kb.builder().create_block("row_head");
    BasicBlock* row_body = kb.builder().create_block("row_body");
    BasicBlock* blk_head = kb.builder().create_block("blk_head");
    BasicBlock* blk_body = kb.builder().create_block("blk_body");
    BasicBlock* blk_exit = kb.builder().create_block("blk_exit");
    BasicBlock* row_exit = kb.builder().create_block("row_exit");

    kb.position_at_end(entry);
    Value* w = kb.builder().add_block_param(entry, Type::ptr());
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());
    Value* k = kb.builder().add_block_param(entry, Type::i64());

    Value* scratch = kb.builder().build_call("brass_get_tls_scratch", Type::ptr(), {});

    Value* zero_i64  = kb.builder().build_iconst_i64(0);
    Value* one_i64   = kb.builder().build_iconst_i64(1);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* c144_i64  = kb.builder().build_iconst_i64(144);
    Value* vzero     = kb.vzero(Type::f32x8());

    // num_blocks = K / 256
    Value* num_blocks = kb.builder().build_lshr(k, eight_i64);
    // row_bytes = num_blocks * 144
    Value* row_bytes = kb.mul(num_blocks, c144_i64);

    kb.builder().build_br(row_head, {zero_i64});

    // --- row_head: params [row] ---
    fn->append_block(row_head);
    Value* row = kb.builder().add_block_param(row_head, Type::i64());
    kb.position_at_end(row_head);
    Value* cond_row = kb.builder().build_slt(row, n);
    kb.builder().build_br_if(cond_row, row_body, row_exit);

    // --- row_body ---
    fn->append_block(row_body);
    kb.position_at_end(row_body);
    Value* row_w_offset = kb.mul(row, row_bytes);
    Value* row_w = kb.add(w, row_w_offset);
    kb.builder().build_br(blk_head, {zero_i64, vzero});

    // --- blk_head: params [blk, vsum] ---
    fn->append_block(blk_head);
    Value* blk = kb.builder().add_block_param(blk_head, Type::i64());
    Value* vsum = kb.builder().add_block_param(blk_head, Type::f32x8());
    kb.position_at_end(blk_head);
    Value* cond_blk = kb.builder().build_slt(blk, num_blocks);
    kb.builder().build_br_if(cond_blk, blk_body, {}, blk_exit, {vsum});

    // --- blk_body ---
    fn->append_block(blk_body);
    kb.position_at_end(blk_body);
    Value* blk_w_offset = kb.mul(blk, c144_i64);
    Value* blk_ptr = kb.add(row_w, blk_w_offset);

    // Dequantize 256 weights to scratch
    kb.builder().build_call("brass_dequant_q4k_block", Type::void_type(), {blk_ptr, scratch});

    // 32 vector FMA steps (each 8 floats = 256 floats)
    Value* blk_k_base = kb.builder().build_shl(blk, eight_i64); // blk * 256
    Value* cur_vsum = vsum;
    for (int c = 0; c < 32; ++c) {
        Value* vw = kb.vload_f32x8(scratch, c * 32);

        Value* sub_elem = kb.builder().build_iconst_i64(c * 8);
        Value* k_idx = kb.add(blk_k_base, sub_elem);
        Value* x_byte_off = kb.builder().build_shl(k_idx, kb.builder().build_iconst_i64(2)); // * 4
        Value* px = kb.add(x, x_byte_off);
        Value* vx = kb.vload_f32x8(px);

        cur_vsum = kb.vfma(vw, vx, cur_vsum);
    }

    Value* next_blk = kb.add(blk, one_i64);
    kb.builder().build_br(blk_head, {next_blk, cur_vsum});

    // --- blk_exit: params [final_vsum] ---
    fn->append_block(blk_exit);
    Value* final_vsum = kb.builder().add_block_param(blk_exit, Type::f32x8());
    kb.position_at_end(blk_exit);
    Value* row_dot = reduce_vsum8(kb, scratch, final_vsum);
    kb.store_f32_indexed(y, row, row_dot, 4, 0);

    Value* next_row = kb.add(row, one_i64);
    kb.builder().build_br(row_head, {next_row});

    // --- row_exit ---
    fn->append_block(row_exit);
    kb.position_at_end(row_exit);
    kb.builder().build_ret_void();

    return fn;
}

KernelFunction MlFusionCompiler::compile_gemv_q4_k() {
    jit_.register_external_symbol("brass_get_tls_scratch", reinterpret_cast<void*>(&brass_get_tls_scratch));
    jit_.register_external_symbol("brass_dequant_q4k_block", reinterpret_cast<void*>(&brass_dequant_q4k_block));
    Module mod("mod_gemv_q4_k");
    build_gemv_q4_k(mod);
    return jit_.compile(mod, "gemv_q4_k");
}

} // namespace brass::codegen
