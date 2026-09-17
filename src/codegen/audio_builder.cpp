#include <brass/codegen/audio_builder.hpp>

namespace brass::codegen {

// ── Static alignment arrays for ramp construction ───────────────────────────
alignas(32) static const float kRampIndices4Data[4] = {0.0f, 1.0f, 2.0f, 3.0f};
alignas(32) static const float kRampIndices8Data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

// ── Stereo Interleaved SIMD Shuffling ───────────────────────────────────────

void AudioKernelBuilder::deinterleave_stereo_f32x4(Value* s0, Value* s1, Value*& out_l, Value*& out_r) {
    // s0 = [L0, R0, L1, R1], s1 = [L2, R2, L3, R3]
    // out_l = [L0, L1, L2, L3] (lanes 0, 2 from s0 and 0, 2 from s1) -> mask 0x88
    out_l = builder().build_vshuffle(s0, s1, 0x88);
    // out_r = [R0, R1, R2, R3] (lanes 1, 3 from s0 and 1, 3 from s1) -> mask 0xDD
    out_r = builder().build_vshuffle(s0, s1, 0xDD);
}

void AudioKernelBuilder::interleave_stereo_f32x4(Value* l, Value* r, Value*& out_s0, Value*& out_s1) {
    // l = [L0, L1, L2, L3], r = [R0, R1, R2, R3]
    // Low unpack: tmp0 = [L0, L1, R0, R1] via mask 0x44
    Value* tmp0 = builder().build_vshuffle(l, r, 0x44);
    // Transpose to [L0, R0, L1, R1] via mask 0xD8
    out_s0 = builder().build_vshuffle(tmp0, tmp0, 0xD8);

    // High unpack: tmp1 = [L2, L3, R2, R3] via mask 0xEE
    Value* tmp1 = builder().build_vshuffle(l, r, 0xEE);
    // Transpose to [L2, R2, L3, R3] via mask 0xD8
    out_s1 = builder().build_vshuffle(tmp1, tmp1, 0xD8);
}

void AudioKernelBuilder::store_stereo_sample(Value* buf_ptr, Value* frame_idx, Value* l, Value* r) {
    Value* two = const_i64(2);
    Value* base_idx = builder().build_mul(frame_idx, two);
    Value* one = const_i64(1);
    Value* idx_r = builder().build_add(base_idx, one);
    store_f32_indexed(buf_ptr, base_idx, l, 4, 0);
    store_f32_indexed(buf_ptr, idx_r, r, 4, 0);
}

// ── Vector Blend & Clamp ────────────────────────────────────────────────────

Value* AudioKernelBuilder::vlerp_f32x4(Value* a, Value* b, Value* t) {
    Value* diff = vsub(b, a);
    return vfma(t, diff, a);
}

Value* AudioKernelBuilder::vlerp_f32x8(Value* a, Value* b, Value* t) {
    Value* diff = vsub(b, a);
    return vfma(t, diff, a);
}

Value* AudioKernelBuilder::vclamp_f32x4(Value* x, Value* min_val, Value* max_val) {
    Value* clamped_min = vmax(x, min_val);
    return vmin(clamped_min, max_val);
}

Value* AudioKernelBuilder::vclamp_f32x8(Value* x, Value* min_val, Value* max_val) {
    Value* clamped_min = vmax(x, min_val);
    return vmin(clamped_min, max_val);
}

Value* AudioKernelBuilder::clamp_f32(Value* x, Value* min_val, Value* max_val) {
    // In MIR: max(x, min_val), then min with max_val
    Value* cond_min = builder().build_sgt(x, min_val);
    Value* t_min = builder().build_select(cond_min, x, min_val);
    Value* cond_max = builder().build_slt(t_min, max_val);
    return builder().build_select(cond_max, t_min, max_val);
}

// ── IIR Biquad (Direct Form II Transposed) ──────────────────────────────────

Value* AudioKernelBuilder::biquad_df2t(const BiquadCoeffs& c, BiquadState& s, Value* x) {
    // y = b0 * x + z1
    Value* y = fma(c.b0, x, s.z1);
    Value* neg_y = builder().build_neg(y);

    // z1' = b1 * x - a1 * y + z2
    Value* term1 = fma(c.a1, neg_y, s.z2);
    Value* new_z1 = fma(c.b1, x, term1);

    // z2' = b2 * x - a2 * y
    Value* new_z2 = fma(c.b2, x, builder().build_mul(c.a2, neg_y));

    s.z1 = new_z1;
    s.z2 = new_z2;
    return y;
}

void AudioKernelBuilder::biquad_df2t_stereo(const BiquadCoeffs& c, BiquadState& s_l, BiquadState& s_r,
                                            Value* in_l, Value* in_r, Value*& out_l, Value*& out_r) {
    out_l = biquad_df2t(c, s_l, in_l);
    out_r = biquad_df2t(c, s_r, in_r);
}

// ── One-Pole Filter / Smoother ──────────────────────────────────────────────

Value* AudioKernelBuilder::one_pole(Value* in, Value* alpha, Value*& state) {
    Value* diff = sub(in, state);
    Value* y = fma(alpha, diff, state);
    state = y;
    return y;
}

// ── Parameter Smoothing Ramps ───────────────────────────────────────────────

Value* AudioKernelBuilder::vramp_init_f32x4(Value* start, Value* delta) {
    Value* vstart = vbroadcast(Type::f32x4(), start);
    Value* vdelta = vbroadcast(Type::f32x4(), delta);
    Value* ptr_indices = builder().build_iconst_i64(reinterpret_cast<uintptr_t>(kRampIndices4Data));
    Value* vindices = vload_f32x4(ptr_indices);
    return vfma(vdelta, vindices, vstart);
}

Value* AudioKernelBuilder::vramp_step_f32x4(Value* current_ramp, Value* delta4) {
    Value* vstep4 = vbroadcast(Type::f32x4(), delta4);
    return vadd(current_ramp, vstep4);
}

Value* AudioKernelBuilder::vramp_init_f32x8(Value* start, Value* delta) {
    Value* vstart = vbroadcast(Type::f32x8(), start);
    Value* vdelta = vbroadcast(Type::f32x8(), delta);
    Value* ptr_indices = builder().build_iconst_i64(reinterpret_cast<uintptr_t>(kRampIndices8Data));
    Value* vindices = vload_f32x8(ptr_indices);
    return vfma(vdelta, vindices, vstart);
}

Value* AudioKernelBuilder::vramp_step_f32x8(Value* current_ramp, Value* delta8) {
    Value* vstep8 = vbroadcast(Type::f32x8(), delta8);
    return vadd(current_ramp, vstep8);
}

// ── Musical Pitch, Decibels, Waveforms & Math ───────────────────────────────

Value* AudioKernelBuilder::exp2_fast(Value* x) {
    // 2^x = exp(x * ln 2)
    Value* cx = clamp_f32(x, const_f32(-30.0f), const_f32(30.0f));
    Value* z = mul(cx, const_f32(0.69314718056f));
    Value* u = mul(z, const_f32(0.03125f)); // u = z / 32

    // 5th-order Taylor polynomial for exp(u)
    Value* p5 = fma(u, const_f32(1.0f / 120.0f), const_f32(1.0f / 24.0f));
    Value* p4 = fma(u, p5, const_f32(1.0f / 6.0f));
    Value* p3 = fma(u, p4, const_f32(0.5f));
    Value* p2 = fma(u, p3, const_f32(1.0f));
    Value* p1 = fma(u, p2, const_f32(1.0f));

    // 5 squarings: (P(u))^32
    Value* s1 = mul(p1, p1);
    Value* s2 = mul(s1, s1);
    Value* s3 = mul(s2, s2);
    Value* s4 = mul(s3, s3);
    return mul(s4, s4);
}

Value* AudioKernelBuilder::vexp2_f32x4(Value* x) {
    Value* c_min = vbroadcast(Type::f32x4(), const_f32(-30.0f));
    Value* c_max = vbroadcast(Type::f32x4(), const_f32(30.0f));
    Value* cx = vclamp_f32x4(x, c_min, c_max);

    Value* c_ln2 = vbroadcast(Type::f32x4(), const_f32(0.69314718056f));
    Value* z = vmul(cx, c_ln2);
    Value* c_inv32 = vbroadcast(Type::f32x4(), const_f32(0.03125f));
    Value* u = vmul(z, c_inv32);

    Value* c_1_120 = vbroadcast(Type::f32x4(), const_f32(1.0f / 120.0f));
    Value* c_1_24  = vbroadcast(Type::f32x4(), const_f32(1.0f / 24.0f));
    Value* c_1_6   = vbroadcast(Type::f32x4(), const_f32(1.0f / 6.0f));
    Value* c_1_2   = vbroadcast(Type::f32x4(), const_f32(0.5f));
    Value* c_1_0   = vbroadcast(Type::f32x4(), const_f32(1.0f));

    Value* p5 = vfma(u, c_1_120, c_1_24);
    Value* p4 = vfma(u, p5, c_1_6);
    Value* p3 = vfma(u, p4, c_1_2);
    Value* p2 = vfma(u, p3, c_1_0);
    Value* p1 = vfma(u, p2, c_1_0);

    Value* s1 = vmul(p1, p1);
    Value* s2 = vmul(s1, s1);
    Value* s3 = vmul(s2, s2);
    Value* s4 = vmul(s3, s3);
    return vmul(s4, s4);
}

Value* AudioKernelBuilder::vexp2_f32x8(Value* x) {
    Value* c_min = vbroadcast(Type::f32x8(), const_f32(-30.0f));
    Value* c_max = vbroadcast(Type::f32x8(), const_f32(30.0f));
    Value* cx = vclamp_f32x8(x, c_min, c_max);

    Value* c_ln2 = vbroadcast(Type::f32x8(), const_f32(0.69314718056f));
    Value* z = vmul(cx, c_ln2);
    Value* c_inv32 = vbroadcast(Type::f32x8(), const_f32(0.03125f));
    Value* u = vmul(z, c_inv32);

    Value* c_1_120 = vbroadcast(Type::f32x8(), const_f32(1.0f / 120.0f));
    Value* c_1_24  = vbroadcast(Type::f32x8(), const_f32(1.0f / 24.0f));
    Value* c_1_6   = vbroadcast(Type::f32x8(), const_f32(1.0f / 6.0f));
    Value* c_1_2   = vbroadcast(Type::f32x8(), const_f32(0.5f));
    Value* c_1_0   = vbroadcast(Type::f32x8(), const_f32(1.0f));

    Value* p5 = vfma(u, c_1_120, c_1_24);
    Value* p4 = vfma(u, p5, c_1_6);
    Value* p3 = vfma(u, p4, c_1_2);
    Value* p2 = vfma(u, p3, c_1_0);
    Value* p1 = vfma(u, p2, c_1_0);

    Value* s1 = vmul(p1, p1);
    Value* s2 = vmul(s1, s1);
    Value* s3 = vmul(s2, s2);
    Value* s4 = vmul(s3, s3);
    return vmul(s4, s4);
}

Value* AudioKernelBuilder::pitch_to_freq_ratio(Value* semitones) {
    Value* inv12 = const_f32(1.0f / 12.0f);
    Value* octaves = mul(semitones, inv12);
    return exp2_fast(octaves);
}

Value* AudioKernelBuilder::vpitch_to_freq_ratio_f32x8(Value* semitones) {
    Value* c_inv12 = vbroadcast(Type::f32x8(), const_f32(1.0f / 12.0f));
    Value* octaves = vmul(semitones, c_inv12);
    return vexp2_f32x8(octaves);
}

Value* AudioKernelBuilder::db_to_linear(Value* db) {
    // 10^(db / 20) = 2^(db * (log2(10) / 20)) = 2^(db * 0.16609640474f)
    Value* scale = const_f32(0.16609640474f);
    return exp2_fast(mul(db, scale));
}

Value* AudioKernelBuilder::vdb_to_linear_f32x8(Value* db) {
    Value* c_scale = vbroadcast(Type::f32x8(), const_f32(0.16609640474f));
    return vexp2_f32x8(vmul(db, c_scale));
}

Value* AudioKernelBuilder::linear_to_db(Value* lin) {
    // 20 * log10(lin) = log2(lin) * (20 * log10(2)) = log2(lin) * 6.0205999f
    Value* l2 = log2_fast(lin);
    return mul(l2, const_f32(6.02059991328f));
}

Value* AudioKernelBuilder::log2_fast(Value* x) {
    // ln(x) approx using rational approximation: 2 * (x - 1) / (x + 1)
    // divided by ln(2): log2(x)
    Value* one = const_f32(1.0f);
    Value* num = sub(x, one);
    Value* den = add(x, one);
    Value* t = div(num, den);
    Value* t2 = mul(t, t);

    // Horner: t * (2.88539008178 + t^2 * (0.9617966939 + t^2 * 0.577078))
    Value* p3 = fma(t2, const_f32(0.577078f), const_f32(0.9617966939f));
    Value* p2 = fma(t2, p3, const_f32(2.88539008178f));
    return mul(t, p2);
}

Value* AudioKernelBuilder::vlog2_f32x8(Value* x) {
    Value* c_1 = vbroadcast(Type::f32x8(), const_f32(1.0f));
    Value* num = vsub(x, c_1);
    Value* den = vadd(x, c_1);
    Value* t = vdiv(num, den);
    Value* t2 = vmul(t, t);

    Value* c_p3 = vbroadcast(Type::f32x8(), const_f32(0.577078f));
    Value* c_p2 = vbroadcast(Type::f32x8(), const_f32(0.9617966939f));
    Value* c_p1 = vbroadcast(Type::f32x8(), const_f32(2.88539008178f));

    Value* p3 = vfma(t2, c_p3, c_p2);
    Value* p2 = vfma(t2, p3, c_p1);
    return vmul(t, p2);
}

Value* AudioKernelBuilder::sin_norm(Value* phase01) {
    // phase01 in [0, 1) -> sin(2 * pi * phase)
    // Reduce to [-0.25, 0.25] via vfold(y) = max(min(y, 0.5 - y), -0.5 - y)
    Value* c_05 = const_f32(0.5f);
    Value* c_neg05 = const_f32(-0.5f);
    Value* y = sub(phase01, c_05);

    Value* fold_pos = sub(c_05, y);
    Value* is_less = builder().build_slt(y, fold_pos);
    Value* t1 = builder().build_select(is_less, y, fold_pos);

    Value* fold_neg = sub(c_neg05, y);
    Value* is_greater = builder().build_sgt(t1, fold_neg);
    Value* t = builder().build_select(is_greater, t1, fold_neg);

    Value* th = mul(t, const_f32(6.28318530718f));
    Value* th2 = mul(th, th);

    Value* p4 = fma(th2, const_f32(1.0f / 362880.0f), const_f32(-1.0f / 5040.0f));
    Value* p3 = fma(th2, p4, const_f32(1.0f / 120.0f));
    Value* p2 = fma(th2, p3, const_f32(-1.0f / 6.0f));
    Value* p1 = fma(th2, p2, const_f32(1.0f));

    return mul(builder().build_neg(th), p1);
}

Value* AudioKernelBuilder::vsin_norm_f32x8(Value* phase01) {
    Value* c_05 = vbroadcast(Type::f32x8(), const_f32(0.5f));
    Value* c_neg05 = vbroadcast(Type::f32x8(), const_f32(-0.5f));
    Value* y = vsub(phase01, c_05);

    Value* fold_pos = vsub(c_05, y);
    Value* t1 = vmin(y, fold_pos);
    Value* fold_neg = vsub(c_neg05, y);
    Value* t = vmax(t1, fold_neg);

    Value* c_2pi = vbroadcast(Type::f32x8(), const_f32(6.28318530718f));
    Value* th = vmul(t, c_2pi);
    Value* th2 = vmul(th, th);

    Value* c_p4 = vbroadcast(Type::f32x8(), const_f32(1.0f / 362880.0f));
    Value* c_p3 = vbroadcast(Type::f32x8(), const_f32(-1.0f / 5040.0f));
    Value* c_p2 = vbroadcast(Type::f32x8(), const_f32(1.0f / 120.0f));
    Value* c_p1 = vbroadcast(Type::f32x8(), const_f32(-1.0f / 6.0f));
    Value* c_1  = vbroadcast(Type::f32x8(), const_f32(1.0f));

    Value* p4 = vfma(th2, c_p4, c_p3);
    Value* p3 = vfma(th2, p4, c_p2);
    Value* p2 = vfma(th2, p3, c_p1);
    Value* p1 = vfma(th2, p2, c_1);

    Value* c_neg1 = vbroadcast(Type::f32x8(), const_f32(-1.0f));
    Value* neg_th = vmul(th, c_neg1);
    return vmul(neg_th, p1);
}

Value* AudioKernelBuilder::cos_norm(Value* phase01) {
    // cos(2*pi*p) is even around 0.5.
    // p_sym = min(p, 1 - p) in [0, 0.5].
    // t = 0.25 - p_sym in [-0.25, 0.25].
    // cos(2*pi*p) = sin(2*pi*(0.25 - p_sym))
    Value* c_1 = const_f32(1.0f);
    Value* c_025 = const_f32(0.25f);
    Value* one_minus_p = sub(c_1, phase01);
    Value* is_less = builder().build_slt(phase01, one_minus_p);
    Value* p_sym = builder().build_select(is_less, phase01, one_minus_p);
    Value* t = sub(c_025, p_sym);

    Value* th = mul(t, const_f32(6.28318530718f));
    Value* th2 = mul(th, th);

    Value* p4 = fma(th2, const_f32(1.0f / 362880.0f), const_f32(-1.0f / 5040.0f));
    Value* p3 = fma(th2, p4, const_f32(1.0f / 120.0f));
    Value* p2 = fma(th2, p3, const_f32(-1.0f / 6.0f));
    Value* p1 = fma(th2, p2, const_f32(1.0f));

    return mul(th, p1);
}

Value* AudioKernelBuilder::vcos_norm_f32x8(Value* phase01) {
    Value* c_1 = vbroadcast(Type::f32x8(), const_f32(1.0f));
    Value* c_025 = vbroadcast(Type::f32x8(), const_f32(0.25f));
    Value* one_minus_p = vsub(c_1, phase01);
    Value* p_sym = vmin(phase01, one_minus_p);
    Value* t = vsub(c_025, p_sym);

    Value* c_2pi = vbroadcast(Type::f32x8(), const_f32(6.28318530718f));
    Value* th = vmul(t, c_2pi);
    Value* th2 = vmul(th, th);

    Value* c_p4 = vbroadcast(Type::f32x8(), const_f32(1.0f / 362880.0f));
    Value* c_p3 = vbroadcast(Type::f32x8(), const_f32(-1.0f / 5040.0f));
    Value* c_p2 = vbroadcast(Type::f32x8(), const_f32(1.0f / 120.0f));
    Value* c_p1 = vbroadcast(Type::f32x8(), const_f32(-1.0f / 6.0f));

    Value* p4 = vfma(th2, c_p4, c_p3);
    Value* p3 = vfma(th2, p4, c_p2);
    Value* p2 = vfma(th2, p3, c_p1);
    Value* p1 = vfma(th2, p2, c_1);

    return vmul(th, p1);
}

Value* AudioKernelBuilder::sin_fast(Value* rad) {
    // Convert rad to normalized phase: phase01 = rad * (1 / (2 * pi))
    Value* inv2pi = const_f32(0.15915494309f);
    Value* p = mul(rad, inv2pi);
    return sin_norm(p);
}

Value* AudioKernelBuilder::vsin_f32x8(Value* rad) {
    Value* c_inv2pi = vbroadcast(Type::f32x8(), const_f32(0.15915494309f));
    Value* p = vmul(rad, c_inv2pi);
    return vsin_norm_f32x8(p);
}

Value* AudioKernelBuilder::cos_fast(Value* rad) {
    Value* inv2pi = const_f32(0.15915494309f);
    Value* p = mul(rad, inv2pi);
    return cos_norm(p);
}

Value* AudioKernelBuilder::vcos_f32x8(Value* rad) {
    Value* c_inv2pi = vbroadcast(Type::f32x8(), const_f32(0.15915494309f));
    Value* p = vmul(rad, c_inv2pi);
    return vcos_norm_f32x8(p);
}

// ── Saturation / Waveshaping ────────────────────────────────────────────────

Value* AudioKernelBuilder::soft_clip(Value* x) {
    // tanh(x) = 2 / (1 + exp(-2x)) - 1
    Value* neg2x = mul(x, const_f32(-2.0f));
    Value* c_neg2x = clamp_f32(neg2x, const_f32(-20.0f), const_f32(20.0f));
    Value* exp_val = exp2_fast(mul(c_neg2x, const_f32(1.4426950408889634f)));
    Value* denom = add(const_f32(1.0f), exp_val);
    Value* two_over_denom = div(const_f32(2.0f), denom);
    return sub(two_over_denom, const_f32(1.0f));
}

Value* AudioKernelBuilder::vsoft_clip_f32x8(Value* x) {
    Value* c_neg2 = vbroadcast(Type::f32x8(), const_f32(-2.0f));
    Value* neg2x = vmul(x, c_neg2);
    Value* c_bound = vbroadcast(Type::f32x8(), const_f32(20.0f));
    Value* c_nbound = vbroadcast(Type::f32x8(), const_f32(-20.0f));
    Value* c_neg2x = vclamp_f32x8(neg2x, c_nbound, c_bound);

    Value* c_log2e = vbroadcast(Type::f32x8(), const_f32(1.4426950408889634f));
    Value* exp_val = vexp2_f32x8(vmul(c_neg2x, c_log2e));

    Value* c_1 = vbroadcast(Type::f32x8(), const_f32(1.0f));
    Value* c_2 = vbroadcast(Type::f32x8(), const_f32(2.0f));
    Value* denom = vadd(c_1, exp_val);
    Value* two_over_denom = vdiv(c_2, denom);
    return vsub(two_over_denom, c_1);
}

Value* AudioKernelBuilder::hard_clip(Value* x, Value* threshold) {
    Value* neg_thresh = builder().build_neg(threshold);
    return clamp_f32(x, neg_thresh, threshold);
}

Value* AudioKernelBuilder::vhard_clip_f32x8(Value* x, Value* threshold) {
    Value* c_neg1 = vbroadcast(Type::f32x8(), const_f32(-1.0f));
    Value* neg_thresh = vmul(threshold, c_neg1);
    return vclamp_f32x8(x, neg_thresh, threshold);
}

Value* AudioKernelBuilder::foldback(Value* x, Value* threshold) {
    // Soft foldback approximation: x - 4 * threshold * floor((x + threshold) / (4 * threshold))
    // Or for audio saturation in [-threshold, threshold]:
    Value* four_th = mul(threshold, const_f32(4.0f));
    Value* inv_four_th = div(const_f32(1.0f), four_th);
    Value* norm = mul(add(x, threshold), inv_four_th);
    Value* int_norm = builder().build_fptosi_i32(norm);
    Value* flr = builder().build_sitofp_f64_i32(int_norm); // f64, cast back to f32
    Value* flr_f32 = builder().build_trunc_i32(int_norm);
    (void)flr_f32;
    Value* sub_term = mul(builder().build_fconst_f32(1.0f), flr); // float
    return sub(x, mul(sub_term, four_th));
}

// ── Circular Delay Line & Fractional Interpolation ──────────────────────────

Value* AudioKernelBuilder::load_circular_mask(Value* base_ptr, Value* idx, Value* mask) {
    Value* wrapped_idx = builder().build_and(idx, mask);
    return load_f32_indexed(base_ptr, wrapped_idx, 4, 0);
}

Instruction* AudioKernelBuilder::store_circular_mask(Value* base_ptr, Value* idx, Value* mask, Value* val) {
    Value* wrapped_idx = builder().build_and(idx, mask);
    return store_f32_indexed(base_ptr, wrapped_idx, val, 4, 0);
}

Value* AudioKernelBuilder::hermite_interp4(Value* p0, Value* p1, Value* p2, Value* p3, Value* t) {
    Value* half = const_f32(0.5f);
    Value* c0 = p1;
    // c1 = 0.5 * (p2 - p0)
    Value* c1 = mul(half, sub(p2, p0));
    // c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
    Value* term2a = fma(const_f32(-2.5f), p1, p0);
    Value* term2b = fma(const_f32(2.0f), p2, term2a);
    Value* c2 = fma(const_f32(-0.5f), p3, term2b);
    // c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2)
    Value* term3a = mul(half, sub(p3, p0));
    Value* c3 = fma(const_f32(1.5f), sub(p1, p2), term3a);

    // Horner form: ((c3 * t + c2) * t + c1) * t + c0
    Value* r2 = fma(c3, t, c2);
    Value* r1 = fma(r2, t, c1);
    return fma(r1, t, c0);
}

Value* AudioKernelBuilder::load_delay_hermite(Value* ring_buf, Value* head, Value* delay_samples, Value* mask) {
    // tap position: head - delay_samples
    Value* tap_pos = sub(head, delay_samples);
    // Integer base
    Value* int_tap = builder().build_fptosi_i64(tap_pos);
    Value* float_base = builder().build_sitofp_f64_i64(int_tap);
    Value* frac = sub(tap_pos, float_base);

    Value* one = const_i64(1);
    Value* two = const_i64(2);

    Value* idx0 = builder().build_sub(int_tap, one);
    Value* idx1 = int_tap;
    Value* idx2 = builder().build_add(int_tap, one);
    Value* idx3 = builder().build_add(int_tap, two);

    Value* p0 = load_circular_mask(ring_buf, idx0, mask);
    Value* p1 = load_circular_mask(ring_buf, idx1, mask);
    Value* p2 = load_circular_mask(ring_buf, idx2, mask);
    Value* p3 = load_circular_mask(ring_buf, idx3, mask);

    return hermite_interp4(p0, p1, p2, p3, frac);
}

} // namespace brass::codegen
