#pragma once

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace brass::codegen {

// ─── ScopedFtzDaz ────────────────────────────────────────────────────────────
//
// RAII guard that enables Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ)
// on the current thread, restoring the previous processor state upon destruction.
// Essential in audio DSP to avoid CPU microcode traps on subnormal float decay.
class ScopedFtzDaz {
public:
    ScopedFtzDaz() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        saved_mxcsr_ = _mm_getcsr();
        // FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040)
        _mm_setcsr(saved_mxcsr_ | 0x8040);
#elif defined(__aarch64__) || defined(_M_ARM64)
        uint64_t fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        saved_fpcr_ = fpcr;
        // FZ = bit 24 (Flush to zero)
        fpcr |= (1ULL << 24);
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
    }

    ~ScopedFtzDaz() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr(saved_mxcsr_);
#elif defined(__aarch64__) || defined(_M_ARM64)
        __asm__ __volatile__("msr fpcr, %0" : : "r"(saved_fpcr_));
#endif
    }

    ScopedFtzDaz(const ScopedFtzDaz&) = delete;
    ScopedFtzDaz& operator=(const ScopedFtzDaz&) = delete;

    static void set_ftz_daz(bool enable) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        unsigned int mxcsr = _mm_getcsr();
        if (enable) {
            mxcsr |= 0x8040;
        } else {
            mxcsr &= ~0x8040;
        }
        _mm_setcsr(mxcsr);
#elif defined(__aarch64__) || defined(_M_ARM64)
        uint64_t fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        if (enable) {
            fpcr |= (1ULL << 24);
        } else {
            fpcr &= ~(1ULL << 24);
        }
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#else
        (void)enable;
#endif
    }

    static bool is_ftz_daz_enabled() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        return (_mm_getcsr() & 0x8040) == 0x8040;
#elif defined(__aarch64__) || defined(_M_ARM64)
        uint64_t fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        return (fpcr & (1ULL << 24)) != 0;
#else
        return false;
#endif
    }

private:
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int saved_mxcsr_ = 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    uint64_t saved_fpcr_ = 0;
#endif
};

// ─── Biquad Coeffs and State ────────────────────────────────────────────────
struct BiquadCoeffs {
    Value* b0 = nullptr;
    Value* b1 = nullptr;
    Value* b2 = nullptr;
    Value* a1 = nullptr;
    Value* a2 = nullptr;
};

struct BiquadState {
    Value* z1 = nullptr;
    Value* z2 = nullptr;
};

// ─── AudioKernelBuilder ──────────────────────────────────────────────────────
//
// Specialized MIR builder for audio DSP, synthesis, and bus FX chain JIT compilation.
// Implements Direct Form II Transposed IIR biquad steps, stereo deinterleave / interleave,
// parameter smoothing ramps, circular delay addressing, fast musical pitch/decibel math,
// and waveshaping / saturation.
class AudioKernelBuilder : public KernelBuilder {
public:
    AudioKernelBuilder(Module& mod, Function* fn = nullptr)
        : KernelBuilder(mod, fn) {}
    explicit AudioKernelBuilder(Builder& b) noexcept
        : KernelBuilder(b) {}

    // ── Stereo Interleaved SIMD Shuffling ───────────────────────────────────
    // Deinterleaves 4 stereo pairs across two f32x4 registers into 4 Left and 4 Right samples.
    // s0 = [L0, R0, L1, R1], s1 = [L2, R2, L3, R3]
    // out_l = [L0, L1, L2, L3], out_r = [R0, R1, R2, R3]
    void deinterleave_stereo_f32x4(Value* s0, Value* s1, Value*& out_l, Value*& out_r);

    // Interleaves 4 Left and 4 Right samples back into two stereo f32x4 registers.
    // l = [L0, L1, L2, L3], r = [R0, R1, R2, R3]
    // out_s0 = [L0, R0, L1, R1], out_s1 = [L2, R2, L3, R3]
    void interleave_stereo_f32x4(Value* l, Value* r, Value*& out_s0, Value*& out_s1);

    // Writes an interleaved stereo sample [l, r] to buf_ptr at index (2 * frame_idx)
    void store_stereo_sample(Value* buf_ptr, Value* frame_idx, Value* l, Value* r);

    // ── Vector Blend & Clamp ────────────────────────────────────────────────
    Value* vlerp_f32x4(Value* a, Value* b, Value* t);
    Value* vlerp_f32x8(Value* a, Value* b, Value* t);
    Value* vclamp_f32x4(Value* x, Value* min_val, Value* max_val);
    Value* vclamp_f32x8(Value* x, Value* min_val, Value* max_val);
    Value* clamp_f32(Value* x, Value* min_val, Value* max_val);

    // ── IIR Biquad (Direct Form II Transposed) ──────────────────────────────
    // Computes:
    //   y   = b0 * x + z1
    //   z1' = b1 * x - a1 * y + z2
    //   z2' = b2 * x - a2 * y
    // Returns y, updates s.z1 and s.z2 in-place.
    Value* biquad_df2t(const BiquadCoeffs& c, BiquadState& s, Value* x);

    // Stereo biquad step applying the same coefficients to Left and Right channels.
    void biquad_df2t_stereo(const BiquadCoeffs& c, BiquadState& s_l, BiquadState& s_r,
                            Value* in_l, Value* in_r, Value*& out_l, Value*& out_r);

    // ── One-Pole Filter / Smoother ──────────────────────────────────────────
    // Computes:
    //   y = state + alpha * (in - state)
    // Returns y, updates state.
    Value* one_pole(Value* in, Value* alpha, Value*& state);

    // ── Parameter Smoothing Ramps ───────────────────────────────────────────
    // Constructs an initial 4-wide vector ramp: [start, start+delta, start+2*delta, start+3*delta]
    Value* vramp_init_f32x4(Value* start, Value* delta);
    // Advances 4-wide vector ramp by 4 * delta
    Value* vramp_step_f32x4(Value* current_ramp, Value* delta4);

    // Constructs an initial 8-wide vector ramp: [start, start+delta, ..., start+7*delta]
    Value* vramp_init_f32x8(Value* start, Value* delta);
    // Advances 8-wide vector ramp by 8 * delta
    Value* vramp_step_f32x8(Value* current_ramp, Value* delta8);

    // ── Musical Pitch, Decibels, Waveforms & Math ────────────────────────────
    // Computes 2^x branch-free via Horner polynomial & squaring.
    Value* exp2_fast(Value* x);
    Value* vexp2_f32x4(Value* x);
    Value* vexp2_f32x8(Value* x);

    // Converts pitch bend in semitones to frequency ratio: 2^(semitones / 12)
    Value* pitch_to_freq_ratio(Value* semitones);
    Value* vpitch_to_freq_ratio_f32x8(Value* semitones);

    // Decibel to linear amplitude conversion: 10^(db / 20) = 2^(db * 0.1660964)
    Value* db_to_linear(Value* db);
    Value* vdb_to_linear_f32x8(Value* db);

    // Linear amplitude to decibels: 20 * log10(lin) = 6.0205999 * log2(lin)
    Value* linear_to_db(Value* lin);

    // Fast log2(x)
    Value* log2_fast(Value* x);
    Value* vlog2_f32x8(Value* x);

    // Trigonometric functions
    Value* sin_fast(Value* rad);
    Value* vsin_f32x8(Value* rad);
    Value* cos_fast(Value* rad);
    Value* vcos_f32x8(Value* rad);

    // Normalized oscillator functions: phase in [0, 1) -> sin/cos(2 * pi * phase)
    Value* sin_norm(Value* phase01);
    Value* vsin_norm_f32x8(Value* phase01);
    Value* cos_norm(Value* phase01);
    Value* vcos_norm_f32x8(Value* phase01);

    // Saturation / waveshaping
    // Soft clip: branchless musical tanh(x)
    Value* soft_clip(Value* x);
    Value* vsoft_clip_f32x8(Value* x);

    // Hard clip to [-threshold, threshold]
    Value* hard_clip(Value* x, Value* threshold);
    Value* vhard_clip_f32x8(Value* x, Value* threshold);

    // Foldback distortion: folds signal back when exceeding [-threshold, threshold]
    Value* foldback(Value* x, Value* threshold);

    // ── Circular Delay Line & Fractional Interpolation ──────────────────────
    // Reads base_ptr[idx & mask]
    Value* load_circular_mask(Value* base_ptr, Value* idx, Value* mask);
    // Writes base_ptr[idx & mask] = val
    Instruction* store_circular_mask(Value* base_ptr, Value* idx, Value* mask, Value* val);

    // 4-point, 3rd-order cubic Hermite interpolation between p0, p1, p2, p3 with frac in [0, 1)
    Value* hermite_interp4(Value* p0, Value* p1, Value* p2, Value* p3, Value* frac);

    // Reads fractional delay tap with cubic Hermite interpolation from a circular ring buffer
    // of size (mask + 1), where head is the write position and delay_samples is a float delay.
    Value* load_delay_hermite(Value* ring_buf, Value* head, Value* delay_samples, Value* mask);
};

} // namespace brass::codegen
