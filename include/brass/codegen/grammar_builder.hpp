#pragma once

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/types.hpp>

#include <cstdint>
#include <string_view>

namespace brass::codegen {

// ── Function Pointer Signatures ──────────────────────────────────────────────
typedef void (*LogitMaskFn)(float* logits, const uint64_t* valid_mask, uint64_t vocab_size, float mask_val);
typedef int32_t (*DfaStepFn)(int32_t current_state, uint8_t byte_val);
typedef int32_t (*DfaTableStepFn)(const int32_t* transition_table, int32_t num_states, int32_t current_state, uint8_t byte_val);
typedef int32_t (*DfaScanStringFn)(const int32_t* transition_table, int32_t start_state, const uint8_t* bytes, uint64_t length);
typedef void (*DfaFilterTokensFn)(const int32_t* transition_table, int32_t current_state, const uint32_t* token_offsets, const uint8_t* token_bytes, uint64_t num_tokens, uint64_t* out_valid_mask);

// ─── GrammarBuilder ──────────────────────────────────────────────────────────
//
// Specialized MIR builder for grammar-constrained LLM decoding:
// 1. High-throughput vectorized logit masking (AVX2 / SIMD) using token validity bitmasks.
// 2. Direct DFA state transitions and unrolled byte string scanning.
// 3. Batched token accept/reject filtering across large vocabulary slices.
class GrammarBuilder : public KernelBuilder {
public:
    GrammarBuilder(Module& mod, Function* fn = nullptr)
        : KernelBuilder(mod, fn) {}
    explicit GrammarBuilder(Builder& b) noexcept
        : KernelBuilder(b) {}

    // ── Low-level Helpers ───────────────────────────────────────────────────
    // Loads an unsigned byte [0..255] as an i32 value from (base_ptr + byte_offset)
    // using aligned, page-safe 64-bit loads and bitwise extraction.
    Value* load_byte(Value* base_ptr, Value* byte_offset);

    // Evaluates a single DFA transition: transition_table[current_state * 256 + byte_val]
    Value* dfa_table_step(Value* transition_table, Value* current_state, Value* byte_val);

    // ── Vectorized / SIMD Logit Masking Kernel ──────────────────────────────
    // Emits a high-throughput loop over vocab_size in chunks of 8 floats (f32x8 / AVX2):
    // - Reads corresponding bits from valid_mask.
    // - If a bit is 0 (invalid token), writes mask_val (-INFINITY) to the logit.
    // - Skips stores entirely when all 8 tokens are valid (0xFF).
    // - Writes 8-wide vector broadcast when all 8 tokens are invalid (0x00).
    // - Handles scalar tail for vocab_size % 8 != 0.
    void emit_logit_mask_kernel(
        Function* fn,
        Value* logits_ptr,
        Value* mask_ptr,
        Value* vocab_size,
        Value* mask_val
    );

    // Convenience function factory for LogitMaskFn
    Function* build_logit_mask_function(std::string_view name = "logit_mask");

    // ── Fast DFA State Transition Jump Kernels ──────────────────────────────
    // Given a state transition table pointer and num_states:
    // Reads transition_table[current_state * 256 + byte_val].
    // Returns -1 if current_state is out of bounds or negative.
    void emit_dfa_table_step_kernel(
        Function* fn,
        Value* transition_table,
        Value* num_states,
        Value* current_state,
        Value* byte_val
    );

    // Emits a specialized DFA step jump kernel where the transition table pointer
    // is embedded directly into the generated code as an immediate constant.
    void emit_dfa_step_kernel(
        Function* fn,
        const int32_t* transition_table,
        Value* current_state,
        Value* byte_val,
        int32_t num_states = -1
    );

    Function* build_dfa_table_step_function(std::string_view name = "dfa_table_step");
    Function* build_dfa_step_function(
        const int32_t* transition_table,
        int32_t num_states = -1,
        std::string_view name = "dfa_step"
    );

    // Emits an unrolled byte string scanner:
    // Loops through the byte buffer, transitioning states.
    // If state becomes -1 (dead), immediately returns -1.
    void emit_dfa_scan_string_kernel(
        Function* fn,
        Value* transition_table,
        Value* start_state,
        Value* bytes,
        Value* length,
        unsigned unroll_factor = 4
    );

    Function* build_dfa_scan_string_function(
        std::string_view name = "dfa_scan_string",
        unsigned unroll_factor = 4
    );

    // ── Batched Token Accept Filter ─────────────────────────────────────────
    // Checks a batch of tokens against the current DFA state and writes 1s/0s to out_valid_mask.
    // token_offsets contains (num_tokens + 1) offsets into token_bytes.
    // Writes full 64-bit bitmasks to out_valid_mask.
    void emit_dfa_filter_tokens_kernel(
        Function* fn,
        Value* transition_table,
        Value* current_state,
        Value* token_offsets,
        Value* token_bytes,
        Value* num_tokens,
        Value* out_valid_mask
    );

    Function* build_dfa_filter_tokens_function(std::string_view name = "dfa_filter_tokens");
};

} // namespace brass::codegen
