#include <brass/codegen/grammar_builder.hpp>

namespace brass::codegen {

// ── Low-level Helpers ────────────────────────────────────────────────────────

Value* GrammarBuilder::load_byte(Value* base_ptr, Value* byte_offset) {
    Value* offset_i64 = byte_offset;
    if (byte_offset->type().is_i32()) {
        offset_i64 = builder().build_zext_i64(byte_offset);
    }
    // Align offset to 8-byte boundary (on i64 integers)
    Value* mask_align = const_i64(~int64_t(7));
    Value* aligned_offset = builder().build_and(offset_i64, mask_align);
    Value* byte_idx_in_word = builder().build_and(offset_i64, const_i64(7));
    Value* shift = builder().build_shl(byte_idx_in_word, const_i64(3)); // byte_idx * 8
    Value* word_ptr = builder().build_add(base_ptr, aligned_offset);
    Value* word = load_i64(word_ptr, 0);
    Value* shifted = builder().build_lshr(word, shift);
    Value* byte_val = builder().build_and(shifted, const_i64(0xFF));
    return builder().build_trunc_i32(byte_val);
}

Value* GrammarBuilder::dfa_table_step(Value* transition_table, Value* current_state, Value* byte_val) {
    Value* byte_clean = byte_val;
    if (byte_val->type().is_i64()) {
        byte_clean = builder().build_trunc_i32(byte_val);
    }
    Value* byte_u8 = builder().build_and(byte_clean, const_i32(0xFF));
    Value* state_256 = builder().build_shl(current_state, const_i32(8));
    Value* index_i32 = builder().build_add(state_256, byte_u8);
    Value* index_i64 = builder().build_sext_i64(index_i32);
    return load_i32_indexed(transition_table, index_i64, 4, 0);
}

// ── Vectorized / SIMD Logit Masking Kernel ───────────────────────────────────

void GrammarBuilder::emit_logit_mask_kernel(
    Function* fn,
    Value* logits_ptr,
    Value* mask_ptr,
    Value* vocab_size,
    Value* mask_val
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }
    Value* zero = const_i64(0);
    Value* eight = const_i64(8);
    Value* rem = builder().build_umod(vocab_size, eight);
    Value* vec_end = builder().build_sub(vocab_size, rem);

    // Vector loop: process in chunks of 8 floats (f32x8 / AVX2)
    for_range(zero, vec_end, eight, [&](Value* i) {
        Value* word_idx = builder().build_lshr(i, const_i64(6));
        Value* bit_offset = builder().build_and(i, const_i64(63));
        Value* word = load_i64_indexed(mask_ptr, word_idx, 8, 0);
        Value* shifted = builder().build_lshr(word, bit_offset);
        Value* mask_byte = builder().build_and(shifted, const_i64(0xFF));

        BasicBlock* check_zero_bb = builder().create_block("check_zero");
        BasicBlock* store_vec_bb = builder().create_block("store_vec");
        BasicBlock* mixed_bb = builder().create_block("mixed_lanes");
        BasicBlock* done_bb = builder().create_block("chunk_done");

        // If mask_byte == 255: all 8 tokens valid, no writes needed
        Value* is_all_valid = builder().build_eq(mask_byte, const_i64(255));
        builder().build_br_if(is_all_valid, done_bb, check_zero_bb);

        // Check if all 8 tokens are invalid (0x00)
        builder().current_function()->append_block(check_zero_bb);
        builder().position_at_end(check_zero_bb);
        Value* is_all_invalid = builder().build_eq(mask_byte, zero);
        builder().build_br_if(is_all_invalid, store_vec_bb, mixed_bb);

        // Fast path: all 8 tokens masked -> vectorized store of mask_val
        builder().current_function()->append_block(store_vec_bb);
        builder().position_at_end(store_vec_bb);
        Value* v_mask = builder().build_vbroadcast(Type::f32x8(), mask_val);
        Value* byte_offset = builder().build_shl(i, const_i64(2)); // i * 4 bytes
        Value* out_ptr = builder().build_add(logits_ptr, byte_offset);
        builder().build_vstore(Type::f32x8(), out_ptr, 0, v_mask);
        builder().build_br(done_bb);

        // Mixed valid/invalid tokens: check individual bits
        builder().current_function()->append_block(mixed_bb);
        builder().position_at_end(mixed_bb);
        for (int lane = 0; lane < 8; ++lane) {
            Value* bit = builder().build_and(builder().build_lshr(mask_byte, const_i64(lane)), const_i64(1));
            Value* is_zero = builder().build_eq(bit, zero);
            if_then(is_zero, [&]() {
                Value* idx = builder().build_add(i, const_i64(lane));
                store_f32_indexed(logits_ptr, idx, mask_val, 4, 0);
            });
        }
        builder().build_br(done_bb);

        builder().current_function()->append_block(done_bb);
        builder().position_at_end(done_bb);
    });

    // Scalar tail: process remainder (vocab_size % 8 != 0)
    for_range(vec_end, vocab_size, const_i64(1), [&](Value* i) {
        Value* word_idx = builder().build_lshr(i, const_i64(6));
        Value* bit_offset = builder().build_and(i, const_i64(63));
        Value* word = load_i64_indexed(mask_ptr, word_idx, 8, 0);
        Value* bit = builder().build_and(builder().build_lshr(word, bit_offset), const_i64(1));
        Value* is_zero = builder().build_eq(bit, zero);
        if_then(is_zero, [&]() {
            store_f32_indexed(logits_ptr, i, mask_val, 4, 0);
        });
    });
}

Function* GrammarBuilder::build_logit_mask_function(std::string_view name) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::void_type(), {
        Type::ptr(), // logits (float*)
        Type::ptr(), // valid_mask (const uint64_t*)
        Type::i64(), // vocab_size (uint64_t)
        Type::f32()  // mask_val (float)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* logits = builder().add_block_param(entry, Type::ptr());
    Value* mask = builder().add_block_param(entry, Type::ptr());
    Value* vocab_size = builder().add_block_param(entry, Type::i64());
    Value* mask_val = builder().add_block_param(entry, Type::f32());

    emit_logit_mask_kernel(fn, logits, mask, vocab_size, mask_val);
    builder().build_ret_void();
    return fn;
}

// ── Fast DFA State Transition Jump Kernels ──────────────────────────────────

void GrammarBuilder::emit_dfa_table_step_kernel(
    Function* fn,
    Value* transition_table,
    Value* num_states,
    Value* current_state,
    Value* byte_val
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    BasicBlock* valid_bb = builder().create_block("state_valid");
    BasicBlock* invalid_bb = builder().create_block("state_invalid");

    // Unsigned comparison checks both current_state < 0 and current_state >= num_states
    Value* is_oob = builder().build_uge(current_state, num_states);
    builder().build_br_if(is_oob, invalid_bb, valid_bb);

    builder().current_function()->append_block(invalid_bb);
    builder().position_at_end(invalid_bb);
    builder().build_ret(const_i32(-1));

    builder().current_function()->append_block(valid_bb);
    builder().position_at_end(valid_bb);
    Value* next_state = dfa_table_step(transition_table, current_state, byte_val);
    builder().build_ret(next_state);
}

void GrammarBuilder::emit_dfa_step_kernel(
    Function* fn,
    const int32_t* transition_table,
    Value* current_state,
    Value* byte_val,
    int32_t num_states
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    BasicBlock* valid_bb = builder().create_block("state_valid");
    BasicBlock* invalid_bb = builder().create_block("state_invalid");

    if (num_states > 0) {
        Value* is_oob = builder().build_uge(current_state, const_i32(num_states));
        builder().build_br_if(is_oob, invalid_bb, valid_bb);
    } else {
        Value* is_neg = builder().build_slt(current_state, const_i32(0));
        builder().build_br_if(is_neg, invalid_bb, valid_bb);
    }

    builder().current_function()->append_block(invalid_bb);
    builder().position_at_end(invalid_bb);
    builder().build_ret(const_i32(-1));

    builder().current_function()->append_block(valid_bb);
    builder().position_at_end(valid_bb);
    Value* table_ptr = const_i64(reinterpret_cast<uintptr_t>(transition_table));
    Value* next_state = dfa_table_step(table_ptr, current_state, byte_val);
    builder().build_ret(next_state);
}

Function* GrammarBuilder::build_dfa_table_step_function(std::string_view name) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::i32(), {
        Type::ptr(), // transition_table (const int32_t*)
        Type::i32(), // num_states (int32_t)
        Type::i32(), // current_state (int32_t)
        Type::i32()  // byte_val (uint8_t passed as i32)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* table = builder().add_block_param(entry, Type::ptr());
    Value* num_states = builder().add_block_param(entry, Type::i32());
    Value* current_state = builder().add_block_param(entry, Type::i32());
    Value* byte_val = builder().add_block_param(entry, Type::i32());

    emit_dfa_table_step_kernel(fn, table, num_states, current_state, byte_val);
    return fn;
}

Function* GrammarBuilder::build_dfa_step_function(
    const int32_t* transition_table,
    int32_t num_states,
    std::string_view name
) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::i32(), {
        Type::i32(), // current_state (int32_t)
        Type::i32()  // byte_val (uint8_t passed as i32)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* current_state = builder().add_block_param(entry, Type::i32());
    Value* byte_val = builder().add_block_param(entry, Type::i32());

    emit_dfa_step_kernel(fn, transition_table, current_state, byte_val, num_states);
    return fn;
}

void GrammarBuilder::emit_dfa_scan_string_kernel(
    Function* fn,
    Value* transition_table,
    Value* start_state,
    Value* bytes,
    Value* length,
    unsigned unroll_factor
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    BasicBlock* early_dead_bb = builder().create_block("early_dead");
    BasicBlock* check_len_bb = builder().create_block("check_len");
    BasicBlock* return_start_bb = builder().create_block("return_start");
    BasicBlock* loop_entry_bb = builder().create_block("loop_entry");
    BasicBlock* done_bb = builder().create_block("scan_done");

    // In current block (entry): check start_state < 0
    Value* is_dead = builder().build_slt(start_state, const_i32(0));
    builder().build_br_if(is_dead, early_dead_bb, check_len_bb);

    // Empty string check
    builder().current_function()->append_block(check_len_bb);
    builder().position_at_end(check_len_bb);
    Value* is_empty = builder().build_eq(length, const_i64(0));
    builder().build_br_if(is_empty, return_start_bb, loop_entry_bb);

    builder().current_function()->append_block(return_start_bb);
    builder().position_at_end(return_start_bb);
    builder().build_ret(start_state);

    builder().current_function()->append_block(early_dead_bb);
    builder().position_at_end(early_dead_bb);
    builder().build_ret(const_i32(-1));

    builder().current_function()->append_block(loop_entry_bb);
    builder().position_at_end(loop_entry_bb);

    unsigned u = unroll_factor < 1 ? 1 : unroll_factor;
    Value* u_val = const_i64(static_cast<int64_t>(u));
    Value* rem = builder().build_umod(length, u_val);
    Value* unroll_limit = builder().build_sub(length, rem);

    BasicBlock* unroll_head = builder().create_block("unroll_head");
    BasicBlock* rem_head = builder().create_block("rem_head");

    // Enter unroll loop
    builder().build_br(unroll_head, {const_i64(0), start_state});

    // Unrolled loop header
    builder().current_function()->append_block(unroll_head);
    builder().position_at_end(unroll_head);
    Value* cur_i = builder().add_block_param(unroll_head, Type::i64());
    Value* cur_state = builder().add_block_param(unroll_head, Type::i32());

    Value* in_unroll = builder().build_slt(cur_i, unroll_limit);
    BasicBlock* unroll_step0 = builder().create_block("unroll_step0");
    builder().build_br_if(in_unroll, unroll_step0, {}, rem_head, {cur_i, cur_state});

    // Build unrolled steps
    builder().current_function()->append_block(unroll_step0);
    BasicBlock* prev_step_bb = unroll_step0;
    Value* step_state = cur_state;

    for (unsigned k = 0; k < u; ++k) {
        builder().position_at_end(prev_step_bb);
        Value* idx = builder().build_add(cur_i, const_i64(static_cast<int64_t>(k)));
        Value* b = load_byte(bytes, idx);
        Value* next_s = dfa_table_step(transition_table, step_state, b);
        Value* dead_k = builder().build_slt(next_s, const_i32(0));

        if (k + 1 < u) {
            BasicBlock* next_step_bb = builder().create_block("unroll_step" + std::to_string(k + 1));
            builder().build_br_if(dead_k, early_dead_bb, next_step_bb);
            builder().current_function()->append_block(next_step_bb);
            prev_step_bb = next_step_bb;
            step_state = next_s;
        } else {
            // End of unrolled iteration
            BasicBlock* unroll_back_bb = builder().create_block("unroll_back");
            builder().build_br_if(dead_k, early_dead_bb, unroll_back_bb);

            builder().current_function()->append_block(unroll_back_bb);
            builder().position_at_end(unroll_back_bb);
            Value* next_i = builder().build_add(cur_i, u_val);
            builder().build_br(unroll_head, {next_i, next_s});
        }
    }

    // Remainder loop header
    builder().current_function()->append_block(rem_head);
    builder().position_at_end(rem_head);
    Value* rem_i = builder().add_block_param(rem_head, Type::i64());
    Value* rem_state = builder().add_block_param(rem_head, Type::i32());

    Value* in_rem = builder().build_slt(rem_i, length);
    BasicBlock* rem_body = builder().create_block("rem_body");
    builder().build_br_if(in_rem, rem_body, {}, done_bb, {rem_state});

    // Remainder body
    builder().current_function()->append_block(rem_body);
    builder().position_at_end(rem_body);
    Value* rb = load_byte(bytes, rem_i);
    Value* rnext_s = dfa_table_step(transition_table, rem_state, rb);
    Value* rdead = builder().build_slt(rnext_s, const_i32(0));

    BasicBlock* rem_back = builder().create_block("rem_back");
    builder().build_br_if(rdead, early_dead_bb, rem_back);

    builder().current_function()->append_block(rem_back);
    builder().position_at_end(rem_back);
    Value* rnext_i = builder().build_add(rem_i, const_i64(1));
    builder().build_br(rem_head, {rnext_i, rnext_s});

    // Done
    builder().current_function()->append_block(done_bb);
    builder().position_at_end(done_bb);
    Value* final_state = builder().add_block_param(done_bb, Type::i32());
    builder().build_ret(final_state);
}

Function* GrammarBuilder::build_dfa_scan_string_function(
    std::string_view name,
    unsigned unroll_factor
) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::i32(), {
        Type::ptr(), // transition_table (const int32_t*)
        Type::i32(), // start_state (int32_t)
        Type::ptr(), // bytes (const uint8_t*)
        Type::i64()  // length (uint64_t)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* table = builder().add_block_param(entry, Type::ptr());
    Value* start_state = builder().add_block_param(entry, Type::i32());
    Value* bytes = builder().add_block_param(entry, Type::ptr());
    Value* length = builder().add_block_param(entry, Type::i64());

    emit_dfa_scan_string_kernel(fn, table, start_state, bytes, length, unroll_factor);
    return fn;
}

// ── Batched Token Accept Filter ──────────────────────────────────────────────

void GrammarBuilder::emit_dfa_filter_tokens_kernel(
    Function* fn,
    Value* transition_table,
    Value* current_state,
    Value* token_offsets,
    Value* token_bytes,
    Value* num_tokens,
    Value* out_valid_mask
) {
    if (fn && builder().current_function() != fn) {
        builder().set_function(fn);
    }

    Value* zero = const_i64(0);
    // 1. Zero out out_valid_mask for all (num_tokens + 63) / 64 words
    Value* words_count = builder().build_lshr(builder().build_add(num_tokens, const_i64(63)), const_i64(6));
    for_range(zero, words_count, const_i64(1), [&](Value* w) {
        store_i64_indexed(out_valid_mask, w, zero, 8, 0);
    });

    // 2. If current_state < 0, all tokens rejected, return early
    Value* is_state_valid = builder().build_sge(current_state, const_i32(0));
    if_then(is_state_valid, [&]() {
        // Iterate through each token in the batch
        for_range(zero, num_tokens, const_i64(1), [&](Value* tok) {
            Value* start_i32 = load_i32_indexed(token_offsets, tok, 4, 0);
            Value* next_tok = builder().build_add(tok, const_i64(1));
            Value* end_i32 = load_i32_indexed(token_offsets, next_tok, 4, 0);

            Value* start_off = builder().build_zext_i64(start_i32);
            Value* end_off = builder().build_zext_i64(end_i32);
            Value* tok_len = builder().build_sub(end_off, start_off);

            // Scan token bytes
            Value* final_state = for_range_reduce(zero, tok_len, const_i64(1), current_state, [&](Value* j, Value* s) {
                Value* is_dead = builder().build_slt(s, const_i32(0));
                BasicBlock* next_bb = builder().create_block("step_bb");
                BasicBlock* skip_bb = builder().create_block("skip_bb");
                BasicBlock* join_bb = builder().create_block("join_bb");

                builder().build_br_if(is_dead, skip_bb, next_bb);

                builder().current_function()->append_block(skip_bb);
                builder().position_at_end(skip_bb);
                builder().build_br(join_bb, {s});

                builder().current_function()->append_block(next_bb);
                builder().position_at_end(next_bb);
                Value* byte_idx = builder().build_add(start_off, j);
                Value* b = load_byte(token_bytes, byte_idx);
                Value* s_next = dfa_table_step(transition_table, s, b);
                builder().build_br(join_bb, {s_next});

                builder().current_function()->append_block(join_bb);
                builder().position_at_end(join_bb);
                Value* res_state = builder().add_block_param(join_bb, Type::i32());
                return res_state;
            });

            // If token accepted (final_state >= 0), set corresponding bit in out_valid_mask
            Value* is_accepted = builder().build_sge(final_state, const_i32(0));
            if_then(is_accepted, [&]() {
                Value* w_idx = builder().build_lshr(tok, const_i64(6));
                Value* bit_pos = builder().build_and(tok, const_i64(63));
                Value* bit_mask = builder().build_shl(const_i64(1), bit_pos);
                Value* cur_word = load_i64_indexed(out_valid_mask, w_idx, 8, 0);
                Value* new_word = builder().build_or(cur_word, bit_mask);
                store_i64_indexed(out_valid_mask, w_idx, new_word, 8, 0);
            });
        });
    });
}

Function* GrammarBuilder::build_dfa_filter_tokens_function(std::string_view name) {
    Module* mod = builder().current_module();
    if (!mod) return nullptr;
    Function* fn = mod->create_function(name, Type::void_type(), {
        Type::ptr(), // transition_table (const int32_t*)
        Type::i32(), // current_state (int32_t)
        Type::ptr(), // token_offsets (const uint32_t*)
        Type::ptr(), // token_bytes (const uint8_t*)
        Type::i64(), // num_tokens (uint64_t)
        Type::ptr()  // out_valid_mask (uint64_t*)
    });
    builder().set_function(fn);
    BasicBlock* entry = builder().append_block("entry");
    builder().position_at_end(entry);

    Value* table = builder().add_block_param(entry, Type::ptr());
    Value* current_state = builder().add_block_param(entry, Type::i32());
    Value* offsets = builder().add_block_param(entry, Type::ptr());
    Value* bytes = builder().add_block_param(entry, Type::ptr());
    Value* num_tokens = builder().add_block_param(entry, Type::i64());
    Value* out_mask = builder().add_block_param(entry, Type::ptr());

    emit_dfa_filter_tokens_kernel(fn, table, current_state, offsets, bytes, num_tokens, out_mask);
    builder().build_ret_void();
    return fn;
}

} // namespace brass::codegen
