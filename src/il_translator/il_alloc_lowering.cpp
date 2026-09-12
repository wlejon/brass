#include <brass/il_translator/il_alloc_lowering.hpp>
#include <string>

namespace brass::il {

Value* AllocLoweringHelper::lower_create_object(Builder& b) {
    if (!enable_tlab_) {
        return b.build_call("bronze_create_object", Type::i64(), {});
    }
    if (model_ == Model::BronzeTLS) {
        return lower_create_object_bronze(b);
    }
    return lower_create_object_brass(b);
}

Value* AllocLoweringHelper::lower_create_array(Builder& b, Value* size_val, uint32_t param_count) {
    if (!enable_tlab_ || param_count > 8) {
        return b.build_call("bronze_create_array", Type::i64(), {size_val});
    }
    if (model_ == Model::BronzeTLS) {
        return lower_create_array_bronze(b, size_val, param_count);
    }
    return lower_create_array_brass(b, size_val, param_count);
}

Value* AllocLoweringHelper::lower_env_create(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count) {
    if (!enable_tlab_ || param_count > 32) {
        return b.build_call("bronze_env_create", Type::i64(), {parent_val, size_val});
    }
    if (model_ == Model::BronzeTLS) {
        return lower_env_create_bronze(b, parent_val, size_val, param_count);
    }
    return lower_env_create_brass(b, parent_val, size_val, param_count);
}

// -----------------------------------------------------------------------------
// Bronze TLS Allocation Model
// -----------------------------------------------------------------------------

Value* AllocLoweringHelper::lower_create_object_bronze(Builder& b) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_obj_bronze_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t PLAIN_OBJECT_BYTES = 56;

    Value* tls_addr = b.build_call("bronze_tls_block_addr", Type::i64(), {});
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, 24);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, 32);
    Value* plain_shape = b.build_load(Type::i64(), tls_addr, 40);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(PLAIN_OBJECT_BYTES)));
    Value* can_fit = b.build_ule(new_cursor, cur_limit);
    Value* has_shape = b.build_ne(plain_shape, b.build_iconst_i64(0));
    Value* can_alloc = b.build_and(can_fit, has_shape);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path: bump pointer and initialize plain object
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, 24, new_cursor);

    // HeapObjectHeader at cur_cursor (offset 0):
    // tag = 0xFFF1 (Tag::Object), flags = 0 (HeapKind::Plain), size = 56
    constexpr uint64_t HEADER_WORD = (static_cast<uint64_t>(PLAIN_OBJECT_BYTES) << 32) | 0xFFF1ULL;
    b.build_store(Type::i64(), cur_cursor, 0, b.build_iconst_i64(static_cast<int64_t>(HEADER_WORD)));

    // ObjectHeader:
    // Offset 8: shape = plain_shape
    b.build_store(Type::i64(), cur_cursor, 8, plain_shape);

    // Offset 16: overflow = Value::fromUndefined() (0xFFF6000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0xFFF6000000000000ULL));
    b.build_store(Type::i64(), cur_cursor, 16, undef_val);

    // Offsets 24, 32, 40, 48: inline_slots[0..3] = undefined
    for (int i = 0; i < 4; ++i) {
        b.build_store(Type::i64(), cur_cursor, 24 + i * 8, undef_val);
    }

    // NaN-box Tag::Object (0xFFF1ULL << 48)
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL));
    Value* masked_ptr = b.build_and(cur_cursor, ptr_mask);
    Value* obj_val = b.build_or(masked_ptr, b.build_iconst_i64(static_cast<int64_t>(0xFFF1000000000000ULL)));

    b.build_br(bb_merge, {obj_val});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_create_object", Type::i64(), {});
    b.build_br(bb_merge, {fallback_val});

    // Merge block
    b.position_at_end(bb_merge);
    return merge_val;
}

Value* AllocLoweringHelper::lower_create_array_bronze(Builder& b, Value* size_val, uint32_t param_count) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_arr_bronze_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t ARR_HDR_BYTES = 40; // BRONZE_ABI_ARRAY_HEADER_BYTES
    uint32_t cap = (param_count < 4) ? 4 : param_count;
    size_t elem_block_bytes = 8 + static_cast<size_t>(cap) * 8; // BRONZE_ABI_HDR_BYTES + cap * 8
    size_t total_needed = ARR_HDR_BYTES + elem_block_bytes;

    Value* tls_addr = b.build_call("bronze_tls_block_addr", Type::i64(), {});
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, 24);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, 32);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(total_needed)));
    Value* can_alloc = b.build_ule(new_cursor, cur_limit);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, 24, new_cursor);

    Value* arr_ptr = cur_cursor;
    Value* elem_ptr = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(ARR_HDR_BYTES)));

    // 1. ArrayHeader (at arr_ptr):
    // Word 0 (offset 0): size=40, flags=HeapKind::Array (1), tag=Tag::Object (0xFFF1)
    constexpr uint64_t ARR_W0 = (static_cast<uint64_t>(ARR_HDR_BYTES) << 32) | (1ULL << 16) | 0xFFF1ULL;
    b.build_store(Type::i64(), arr_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(ARR_W0)));

    // Word 1 (offset 8): length (lower 32) = param_count, capacity (upper 32) = cap
    uint64_t arr_w1 = static_cast<uint64_t>(param_count) | (static_cast<uint64_t>(cap) << 32);
    b.build_store(Type::i64(), arr_ptr, 8, b.build_iconst_i64(static_cast<int64_t>(arr_w1)));

    // Word 2 (offset 16): head_offset=0, reserved=0
    b.build_store(Type::i64(), arr_ptr, 16, b.build_iconst_i64(0));

    // Word 3 (offset 24): elements = Tag::Object boxed elem_ptr
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL));
    Value* masked_elem_ptr = b.build_and(elem_ptr, ptr_mask);
    Value* elem_val = b.build_or(masked_elem_ptr, b.build_iconst_i64(static_cast<int64_t>(0xFFF1000000000000ULL)));
    b.build_store(Type::i64(), arr_ptr, 24, elem_val);

    // Word 4 (offset 32): properties = Value::fromUndefined() (0xFFF6000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0xFFF6000000000000ULL));
    b.build_store(Type::i64(), arr_ptr, 32, undef_val);

    // 2. Elements Block (at elem_ptr):
    // Word 0 (offset 0): size=elem_block_bytes, flags=HeapKind::ValueBlock (19), tag=Tag::Object (0xFFF1)
    uint64_t elem_w0 = (static_cast<uint64_t>(elem_block_bytes) << 32) | (19ULL << 16) | 0xFFF1ULL;
    b.build_store(Type::i64(), elem_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(elem_w0)));

    // Offsets 8..8+cap*8: slots initialized to Value::fromHole() (0xFFF7000000000000ULL)
    Value* hole_val = b.build_iconst_i64(static_cast<int64_t>(0xFFF7000000000000ULL));
    for (uint32_t i = 0; i < cap; ++i) {
        b.build_store(Type::i64(), elem_ptr, static_cast<int32_t>(8 + i * 8), hole_val);
    }

    // Tagged array Value
    Value* masked_arr_ptr = b.build_and(arr_ptr, ptr_mask);
    Value* res_val = b.build_or(masked_arr_ptr, b.build_iconst_i64(static_cast<int64_t>(0xFFF1000000000000ULL)));
    b.build_br(bb_merge, {res_val});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_create_array", Type::i64(), {size_val});
    b.build_br(bb_merge, {fallback_val});

    // Merge block
    b.position_at_end(bb_merge);
    return merge_val;
}

Value* AllocLoweringHelper::lower_env_create_bronze(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_env_bronze_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    size_t total_size = 16 + static_cast<size_t>(param_count) * 8;

    Value* tls_addr = b.build_call("bronze_tls_block_addr", Type::i64(), {});
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, 24);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, 32);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(total_size)));
    Value* can_alloc = b.build_ule(new_cursor, cur_limit);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, 24, new_cursor);

    Value* env_ptr = cur_cursor;

    // Word 0 (offset 0): size=total_size, flags=HeapKind::Env (12), tag=Tag::Object (0xFFF1)
    uint64_t w0 = (static_cast<uint64_t>(total_size) << 32) | (12ULL << 16) | 0xFFF1ULL;
    b.build_store(Type::i64(), env_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));

    // Word 1 (offset 8): parent = parent_val
    b.build_store(Type::i64(), env_ptr, 8, parent_val);

    // Slots (offsets 16, 24, ...): Value::fromUndefined() (0xFFF6000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0xFFF6000000000000ULL));
    for (uint32_t i = 0; i < param_count; ++i) {
        b.build_store(Type::i64(), env_ptr, static_cast<int32_t>(16 + i * 8), undef_val);
    }

    // Tagged env Value
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL));
    Value* masked_env_ptr = b.build_and(env_ptr, ptr_mask);
    Value* res_val = b.build_or(masked_env_ptr, b.build_iconst_i64(static_cast<int64_t>(0xFFF1000000000000ULL)));
    b.build_br(bb_merge, {res_val});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_env_create", Type::i64(), {parent_val, size_val});
    b.build_br(bb_merge, {fallback_val});

    // Merge block
    b.position_at_end(bb_merge);
    return merge_val;
}

// -----------------------------------------------------------------------------
// Brass HostGC Allocation Model (Standalone Tests)
// -----------------------------------------------------------------------------

Value* AllocLoweringHelper::lower_create_object_brass(Builder& b) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_obj_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t PAYLOAD_SIZE = 104; // sizeof(DynamicObject)
    constexpr size_t TOTAL_SIZE = 24 + PAYLOAD_SIZE; // 128 bytes

    Value* top_ptr = b.build_func_addr("brass_tlab_top");
    Value* end_ptr = b.build_func_addr("brass_tlab_end");
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(TOTAL_SIZE)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path: inline bump pointer and initialize object
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* obj_addr = b.build_add(cur_top, b.build_iconst_i64(24));

    // HostGcHeader at cur_top
    // Word 0: size (104) | (type_tag (100) << 32)
    uint64_t w0 = static_cast<uint64_t>(PAYLOAD_SIZE) | (100ULL << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));
    // Word 1: pointer_mask = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12)
    constexpr uint64_t POINTER_MASK = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12);
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(static_cast<int64_t>(POINTER_MASK)));
    // Word 2: forwarding_address = 0
    b.build_store(Type::i64(), cur_top, 16, b.build_iconst_i64(0));

    // DynamicObject payload at obj_addr
    // Word 0 (offset 0): shape = root_shape
    Value* root_shape_addr = b.build_func_addr("brass_root_shape");
    Value* root_shape = b.build_load(Type::i64(), root_shape_addr, 0);
    b.build_store(Type::i64(), obj_addr, 0, root_shape);

    // Word 1 (offset 8): inline_capacity = 8, out_of_line_capacity = 0
    b.build_store(Type::i64(), obj_addr, 8, b.build_iconst_i64(8));

    // Word 2 (offset 16): out_of_line_slots = 0
    b.build_store(Type::i64(), obj_addr, 16, b.build_iconst_i64(0));

    // Words 3..10 (offsets 24..80): inline_slots[0..7] = HostValue::undefined_val() (0x7FFC000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0x7FFC000000000000ULL));
    for (int i = 0; i < 8; ++i) {
        b.build_store(Type::i64(), obj_addr, 24 + i * 8, undef_val);
    }

    // Word 11 (offset 88): element_count = 0, element_capacity = 0
    b.build_store(Type::i64(), obj_addr, 88, b.build_iconst_i64(0));

    // Word 12 (offset 96): elements = 0
    b.build_store(Type::i64(), obj_addr, 96, b.build_iconst_i64(0));

    b.build_br(bb_merge, {obj_addr});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_create_object", Type::i64(), {});
    b.build_br(bb_merge, {fallback_val});

    // Merge block
    b.position_at_end(bb_merge);
    return merge_val;
}

Value* AllocLoweringHelper::lower_create_array_brass(Builder& b, Value* size_val, uint32_t param_count) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_arr_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t OBJ_PAYLOAD = 104;
    constexpr size_t OBJ_TOTAL = 24 + OBJ_PAYLOAD; // 128 bytes
    constexpr size_t CAP = 8;
    constexpr size_t BUF_PAYLOAD = 8 + CAP * 8; // 72 bytes
    constexpr size_t BUF_TOTAL = 24 + BUF_PAYLOAD; // 96 bytes
    constexpr size_t TOTAL_SIZE = OBJ_TOTAL + BUF_TOTAL; // 224 bytes

    Value* top_ptr = b.build_func_addr("brass_tlab_top");
    Value* end_ptr = b.build_func_addr("brass_tlab_end");
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(TOTAL_SIZE)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* obj_addr = b.build_add(cur_top, b.build_iconst_i64(24));
    Value* buf_hdr_addr = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(OBJ_TOTAL)));
    Value* buf_addr = b.build_add(buf_hdr_addr, b.build_iconst_i64(24));

    // DynamicObject Header at cur_top
    uint64_t w0_obj = static_cast<uint64_t>(OBJ_PAYLOAD) | (100ULL << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0_obj)));
    constexpr uint64_t POINTER_MASK = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12);
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(static_cast<int64_t>(POINTER_MASK)));
    b.build_store(Type::i64(), cur_top, 16, b.build_iconst_i64(0));

    // DynamicObject Payload at obj_addr
    Value* root_shape_addr = b.build_func_addr("brass_root_shape");
    Value* root_shape = b.build_load(Type::i64(), root_shape_addr, 0);
    b.build_store(Type::i64(), obj_addr, 0, root_shape);
    b.build_store(Type::i64(), obj_addr, 8, b.build_iconst_i64(8));
    b.build_store(Type::i64(), obj_addr, 16, b.build_iconst_i64(0));

    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0x7FFC000000000000ULL));
    for (int i = 0; i < 8; ++i) {
        b.build_store(Type::i64(), obj_addr, 24 + i * 8, undef_val);
    }

    // Word 11 (offset 88): element_count = param_count, element_capacity = CAP
    uint64_t w11 = static_cast<uint64_t>(param_count) | (static_cast<uint64_t>(CAP) << 32);
    b.build_store(Type::i64(), obj_addr, 88, b.build_iconst_i64(static_cast<int64_t>(w11)));

    // Word 12 (offset 96): elements = buf_addr
    b.build_store(Type::i64(), obj_addr, 96, buf_addr);

    // DynamicObjectBuffer Header at buf_hdr_addr
    uint64_t w0_buf = static_cast<uint64_t>(BUF_PAYLOAD) | (102ULL << 32);
    b.build_store(Type::i64(), buf_hdr_addr, 0, b.build_iconst_i64(static_cast<int64_t>(w0_buf)));
    constexpr uint64_t BUF_MASK = 0x1FEULL; // ((1ULL << 8) - 1ULL) << 1
    b.build_store(Type::i64(), buf_hdr_addr, 8, b.build_iconst_i64(static_cast<int64_t>(BUF_MASK)));
    b.build_store(Type::i64(), buf_hdr_addr, 16, b.build_iconst_i64(0));

    // DynamicObjectBuffer Payload at buf_addr
    b.build_store(Type::i64(), buf_addr, 0, b.build_iconst_i64(static_cast<int64_t>(CAP)));
    for (size_t i = 0; i < CAP; ++i) {
        b.build_store(Type::i64(), buf_addr, static_cast<int32_t>(8 + i * 8), undef_val);
    }

    b.build_br(bb_merge, {obj_addr});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_create_array", Type::i64(), {size_val});
    b.build_br(bb_merge, {fallback_val});

    // Merge
    b.position_at_end(bb_merge);
    return merge_val;
}

Value* AllocLoweringHelper::lower_env_create_brass(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count) {
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_env_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    size_t alloc_size = (param_count <= 1) ? 24 : (16 + static_cast<size_t>(param_count) * 8);
    size_t total_size = 24 + alloc_size;

    Value* top_ptr = b.build_func_addr("brass_tlab_top");
    Value* end_ptr = b.build_func_addr("brass_tlab_end");
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(total_size)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* env_addr = b.build_add(cur_top, b.build_iconst_i64(24));

    // HostGcHeader at cur_top
    uint64_t w0 = static_cast<uint64_t>(alloc_size) | (1ULL << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(1));
    b.build_store(Type::i64(), cur_top, 16, b.build_iconst_i64(0));

    // BronzeEnv payload at env_addr
    b.build_store(Type::i64(), env_addr, 0, parent_val);
    b.build_store(Type::i32(), env_addr, 8, b.build_iconst_i32(static_cast<int32_t>(param_count)));
    b.build_store(Type::i32(), env_addr, 12, b.build_iconst_i32(0)); // padding

    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(0xFFF6000000000000ULL));
    for (uint32_t i = 0; i < param_count; ++i) {
        b.build_store(Type::i64(), env_addr, static_cast<int32_t>(16 + i * 8), undef_val);
    }

    b.build_br(bb_merge, {env_addr});

    // Fallback path
    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_env_create", Type::i64(), {parent_val, size_val});
    b.build_br(bb_merge, {fallback_val});

    // Merge block
    b.position_at_end(bb_merge);
    return merge_val;
}

} // namespace brass::il
