#include <brass/il_translator/il_alloc_lowering.hpp>
#include <string>

namespace brass::il {

Value* AllocLoweringHelper::lower_create_object(Builder& b) {
    if (!enable_tlab_) {
        return b.build_call("bronze_create_object", Type::i64(), {});
    }

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

Value* AllocLoweringHelper::lower_create_array(Builder& b, Value* size_val, uint32_t param_count) {
    if (!enable_tlab_ || param_count > 8) {
        return b.build_call("bronze_create_array", Type::i64(), {size_val});
    }

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

Value* AllocLoweringHelper::lower_env_create(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count) {
    if (!enable_tlab_ || param_count > 32) {
        return b.build_call("bronze_env_create", Type::i64(), {parent_val, size_val});
    }

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
