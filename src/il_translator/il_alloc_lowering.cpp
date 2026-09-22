#include <brass/il_translator/il_alloc_lowering.hpp>
#include <brass/mir/module.hpp>
#include <string>

namespace brass::il {

// The bronze thread-local block: through the pinned register when the module
// has one, otherwise the helper call.
static Value* bronze_tls_addr(Builder& b) {
    Module* mod = b.current_block()->parent()->parent();
    if (mod && mod->pinned_tls_register()) {
        return b.build_pinned_tls_read();
    }
    return b.build_call("bronze_tls_block_addr", Type::i64(), {});
}

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
// Bronze Heap & TLAB ABI Constants & Static Invariants
// Verified against bronze_abi.h and bronze_abi_tls.h conventions
// -----------------------------------------------------------------------------
namespace bronze_heap_abi {

// Thread-local allocation buffer offsets in bronze_tls_block (bronze_abi_tls.h)
constexpr int32_t TLS_ALLOC_CURSOR_OFF = 24; // BRONZE_TLS_ALLOC_CURSOR_OFF
constexpr int32_t TLS_ALLOC_LIMIT_OFF  = 32; // BRONZE_TLS_ALLOC_LIMIT_OFF
constexpr int32_t TLS_PLAIN_SHAPE_OFF  = 40; // BRONZE_TLS_PLAIN_SHAPE_OFF

// Allocation sizes (bronze_abi.h)
constexpr size_t PLAIN_OBJECT_BYTES   = 56; // BRONZE_ABI_PLAIN_OBJECT_BYTES
constexpr size_t ARRAY_HEADER_BYTES   = 40; // BRONZE_ABI_ARRAY_HEADER_BYTES
constexpr size_t ARRAY_MIN_CAPACITY   = 4;  // BRONZE_ABI_ARRAY_MIN_CAPACITY
constexpr size_t HDR_BYTES            = 8;  // BRONZE_ABI_HDR_BYTES

// HeapObjectHeader kind flags: bits 16..31 of header word (bronze_abi.h)
constexpr uint64_t OBJ_FLAGS_PLAIN       = 0;  // BRONZE_ABI_OBJ_FLAGS_PLAIN
constexpr uint64_t OBJ_FLAGS_ARRAY       = 1;  // BRONZE_ABI_OBJ_FLAGS_ARRAY
constexpr uint64_t OBJ_FLAGS_ENV         = 12; // BRONZE_ABI_OBJ_FLAGS_ENV
constexpr uint64_t OBJ_FLAGS_VALUE_BLOCK = 19; // BRONZE_ABI_OBJ_FLAGS_VALUE_BLOCK

// NaN-boxing tags & masks (bronze_abi.h)
constexpr uint32_t VALUE_TAG_SHIFT       = 48; // BRONZE_ABI_VALUE_TAG_SHIFT
constexpr uint64_t VALUE_PAYLOAD_MASK    = 0x0000FFFFFFFFFFFFULL; // BRONZE_ABI_VALUE_PAYLOAD_MASK
constexpr uint64_t TAG_OBJECT            = 0xFFF1ULL; // BRONZE_ABI_TAG_OBJECT
constexpr uint64_t TAG_UNDEFINED         = 0xFFF6ULL; // BRONZE_ABI_TAG_UNDEFINED
constexpr uint64_t TAG_HOLE              = 0xFFF7ULL; // BRONZE_ABI_TAG_HOLE

constexpr uint64_t VALUE_TAG_OBJECT      = TAG_OBJECT << VALUE_TAG_SHIFT;
constexpr uint64_t VALUE_TAG_UNDEFINED   = TAG_UNDEFINED << VALUE_TAG_SHIFT;
constexpr uint64_t VALUE_TAG_HOLE        = TAG_HOLE << VALUE_TAG_SHIFT;

// Compile-time static assertions verifying Bronze ABI invariants
static_assert(TLS_ALLOC_CURSOR_OFF == 24, "Bronze TLS alloc_cursor offset must be 24");
static_assert(TLS_ALLOC_LIMIT_OFF == 32, "Bronze TLS alloc_limit offset must be 32");
static_assert(TLS_PLAIN_SHAPE_OFF == 40, "Bronze TLS plain_shape offset must be 40");
static_assert(PLAIN_OBJECT_BYTES == 56, "Bronze plain object size must be 56 bytes");
static_assert(ARRAY_HEADER_BYTES == 40, "Bronze array header size must be 40 bytes");
static_assert(ARRAY_MIN_CAPACITY == 4, "Bronze array min capacity must be 4");
static_assert(HDR_BYTES == 8, "Bronze header bytes must be 8");
static_assert(OBJ_FLAGS_PLAIN == 0, "Bronze HeapKind::Plain must be 0");
static_assert(OBJ_FLAGS_ARRAY == 1, "Bronze HeapKind::Array must be 1");
static_assert(OBJ_FLAGS_ENV == 12, "Bronze HeapKind::Env must be 12");
static_assert(OBJ_FLAGS_VALUE_BLOCK == 19, "Bronze HeapKind::ValueBlock must be 19");
static_assert(VALUE_TAG_OBJECT == 0xFFF1000000000000ULL, "Bronze Object tag bits mismatch");
static_assert(VALUE_TAG_UNDEFINED == 0xFFF6000000000000ULL, "Bronze Undefined tag bits mismatch");
static_assert(VALUE_TAG_HOLE == 0xFFF7000000000000ULL, "Bronze Hole tag bits mismatch");
static_assert(VALUE_PAYLOAD_MASK == 0x0000FFFFFFFFFFFFULL, "Bronze payload mask mismatch");

// Helper to construct Bronze HeapObjectHeader word:
// size (bits 32..63) | flags (bits 16..31) | tag (bits 0..15)
constexpr uint64_t make_header_word(size_t size_bytes, uint64_t flags, uint64_t tag = TAG_OBJECT) {
    return (static_cast<uint64_t>(size_bytes) << 32) | ((flags & 0xFFFFULL) << 16) | (tag & 0xFFFFULL);
}

} // namespace bronze_heap_abi

// -----------------------------------------------------------------------------
// Brass HostGC ABI Constants (Standalone Tests)
// -----------------------------------------------------------------------------
namespace brass_hostgc_abi {

constexpr size_t HOSTGC_HEADER_SIZE = 24;
constexpr size_t DYNAMIC_OBJECT_PAYLOAD = 104;
constexpr size_t DYNAMIC_OBJECT_TOTAL = HOSTGC_HEADER_SIZE + DYNAMIC_OBJECT_PAYLOAD; // 128 bytes
constexpr uint64_t HOSTGC_TYPE_DYNAMIC_OBJECT = 100;
constexpr uint64_t HOSTGC_TYPE_DYNAMIC_OBJECT_BUFFER = 102;
constexpr uint64_t HOSTGC_TYPE_ENV = 1;
constexpr uint64_t DYNAMIC_OBJECT_POINTER_MASK = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12);
constexpr uint64_t BRASS_UNDEFINED_BITS = 0x7FFC000000000000ULL;

static_assert(DYNAMIC_OBJECT_TOTAL == 128, "DynamicObject total size must be 128 bytes");

} // namespace brass_hostgc_abi

// -----------------------------------------------------------------------------
// Bronze TLS Allocation Model
// -----------------------------------------------------------------------------

Value* AllocLoweringHelper::lower_create_object_bronze(Builder& b) {
    using namespace bronze_heap_abi;
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_obj_bronze_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    Value* tls_addr = bronze_tls_addr(b);
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_LIMIT_OFF);
    Value* plain_shape = b.build_load(Type::i64(), tls_addr, TLS_PLAIN_SHAPE_OFF);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(PLAIN_OBJECT_BYTES)));
    Value* can_fit = b.build_ule(new_cursor, cur_limit);
    Value* has_shape = b.build_ne(plain_shape, b.build_iconst_i64(0));
    Value* can_alloc = b.build_and(can_fit, has_shape);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path: bump pointer and initialize plain object
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF, new_cursor);

    // HeapObjectHeader at cur_cursor (offset 0):
    // tag = Tag::Object, flags = HeapKind::Plain, size = 56
    constexpr uint64_t HEADER_WORD = make_header_word(PLAIN_OBJECT_BYTES, OBJ_FLAGS_PLAIN, TAG_OBJECT);
    b.build_store(Type::i64(), cur_cursor, 0, b.build_iconst_i64(static_cast<int64_t>(HEADER_WORD)));

    // ObjectHeader:
    // Offset 8: shape = plain_shape
    b.build_store(Type::i64(), cur_cursor, 8, plain_shape);

    // Offset 16: overflow = Value::fromUndefined()
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_UNDEFINED));
    b.build_store(Type::i64(), cur_cursor, 16, undef_val);

    // Offsets 24, 32, 40, 48: inline_slots[0..3] = undefined
    for (int i = 0; i < 4; ++i) {
        b.build_store(Type::i64(), cur_cursor, 24 + i * 8, undef_val);
    }

    // NaN-box Tag::Object (0xFFF1ULL << 48)
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(VALUE_PAYLOAD_MASK));
    Value* masked_ptr = b.build_and(cur_cursor, ptr_mask);
    Value* obj_val = b.build_or(masked_ptr, b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_OBJECT)));

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
    using namespace bronze_heap_abi;
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_arr_bronze_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    uint32_t cap = (param_count < ARRAY_MIN_CAPACITY) ? static_cast<uint32_t>(ARRAY_MIN_CAPACITY) : param_count;
    size_t elem_block_bytes = HDR_BYTES + static_cast<size_t>(cap) * 8;
    size_t total_needed = ARRAY_HEADER_BYTES + elem_block_bytes;

    Value* tls_addr = bronze_tls_addr(b);
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_LIMIT_OFF);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(total_needed)));
    Value* can_alloc = b.build_ule(new_cursor, cur_limit);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF, new_cursor);

    Value* arr_ptr = cur_cursor;
    Value* elem_ptr = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(ARRAY_HEADER_BYTES)));

    // 1. ArrayHeader (at arr_ptr):
    // Word 0 (offset 0): size=40, flags=HeapKind::Array (1), tag=Tag::Object (0xFFF1)
    constexpr uint64_t ARR_W0 = make_header_word(ARRAY_HEADER_BYTES, OBJ_FLAGS_ARRAY, TAG_OBJECT);
    b.build_store(Type::i64(), arr_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(ARR_W0)));

    // Word 1 (offset 8): length (lower 32) = param_count, capacity (upper 32) = cap
    uint64_t arr_w1 = static_cast<uint64_t>(param_count) | (static_cast<uint64_t>(cap) << 32);
    b.build_store(Type::i64(), arr_ptr, 8, b.build_iconst_i64(static_cast<int64_t>(arr_w1)));

    // Word 2 (offset 16): head_offset=0, reserved=0
    b.build_store(Type::i64(), arr_ptr, 16, b.build_iconst_i64(0));

    // Word 3 (offset 24): elements = Tag::Object boxed elem_ptr
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(VALUE_PAYLOAD_MASK));
    Value* masked_elem_ptr = b.build_and(elem_ptr, ptr_mask);
    Value* elem_val = b.build_or(masked_elem_ptr, b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_OBJECT)));
    b.build_store(Type::i64(), arr_ptr, 24, elem_val);

    // Word 4 (offset 32): properties = Value::fromUndefined() (0xFFF6000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_UNDEFINED));
    b.build_store(Type::i64(), arr_ptr, 32, undef_val);

    // 2. Elements Block (at elem_ptr):
    // Word 0 (offset 0): size=elem_block_bytes, flags=HeapKind::ValueBlock (19), tag=Tag::Object (0xFFF1)
    uint64_t elem_w0 = make_header_word(elem_block_bytes, OBJ_FLAGS_VALUE_BLOCK, TAG_OBJECT);
    b.build_store(Type::i64(), elem_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(elem_w0)));

    // Offsets 8..8+cap*8: slots initialized to Value::fromHole() (0xFFF7000000000000ULL)
    Value* hole_val = b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_HOLE));
    for (uint32_t i = 0; i < cap; ++i) {
        b.build_store(Type::i64(), elem_ptr, static_cast<int32_t>(8 + i * 8), hole_val);
    }

    // Tagged array Value
    Value* masked_arr_ptr = b.build_and(arr_ptr, ptr_mask);
    Value* res_val = b.build_or(masked_arr_ptr, b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_OBJECT)));
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
    using namespace bronze_heap_abi;
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

    Value* tls_addr = bronze_tls_addr(b);
    Value* cur_cursor = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF);
    Value* cur_limit = b.build_load(Type::i64(), tls_addr, TLS_ALLOC_LIMIT_OFF);

    Value* new_cursor = b.build_add(cur_cursor, b.build_iconst_i64(static_cast<int64_t>(total_size)));
    Value* can_alloc = b.build_ule(new_cursor, cur_limit);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), tls_addr, TLS_ALLOC_CURSOR_OFF, new_cursor);

    Value* env_ptr = cur_cursor;

    // Word 0 (offset 0): size=total_size, flags=HeapKind::Env (12), tag=Tag::Object (0xFFF1)
    uint64_t w0 = make_header_word(total_size, OBJ_FLAGS_ENV, TAG_OBJECT);
    b.build_store(Type::i64(), env_ptr, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));

    // Word 1 (offset 8): parent = parent_val
    b.build_store(Type::i64(), env_ptr, 8, parent_val);

    // Slots (offsets 16, 24, ...): Value::fromUndefined() (0xFFF6000000000000ULL)
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_UNDEFINED));
    for (uint32_t i = 0; i < param_count; ++i) {
        b.build_store(Type::i64(), env_ptr, static_cast<int32_t>(16 + i * 8), undef_val);
    }

    // Tagged env Value
    Value* ptr_mask = b.build_iconst_i64(static_cast<int64_t>(VALUE_PAYLOAD_MASK));
    Value* masked_env_ptr = b.build_and(env_ptr, ptr_mask);
    Value* res_val = b.build_or(masked_env_ptr, b.build_iconst_i64(static_cast<int64_t>(VALUE_TAG_OBJECT)));
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
    using namespace brass_hostgc_abi;
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_obj_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t TOTAL_SIZE = DYNAMIC_OBJECT_TOTAL; // 128 bytes

    Value* top_ptr = b.build_call("brass_current_thread_tlab_top", Type::ptr());
    Value* end_ptr = b.build_call("brass_current_thread_tlab_end", Type::ptr());
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(TOTAL_SIZE)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path: inline bump pointer and initialize object
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* obj_addr = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(HOSTGC_HEADER_SIZE)));

    // HostGcHeader at cur_top
    // Word 0: size (104) | (type_tag (100) << 32)
    uint64_t w0 = static_cast<uint64_t>(DYNAMIC_OBJECT_PAYLOAD) | (HOSTGC_TYPE_DYNAMIC_OBJECT << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));
    // Word 1: pointer_mask = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12)
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(static_cast<int64_t>(DYNAMIC_OBJECT_POINTER_MASK)));
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
    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(BRASS_UNDEFINED_BITS));
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
    using namespace brass_hostgc_abi;
    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "tlab_arr_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    constexpr size_t OBJ_TOTAL = DYNAMIC_OBJECT_TOTAL; // 128 bytes
    constexpr size_t CAP = 8;
    constexpr size_t BUF_PAYLOAD = 8 + CAP * 8; // 72 bytes
    constexpr size_t BUF_TOTAL = HOSTGC_HEADER_SIZE + BUF_PAYLOAD; // 96 bytes
    constexpr size_t TOTAL_SIZE = OBJ_TOTAL + BUF_TOTAL; // 224 bytes

    Value* top_ptr = b.build_call("brass_current_thread_tlab_top", Type::ptr());
    Value* end_ptr = b.build_call("brass_current_thread_tlab_end", Type::ptr());
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(TOTAL_SIZE)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* obj_addr = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(HOSTGC_HEADER_SIZE)));
    Value* buf_hdr_addr = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(OBJ_TOTAL)));
    Value* buf_addr = b.build_add(buf_hdr_addr, b.build_iconst_i64(static_cast<int64_t>(HOSTGC_HEADER_SIZE)));

    // DynamicObject Header at cur_top
    uint64_t w0_obj = static_cast<uint64_t>(DYNAMIC_OBJECT_PAYLOAD) | (HOSTGC_TYPE_DYNAMIC_OBJECT << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0_obj)));
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(static_cast<int64_t>(DYNAMIC_OBJECT_POINTER_MASK)));
    b.build_store(Type::i64(), cur_top, 16, b.build_iconst_i64(0));

    // DynamicObject Payload at obj_addr
    Value* root_shape_addr = b.build_func_addr("brass_root_shape");
    Value* root_shape = b.build_load(Type::i64(), root_shape_addr, 0);
    b.build_store(Type::i64(), obj_addr, 0, root_shape);
    b.build_store(Type::i64(), obj_addr, 8, b.build_iconst_i64(8));
    b.build_store(Type::i64(), obj_addr, 16, b.build_iconst_i64(0));

    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(BRASS_UNDEFINED_BITS));
    for (int i = 0; i < 8; ++i) {
        b.build_store(Type::i64(), obj_addr, 24 + i * 8, undef_val);
    }

    // Word 11 (offset 88): element_count = param_count, element_capacity = CAP
    uint64_t w11 = static_cast<uint64_t>(param_count) | (static_cast<uint64_t>(CAP) << 32);
    b.build_store(Type::i64(), obj_addr, 88, b.build_iconst_i64(static_cast<int64_t>(w11)));

    // Word 12 (offset 96): elements = buf_addr
    b.build_store(Type::i64(), obj_addr, 96, buf_addr);

    // DynamicObjectBuffer Header at buf_hdr_addr
    uint64_t w0_buf = static_cast<uint64_t>(BUF_PAYLOAD) | (HOSTGC_TYPE_DYNAMIC_OBJECT_BUFFER << 32);
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
    using namespace brass_hostgc_abi;
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
    size_t total_size = HOSTGC_HEADER_SIZE + alloc_size;

    Value* top_ptr = b.build_call("brass_current_thread_tlab_top", Type::ptr());
    Value* end_ptr = b.build_call("brass_current_thread_tlab_end", Type::ptr());
    Value* cur_top = b.build_load(Type::i64(), top_ptr, 0);
    Value* cur_end = b.build_load(Type::i64(), end_ptr, 0);

    Value* new_top = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(total_size)));
    Value* can_alloc = b.build_ule(new_top, cur_end);
    b.build_br_if(can_alloc, bb_fast, bb_fallback);

    // Fast path
    b.position_at_end(bb_fast);
    b.build_store(Type::i64(), top_ptr, 0, new_top);

    Value* env_addr = b.build_add(cur_top, b.build_iconst_i64(static_cast<int64_t>(HOSTGC_HEADER_SIZE)));

    // HostGcHeader at cur_top
    uint64_t w0 = static_cast<uint64_t>(alloc_size) | (HOSTGC_TYPE_ENV << 32);
    b.build_store(Type::i64(), cur_top, 0, b.build_iconst_i64(static_cast<int64_t>(w0)));
    uint64_t ptr_mask = 1ULL;
    for (uint32_t i = 0; i < param_count; ++i) {
        if (2 + i < 64) {
            ptr_mask |= (1ULL << (2 + i));
        }
    }
    b.build_store(Type::i64(), cur_top, 8, b.build_iconst_i64(static_cast<int64_t>(ptr_mask)));
    b.build_store(Type::i64(), cur_top, 16, b.build_iconst_i64(0));

    // BronzeEnv payload at env_addr
    b.build_store(Type::i64(), env_addr, 0, parent_val);
    b.build_store(Type::i32(), env_addr, 8, b.build_iconst_i32(static_cast<int32_t>(param_count)));
    b.build_store(Type::i32(), env_addr, 12, b.build_iconst_i32(0)); // padding

    Value* undef_val = b.build_iconst_i64(static_cast<int64_t>(bronze_heap_abi::VALUE_TAG_UNDEFINED));
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
