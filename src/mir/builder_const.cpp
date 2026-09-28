// Integer constants, and the per-block reuse a front end can turn on
// (Builder::set_const_reuse).
//
// A function-wide pool (each constant placed once in a block dominating the
// body) was measured on a whole-program workload and left out: it cut the
// MIR a further fifth but made lowering slower and the code no faster.

#include <brass/mir/builder.hpp>

namespace brass {

namespace {

// `inst` is still the constant `val` of kind `op` in `block`: every check a
// pass that edited the block since could have falsified.
Value* still_const(const Instruction* inst, Opcode op, int64_t val, const BasicBlock* block) {
    if (inst->parent() != block || inst->opcode() != op || inst->imm_i64() != val) return nullptr;
    Value* res = inst->result();
    return res != nullptr && res->defining_instruction() == inst ? res : nullptr;
}

}  // namespace

// A constant this builder appended to the current block since it was last
// positioned in another one, and which is still there.
Value* Builder::reuse_const(Opcode op, int64_t val) const noexcept {
    for (uint32_t i = 0; i < const_count_; ++i) {
        if (Value* res = still_const(const_cache_[i], op, val, block_)) return res;
    }
    return nullptr;
}

void Builder::remember_const(Instruction* inst) noexcept {
    if (const_count_ < kConstCache) {
        const_cache_[const_count_++] = inst;
        return;
    }
    const_cache_[const_next_] = inst;
    const_next_ = (const_next_ + 1) % kConstCache;
}

Value* Builder::build_iconst_i32(int32_t val) {
    // Keyed as the i64 the instruction keeps (set_imm_i32 widens it).
    const bool reuse = const_reuse_ && insert_before_ == nullptr && block_ != nullptr;
    if (reuse) {
        if (Value* hit = reuse_const(Opcode::iconst_i32, static_cast<int64_t>(val))) return hit;
    }
    Instruction* inst = get_arena().make<Instruction>(Opcode::iconst_i32, Type::i32());
    inst->set_imm_i32(val);
    Value* res = create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    if (reuse) remember_const(inst);
    return res;
}

Value* Builder::build_iconst_i64(int64_t val) {
    const bool reuse = const_reuse_ && insert_before_ == nullptr && block_ != nullptr;
    if (reuse) {
        if (Value* hit = reuse_const(Opcode::iconst_i64, val)) return hit;
    }
    Instruction* inst = get_arena().make<Instruction>(Opcode::iconst_i64, Type::i64());
    inst->set_imm_i64(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    if (reuse) remember_const(inst);
    return res;
}

}  // namespace brass
