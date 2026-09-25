// Builders for tagged values: the two bitcasts between a tagged value and its
// bits, the frame buffer of tagged words, and keep_alive, which holds a value
// (and so its referent) live to a point (docs/gc_contract.md).

#include <brass/mir/builder.hpp>

namespace brass {

namespace {

Value* build_unary(Builder& b, Opcode op, Type result_type, Value* val) {
    Instruction* inst = b.arena().make<Instruction>(op, result_type);
    inst->add_operand(val);
    Value* res = b.create_value(result_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b.insert(inst);
    return res;
}

} // namespace

Value* Builder::build_bitcast_i64_tagged(Value* val) {
    return build_unary(*this, Opcode::bitcast_i64_tagged, Type::i64(), val);
}

Value* Builder::build_bitcast_tagged_i64(Value* val) {
    return build_unary(*this, Opcode::bitcast_tagged_i64, Type::tagged(), val);
}

Instruction* Builder::build_keep_alive(Value* val) {
    Instruction* inst = arena().make<Instruction>(Opcode::keep_alive, Type::void_type());
    inst->add_operand(val);
    insert(inst);
    return inst;
}

Value* Builder::build_alloca_tagged(uint32_t words) {
    Value* res = build_alloca(words * 8, 8);
    res->defining_instruction()->set_memory_type(Type::tagged());
    return res;
}

} // namespace brass
