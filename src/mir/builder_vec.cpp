#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>

namespace brass {

static Value* build_vec_bin_op(Builder* b, Arena& arena, Opcode op, Value* lhs, Value* rhs) {
    Type res_type = lhs ? lhs->type() : (rhs ? rhs->type() : Type::f32x4());
    Instruction* inst = arena.make<Instruction>(op, res_type);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    Value* res = b->create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

static Value* build_vec_un_op(Builder* b, Arena& arena, Opcode op, Value* val) {
    Type res_type = val ? val->type() : Type::f32x4();
    Instruction* inst = arena.make<Instruction>(op, res_type);
    inst->add_operand(val);
    Value* res = b->create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

Value* Builder::build_fadd(Value* lhs, Value* rhs) {
    Type res_type = lhs ? lhs->type() : (rhs ? rhs->type() : Type::f32());
    Instruction* inst = get_arena().make<Instruction>(Opcode::add, res_type);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_vadd(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vadd, lhs, rhs); }
Value* Builder::build_vsub(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vsub, lhs, rhs); }
Value* Builder::build_vmul(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vmul, lhs, rhs); }
Value* Builder::build_vfma(Value* a, Value* b, Value* c) {
    Type res_type = a ? a->type() : (b ? b->type() : (c ? c->type() : Type::f32x4()));
    Instruction* inst = get_arena().make<Instruction>(Opcode::vfma, res_type);
    inst->add_operand(a);
    inst->add_operand(b);
    inst->add_operand(c);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}
Value* Builder::build_vdiv(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vdiv, lhs, rhs); }
Value* Builder::build_vneg(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vneg, val); }
Value* Builder::build_vmin(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vmin, lhs, rhs); }
Value* Builder::build_vmax(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vmax, lhs, rhs); }
Value* Builder::build_vsqrt(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vsqrt, val); }
Value* Builder::build_vand(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vand, lhs, rhs); }
Value* Builder::build_vor(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vor, lhs, rhs); }
Value* Builder::build_vxor(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vxor, lhs, rhs); }
Value* Builder::build_vnot(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vnot, val); }

Value* Builder::build_vload(Type type, Value* base) {
    return build_vload(type, base, 0);
}

Value* Builder::build_vload(Type type, Value* base, int32_t offset) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::vload, type);
    inst->add_operand(base);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    Value* res = create_value(type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Instruction* Builder::build_vstore(Type type, Value* base, Value* val) {
    return build_vstore(type, base, 0, val);
}

Instruction* Builder::build_vstore(Type type, Value* base, int32_t offset, Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::vstore, Type::void_type());
    inst->add_operand(base);
    inst->add_operand(val);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    insert(inst);
    return inst;
}

Value* Builder::build_vbroadcast(Type vec_type, Value* scalar_val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::vbroadcast, vec_type);
    inst->add_operand(scalar_val);
    Value* res = create_value(vec_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_vextract_lane(Value* vec_val, uint32_t lane) {
    Type res_type = vec_val ? vec_val->type().element_type() : Type::f32();
    Instruction* inst = get_arena().make<Instruction>(Opcode::vextract_lane, res_type);
    inst->add_operand(vec_val);
    inst->set_lane(lane);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_vinsert_lane(Value* vec_val, Value* scalar_val, uint32_t lane) {
    Type res_type = vec_val ? vec_val->type() : Type::f32x4();
    Instruction* inst = get_arena().make<Instruction>(Opcode::vinsert_lane, res_type);
    inst->add_operand(vec_val);
    inst->add_operand(scalar_val);
    inst->set_lane(lane);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_vshuffle(Value* v1, Value* v2, uint32_t mask) {
    Type res_type = v1 ? v1->type() : (v2 ? v2->type() : Type::f32x4());
    Instruction* inst = get_arena().make<Instruction>(Opcode::vshuffle, res_type);
    inst->add_operand(v1);
    inst->add_operand(v2);
    inst->set_shuffle_mask(mask);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_vzero(Type vec_type) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::vzero, vec_type);
    Value* res = create_value(vec_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

static Value* build_vec_typed(Builder* b, Arena& arena, Opcode op, Type res_type,
                              std::initializer_list<Value*> operands) {
    Instruction* inst = arena.make<Instruction>(op, res_type);
    for (Value* v : operands) inst->add_operand(v);
    Value* res = b->create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

Value* Builder::build_vcmp(Opcode cmp, Value* lhs, Value* rhs) {
    Type res_type = (lhs ? lhs->type() : Type::f32x4()).vector_mask_type();
    return build_vec_typed(this, get_arena(), cmp, res_type, {lhs, rhs});
}
Value* Builder::build_vselect(Value* mask, Value* a, Value* b) {
    return build_vec_typed(this, get_arena(), Opcode::vselect, a ? a->type() : Type::f32x4(), {mask, a, b});
}
Value* Builder::build_vshl(Value* val, Value* amount) { return build_vec_bin_op(this, get_arena(), Opcode::vshl, val, amount); }
Value* Builder::build_vlshr(Value* val, Value* amount) { return build_vec_bin_op(this, get_arena(), Opcode::vlshr, val, amount); }
Value* Builder::build_vashr(Value* val, Value* amount) { return build_vec_bin_op(this, get_arena(), Opcode::vashr, val, amount); }
Value* Builder::build_vconvert(Type to, Value* val) {
    return build_vec_typed(this, get_arena(), Opcode::vconvert, to, {val});
}
Value* Builder::build_vbitcast(Type to, Value* val) {
    return build_vec_typed(this, get_arena(), Opcode::vbitcast, to, {val});
}
Value* Builder::build_vfloor(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vfloor, val); }
Value* Builder::build_vceil(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vceil, val); }
Value* Builder::build_vtrunc(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vtrunc, val); }
Value* Builder::build_vround(Value* val) { return build_vec_un_op(this, get_arena(), Opcode::vround, val); }
Value* Builder::build_vany(Value* mask) {
    return build_vec_typed(this, get_arena(), Opcode::vany, Type::i32(), {mask});
}

} // namespace brass
