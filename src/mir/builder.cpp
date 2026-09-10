#include <brass/mir/builder.hpp>
#include <stdexcept>

namespace brass {

Builder::Builder(Module& module) noexcept
    : module_(&module) {}

Builder::Builder(Function& fn) noexcept
    : module_(fn.parent()), function_(&fn) {}

void Builder::set_function(Function* fn) noexcept {
    function_ = fn;
    if (fn && fn->parent()) {
        module_ = fn->parent();
    }
}

void Builder::position_at_end(BasicBlock* bb) noexcept {
    block_ = bb;
    insert_before_ = nullptr;
    if (bb && bb->parent()) {
        function_ = bb->parent();
        if (function_ && function_->parent()) {
            module_ = function_->parent();
        }
    }
}

void Builder::position_before(Instruction* inst) noexcept {
    if (inst) {
        block_ = inst->parent();
        insert_before_ = inst;
        if (block_ && block_->parent()) {
            function_ = block_->parent();
            if (function_ && function_->parent()) {
                module_ = function_->parent();
            }
        }
    }
}

void Builder::position_after(Instruction* inst) noexcept {
    if (inst) {
        block_ = inst->parent();
        insert_before_ = inst->next();
        if (block_ && block_->parent()) {
            function_ = block_->parent();
            if (function_ && function_->parent()) {
                module_ = function_->parent();
            }
        }
    }
}

Arena& Builder::get_arena() {
    if (module_) {
        return module_->arena();
    }
    static Arena fallback_arena;
    return fallback_arena;
}

StringPool& Builder::get_string_pool() {
    if (module_) {
        return module_->string_pool();
    }
    static StringPool fallback_pool;
    return fallback_pool;
}

Value* Builder::create_value(Type type) {
    uint32_t id = function_ ? function_->next_value_id() : 0;
    return get_arena().make<Value>(id, type, ValueKind::InstructionResult);
}

BasicBlock* Builder::create_block() {
    return create_block("");
}

BasicBlock* Builder::create_block(std::string_view name) {
    uint32_t id = function_ ? function_->next_block_id() : 0;
    std::string_view sym = name.empty() ? "" : get_string_pool().intern(name);
    BasicBlock* bb = get_arena().make<BasicBlock>(id, sym);
    if (function_) {
        bb->set_parent(function_);
    }
    return bb;
}

BasicBlock* Builder::append_block() {
    return append_block("");
}

BasicBlock* Builder::append_block(std::string_view name) {
    BasicBlock* bb = create_block(name);
    if (function_) {
        function_->append_block(bb);
    }
    position_at_end(bb);
    return bb;
}

Value* Builder::add_block_param(BasicBlock* block, Type type) {
    if (!block) return nullptr;
    uint32_t id = function_ ? function_->next_value_id() : 0;
    Value* val = get_arena().make<Value>(id, type, ValueKind::BlockParam);
    block->add_param(val);
    return val;
}

Value* Builder::add_param(Type type) {
    return add_block_param(block_, type);
}

Instruction* Builder::insert(Instruction* inst) {
    if (!inst) return nullptr;
    if (current_loc_.is_valid()) {
        inst->set_loc(current_loc_);
    }
    if (block_) {
        if (insert_before_) {
            block_->insert_before(inst, insert_before_);
        } else {
            block_->append_instruction(inst);
        }
    }
    return inst;
}

Value* Builder::build_iconst_i32(int32_t val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::iconst_i32, Type::i32());
    inst->set_imm_i32(val);
    Value* res = create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_iconst_i64(int64_t val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::iconst_i64, Type::i64());
    inst->set_imm_i64(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_fconst_f64(double val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::fconst_f64, Type::f64());
    inst->set_imm_f64(val);
    Value* res = create_value(Type::f64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_patchable_const_i32(std::string_view symbol, int32_t initial_val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::patchable_const_i32, Type::i32());
    inst->set_symbol(get_string_pool().intern(symbol));
    inst->set_imm_i32(initial_val);
    Value* res = create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_patchable_const_i64(std::string_view symbol, int64_t initial_val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::patchable_const_i64, Type::i64());
    inst->set_symbol(get_string_pool().intern(symbol));
    inst->set_imm_i64(initial_val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_sext_i64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::sext_i64, Type::i64());
    inst->add_operand(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_zext_i64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::zext_i64, Type::i64());
    inst->add_operand(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_trunc_i32(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::trunc_i32, Type::i32());
    inst->add_operand(val);
    Value* res = create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_fptosi_i32(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::fptosi_i32, Type::i32());
    inst->add_operand(val);
    Value* res = create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_fptosi_i64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::fptosi_i64, Type::i64());
    inst->add_operand(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_sitofp_f64_i32(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::sitofp_f64_i32, Type::f64());
    inst->add_operand(val);
    Value* res = create_value(Type::f64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_sitofp_f64_i64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::sitofp_f64_i64, Type::f64());
    inst->add_operand(val);
    Value* res = create_value(Type::f64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_bitcast_i64_f64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::bitcast_i64_f64, Type::i64());
    inst->add_operand(val);
    Value* res = create_value(Type::i64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_bitcast_f64_i64(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::bitcast_f64_i64, Type::f64());
    inst->add_operand(val);
    Value* res = create_value(Type::f64());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

static Value* build_bin_op(Builder* b, Arena& arena, Opcode op, Value* lhs, Value* rhs) {
    Type res_type = lhs ? lhs->type() : Type::i32();
    Instruction* inst = arena.make<Instruction>(op, res_type);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    Value* res = b->create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

static Value* build_un_op(Builder* b, Arena& arena, Opcode op, Value* val) {
    Type res_type = val ? val->type() : Type::i32();
    Instruction* inst = arena.make<Instruction>(op, res_type);
    inst->add_operand(val);
    Value* res = b->create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

static Value* build_cmp_op(Builder* b, Arena& arena, Opcode op, Value* lhs, Value* rhs) {
    Instruction* inst = arena.make<Instruction>(op, Type::i32());
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    Value* res = b->create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

Value* Builder::build_add(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::add, lhs, rhs); }
Value* Builder::build_sub(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::sub, lhs, rhs); }
Value* Builder::build_mul(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::mul, lhs, rhs); }
Value* Builder::build_sdiv(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::sdiv, lhs, rhs); }
Value* Builder::build_udiv(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::udiv, lhs, rhs); }
Value* Builder::build_smod(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::smod, lhs, rhs); }
Value* Builder::build_umod(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::umod, lhs, rhs); }
Value* Builder::build_neg(Value* val) { return build_un_op(this, get_arena(), Opcode::neg, val); }
Value* Builder::build_and(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::and_, lhs, rhs); }
Value* Builder::build_or(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::or_, lhs, rhs); }
Value* Builder::build_xor(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::xor_, lhs, rhs); }
Value* Builder::build_shl(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::shl, lhs, rhs); }
Value* Builder::build_lshr(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::lshr, lhs, rhs); }
Value* Builder::build_ashr(Value* lhs, Value* rhs) { return build_bin_op(this, get_arena(), Opcode::ashr, lhs, rhs); }
Value* Builder::build_not(Value* val) { return build_un_op(this, get_arena(), Opcode::not_, val); }
Value* Builder::build_clz(Value* val) { return build_un_op(this, get_arena(), Opcode::clz, val); }
Value* Builder::build_ctz(Value* val) { return build_un_op(this, get_arena(), Opcode::ctz, val); }
Value* Builder::build_popcnt(Value* val) { return build_un_op(this, get_arena(), Opcode::popcnt, val); }

Value* Builder::build_eq(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::eq, lhs, rhs); }
Value* Builder::build_ne(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::ne, lhs, rhs); }
Value* Builder::build_slt(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::slt, lhs, rhs); }
Value* Builder::build_ult(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::ult, lhs, rhs); }
Value* Builder::build_sle(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::sle, lhs, rhs); }
Value* Builder::build_ule(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::ule, lhs, rhs); }
Value* Builder::build_sgt(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::sgt, lhs, rhs); }
Value* Builder::build_ugt(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::ugt, lhs, rhs); }
Value* Builder::build_sge(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::sge, lhs, rhs); }
Value* Builder::build_uge(Value* lhs, Value* rhs) { return build_cmp_op(this, get_arena(), Opcode::uge, lhs, rhs); }

static Value* build_overflow_op(Builder* b, Arena& arena, Opcode op, Value* lhs, Value* rhs) {
    Instruction* inst = arena.make<Instruction>(op, Type::i32());
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    Value* res = b->create_value(Type::i32());
    res->set_defining_instruction(inst);
    inst->set_result(res);
    b->insert(inst);
    return res;
}

Value* Builder::build_sadd_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::sadd_overflow, lhs, rhs); }
Value* Builder::build_ssub_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::ssub_overflow, lhs, rhs); }
Value* Builder::build_smul_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::smul_overflow, lhs, rhs); }
Value* Builder::build_uadd_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::uadd_overflow, lhs, rhs); }
Value* Builder::build_usub_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::usub_overflow, lhs, rhs); }
Value* Builder::build_umul_overflow(Value* lhs, Value* rhs) { return build_overflow_op(this, get_arena(), Opcode::umul_overflow, lhs, rhs); }

Value* Builder::build_select(Value* cond, Value* true_val, Value* false_val) {
    Type res_type = true_val ? true_val->type() : (false_val ? false_val->type() : Type::i64());
    Instruction* inst = get_arena().make<Instruction>(Opcode::select, res_type);
    inst->add_operand(cond);
    inst->add_operand(true_val);
    inst->add_operand(false_val);
    Value* res = create_value(res_type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Value* Builder::build_load(Type type, Value* base) {
    return build_load(type, base, 0);
}

Value* Builder::build_load(Type type, Value* base, int32_t offset) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::load, type);
    inst->add_operand(base);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    Value* res = create_value(type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Instruction* Builder::build_store(Type type, Value* base, int32_t offset, Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::store, Type::void_type());
    inst->add_operand(base);
    inst->add_operand(val);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    insert(inst);
    return inst;
}

Instruction* Builder::build_store(Type type, Value* base, Value* val) {
    return build_store(type, base, 0, val);
}

Value* Builder::build_load_indexed(Type type, Value* base, Value* index, uint8_t scale) {
    return build_load_indexed(type, base, index, scale, 0);
}

Value* Builder::build_load_indexed(Type type, Value* base, Value* index, uint8_t scale, int32_t offset) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::load_indexed, type);
    inst->add_operand(base);
    inst->add_operand(index);
    inst->set_scale(scale);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    Value* res = create_value(type);
    res->set_defining_instruction(inst);
    inst->set_result(res);
    insert(inst);
    return res;
}

Instruction* Builder::build_store_indexed(Type type, Value* base, Value* index, uint8_t scale, int32_t offset, Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::store_indexed, Type::void_type());
    inst->add_operand(base);
    inst->add_operand(index);
    inst->add_operand(val);
    inst->set_scale(scale);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    insert(inst);
    return inst;
}

Instruction* Builder::build_store_indexed(Type type, Value* base, Value* index, uint8_t scale, Value* val) {
    return build_store_indexed(type, base, index, scale, 0, val);
}

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

Value* Builder::build_vadd(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vadd, lhs, rhs); }
Value* Builder::build_vsub(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vsub, lhs, rhs); }
Value* Builder::build_vmul(Value* lhs, Value* rhs) { return build_vec_bin_op(this, get_arena(), Opcode::vmul, lhs, rhs); }
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

Value* Builder::build_call(std::string_view callee, Type return_type, Span<Value* const> args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::call, return_type);
    inst->set_symbol(get_string_pool().intern(callee));
    for (size_t i = 0; i < args.size(); ++i) {
        inst->add_operand(args[i]);
    }
    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        res->set_defining_instruction(inst);
        inst->set_result(res);
    }
    insert(inst);
    return inst->result();
}

Value* Builder::build_call(std::string_view callee, Type return_type, std::initializer_list<Value*> args) {
    return build_call(callee, return_type, Span<Value* const>(args.begin(), args.size()));
}

Value* Builder::build_call(std::string_view callee, Type return_type) {
    return build_call(callee, return_type, Span<Value* const>());
}

Value* Builder::build_call_indirect(Value* callee_ptr, Type return_type, Span<Value* const> args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::call_indirect, return_type);
    inst->add_operand(callee_ptr);
    for (size_t i = 0; i < args.size(); ++i) {
        inst->add_operand(args[i]);
    }
    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        res->set_defining_instruction(inst);
        inst->set_result(res);
    }
    insert(inst);
    return inst->result();
}

Value* Builder::build_call_indirect(Value* callee_ptr, Type return_type, std::initializer_list<Value*> args) {
    return build_call_indirect(callee_ptr, return_type, Span<Value* const>(args.begin(), args.size()));
}

Value* Builder::build_call_indirect(Value* callee_ptr, Type return_type) {
    return build_call_indirect(callee_ptr, return_type, Span<Value* const>());
}

Value* Builder::build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type, Span<Value* const> args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::patchable_call, return_type);
    inst->set_symbol(get_string_pool().intern(patch_symbol));
    inst->set_extra_symbol(get_string_pool().intern(callee));
    for (size_t i = 0; i < args.size(); ++i) {
        inst->add_operand(args[i]);
    }
    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        res->set_defining_instruction(inst);
        inst->set_result(res);
    }
    insert(inst);
    return inst->result();
}

Value* Builder::build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type, std::initializer_list<Value*> args) {
    return build_patchable_call(patch_symbol, callee, return_type, Span<Value* const>(args.begin(), args.size()));
}

Value* Builder::build_patchable_call(std::string_view patch_symbol, std::string_view callee, Type return_type) {
    return build_patchable_call(patch_symbol, callee, return_type, Span<Value* const>());
}

Instruction* Builder::build_safepoint() {
    Instruction* inst = get_arena().make<Instruction>(Opcode::safepoint, Type::void_type());
    insert(inst);
    return inst;
}

Instruction* Builder::build_guard(Value* cond, std::string_view exit_label, Span<Value* const> state_values) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::guard, Type::void_type());
    inst->add_operand(cond);
    inst->set_symbol(get_string_pool().intern(exit_label));
    for (size_t i = 0; i < state_values.size(); ++i) {
        inst->add_state_value(state_values[i]);
    }
    insert(inst);
    return inst;
}

Instruction* Builder::build_guard(Value* cond, std::string_view exit_label, std::initializer_list<Value*> state_values) {
    return build_guard(cond, exit_label, Span<Value* const>(state_values.begin(), state_values.size()));
}

Instruction* Builder::build_guard(Value* cond, std::string_view exit_label) {
    return build_guard(cond, exit_label, Span<Value* const>());
}

Instruction* Builder::build_resume_point(uint32_t resume_id) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::resume_point, Type::void_type());
    inst->set_resume_id(resume_id);
    insert(inst);
    return inst;
}

Instruction* Builder::build_br(BasicBlock* target) {
    return build_br(target, Span<Value* const>());
}

Instruction* Builder::build_br(BasicBlock* target, Span<Value* const> args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::br, Type::void_type());
    std::vector<Value*> branch_args;
    branch_args.reserve(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        branch_args.push_back(args[i]);
    }
    inst->set_branch_target(BranchTarget(target, std::move(branch_args)));
    insert(inst);
    return inst;
}

Instruction* Builder::build_br(BasicBlock* target, std::initializer_list<Value*> args) {
    return build_br(target, Span<Value* const>(args.begin(), args.size()));
}

Instruction* Builder::build_br_if(Value* cond, BasicBlock* true_target, BasicBlock* false_target) {
    return build_br_if(cond, true_target, Span<Value* const>(), false_target, Span<Value* const>());
}

Instruction* Builder::build_br_if(Value* cond, BasicBlock* true_target, Span<Value* const> true_args,
                                 BasicBlock* false_target, Span<Value* const> false_args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::br_if, Type::void_type());
    inst->add_operand(cond);

    std::vector<Value*> t_args;
    t_args.reserve(true_args.size());
    for (size_t i = 0; i < true_args.size(); ++i) {
        t_args.push_back(true_args[i]);
    }

    std::vector<Value*> f_args;
    f_args.reserve(false_args.size());
    for (size_t i = 0; i < false_args.size(); ++i) {
        f_args.push_back(false_args[i]);
    }

    inst->set_true_target(BranchTarget(true_target, std::move(t_args)));
    inst->set_false_target(BranchTarget(false_target, std::move(f_args)));
    insert(inst);
    return inst;
}

Instruction* Builder::build_br_if(Value* cond, BasicBlock* true_target, std::initializer_list<Value*> true_args,
                                 BasicBlock* false_target, std::initializer_list<Value*> false_args) {
    return build_br_if(cond, true_target, Span<Value* const>(true_args.begin(), true_args.size()),
                       false_target, Span<Value* const>(false_args.begin(), false_args.size()));
}

Instruction* Builder::build_switch(Value* val, BasicBlock* default_target, Span<const SwitchCase> cases) {
    return build_switch(val, default_target, Span<Value* const>(), cases);
}

Instruction* Builder::build_switch(Value* val, BasicBlock* default_target, std::initializer_list<SwitchCase> cases) {
    return build_switch(val, default_target, Span<Value* const>(), Span<const SwitchCase>(cases.begin(), cases.size()));
}

Instruction* Builder::build_switch(Value* val, BasicBlock* default_target, Span<Value* const> default_args, Span<const SwitchCase> cases) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::switch_, Type::void_type());
    inst->add_operand(val);
    std::vector<Value*> def_args;
    def_args.reserve(default_args.size());
    for (size_t i = 0; i < default_args.size(); ++i) {
        def_args.push_back(default_args[i]);
    }
    inst->set_default_target(BranchTarget(default_target, std::move(def_args)));
    for (size_t i = 0; i < cases.size(); ++i) {
        inst->add_switch_case(cases[i].value, cases[i].target);
    }
    insert(inst);
    return inst;
}

Instruction* Builder::build_switch(Value* val, BasicBlock* default_target, std::initializer_list<Value*> default_args, std::initializer_list<SwitchCase> cases) {
    return build_switch(val, default_target, Span<Value* const>(default_args.begin(), default_args.size()),
                        Span<const SwitchCase>(cases.begin(), cases.size()));
}

Instruction* Builder::build_ret(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::ret, Type::void_type());
    if (val) {
        inst->add_operand(val);
    }
    insert(inst);
    return inst;
}

Instruction* Builder::build_ret_void() {
    return build_ret(nullptr);
}

Instruction* Builder::build_unreachable() {
    Instruction* inst = get_arena().make<Instruction>(Opcode::unreachable, Type::void_type());
    insert(inst);
    return inst;
}

Instruction* Builder::build_throw(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::throw_, Type::void_type());
    if (val) {
        inst->add_operand(val);
    }
    insert(inst);
    return inst;
}

Instruction* Builder::build_resume(Value* val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::resume, Type::void_type());
    if (val) {
        inst->add_operand(val);
    }
    insert(inst);
    return inst;
}

Value* Builder::build_landing_pad(Type type) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::landing_pad, type);
    Value* res = create_value(type);
    inst->set_result(res);
    res->set_defining_instruction(inst);
    insert(inst);
    return res;
}

Instruction* Builder::build_invoke(std::string_view callee, Type return_type, Span<Value* const> args,
                                   BasicBlock* normal_target, BasicBlock* unwind_target) {
    return build_invoke(callee, return_type, args, normal_target, {}, unwind_target, {});
}

Instruction* Builder::build_invoke(std::string_view callee, Type return_type, std::initializer_list<Value*> args,
                                   BasicBlock* normal_target, BasicBlock* unwind_target) {
    return build_invoke(callee, return_type, Span<Value* const>(args.begin(), args.size()),
                        normal_target, {}, unwind_target, {});
}

Instruction* Builder::build_invoke(std::string_view callee, Type return_type, Span<Value* const> args,
                                   BasicBlock* normal_target, Span<Value* const> normal_args,
                                   BasicBlock* unwind_target, Span<Value* const> unwind_args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::invoke, return_type);
    inst->set_symbol(get_string_pool().intern(callee));
    for (Value* arg : args) {
        inst->add_operand(arg);
    }
    std::vector<Value*> n_args(normal_args.begin(), normal_args.end());
    inst->set_normal_target(BranchTarget(normal_target, std::move(n_args)));

    std::vector<Value*> u_args(unwind_args.begin(), unwind_args.end());
    inst->set_unwind_target(BranchTarget(unwind_target, std::move(u_args)));

    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        inst->set_result(res);
        res->set_defining_instruction(inst);
    }

    insert(inst);
    return inst;
}

Instruction* Builder::build_invoke(std::string_view callee, Type return_type, std::initializer_list<Value*> args,
                                   BasicBlock* normal_target, std::initializer_list<Value*> normal_args,
                                   BasicBlock* unwind_target, std::initializer_list<Value*> unwind_args) {
    return build_invoke(callee, return_type, Span<Value* const>(args.begin(), args.size()),
                        normal_target, Span<Value* const>(normal_args.begin(), normal_args.size()),
                        unwind_target, Span<Value* const>(unwind_args.begin(), unwind_args.size()));
}

Value* Builder::build_coro_create(std::string_view callee, Span<Value* const> args) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::coro_create, Type::gcref());
    inst->set_symbol(get_string_pool().intern(callee));
    for (Value* arg : args) {
        inst->add_operand(arg);
    }
    Value* res = create_value(Type::gcref());
    inst->set_result(res);
    res->set_defining_instruction(inst);
    insert(inst);
    return res;
}

Value* Builder::build_coro_create(std::string_view callee, std::initializer_list<Value*> args) {
    return build_coro_create(callee, Span<Value* const>(args.begin(), args.size()));
}

Value* Builder::build_coro_suspend(Value* yield_val, uint32_t state_id, Type return_type) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::coro_suspend, return_type);
    if (yield_val) {
        inst->add_operand(yield_val);
    }
    inst->set_resume_id(state_id);
    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        inst->set_result(res);
        res->set_defining_instruction(inst);
        insert(inst);
        return res;
    }
    insert(inst);
    return nullptr;
}

Value* Builder::build_coro_resume(Value* coro_val, Value* input_val, Type return_type) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::coro_resume, return_type);
    if (coro_val) {
        inst->add_operand(coro_val);
    }
    if (input_val) {
        inst->add_operand(input_val);
    }
    if (!return_type.is_void()) {
        Value* res = create_value(return_type);
        inst->set_result(res);
        res->set_defining_instruction(inst);
        insert(inst);
        return res;
    }
    insert(inst);
    return nullptr;
}

Instruction* Builder::build_coro_destroy(Value* coro_val) {
    Instruction* inst = get_arena().make<Instruction>(Opcode::coro_destroy, Type::void_type());
    if (coro_val) {
        inst->add_operand(coro_val);
    }
    insert(inst);
    return inst;
}

} // namespace brass
