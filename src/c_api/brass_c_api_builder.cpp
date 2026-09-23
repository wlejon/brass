#include "brass_c_api_internal.hpp"

using namespace brass;

namespace {

// Every builder entry point goes through builder_ready: a builder with no
// function (created from a null or dead function handle), whose module was
// destroyed, or that has not been positioned at a block is a reported error,
// never a silent no-op or a handle to an instruction inserted nowhere.
bool builder_ready(BrassBuilder b, const char* api) {
    if (!b) return false;
    if (!b->is_valid || !b->func || !b->mod) {
        set_ctx_error(b->ctx, std::string(api) +
            ": builder is invalid (it has no function, or its module was destroyed)");
        return false;
    }
    if (!b->builder.current_block()) {
        set_ctx_error(b->ctx, std::string(api) + ": builder is not positioned at a block");
        return false;
    }
    return true;
}

bool operand_ok(BrassBuilder b, BrassValue v, const char* api) {
    if (!v || !v->val) {
        set_ctx_error(b->ctx, std::string(api) +
            ": value operand is null or belongs to a destroyed module");
        return false;
    }
    if (v->mod != b->mod) {
        set_ctx_error(b->ctx, std::string(api) +
            ": value operand belongs to a different module than the builder");
        return false;
    }
    return true;
}

bool operands_ok(BrassBuilder b, const BrassValue* args, size_t count, const char* api,
                 std::vector<Value*>& out) {
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (!args || !operand_ok(b, args[i], api)) return false;
        out.push_back(args[i]->val);
    }
    return true;
}

bool type_ok(BrassBuilder b, BrassType t, const char* api) {
    if (!t) {
        set_ctx_error(b->ctx, std::string(api) + ": type is null");
        return false;
    }
    return true;
}

bool target_ok(BrassBuilder b, BrassBlock blk, const char* api) {
    if (!blk || !blk->block) {
        set_ctx_error(b->ctx, std::string(api) +
            ": target block is null or belongs to a destroyed module");
        return false;
    }
    if (blk->func != b->func) {
        set_ctx_error(b->ctx, std::string(api) +
            ": target block belongs to a different function than the builder");
        return false;
    }
    return true;
}

BrassValue wrap(BrassBuilder b, Value* res, const char* api) {
    if (!res) {
        set_ctx_error(b->ctx, std::string(api) + ": builder produced no value");
        return nullptr;
    }
    return b->ctx->wrap_value(res, b->mod);
}

template <typename F>
BrassValue guarded_value(BrassBuilder b, const char* api, F&& build) {
    try {
        return wrap(b, build(), api);
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, api, e);
        return nullptr;
    }
}

template <typename F>
BrassStatus guarded_status(BrassBuilder b, const char* api, F&& build) {
    try {
        build();
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, api, e);
        return BRASS_ERR_GENERIC;
    }
}

template <typename F>
BrassValue binary(BrassBuilder b, BrassValue lhs, BrassValue rhs, const char* api, F&& build) {
    if (!builder_ready(b, api) || !operand_ok(b, lhs, api) || !operand_ok(b, rhs, api)) return nullptr;
    return guarded_value(b, api, [&] { return build(lhs->val, rhs->val); });
}

} // namespace

extern "C" {

BrassBuilder brass_builder_create(BrassContext ctx, BrassFunction fn) {
    if (!ctx) return nullptr;
    if (!fn || !fn->func || !fn->mod) {
        set_ctx_error(ctx, "brass_builder_create: function is null or belongs to a destroyed module");
        return nullptr;
    }
    if (fn->ctx != ctx) {
        set_ctx_error(ctx, "brass_builder_create: function belongs to a different context");
        return nullptr;
    }
    try {
        auto b = std::make_unique<BrassBuilder_T>();
        b->ctx = ctx;
        b->func = fn->func;
        b->mod = fn->mod;
        b->builder.set_function(fn->func);
        b->is_valid = true;
        ctx->builders.push_back(b.get());
        ctx_retain(ctx);
        return b.release();
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_builder_create", e);
        return nullptr;
    }
}

void brass_builder_destroy(BrassBuilder b) {
    if (!b) return;
    BrassContext ctx = b->ctx;
    if (ctx) {
        auto& list = ctx->builders;
        list.erase(std::remove(list.begin(), list.end(), b), list.end());
    }
    delete b;
    ctx_release(ctx);
}

void brass_builder_position_at_end(BrassBuilder b, BrassBlock blk) {
    if (!b) return;
    if (!b->is_valid || !b->func || !b->mod) {
        set_ctx_error(b->ctx, "brass_builder_position_at_end: builder is invalid "
            "(it has no function, or its module was destroyed)");
        return;
    }
    if (!blk || !blk->block || !blk->func) {
        set_ctx_error(b->ctx, "brass_builder_position_at_end: block is null or belongs to a destroyed module");
        return;
    }
    if (blk->mod != b->mod) {
        set_ctx_error(b->ctx, "brass_builder_position_at_end: block belongs to a different module than the builder");
        return;
    }
    b->func = blk->func;
    b->builder.position_at_end(blk->block);
}

/* Constants */
BrassValue brass_build_iconst_i32(BrassBuilder b, int32_t val) {
    const char* api = "brass_build_iconst_i32";
    if (!builder_ready(b, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_iconst_i32(val); });
}

BrassValue brass_build_iconst_i64(BrassBuilder b, int64_t val) {
    const char* api = "brass_build_iconst_i64";
    if (!builder_ready(b, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_iconst_i64(val); });
}

BrassValue brass_build_fconst_f32(BrassBuilder b, float val) {
    const char* api = "brass_build_fconst_f32";
    if (!builder_ready(b, api)) return nullptr;
    return guarded_value(b, api, [&] {
        Instruction* inst = b->builder.arena().make<Instruction>(Opcode::fconst_f64, Type::f32());
        inst->set_imm_f64(static_cast<double>(val));
        Value* res = b->builder.create_value(Type::f32());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b->builder.insert(inst);
        return res;
    });
}

BrassValue brass_build_fconst_f64(BrassBuilder b, double val) {
    const char* api = "brass_build_fconst_f64";
    if (!builder_ready(b, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_fconst_f64(val); });
}

BrassValue brass_build_bconst(BrassBuilder b, int val) {
    return brass_build_iconst_i32(b, val ? 1 : 0);
}

/* Arithmetic & Bitwise */
BrassValue brass_build_add(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_add", [&](Value* l, Value* r) { return b->builder.build_add(l, r); });
}

BrassValue brass_build_sub(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_sub", [&](Value* l, Value* r) { return b->builder.build_sub(l, r); });
}

BrassValue brass_build_mul(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_mul", [&](Value* l, Value* r) { return b->builder.build_mul(l, r); });
}

BrassValue brass_build_sdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_sdiv", [&](Value* l, Value* r) { return b->builder.build_sdiv(l, r); });
}

BrassValue brass_build_udiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_udiv", [&](Value* l, Value* r) { return b->builder.build_udiv(l, r); });
}

BrassValue brass_build_fdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_fdiv", [&](Value* l, Value* r) {
        Type t = l->type().is_float() ? l->type() : Type::f64();
        Instruction* inst = b->builder.arena().make<Instruction>(Opcode::sdiv, t);
        inst->add_operand(l);
        inst->add_operand(r);
        Value* res = b->builder.create_value(t);
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b->builder.insert(inst);
        return res;
    });
}

BrassValue brass_build_and(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_and", [&](Value* l, Value* r) { return b->builder.build_and(l, r); });
}

BrassValue brass_build_or(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_or", [&](Value* l, Value* r) { return b->builder.build_or(l, r); });
}

BrassValue brass_build_xor(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_xor", [&](Value* l, Value* r) { return b->builder.build_xor(l, r); });
}

BrassValue brass_build_shl(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_shl", [&](Value* l, Value* r) { return b->builder.build_shl(l, r); });
}

BrassValue brass_build_shr(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_shr", [&](Value* l, Value* r) { return b->builder.build_lshr(l, r); });
}

BrassValue brass_build_sar(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_sar", [&](Value* l, Value* r) { return b->builder.build_ashr(l, r); });
}

/* Comparisons */
BrassValue brass_build_cmp(BrassBuilder b, BrassCmpOp op, BrassValue lhs, BrassValue rhs) {
    const char* api = "brass_build_cmp";
    if (op < BRASS_CMP_EQ || op > BRASS_CMP_UGE) {
        if (b) set_ctx_error(b->ctx, "brass_build_cmp: unknown comparison operator");
        return nullptr;
    }
    return binary(b, lhs, rhs, api, [&](Value* l, Value* r) -> Value* {
        switch (op) {
            case BRASS_CMP_EQ:  return b->builder.build_eq(l, r);
            case BRASS_CMP_NE:  return b->builder.build_ne(l, r);
            case BRASS_CMP_SLT: return b->builder.build_slt(l, r);
            case BRASS_CMP_SLE: return b->builder.build_sle(l, r);
            case BRASS_CMP_SGT: return b->builder.build_sgt(l, r);
            case BRASS_CMP_SGE: return b->builder.build_sge(l, r);
            case BRASS_CMP_ULT: return b->builder.build_ult(l, r);
            case BRASS_CMP_ULE: return b->builder.build_ule(l, r);
            case BRASS_CMP_UGT: return b->builder.build_ugt(l, r);
            case BRASS_CMP_UGE: return b->builder.build_uge(l, r);
        }
        return nullptr;
    });
}

/* Control Flow */
BrassStatus brass_build_br(BrassBuilder b, BrassBlock target, const BrassValue* args, size_t arg_count) {
    const char* api = "brass_build_br";
    std::vector<Value*> vargs;
    if (!builder_ready(b, api) || !target_ok(b, target, api) ||
        !operands_ok(b, args, arg_count, api, vargs)) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] {
        b->builder.build_br(target->block, Span<Value* const>(vargs.data(), vargs.size()));
    });
}

BrassStatus brass_build_br_if(BrassBuilder b, BrassValue cond,
                              BrassBlock true_target, const BrassValue* true_args, size_t true_arg_count,
                              BrassBlock false_target, const BrassValue* false_args, size_t false_arg_count) {
    const char* api = "brass_build_br_if";
    std::vector<Value*> targs;
    std::vector<Value*> fargs;
    if (!builder_ready(b, api) || !operand_ok(b, cond, api) ||
        !target_ok(b, true_target, api) || !target_ok(b, false_target, api) ||
        !operands_ok(b, true_args, true_arg_count, api, targs) ||
        !operands_ok(b, false_args, false_arg_count, api, fargs)) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] {
        b->builder.build_br_if(cond->val,
            true_target->block, Span<Value* const>(targs.data(), targs.size()),
            false_target->block, Span<Value* const>(fargs.data(), fargs.size()));
    });
}

/* A NULL value is `ret void` and is only accepted in a void function; a
 * value must match the function's return type. */
BrassStatus brass_build_ret(BrassBuilder b, BrassValue val) {
    const char* api = "brass_build_ret";
    if (!builder_ready(b, api)) return BRASS_ERR_INVALID_ARGUMENT;
    Type rt = b->func->return_type();
    if (!val) {
        if (!rt.is_void()) {
            set_ctx_error(b->ctx, std::string(api) + ": function '" + std::string(b->func->name()) +
                "' is not void; a return value is required");
            return BRASS_ERR_INVALID_ARGUMENT;
        }
        return guarded_status(b, api, [&] { b->builder.build_ret_void(); });
    }
    if (!operand_ok(b, val, api)) return BRASS_ERR_INVALID_ARGUMENT;
    if (rt.is_void()) {
        set_ctx_error(b->ctx, std::string(api) + ": function '" + std::string(b->func->name()) +
            "' is void; pass NULL to return without a value");
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    if (val->val->type() != rt) {
        set_ctx_error(b->ctx, std::string(api) + ": return value type does not match the return type of function '" +
            std::string(b->func->name()) + "'");
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] { b->builder.build_ret(val->val); });
}

BrassStatus brass_build_unreachable(BrassBuilder b) {
    const char* api = "brass_build_unreachable";
    if (!builder_ready(b, api)) return BRASS_ERR_INVALID_ARGUMENT;
    return guarded_status(b, api, [&] { b->builder.build_unreachable(); });
}

/* Calls & Function Pointers */
BrassValue brass_build_call(BrassBuilder b, const char* callee, BrassType return_type, const BrassValue* args, size_t arg_count) {
    const char* api = "brass_build_call";
    std::vector<Value*> vargs;
    if (!builder_ready(b, api) || !type_ok(b, return_type, api) ||
        !operands_ok(b, args, arg_count, api, vargs)) {
        return nullptr;
    }
    if (!callee) {
        set_ctx_error(b->ctx, "brass_build_call: callee name is null");
        return nullptr;
    }
    try {
        Value* res = b->builder.build_call(callee, return_type->type, Span<Value* const>(vargs.data(), vargs.size()));
        // A void call has no result: it is emitted and NULL is returned.
        if (return_type->type.is_void()) return nullptr;
        return wrap(b, res, api);
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, api, e);
        return nullptr;
    }
}

BrassValue brass_build_func_addr(BrassBuilder b, const char* name) {
    const char* api = "brass_build_func_addr";
    if (!builder_ready(b, api)) return nullptr;
    if (!name) {
        set_ctx_error(b->ctx, "brass_build_func_addr: function name is null");
        return nullptr;
    }
    return guarded_value(b, api, [&] { return b->builder.build_func_addr(name); });
}

/* Memory */
BrassValue brass_build_load(BrassBuilder b, BrassType type, BrassValue base, int32_t offset) {
    const char* api = "brass_build_load";
    if (!builder_ready(b, api) || !type_ok(b, type, api) || !operand_ok(b, base, api)) return nullptr;
    return guarded_value(b, api, [&] {
        return (offset == 0)
            ? b->builder.build_load(type->type, base->val)
            : b->builder.build_load(type->type, base->val, offset);
    });
}

BrassStatus brass_build_store(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val) {
    const char* api = "brass_build_store";
    if (!builder_ready(b, api) || !type_ok(b, type, api) ||
        !operand_ok(b, base, api) || !operand_ok(b, val, api)) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] {
        if (offset == 0) {
            b->builder.build_store(type->type, base->val, val->val);
        } else {
            b->builder.build_store(type->type, base->val, offset, val->val);
        }
    });
}

BrassValue brass_build_load_indexed(BrassBuilder b, BrassType type, BrassValue base, BrassValue index, uint8_t scale, int32_t offset) {
    const char* api = "brass_build_load_indexed";
    if (!builder_ready(b, api) || !type_ok(b, type, api) ||
        !operand_ok(b, base, api) || !operand_ok(b, index, api)) {
        return nullptr;
    }
    return guarded_value(b, api, [&] {
        return b->builder.build_load_indexed(type->type, base->val, index->val, scale, offset);
    });
}

BrassStatus brass_build_store_indexed(BrassBuilder b, BrassType type, BrassValue base, BrassValue index, uint8_t scale, int32_t offset, BrassValue val) {
    const char* api = "brass_build_store_indexed";
    if (!builder_ready(b, api) || !type_ok(b, type, api) || !operand_ok(b, base, api) ||
        !operand_ok(b, index, api) || !operand_ok(b, val, api)) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] {
        b->builder.build_store_indexed(type->type, base->val, index->val, scale, offset, val->val);
    });
}

/* Vector (SIMD) & FMA */
BrassValue brass_build_fma(BrassBuilder b, BrassValue a, BrassValue b_val, BrassValue c) {
    const char* api = "brass_build_fma";
    if (!builder_ready(b, api) || !operand_ok(b, a, api) || !operand_ok(b, b_val, api) || !operand_ok(b, c, api)) {
        return nullptr;
    }
    return guarded_value(b, api, [&] { return b->builder.build_fma(a->val, b_val->val, c->val); });
}

BrassValue brass_build_vfma(BrassBuilder b, BrassValue a, BrassValue b_val, BrassValue c) {
    const char* api = "brass_build_vfma";
    if (!builder_ready(b, api) || !operand_ok(b, a, api) || !operand_ok(b, b_val, api) || !operand_ok(b, c, api)) {
        return nullptr;
    }
    return guarded_value(b, api, [&] { return b->builder.build_vfma(a->val, b_val->val, c->val); });
}

BrassValue brass_build_vload(BrassBuilder b, BrassType type, BrassValue base, int32_t offset) {
    const char* api = "brass_build_vload";
    if (!builder_ready(b, api) || !type_ok(b, type, api) || !operand_ok(b, base, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_vload(type->type, base->val, offset); });
}

BrassStatus brass_build_vstore(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val) {
    const char* api = "brass_build_vstore";
    if (!builder_ready(b, api) || !type_ok(b, type, api) ||
        !operand_ok(b, base, api) || !operand_ok(b, val, api)) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return guarded_status(b, api, [&] { b->builder.build_vstore(type->type, base->val, offset, val->val); });
}

BrassValue brass_build_vadd(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vadd", [&](Value* l, Value* r) { return b->builder.build_vadd(l, r); });
}

BrassValue brass_build_vsub(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vsub", [&](Value* l, Value* r) { return b->builder.build_vsub(l, r); });
}

BrassValue brass_build_vmul(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vmul", [&](Value* l, Value* r) { return b->builder.build_vmul(l, r); });
}

BrassValue brass_build_vdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vdiv", [&](Value* l, Value* r) { return b->builder.build_vdiv(l, r); });
}

BrassValue brass_build_vmin(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vmin", [&](Value* l, Value* r) { return b->builder.build_vmin(l, r); });
}

BrassValue brass_build_vmax(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    return binary(b, lhs, rhs, "brass_build_vmax", [&](Value* l, Value* r) { return b->builder.build_vmax(l, r); });
}

BrassValue brass_build_vbroadcast(BrassBuilder b, BrassType vec_type, BrassValue scalar_val) {
    const char* api = "brass_build_vbroadcast";
    if (!builder_ready(b, api) || !type_ok(b, vec_type, api) || !operand_ok(b, scalar_val, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_vbroadcast(vec_type->type, scalar_val->val); });
}

BrassValue brass_build_vextract_lane(BrassBuilder b, BrassValue vec_val, uint32_t lane) {
    const char* api = "brass_build_vextract_lane";
    if (!builder_ready(b, api) || !operand_ok(b, vec_val, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_vextract_lane(vec_val->val, lane); });
}

BrassValue brass_build_vinsert_lane(BrassBuilder b, BrassValue vec_val, BrassValue scalar_val, uint32_t lane) {
    const char* api = "brass_build_vinsert_lane";
    if (!builder_ready(b, api) || !operand_ok(b, vec_val, api) || !operand_ok(b, scalar_val, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_vinsert_lane(vec_val->val, scalar_val->val, lane); });
}

BrassValue brass_build_vzero(BrassBuilder b, BrassType vec_type) {
    const char* api = "brass_build_vzero";
    if (!builder_ready(b, api) || !type_ok(b, vec_type, api)) return nullptr;
    return guarded_value(b, api, [&] { return b->builder.build_vzero(vec_type->type); });
}

} /* extern "C" */
