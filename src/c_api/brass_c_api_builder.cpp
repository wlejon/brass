#include "brass_c_api_internal.hpp"

using namespace brass;

extern "C" {

BrassBuilder brass_builder_create(BrassContext ctx, BrassFunction fn) {
    try {
        auto* b = new BrassBuilder_T();
        b->ctx = ctx;
        if (fn && fn->func) {
            b->func = fn->func;
            b->builder.set_function(fn->func);
        }
        return b;
    } catch (const std::exception& e) {
        set_ctx_exception(ctx, "brass_builder_create", e);
        return nullptr;
    }
}

void brass_builder_destroy(BrassBuilder b) {
    delete b;
}

void brass_builder_position_at_end(BrassBuilder b, BrassBlock blk) {
    if (!b || !blk || !blk->block) return;
    b->builder.position_at_end(blk->block);
}

/* Constants */
BrassValue brass_build_iconst_i32(BrassBuilder b, int32_t val) {
    if (!b) return nullptr;
    try {
        Value* res = b->builder.build_iconst_i32(val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_iconst_i32", e);
        return nullptr;
    }
}

BrassValue brass_build_iconst_i64(BrassBuilder b, int64_t val) {
    if (!b) return nullptr;
    try {
        Value* res = b->builder.build_iconst_i64(val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_iconst_i64", e);
        return nullptr;
    }
}

BrassValue brass_build_fconst_f32(BrassBuilder b, float val) {
    if (!b) return nullptr;
    try {
        Instruction* inst = b->builder.arena().make<Instruction>(Opcode::fconst_f64, Type::f32());
        inst->set_imm_f64(static_cast<double>(val));
        Value* res = b->builder.create_value(Type::f32());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b->builder.insert(inst);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_fconst_f32", e);
        return nullptr;
    }
}

BrassValue brass_build_fconst_f64(BrassBuilder b, double val) {
    if (!b) return nullptr;
    try {
        Value* res = b->builder.build_fconst_f64(val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_fconst_f64", e);
        return nullptr;
    }
}

BrassValue brass_build_bconst(BrassBuilder b, int val) {
    return brass_build_iconst_i32(b, val ? 1 : 0);
}

/* Arithmetic & Bitwise */
BrassValue brass_build_add(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_add(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_add", e);
        return nullptr;
    }
}

BrassValue brass_build_sub(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_sub(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_sub", e);
        return nullptr;
    }
}

BrassValue brass_build_mul(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_mul(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_mul", e);
        return nullptr;
    }
}

BrassValue brass_build_sdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_sdiv(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_sdiv", e);
        return nullptr;
    }
}

BrassValue brass_build_udiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_udiv(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_udiv", e);
        return nullptr;
    }
}

BrassValue brass_build_fdiv(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Type t = lhs->val->type().is_float() ? lhs->val->type() : Type::f64();
        Instruction* inst = b->builder.arena().make<Instruction>(Opcode::sdiv, t);
        inst->add_operand(lhs->val);
        inst->add_operand(rhs->val);
        Value* res = b->builder.create_value(t);
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b->builder.insert(inst);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_fdiv", e);
        return nullptr;
    }
}

BrassValue brass_build_and(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_and(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_and", e);
        return nullptr;
    }
}

BrassValue brass_build_or(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_or(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_or", e);
        return nullptr;
    }
}

BrassValue brass_build_xor(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_xor(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_xor", e);
        return nullptr;
    }
}

BrassValue brass_build_shl(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_shl(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_shl", e);
        return nullptr;
    }
}

BrassValue brass_build_shr(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_lshr(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_shr", e);
        return nullptr;
    }
}

BrassValue brass_build_sar(BrassBuilder b, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = b->builder.build_ashr(lhs->val, rhs->val);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_sar", e);
        return nullptr;
    }
}

/* Comparisons */
BrassValue brass_build_cmp(BrassBuilder b, BrassCmpOp op, BrassValue lhs, BrassValue rhs) {
    if (!b || !lhs || !lhs->val || !rhs || !rhs->val) return nullptr;
    try {
        Value* res = nullptr;
        switch (op) {
            case BRASS_CMP_EQ:  res = b->builder.build_eq(lhs->val, rhs->val); break;
            case BRASS_CMP_NE:  res = b->builder.build_ne(lhs->val, rhs->val); break;
            case BRASS_CMP_SLT: res = b->builder.build_slt(lhs->val, rhs->val); break;
            case BRASS_CMP_SLE: res = b->builder.build_sle(lhs->val, rhs->val); break;
            case BRASS_CMP_SGT: res = b->builder.build_sgt(lhs->val, rhs->val); break;
            case BRASS_CMP_SGE: res = b->builder.build_sge(lhs->val, rhs->val); break;
            case BRASS_CMP_ULT: res = b->builder.build_ult(lhs->val, rhs->val); break;
            case BRASS_CMP_ULE: res = b->builder.build_ule(lhs->val, rhs->val); break;
            case BRASS_CMP_UGT: res = b->builder.build_ugt(lhs->val, rhs->val); break;
            case BRASS_CMP_UGE: res = b->builder.build_uge(lhs->val, rhs->val); break;
            default: return nullptr;
        }
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_cmp", e);
        return nullptr;
    }
}

/* Control Flow */
BrassStatus brass_build_br(BrassBuilder b, BrassBlock target, const BrassValue* args, size_t arg_count) {
    if (!b || !target || !target->block) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        std::vector<Value*> vargs;
        vargs.reserve(arg_count);
        for (size_t i = 0; i < arg_count; ++i) {
            if (!args || !args[i] || !args[i]->val) return BRASS_ERR_INVALID_ARGUMENT;
            vargs.push_back(args[i]->val);
        }
        b->builder.build_br(target->block, Span<Value* const>(vargs.data(), vargs.size()));
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_br", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassStatus brass_build_br_if(BrassBuilder b, BrassValue cond,
                              BrassBlock true_target, const BrassValue* true_args, size_t true_arg_count,
                              BrassBlock false_target, const BrassValue* false_args, size_t false_arg_count) {
    if (!b || !cond || !cond->val || !true_target || !true_target->block || !false_target || !false_target->block) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<Value*> targs;
        targs.reserve(true_arg_count);
        for (size_t i = 0; i < true_arg_count; ++i) {
            if (!true_args || !true_args[i] || !true_args[i]->val) return BRASS_ERR_INVALID_ARGUMENT;
            targs.push_back(true_args[i]->val);
        }
        std::vector<Value*> fargs;
        fargs.reserve(false_arg_count);
        for (size_t i = 0; i < false_arg_count; ++i) {
            if (!false_args || !false_args[i] || !false_args[i]->val) return BRASS_ERR_INVALID_ARGUMENT;
            fargs.push_back(false_args[i]->val);
        }
        b->builder.build_br_if(cond->val,
            true_target->block, Span<Value* const>(targs.data(), targs.size()),
            false_target->block, Span<Value* const>(fargs.data(), fargs.size()));
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_br_if", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassStatus brass_build_ret(BrassBuilder b, BrassValue val) {
    if (!b) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        if (val && val->val) {
            b->builder.build_ret(val->val);
        } else {
            b->builder.build_ret_void();
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_ret", e);
        return BRASS_ERR_GENERIC;
    }
}

BrassStatus brass_build_unreachable(BrassBuilder b) {
    if (!b) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        b->builder.build_unreachable();
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_unreachable", e);
        return BRASS_ERR_GENERIC;
    }
}

/* Calls & Function Pointers */
BrassValue brass_build_call(BrassBuilder b, const char* callee, BrassType return_type, const BrassValue* args, size_t arg_count) {
    if (!b || !callee || !return_type) return nullptr;
    try {
        std::vector<Value*> vargs;
        vargs.reserve(arg_count);
        for (size_t i = 0; i < arg_count; ++i) {
            if (!args || !args[i] || !args[i]->val) return nullptr;
            vargs.push_back(args[i]->val);
        }
        Value* res = b->builder.build_call(callee, return_type->type, Span<Value* const>(vargs.data(), vargs.size()));
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_call", e);
        return nullptr;
    }
}

BrassValue brass_build_func_addr(BrassBuilder b, const char* name) {
    if (!b || !name) return nullptr;
    try {
        Value* res = b->builder.build_func_addr(name);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_func_addr", e);
        return nullptr;
    }
}

/* Memory */
BrassValue brass_build_load(BrassBuilder b, BrassType type, BrassValue base, int32_t offset) {
    if (!b || !type || !base || !base->val) return nullptr;
    try {
        Value* res = (offset == 0)
            ? b->builder.build_load(type->type, base->val)
            : b->builder.build_load(type->type, base->val, offset);
        return b->ctx ? b->ctx->wrap_value(res) : nullptr;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_load", e);
        return nullptr;
    }
}

BrassStatus brass_build_store(BrassBuilder b, BrassType type, BrassValue base, int32_t offset, BrassValue val) {
    if (!b || !type || !base || !base->val || !val || !val->val) return BRASS_ERR_INVALID_ARGUMENT;
    try {
        if (offset == 0) {
            b->builder.build_store(type->type, base->val, val->val);
        } else {
            b->builder.build_store(type->type, base->val, offset, val->val);
        }
        return BRASS_OK;
    } catch (const std::exception& e) {
        set_ctx_exception(b->ctx, "brass_build_store", e);
        return BRASS_ERR_GENERIC;
    }
}

} /* extern "C" */
