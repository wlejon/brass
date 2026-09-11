#include "il_lowering_ops.hpp"
#include "il_lowering.hpp"

namespace brass::il {

bool is_ops_il_op(BronzeOp op) {
    switch (op) {
        case BronzeOp::Add:
        case BronzeOp::Sub:
        case BronzeOp::Mul:
        case BronzeOp::Div:
        case BronzeOp::Mod:
        case BronzeOp::Neg:
        case BronzeOp::BitAnd:
        case BronzeOp::BitOr:
        case BronzeOp::BitXor:
        case BronzeOp::Shl:
        case BronzeOp::Shr:
        case BronzeOp::UShr:
        case BronzeOp::BitNot:
        case BronzeOp::ToInt32:
        case BronzeOp::ToNumeric:
        case BronzeOp::NumericStep:
        case BronzeOp::RelLt:
        case BronzeOp::RelLe:
        case BronzeOp::RelGt:
        case BronzeOp::RelGe:
        case BronzeOp::StrictEq:
        case BronzeOp::LooseEq:
        case BronzeOp::CmpLt:
        case BronzeOp::CmpLe:
        case BronzeOp::CmpGt:
        case BronzeOp::CmpGe:
        case BronzeOp::CmpEq:
        case BronzeOp::CmpNe:
            return true;
        default:
            return false;
    }
}

bool lower_ops_instruction(
    IlLowering* lowering,
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    Value*& res_val
) {
    Type res_type = lower_type(inst_ast.result_type);
    auto get_opd = [&](size_t idx) -> Value* {
        if (idx < inst_ast.operands.size()) {
            uint32_t id = inst_ast.operands[idx];
            if (val_map.count(id)) return val_map[id];
        }
        return nullptr;
    };

    switch (inst_ast.op) {
        case BronzeOp::Add: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64()) {
                op0 = lowering->ensure_type(op0, Type::f64(), b);
                op1 = lowering->ensure_type(op1, Type::f64(), b);
                res_val = b.build_add(op0, op1);
            } else if (res_type == Type::i64()) {
                op0 = lowering->ensure_type(op0, Type::i64(), b);
                op1 = lowering->ensure_type(op1, Type::i64(), b);
                res_val = b.build_call("bronze_dynamic_add", Type::i64(), {op0, op1});
            } else {
                op0 = lowering->ensure_type(op0, Type::i32(), b);
                op1 = lowering->ensure_type(op1, Type::i32(), b);
                res_val = b.build_add(op0, op1);
            }
            return true;
        }

        case BronzeOp::Sub:
        case BronzeOp::Mul:
        case BronzeOp::Div: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            bool is_float = (res_type == Type::f64() || res_type == Type::i64());
            op0 = lowering->ensure_type(op0, is_float ? Type::f64() : Type::i32(), b);
            op1 = lowering->ensure_type(op1, is_float ? Type::f64() : Type::i32(), b);
            Value* r = (inst_ast.op == BronzeOp::Sub) ? b.build_sub(op0, op1) :
                       (inst_ast.op == BronzeOp::Mul) ? b.build_mul(op0, op1) : b.build_sdiv(op0, op1);
            res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            return true;
        }

        case BronzeOp::Mod: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (!op0 || !op1) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = lowering->ensure_type(op0, Type::f64(), b);
                op1 = lowering->ensure_type(op1, Type::f64(), b);
                Value* r = b.build_call("bronze_f64_mod", Type::f64(), {op0, op1});
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = lowering->ensure_type(op0, Type::i32(), b);
                op1 = lowering->ensure_type(op1, Type::i32(), b);
                res_val = b.build_smod(op0, op1);
            }
            return true;
        }

        case BronzeOp::Neg: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (res_type == Type::f64() || res_type == Type::i64()) {
                op0 = lowering->ensure_type(op0, Type::f64(), b);
                Value* r = b.build_neg(op0);
                res_val = (res_type == Type::i64()) ? b.build_bitcast_i64_f64(r) : r;
            } else {
                op0 = lowering->ensure_type(op0, Type::i32(), b);
                res_val = b.build_neg(op0);
            }
            return true;
        }

        case BronzeOp::BitAnd: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_and(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::BitOr: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_or(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::BitXor: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_xor(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::Shl: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_shl(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::Shr: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_ashr(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::UShr: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i32(), b);
            Value* r = b.build_lshr(op0, op1);
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }
        case BronzeOp::BitNot: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i32(), b);
            Value* r = b.build_xor(op0, b.build_iconst_i32(-1));
            res_val = (res_type == Type::f64()) ? b.build_sitofp_f64_i32(r) : r;
            return true;
        }

        case BronzeOp::ToInt32: {
            Value* op0 = get_opd(0);
            res_val = lowering->ensure_type(op0, Type::i32(), b);
            return true;
        }
        case BronzeOp::ToNumeric: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            if (res_type == Type::f64()) {
                res_val = lowering->ensure_type(op0, Type::f64(), b);
            } else {
                Value* lhs = lowering->ensure_type(op0, Type::i64(), b);
                res_val = b.build_call("bronze_to_numeric", Type::i64(), {lhs});
            }
            return true;
        }
        case BronzeOp::NumericStep: {
            Value* op0 = get_opd(0);
            if (!op0) return false;
            Value* lhs = lowering->ensure_type(op0, Type::i64(), b);
            Value* is_inc = b.build_iconst_i32(inst_ast.imm_i64 >= 0 ? 1 : 0);
            res_val = b.build_call("bronze_numeric_step", Type::i64(), {lhs, is_inc});
            return true;
        }

        case BronzeOp::RelLt:
        case BronzeOp::RelLe:
        case BronzeOp::RelGt:
        case BronzeOp::RelGe: {
            Value* op0 = lowering->ensure_type(get_opd(0), Type::i64(), b);
            Value* op1 = lowering->ensure_type(get_opd(1), Type::i64(), b);
            const char* helper = (inst_ast.op == BronzeOp::RelLt) ? "bronze_rel_lt" :
                                 (inst_ast.op == BronzeOp::RelLe) ? "bronze_rel_le" :
                                 (inst_ast.op == BronzeOp::RelGt) ? "bronze_rel_gt" : "bronze_rel_ge";
            res_val = b.build_call(helper, Type::i32(), {op0, op1});
            res_val = b.build_and(res_val, b.build_iconst_i32(1));
            return true;
        }

        case BronzeOp::StrictEq: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() && op1->type() == Type::f64()) {
                res_val = b.build_eq(op0, op1);
            } else if (op0->type() == Type::i32() && op1->type() == Type::i32()) {
                res_val = b.build_eq(op0, op1);
            } else {
                op0 = lowering->ensure_type(op0, Type::i64(), b);
                op1 = lowering->ensure_type(op1, Type::i64(), b);
                res_val = b.build_call("bronze_strict_eq", Type::i32(), {op0, op1});
                res_val = b.build_and(res_val, b.build_iconst_i32(1));
            }
            return true;
        }

        case BronzeOp::LooseEq: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() && op1->type() == Type::f64()) {
                res_val = b.build_eq(op0, op1);
            } else if (op0->type() == Type::i32() && op1->type() == Type::i32()) {
                res_val = b.build_eq(op0, op1);
            } else {
                op0 = lowering->ensure_type(op0, Type::i64(), b);
                op1 = lowering->ensure_type(op1, Type::i64(), b);
                res_val = b.build_call("bronze_loose_eq", Type::i32(), {op0, op1});
                res_val = b.build_and(res_val, b.build_iconst_i32(1));
            }
            return true;
        }

        case BronzeOp::CmpLt:
        case BronzeOp::CmpLe:
        case BronzeOp::CmpGt:
        case BronzeOp::CmpGe:
        case BronzeOp::CmpEq:
        case BronzeOp::CmpNe: {
            Value* op0 = get_opd(0);
            Value* op1 = get_opd(1);
            if (op0->type() == Type::f64() || op1->type() == Type::f64()) {
                op0 = lowering->ensure_type(op0, Type::f64(), b);
                op1 = lowering->ensure_type(op1, Type::f64(), b);
            } else if (op0->type() == Type::i64() || op1->type() == Type::i64()) {
                op0 = lowering->ensure_type(op0, Type::i64(), b);
                op1 = lowering->ensure_type(op1, Type::i64(), b);
            } else {
                op0 = lowering->ensure_type(op0, Type::i32(), b);
                op1 = lowering->ensure_type(op1, Type::i32(), b);
            }
            if (inst_ast.op == BronzeOp::CmpLt) res_val = b.build_slt(op0, op1);
            else if (inst_ast.op == BronzeOp::CmpLe) res_val = b.build_sle(op0, op1);
            else if (inst_ast.op == BronzeOp::CmpGt) res_val = b.build_sgt(op0, op1);
            else if (inst_ast.op == BronzeOp::CmpGe) res_val = b.build_sge(op0, op1);
            else if (inst_ast.op == BronzeOp::CmpEq) res_val = b.build_eq(op0, op1);
            else res_val = b.build_ne(op0, op1);
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::il
