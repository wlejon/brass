#include "gvn_table.hpp"

namespace brass {

bool is_commutative_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::add:
        case Opcode::mul:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::vadd:
        case Opcode::vmul:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
            return true;
        default:
            return false;
    }
}

bool is_pure_gvn_op(const Instruction* inst) noexcept {
    if (!inst || !inst->produces_value()) return false;
    Opcode op = inst->opcode();
    if (op == Opcode::call) {
        return inst->symbol() == "bronze_tls_block_addr";
    }
    switch (op) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::select:
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vshuffle:
        case Opcode::vzero:
            return true;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod: {
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            const Value* denom = inst->operand(1);
            if (denom->is_instruction()) {
                const Instruction* ddef = denom->defining_instruction();
                if (ddef && (ddef->opcode() == Opcode::iconst_i32 || ddef->opcode() == Opcode::iconst_i64)) {
                    return ddef->imm_i64() != 0;
                }
            }
            return false;
        }
        default:
            return false;
    }
}

GvnExpression GvnExpression::from_instruction(const Instruction* inst) {
    GvnExpression expr;
    if (!inst) return expr;

    expr.opcode = inst->opcode();
    expr.type = inst->type();
    expr.offset = inst->offset();
    expr.scale = inst->scale();
    expr.symbol = inst->symbol();

    if (inst->opcode() == Opcode::fconst_f64) {
        double f = inst->imm_f64();
        std::memcpy(&expr.imm_bits, &f, sizeof(double));
    } else {
        expr.imm_bits = static_cast<uint64_t>(inst->imm_i64());
    }

    size_t op_count = inst->operand_count();
    if (op_count >= 1) expr.op0 = inst->operand(0);
    if (op_count >= 2) expr.op1 = inst->operand(1);
    if (op_count >= 3) expr.op2 = inst->operand(2);

    // Canonicalize commutative expressions
    if (is_commutative_op(expr.opcode) && expr.op0 && expr.op1) {
        if (expr.op0->id() > expr.op1->id()) {
            std::swap(expr.op0, expr.op1);
        }
    }

    return expr;
}

bool GvnExpression::operator==(const GvnExpression& other) const noexcept {
    return opcode == other.opcode &&
           type == other.type &&
           op0 == other.op0 &&
           op1 == other.op1 &&
           op2 == other.op2 &&
           imm_bits == other.imm_bits &&
           offset == other.offset &&
           scale == other.scale &&
           symbol == other.symbol;
}

size_t GvnExprHash::operator()(const GvnExpression& k) const noexcept {
    size_t h = static_cast<size_t>(k.opcode);
    h = h * 31 + static_cast<size_t>(k.type.kind());
    h = h * 31 + std::hash<const void*>()(k.op0);
    h = h * 31 + std::hash<const void*>()(k.op1);
    h = h * 31 + std::hash<const void*>()(k.op2);
    h = h * 31 + static_cast<size_t>(k.imm_bits);
    h = h * 31 + static_cast<size_t>(k.offset);
    h = h * 31 + static_cast<size_t>(k.scale);
    h = h * 31 + std::hash<std::string_view>()(k.symbol);
    return h;
}

void GvnTable::enter_scope() {
    scopes_.emplace_back();
}

void GvnTable::exit_scope() {
    if (scopes_.empty()) return;
    const ScopeFrame& frame = scopes_.back();
    for (const auto& expr : frame.exprs) {
        expr_map_.erase(expr);
    }
    for (const auto& load : frame.loads) {
        load_map_.erase(load);
    }
    scopes_.pop_back();
}

Value* GvnTable::lookup_expression(const GvnExpression& expr) const {
    auto it = expr_map_.find(expr);
    return (it != expr_map_.end()) ? it->second : nullptr;
}

void GvnTable::insert_expression(const GvnExpression& expr, Value* val) {
    if (scopes_.empty()) {
        enter_scope();
    }
    auto res = expr_map_.emplace(expr, val);
    if (res.second) {
        scopes_.back().exprs.push_back(expr);
    }
}

Value* GvnTable::lookup_load(const AvailableLoadKey& key) const {
    auto it = load_map_.find(key);
    return (it != load_map_.end()) ? it->second : nullptr;
}

void GvnTable::insert_load(const AvailableLoadKey& key, Value* val) {
    if (scopes_.empty()) {
        enter_scope();
    }
    auto res = load_map_.emplace(key, val);
    if (res.second) {
        scopes_.back().loads.push_back(key);
    }
}

} // namespace brass
