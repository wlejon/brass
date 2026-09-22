// Subscript and pairwise dependence queries for loop parallelization. The
// pass itself decides legality in loop_parallel_analysis.cpp; these are the
// standalone per-access queries its reporting API exposes.

#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <sstream>
#include <cstdlib>

namespace brass {

std::string ParallelLoopStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Parallel Loop Statistics ===\n"
       << "  Loops analyzed: " << loops_analyzed << "\n"
       << "  DOALL loops found: " << doall_loops_found << "\n"
       << "  Reduction loops found: " << reduction_loops_found << "\n"
       << "  Loops transformed: " << parallel_loops_transformed << "\n"
       << "  Rejected (carried dependence): " << loops_rejected_carried_dependence << "\n"
       << "  Rejected (uncontrolled effects): " << loops_rejected_uncontrolled_effects << "\n"
       << "  Rejected (below cost threshold): " << loops_rejected_cost << "\n";
    return ss.str();
}

namespace {

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

bool decompose_affine_expr(
    const Value* val,
    const Value* iv_val,
    const Value*& out_base,
    int64_t& out_stride,
    int64_t& out_offset
) {
    if (!val) return false;
    if (val == iv_val) {
        out_stride += 1;
        return true;
    }
    int64_t c = 0;
    if (get_const_int(val, c)) {
        out_offset += c;
        return true;
    }
    if (!val->is_instruction()) {
        if (!out_base) {
            out_base = val;
            return true;
        }
        return false;
    }
    const Instruction* inst = val->defining_instruction();
    if (!inst) return false;

    if (inst->opcode() == Opcode::add) {
        return decompose_affine_expr(inst->operand(0), iv_val, out_base, out_stride, out_offset) &&
               decompose_affine_expr(inst->operand(1), iv_val, out_base, out_stride, out_offset);
    }
    if (inst->opcode() == Opcode::sub) {
        int64_t rhs_stride = 0, rhs_offset = 0;
        const Value* rhs_base = nullptr;
        if (!decompose_affine_expr(inst->operand(0), iv_val, out_base, out_stride, out_offset)) return false;
        if (!decompose_affine_expr(inst->operand(1), iv_val, rhs_base, rhs_stride, rhs_offset)) return false;
        if (rhs_base != nullptr) return false;
        out_stride -= rhs_stride;
        out_offset -= rhs_offset;
        return true;
    }
    if (inst->opcode() == Opcode::mul) {
        int64_t k = 0;
        if (inst->operand(0) == iv_val && get_const_int(inst->operand(1), k)) {
            out_stride += k;
            return true;
        }
        if (inst->operand(1) == iv_val && get_const_int(inst->operand(0), k)) {
            out_stride += k;
            return true;
        }
        return false;
    }
    if (inst->opcode() == Opcode::shl) {
        int64_t shift = 0;
        if (inst->operand(0) == iv_val && get_const_int(inst->operand(1), shift) && shift >= 0 && shift < 63) {
            out_stride += (1LL << shift);
            return true;
        }
        return false;
    }
    if (!out_base) {
        out_base = val;
        return true;
    }
    return false;
}

} // namespace

bool parse_subscript_expression(
    const Instruction* inst,
    const Value* iv_val,
    SubscriptExpr& out_expr
) {
    if (!inst || !iv_val) return false;
    out_expr.is_valid = false;
    Opcode op = inst->opcode();

    if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
        out_expr.base = inst->operand(0);
        out_expr.scale = inst->scale() > 0 ? inst->scale() : 1;
        out_expr.offset = inst->offset();
        out_expr.iv = const_cast<Value*>(iv_val);

        const Value* idx = inst->operand(1);
        const Value* dummy_base = nullptr;
        int64_t idx_stride = 0;
        int64_t idx_offset = 0;

        if (decompose_affine_expr(idx, iv_val, dummy_base, idx_stride, idx_offset) && dummy_base == nullptr) {
            out_expr.stride = idx_stride * static_cast<int64_t>(out_expr.scale);
            out_expr.offset += idx_offset * static_cast<int64_t>(out_expr.scale);
            out_expr.is_valid = true;
            return true;
        }
    } else if (op == Opcode::load || op == Opcode::store) {
        out_expr.scale = 1;
        out_expr.offset = inst->offset();
        out_expr.iv = const_cast<Value*>(iv_val);

        const Value* ptr = inst->operand(0);
        int64_t ptr_stride = 0;
        int64_t ptr_offset = 0;
        const Value* base = nullptr;

        if (decompose_affine_expr(ptr, iv_val, base, ptr_stride, ptr_offset)) {
            out_expr.base = const_cast<Value*>(base);
            out_expr.stride = ptr_stride;
            out_expr.offset += ptr_offset;
            out_expr.is_valid = true;
            return true;
        }
    }

    return false;
}

ParallelDependence check_subscript_dependence(
    const ParallelMemAccess& a1,
    const ParallelMemAccess& a2,
    const AliasAnalysis* aa
) {
    ParallelDependence dep;
    dep.src = a1.inst;
    dep.dst = a2.inst;

    auto unknown = [&dep]() {
        dep.dir = DependenceDir::Any;
        dep.is_loop_carried = true;
        dep.has_distance = false;
        return dep;
    };

    // Read-After-Read is never a data conflict
    if (!a1.is_store && !a2.is_store) {
        dep.kind = DependenceKind::None;
        dep.is_loop_carried = false;
        return dep;
    }

    if (a1.is_store && !a2.is_store) {
        dep.kind = DependenceKind::RAW;
    } else if (!a1.is_store && a2.is_store) {
        dep.kind = DependenceKind::WAR;
    } else {
        dep.kind = DependenceKind::WAW;
    }

    if (!a1.expr.is_valid || !a2.expr.is_valid) return unknown();

    // Different bases: only provably different objects are independent.
    if (a1.expr.base != a2.expr.base) {
        if (a1.expr.base && a2.expr.base && aa &&
            aa->alias(a1.expr.base, a2.expr.base) == AliasResult::NoAlias) {
            dep.kind = DependenceKind::None;
            dep.is_loop_carried = false;
            return dep;
        }
        return unknown();
    }

    const int64_t stride = a1.expr.stride;
    if (stride != a2.expr.stride || stride == 0) return unknown();

    const int64_t diff = a2.expr.offset - a1.expr.offset;
    if (diff % stride == 0) {
        const int64_t dist = diff / stride;
        dep.distance = dist;
        dep.has_distance = true;
        dep.is_loop_carried = dist != 0;
        dep.dir = dist == 0 ? DependenceDir::Equal : (dist > 0 ? DependenceDir::Forward : DependenceDir::Backward);
        return dep;
    }

    // Incongruent offsets keep the two streams apart only when, within one
    // stride period, the second access's bytes fall entirely in the gap
    // after the first access's bytes.
    const int64_t period = std::llabs(stride);
    const int64_t size1 = static_cast<int64_t>(a1.elem_type.size_in_bytes());
    const int64_t size2 = static_cast<int64_t>(a2.elem_type.size_in_bytes());
    const int64_t r = ((diff % period) + period) % period;
    if (size1 > 0 && size2 > 0 && r >= size1 && r + size2 <= period) {
        dep.kind = DependenceKind::None;
        dep.is_loop_carried = false;
        return dep;
    }
    return unknown();
}

} // namespace brass
