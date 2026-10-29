#include "verifier_vec.hpp"
#include <string>

namespace brass {

bool verify_vector_instruction(
    const Instruction* inst,
    const std::string& inst_prefix,
    const std::function<void(const std::string&)>& report_error
) {
    if (!inst) return false;
    Opcode op = inst->opcode();

    switch (op) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vmin:
        case Opcode::vmax: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            Type t1 = inst->operand(1)->type();
            if (!t0.is_vector() || !t1.is_vector()) {
                report_error(inst_prefix + "Operands must be vector types.");
                return false;
            }
            if (t0 != t1) {
                report_error(inst_prefix + "Vector operand types mismatch: " +
                             std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                             ") must match operand type (" + std::string(t0.name()) + ").");
                return false;
            }
            return true;
        }

        case Opcode::vfma: {
            if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                report_error(inst_prefix + "Requires 3 operands.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            Type t1 = inst->operand(1)->type();
            Type t2 = inst->operand(2)->type();
            if (!t0.is_vector() || !t1.is_vector() || !t2.is_vector()) {
                report_error(inst_prefix + "Operands must be vector types.");
                return false;
            }
            if (t0 != t1 || t0 != t2) {
                report_error(inst_prefix + "Vector operand types mismatch.");
                return false;
            }
            if (!t0.element_type().is_float()) {
                report_error(inst_prefix + "vfma operands must be float vector types.");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                             ") must match operand type (" + std::string(t0.name()) + ").");
                return false;
            }
            return true;
        }

        case Opcode::vneg: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector()) {
                report_error(inst_prefix + "Operand must be vector type.");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type must match operand type.");
                return false;
            }
            return true;
        }

        case Opcode::vsqrt: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector() || !t0.element_type().is_float()) {
                report_error(inst_prefix + "Operand must be float vector type.");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type must match operand type.");
                return false;
            }
            return true;
        }

        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            Type t1 = inst->operand(1)->type();
            if (!t0.is_vector() || !t1.is_vector()) {
                report_error(inst_prefix + "Operands must be vector types.");
                return false;
            }
            if (t0 != t1) {
                report_error(inst_prefix + "Vector operand types mismatch: " +
                             std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type must match operand type.");
                return false;
            }
            return true;
        }

        case Opcode::vnot: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector()) {
                report_error(inst_prefix + "Operand must be vector type.");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type must match operand type.");
                return false;
            }
            return true;
        }

        case Opcode::vload: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand (base pointer).");
                return false;
            }
            Type base_t = inst->operand(0)->type();
            if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                report_error(inst_prefix + "Base operand must be pointer or gcref.");
                return false;
            }
            if (!inst->memory_type().is_vector()) {
                report_error(inst_prefix + "Memory type must be a vector type.");
                return false;
            }
            if (inst->type() != inst->memory_type()) {
                report_error(inst_prefix + "Result type must match memory type.");
                return false;
            }
            return true;
        }

        case Opcode::vstore: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands (base pointer and value).");
                return false;
            }
            Type base_t = inst->operand(0)->type();
            if (!base_t.is_pointer_or_gcref() && base_t != Type::i64()) {
                report_error(inst_prefix + "Base operand must be pointer or gcref.");
                return false;
            }
            if (!inst->memory_type().is_vector()) {
                report_error(inst_prefix + "Memory type must be a vector type.");
                return false;
            }
            Type val_t = inst->operand(1)->type();
            if (val_t != inst->memory_type()) {
                report_error(inst_prefix + "Stored value type (" + std::string(val_t.name()) +
                             ") must match memory type (" + std::string(inst->memory_type().name()) + ").");
                return false;
            }
            if (!inst->type().is_void()) {
                report_error(inst_prefix + "Store result type must be void.");
                return false;
            }
            return true;
        }

        case Opcode::vbroadcast: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 scalar operand.");
                return false;
            }
            if (!inst->type().is_vector()) {
                report_error(inst_prefix + "Result type must be a vector type.");
                return false;
            }
            Type elem_t = inst->type().element_type();
            Type src_t = inst->operand(0)->type();
            if (src_t != elem_t) {
                report_error(inst_prefix + "Scalar operand type (" + std::string(src_t.name()) +
                             ") must match vector element type (" + std::string(elem_t.name()) + ").");
                return false;
            }
            return true;
        }

        case Opcode::vextract_lane: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 vector operand.");
                return false;
            }
            Type vec_t = inst->operand(0)->type();
            if (!vec_t.is_vector()) {
                report_error(inst_prefix + "Operand must be a vector type.");
                return false;
            }
            uint32_t lanes = vec_t.vector_lanes();
            if (inst->lane() >= lanes) {
                report_error(inst_prefix + "Lane index " + std::to_string(inst->lane()) +
                             " out of bounds for " + std::string(vec_t.name()) + " (lanes: " + std::to_string(lanes) + ").");
                return false;
            }
            if (inst->type() != vec_t.element_type()) {
                report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                             ") must match element type (" + std::string(vec_t.element_type().name()) + ").");
                return false;
            }
            return true;
        }

        case Opcode::vinsert_lane: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands (vector, scalar).");
                return false;
            }
            Type vec_t = inst->operand(0)->type();
            Type scalar_t = inst->operand(1)->type();
            if (!vec_t.is_vector()) {
                report_error(inst_prefix + "First operand must be a vector type.");
                return false;
            }
            uint32_t lanes = vec_t.vector_lanes();
            if (inst->lane() >= lanes) {
                report_error(inst_prefix + "Lane index " + std::to_string(inst->lane()) +
                             " out of bounds for " + std::string(vec_t.name()) + " (lanes: " + std::to_string(lanes) + ").");
                return false;
            }
            if (scalar_t != vec_t.element_type()) {
                report_error(inst_prefix + "Scalar operand type (" + std::string(scalar_t.name()) +
                             ") must match element type (" + std::string(vec_t.element_type().name()) + ").");
                return false;
            }
            if (inst->type() != vec_t) {
                report_error(inst_prefix + "Result type must match input vector type.");
                return false;
            }
            return true;
        }

        case Opcode::vshuffle: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands (v1, v2).");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            Type t1 = inst->operand(1)->type();
            if (!t0.is_vector() || !t1.is_vector()) {
                report_error(inst_prefix + "Operands must be vector types.");
                return false;
            }
            if (t0 != t1) {
                report_error(inst_prefix + "Vector operand types mismatch: " +
                             std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                return false;
            }
            if (inst->type() != t0) {
                report_error(inst_prefix + "Result type must match vector operand type.");
                return false;
            }
            return true;
        }

        case Opcode::vzero: {
            if (inst->operand_count() != 0) {
                report_error(inst_prefix + "vzero takes 0 operands.");
                return false;
            }
            if (!inst->type().is_vector()) {
                report_error(inst_prefix + "Result type must be a vector type.");
                return false;
            }
            return true;
        }

        case Opcode::vcmp_eq:
        case Opcode::vcmp_ne:
        case Opcode::vcmp_lt:
        case Opcode::vcmp_le:
        case Opcode::vcmp_gt:
        case Opcode::vcmp_ge: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector() || inst->operand(1)->type() != t0) {
                report_error(inst_prefix + "Operands must be vectors of one type.");
                return false;
            }
            if (inst->type() != t0.vector_mask_type()) {
                report_error(inst_prefix + "Result type must be " +
                             std::string(t0.vector_mask_type().name()) + ".");
                return false;
            }
            return true;
        }

        case Opcode::vselect: {
            if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                report_error(inst_prefix + "Requires 3 operands (mask, a, b).");
                return false;
            }
            Type t = inst->operand(1)->type();
            if (!t.is_vector() || inst->operand(2)->type() != t || inst->type() != t) {
                report_error(inst_prefix + "Values and result must be vectors of one type.");
                return false;
            }
            if (inst->operand(0)->type() != t.vector_mask_type()) {
                report_error(inst_prefix + "Mask must be " + std::string(t.vector_mask_type().name()) + ".");
                return false;
            }
            return true;
        }

        case Opcode::vshl:
        case Opcode::vlshr:
        case Opcode::vashr: {
            if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                report_error(inst_prefix + "Requires 2 operands.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector() || !t0.element_type().is_integer() ||
                inst->operand(1)->type() != t0 || inst->type() != t0) {
                report_error(inst_prefix + "Operands and result must be integer vectors of one type.");
                return false;
            }
            return true;
        }

        case Opcode::vconvert: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type from = inst->operand(0)->type();
            Type to = inst->type();
            auto pair = [&](Type a, Type b) {
                return (from == a && to == b) || (from == b && to == a);
            };
            if (!(pair(Type::i32x4(), Type::f32x4()) || pair(Type::i32x8(), Type::f32x8()) ||
                  pair(Type::i32x4(), Type::f64x4()) || pair(Type::f32x4(), Type::f64x4()) ||
                  pair(Type::i32x4(), Type::i64x4()))) {
                report_error(inst_prefix + "No vconvert from " + std::string(from.name()) +
                             " to " + std::string(to.name()) + ".");
                return false;
            }
            return true;
        }

        case Opcode::vbitcast: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type from = inst->operand(0)->type();
            if (!from.is_vector() || !inst->type().is_vector() ||
                from.size_in_bytes() != inst->type().size_in_bytes()) {
                report_error(inst_prefix + "Operand and result must be vectors of one size.");
                return false;
            }
            return true;
        }

        case Opcode::vfloor:
        case Opcode::vceil:
        case Opcode::vtrunc:
        case Opcode::vround: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector() || !t0.element_type().is_float() || inst->type() != t0) {
                report_error(inst_prefix + "Operand and result must be one float vector type.");
                return false;
            }
            return true;
        }

        case Opcode::vany: {
            if (inst->operand_count() != 1 || !inst->operand(0)) {
                report_error(inst_prefix + "Requires 1 operand.");
                return false;
            }
            Type t0 = inst->operand(0)->type();
            if (!t0.is_vector() || !t0.element_type().is_integer() || inst->type() != Type::i32()) {
                report_error(inst_prefix + "Operand must be an integer vector, the result i32.");
                return false;
            }
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass
