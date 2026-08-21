#include <brass/mir/opcodes.hpp>
#include <ostream>

namespace brass {

std::string_view opcode_name(Opcode op) noexcept {
    switch (op) {
        case Opcode::iconst_i32: return "iconst_i32";
        case Opcode::iconst_i64: return "iconst_i64";
        case Opcode::fconst_f64: return "fconst_f64";
        case Opcode::patchable_const_i32: return "patchable_const_i32";
        case Opcode::patchable_const_i64: return "patchable_const_i64";

        case Opcode::sext_i64: return "sext_i64";
        case Opcode::zext_i64: return "zext_i64";
        case Opcode::trunc_i32: return "trunc_i32";
        case Opcode::fptosi_i32: return "fptosi_i32";
        case Opcode::fptosi_i64: return "fptosi_i64";
        case Opcode::sitofp_f64_i32: return "sitofp_f64_i32";
        case Opcode::sitofp_f64_i64: return "sitofp_f64_i64";
        case Opcode::bitcast_i64_f64: return "bitcast_i64_f64";
        case Opcode::bitcast_f64_i64: return "bitcast_f64_i64";

        case Opcode::add: return "add";
        case Opcode::sub: return "sub";
        case Opcode::mul: return "mul";
        case Opcode::sdiv: return "sdiv";
        case Opcode::udiv: return "udiv";
        case Opcode::smod: return "smod";
        case Opcode::umod: return "umod";
        case Opcode::neg: return "neg";
        case Opcode::and_: return "and";
        case Opcode::or_: return "or";
        case Opcode::xor_: return "xor";
        case Opcode::shl: return "shl";
        case Opcode::lshr: return "lshr";
        case Opcode::ashr: return "ashr";
        case Opcode::not_: return "not";
        case Opcode::clz: return "clz";
        case Opcode::ctz: return "ctz";
        case Opcode::popcnt: return "popcnt";

        case Opcode::eq: return "eq";
        case Opcode::ne: return "ne";
        case Opcode::slt: return "slt";
        case Opcode::ult: return "ult";
        case Opcode::sle: return "sle";
        case Opcode::ule: return "ule";
        case Opcode::sgt: return "sgt";
        case Opcode::ugt: return "ugt";
        case Opcode::sge: return "sge";
        case Opcode::uge: return "uge";

        case Opcode::select: return "select";

        case Opcode::load: return "load";
        case Opcode::store: return "store";
        case Opcode::load_indexed: return "load_indexed";
        case Opcode::store_indexed: return "store_indexed";

        case Opcode::call: return "call";
        case Opcode::call_indirect: return "call_indirect";
        case Opcode::patchable_call: return "patchable_call";
        case Opcode::safepoint: return "safepoint";

        case Opcode::guard: return "guard";
        case Opcode::resume_point: return "resume_point";

        case Opcode::br: return "br";
        case Opcode::br_if: return "br_if";
        case Opcode::ret: return "ret";
        case Opcode::unreachable: return "unreachable";
    }
    return "unknown";
}

bool is_terminator(Opcode op) noexcept {
    switch (op) {
        case Opcode::br:
        case Opcode::br_if:
        case Opcode::ret:
        case Opcode::unreachable:
            return true;
        default:
            return false;
    }
}

bool is_branch(Opcode op) noexcept {
    return op == Opcode::br || op == Opcode::br_if;
}

bool is_call(Opcode op) noexcept {
    return op == Opcode::call || op == Opcode::call_indirect || op == Opcode::patchable_call;
}

bool is_constant(Opcode op) noexcept {
    switch (op) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64:
            return true;
        default:
            return false;
    }
}

bool is_conversion(Opcode op) noexcept {
    switch (op) {
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
            return true;
        default:
            return false;
    }
}

bool is_arithmetic(Opcode op) noexcept {
    switch (op) {
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod:
        case Opcode::neg:
            return true;
        default:
            return false;
    }
}

bool is_bitwise(Opcode op) noexcept {
    switch (op) {
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
            return true;
        default:
            return false;
    }
}

bool is_comparison(Opcode op) noexcept {
    switch (op) {
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
            return true;
        default:
            return false;
    }
}

bool is_memory(Opcode op) noexcept {
    switch (op) {
        case Opcode::load:
        case Opcode::store:
        case Opcode::load_indexed:
        case Opcode::store_indexed:
            return true;
        default:
            return false;
    }
}

bool is_select(Opcode op) noexcept {
    return op == Opcode::select;
}

bool has_side_effects(Opcode op) noexcept {
    if (is_terminator(op)) return true;
    if (is_call(op)) return true;
    switch (op) {
        case Opcode::store:
        case Opcode::store_indexed:
        case Opcode::safepoint:
        case Opcode::guard:
        case Opcode::resume_point:
            return true;
        default:
            return false;
    }
}

std::ostream& operator<<(std::ostream& os, Opcode op) {
    return os << opcode_name(op);
}

} // namespace brass
