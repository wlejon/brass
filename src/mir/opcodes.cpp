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
        case Opcode::fma_f32: return "fma_f32";
        case Opcode::fma_f64: return "fma_f64";
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

        case Opcode::sadd_overflow: return "sadd_overflow";
        case Opcode::ssub_overflow: return "ssub_overflow";
        case Opcode::smul_overflow: return "smul_overflow";
        case Opcode::uadd_overflow: return "uadd_overflow";
        case Opcode::usub_overflow: return "usub_overflow";
        case Opcode::umul_overflow: return "umul_overflow";

        case Opcode::select: return "select";

        case Opcode::load: return "load";
        case Opcode::store: return "store";
        case Opcode::load_indexed: return "load_indexed";
        case Opcode::store_indexed: return "store_indexed";
        case Opcode::write_barrier: return "write_barrier";

        case Opcode::call: return "call";
        case Opcode::call_indirect: return "call_indirect";
        case Opcode::patchable_call: return "patchable_call";
        case Opcode::safepoint: return "safepoint";

        case Opcode::guard: return "guard";
        case Opcode::resume_point: return "resume_point";
        case Opcode::osr_entry: return "osr_entry";

        case Opcode::br: return "br";
        case Opcode::br_if: return "br_if";
        case Opcode::switch_: return "switch";
        case Opcode::ret: return "ret";
        case Opcode::unreachable: return "unreachable";
        case Opcode::throw_: return "throw";
        case Opcode::invoke: return "invoke";
        case Opcode::landing_pad: return "landing_pad";
        case Opcode::resume: return "resume";

        case Opcode::vadd: return "vadd";
        case Opcode::vsub: return "vsub";
        case Opcode::vmul: return "vmul";
        case Opcode::vdiv: return "vdiv";
        case Opcode::vfma: return "vfma";
        case Opcode::vneg: return "vneg";
        case Opcode::vmin: return "vmin";
        case Opcode::vmax: return "vmax";
        case Opcode::vsqrt: return "vsqrt";
        case Opcode::vand: return "vand";
        case Opcode::vor: return "vor";
        case Opcode::vxor: return "vxor";
        case Opcode::vnot: return "vnot";
        case Opcode::vload: return "vload";
        case Opcode::vstore: return "vstore";
        case Opcode::vbroadcast: return "vbroadcast";
        case Opcode::vextract_lane: return "vextract_lane";
        case Opcode::vinsert_lane: return "vinsert_lane";
        case Opcode::vshuffle: return "vshuffle";
        case Opcode::vzero: return "vzero";

        case Opcode::coro_create: return "coro_create";
        case Opcode::coro_suspend: return "coro_suspend";
        case Opcode::coro_resume: return "coro_resume";
        case Opcode::coro_destroy: return "coro_destroy";
    }
    return "unknown";
}

bool is_terminator(Opcode op) noexcept {
    switch (op) {
        case Opcode::br:
        case Opcode::br_if:
        case Opcode::switch_:
        case Opcode::ret:
        case Opcode::unreachable:
        case Opcode::throw_:
        case Opcode::invoke:
        case Opcode::resume:
            return true;
        default:
            return false;
    }
}

bool is_branch(Opcode op) noexcept {
    return op == Opcode::br || op == Opcode::br_if || op == Opcode::switch_ || op == Opcode::invoke;
}

bool is_call(Opcode op) noexcept {
    return op == Opcode::call || op == Opcode::call_indirect || op == Opcode::patchable_call || op == Opcode::invoke;
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
        case Opcode::fma_f32:
        case Opcode::fma_f64:
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod:
        case Opcode::neg:
        case Opcode::sadd_overflow:
        case Opcode::ssub_overflow:
        case Opcode::smul_overflow:
        case Opcode::uadd_overflow:
        case Opcode::usub_overflow:
        case Opcode::umul_overflow:
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
        case Opcode::vload:
        case Opcode::vstore:
            return true;
        default:
            return false;
    }
}

bool is_write_barrier(Opcode op) noexcept {
    return op == Opcode::write_barrier;
}

bool is_select(Opcode op) noexcept {
    return op == Opcode::select;
}

bool is_vector_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vfma:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vload:
        case Opcode::vstore:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vshuffle:
        case Opcode::vzero:
            return true;
        default:
            return false;
    }
}

bool is_coro_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::coro_create:
        case Opcode::coro_suspend:
        case Opcode::coro_resume:
        case Opcode::coro_destroy:
            return true;
        default:
            return false;
    }
}

bool is_coro_suspend(Opcode op) noexcept {
    return op == Opcode::coro_suspend;
}

bool is_coro_resume(Opcode op) noexcept {
    return op == Opcode::coro_resume;
}

bool is_osr_entry(Opcode op) noexcept {
    return op == Opcode::osr_entry;
}

bool has_side_effects(Opcode op) noexcept {
    if (is_terminator(op)) return true;
    if (is_call(op)) return true;
    if (is_coro_op(op)) return true;
    switch (op) {
        case Opcode::store:
        case Opcode::store_indexed:
        case Opcode::vstore:
        case Opcode::write_barrier:
        case Opcode::safepoint:
        case Opcode::guard:
        case Opcode::resume_point:
        case Opcode::osr_entry:
        case Opcode::landing_pad:
            return true;
        default:
            return false;
    }
}

std::ostream& operator<<(std::ostream& os, Opcode op) {
    return os << opcode_name(op);
}

} // namespace brass
