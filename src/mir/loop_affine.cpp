#include "loop_affine.hpp"
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/opcodes.hpp>
#include <limits>

namespace brass::affine {

bool checked_add(int64_t a, int64_t b, int64_t& out) noexcept {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    if ((b > 0 && a > kMax - b) || (b < 0 && a < kMin - b)) return false;
    out = a + b;
    return true;
}

bool checked_sub(int64_t a, int64_t b, int64_t& out) noexcept {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    if ((b < 0 && a > kMax + b) || (b > 0 && a < kMin + b)) return false;
    out = a - b;
    return true;
}

bool checked_mul(int64_t a, int64_t b, int64_t& out) noexcept {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if ((a == -1 && b == kMin) || (b == -1 && a == kMin)) return false;
    if (a > 0 ? (b > 0 ? a > kMax / b : b < kMin / a)
              : (b > 0 ? a < kMin / b : a < kMax / b)) {
        return false;
    }
    out = a * b;
    return true;
}

namespace {

constexpr int kMaxDepth = 24;

// A linear combination under construction (see Form).
struct Lin {
    std::map<const Value*, int64_t> inv;
    std::map<std::pair<const Value*, const Value*>, int64_t> terms;
    int64_t c = 0;
};

bool add_to(int64_t& dst, int64_t v) { return checked_add(dst, v, dst); }
bool mul_ok(int64_t a, int64_t b, int64_t& out) { return checked_mul(a, b, out); }

bool accumulate(Lin& dst, const Lin& src, int64_t factor) {
    int64_t t = 0;
    for (const auto& [v, k] : src.inv) {
        if (!mul_ok(k, factor, t) || !add_to(dst.inv[v], t)) return false;
    }
    for (const auto& [key, k] : src.terms) {
        if (!mul_ok(k, factor, t) || !add_to(dst.terms[key], t)) return false;
    }
    return mul_ok(src.c, factor, t) && add_to(dst.c, t);
}

void prune(Lin& l) {
    for (auto it = l.inv.begin(); it != l.inv.end();) it = it->second == 0 ? l.inv.erase(it) : std::next(it);
    for (auto it = l.terms.begin(); it != l.terms.end();) it = it->second == 0 ? l.terms.erase(it) : std::next(it);
}

bool is_wide(Type t) noexcept { return t == Type::i64() || t.is_pointer_or_gcref(); }

struct Decomposer {
    const std::vector<const Value*>& ivs;
    const InvariantFn& invariant;

    bool is_iv(const Value* v) const {
        for (const Value* iv : ivs) if (iv == v) return true;
        return false;
    }

    bool run(const Value* v, Lin& out, int depth) const {
        if (!v || depth > kMaxDepth) return false;
        int64_t k = 0;
        if (is_int_constant(v, k)) {
            out.c = k;
            return true;
        }
        if (is_iv(v)) {
            out.terms[{v, nullptr}] = 1;
            return true;
        }
        if (invariant(v)) {
            out.inv[v] = 1;
            return true;
        }
        if (!v->is_instruction()) return false;
        const Instruction* inst = v->defining_instruction();
        if (!inst) return false;
        const Opcode op = inst->opcode();
        if (op == Opcode::sext_i64) {
            // An induction variable never wraps inside its loop, so widening
            // it is exact; widening anything else could hide a 32-bit wrap.
            const Value* x = inst->operand(0);
            if (!is_iv(x)) return false;
            out.terms[{x, nullptr}] = 1;
            return true;
        }
        if (!is_wide(inst->type())) return false;
        if (op == Opcode::add || op == Opcode::sub) {
            Lin a, b;
            if (!run(inst->operand(0), a, depth + 1) || !run(inst->operand(1), b, depth + 1)) return false;
            return accumulate(out, a, 1) && accumulate(out, b, op == Opcode::add ? 1 : -1);
        }
        if (op == Opcode::shl) {
            int64_t s = 0;
            if (!is_int_constant(inst->operand(1), s) || s < 0 || s > 62) return false;
            Lin a;
            return run(inst->operand(0), a, depth + 1) && accumulate(out, a, int64_t{1} << s);
        }
        if (op == Opcode::mul) {
            const Value* lhs = inst->operand(0);
            const Value* rhs = inst->operand(1);
            if (is_int_constant(lhs, k)) std::swap(lhs, rhs);
            if (is_int_constant(rhs, k)) {
                Lin a;
                return run(lhs, a, depth + 1) && accumulate(out, a, k);
            }
            // iv-affine * invariant: each coefficient gains the invariant as
            // a symbolic factor.
            if (invariant(lhs)) std::swap(lhs, rhs);
            if (!invariant(rhs)) return false;
            Lin a;
            if (!run(lhs, a, depth + 1) || !a.inv.empty()) return false;
            for (const auto& [key, coeff] : a.terms) {
                if (key.second != nullptr) return false;
                if (!add_to(out.terms[{key.first, rhs}], coeff)) return false;
            }
            return a.c == 0 || add_to(out.inv[rhs], a.c);
        }
        return false;
    }
};

} // namespace

bool is_plain_memory_access(Opcode op) noexcept {
    return op == Opcode::load || op == Opcode::store || op == Opcode::load_indexed || op == Opcode::store_indexed ||
           op == Opcode::vload || op == Opcode::vstore;
}

bool is_plain_store(Opcode op) noexcept {
    return op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore;
}

bool is_int_constant(const Value* v, int64_t& out) noexcept {
    if (!v || !v->is_instruction() || !v->defining_instruction()) return false;
    const Instruction* def = v->defining_instruction();
    if (def->opcode() == Opcode::iconst_i32) {
        out = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out = def->imm_i64();
        return true;
    }
    return false;
}

bool is_plain_constant(const Value* v) noexcept {
    if (!v || !v->is_instruction() || !v->defining_instruction()) return false;
    const Opcode op = v->defining_instruction()->opcode();
    return op == Opcode::iconst_i32 || op == Opcode::iconst_i64 || op == Opcode::fconst_f64;
}

Form address_form(const Instruction& mem, const std::vector<const Value*>& ivs, const InvariantFn& invariant) {
    Form f;
    const Opcode op = mem.opcode();
    if (!is_plain_memory_access(op)) return f;
    f.is_store = is_plain_store(op);
    f.size = static_cast<uint32_t>(mem.memory_type().size_in_bytes());
    f.pointer = mem.operand(0);
    if (f.size == 0) return f;

    Decomposer d{ivs, invariant};
    Lin lin;
    if (!d.run(mem.operand(0), lin, 0)) return f;
    if (op == Opcode::load_indexed || op == Opcode::store_indexed) {
        Lin idx;
        if (!d.run(mem.operand(1), idx, 0)) return f;
        const int64_t scale = mem.scale() > 0 ? mem.scale() : 1;
        if (!accumulate(lin, idx, scale)) return f;
    }
    if (!add_to(lin.c, mem.offset())) return f;
    prune(lin);
    f.invariants = std::move(lin.inv);
    f.terms = std::move(lin.terms);
    f.constant = lin.c;
    f.ok = true;
    return f;
}

bool same_form(const Form& a, const Form& b) noexcept {
    return a.ok && b.ok && a.size == b.size && a.constant == b.constant && a.invariants == b.invariants &&
           a.terms == b.terms;
}

bool distinct_objects(const AliasAnalysis& aa, const Value* p1, const Value* p2) {
    if (!p1 || !p2) return false;
    int64_t o1 = 0, o2 = 0;
    const Value* b1 = aa.get_underlying_base(p1, o1);
    const Value* b2 = aa.get_underlying_base(p2, o2);
    if (b1 && b1 == b2) return false;
    return aa.alias(p1, p2) == AliasResult::NoAlias;
}

bool is_pure_nontrapping(const Instruction& inst) noexcept {
    switch (inst.opcode()) {
        case Opcode::iconst_i32: case Opcode::iconst_i64: case Opcode::fconst_f64:
        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32: case Opcode::trunc_i8:
        case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64: case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f32_i64: case Opcode::fptrunc_f32_f64: case Opcode::fpext_f64_f32:
        case Opcode::bitcast_i64_f64: case Opcode::bitcast_f64_i64:
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::neg:
        case Opcode::fma_f32: case Opcode::fma_f64: case Opcode::sqrt_f32: case Opcode::sqrt_f64:
        case Opcode::floor_f32: case Opcode::floor_f64: case Opcode::ceil_f32: case Opcode::ceil_f64:
        case Opcode::round_f32: case Opcode::round_f64: case Opcode::fabs_f32: case Opcode::fabs_f64:
        case Opcode::fmin_f32: case Opcode::fmin_f64: case Opcode::fmax_f32: case Opcode::fmax_f64:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_: case Opcode::shl: case Opcode::lshr:
        case Opcode::ashr: case Opcode::not_: case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
        case Opcode::eq: case Opcode::ne: case Opcode::slt: case Opcode::ult: case Opcode::sle:
        case Opcode::ule: case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
        case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow:
        case Opcode::select:
            return !inst.type().is_vector();
        case Opcode::vadd: case Opcode::vsub: case Opcode::vmul: case Opcode::vfma: case Opcode::vneg:
        case Opcode::vmin: case Opcode::vmax: case Opcode::vsqrt: case Opcode::vand: case Opcode::vor:
        case Opcode::vxor: case Opcode::vnot: case Opcode::vbroadcast: case Opcode::vextract_lane:
        case Opcode::vinsert_lane: case Opcode::vshuffle: case Opcode::vzero:
            return true;
        case Opcode::vdiv:
            // Lane-wise integer division could divide by zero.
            return inst.type().is_vector() && inst.type().element_type().is_float();
        case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod: {
            // Division by zero is a program error: only a known non-zero
            // divisor makes the instruction safe to repeat or reorder.
            int64_t d = 0;
            return !inst.type().is_vector() && is_int_constant(inst.operand(1), d) && d != 0;
        }
        default:
            return false;
    }
}

bool defined_outside(const LoopInfo& loop, const Value* v) noexcept {
    if (!v) return false;
    if (v->is_block_param()) return !loop.contains(v->defining_block());
    if (v->is_instruction()) {
        const Instruction* def = v->defining_instruction();
        return def && def->parent() && !loop.contains(def->parent());
    }
    return false;
}

bool used_outside(const Function& fn, const LoopInfo& loop, const Value* v) {
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || loop.contains(bb)) continue;
        for (Instruction* inst : *bb) {
            if (inst && uses_value(*inst, v)) return true;
        }
    }
    return false;
}

} // namespace brass::affine
