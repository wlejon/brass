#include <brass/mir/fdiv_pow2.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/uses.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace brass {

namespace {

// The divisor's constant, when the division is a scalar float one by a
// constant (`fconst_f64` carries an f32 constant too, typed f32).
bool const_divisor(const Instruction* inst, double& c) {
    if (inst->opcode() != Opcode::sdiv || inst->operand_count() != 2 || !inst->result()) return false;
    const Type t = inst->type();
    if (t != Type::f64() && t != Type::f32()) return false;
    const Value* d = inst->operand(1);
    const Instruction* def = d && d->is_instruction() ? d->defining_instruction() : nullptr;
    if (!def || def->opcode() != Opcode::fconst_f64 || def->type() != t) return false;
    c = def->imm_f64();
    return true;
}

}  // namespace

bool exact_pow2_reciprocal(double c, bool f32, double& recip) {
    if (!std::isfinite(c) || c == 0.0) return false;
    int e = 0;
    // A power of two (or its negation) is the one value whose frexp mantissa
    // is exactly one half.
    if (std::fabs(std::frexp(c, &e)) != 0.5) return false;
    const double r = 1.0 / c;
    // Checked rather than argued: the reciprocal must be finite, itself a
    // power of two (so the division by c and the product by r are the same
    // exact scaling), and give back c when inverted.
    int er = 0;
    if (!std::isfinite(r) || r == 0.0 || std::fabs(std::frexp(r, &er)) != 0.5 || 1.0 / r != c) return false;
    // A subnormal reciprocal (c = 2^1023) is exact as a value, but a core
    // running with denormals-as-zero would read it as 0 where the division
    // would not have; a normal one reads the same under every mode.
    if (std::fpclassify(r) != FP_NORMAL) return false;
    if (f32) {
        // Both must be floats exactly: the constant was one, and an f32
        // multiply by a reciprocal that does not fit would round it.
        const float cf = static_cast<float>(c);
        const float rf = static_cast<float>(r);
        if (static_cast<double>(cf) != c || static_cast<double>(rf) != r || std::fpclassify(rf) != FP_NORMAL) {
            return false;
        }
    }
    recip = r;
    return true;
}

size_t rewrite_fdiv_pow2(Function& fn) {
    struct Site {
        Instruction* div;
        double recip;
    };
    std::vector<Site> sites;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            double c = 0.0;
            double r = 0.0;
            if (!const_divisor(inst, c)) continue;
            if (!exact_pow2_reciprocal(c, inst->type() == Type::f32(), r)) continue;
            sites.push_back({inst, r});
        }
    }
    if (sites.empty()) return 0;
    Builder b(fn);
    for (const Site& s : sites) {
        b.position_before(s.div);
        const bool f32 = s.div->type() == Type::f32();
        Value* k = f32 ? b.build_fconst_f32(static_cast<float>(s.recip)) : b.build_fconst_f64(s.recip);
        Value* m = b.build_mul(s.div->operand(0), k);
        replace_all_uses(fn, s.div->result(), m);
        s.div->parent()->remove_instruction(s.div);
    }
    return sites.size();
}

}  // namespace brass
