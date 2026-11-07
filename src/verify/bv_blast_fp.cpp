#include "bv_blast_fp.hpp"
#include <algorithm>

namespace brass::verify {

namespace {

using Lit = CDCLSolver::Lit;

uint32_t fp_mant_bits(uint32_t w) noexcept {
    return (w == 32) ? 23u : 52u;
}

Lit is_exp_all_ones(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    uint32_t mb = fp_mant_bits(w);
    Lit acc = sat.true_lit();
    for (uint32_t i = mb; i + 1 < w; ++i) {
        acc = sat.mk_and(acc, b[i]);
    }
    return acc;
}

Lit is_exp_all_zeros(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    uint32_t mb = fp_mant_bits(w);
    Lit acc = sat.true_lit();
    for (uint32_t i = mb; i + 1 < w; ++i) {
        acc = sat.mk_and(acc, CDCLSolver::neg(b[i]));
    }
    return acc;
}

Lit is_exp_max_minus_one(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    uint32_t mb = fp_mant_bits(w);
    Lit acc = CDCLSolver::neg(b[mb]);
    for (uint32_t i = mb + 1; i + 1 < w; ++i) {
        acc = sat.mk_and(acc, b[i]);
    }
    return acc;
}

Lit is_mant_nonzero(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    uint32_t mb = fp_mant_bits(w);
    Lit acc = sat.false_lit();
    for (uint32_t i = 0; i < mb; ++i) {
        acc = sat.mk_or(acc, b[i]);
    }
    return acc;
}

Lit is_nan(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    return sat.mk_and(is_exp_all_ones(sat, b, w), is_mant_nonzero(sat, b, w));
}

Lit is_zero(CDCLSolver& sat, const std::vector<Lit>& b, uint32_t w) {
    Lit acc = sat.true_lit();
    for (uint32_t i = 0; i + 1 < w; ++i) {
        acc = sat.mk_and(acc, CDCLSolver::neg(b[i]));
    }
    return acc;
}

Lit blast_mag_eq(CDCLSolver& sat, const std::vector<Lit>& a, const std::vector<Lit>& b, uint32_t w) {
    Lit acc = sat.true_lit();
    for (uint32_t i = 0; i + 1 < w; ++i) {
        Lit eq_i = CDCLSolver::neg(sat.mk_xor(a[i], b[i]));
        acc = sat.mk_and(acc, eq_i);
    }
    return acc;
}

Lit blast_mag_ult(CDCLSolver& sat, const std::vector<Lit>& a, const std::vector<Lit>& b, uint32_t w) {
    Lit carry = sat.true_lit();
    for (uint32_t i = 0; i + 1 < w; ++i) {
        Lit ai = a[i];
        Lit nbi = CDCLSolver::neg(b[i]);
        Lit axb = sat.mk_xor(ai, nbi);
        Lit ab = sat.mk_and(ai, nbi);
        Lit c_axb = sat.mk_and(carry, axb);
        carry = sat.mk_or(ab, c_axb);
    }
    return CDCLSolver::neg(carry);
}

Lit blast_fp_lt(CDCLSolver& sat, const std::vector<Lit>& a, const std::vector<Lit>& b,
                uint32_t w, Lit any_nan, Lit both_zero) {
    Lit sign_a = a[w - 1];
    Lit sign_b = b[w - 1];
    Lit mag_lt_ab = blast_mag_ult(sat, a, b, w);
    Lit mag_lt_ba = blast_mag_ult(sat, b, a, w);
    Lit neg_pos = sat.mk_and(sign_a, CDCLSolver::neg(sign_b));
    Lit both_pos = sat.mk_and(CDCLSolver::neg(sign_a), CDCLSolver::neg(sign_b));
    Lit both_neg = sat.mk_and(sign_a, sign_b);
    Lit raw_lt = sat.mk_or(neg_pos,
                           sat.mk_or(sat.mk_and(both_pos, mag_lt_ab),
                                     sat.mk_and(both_neg, mag_lt_ba)));
    return sat.mk_and(CDCLSolver::neg(any_nan),
                      sat.mk_and(CDCLSolver::neg(both_zero), raw_lt));
}

void blast_quiet_copy(CDCLSolver& sat, const std::vector<Lit>& bx, uint32_t w, Lit nan_x, std::vector<Lit>& out) {
    uint32_t qbit = fp_mant_bits(w) - 1;
    for (uint32_t i = 0; i < w; ++i) {
        out[i] = (i == qbit) ? sat.mk_or(bx[i], nan_x) : bx[i];
    }
}

void blast_scale_two(CDCLSolver& sat, const std::vector<Lit>& bx, uint32_t w, std::vector<Lit>& out) {
    uint32_t mb = fp_mant_bits(w);
    Lit exp_ones = is_exp_all_ones(sat, bx, w);
    Lit exp_zeros = is_exp_all_zeros(sat, bx, w);
    Lit exp_ovf = is_exp_max_minus_one(sat, bx, w);
    Lit nan_x = is_nan(sat, bx, w);

    std::vector<Lit> norm_bits(w, sat.false_lit());
    for (uint32_t i = 0; i < mb; ++i) norm_bits[i] = bx[i];
    Lit carry = sat.true_lit();
    for (uint32_t i = mb; i + 1 < w; ++i) {
        norm_bits[i] = sat.mk_xor(bx[i], carry);
        carry = sat.mk_and(bx[i], carry);
    }

    std::vector<Lit> sub_bits(w, sat.false_lit());
    for (uint32_t i = 1; i + 1 < w; ++i) {
        sub_bits[i] = bx[i - 1];
    }

    std::vector<Lit> inf_nan_bits(w, sat.false_lit());
    for (uint32_t i = 0; i < mb; ++i) {
        Lit m_bit = (i == mb - 1) ? sat.mk_or(bx[i], nan_x) : bx[i];
        inf_nan_bits[i] = sat.mk_and(exp_ones, m_bit);
    }
    for (uint32_t i = mb; i + 1 < w; ++i) {
        inf_nan_bits[i] = sat.true_lit();
    }

    Lit use_inf_nan = sat.mk_or(exp_ones, exp_ovf);
    for (uint32_t i = 0; i + 1 < w; ++i) {
        out[i] = sat.mk_ite(use_inf_nan, inf_nan_bits[i],
                            sat.mk_ite(exp_zeros, sub_bits[i], norm_bits[i]));
    }
    out[w - 1] = bx[w - 1];
}

constexpr bool is_const_pos_zero(const BVNode& n) noexcept {
    return n.op == BVOp::Const && n.imm == 0;
}

constexpr bool is_const_neg_zero(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x80000000ULL) ||
            (n.width == 64 && n.imm == 0x8000000000000000ULL));
}

constexpr bool is_const_pos_one(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x3F800000ULL) ||
            (n.width == 64 && n.imm == 0x3FF0000000000000ULL));
}

constexpr bool is_const_neg_one(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0xBF800000ULL) ||
            (n.width == 64 && n.imm == 0xBFF0000000000000ULL));
}

constexpr bool is_const_pos_two(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x40000000ULL) ||
            (n.width == 64 && n.imm == 0x4000000000000000ULL));
}

} // namespace

bool blast_fp_node(
    CDCLSolver& sat,
    const BVContext& ctx,
    const BVNode& n,
    const std::vector<std::vector<CDCLSolver::Lit>>& bits,
    std::vector<CDCLSolver::Lit>& out
) {
    const auto& nodes = ctx.nodes();
    const uint32_t w = n.width;

    switch (n.op) {
        case BVOp::FAbs: {
            if (w != 32 && w != 64) return false;
            const auto& ba = bits[n.a];
            for (uint32_t i = 0; i + 1 < w; ++i) out[i] = ba[i];
            out[w - 1] = sat.false_lit();
            return true;
        }

        case BVOp::FNeg: {
            if (w != 32 && w != 64) return false;
            const auto& ba = bits[n.a];
            for (uint32_t i = 0; i + 1 < w; ++i) out[i] = ba[i];
            out[w - 1] = CDCLSolver::neg(ba[w - 1]);
            return true;
        }

        case BVOp::FEq:
        case BVOp::FNe:
        case BVOp::FLt:
        case BVOp::FLe:
        case BVOp::FGt:
        case BVOp::FGe: {
            const auto& ba = bits[n.a];
            const auto& bb = bits[n.b];
            const uint32_t aw = static_cast<uint32_t>(ba.size());
            if ((aw != 32 && aw != 64) || bb.size() != aw) return false;

            Lit nan_a = is_nan(sat, ba, aw);
            Lit nan_b = is_nan(sat, bb, aw);
            Lit any_nan = sat.mk_or(nan_a, nan_b);
            Lit both_zero = sat.mk_and(is_zero(sat, ba, aw), is_zero(sat, bb, aw));
            Lit same_sign = CDCLSolver::neg(sat.mk_xor(ba[aw - 1], bb[aw - 1]));
            Lit mag_eq = blast_mag_eq(sat, ba, bb, aw);
            Lit full_eq = sat.mk_or(both_zero, sat.mk_and(same_sign, mag_eq));

            if (n.op == BVOp::FEq) {
                out[0] = sat.mk_and(CDCLSolver::neg(any_nan), full_eq);
            } else if (n.op == BVOp::FNe) {
                out[0] = sat.mk_or(any_nan, CDCLSolver::neg(full_eq));
            } else if (n.op == BVOp::FLt) {
                out[0] = blast_fp_lt(sat, ba, bb, aw, any_nan, both_zero);
            } else if (n.op == BVOp::FLe) {
                Lit lt = blast_fp_lt(sat, ba, bb, aw, any_nan, both_zero);
                out[0] = sat.mk_and(CDCLSolver::neg(any_nan), sat.mk_or(lt, full_eq));
            } else if (n.op == BVOp::FGt) {
                out[0] = blast_fp_lt(sat, bb, ba, aw, any_nan, both_zero);
            } else {
                Lit gt = blast_fp_lt(sat, bb, ba, aw, any_nan, both_zero);
                out[0] = sat.mk_and(CDCLSolver::neg(any_nan), sat.mk_or(gt, full_eq));
            }
            return true;
        }

        case BVOp::FMin:
        case BVOp::FMax: {
            if (w != 32 && w != 64) return false;
            const auto& ba = bits[n.a];
            const auto& bb = bits[n.b];
            if (ba.size() != w || bb.size() != w) return false;

            Lit nan_a = is_nan(sat, ba, w);
            Lit nan_b = is_nan(sat, bb, w);
            Lit any_nan = sat.mk_or(nan_a, nan_b);
            Lit both_nan = sat.mk_and(nan_a, nan_b);
            Lit both_zero = sat.mk_and(is_zero(sat, ba, w), is_zero(sat, bb, w));
            Lit pick_a = (n.op == BVOp::FMin)
                ? blast_fp_lt(sat, ba, bb, w, any_nan, both_zero)
                : blast_fp_lt(sat, bb, ba, w, any_nan, both_zero);

            uint32_t qbit = fp_mant_bits(w) - 1;
            for (uint32_t i = 0; i < w; ++i) {
                Lit ord_i = sat.mk_ite(pick_a, ba[i], bb[i]);
                Lit zero_i = sat.false_lit();
                if (i == w - 1) {
                    zero_i = (n.op == BVOp::FMin)
                        ? sat.mk_or(ba[w - 1], bb[w - 1])
                        : sat.mk_and(ba[w - 1], bb[w - 1]);
                }
                Lit non_nan_i = sat.mk_ite(both_zero, zero_i, ord_i);
                Lit qa_i = (i == qbit) ? sat.true_lit() : ba[i];
                out[i] = sat.mk_ite(both_nan, qa_i,
                                    sat.mk_ite(nan_a, bb[i],
                                               sat.mk_ite(nan_b, ba[i], non_nan_i)));
            }
            return true;
        }

        case BVOp::FpExt: {
            const auto& ba = bits[n.a];
            if (ba.size() != 32 || w != 64) return false;
            Lit exp_ones = is_exp_all_ones(sat, ba, 32);
            Lit exp_zeros = is_exp_all_zeros(sat, ba, 32);
            Lit nan_a = is_nan(sat, ba, 32);

            std::vector<Lit> norm_bits(64, sat.false_lit());
            for (uint32_t i = 0; i < 23; ++i) norm_bits[29 + i] = ba[i];
            for (uint32_t i = 0; i < 7; ++i) norm_bits[52 + i] = ba[23 + i];
            Lit e7 = ba[30];
            norm_bits[59] = CDCLSolver::neg(e7);
            norm_bits[60] = CDCLSolver::neg(e7);
            norm_bits[61] = CDCLSolver::neg(e7);
            norm_bits[62] = e7;

            std::vector<Lit> inf_nan_bits(64, sat.false_lit());
            for (uint32_t i = 0; i < 22; ++i) inf_nan_bits[29 + i] = ba[i];
            inf_nan_bits[51] = sat.mk_or(ba[22], nan_a);
            for (uint32_t i = 52; i < 63; ++i) inf_nan_bits[i] = sat.true_lit();

            std::vector<Lit> sub_bits(64, sat.false_lit());
            for (uint32_t bit = 0; bit < 23; ++bit) {
                Lit b_lit = ba[bit];
                if (b_lit == sat.false_lit()) continue;
                uint64_t exp11 = 874ULL + bit;
                for (uint32_t j = 0; j < 63; ++j) {
                    Lit cand_j = sat.false_lit();
                    if (j >= 52) {
                        cand_j = ((exp11 >> (j - 52)) & 1ULL) ? sat.true_lit() : sat.false_lit();
                    } else if (j >= 52 - bit) {
                        cand_j = ba[j - (52 - bit)];
                    }
                    sub_bits[j] = sat.mk_ite(b_lit, cand_j, sub_bits[j]);
                }
            }

            for (uint32_t j = 0; j < 63; ++j) {
                out[j] = sat.mk_ite(exp_ones, inf_nan_bits[j],
                                    sat.mk_ite(exp_zeros, sub_bits[j], norm_bits[j]));
            }
            out[63] = ba[31];
            return true;
        }

        case BVOp::FAdd: {
            if (w != 32 && w != 64) return false;
            const BVNode& na = nodes[n.a];
            const BVNode& nb = nodes[n.b];
            if (n.a == n.b) {
                blast_scale_two(sat, bits[n.a], w, out);
                return true;
            }
            if (is_const_pos_zero(nb) || is_const_pos_zero(na)) {
                const auto& bx = is_const_pos_zero(nb) ? bits[n.a] : bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                Lit zero_x = is_zero(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                out[w - 1] = sat.mk_and(CDCLSolver::neg(zero_x), bx[w - 1]);
                return true;
            }
            if (is_const_neg_zero(nb) || is_const_neg_zero(na)) {
                const auto& bx = is_const_neg_zero(nb) ? bits[n.a] : bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                return true;
            }
            return false;
        }

        case BVOp::FSub: {
            if (w != 32 && w != 64) return false;
            const BVNode& na = nodes[n.a];
            const BVNode& nb = nodes[n.b];
            if (is_const_pos_zero(nb)) {
                const auto& bx = bits[n.a];
                Lit nan_x = is_nan(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                return true;
            }
            if (is_const_neg_zero(nb)) {
                const auto& bx = bits[n.a];
                Lit nan_x = is_nan(sat, bx, w);
                Lit zero_x = is_zero(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                out[w - 1] = sat.mk_and(CDCLSolver::neg(zero_x), bx[w - 1]);
                return true;
            }
            if (is_const_neg_zero(na)) {
                const auto& bx = bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                out[w - 1] = sat.mk_ite(nan_x, bx[w - 1], CDCLSolver::neg(bx[w - 1]));
                return true;
            }
            if (is_const_pos_zero(na)) {
                const auto& bx = bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                Lit zero_x = is_zero(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                out[w - 1] = sat.mk_ite(nan_x, bx[w - 1],
                                        sat.mk_and(CDCLSolver::neg(zero_x), CDCLSolver::neg(bx[w - 1])));
                return true;
            }
            return false;
        }

        case BVOp::FMul: {
            if (w != 32 && w != 64) return false;
            const BVNode& na = nodes[n.a];
            const BVNode& nb = nodes[n.b];
            if (is_const_pos_one(nb) || is_const_pos_one(na)) {
                const auto& bx = is_const_pos_one(nb) ? bits[n.a] : bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                return true;
            }
            if (is_const_neg_one(nb) || is_const_neg_one(na)) {
                const auto& bx = is_const_neg_one(nb) ? bits[n.a] : bits[n.b];
                Lit nan_x = is_nan(sat, bx, w);
                blast_quiet_copy(sat, bx, w, nan_x, out);
                out[w - 1] = sat.mk_ite(nan_x, bx[w - 1], CDCLSolver::neg(bx[w - 1]));
                return true;
            }
            if (is_const_pos_two(nb) || is_const_pos_two(na)) {
                const auto& bx = is_const_pos_two(nb) ? bits[n.a] : bits[n.b];
                blast_scale_two(sat, bx, w, out);
                return true;
            }
            return false;
        }

        default:
            return false;
    }
}

} // namespace brass::verify
