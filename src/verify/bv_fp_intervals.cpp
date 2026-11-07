#include "bv_solver.hpp"
#include <cmath>
#include <cstring>

namespace brass::verify {

namespace {

constexpr bool is_fp_pos_zero(const BVNode& n) noexcept {
    return n.op == BVOp::Const && n.imm == 0;
}

constexpr bool is_fp_neg_zero(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x80000000ULL) ||
            (n.width == 64 && n.imm == 0x8000000000000000ULL));
}

constexpr bool is_fp_one(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x3F800000ULL) ||
            (n.width == 64 && n.imm == 0x3FF0000000000000ULL));
}

constexpr bool is_fp_two(const BVNode& n) noexcept {
    return n.op == BVOp::Const &&
           ((n.width == 32 && n.imm == 0x40000000ULL) ||
            (n.width == 64 && n.imm == 0x4000000000000000ULL));
}

bool is_non_neg_fp_const(const BVNode& n, const BVInterval& iv) noexcept {
    if (n.op != BVOp::Const || !iv.fp_non_nan) return false;
    if (n.imm == 0 || is_fp_neg_zero(n)) return true;
    uint64_t sign_bit = (n.width == 32) ? (1ULL << 31) : (1ULL << 63);
    return (n.imm & sign_bit) == 0;
}

bool is_non_pos_fp_const(const BVNode& n, const BVInterval& iv) noexcept {
    if (n.op != BVOp::Const || !iv.fp_non_nan) return false;
    if (n.imm == 0 || is_fp_neg_zero(n)) return true;
    uint64_t sign_bit = (n.width == 32) ? (1ULL << 31) : (1ULL << 63);
    return (n.imm & sign_bit) != 0;
}

} // namespace

void BVContext::compute_fp_interval(const BVNode& n, BVInterval& iv) const noexcept {
    const uint32_t w = n.width;
    if (w != 32 && w != 64) return;

    switch (n.op) {
        case BVOp::Const: {
            if (w == 32) {
                uint32_t bits = static_cast<uint32_t>(n.imm & 0xFFFFFFFFULL);
                uint32_t exp = (bits >> 23) & 0xFFu;
                uint32_t mant = bits & 0x7FFFFFu;
                bool is_nan = (exp == 0xFFu) && (mant != 0);
                bool is_inf = (exp == 0xFFu) && (mant == 0);
                bool is_neg_zero = (bits == 0x80000000u);
                iv.fp_non_nan = !is_nan;
                iv.fp_non_inf = !is_inf;
                iv.fp_non_neg_zero = !is_neg_zero;
                if (!is_nan && !is_inf) {
                    float f = 0.0f;
                    std::memcpy(&f, &bits, sizeof(float));
                    iv.fp_exact_int = (std::floor(f) == f) && (std::fabs(f) <= 16777216.0f);
                }
            } else {
                uint64_t bits = n.imm;
                uint64_t exp = (bits >> 52) & 0x7FFULL;
                uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;
                bool is_nan = (exp == 0x7FFULL) && (mant != 0);
                bool is_inf = (exp == 0x7FFULL) && (mant == 0);
                bool is_neg_zero = (bits == 0x8000000000000000ULL);
                iv.fp_non_nan = !is_nan;
                iv.fp_non_inf = !is_inf;
                iv.fp_non_neg_zero = !is_neg_zero;
                if (!is_nan && !is_inf) {
                    double d = 0.0;
                    std::memcpy(&d, &bits, sizeof(double));
                    iv.fp_exact_int = (std::floor(d) == d) && (std::fabs(d) <= 9007199254740992.0);
                }
            }
            break;
        }

        case BVOp::SiToFp: {
            iv.fp_non_nan = true;
            iv.fp_non_inf = true;
            iv.fp_non_neg_zero = true;
            uint32_t aw = nodes_[n.a].width;
            const auto& ia = intervals_[n.a];
            if (w == 64) {
                if (aw <= 32 || (ia.smin >= -(1LL << 53) && ia.smax <= (1LL << 53))) {
                    iv.fp_exact_int = true;
                }
            } else {
                if (aw <= 24 || (ia.smin >= -(1LL << 24) && ia.smax <= (1LL << 24))) {
                    iv.fp_exact_int = true;
                }
            }
            if (ia.smin >= 0) {
                uint64_t sign_bit = 1ULL << (w - 1);
                iv.known_zeros |= sign_bit;
                iv.umax = width_mask(w) & ~iv.known_zeros;
            }
            break;
        }

        case BVOp::FAbs: {
            const auto& ia = intervals_[n.a];
            iv.fp_non_nan = ia.fp_non_nan;
            iv.fp_non_inf = ia.fp_non_inf;
            iv.fp_non_neg_zero = true;
            iv.fp_exact_int = ia.fp_exact_int;
            uint64_t sign_bit = 1ULL << (w - 1);
            iv.known_zeros = ia.known_zeros | sign_bit;
            iv.known_ones = ia.known_ones & ~sign_bit;
            iv.umin = iv.known_ones;
            iv.umax = width_mask(w) & ~iv.known_zeros;
            break;
        }

        case BVOp::FNeg: {
            const auto& ia = intervals_[n.a];
            iv.fp_non_nan = ia.fp_non_nan;
            iv.fp_non_inf = ia.fp_non_inf;
            iv.fp_exact_int = ia.fp_exact_int;
            break;
        }

        case BVOp::FFloor:
        case BVOp::FCeil:
        case BVOp::FRound: {
            const auto& ia = intervals_[n.a];
            iv.fp_non_nan = ia.fp_non_nan;
            iv.fp_non_inf = ia.fp_non_inf;
            iv.fp_exact_int = ia.fp_exact_int;
            if ((ia.fp_exact_int && ia.fp_non_neg_zero) ||
                (n.op == BVOp::FFloor && ia.fp_non_neg_zero)) {
                iv.fp_non_neg_zero = true;
            }
            break;
        }

        case BVOp::FMin:
        case BVOp::FMax: {
            const auto& ia = intervals_[n.a];
            const auto& ib = intervals_[n.b];
            iv.fp_non_nan = ia.fp_non_nan && ib.fp_non_nan;
            iv.fp_non_inf = ia.fp_non_inf && ib.fp_non_inf;
            iv.fp_non_neg_zero = ia.fp_non_neg_zero && ib.fp_non_neg_zero;
            iv.fp_exact_int = ia.fp_exact_int && ib.fp_exact_int;
            break;
        }

        case BVOp::FpExt: {
            const auto& ia = intervals_[n.a];
            iv.fp_non_nan = ia.fp_non_nan;
            iv.fp_non_inf = ia.fp_non_inf;
            iv.fp_non_neg_zero = ia.fp_non_neg_zero;
            iv.fp_exact_int = ia.fp_exact_int;
            break;
        }

        case BVOp::Ite: {
            const auto& it = intervals_[n.b];
            const auto& if_ = intervals_[n.c];
            iv.fp_non_nan = it.fp_non_nan && if_.fp_non_nan;
            iv.fp_non_inf = it.fp_non_inf && if_.fp_non_inf;
            iv.fp_non_neg_zero = it.fp_non_neg_zero && if_.fp_non_neg_zero;
            iv.fp_exact_int = it.fp_exact_int && if_.fp_exact_int;
            break;
        }

        default:
            break;
    }
}

BVNodeId BVContext::simplify_fp_unary(BVOp op, BVNodeId a, uint32_t w) {
    const BVNode na = nodes_[a];
    const BVInterval ia = intervals_[a];

    switch (op) {
        case BVOp::FAbs: {
            if (na.op == BVOp::FAbs) return a;
            if (na.op == BVOp::FNeg) return make_unary(BVOp::FAbs, na.a, w);
            if ((w == 32 || w == 64) && (ia.known_zeros & (1ULL << (w - 1))) != 0) {
                return a;
            }
            break;
        }

        case BVOp::FNeg: {
            if (na.op == BVOp::FNeg) return na.a;
            break;
        }

        case BVOp::FFloor:
        case BVOp::FCeil:
        case BVOp::FRound: {
            if (ia.fp_exact_int && ia.fp_non_neg_zero && ia.fp_non_nan && ia.fp_non_inf) {
                return a;
            }
            if (na.op == BVOp::FFloor || na.op == BVOp::FCeil || na.op == BVOp::FRound) {
                return a;
            }
            break;
        }

        case BVOp::FpToSi: {
            if (w == 32 && na.op == BVOp::SiToFp && na.width == 64 && nodes_[na.a].width == 32) {
                return na.a;
            }
            if (w == 64 && na.op == BVOp::SiToFp && na.width == 64 && nodes_[na.a].width == 32) {
                return make_sext(na.a, 32);
            }
            break;
        }

        default:
            break;
    }

    return UINT32_MAX;
}

BVNodeId BVContext::simplify_fp_binary(BVOp op, BVNodeId a, BVNodeId b, uint32_t w) {
    const BVNode ca = nodes_[a];
    const BVNode cb = nodes_[b];
    const BVInterval ia = intervals_[a];
    const BVInterval ib = intervals_[b];

    switch (op) {
        case BVOp::FEq:
        case BVOp::FLe:
        case BVOp::FGe:
            if (a == b && ia.fp_non_nan) return make_const(1, 1);
            break;

        case BVOp::FNe:
        case BVOp::FLt:
        case BVOp::FGt:
            if (a == b && ia.fp_non_nan) return make_const(0, 1);
            break;

        case BVOp::FMin:
        case BVOp::FMax:
            if (a == b && ia.fp_non_nan) return a;
            break;

        case BVOp::FAdd:
            if (is_fp_pos_zero(cb) && ia.fp_non_nan && ia.fp_non_neg_zero) return a;
            if (is_fp_pos_zero(ca) && ib.fp_non_nan && ib.fp_non_neg_zero) return b;
            if (is_fp_neg_zero(cb) && ia.fp_non_nan) return a;
            if (is_fp_neg_zero(ca) && ib.fp_non_nan) return b;
            break;

        case BVOp::FSub:
            if (is_fp_pos_zero(cb) && ia.fp_non_nan && ia.fp_non_neg_zero) return a;
            if (is_fp_neg_zero(cb) && ia.fp_non_nan && ia.fp_non_neg_zero) return a;
            break;

        case BVOp::FMul:
            if (is_fp_one(cb) && ia.fp_non_nan) return a;
            if (is_fp_one(ca) && ib.fp_non_nan) return b;
            if ((w == 32 || w == 64) && is_fp_two(cb)) return make_binary(BVOp::FAdd, a, a, w);
            if ((w == 32 || w == 64) && is_fp_two(ca)) return make_binary(BVOp::FAdd, b, b, w);
            break;

        default:
            break;
    }

    return UINT32_MAX;
}

BVNodeId BVContext::simplify_fp_ite(BVNodeId cond, BVNodeId t_val, BVNodeId f_val) {
    const BVNode nc = nodes_[cond];
    BVNodeId guarded_var = UINT32_MAX;
    bool guarded_pos = false;

    if ((nc.op == BVOp::FGt && is_non_neg_fp_const(nodes_[nc.b], intervals_[nc.b])) ||
        (nc.op == BVOp::FLt && is_non_pos_fp_const(nodes_[nc.b], intervals_[nc.b]))) {
        guarded_var = nc.a;
        guarded_pos = (nc.op == BVOp::FGt);
    } else if ((nc.op == BVOp::FLt && is_non_neg_fp_const(nodes_[nc.a], intervals_[nc.a])) ||
               (nc.op == BVOp::FGt && is_non_pos_fp_const(nodes_[nc.a], intervals_[nc.a]))) {
        guarded_var = nc.b;
        guarded_pos = (nc.op == BVOp::FLt);
    }

    if (guarded_var != UINT32_MAX) {
        const BVNode nt = nodes_[t_val];
        BVNodeId new_t = t_val;
        if (nt.op == BVOp::FAdd) {
            if (nt.a == guarded_var && (is_fp_pos_zero(nodes_[nt.b]) || is_fp_neg_zero(nodes_[nt.b]))) {
                new_t = guarded_var;
            } else if (nt.b == guarded_var && (is_fp_pos_zero(nodes_[nt.a]) || is_fp_neg_zero(nodes_[nt.a]))) {
                new_t = guarded_var;
            }
        } else if (nt.op == BVOp::FSub) {
            if (nt.a == guarded_var && (is_fp_pos_zero(nodes_[nt.b]) || is_fp_neg_zero(nodes_[nt.b]))) {
                new_t = guarded_var;
            }
        } else if (nt.op == BVOp::FMul) {
            if (nt.a == guarded_var && is_fp_one(nodes_[nt.b])) new_t = guarded_var;
            else if (nt.b == guarded_var && is_fp_one(nodes_[nt.a])) new_t = guarded_var;
        } else if (nt.op == BVOp::FAbs && guarded_pos && nt.a == guarded_var) {
            new_t = guarded_var;
        }
        if (new_t != t_val) {
            return make_ite(cond, new_t, f_val);
        }
    }

    return UINT32_MAX;
}

} // namespace brass::verify
