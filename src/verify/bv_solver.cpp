#include "bv_solver.hpp"
#include <algorithm>
#include <bit>
#include <limits>

namespace brass::verify {

namespace {

// a + b against a bound, exactly, without 128-bit integers (MSVC has none).
constexpr bool sum_at_least(int64_t a, int64_t b, int64_t lo) noexcept {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) return true;    // over INT64_MAX
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) return false;   // under INT64_MIN
    return a + b >= lo;
}
constexpr bool sum_at_most(int64_t a, int64_t b, int64_t hi) noexcept {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) return false;
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) return true;
    return a + b <= hi;
}
// a + b <= m for unsigned a, b, without wrapping.
constexpr bool usum_at_most(uint64_t a, uint64_t b, uint64_t m) noexcept { return a <= m && b <= m - a; }

constexpr bool is_commutative(BVOp op) noexcept {
    switch (op) {
        case BVOp::And: case BVOp::Or: case BVOp::Xor: case BVOp::Add: case BVOp::Mul:
        case BVOp::Eq: case BVOp::Ne:
        case BVOp::UAddOvf: case BVOp::SAddOvf: case BVOp::UMulOvf: case BVOp::SMulOvf:
            return true;
        default:
            return false;
    }
}

} // namespace

BVContext::BVContext() {
    // Node 0 is always 1-bit constant 0
    BVNode zero;
    zero.op = BVOp::Const;
    zero.width = 1;
    zero.imm = 0;
    intern(zero);
}

BVNodeId BVContext::intern(const BVNode& node) {
    auto it = memo_.find(node);
    if (it != memo_.end()) {
        return it->second;
    }
    BVNodeId id = static_cast<BVNodeId>(nodes_.size());
    nodes_.push_back(node);
    intervals_.push_back(compute_interval(node));
    memo_.emplace(node, id);
    return id;
}

BVInterval BVContext::compute_interval(const BVNode& n) const noexcept {
    BVInterval iv;
    const uint32_t w = n.width;
    const uint64_t m = width_mask(w);
    iv.umin = 0;
    iv.umax = m;
    iv.smin = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
    iv.smax = (w >= 64) ? INT64_MAX : ((1LL << (w - 1)) - 1LL);
    iv.known_zeros = ~m;
    iv.known_ones = 0;

    if (n.op == BVOp::Const) {
        uint64_t c = n.imm & m;
        int64_t sc = sign_extend_bits(c, w);
        iv.umin = iv.umax = c;
        iv.smin = iv.smax = sc;
        iv.known_zeros = ~c;
        iv.known_ones = c;
        return iv;
    }

    if (n.op == BVOp::Extract) {
        const auto& a = intervals_[n.a];
        uint32_t lo = n.extra;
        iv.known_zeros = ((a.known_zeros >> lo) & m) | ~m;
        iv.known_ones = (a.known_ones >> lo) & m;
        iv.umin = iv.known_ones;
        iv.umax = m & ~iv.known_zeros;
        uint64_t sign_bit = (w >= 64) ? (1ULL << 63) : (1ULL << (w - 1));
        if (iv.known_zeros & sign_bit) {
            iv.smin = static_cast<int64_t>(iv.umin);
            iv.smax = static_cast<int64_t>(iv.umax);
        }
        return iv;
    }

    if (n.op == BVOp::ZExt) {
        const auto& a = intervals_[n.a];
        uint32_t aw = nodes_[n.a].width;
        uint64_t am = width_mask(aw);
        iv.umin = a.umin & am;
        iv.umax = a.umax & am;
        if (aw < w) {
            iv.smin = static_cast<int64_t>(iv.umin);
            iv.smax = static_cast<int64_t>(iv.umax);
        }
        uint32_t bits = (iv.umax == 0) ? 0u : static_cast<uint32_t>(64 - std::countl_zero(iv.umax));
        iv.known_zeros = a.known_zeros | ~am | ~width_mask(bits);
        iv.known_ones = a.known_ones & am;
        return iv;
    }

    if (n.op == BVOp::SExt) {
        const auto& a = intervals_[n.a];
        iv.smin = a.smin;
        iv.smax = a.smax;
        if (a.smin >= 0) {
            iv.umin = static_cast<uint64_t>(a.smin);
            iv.umax = static_cast<uint64_t>(a.smax);
            uint32_t bits = (iv.umax == 0) ? 0u : static_cast<uint32_t>(64 - std::countl_zero(iv.umax));
            iv.known_zeros = a.known_zeros | ~width_mask(bits);
        } else if (a.smax < 0) {
            iv.umin = static_cast<uint64_t>(a.smin) & m;
            iv.umax = static_cast<uint64_t>(a.smax) & m;
        }
        return iv;
    }

    if (n.op == BVOp::And) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        iv.known_zeros = a.known_zeros | b.known_zeros;
        iv.known_ones = a.known_ones & b.known_ones;
        iv.umax = std::min({a.umax, b.umax, m & ~iv.known_zeros});
        iv.umin = iv.known_ones;
        uint64_t sign_bit = (w >= 64) ? (1ULL << 63) : (1ULL << (w - 1));
        if (iv.known_zeros & sign_bit) {
            iv.smin = static_cast<int64_t>(iv.umin);
            iv.smax = static_cast<int64_t>(iv.umax);
        }
        return iv;
    }

    if (n.op == BVOp::Or) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        iv.known_zeros = a.known_zeros & b.known_zeros;
        iv.known_ones = a.known_ones | b.known_ones;
        iv.umin = std::max({a.umin, b.umin, iv.known_ones});
        iv.umax = m & ~iv.known_zeros;
        return iv;
    }

    if (n.op == BVOp::LShr) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (b.umin == b.umax) {
            uint32_t sh = static_cast<uint32_t>(b.umin & shift_mask(w));
            if (sh > 0 && sh < w) {
                iv.umin = a.umin >> sh;
                iv.umax = a.umax >> sh;
                iv.smin = static_cast<int64_t>(iv.umin);
                iv.smax = static_cast<int64_t>(iv.umax);
                uint64_t low_mask = width_mask(w - sh);
                iv.known_zeros = (a.known_zeros >> sh) | ~low_mask;
                iv.known_ones = (a.known_ones >> sh) & low_mask;
            }
        }
        return iv;
    }

    if (n.op == BVOp::Shl) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (b.umin == b.umax) {
            uint32_t sh = static_cast<uint32_t>(b.umin & shift_mask(w));
            if (sh < w) {
                iv.known_zeros = ((a.known_zeros << sh) | width_mask(sh)) | ~m;
                iv.known_ones = (a.known_ones << sh) & m;
                if (sh == 0 || (a.umax <= (m >> sh))) {
                    iv.umin = (a.umin << sh) & m;
                    iv.umax = (a.umax << sh) & m;
                }
                if (w + sh < 63) {
                    int64_t lo = a.smin * (1LL << sh);
                    int64_t hi = a.smax * (1LL << sh);
                    int64_t min_s = -(1LL << (w - 1));
                    int64_t max_s = (1LL << (w - 1)) - 1LL;
                    if (lo >= min_s && hi <= max_s) {
                        iv.smin = lo;
                        iv.smax = hi;
                    }
                }
            }
        }
        return iv;
    }

    if (n.op == BVOp::Add && w <= 64) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (usum_at_most(a.umin, b.umin, m) && usum_at_most(a.umax, b.umax, m)) {
            iv.umin = a.umin + b.umin;
            iv.umax = a.umax + b.umax;
            uint32_t bits = (iv.umax == 0) ? 0u : static_cast<uint32_t>(64 - std::countl_zero(iv.umax));
            iv.known_zeros = ~width_mask(bits);
        }
        int64_t min_s = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
        int64_t max_s = (w >= 64) ? INT64_MAX : ((1LL << (w - 1)) - 1LL);
        if (sum_at_least(a.smin, b.smin, min_s) && sum_at_most(a.smax, b.smax, max_s)) {
            iv.smin = a.smin + b.smin;
            iv.smax = a.smax + b.smax;
            if (iv.smin >= 0) {
                iv.umin = std::max(iv.umin, static_cast<uint64_t>(iv.smin));
                iv.umax = std::min(iv.umax, static_cast<uint64_t>(iv.smax));
                uint32_t bits = (iv.umax == 0) ? 0u : static_cast<uint32_t>(64 - std::countl_zero(iv.umax));
                iv.known_zeros |= ~width_mask(bits);
            } else if (iv.smax < 0) {
                iv.umin = std::max(iv.umin, static_cast<uint64_t>(iv.smin) & m);
                iv.umax = std::min(iv.umax, static_cast<uint64_t>(iv.smax) & m);
            }
        }
        return iv;
    }

    if (n.op == BVOp::Mul && w <= 64) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (a.smin >= 0 && b.smin >= 0) {
            int64_t max_s = (w >= 64) ? INT64_MAX : ((1LL << (w - 1)) - 1LL);
            if (a.smax == 0 || b.smax == 0 || a.smax <= max_s / b.smax) {
                iv.smin = a.smin * b.smin;
                iv.smax = a.smax * b.smax;
                iv.umin = static_cast<uint64_t>(iv.smin);
                iv.umax = static_cast<uint64_t>(iv.smax);
                uint32_t bits = (iv.umax == 0) ? 0u : static_cast<uint32_t>(64 - std::countl_zero(iv.umax));
                iv.known_zeros = ~width_mask(bits);
            }
        }
        return iv;
    }

    if (n.op == BVOp::Ite) {
        const auto& t = intervals_[n.b];
        const auto& f = intervals_[n.c];
        iv.umin = std::min(t.umin, f.umin);
        iv.umax = std::max(t.umax, f.umax);
        iv.smin = std::min(t.smin, f.smin);
        iv.smax = std::max(t.smax, f.smax);
        iv.known_zeros = t.known_zeros & f.known_zeros;
        iv.known_ones = t.known_ones & f.known_ones;
        return iv;
    }

    return iv;
}

BVNodeId BVContext::make_const(uint64_t val, uint32_t width) {
    BVNode n;
    n.op = BVOp::Const;
    n.width = width;
    n.imm = val & width_mask(width);
    return intern(n);
}

BVNodeId BVContext::make_var(std::string name, uint32_t width, bool is_float) {
    uint32_t idx = static_cast<uint32_t>(vars_.size());
    BVNode n;
    n.op = BVOp::Var;
    n.width = width;
    n.imm = idx;
    BVNodeId id = intern(n);
    vars_.push_back(BVVarInfo{std::move(name), width, id, is_float});
    return id;
}

BVNodeId BVContext::make_extract(BVNodeId a, uint32_t hi, uint32_t lo) {
    const BVNode na = nodes_[a];
    uint32_t w = hi - lo + 1;
    uint64_t w_mask = width_mask(w);
    if (lo == 0 && hi + 1 == na.width) {
        return a;
    }
    if (na.op == BVOp::Const) {
        return make_const((na.imm >> lo) & w_mask, w);
    }
    if (((intervals_[a].known_zeros >> lo) & w_mask) == w_mask) {
        return make_const(0, w);
    }
    if (na.op == BVOp::Ite) {
        return make_ite(na.a, make_extract(na.b, hi, lo), make_extract(na.c, hi, lo));
    }
    if (na.op == BVOp::Extract) {
        return make_extract(na.a, na.extra + hi, na.extra + lo);
    }
    if ((na.op == BVOp::ZExt || na.op == BVOp::SExt) && hi < nodes_[na.a].width) {
        return make_extract(na.a, hi, lo);
    }
    if (na.op == BVOp::ZExt && lo >= nodes_[na.a].width) {
        return make_const(0, w);
    }
    if (na.op == BVOp::Concat) {
        uint32_t lo_w = nodes_[na.b].width;
        if (hi < lo_w) {
            return make_extract(na.b, hi, lo);
        }
        if (lo >= lo_w) {
            return make_extract(na.a, hi - lo_w, lo - lo_w);
        }
    }
    if (lo == 0 && (na.op == BVOp::And || na.op == BVOp::Or || na.op == BVOp::Xor ||
                    na.op == BVOp::Add || na.op == BVOp::Sub || na.op == BVOp::Mul)) {
        return make_binary(na.op, make_extract(na.a, hi, 0), make_extract(na.b, hi, 0), w);
    }
    if (lo == 0 && na.op == BVOp::Shl && nodes_[na.b].op == BVOp::Const &&
        (nodes_[na.b].imm & shift_mask(na.width)) < w) {
        return make_binary(BVOp::Shl, make_extract(na.a, hi, 0),
                           make_const(nodes_[na.b].imm & shift_mask(na.width), w), w);
    }
    BVNode n;
    n.op = BVOp::Extract;
    n.width = w;
    n.a = a;
    n.imm = hi;
    n.extra = lo;
    return intern(n);
}

BVNodeId BVContext::make_concat(BVNodeId hi, BVNodeId lo) {
    const BVNode nhi = nodes_[hi];
    const BVNode nlo = nodes_[lo];
    uint32_t w = nhi.width + nlo.width;
    if (nhi.op == BVOp::Const && nlo.op == BVOp::Const) {
        uint64_t v = ((nhi.imm & width_mask(nhi.width)) << nlo.width) | (nlo.imm & width_mask(nlo.width));
        return make_const(v, w);
    }
    if (nhi.op == BVOp::Const && nhi.imm == 0) {
        return make_zext(lo, nhi.width);
    }
    if (nhi.op == BVOp::Extract && nlo.op == BVOp::Extract && nhi.a == nlo.a &&
        nlo.imm + 1 == nhi.extra) {
        return make_extract(nhi.a, static_cast<uint32_t>(nhi.imm), nlo.extra);
    }
    BVNode n;
    n.op = BVOp::Concat;
    n.width = w;
    n.a = hi;
    n.b = lo;
    return intern(n);
}

BVNodeId BVContext::make_zext(BVNodeId a, uint32_t ext_bits) {
    if (ext_bits == 0) return a;
    const BVNode na = nodes_[a];
    uint32_t w = na.width + ext_bits;
    if (na.op == BVOp::Const) {
        return make_const(na.imm & width_mask(na.width), w);
    }
    if (na.op == BVOp::ZExt) {
        return make_zext(na.a, na.extra + ext_bits);
    }
    if (na.op == BVOp::Extract && na.extra == 0 && nodes_[na.a].width == w) {
        return make_binary(BVOp::And, na.a, make_const(width_mask(na.width), w), w);
    }
    BVNode n;
    n.op = BVOp::ZExt;
    n.width = w;
    n.a = a;
    n.extra = ext_bits;
    return intern(n);
}

BVNodeId BVContext::make_sext(BVNodeId a, uint32_t ext_bits) {
    if (ext_bits == 0) return a;
    const BVNode na = nodes_[a];
    uint32_t w = na.width + ext_bits;
    if (na.op == BVOp::Const) {
        int64_t sv = sign_extend_bits(na.imm, na.width);
        return make_const(static_cast<uint64_t>(sv), w);
    }
    if (na.op == BVOp::SExt) {
        return make_sext(na.a, na.extra + ext_bits);
    }
    if (na.op == BVOp::ZExt) {
        return make_zext(na.a, na.extra + ext_bits);
    }
    const auto& iv = intervals_[a];
    if (iv.smin >= 0) {
        return make_zext(a, ext_bits);
    }
    BVNode n;
    n.op = BVOp::SExt;
    n.width = w;
    n.a = a;
    n.extra = ext_bits;
    return intern(n);
}

BVNodeId BVContext::make_unary(BVOp op, BVNodeId a, uint32_t res_width) {
    const BVNode na = nodes_[a];
    uint32_t w = (res_width != 0) ? res_width : na.width;
    if (na.op == BVOp::Const) {
        BVNode tmp;
        tmp.op = op;
        tmp.width = w;
        tmp.a = a;
        return make_const(eval_node_concrete(tmp, na.imm, 0, 0), w);
    }
    if (op == BVOp::Not && na.op == BVOp::Not) {
        return na.a;
    }
    if (op == BVOp::Neg) {
        return make_binary(BVOp::Sub, make_const(0, w), a, w);
    }
    BVNode n;
    n.op = op;
    n.width = w;
    n.a = a;
    return intern(n);
}

BVNodeId BVContext::make_binary(BVOp op, BVNodeId a, BVNodeId b, uint32_t res_width) {
    const BVNode na = nodes_[a];
    const BVNode nb = nodes_[b];
    bool is_pred = (op == BVOp::Eq || op == BVOp::Ne ||
                    op == BVOp::Ult || op == BVOp::Ule || op == BVOp::Ugt || op == BVOp::Uge ||
                    op == BVOp::Slt || op == BVOp::Sle || op == BVOp::Sgt || op == BVOp::Sge ||
                    op == BVOp::FEq || op == BVOp::FNe || op == BVOp::FLt || op == BVOp::FLe ||
                    op == BVOp::FGt || op == BVOp::FGe ||
                    op == BVOp::UAddOvf || op == BVOp::USubOvf || op == BVOp::UMulOvf ||
                    op == BVOp::SAddOvf || op == BVOp::SSubOvf || op == BVOp::SMulOvf);
    uint32_t w = (res_width != 0) ? res_width : (is_pred ? 1u : std::max(na.width, nb.width));
    const uint64_t m = width_mask(na.width);

    if (op == BVOp::Ugt) return make_binary(BVOp::Ult, b, a, 1);
    if (op == BVOp::Uge) return make_unary(BVOp::Not, make_binary(BVOp::Ult, a, b, 1), 1);
    if (op == BVOp::Ule) return make_unary(BVOp::Not, make_binary(BVOp::Ult, b, a, 1), 1);
    if (op == BVOp::Sgt) return make_binary(BVOp::Slt, b, a, 1);
    if (op == BVOp::Sge) return make_unary(BVOp::Not, make_binary(BVOp::Slt, a, b, 1), 1);
    if (op == BVOp::Sle) return make_unary(BVOp::Not, make_binary(BVOp::Slt, b, a, 1), 1);
    if (op == BVOp::Ne)  return make_unary(BVOp::Not, make_binary(BVOp::Eq, a, b, 1), 1);

    if (is_commutative(op)) {
        bool swap = false;
        if (na.op == BVOp::Const && nb.op != BVOp::Const) {
            swap = true;
        } else if ((na.op == BVOp::Const) == (nb.op == BVOp::Const) && a > b) {
            swap = true;
        }
        if (swap) {
            std::swap(a, b);
        }
    }

    const BVNode ca = nodes_[a];
    const BVNode cb = nodes_[b];

    if (ca.op == BVOp::Const && cb.op == BVOp::Const) {
        BVNode tmp;
        tmp.op = op;
        tmp.width = w;
        tmp.a = a;
        tmp.b = b;
        return make_const(eval_node_concrete(tmp, ca.imm, cb.imm, 0), w);
    }

    switch (op) {
        case BVOp::Add:
            if (cb.op == BVOp::Const && cb.imm == 0) return a;
            if (a == b) return (w == 1) ? make_const(0, 1) : make_binary(BVOp::Shl, a, make_const(1, w), w);
            if (ca.op == BVOp::Add && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                return make_binary(BVOp::Add, ca.a, make_const(nodes_[ca.b].imm + cb.imm, w), w);
            }
            if (ca.op == BVOp::Sub && nodes_[ca.a].op == BVOp::Const && nodes_[ca.a].imm == 0) {
                return make_binary(BVOp::Sub, b, ca.b, w);
            }
            if (cb.op == BVOp::Sub && nodes_[cb.a].op == BVOp::Const && nodes_[cb.a].imm == 0) {
                return make_binary(BVOp::Sub, a, cb.b, w);
            }
            if (ca.op == BVOp::Sub && ca.b == b) return ca.a;
            if (cb.op == BVOp::Sub && cb.b == a) return cb.a;
            if (((intervals_[a].known_zeros | intervals_[b].known_zeros) & m) == m) {
                return make_binary(BVOp::Or, a, b, w);
            }
            break;

        case BVOp::Sub:
            if (cb.op == BVOp::Const && cb.imm == 0) return a;
            if (a == b) return make_const(0, w);
            if (cb.op == BVOp::Const) {
                return make_binary(BVOp::Add, a, make_const((~cb.imm + 1ULL) & m, w), w);
            }
            if (ca.op == BVOp::Add && ca.a == b) return ca.b;
            if (ca.op == BVOp::Add && ca.b == b) return ca.a;
            if (cb.op == BVOp::Add && cb.a == a) {
                return make_binary(BVOp::Sub, make_const(0, w), cb.b, w);
            }
            if (cb.op == BVOp::Add && cb.b == a) {
                return make_binary(BVOp::Sub, make_const(0, w), cb.a, w);
            }
            break;

        case BVOp::Mul:
            if (cb.op == BVOp::Const) {
                if (cb.imm == 0) return make_const(0, w);
                if (cb.imm == 1) return a;
                if (cb.imm == m) return make_binary(BVOp::Sub, make_const(0, w), a, w);
                if (std::has_single_bit(cb.imm)) {
                    uint32_t sh = static_cast<uint32_t>(std::countr_zero(cb.imm));
                    if (sh < w) return make_binary(BVOp::Shl, a, make_const(sh, w), w);
                }
            }
            if (ca.op == BVOp::Mul && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                return make_binary(BVOp::Mul, ca.a, make_const(nodes_[ca.b].imm * cb.imm, w), w);
            }
            if (ca.op == BVOp::Add && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                BVNodeId t1 = make_binary(BVOp::Mul, ca.a, b, w);
                BVNodeId t2 = make_const(nodes_[ca.b].imm * cb.imm, w);
                return make_binary(BVOp::Add, t1, t2, w);
            }
            break;

        case BVOp::And:
            if (a == b) return a;
            if (cb.op == BVOp::Const) {
                if (cb.imm == 0) return make_const(0, w);
                if (cb.imm == m) return a;
                if ((intervals_[a].known_zeros | ~cb.imm) == intervals_[a].known_zeros) {
                    return a;
                }
                if ((intervals_[a].known_zeros & cb.imm) == cb.imm) {
                    return make_const(0, w);
                }
            }
            if (ca.op == BVOp::Not && ca.a == b) return make_const(0, w);
            if (cb.op == BVOp::Not && cb.a == a) return make_const(0, w);
            if (ca.op == BVOp::And && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                return make_binary(BVOp::And, ca.a, make_const(nodes_[ca.b].imm & cb.imm, w), w);
            }
            break;

        case BVOp::Or:
            if (a == b) return a;
            if (cb.op == BVOp::Const) {
                if (cb.imm == 0) return a;
                if (cb.imm == m) return make_const(m, w);
            }
            if (ca.op == BVOp::Not && ca.a == b) return make_const(m, w);
            if (cb.op == BVOp::Not && cb.a == a) return make_const(m, w);
            if (ca.op == BVOp::Or && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                return make_binary(BVOp::Or, ca.a, make_const(nodes_[ca.b].imm | cb.imm, w), w);
            }
            if (ca.op == BVOp::And && cb.op == BVOp::And) {
                if (ca.a == cb.a) return make_binary(BVOp::And, ca.a, make_binary(BVOp::Or, ca.b, cb.b, w), w);
                if (ca.a == cb.b) return make_binary(BVOp::And, ca.a, make_binary(BVOp::Or, ca.b, cb.a, w), w);
                if (ca.b == cb.a) return make_binary(BVOp::And, ca.b, make_binary(BVOp::Or, ca.a, cb.b, w), w);
                if (ca.b == cb.b) return make_binary(BVOp::And, ca.b, make_binary(BVOp::Or, ca.a, cb.a, w), w);
            }
            break;

        case BVOp::Xor:
            if (a == b) return make_const(0, w);
            if (cb.op == BVOp::Const) {
                if (cb.imm == 0) return a;
                if (cb.imm == m) return make_unary(BVOp::Not, a, w);
            }
            if (ca.op == BVOp::Xor && ca.a == b) return ca.b;
            if (ca.op == BVOp::Xor && ca.b == b) return ca.a;
            if (cb.op == BVOp::Xor && cb.a == a) return cb.b;
            if (cb.op == BVOp::Xor && cb.b == a) return cb.a;
            if (ca.op == BVOp::Xor && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                return make_binary(BVOp::Xor, ca.a, make_const(nodes_[ca.b].imm ^ cb.imm, w), w);
            }
            break;

        case BVOp::Shl:
        case BVOp::LShr:
        case BVOp::AShr: {
            if (cb.op == BVOp::Const) {
                uint64_t sh = cb.imm & shift_mask(w);
                if (sh != cb.imm) {
                    return make_binary(op, a, make_const(sh, cb.width), w);
                }
                if (sh == 0) return a;
                if (sh >= w && op != BVOp::AShr) {
                    return make_const(0, w);
                }
                if (ca.op == op && nodes_[ca.b].op == BVOp::Const) {
                    uint64_t prev_sh = nodes_[ca.b].imm & shift_mask(w);
                    if (prev_sh + sh < w) {
                        return make_binary(op, ca.a, make_const(prev_sh + sh, w), w);
                    } else if (op != BVOp::AShr) {
                        return make_const(0, w);
                    }
                }
                if (op == BVOp::Shl && sh < w && ca.op == BVOp::Add && nodes_[ca.b].op == BVOp::Const) {
                    BVNodeId t1 = make_binary(BVOp::Shl, ca.a, b, w);
                    BVNodeId t2 = make_const((nodes_[ca.b].imm << sh) & m, w);
                    return make_binary(BVOp::Add, t1, t2, w);
                }
                if (op == BVOp::Shl && sh < w && ca.op == BVOp::LShr && nodes_[ca.b].op == BVOp::Const && nodes_[ca.b].imm == sh) {
                    return make_binary(BVOp::And, ca.a, make_const((m << sh) & m, w), w);
                }
                if (op == BVOp::LShr && sh < w && ca.op == BVOp::Shl && nodes_[ca.b].op == BVOp::Const && nodes_[ca.b].imm == sh) {
                    return make_binary(BVOp::And, ca.a, make_const(m >> sh, w), w);
                }
            }
            break;
        }

        case BVOp::UDiv:
        case BVOp::SDiv:
        case BVOp::URem:
        case BVOp::SRem: {
            BVNodeId exp = expand_const_div_rem_bv(*this, op, a, b, w);
            if (exp != UINT32_MAX) return exp;
            break;
        }

        case BVOp::Eq:
        case BVOp::Ne: {
            bool is_eq = (op == BVOp::Eq);
            if (a == b) return make_const(is_eq ? 1 : 0, 1);
            if (ca.op == BVOp::Add && cb.op == BVOp::Add) {
                if (ca.a == cb.a) return make_binary(op, ca.b, cb.b, 1);
                if (ca.b == cb.b) return make_binary(op, ca.a, cb.a, 1);
                if (ca.a == cb.b) return make_binary(op, ca.b, cb.a, 1);
                if (ca.b == cb.a) return make_binary(op, ca.a, cb.b, 1);
            }
            if (ca.op == BVOp::Add && nodes_[ca.b].op == BVOp::Const && cb.op == BVOp::Const) {
                uint64_t diff = (cb.imm - nodes_[ca.b].imm) & m;
                return make_binary(op, ca.a, make_const(diff, ca.width), 1);
            }
            if ((ca.op == BVOp::ZExt && cb.op == BVOp::ZExt && nodes_[ca.a].width == nodes_[cb.a].width) ||
                (ca.op == BVOp::SExt && cb.op == BVOp::SExt && nodes_[ca.a].width == nodes_[cb.a].width)) {
                return make_binary(op, ca.a, cb.a, 1);
            }
            if (ca.op == BVOp::ZExt && nodes_[ca.a].width == 1 && cb.op == BVOp::Const) {
                if (cb.imm == 0) return make_unary(BVOp::Not, ca.a, 1);
                if (cb.imm == 1) return is_eq ? ca.a : make_unary(BVOp::Not, ca.a, 1);
                return make_const(is_eq ? 0 : 1, 1);
            }
            const auto& ia = intervals_[a];
            const auto& ib = intervals_[b];
            if (ia.umax < ib.umin || ib.umax < ia.umin || ia.smax < ib.smin || ib.smax < ia.smin ||
                ((ia.known_ones & ib.known_zeros) != 0) || ((ib.known_ones & ia.known_zeros) != 0)) {
                return make_const(is_eq ? 0 : 1, 1);
            }
            break;
        }

        case BVOp::Ult:
        case BVOp::Ule: {
            bool is_lt = (op == BVOp::Ult);
            if (a == b) return make_const(is_lt ? 0 : 1, 1);
            const auto& ia = intervals_[a];
            const auto& ib = intervals_[b];
            if (is_lt) {
                if (ia.umax < ib.umin) return make_const(1, 1);
                if (ia.umin >= ib.umax) return make_const(0, 1);
            } else {
                if (ia.umax <= ib.umin) return make_const(1, 1);
                if (ia.umin > ib.umax) return make_const(0, 1);
            }
            if (ca.op == BVOp::ZExt && cb.op == BVOp::ZExt && nodes_[ca.a].width == nodes_[cb.a].width) {
                return make_binary(op, ca.a, cb.a, 1);
            }
            break;
        }

        case BVOp::Slt:
        case BVOp::Sle: {
            bool is_lt = (op == BVOp::Slt);
            if (a == b) return make_const(is_lt ? 0 : 1, 1);
            const auto& ia = intervals_[a];
            const auto& ib = intervals_[b];
            if (is_lt) {
                if (ia.smax < ib.smin) return make_const(1, 1);
                if (ia.smin >= ib.smax) return make_const(0, 1);
            } else {
                if (ia.smax <= ib.smin) return make_const(1, 1);
                if (ia.smin > ib.smax) return make_const(0, 1);
            }
            if (ca.op == BVOp::SExt && cb.op == BVOp::SExt && nodes_[ca.a].width == nodes_[cb.a].width) {
                return make_binary(op, ca.a, cb.a, 1);
            }
            break;
        }

        default:
            break;
    }

    BVNode n;
    n.op = op;
    n.width = w;
    n.a = a;
    n.b = b;
    return intern(n);
}

BVNodeId BVContext::make_ite(BVNodeId cond, BVNodeId t_val, BVNodeId f_val) {
    if (t_val == f_val) return t_val;
    const BVNode nc = nodes_[cond];
    if (nc.op == BVOp::Const) {
        return (nc.imm != 0) ? t_val : f_val;
    }
    if (nc.op == BVOp::Not) {
        return make_ite(nc.a, f_val, t_val);
    }
    if (nodes_[t_val].op == BVOp::Ite && nodes_[t_val].a == cond) {
        t_val = nodes_[t_val].b;
    }
    if (nodes_[f_val].op == BVOp::Ite && nodes_[f_val].a == cond) {
        f_val = nodes_[f_val].c;
    }
    if (t_val == f_val) return t_val;
    uint32_t w = nodes_[t_val].width;
    if (w == 1) {
        const BVNode nt = nodes_[t_val];
        const BVNode nf = nodes_[f_val];
        if (nt.op == BVOp::Const && nf.op == BVOp::Const) {
            if (nt.imm == 1 && nf.imm == 0) return cond;
            if (nt.imm == 0 && nf.imm == 1) return make_unary(BVOp::Not, cond, 1);
        }
    }
    BVNode n;
    n.op = BVOp::Ite;
    n.width = w;
    n.a = cond;
    n.b = t_val;
    n.c = f_val;
    return intern(n);
}

} // namespace brass::verify
