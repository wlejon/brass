#include "bv_solver.hpp"
#include <brass/interpreter/float_arith.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace brass::verify {

namespace {

constexpr bool is_commutative(BVOp op) noexcept {
    switch (op) {
        case BVOp::And:
        case BVOp::Or:
        case BVOp::Xor:
        case BVOp::Add:
        case BVOp::Mul:
        case BVOp::Eq:
        case BVOp::Ne:
        case BVOp::UAddOvf:
        case BVOp::SAddOvf:
        case BVOp::UMulOvf:
        case BVOp::SMulOvf:
            return true;
        default:
            return false;
    }
}

uint64_t f32_to_bits(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    return u;
}

float bits_to_f32(uint64_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b & 0xFFFFFFFFULL);
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

uint64_t f64_to_bits(double d) noexcept {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof(double));
    return u;
}

double bits_to_f64(uint64_t b) noexcept {
    double d = 0.0;
    std::memcpy(&d, &b, sizeof(double));
    return d;
}

bool mul_u64_overflow(uint64_t a, uint64_t b) noexcept {
    if (a == 0 || b == 0) return false;
    uint64_t prod = a * b;
    return (prod / a) != b;
}

bool mul_s64_overflow(int64_t a, int64_t b) noexcept {
    if (a == 0 || b == 0) return false;
    if (a == -1 && b == INT64_MIN) return true;
    if (b == -1 && a == INT64_MIN) return true;
    if (a > 0 && b > 0 && a > INT64_MAX / b) return true;
    if (a > 0 && b < 0 && b < INT64_MIN / a) return true;
    if (a < 0 && b > 0 && a < INT64_MIN / b) return true;
    if (a < 0 && b < 0 && a < INT64_MAX / b) return true;
    return false;
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
        iv.known_zeros = a.known_zeros | ~am;
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
        }
        return iv;
    }

    if (n.op == BVOp::And) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        iv.known_zeros = a.known_zeros | b.known_zeros;
        iv.known_ones = a.known_ones & b.known_ones;
        iv.umax = std::min(a.umax, b.umax) & ~iv.known_zeros;
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
        iv.umin = std::max(a.umin, b.umin) | iv.known_ones;
        iv.umax = m & ~iv.known_zeros;
        return iv;
    }

    if (n.op == BVOp::LShr) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (b.umin == b.umax && b.umin > 0 && b.umin < w) {
            uint32_t sh = static_cast<uint32_t>(b.umin);
            iv.umin = a.umin >> sh;
            iv.umax = a.umax >> sh;
            iv.smin = static_cast<int64_t>(iv.umin);
            iv.smax = static_cast<int64_t>(iv.umax);
            uint64_t low_mask = width_mask(w - sh);
            iv.known_zeros = (a.known_zeros >> sh) | ~low_mask;
            iv.known_ones = (a.known_ones >> sh) & low_mask;
        }
        return iv;
    }

    if (n.op == BVOp::Shl) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (b.umin == b.umax && b.umin < w) {
            uint32_t sh = static_cast<uint32_t>(b.umin);
            iv.known_zeros = ((a.known_zeros << sh) | width_mask(sh)) | ~m;
            iv.known_ones = (a.known_ones << sh) & m;
            if (sh == 0 || (a.umax <= (m >> sh))) {
                iv.umin = (a.umin << sh) & m;
                iv.umax = (a.umax << sh) & m;
            }
            if (w < 64) {
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
        return iv;
    }

    if (n.op == BVOp::Add && w <= 32) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (a.umin + b.umin <= m && a.umax + b.umax <= m) {
            iv.umin = a.umin + b.umin;
            iv.umax = a.umax + b.umax;
        }
        int64_t min_s = -(1LL << (w - 1));
        int64_t max_s = (1LL << (w - 1)) - 1LL;
        if (a.smin + b.smin >= min_s && a.smax + b.smax <= max_s) {
            iv.smin = a.smin + b.smin;
            iv.smax = a.smax + b.smax;
        }
        return iv;
    }

    if (n.op == BVOp::Mul && w <= 32) {
        const auto& a = intervals_[n.a];
        const auto& b = intervals_[n.b];
        if (a.smin >= 0 && b.smin >= 0) {
            int64_t max_s = (1LL << (w - 1)) - 1LL;
            if (a.smax == 0 || b.smax == 0 || a.smax <= max_s / b.smax) {
                iv.smin = a.smin * b.smin;
                iv.smax = a.smax * b.smax;
                iv.umin = static_cast<uint64_t>(iv.smin);
                iv.umax = static_cast<uint64_t>(iv.smax);
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
    if (lo == 0 && hi + 1 == na.width) {
        return a;
    }
    if (na.op == BVOp::Const) {
        return make_const((na.imm >> lo) & width_mask(w), w);
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
    uint32_t w = (res_width != 0) ? res_width : na.width;
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
            if (a == b) return make_binary(BVOp::Shl, a, make_const(1, w), w);
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
                    return make_binary(BVOp::Shl, a, make_const(sh, w), w);
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
                uint64_t sh = cb.imm & (w - 1);
                if (sh != cb.imm) {
                    return make_binary(op, a, make_const(sh, cb.width), w);
                }
                if (sh == 0) return a;
                if (ca.op == op && nodes_[ca.b].op == BVOp::Const) {
                    uint64_t prev_sh = nodes_[ca.b].imm;
                    if (prev_sh + sh < w) {
                        return make_binary(op, ca.a, make_const(prev_sh + sh, w), w);
                    } else if (op != BVOp::AShr) {
                        return make_const(0, w);
                    }
                }
                if (op == BVOp::Shl && ca.op == BVOp::Add && nodes_[ca.b].op == BVOp::Const) {
                    BVNodeId t1 = make_binary(BVOp::Shl, ca.a, b, w);
                    BVNodeId t2 = make_const((nodes_[ca.b].imm << sh) & m, w);
                    return make_binary(BVOp::Add, t1, t2, w);
                }
            }
            break;
        }

        case BVOp::UDiv:
        case BVOp::SDiv:
            if (cb.op == BVOp::Const && cb.imm == 1) return a;
            if (a == b && intervals_[b].umin > 0) return make_const(1, w);
            break;

        case BVOp::URem:
        case BVOp::SRem:
            if (cb.op == BVOp::Const && cb.imm == 1) return make_const(0, w);
            if (a == b && intervals_[b].umin > 0) return make_const(0, w);
            if (op == BVOp::URem && cb.op == BVOp::Const && std::has_single_bit(cb.imm)) {
                return make_binary(BVOp::And, a, make_const(cb.imm - 1, w), w);
            }
            break;

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

uint64_t BVContext::eval_node_concrete(const BVNode& n, uint64_t va, uint64_t vb, uint64_t vc) const noexcept {
    const uint32_t w = n.width;
    const uint64_t m = width_mask(w);
    const uint32_t aw = (n.a < nodes_.size()) ? nodes_[n.a].width : w;
    const uint64_t am = width_mask(aw);
    va &= am;
    vb &= (n.b < nodes_.size()) ? width_mask(nodes_[n.b].width) : m;
    vc &= (n.c < nodes_.size()) ? width_mask(nodes_[n.c].width) : m;

    switch (n.op) {
        case BVOp::Const:   return n.imm & m;
        case BVOp::Var:     return va & m;
        case BVOp::Extract: return (va >> n.extra) & m;
        case BVOp::Concat: {
            uint32_t lo_w = nodes_[n.b].width;
            return ((va << lo_w) | (vb & width_mask(lo_w))) & m;
        }
        case BVOp::ZExt:    return va & m;
        case BVOp::SExt:    return static_cast<uint64_t>(sign_extend_bits(va, aw)) & m;
        case BVOp::Not:     return (~va) & m;
        case BVOp::Neg:     return (~va + 1ULL) & m;
        case BVOp::And:     return (va & vb) & m;
        case BVOp::Or:      return (va | vb) & m;
        case BVOp::Xor:     return (va ^ vb) & m;
        case BVOp::Add:     return (va + vb) & m;
        case BVOp::Sub:     return (va - vb) & m;
        case BVOp::Mul:     return (va * vb) & m;
        case BVOp::UDiv:    return (vb == 0) ? 0 : ((va / vb) & m);
        case BVOp::URem:    return (vb == 0) ? 0 : ((va % vb) & m);
        case BVOp::SDiv: {
            if (vb == 0) return 0;
            int64_t sa = sign_extend_bits(va, w);
            int64_t sb = sign_extend_bits(vb, w);
            int64_t min_s = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
            if (sa == min_s && sb == -1) return static_cast<uint64_t>(min_s) & m;
            return static_cast<uint64_t>(sa / sb) & m;
        }
        case BVOp::SRem: {
            if (vb == 0) return 0;
            int64_t sa = sign_extend_bits(va, w);
            int64_t sb = sign_extend_bits(vb, w);
            int64_t min_s = (w >= 64) ? INT64_MIN : -(1LL << (w - 1));
            if (sa == min_s && sb == -1) return 0;
            return static_cast<uint64_t>(sa % sb) & m;
        }
        case BVOp::Shl: {
            uint32_t sh = static_cast<uint32_t>(vb & (w - 1));
            return (va << sh) & m;
        }
        case BVOp::LShr: {
            uint32_t sh = static_cast<uint32_t>(vb & (w - 1));
            return (va >> sh) & m;
        }
        case BVOp::AShr: {
            uint32_t sh = static_cast<uint32_t>(vb & (w - 1));
            int64_t sa = sign_extend_bits(va, w);
            return static_cast<uint64_t>(sa >> sh) & m;
        }
        case BVOp::Eq:  return (va == vb) ? 1 : 0;
        case BVOp::Ne:  return (va != vb) ? 1 : 0;
        case BVOp::Ult: return (va < vb) ? 1 : 0;
        case BVOp::Ule: return (va <= vb) ? 1 : 0;
        case BVOp::Ugt: return (va > vb) ? 1 : 0;
        case BVOp::Uge: return (va >= vb) ? 1 : 0;
        case BVOp::Slt: return (sign_extend_bits(va, aw) < sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sle: return (sign_extend_bits(va, aw) <= sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sgt: return (sign_extend_bits(va, aw) > sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Sge: return (sign_extend_bits(va, aw) >= sign_extend_bits(vb, aw)) ? 1 : 0;
        case BVOp::Ite: return (va != 0) ? (vb & m) : (vc & m);
        case BVOp::Clz: {
            if (va == 0) return w;
            return static_cast<uint64_t>(std::countl_zero(va) - (64 - w));
        }
        case BVOp::Ctz: {
            if (va == 0) return w;
            return static_cast<uint64_t>(std::countr_zero(va));
        }
        case BVOp::Popcnt:
            return static_cast<uint64_t>(std::popcount(va & m));

        case BVOp::UAddOvf:
            return (((va + vb) & am) < va) ? 1 : 0;
        case BVOp::USubOvf:
            return (va < vb) ? 1 : 0;
        case BVOp::UMulOvf: {
            if (aw <= 32) {
                return ((va * vb) > am) ? 1 : 0;
            }
            return mul_u64_overflow(va, vb) ? 1 : 0;
        }
        case BVOp::SAddOvf: {
            int64_t sa = sign_extend_bits(va, aw);
            int64_t sb = sign_extend_bits(vb, aw);
            int64_t sr = sign_extend_bits(va + vb, aw);
            return (((sa >= 0 && sb >= 0 && sr < 0) || (sa < 0 && sb < 0 && sr >= 0))) ? 1 : 0;
        }
        case BVOp::SSubOvf: {
            int64_t sa = sign_extend_bits(va, aw);
            int64_t sb = sign_extend_bits(vb, aw);
            int64_t sr = sign_extend_bits(va - vb, aw);
            return (((sa >= 0 && sb < 0 && sr < 0) || (sa < 0 && sb >= 0 && sr >= 0))) ? 1 : 0;
        }
        case BVOp::SMulOvf: {
            int64_t sa = sign_extend_bits(va, aw);
            int64_t sb = sign_extend_bits(vb, aw);
            if (aw <= 32) {
                int64_t p = sa * sb;
                int64_t min_s = -(1LL << (aw - 1));
                int64_t max_s = (1LL << (aw - 1)) - 1LL;
                return (p < min_s || p > max_s) ? 1 : 0;
            }
            return mul_s64_overflow(sa, sb) ? 1 : 0;
        }

        case BVOp::FAdd:
            return (w == 32) ? f32_to_bits(fparith::add(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(fparith::add(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FSub:
            return (w == 32) ? f32_to_bits(fparith::sub(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(fparith::sub(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FMul:
            return (w == 32) ? f32_to_bits(fparith::mul(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(fparith::mul(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FDiv:
            return (w == 32) ? f32_to_bits(fparith::div(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(fparith::div(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FSqrt:
            return (w == 32) ? f32_to_bits(std::sqrt(bits_to_f32(va)))
                             : f64_to_bits(std::sqrt(bits_to_f64(va)));
        case BVOp::FAbs:
            return (w == 32) ? f32_to_bits(std::fabs(bits_to_f32(va)))
                             : f64_to_bits(std::fabs(bits_to_f64(va)));
        case BVOp::FNeg:
            return (w == 32) ? f32_to_bits(-bits_to_f32(va))
                             : f64_to_bits(-bits_to_f64(va));
        case BVOp::FFloor:
            return (w == 32) ? f32_to_bits(std::floor(bits_to_f32(va)))
                             : f64_to_bits(std::floor(bits_to_f64(va)));
        case BVOp::FCeil:
            return (w == 32) ? f32_to_bits(std::ceil(bits_to_f32(va)))
                             : f64_to_bits(std::ceil(bits_to_f64(va)));
        case BVOp::FRound:
            return (w == 32) ? f32_to_bits(std::round(bits_to_f32(va)))
                             : f64_to_bits(std::round(bits_to_f64(va)));
        case BVOp::FMin:
            return (w == 32) ? f32_to_bits(std::fmin(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(std::fmin(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FMax:
            return (w == 32) ? f32_to_bits(std::fmax(bits_to_f32(va), bits_to_f32(vb)))
                             : f64_to_bits(std::fmax(bits_to_f64(va), bits_to_f64(vb)));
        case BVOp::FEq:
            return (aw == 32) ? (bits_to_f32(va) == bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) == bits_to_f64(vb) ? 1 : 0);
        case BVOp::FNe:
            return (aw == 32) ? (bits_to_f32(va) != bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) != bits_to_f64(vb) ? 1 : 0);
        case BVOp::FLt:
            return (aw == 32) ? (bits_to_f32(va) < bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) < bits_to_f64(vb) ? 1 : 0);
        case BVOp::FLe:
            return (aw == 32) ? (bits_to_f32(va) <= bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) <= bits_to_f64(vb) ? 1 : 0);
        case BVOp::FGt:
            return (aw == 32) ? (bits_to_f32(va) > bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) > bits_to_f64(vb) ? 1 : 0);
        case BVOp::FGe:
            return (aw == 32) ? (bits_to_f32(va) >= bits_to_f32(vb) ? 1 : 0)
                              : (bits_to_f64(va) >= bits_to_f64(vb) ? 1 : 0);
        case BVOp::SiToFp: {
            int64_t sv = sign_extend_bits(va, aw);
            return (w == 32) ? f32_to_bits(static_cast<float>(sv))
                             : f64_to_bits(static_cast<double>(sv));
        }
        case BVOp::FpToSi: {
            double d = (aw == 32) ? static_cast<double>(bits_to_f32(va)) : bits_to_f64(va);
            if (!std::isfinite(d)) return 0;
            if (w == 32) {
                if (d <= static_cast<double>(INT32_MIN)) return static_cast<uint32_t>(INT32_MIN);
                if (d >= static_cast<double>(INT32_MAX)) return static_cast<uint32_t>(INT32_MAX);
                return static_cast<uint32_t>(static_cast<int32_t>(d));
            }
            if (d <= -9223372036854775808.0) return static_cast<uint64_t>(INT64_MIN);
            if (d >= 9223372036854775807.0) return static_cast<uint64_t>(INT64_MAX);
            return static_cast<uint64_t>(static_cast<int64_t>(d)) & m;
        }
        case BVOp::FpExt:
            return f64_to_bits(static_cast<double>(bits_to_f32(va)));
        case BVOp::FpTrunc:
            return f32_to_bits(static_cast<float>(bits_to_f64(va)));
    }
    return 0;
}

void BVContext::evaluate_all(const std::vector<uint64_t>& var_values, std::vector<uint64_t>& node_vals) const {
    node_vals.resize(nodes_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const BVNode& n = nodes_[i];
        if (n.op == BVOp::Var) {
            uint64_t v = (n.imm < var_values.size()) ? var_values[static_cast<size_t>(n.imm)] : 0;
            node_vals[i] = v & width_mask(n.width);
            continue;
        }
        if (n.op == BVOp::Const) {
            node_vals[i] = n.imm & width_mask(n.width);
            continue;
        }
        node_vals[i] = eval_node_concrete(n, node_vals[n.a], node_vals[n.b], node_vals[n.c]);
    }
}

uint64_t BVContext::eval_concrete(BVNodeId root, const std::vector<uint64_t>& var_values) const {
    std::vector<uint64_t> node_vals;
    evaluate_all(var_values, node_vals);
    return (root < node_vals.size()) ? node_vals[root] : 0;
}

std::string BVContext::export_smt2(BVNodeId bad_cond, std::string_view comment) const {
    std::ostringstream os;
    os << "; Brass Symbolic Bitvector Equivalence Query\n";
    if (!comment.empty()) {
        os << "; " << comment << "\n";
    }
    os << "(set-logic QF_BV)\n";
    for (const auto& v : vars_) {
        os << "(declare-fun " << v.name << " () (_ BitVec " << v.width << "))\n";
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const BVNode& n = nodes_[i];
        os << "(define-fun t" << i << " () (_ BitVec " << n.width << ") ";
        auto r = [](BVNodeId id) { return "t" + std::to_string(id); };
        switch (n.op) {
            case BVOp::Const:
                os << "(_ bv" << n.imm << " " << n.width << ")";
                break;
            case BVOp::Var:
                os << vars_[static_cast<size_t>(n.imm)].name;
                break;
            case BVOp::Extract:
                os << "((_ extract " << n.imm << " " << n.extra << ") " << r(n.a) << ")";
                break;
            case BVOp::Concat:
                os << "(concat " << r(n.a) << " " << r(n.b) << ")";
                break;
            case BVOp::ZExt:
                os << "((_ zero_extend " << n.extra << ") " << r(n.a) << ")";
                break;
            case BVOp::SExt:
                os << "((_ sign_extend " << n.extra << ") " << r(n.a) << ")";
                break;
            case BVOp::Not: os << "(bvnot " << r(n.a) << ")"; break;
            case BVOp::Neg: os << "(bvneg " << r(n.a) << ")"; break;
            case BVOp::And: os << "(bvand " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Or:  os << "(bvor "  << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Xor: os << "(bvxor " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Add: os << "(bvadd " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Sub: os << "(bvsub " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Mul: os << "(bvmul " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::UDiv: os << "(bvudiv " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::SDiv: os << "(bvsdiv " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::URem: os << "(bvurem " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::SRem: os << "(bvsrem " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Shl:  os << "(bvshl "  << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::LShr: os << "(bvlshr " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::AShr: os << "(bvashr " << r(n.a) << " " << r(n.b) << ")"; break;
            case BVOp::Eq:  os << "(ite (= "     << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ne:  os << "(ite (distinct " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ult: os << "(ite (bvult " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ule: os << "(ite (bvule " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ugt: os << "(ite (bvugt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Uge: os << "(ite (bvuge " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Slt: os << "(ite (bvslt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sle: os << "(ite (bvsle " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sgt: os << "(ite (bvsgt " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Sge: os << "(ite (bvsge " << r(n.a) << " " << r(n.b) << ") #b1 #b0)"; break;
            case BVOp::Ite: os << "(ite (= " << r(n.a) << " #b1) " << r(n.b) << " " << r(n.c) << ")"; break;
            default:        os << "(_ bv0 " << n.width << ")"; break;
        }
        os << ")\n";
    }
    os << "(assert (= t" << bad_cond << " #b1))\n";
    os << "(check-sat)\n(get-model)\n";
    return os.str();
}

} // namespace brass::verify
